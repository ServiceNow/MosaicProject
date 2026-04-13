"""Backend state for the privacy-agent retrieval helper service."""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from trajectory_generation import config
from trajectory_generation.src.retrieval.browsecomp import BrowseCompHybridSearcher
from trajectory_generation.src.retrieval.composite import FilteredSearcher, WeightedRRFSearcher
from trajectory_generation.src.retrieval.dense import OpenAICompatibleBackend
from trajectory_generation.src.retrieval.hybrid import HybridSearcher


def _maybe_v1_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[rank])


class EndpointMetrics:
    """Small in-memory latency window for helper health reporting."""

    def __init__(self, window_size: int = 256) -> None:
        self.window_size = int(window_size)
        self._counts: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._active: dict[str, int] = defaultdict(int)
        self._latencies_ms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._lock = threading.Lock()

    def start(self, endpoint: str) -> float:
        with self._lock:
            self._active[endpoint] += 1
        return time.perf_counter()

    def finish(self, endpoint: str, start_time: float, ok: bool) -> None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        with self._lock:
            self._counts[endpoint] += 1
            if not ok:
                self._errors[endpoint] += 1
            self._active[endpoint] = max(0, self._active[endpoint] - 1)
            self._latencies_ms[endpoint].append(elapsed_ms)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            payload: dict[str, dict[str, float | int]] = {}
            endpoints = set(self._counts) | set(self._active) | set(self._latencies_ms)
            for endpoint in sorted(endpoints):
                samples = list(self._latencies_ms.get(endpoint, ()))
                payload[endpoint] = {
                    "count": int(self._counts.get(endpoint, 0)),
                    "errors": int(self._errors.get(endpoint, 0)),
                    "active": int(self._active.get(endpoint, 0)),
                    "p50_ms": _percentile(samples, 50),
                    "p95_ms": _percentile(samples, 95),
                    "max_ms": max(samples) if samples else 0.0,
                }
            return payload


class EmbeddingService:
    """Shared client for the colocated vLLM embedding endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        batch_size: int,
    ) -> None:
        self.base_url = _maybe_v1_url(base_url)
        self.model_name = model_name
        self.health_url = self.base_url.rsplit("/v1", 1)[0] + "/health"
        self._health_client = httpx.Client(timeout=timeout_seconds)
        self._backend = OpenAICompatibleBackend(
            model=model_name,
            base_url=self.base_url,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            query_prefix="",
        )

    def embed_texts(self, texts: list[str], model_name: str | None = None) -> np.ndarray:
        if model_name is not None and model_name != self.model_name:
            raise ValueError(
                f"Helper was started with embedding model {self.model_name}, but request asked for {model_name}."
            )
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(self._backend.embed_texts(texts), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(self._backend.embed_query(query), dtype=np.float32)

    def health(self) -> bool:
        try:
            response = self._health_client.get(self.health_url)
            return response.status_code == 200
        except Exception:
            return False


class LocalTaskIndex:
    """Read-only task-local index built by PipelineRL privacy_agent."""

    def __init__(self, storage_dir: Path, mmap_mode: str = "r") -> None:
        self.storage_dir = storage_dir
        self.documents = json.loads((storage_dir / "documents.json").read_text(encoding="utf-8"))
        self.doc_ids = json.loads((storage_dir / "index.json").read_text(encoding="utf-8")).get(
            "doc_ids", []
        )
        self.embeddings = np.load(storage_dir / "embeddings.npy", mmap_mode=mmap_mode)
        self._embedding_norms: np.ndarray | None = None

    @property
    def embedding_norms(self) -> np.ndarray:
        if self._embedding_norms is None:
            self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        return self._embedding_norms

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def search(self, query_vector: np.ndarray, k: int, threshold: float) -> list[dict[str, Any]]:
        if k <= 0 or not self.doc_ids:
            return []

        query_norm = float(np.linalg.norm(query_vector))
        if not query_norm:
            return []

        similarities = np.dot(self.embeddings, query_vector) / (self.embedding_norms * query_norm)
        top_indices = np.argsort(similarities)[::-1][: int(k)]

        hits: list[dict[str, Any]] = []
        for raw_idx in top_indices:
            idx = int(raw_idx)
            score = float(similarities[idx])
            if score < threshold:
                continue
            doc_id = str(self.doc_ids[idx])
            doc_info = self.documents[doc_id]
            hits.append(
                {
                    "doc_id": doc_id,
                    "similarity_score": score,
                    "content": doc_info["content"],
                    "metadata": doc_info.get("metadata", {}),
                    "preview": doc_info.get("content_preview", doc_info["content"][:300]),
                    "timestamp": doc_info.get("timestamp"),
                }
            )
        return hits


class LocalIndexRepository:
    """Lazy cache of task-local read-only indices."""

    def __init__(self, index_root: str | Path, mmap_mode: str = "r") -> None:
        self.index_root = Path(index_root).expanduser()
        self.mmap_mode = mmap_mode
        self._cache: dict[str, LocalTaskIndex] = {}
        self._lock = threading.Lock()

    @property
    def total_tasks(self) -> int:
        if not self.index_root.exists():
            return 0
        return sum(1 for child in self.index_root.iterdir() if child.is_dir())

    @property
    def loaded_tasks(self) -> int:
        return len(self._cache)

    def get(self, task_id: str) -> LocalTaskIndex:
        with self._lock:
            index = self._cache.get(task_id)
            if index is not None:
                return index

            storage_dir = self.index_root / task_id
            if not storage_dir.exists():
                raise FileNotFoundError(f"Missing local index for task {task_id} at {storage_dir}")

            index = LocalTaskIndex(storage_dir=storage_dir, mmap_mode=self.mmap_mode)
            self._cache[task_id] = index
            return index


def _is_direct_curated_doc_record(row: dict[str, Any]) -> bool:
    if row.get("extractor") == "insights_fallback":
        return False
    text = (row.get("text") or "").strip()
    err = row.get("error")
    status = row.get("http_status")
    return bool(text) and not err and isinstance(status, int) and 200 <= status < 300


def _load_curated_doc_metadata(docs_path: Path) -> tuple[set[str], dict[str, set[str]]]:
    allowed_doc_ids: set[str] = set()
    by_task: dict[str, set[str]] = defaultdict(set)
    if not docs_path.exists():
        return allowed_doc_ids, by_task

    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                continue
            if not _is_direct_curated_doc_record(row):
                continue
            allowed_doc_ids.add(doc_id)
            for task_id in row.get("task_ids") or []:
                if isinstance(task_id, str) and task_id:
                    by_task[task_id].add(doc_id)
    return allowed_doc_ids, by_task


class CuratedWebRepository:
    """Optional direct-fetch web sidecar keyed by task id."""

    def __init__(self, docs_path: str | Path, chunk_paths: list[str | Path]) -> None:
        self.docs_path = Path(docs_path).expanduser()
        self.chunk_paths = [Path(path).expanduser() for path in chunk_paths]
        self.allowed_doc_ids, self.doc_ids_by_task = _load_curated_doc_metadata(self.docs_path)
        existing_paths = [path for path in self.chunk_paths if path.exists()]
        self.curated_web_doc_ids = set(self.allowed_doc_ids)
        self.searcher = None
        if existing_paths and self.allowed_doc_ids:
            try:
                self.searcher = HybridSearcher.from_paths(
                    existing_paths,
                    doc_ids=self.allowed_doc_ids,
                )
            except ValueError:
                self.searcher = None

    @property
    def enabled(self) -> bool:
        return self.searcher is not None and bool(self.allowed_doc_ids)

    def allowed_for_task(self, task_id: str | None) -> set[str]:
        if not task_id:
            return set()
        return self.doc_ids_by_task.get(task_id, set()) & self.curated_web_doc_ids


@dataclass(slots=True)
class PrivacyHelperConfig:
    local_index_root: str
    local_index_mmap_mode: str
    embedding_base_url: str
    embedding_model: str
    embedding_timeout_s: float
    embedding_batch_size: int
    browsecomp_index_glob: str
    browsecomp_corpus: str
    browsecomp_index_device: str
    browsecomp_bm25_recall_k: int
    browsecomp_dense_recall_k: int
    browsecomp_max_chars: int
    curated_web_docs: str
    curated_web_chunks: list[str]


class PrivacyHelperBackend:
    """Long-lived retrieval state shared across helper requests."""

    def __init__(self, config_obj: PrivacyHelperConfig) -> None:
        self.config = config_obj
        self.metrics = EndpointMetrics()
        self.embeddings = EmbeddingService(
            base_url=config_obj.embedding_base_url,
            model_name=config_obj.embedding_model,
            timeout_seconds=config_obj.embedding_timeout_s,
            batch_size=config_obj.embedding_batch_size,
        )
        self.local_indices = LocalIndexRepository(
            index_root=config_obj.local_index_root,
            mmap_mode=config_obj.local_index_mmap_mode,
        )
        self.browsecomp = BrowseCompHybridSearcher(
            index_glob=config_obj.browsecomp_index_glob,
            dataset_name=config_obj.browsecomp_corpus,
            embedder=self.embeddings,
            bm25_recall_k=config_obj.browsecomp_bm25_recall_k,
            dense_recall_k=config_obj.browsecomp_dense_recall_k,
            max_chars=config_obj.browsecomp_max_chars,
            index_device=config_obj.browsecomp_index_device,
        )
        self.curated = CuratedWebRepository(
            docs_path=config_obj.curated_web_docs,
            chunk_paths=config_obj.curated_web_chunks,
        )
        self._task_web_cache: dict[str, Any] = {}
        self._task_web_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "privacy_helper",
            "embedding_base_url": self.embeddings.base_url,
            "embedding_model": self.embeddings.model_name,
            "embedding_server_ok": self.embeddings.health(),
            "local_index_root": str(self.local_indices.index_root),
            "local_task_count": self.local_indices.total_tasks,
            "loaded_local_tasks": self.local_indices.loaded_tasks,
            "browsecomp_size": self.browsecomp.size,
            "browsecomp_index_device": self.config.browsecomp_index_device,
            "curated_enabled": self.curated.enabled,
            "curated_doc_count": len(self.curated.curated_web_doc_ids),
            "curated_task_count": len(self.curated.doc_ids_by_task),
            "metrics": self.metrics.snapshot(),
        }

    def embed(self, texts: list[str], model_name: str | None = None) -> list[list[float]]:
        return self.embeddings.embed_texts(texts, model_name=model_name).tolist()

    def search_local(
        self,
        task_id: str,
        query: str,
        k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        query_vector = self.embeddings.embed_query(query)
        task_index = self.local_indices.get(task_id)
        return task_index.search(query_vector, k=k, threshold=threshold)

    def _task_web_searcher(self, task_id: str | None) -> Any:
        task_key = task_id or ""
        with self._task_web_lock:
            cached = self._task_web_cache.get(task_key)
            if cached is not None:
                return cached

            searchers: list[tuple[Any, float]] = [
                (self.browsecomp, config.BROWSECOMP_WEB_WEIGHT),
            ]

            curated_ids = self.curated.allowed_for_task(task_id)
            if self.curated.searcher is not None and curated_ids:
                searchers.append(
                    (
                        FilteredSearcher(self.curated.searcher, allowed_doc_ids=curated_ids),
                        config.CURATED_WEB_WEIGHT,
                    )
                )

            if len(searchers) == 1:
                searcher = searchers[0][0]
            else:
                searcher = WeightedRRFSearcher(searchers, rrf_k=config.RRF_K)

            self._task_web_cache[task_key] = searcher
            return searcher

    def search_browsecomp(
        self,
        query: str,
        task_id: str | None,
        k: int,
        max_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        searcher = self._task_web_searcher(task_id)
        hits = searcher.search(query=query, k=k, mode="hybrid")
        char_limit = int(max_chars) if max_chars is not None else self.browsecomp.max_chars

        results: list[dict[str, Any]] = []
        for hit in hits:
            text = hit.text
            if len(text) > char_limit:
                text = text[:char_limit] + "\n\n[truncated]"
            meta = hit.meta or {}
            results.append(
                {
                    "docid": hit.doc_id,
                    "score": float(hit.score),
                    "url": meta.get("url", ""),
                    "text": text,
                    "source": meta.get("source", ""),
                    "raw_docid": meta.get("raw_docid", ""),
                }
            )
        return results
