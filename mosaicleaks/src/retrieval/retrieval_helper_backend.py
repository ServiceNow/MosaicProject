"""Backend state for the MosaicLeaks retrieval helper service."""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from mosaicleaks import config
from mosaicleaks.src.retrieval.bm25 import BM25Index
from mosaicleaks.src.retrieval.browsecomp import BrowseCompHybridSearcher
from mosaicleaks.src.retrieval.composite import (
    ExcludedSearcher,
    FilteredSearcher,
    FixedKwargSearcher,
    WeightedRRFSearcher,
)
from mosaicleaks.src.retrieval.dense import OpenAICompatibleBackend
from mosaicleaks.src.retrieval.hybrid import HybridSearcher


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
    """Shared client for an OpenAI-compatible embedding endpoint."""

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
    """Read-only task-local index used by the retrieval helper."""

    def __init__(self, storage_dir: Path, mmap_mode: str = "r") -> None:
        self.storage_dir = storage_dir
        self.documents = json.loads((storage_dir / "documents.json").read_text(encoding="utf-8"))
        self.doc_ids = json.loads((storage_dir / "index.json").read_text(encoding="utf-8")).get(
            "doc_ids", []
        )
        self.embeddings = np.load(storage_dir / "embeddings.npy", mmap_mode=mmap_mode)
        self._embedding_norms: np.ndarray | None = None
        self._bm25_doc_ids: list[str] = []
        bm25_texts: list[str] = []
        for doc_id in self.doc_ids:
            doc_info = self.documents.get(str(doc_id))
            if not isinstance(doc_info, dict):
                continue
            content = str(doc_info.get("content") or "")
            metadata = json.dumps(doc_info.get("metadata", {}), sort_keys=True)
            text = f"{content}\n{metadata}".strip()
            if text:
                self._bm25_doc_ids.append(str(doc_id))
                bm25_texts.append(text)
        self.bm25 = BM25Index(bm25_texts) if bm25_texts else None

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

    def bm25_search(self, query: str, k: int) -> list[dict[str, Any]]:
        if k <= 0 or self.bm25 is None:
            return []

        hits: list[dict[str, Any]] = []
        for bm25_hit in self.bm25.search(query, k=k):
            doc_id = self._bm25_doc_ids[int(bm25_hit.doc_idx)]
            doc_info = self.documents[doc_id]
            hits.append(
                {
                    "doc_id": doc_id,
                    "similarity_score": float(bm25_hit.score),
                    "bm25_score": float(bm25_hit.score),
                    "content": doc_info["content"],
                    "metadata": doc_info.get("metadata", {}),
                    "preview": doc_info.get("content_preview", doc_info["content"][:300]),
                    "timestamp": doc_info.get("timestamp"),
                    "retrieval_mode": "bm25",
                }
            )
        return hits

    def rrf_search(
        self,
        query_vector: np.ndarray,
        query: str,
        *,
        k: int,
        threshold: float,
        dense_k: int,
        bm25_k: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        dense_hits = self.search(query_vector, k=max(k, dense_k), threshold=threshold)
        bm25_hits = self.bm25_search(query, k=max(k, bm25_k))

        ranked_by_doc: dict[str, dict[str, int]] = defaultdict(dict)
        best_hit_by_doc: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            doc_id = str(hit.get("doc_id") or "")
            if not doc_id:
                continue
            ranked_by_doc[doc_id]["dense_rank"] = rank
            best_hit_by_doc[doc_id] = hit

        for rank, hit in enumerate(bm25_hits, start=1):
            doc_id = str(hit.get("doc_id") or "")
            if not doc_id:
                continue
            ranked_by_doc[doc_id]["bm25_rank"] = rank
            best_hit_by_doc.setdefault(doc_id, hit)

        base = max(1, int(rrf_k))
        scored: list[tuple[float, str]] = []
        for doc_id, ranks in ranked_by_doc.items():
            score = 0.0
            if "dense_rank" in ranks:
                score += 1.0 / (base + ranks["dense_rank"])
            if "bm25_rank" in ranks:
                score += 1.0 / (base + ranks["bm25_rank"])
            scored.append((score, doc_id))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, doc_id in scored[:k]:
            hit = dict(best_hit_by_doc[doc_id])
            hit["similarity_score"] = float(score)
            hit["rrf_score"] = float(score)
            hit["dense_rank"] = ranked_by_doc[doc_id].get("dense_rank")
            hit["bm25_rank"] = ranked_by_doc[doc_id].get("bm25_rank")
            hit["retrieval_mode"] = "rrf"
            results.append(hit)
        return results


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


def _load_doc_id_file(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return set()
    doc_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            doc_ids.add(line)
            if line.startswith("web/"):
                doc_ids.add(line.removeprefix("web/"))
            else:
                doc_ids.add(f"web/{line}")
    return doc_ids


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
class RetrievalHelperConfig:
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
    browsecomp_rank_mode: str = "hybrid"
    browsecomp_rrf_k: int = 20
    local_rank_mode: str = "dense"
    local_bm25_recall_k: int = 100
    local_dense_recall_k: int = 100
    local_rrf_k: int = 20
    browsecomp_exclude_doc_ids_file: str = ""
    browsecomp_max_doc_windows: int = 10
    browsecomp_reader_window_chars: int = 16000
    browsecomp_reader_window_overlap_chars: int = 1000
    threadpool_tokens: int = 16


class RetrievalHelperBackend:
    """Long-lived retrieval state shared across helper requests."""

    def __init__(self, config_obj: RetrievalHelperConfig) -> None:
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
        self.browsecomp_excluded_doc_ids = _load_doc_id_file(
            config_obj.browsecomp_exclude_doc_ids_file
        )
        window_chars = max(1, int(config_obj.browsecomp_reader_window_chars))
        overlap_chars = min(max(0, int(config_obj.browsecomp_reader_window_overlap_chars)), window_chars - 1)
        window_step_chars = max(1, window_chars - overlap_chars)
        max_doc_windows = max(0, int(config_obj.browsecomp_max_doc_windows))
        self.browsecomp_max_doc_chars = (
            overlap_chars + (max_doc_windows * window_step_chars)
            if max_doc_windows > 0
            else 0
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
            "service": "retrieval_helper",
            "embedding_base_url": self.embeddings.base_url,
            "embedding_model": self.embeddings.model_name,
            "embedding_server_ok": self.embeddings.health(),
            "local_index_root": str(self.local_indices.index_root),
            "local_rank_mode": self.config.local_rank_mode,
            "local_bm25_recall_k": self.config.local_bm25_recall_k,
            "local_dense_recall_k": self.config.local_dense_recall_k,
            "local_rrf_k": self.config.local_rrf_k,
            "local_task_count": self.local_indices.total_tasks,
            "loaded_local_tasks": self.local_indices.loaded_tasks,
            "browsecomp_size": self.browsecomp.size,
            "browsecomp_rank_mode": self.config.browsecomp_rank_mode,
            "browsecomp_bm25_recall_k": self.config.browsecomp_bm25_recall_k,
            "browsecomp_dense_recall_k": self.config.browsecomp_dense_recall_k,
            "browsecomp_rrf_k": self.config.browsecomp_rrf_k,
            "browsecomp_excluded_doc_count": len(self.browsecomp_excluded_doc_ids),
            "browsecomp_max_doc_windows": self.config.browsecomp_max_doc_windows,
            "browsecomp_max_doc_chars": self.browsecomp_max_doc_chars,
            "browsecomp_reader_window_chars": self.config.browsecomp_reader_window_chars,
            "browsecomp_reader_window_overlap_chars": self.config.browsecomp_reader_window_overlap_chars,
            "browsecomp_index_device": self.config.browsecomp_index_device,
            "browsecomp_dense_backend": self.browsecomp.dense_backend,
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
        mode = self.config.local_rank_mode
        if mode == "dense":
            return task_index.search(query_vector, k=k, threshold=threshold)
        if mode == "bm25":
            return task_index.bm25_search(query, k=k)
        if mode == "rrf":
            return task_index.rrf_search(
                query_vector,
                query,
                k=k,
                threshold=threshold,
                dense_k=self.config.local_dense_recall_k,
                bm25_k=self.config.local_bm25_recall_k,
                rrf_k=self.config.local_rrf_k,
            )
        raise ValueError(f"Unsupported local_rank_mode={mode!r}")

    def _task_web_searcher(self, task_id: str | None) -> Any:
        task_key = task_id or ""
        with self._task_web_lock:
            cached = self._task_web_cache.get(task_key)
            if cached is not None:
                return cached

            browsecomp_searcher: Any = self.browsecomp
            if self.browsecomp_excluded_doc_ids:
                browsecomp_searcher = ExcludedSearcher(
                    self.browsecomp,
                    excluded_doc_ids=self.browsecomp_excluded_doc_ids,
                )
            browsecomp_searcher = FixedKwargSearcher(
                browsecomp_searcher,
                mode=self.config.browsecomp_rank_mode,
                rrf_k=self.config.browsecomp_rrf_k,
            )

            searchers: list[tuple[Any, float]] = [
                (browsecomp_searcher, config.BROWSECOMP_WEB_WEIGHT),
            ]

            curated_ids = self.curated.allowed_for_task(task_id)
            if self.curated.searcher is not None and curated_ids:
                searchers.append(
                    (
                        FixedKwargSearcher(
                            FilteredSearcher(self.curated.searcher, allowed_doc_ids=curated_ids),
                            mode="bm25",
                        ),
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
        char_limit = int(max_chars) if max_chars is not None else self.browsecomp.max_chars
        if char_limit <= 0:
            char_limit = 0
        search_k = max(k, k * 20) if self.browsecomp_max_doc_chars > 0 else k
        hits = searcher.search(query=query, k=search_k, max_chars=char_limit)

        results: list[dict[str, Any]] = []
        for hit in hits:
            meta = hit.meta or {}
            source = str(meta.get("source", ""))
            if self.browsecomp_max_doc_chars > 0 and source.startswith("browsecomp"):
                raw_docid = str(meta.get("raw_docid") or hit.doc_id)
                doc = self.browsecomp._get_doc(raw_docid)
                if len(doc.get("text") or "") > self.browsecomp_max_doc_chars:
                    continue
            text = hit.text
            if char_limit > 0 and len(text) > char_limit:
                text = text[:char_limit] + "\n\n[truncated]"
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
            if len(results) >= k:
                break
        return results
