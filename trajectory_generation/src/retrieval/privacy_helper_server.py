#!/usr/bin/env python3
"""Reusable helper service for privacy-agent retrieval workloads.

This server keeps the expensive retrieval-side models loaded once and exposes a
small HTTP API that actor jobs can call:

- ``POST /embed`` for general embedding generation
- ``POST /local/search`` for task-local private document retrieval
- ``POST /browsecomp/search`` for BrowseComp-Plus hybrid retrieval
"""

import argparse
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_generation import config
from trajectory_generation.src.retrieval.browsecomp import BrowseCompHybridSearcher

logger = logging.getLogger(__name__)


class SentenceEmbeddingService:
    """SentenceTransformer wrapper used for local-doc and overlay embeddings."""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self._lock = threading.Lock()

        model_kwargs: dict[str, Any] = {}
        init_kwargs: dict[str, Any] = {
            "device": device,
            "tokenizer_kwargs": {"padding_side": "left"},
        }

        if device.startswith("cuda"):
            import torch

            model_kwargs = {
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.float16,
            }
            init_kwargs["model_kwargs"] = model_kwargs

        try:
            self.model = SentenceTransformer(model_name, **init_kwargs)
        except Exception:
            init_kwargs.pop("model_kwargs", None)
            self.model = SentenceTransformer(model_name, **init_kwargs)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._lock:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(embeddings, dtype=np.float32)


class LocalTaskIndex:
    """Read-only task-local index built by PipelineRL privacy_agent."""

    def __init__(self, storage_dir: Path, mmap_mode: str = "r") -> None:
        self.storage_dir = storage_dir
        self.documents = json.loads((storage_dir / "documents.json").read_text(encoding="utf-8"))
        self.doc_ids = json.loads((storage_dir / "index.json").read_text(encoding="utf-8")).get("doc_ids", [])
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


class PrivacyHelperBackend:
    """Backend object shared across all HTTP requests."""

    def __init__(
        self,
        *,
        local_index_root: str,
        local_embedding_model: str,
        local_embedding_device: str,
        local_index_mmap_mode: str,
        browsecomp_index_glob: str,
        browsecomp_model_name: str,
        browsecomp_corpus: str,
        browsecomp_device: str,
        browsecomp_bm25_recall_k: int,
        browsecomp_dense_recall_k: int,
        browsecomp_max_chars: int,
    ) -> None:
        self.embedder = SentenceEmbeddingService(
            model_name=local_embedding_model,
            device=local_embedding_device,
        )
        self.local_indices = LocalIndexRepository(
            index_root=local_index_root,
            mmap_mode=local_index_mmap_mode,
        )
        self.browsecomp = BrowseCompHybridSearcher(
            index_glob=browsecomp_index_glob,
            model_name=browsecomp_model_name,
            dataset_name=browsecomp_corpus,
            device=browsecomp_device,
            bm25_recall_k=browsecomp_bm25_recall_k,
            dense_recall_k=browsecomp_dense_recall_k,
            max_chars=browsecomp_max_chars,
        )

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "privacy_helper",
            "local_embedding_model": self.embedder.model_name,
            "local_embedding_device": self.embedder.device,
            "local_index_root": str(self.local_indices.index_root),
            "local_task_count": self.local_indices.total_tasks,
            "loaded_local_tasks": self.local_indices.loaded_tasks,
            "browsecomp_size": self.browsecomp.size,
        }

    def embed(self, texts: list[str], model_name: str | None = None) -> list[list[float]]:
        if model_name and model_name != self.embedder.model_name:
            raise ValueError(
                f"Helper was started with embedding model {self.embedder.model_name}, "
                f"but request asked for {model_name}."
            )
        return self.embedder.embed(texts).tolist()

    def search_local(
        self,
        task_id: str,
        query: str,
        k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed([query])[0]
        task_index = self.local_indices.get(task_id)
        return task_index.search(query_vector, k=k, threshold=threshold)

    def search_browsecomp(
        self,
        query: str,
        k: int,
        max_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        hits = self.browsecomp.search(query=query, k=k, mode="hybrid")
        char_limit = int(max_chars) if max_chars is not None else self.browsecomp.max_chars
        results: list[dict[str, Any]] = []
        for hit in hits:
            text = hit.text
            if len(text) > char_limit:
                text = text[:char_limit] + "\n\n[truncated]"
            results.append(
                {
                    "docid": hit.doc_id,
                    "score": float(hit.score),
                    "url": (hit.meta or {}).get("url", ""),
                    "text": text,
                }
            )
        return results


class _Handler(BaseHTTPRequestHandler):
    backend: PrivacyHelperBackend | None = None

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        return data

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        backend = self.backend
        if backend is None:
            self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "backend unavailable"})
            return
        self._write_json(HTTPStatus.OK, backend.health())

    def do_POST(self) -> None:
        backend = self.backend
        if backend is None:
            self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "backend unavailable"})
            return

        try:
            data = self._read_json()
            if self.path == "/embed":
                texts = data.get("texts")
                if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                    raise ValueError("texts must be a list of strings")
                payload = {
                    "embeddings": backend.embed(texts, model_name=data.get("model")),
                }
                self._write_json(HTTPStatus.OK, payload)
                return

            if self.path == "/local/search":
                task_id = str(data.get("task_id", "")).strip()
                query = str(data.get("query", "")).strip()
                if not task_id or not query:
                    raise ValueError("task_id and query are required")
                hits = backend.search_local(
                    task_id,
                    query,
                    int(data.get("k", 10)),
                    float(data.get("threshold", 0.0)),
                )
                self._write_json(HTTPStatus.OK, {"hits": hits})
                return

            if self.path == "/browsecomp/search":
                query = str(data.get("query", "")).strip()
                if not query:
                    raise ValueError("query is required")
                max_chars = int(data["max_chars"]) if data.get("max_chars") is not None else None
                hits = backend.search_browsecomp(
                    query,
                    int(data.get("k", 5)),
                    max_chars,
                )
                self._write_json(HTTPStatus.OK, {"hits": hits})
                return

            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except FileNotFoundError as exc:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            logger.exception("helper request failed")
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reusable privacy-agent helper service.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8012, help="Port to bind to.")
    parser.add_argument(
        "--local-index-root",
        default=str(Path.home() / "privacy_agent" / "local_indices"),
        help="Root directory containing prebuilt per-task local indices.",
    )
    parser.add_argument(
        "--local-embedding-model",
        default="Qwen/Qwen3-Embedding-4B",
        help="SentenceTransformer model used for local-doc and overlay embeddings.",
    )
    parser.add_argument(
        "--local-embedding-device",
        default=config.BROWSECOMP_EMBEDDING_DEVICE,
        help="Device for the local embedding model.",
    )
    parser.add_argument(
        "--local-index-mmap-mode",
        default="r",
        help="np.load mmap_mode for local embedding matrices.",
    )
    parser.add_argument(
        "--browsecomp-index-glob",
        default=config.BROWSECOMP_EMBEDDING_INDEX_GLOB,
        help="Path glob for BrowseComp index shards.",
    )
    parser.add_argument(
        "--browsecomp-model-name",
        default=config.BROWSECOMP_EMBEDDING_MODEL,
        help="BrowseComp embedding model name.",
    )
    parser.add_argument(
        "--browsecomp-corpus",
        default=config.BROWSECOMP_DATASET_NAME,
        help="BrowseComp dataset name or local corpus path.",
    )
    parser.add_argument(
        "--browsecomp-device",
        default=config.BROWSECOMP_EMBEDDING_DEVICE,
        help="Device for BrowseComp query encoding.",
    )
    parser.add_argument(
        "--browsecomp-bm25-recall-k",
        type=int,
        default=config.BROWSECOMP_BM25_RECALL_K,
        help="BM25 recall fanout before reranking.",
    )
    parser.add_argument(
        "--browsecomp-dense-recall-k",
        type=int,
        default=config.BROWSECOMP_DENSE_RECALL_K,
        help="Dense recall fanout before reranking.",
    )
    parser.add_argument(
        "--browsecomp-max-chars",
        type=int,
        default=8000,
        help="Maximum text length returned per BrowseComp hit.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading privacy helper backend...")
    backend = PrivacyHelperBackend(
        local_index_root=args.local_index_root,
        local_embedding_model=args.local_embedding_model,
        local_embedding_device=args.local_embedding_device,
        local_index_mmap_mode=args.local_index_mmap_mode,
        browsecomp_index_glob=args.browsecomp_index_glob,
        browsecomp_model_name=args.browsecomp_model_name,
        browsecomp_corpus=args.browsecomp_corpus,
        browsecomp_device=args.browsecomp_device,
        browsecomp_bm25_recall_k=args.browsecomp_bm25_recall_k,
        browsecomp_dense_recall_k=args.browsecomp_dense_recall_k,
        browsecomp_max_chars=args.browsecomp_max_chars,
    )
    _Handler.backend = backend

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info("Privacy helper listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
