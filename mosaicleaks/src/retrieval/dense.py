"""Dense (embedding) index and embedding backends for retrieval."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.ndim == 1:
        denom = float(np.linalg.norm(x) or 1.0)
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:  # shape (n, d)
        ...

    def embed_query(self, query: str) -> np.ndarray:  # shape (d,)
        ...


class HashEmbeddingBackend:
    """Dependency-free hashing embeddings."""

    def __init__(self, *, dim: int = 256) -> None:
        self.dim = int(dim)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros((self.dim,), dtype=np.float32)
        for tok in text.split():
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            vec[idx] += 1.0
        return _l2_normalize(vec)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.stack([self._embed_one(t) for t in texts], axis=0)
        return mat

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_one(query)


class SentenceTransformerBackend:
    def __init__(self, model_name_or_path: str, *, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or use --backend hash."
            ) from exc
        self.model = SentenceTransformer(model_name_or_path)
        self.batch_size = int(batch_size)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], normalize_embeddings=True)
        return np.asarray(emb[0], dtype=np.float32)


class LocalQwen3Backend:
    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_query(self, query: str) -> np.ndarray:
        return self._embedder.encode_queries([query])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        saved = self._embedder.query_prefix
        self._embedder.query_prefix = ""
        try:
            return self._embedder.encode_queries(list(texts))
        finally:
            self._embedder.query_prefix = saved


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        batch_size: int = 256,
        query_prefix: str = "",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai python package is required for this backend.") from exc

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or ""
        if not api_key:
            if base_url is None:
                raise RuntimeError("No API key found and no base_url provided.")
            api_key = "DUMMY"

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model
        self.batch_size = int(batch_size)
        self.query_prefix = query_prefix

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            resp = self.client.embeddings.create(model=self.model, input=batch)
            for d in resp.data:
                all_vecs.append(np.asarray(d.embedding, dtype=np.float32))
        return _l2_normalize(np.stack(all_vecs, axis=0))

    def embed_query(self, query: str) -> np.ndarray:
        text = f"{self.query_prefix} {query}".strip() if self.query_prefix else query
        resp = self.client.embeddings.create(model=self.model, input=[text])
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        return _l2_normalize(vec)


@dataclass(frozen=True)
class DenseHit:
    idx: int
    score: float


class DenseIndex:
    def __init__(self, *, embeddings: np.ndarray, chunk_ids: list[str], doc_ids: list[str]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if embeddings.shape[0] != len(chunk_ids) or len(chunk_ids) != len(doc_ids):
            raise ValueError("embeddings/chunk_ids/doc_ids length mismatch")
        self.embeddings = _l2_normalize(embeddings.astype(np.float32, copy=False))
        self.chunk_ids = list(chunk_ids)
        self.doc_ids = list(doc_ids)
        self._idx_by_chunk_id = {cid: i for i, cid in enumerate(self.chunk_ids)}

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def chunk_index(self, chunk_id: str) -> int | None:
        return self._idx_by_chunk_id.get(chunk_id)

    def score_chunk(self, chunk_id: str, query_emb: np.ndarray) -> float | None:
        idx = self.chunk_index(chunk_id)
        if idx is None:
            return None
        q = _l2_normalize(query_emb)
        return float(self.embeddings[idx] @ q)

    def search(self, query_emb: np.ndarray, *, k: int = 10) -> list[DenseHit]:
        if k <= 0:
            return []
        q = _l2_normalize(query_emb).astype(np.float32, copy=False)
        scores = self.embeddings @ q
        k = min(int(k), int(scores.shape[0]))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [DenseHit(idx=int(i), score=float(scores[int(i)])) for i in idxs]

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            embeddings=self.embeddings,
            chunk_ids=np.asarray(self.chunk_ids),
            doc_ids=np.asarray(self.doc_ids),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "DenseIndex":
        data = np.load(str(path), allow_pickle=True)
        embeddings = data["embeddings"]
        chunk_ids = [str(x) for x in data["chunk_ids"].tolist()]
        doc_ids = [str(x) for x in data["doc_ids"].tolist()]
        return cls(embeddings=embeddings, chunk_ids=chunk_ids, doc_ids=doc_ids)
