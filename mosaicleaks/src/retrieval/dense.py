"""Dense retrieval primitives used by the helper and BrowseComp searchers."""

import os
from pathlib import Path
from typing import Sequence

import numpy as np
from openai import OpenAI


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.ndim == 1:
        denom = float(np.linalg.norm(x) or 1.0)
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


class OpenAICompatibleBackend:
    """Embedding client for OpenAI-compatible /embeddings APIs."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        batch_size: int = 256,
        query_prefix: str = "",
    ) -> None:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or ""
        if not api_key:
            if base_url is None:
                raise RuntimeError("No API key found and no base_url provided.")
            # Local OpenAI-compatible servers usually ignore the key, but the
            # OpenAI client still requires a non-empty value.
            api_key = "not-used"

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model
        self.batch_size = int(batch_size)
        self.query_prefix = query_prefix

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
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


class DenseIndex:
    """In-memory cosine-search index over precomputed chunk embeddings."""

    def __init__(self, embeddings: np.ndarray, chunk_ids: list[str], doc_ids: list[str]) -> None:
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

    def search(self, query_emb: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if k <= 0 or self.size == 0:
            return []
        q = _l2_normalize(query_emb).astype(np.float32, copy=False)
        scores = self.embeddings @ q
        k = min(int(k), int(scores.shape[0]))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [(int(i), float(scores[int(i)])) for i in idxs]

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
        data = np.load(str(path), allow_pickle=False)
        embeddings = data["embeddings"]
        chunk_ids = [str(x) for x in data["chunk_ids"].tolist()]
        doc_ids = [str(x) for x in data["doc_ids"].tolist()]
        return cls(embeddings=embeddings, chunk_ids=chunk_ids, doc_ids=doc_ids)
