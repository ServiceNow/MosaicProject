"""Hybrid retrieval over one or more chunk JSONL files."""

from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .chunks import iter_chunks
from .dense import DenseIndex
from .types import RetrievalHit


class HybridSearcher:
    """BM25-first searcher over one or more prepared chunk JSONL files."""

    def __init__(
        self,
        chunks_path: Path | list[Path],
        dense_index_path: Path | None = None,
        doc_ids: set[str] | None = None,
    ) -> None:
        paths = chunks_path if isinstance(chunks_path, list) else [chunks_path]
        if dense_index_path is not None and len(paths) != 1:
            raise ValueError("dense_index_path is only supported when loading one chunk file")

        self.chunks_path = paths[0]
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, Any] | None] = []

        for path in paths:
            for rec in iter_chunks(path):
                text = (rec.get("text") or "").strip()
                cid = rec.get("chunk_id")
                did = rec.get("doc_id")
                if not text or not cid or not did:
                    continue
                if doc_ids is not None and str(did) not in doc_ids:
                    continue
                self._chunk_ids.append(str(cid))
                self._doc_ids.append(str(did))
                self._texts.append(text)
                meta = rec.get("meta")
                self._metas.append(meta if isinstance(meta, dict) else None)

        if not self._texts:
            raise ValueError(f"No chunks loaded from {paths}")

        self._idx_by_chunk_id = {cid: i for i, cid in enumerate(self._chunk_ids)}
        self.bm25 = BM25Index(self._texts)

        self.dense: DenseIndex | None = None
        if dense_index_path is not None and dense_index_path.exists():
            self.dense = DenseIndex.load_npz(dense_index_path)

    @classmethod
    def from_paths(
        cls,
        paths: list[Path],
        doc_ids: set[str] | None = None,
        dense_index_path: Path | None = None,
    ) -> "HybridSearcher":
        return cls(chunks_path=paths, dense_index_path=dense_index_path, doc_ids=doc_ids)

    @property
    def size(self) -> int:
        return len(self._texts)

    @staticmethod
    def _truncate(text: str, max_chars: int | None) -> str:
        if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[truncated]"

    def _hit(self, idx: int, score: float, max_chars: int | None = None) -> RetrievalHit:
        return RetrievalHit(
            chunk_id=self._chunk_ids[idx],
            doc_id=self._doc_ids[idx],
            score=float(score),
            text=self._truncate(self._texts[idx], max_chars),
            meta=self._metas[idx],
        )

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        embedder: Any | None = None,
        bm25_k: int = 200,
        dense_k: int = 200,
        max_chars: int | None = None,
    ) -> list[RetrievalHit]:
        if k <= 0:
            return []

        mode = mode.lower()
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Invalid mode: {mode}")

        if mode in {"dense", "hybrid"}:
            if self.dense is None:
                raise ValueError("Dense mode requested but dense index is not available.")
            if embedder is None:
                raise ValueError("Dense mode requested but no embedder was provided.")

        if mode == "bm25" or self.dense is None:
            hits = self.bm25.search(query, k=k)
            return [self._hit(idx, score, max_chars) for idx, score in hits]

        assert self.dense is not None
        assert embedder is not None
        # The dense branch is used only when the caller supplies the embedding
        # client for its environment, usually OpenAICompatibleBackend locally
        # or EmbeddingService inside the retrieval helper.
        query_emb = embedder.embed_query(query)

        cand_idxs: set[int] = set()
        if mode == "hybrid":
            bm25_hits = self.bm25.search(query, k=int(bm25_k))
            cand_idxs.update(idx for idx, _score in bm25_hits)

        dense_hits = self.dense.search(query_emb, k=int(dense_k))
        for dense_idx, _score in dense_hits:
            cid = self.dense.chunk_ids[dense_idx]
            idx = self._idx_by_chunk_id.get(cid)
            if idx is not None:
                cand_idxs.add(idx)

        scored: list[tuple[float, int]] = []
        for idx in cand_idxs:
            score = self.dense.score_chunk(self._chunk_ids[idx], query_emb)
            if score is not None:
                scored.append((float(score), idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._hit(idx, score, max_chars) for score, idx in scored[:k]]
