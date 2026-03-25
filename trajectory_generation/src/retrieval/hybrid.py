"""Hybrid retrieval over a chunk JSONL corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .bm25 import BM25Index
from .chunks import iter_chunks
from .dense import DenseIndex, EmbeddingBackend
from .types import RetrievalHit

Mode = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class Candidate:
    idx: int
    chunk_id: str
    doc_id: str
    text: str
    meta: dict[str, Any] | None


class HybridSearcher:
    def __init__(self, *, chunks_path: Path, dense_index_path: Path | None = None) -> None:
        self.chunks_path = chunks_path
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, Any] | None] = []

        for rec in iter_chunks(chunks_path):
            text = (rec.get("text") or "").strip()
            cid = rec.get("chunk_id")
            did = rec.get("doc_id")
            if not text or not cid or not did:
                continue
            self._chunk_ids.append(str(cid))
            self._doc_ids.append(str(did))
            self._texts.append(text)
            meta = rec.get("meta")
            self._metas.append(meta if isinstance(meta, dict) else None)

        if not self._texts:
            raise ValueError(f"No chunks loaded from {chunks_path}")

        self._idx_by_chunk_id = {cid: i for i, cid in enumerate(self._chunk_ids)}
        self.bm25 = BM25Index(self._texts)

        self.dense: DenseIndex | None = None
        if dense_index_path is not None and dense_index_path.exists():
            self.dense = DenseIndex.load_npz(dense_index_path)

    @classmethod
    def from_paths(cls, paths: list[Path], **kwargs) -> "HybridSearcher":
        if len(paths) == 1:
            return cls(chunks_path=paths[0], **kwargs)
        obj = cls.__new__(cls)
        obj._chunk_ids = []
        obj._doc_ids = []
        obj._texts = []
        obj._metas = []
        for path in paths:
            for rec in iter_chunks(path):
                text = (rec.get("text") or "").strip()
                cid = rec.get("chunk_id")
                did = rec.get("doc_id")
                if not text or not cid or not did:
                    continue
                obj._chunk_ids.append(str(cid))
                obj._doc_ids.append(str(did))
                obj._texts.append(text)
                meta = rec.get("meta")
                obj._metas.append(meta if isinstance(meta, dict) else None)
        if not obj._texts:
            raise ValueError(f"No chunks loaded from {paths}")
        obj._idx_by_chunk_id = {cid: i for i, cid in enumerate(obj._chunk_ids)}
        obj.bm25 = BM25Index(obj._texts)
        obj.dense = None
        obj.chunks_path = paths[0]
        return obj

    @property
    def size(self) -> int:
        return len(self._texts)

    def _candidate(self, idx: int) -> Candidate:
        return Candidate(
            idx=idx,
            chunk_id=self._chunk_ids[idx],
            doc_id=self._doc_ids[idx],
            text=self._texts[idx],
            meta=self._metas[idx],
        )

    def _hit(self, cand: Candidate, score: float) -> RetrievalHit:
        return RetrievalHit(
            chunk_id=cand.chunk_id,
            doc_id=cand.doc_id,
            score=float(score),
            text=cand.text,
            meta=cand.meta,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: Mode = "hybrid",
        embedder: EmbeddingBackend | None = None,
        bm25_k: int = 200,
        dense_k: int = 200,
    ) -> list[RetrievalHit]:
        if k <= 0:
            return []

        mode = mode.lower()  # type: ignore[assignment]
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Invalid mode: {mode}")

        if mode in {"dense", "hybrid"}:
            if self.dense is None:
                raise ValueError("Dense mode requested but dense index is not available.")
            if embedder is None:
                raise ValueError("Dense mode requested but no embedder was provided.")

        if mode == "bm25" or self.dense is None:
            hits = self.bm25.search(query, k=k)
            return [self._hit(self._candidate(h.doc_idx), h.score) for h in hits]

        assert self.dense is not None
        assert embedder is not None
        query_emb = embedder.embed_query(query)

        cand_idxs: set[int] = set()
        if mode == "hybrid":
            bm25_hits = self.bm25.search(query, k=int(bm25_k))
            cand_idxs.update(h.doc_idx for h in bm25_hits)

        dense_hits = self.dense.search(query_emb, k=int(dense_k))
        for dh in dense_hits:
            cid = self.dense.chunk_ids[dh.idx]
            idx = self._idx_by_chunk_id.get(cid)
            if idx is not None:
                cand_idxs.add(idx)

        scored: list[tuple[float, Candidate]] = []
        for i in cand_idxs:
            c = self._candidate(i)
            s = self.dense.score_chunk(c.chunk_id, query_emb)
            if s is not None:
                scored.append((float(s), c))

        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [self._hit(c, s) for s, c in scored[:k]]


def merge_results(
    *result_lists: list[RetrievalHit],
    k: int = 10,
) -> list[RetrievalHit]:
    best_by_doc: dict[str, RetrievalHit] = {}
    for results in result_lists:
        for hit in results:
            prev = best_by_doc.get(hit.doc_id)
            if prev is None or hit.score > prev.score:
                best_by_doc[hit.doc_id] = hit
    merged = sorted(best_by_doc.values(), key=lambda h: h.score, reverse=True)
    return merged[:k]
