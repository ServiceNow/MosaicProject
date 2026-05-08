"""In-memory BM25 index. Zero external dependencies."""

import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunks import iter_chunks
from .types import RetrievalHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if len(t) >= 2]


@dataclass(frozen=True)
class SearchHit:
    doc_idx: int
    score: float


@dataclass(frozen=True)
class BM25State:
    k1: float
    b: float
    N: int
    doc_len: list[int]
    avgdl: float
    df: dict[str, int]
    inv: dict[str, list[tuple[int, int]]]


class BM25Index:
    """Minimal BM25 implementation (dependency-free)."""

    def __init__(
        self,
        docs: list[str] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        state: BM25State | None = None,
    ) -> None:
        if state is not None:
            self._docs = []
            self._k1 = float(state.k1)
            self._b = float(state.b)
            self._N = int(state.N)
            self._doc_len = list(state.doc_len)
            self._avgdl = float(state.avgdl)
            self._df = defaultdict(int, state.df)
            self._inv = defaultdict(list, state.inv)
            return

        if not docs:
            raise ValueError("BM25Index requires at least one doc")

        self._docs = docs
        self._k1 = float(k1)
        self._b = float(b)
        self._N = len(docs)
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = defaultdict(int)
        self._inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        total_len = 0
        for idx, text in enumerate(self._docs):
            tf = Counter(tokenize(text))
            dl = sum(tf.values())
            self._doc_len.append(dl)
            total_len += dl
            for term, f in tf.items():
                self._df[term] += 1
                self._inv[term].append((idx, f))
        self._avgdl = (total_len / self._N) if self._N else 0.0

    def _idf(self, df: int) -> float:
        return math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, k: int = 10) -> list[SearchHit]:
        if k <= 0:
            return []
        qtf = Counter(tokenize(query))
        if not qtf:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in qtf.keys():
            df = self._df.get(term)
            if not df:
                continue
            idf = self._idf(df)
            for doc_idx, f in self._inv.get(term, []):
                dl = self._doc_len[doc_idx]
                denom = f + self._k1 * (1.0 - self._b + self._b * (dl / (self._avgdl or 1.0)))
                scores[doc_idx] += idf * (f * (self._k1 + 1.0) / denom)

        if not scores:
            return []
        top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        return [SearchHit(doc_idx=int(doc_idx), score=float(score)) for doc_idx, score in top]

    def doc_text(self, doc_idx: int) -> str:
        return self._docs[doc_idx]

    @property
    def size(self) -> int:
        return self._N

    def to_state(self) -> BM25State:
        return BM25State(
            k1=self._k1,
            b=self._b,
            N=self._N,
            doc_len=list(self._doc_len),
            avgdl=self._avgdl,
            df=dict(self._df),
            inv={term: list(postings) for term, postings in self._inv.items()},
        )

    @classmethod
    def from_state(cls, state: BM25State) -> "BM25Index":
        return cls(state=state)


class BM25Searcher:
    def __init__(self, chunks_path: Path) -> None:
        self.chunks_path = chunks_path
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, Any] | None] = []

        for rec in iter_chunks(chunks_path):
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            cid = rec.get("chunk_id")
            did = rec.get("doc_id")
            if not cid or not did:
                continue
            self._chunk_ids.append(str(cid))
            self._doc_ids.append(str(did))
            self._texts.append(text)
            meta = rec.get("meta")
            self._metas.append(meta if isinstance(meta, dict) else None)

        if not self._texts:
            raise ValueError(f"No chunks loaded from {chunks_path}")

        self.index = BM25Index(self._texts)

    def search(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        hits = self.index.search(query, k=k)
        out: list[RetrievalHit] = []
        for h in hits:
            idx = h.doc_idx
            out.append(
                RetrievalHit(
                    chunk_id=self._chunk_ids[idx],
                    doc_id=self._doc_ids[idx],
                    score=float(h.score),
                    text=self._texts[idx],
                    meta=self._metas[idx],
                )
            )
        return out
