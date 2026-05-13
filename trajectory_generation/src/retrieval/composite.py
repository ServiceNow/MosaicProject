"""Composable retrieval adapters for trajectory generation."""

from dataclasses import dataclass
import logging
from typing import Any, Iterable

from .types import RetrievalHit

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _RankedHit:
    hit: RetrievalHit
    rank: int
    weight: float


class FilteredSearcher:
    """Restrict an underlying searcher to an allowlist of document IDs."""

    def __init__(self, base_searcher: Any, *, allowed_doc_ids: set[str]) -> None:
        self.base_searcher = base_searcher
        self.allowed_doc_ids = set(allowed_doc_ids)

    @property
    def size(self) -> int:
        return len(self.allowed_doc_ids)

    def search(self, query: str, *, k: int = 10, **kwargs) -> list[RetrievalHit]:
        if not self.allowed_doc_ids or k <= 0:
            return []
        search_k = max(k, getattr(self.base_searcher, "size", k))
        hits = self.base_searcher.search(query, k=search_k, **kwargs)
        filtered = [hit for hit in hits if hit.doc_id in self.allowed_doc_ids]
        return filtered[:k]


class ExcludedSearcher:
    """Exclude selected document IDs from an underlying searcher."""

    def __init__(self, base_searcher: Any, *, excluded_doc_ids: set[str], overfetch_factor: int = 20) -> None:
        self.base_searcher = base_searcher
        self.excluded_doc_ids = set(excluded_doc_ids)
        self.overfetch_factor = max(1, int(overfetch_factor))

    @property
    def size(self) -> int:
        base_size = int(getattr(self.base_searcher, "size", 0))
        return max(0, base_size - len(self.excluded_doc_ids))

    def search(self, query: str, *, k: int = 10, **kwargs) -> list[RetrievalHit]:
        if k <= 0:
            return []
        if not self.excluded_doc_ids:
            return self.base_searcher.search(query, k=k, **kwargs)
        base_size = int(getattr(self.base_searcher, "size", k))
        search_k = min(base_size, max(k, k * self.overfetch_factor))
        hits = self.base_searcher.search(query, k=search_k, **kwargs)
        filtered = [hit for hit in hits if hit.doc_id not in self.excluded_doc_ids]
        return filtered[:k]


class FixedKwargSearcher:
    """Force selected search kwargs for an underlying searcher."""

    def __init__(self, base_searcher: Any, **fixed_kwargs: Any) -> None:
        self.base_searcher = base_searcher
        self.fixed_kwargs = dict(fixed_kwargs)

    @property
    def size(self) -> int:
        return int(getattr(self.base_searcher, "size", 0))

    def search(self, query: str, *, k: int = 10, **kwargs) -> list[RetrievalHit]:
        merged_kwargs = dict(kwargs)
        merged_kwargs.update(self.fixed_kwargs)
        return self.base_searcher.search(query, k=k, **merged_kwargs)


class WeightedRRFSearcher:
    """Blend ranked lists from multiple retrieval sources."""

    def __init__(
        self,
        searchers: Iterable[tuple[Any, float]],
        *,
        rrf_k: int = 60,
    ) -> None:
        self._searchers = [(searcher, float(weight)) for searcher, weight in searchers if searcher is not None]
        self.rrf_k = int(rrf_k)

    @property
    def size(self) -> int:
        return sum(int(getattr(searcher, "size", 0)) for searcher, _ in self._searchers)

    def search(self, query: str, *, k: int = 10, **kwargs) -> list[RetrievalHit]:
        if k <= 0 or not self._searchers:
            return []

        ranked_by_doc: dict[str, list[_RankedHit]] = {}
        best_hit_by_doc: dict[str, RetrievalHit] = {}

        for searcher, weight in self._searchers:
            try:
                hits = searcher.search(query, k=k, **kwargs)
            except Exception as exc:
                logger.warning("Skipping failing searcher %s: %s", type(searcher).__name__, exc)
                continue
            for rank, hit in enumerate(hits, start=1):
                ranked_by_doc.setdefault(hit.doc_id, []).append(_RankedHit(hit=hit, rank=rank, weight=weight))
                prev = best_hit_by_doc.get(hit.doc_id)
                if prev is None or hit.score > prev.score:
                    best_hit_by_doc[hit.doc_id] = hit

        fused: list[RetrievalHit] = []
        for doc_id, ranked_hits in ranked_by_doc.items():
            fused_score = sum(rh.weight / (self.rrf_k + rh.rank) for rh in ranked_hits)
            best = best_hit_by_doc[doc_id]
            fused.append(RetrievalHit(
                chunk_id=best.chunk_id,
                doc_id=best.doc_id,
                score=float(fused_score),
                text=best.text,
                meta=best.meta,
            ))

        fused.sort(key=lambda hit: hit.score, reverse=True)
        return fused[:k]
