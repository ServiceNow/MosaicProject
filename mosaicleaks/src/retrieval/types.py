"""Shared data types for the retrieval module."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalHit:
    """Public retrieval result returned by all searcher adapters."""

    chunk_id: str
    doc_id: str
    score: float
    text: str
    meta: dict[str, Any] | None = None
