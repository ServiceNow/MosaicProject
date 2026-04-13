"""Shared data types for the retrieval module."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    meta: dict[str, Any] | None = None
