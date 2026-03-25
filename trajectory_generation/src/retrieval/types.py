"""Shared data types for the retrieval module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    meta: dict[str, Any] | None = None
