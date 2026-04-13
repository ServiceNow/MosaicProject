"""Data structures for the multi-hop chain builder."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HopRecord:
    """One entry in the per-hop history."""

    hop_number: int
    hop_type: str  # "L" or "W"
    question: str
    answer: str
    doc_id: str
    doc_text: str
    quote: str = ""  # Excerpt(s) from doc supporting the Q/A; use " | " to separate disjoint parts


@dataclass
class ChainState:
    """Full state of a chain being built."""

    pattern: str  # Document type pattern, e.g., "LWL"
    hop_history: list[HopRecord] = field(default_factory=list)
    global_question: str = ""
    global_answer: str = ""
    used_doc_ids: set[str] = field(default_factory=set)
    task_id: str | None = None
    company: str | None = None
    trace: list[dict] = field(default_factory=list)  # Per-step traces
    jumps_completed: int = 0  # Number of document transitions completed

    @property
    def current_hop(self) -> int:
        return len(self.hop_history)

    @property
    def is_complete(self) -> bool:
        """Complete when all document transitions are done."""
        return self.jumps_completed >= len(self.pattern) - 1


@dataclass
class VerificationResult:
    """Result from Step 7 chain verification."""

    no_docs_pass: bool  # Should be NOT_ANSWERABLE
    first_only_pass: bool  # Should be NOT_ANSWERABLE
    last_only_pass: bool  # Should be NOT_ANSWERABLE
    all_docs_pass: bool  # Should be ANSWERABLE with correct answer
    all_docs_answer: str  # The answer given with all docs
    is_valid: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chain:
    """A complete, verified multi-hop chain."""

    chain_id: str
    pattern: str
    hop_history: list[HopRecord]
    global_question: str
    global_answer: str
    verification: VerificationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
