"""Parse structured outputs from LLM responses (extract <answer> tags)."""

from __future__ import annotations

import re
from typing import NamedTuple

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    """Extract content between <answer>...</answer> tags.

    Takes the LAST match to handle thinking models that output draft
    <answer> blocks inside <think>...</think> before the real answer.
    """
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return _THINK_RE.sub("", text).strip()


# Keys that start new fields (order matters for multiline capture)
_QA_KEYS = ["QUOTE", "REASONING", "QUESTION", "ANSWER"]


def _parse_kv(block: str, key: str) -> str:
    """Extract value for KEY: value from a block. Handles multiline values."""
    lines = block.splitlines()
    prefix = f"{key}:"
    next_keys = [f"{k}:" for k in _QA_KEYS if k != key]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.upper().startswith(prefix.upper()):
            continue
        value = stripped[len(prefix):].strip().strip('"')
        # Accumulate continuation lines until next key
        for j in range(i + 1, len(lines)):
            next_line = lines[j].strip()
            if not next_line:
                continue
            if any(next_line.upper().startswith(k.upper()) for k in next_keys):
                break
            value += " " + next_line.strip('"')
        return value
    return ""


def parse_qa(text: str) -> tuple[str, str] | None:
    """Parse QUESTION/ANSWER output from atomic prompts.

    Returns (question, answer) or None if unparseable.
    """
    text = _strip_think(text)

    ans = extract_answer(text)
    if ans:
        q = _parse_kv(ans, "QUESTION")
        a = _parse_kv(ans, "ANSWER")
        if q and a:
            return q, a

    q = _parse_kv(text, "QUESTION")
    a = _parse_kv(text, "ANSWER")
    if q and a:
        return q, a

    return None


def parse_qa_with_quote(text: str) -> tuple[str, str, str] | None:
    """Parse QUOTE/QUESTION/ANSWER output from atomic prompts.

    Returns (question, answer, quote) or None if unparseable.
    Quote may be empty if not provided.
    """
    text = _strip_think(text)

    ans = extract_answer(text)
    block = ans if ans else text

    q = _parse_kv(block, "QUESTION")
    a = _parse_kv(block, "ANSWER")
    quote = _parse_kv(block, "QUOTE")

    if q and a:
        return q, a, quote
    return None


class VerificationOutput(NamedTuple):
    answerable: bool
    content: str  # The answer if answerable, or the missing info if not


def parse_verification(text: str) -> VerificationOutput:
    """Parse verification output."""
    ans = extract_answer(text)
    if ans is None:
        ans = text

    upper = ans.upper().strip()
    if upper.startswith("ANSWERABLE"):
        content = ans.split(":", 1)[1].strip() if ":" in ans else ""
        return VerificationOutput(answerable=True, content=content)
    if upper.startswith("NOT_ANSWERABLE"):
        content = ans.split(":", 1)[1].strip() if ":" in ans else ""
        return VerificationOutput(answerable=False, content=content)

    if "NOT_ANSWERABLE" in upper or "NOT ANSWERABLE" in upper:
        return VerificationOutput(answerable=False, content=ans)
    return VerificationOutput(answerable=True, content=ans)
