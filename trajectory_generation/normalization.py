"""Normalization helpers for trajectory generation.

These helpers intentionally keep a simple, human-readable canonical form:
- percentages normalize to ``30%``
- month abbreviations normalize to full month names
- weak modifiers like ``about`` are dropped
- punctuation and spacing are simplified
"""

from __future__ import annotations

import re

_MODIFIERS = {
    "about",
    "around",
    "approximately",
    "approx",
    "approx.",
    "nearly",
    "roughly",
}

_MONTH_MAP = {
    "jan": "january",
    "jan.": "january",
    "january": "january",
    "feb": "february",
    "feb.": "february",
    "february": "february",
    "mar": "march",
    "mar.": "march",
    "march": "march",
    "apr": "april",
    "apr.": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "jun.": "june",
    "june": "june",
    "jul": "july",
    "jul.": "july",
    "july": "july",
    "aug": "august",
    "aug.": "august",
    "august": "august",
    "sep": "september",
    "sep.": "september",
    "sept": "september",
    "sept.": "september",
    "september": "september",
    "oct": "october",
    "oct.": "october",
    "october": "october",
    "nov": "november",
    "nov.": "november",
    "november": "november",
    "dec": "december",
    "dec.": "december",
    "december": "december",
}

_PERCENT_RE = re.compile(
    r"\b(?:about|around|approximately|approx\.?|nearly|roughly)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent|percentage|pct)\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\s\"'`.,;:!?()\[\]{}-]+|[\s\"'`.,;:!?()\[\]{}-]+$")


def _normalize_month_token(token: str) -> str:
    return _MONTH_MAP.get(token, token)


def _normalize_percent_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        number = match.group(1)
        return f"{number}%"

    return _PERCENT_RE.sub(_repl, text)


def canonical_tokens(text: str) -> list[str]:
    """Return canonical tokens for dedupe and relaxed matching."""
    if not text:
        return []
    normalized = text.strip().lower()
    normalized = _normalize_percent_text(normalized)
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("–", " ")
    normalized = normalized.replace("—", " ")
    normalized = normalized.replace(",", " ")
    normalized = _SPACE_RE.sub(" ", normalized).strip()

    tokens: list[str] = []
    for raw_tok in normalized.split():
        tok = _PUNCT_EDGE_RE.sub("", raw_tok)
        if not tok or tok in _MODIFIERS:
            continue
        tok = _normalize_month_token(tok)
        tokens.append(tok)
    return tokens


def canonicalize_surface(text: str) -> str:
    """Canonical string form used for equality / dedupe checks."""
    return " ".join(canonical_tokens(text))


def canonical_match(a: str, b: str) -> bool:
    return bool(a and b) and canonicalize_surface(a) == canonicalize_surface(b)


def pick_preferred_surface(raw_values: list[str]) -> str:
    """Choose a representative surface form for a canonicalized group."""
    if not raw_values:
        return ""

    def _score(value: str) -> tuple[int, int]:
        canonical = canonicalize_surface(value)
        return (len(canonical), len(value))

    return sorted(raw_values, key=_score)[0]
