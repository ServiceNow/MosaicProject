"""Simplified validation helpers for the current trajectory generator."""

from __future__ import annotations

import re
from dataclasses import dataclass

from trajectory_generation import config
from trajectory_generation.normalization import canonical_match, canonical_tokens, canonicalize_surface

ANSWER_WITHOUT_DOC_PROMPT = """Is the answer to this question obvious or common knowledge?

Question: {question}

If the answer is widely known, a well-known fact, or can be determined through
basic reasoning without needing any specific document, provide the answer.
Otherwise output NOT_ANSWERABLE.

End your response with exactly these two lines:
Answer: <the actual answer if known, or the literal text NOT_ANSWERABLE>
Justification: <one sentence explaining why>
"""

BACKREF_CHECK_PROMPT = """Can you answer this question WITHOUT knowing what "{placeholder}" refers to?

Question: {question}

Rules:
- "{placeholder}" is an unknown value.
- Try hard to answer. If only one answer fits regardless of the unknown value,
  then the question does NOT depend on the back-reference.
- Output NOT_ANSWERABLE only if changing "{placeholder}" would really change the answer.

End your response with exactly these two lines:
Answer: <the actual answer if you can determine it, or the literal text NOT_ANSWERABLE>
Justification: <one sentence explaining why>
"""

SEARCH_LEAK_PROMPT = """You are a research assistant. You need to answer this question by searching the web.

Question: {question}

Write 2-3 web search queries you would use to find the answer.

End your response with a numbered list of queries like this:
1. first search query here
2. second search query here
3. third search query here
"""


@dataclass
class TrivialCheckResult:
    trivial: bool
    answer: str | None
    justification: str


@dataclass
class BackrefCheckResult:
    independent: bool
    answer: str | None
    justification: str


@dataclass
class SearchLeakResult:
    would_leak: bool
    queries: list[str]
    justification: str


def _strip_think_tags(raw: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _parse_answer_justification(raw: str) -> tuple[str, str]:
    cleaned = _strip_think_tags(raw)
    answer_matches = list(re.finditer(r"^Answer:\s*(.+)$", cleaned, re.MULTILINE | re.IGNORECASE))
    just_matches = list(re.finditer(r"^Justification:\s*(.+)$", cleaned, re.MULTILINE | re.IGNORECASE))
    answer_line = answer_matches[-1].group(1).strip() if answer_matches else ""
    justification = just_matches[-1].group(1).strip() if just_matches else ""
    if not answer_line or answer_line.startswith("<"):
        answer_line = "NOT_ANSWERABLE"
        if not justification:
            justification = "unparseable response"
    return answer_line, justification


def check_answerable_without_doc(
    question: str,
    llm,
    *,
    correct_answer: str | None = None,
) -> TrivialCheckResult:
    raw = llm.chat(
        [{"role": "user", "content": ANSWER_WITHOUT_DOC_PROMPT.format(question=question)}],
        temperature=0.0,
        max_tokens=config.SHORT_CALL_MAX_TOKENS,
        use_min_max_tokens=False,
    )
    answer_line, justification = _parse_answer_justification(raw)

    if answer_line.upper() == "NOT_ANSWERABLE":
        return TrivialCheckResult(trivial=False, answer=None, justification=justification)
    if correct_answer is not None and not canonical_match(answer_line, correct_answer):
        return TrivialCheckResult(
            trivial=False,
            answer=None,
            justification=(
                f"model answered {answer_line!r} vs correct {correct_answer!r} — "
                "override trivial=False"
            ),
        )
    return TrivialCheckResult(trivial=True, answer=answer_line, justification=justification)


def check_answer_needs_backref(question: str, prev_answer: str, llm) -> BackrefCheckResult:
    placeholder = "an unknown entity"
    blanked = question.replace(prev_answer, placeholder)
    if blanked == question:
        return BackrefCheckResult(independent=False, answer=None, justification="prev_answer not in question")

    raw = llm.chat(
        [{"role": "user", "content": BACKREF_CHECK_PROMPT.format(
            question=blanked,
            placeholder=placeholder,
        )}],
        temperature=0.0,
        max_tokens=config.SHORT_CALL_MAX_TOKENS,
        use_min_max_tokens=False,
    )
    answer_line, justification = _parse_answer_justification(raw)
    if answer_line.upper() == "NOT_ANSWERABLE":
        return BackrefCheckResult(independent=False, answer=None, justification=justification)
    return BackrefCheckResult(independent=True, answer=answer_line, justification=justification)


def check_search_leaks_bridge(question: str, bridge_value: str, llm) -> SearchLeakResult:
    raw = llm.chat(
        [{"role": "user", "content": SEARCH_LEAK_PROMPT.format(question=question)}],
        temperature=0.0,
        max_tokens=config.SHORT_CALL_MAX_TOKENS,
        use_min_max_tokens=False,
    )
    cleaned = _strip_think_tags(raw)
    queries = [m.group(1).strip() for m in re.finditer(r"^\d+\.\s*(.+)$", cleaned, re.MULTILINE)]

    bridge_tokens = canonical_tokens(bridge_value)
    query_token_union: set[str] = set()
    covered_queries = 0
    for query in queries:
        tokens = set(canonical_tokens(query))
        query_token_union.update(tokens)
        if bridge_tokens and all(tok in tokens for tok in bridge_tokens):
            covered_queries += 1
    would_leak = bool(bridge_tokens) and all(tok in query_token_union for tok in bridge_tokens)
    justification = (
        f"bridge {canonicalize_surface(bridge_value)!r} covered across queries; "
        f"{covered_queries}/{len(queries)} queries contain all tokens"
        if would_leak
        else f"bridge {canonicalize_surface(bridge_value)!r} not fully covered across {len(queries)} queries"
    )
    return SearchLeakResult(would_leak=would_leak, queries=queries, justification=justification)


def check_question(
    question: str,
    answer: str,
    *,
    prev_answers: list[str] | None = None,
) -> str | None:
    """Minimal deterministic validation.

    Keep this intentionally small: structure, length, and duplicate suppression.
    """
    q = question.strip()
    a = answer.strip()
    if not q:
        return "empty question"
    if not a:
        return "empty answer"
    if len(a.split()) > 5:
        return f"answer too long ({len(a.split())} words): {a!r}"
    if prev_answers:
        a_canonical = canonicalize_surface(a)
        for prev in prev_answers:
            if a_canonical and a_canonical == canonicalize_surface(prev):
                return f"answer {a!r} duplicates previous answer"
    return None


__all__ = [
    "BackrefCheckResult",
    "SearchLeakResult",
    "TrivialCheckResult",
    "check_answer_needs_backref",
    "check_answerable_without_doc",
    "check_question",
    "check_search_leaks_bridge",
]
