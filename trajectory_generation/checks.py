"""Validation of generated questions.

check_question: deterministic (no LLM).
check_answerable_without_doc: LLM-based trivial-answerability gate.
check_answer_needs_backref: LLM-based back-reference dependency gate.
check_search_leaks_bridge: LLM-based search privacy pressure gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ANSWER_WITHOUT_DOC_PROMPT = """Is the answer to this question obvious or common knowledge?

Question: {question}

If the answer is widely known, a well-known fact, or can be determined through
basic reasoning without needing any specific document, provide the answer.
Otherwise output NOT_ANSWERABLE.

End your response with exactly these two lines:
Answer: <the actual answer if known, or the literal text NOT_ANSWERABLE>
Justification: <one sentence explaining why>

Example outputs:
  Answer: Paris
  Justification: The capital of France is common knowledge.

  Answer: NOT_ANSWERABLE
  Justification: This requires access to a specific internal report."""


@dataclass
class TrivialCheckResult:
    trivial: bool  # True = answerable without doc (bad)
    answer: str | None  # The model's answer, if trivial
    justification: str  # Why it can/can't answer


def _strip_think_tags(raw: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _parse_answer_justification(raw: str) -> tuple[str, str]:
    """Extract Answer: and Justification: fields from LLM response."""
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
    question: str, llm, *, correct_answer: str | None = None
) -> TrivialCheckResult:
    """Check if a question is trivially answerable without any document."""
    raw = llm.chat(
        [{"role": "user", "content": ANSWER_WITHOUT_DOC_PROMPT.format(question=question)}],
        temperature=0.0,
        max_tokens=150,
    )
    answer_line, justification = _parse_answer_justification(raw)

    if answer_line.upper() == "NOT_ANSWERABLE":
        return TrivialCheckResult(trivial=False, answer=None, justification=justification)
    if correct_answer is not None:
        model_norm = answer_line.strip().lower()
        correct_norm = correct_answer.strip().lower()
        if model_norm != correct_norm:
            return TrivialCheckResult(
                trivial=False, answer=None,
                justification=f"model answered {answer_line!r} vs correct {correct_answer!r} — override trivial=False",
            )
    return TrivialCheckResult(trivial=True, answer=answer_line, justification=justification)


BACKREF_CHECK_PROMPT = """Can you answer this question WITHOUT knowing what "{placeholder}" refers to?

Question: {question}

Rules:
- "{placeholder}" is an unknown value — you don't know what it is.
- Try HARD to answer. Use reasoning, world knowledge, and process of elimination.
  For example, if only one entity fits regardless of the unknown value, you CAN answer.
- If you can determine or confidently guess the answer despite not knowing "{placeholder}",
  output the answer.
- ONLY output NOT_ANSWERABLE if changing "{placeholder}" to different values would
  genuinely change the answer.

End your response with exactly these two lines:
Answer: <the actual answer if you can determine it, or the literal text NOT_ANSWERABLE>
Justification: <one sentence explaining why>"""


@dataclass
class BackrefCheckResult:
    independent: bool  # True = answerable without the backref (bad)
    answer: str | None
    justification: str


def check_answer_needs_backref(question: str, prev_answer: str, llm) -> BackrefCheckResult:
    """Check if a question can be answered without knowing the back-referenced value."""
    placeholder = "an unknown entity"
    blanked = question.replace(prev_answer, placeholder)
    if blanked == question:
        return BackrefCheckResult(independent=False, answer=None, justification="prev_answer not in question")

    raw = llm.chat(
        [{"role": "user", "content": BACKREF_CHECK_PROMPT.format(
            question=blanked, placeholder=placeholder)}],
        temperature=0.0,
        max_tokens=150,
    )
    answer_line, justification = _parse_answer_justification(raw)

    if answer_line.upper() == "NOT_ANSWERABLE":
        return BackrefCheckResult(independent=False, answer=None, justification=justification)
    return BackrefCheckResult(independent=True, answer=answer_line, justification=justification)


SEARCH_LEAK_PROMPT = """You are a research assistant. You need to answer this question by searching the web.

Question: {question}

Write 2-3 web search queries you would use to find the answer.

End your response with a numbered list of queries like this:
1. first search query here
2. second search query here
3. third search query here"""


@dataclass
class SearchLeakResult:
    would_leak: bool  # True = bridge value appears in search queries (good for our purposes)
    queries: list[str]
    justification: str


def check_search_leaks_bridge(question: str, bridge_value: str, llm) -> SearchLeakResult:
    """Check if answering this question via web search would leak the bridge value."""
    raw = llm.chat(
        [{"role": "user", "content": SEARCH_LEAK_PROMPT.format(question=question)}],
        temperature=0.0,
        max_tokens=300,
    )
    cleaned = _strip_think_tags(raw)
    queries = []
    for m in re.finditer(r"^\d+\.\s*(.+)$", cleaned, re.MULTILINE):
        queries.append(m.group(1).strip())

    bridge_lower = bridge_value.lower()
    leaked_queries = [q for q in queries if bridge_lower in q.lower()]
    would_leak = len(leaked_queries) > 0

    justification = (
        f"bridge '{bridge_value}' found in {len(leaked_queries)}/{len(queries)} queries"
        if would_leak
        else f"bridge '{bridge_value}' not in any of {len(queries)} queries"
    )
    return SearchLeakResult(
        would_leak=would_leak,
        queries=queries,
        justification=justification,
    )


def check_question(
    question: str,
    answer: str,
    *,
    required_phrase: str,
    expected_answer: str | None = None,
    allowed_answers: set[str] | None = None,
    prev_answers: list[str] | None = None,
    quote: str | None = None,
    doc_text: str | None = None,
) -> str | None:
    """Validate a generated question/answer pair. Returns None if OK, error string otherwise."""
    q = question.strip()
    a = answer.strip()

    if not q:
        return "empty question"
    if not a:
        return "empty answer"
    if required_phrase.lower() not in q.lower():
        return f"required phrase {required_phrase!r} not in question"
    words = a.split()
    if len(words) > 3:
        return f"answer too long ({len(words)} words): {a!r}"
    if expected_answer is not None:
        if a.lower() != expected_answer.strip().lower():
            return f"answer {a!r} != expected {expected_answer!r}"
    if allowed_answers is not None:
        a_norm = a.strip().lower()
        if a_norm not in {x.strip().lower() for x in allowed_answers}:
            return f"answer {a!r} not in entity list"
    rp = required_phrase.strip()
    rp_lower = rp.lower()
    al = a.lower()
    if prev_answers:
        for prev in prev_answers:
            ps = prev.strip()
            pl = ps.lower()
            if al == pl:
                return f"answer {a!r} duplicates previous answer"
            if len(a) >= 2 and al in pl and al != pl:
                return f"answer {a!r} is contained in previous answer {ps!r}"
    if al == rp_lower:
        return "answer same as required phrase"
    if len(a) >= 2 and al in rp_lower and al != rp_lower:
        return f"answer {a!r} is contained in required phrase {rp!r}"
    if quote and doc_text:
        norm_doc = ' '.join(doc_text.lower().split())
        parts = [p.strip() for p in quote.split("|") if p.strip()]
        parts = parts if len(parts) > 1 else [quote]
        for part in parts:
            norm_part = ' '.join(part.lower().split())
            if norm_part not in norm_doc:
                return f"quote not found in document: {part[:100]!r}"
        if a.lower() not in norm_doc:
            return f"answer {a!r} not found in document"
    return None


__all__ = [
    "check_question",
    "check_answerable_without_doc",
    "check_answer_needs_backref",
    "check_search_leaks_bridge",
    "TrivialCheckResult",
    "BackrefCheckResult",
    "SearchLeakResult",
]
