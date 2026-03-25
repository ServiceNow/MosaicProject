"""Question generation using improved prompts."""

from __future__ import annotations

import logging
import re
import time

from trajectory_generation.src.llm import LLMClient
from trajectory_generation.src.parsing import parse_qa_with_quote

from trajectory_generation import config
from trajectory_generation.prompts import PROMPT_CONSTRAINED, PROMPT_PICK, PROMPT_RANK

logger = logging.getLogger(__name__)

DOC_TEXT_LIMIT = 8_000


def generate_question_constrained(
    doc_text: str,
    prev_answer: str,
    target_answer: str,
    llm: LLMClient,
) -> tuple[str, str, dict]:
    """Generate intra-doc question. Returns (question, answer, trace)."""
    prompt = PROMPT_CONSTRAINED.format(
        prev_answer=prev_answer,
        target_answer=target_answer,
        doc_text=doc_text[:DOC_TEXT_LIMIT],
    )
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
        top_k=config.DEFAULT_TOP_K,
    )
    duration = round(time.time() - t0, 2)

    result = parse_qa_with_quote(raw)
    if result is None:
        raise ValueError(f"Constrained question parse failed. Raw: {raw[:500]}")

    question, answer, quote = result
    return question, answer, {
        "step": "question_constrained",
        "question": question,
        "answer": answer,
        "quote": quote,
        "duration": duration,
    }


def generate_question_pick(
    doc_text: str,
    prev_answer: str,
    entity_list: list[str],
    llm: LLMClient,
) -> tuple[str, str, dict]:
    """Generate inter-doc question. Returns (question, answer, trace)."""
    entity_str = "\n".join(f"- {e}" for e in entity_list)
    prompt = PROMPT_PICK.format(
        prev_answer=prev_answer,
        entity_list=entity_str,
        doc_text=doc_text[:DOC_TEXT_LIMIT],
    )
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
        top_k=config.DEFAULT_TOP_K,
    )
    duration = round(time.time() - t0, 2)

    result = parse_qa_with_quote(raw)
    if result is None:
        raise ValueError(f"Pick question parse failed. Raw: {raw[:500]}")

    question, answer, quote = result
    return question, answer, {
        "step": "question_pick",
        "question": question,
        "answer": answer,
        "quote": quote,
        "entity_list": entity_list,
        "duration": duration,
    }


def rank_questions(
    candidates: list[dict],
    prev_answer: str,
    llm: LLMClient,
) -> list[int]:
    """Rank candidates best-first. Returns indices."""
    if len(candidates) <= 1:
        return list(range(len(candidates)))

    cand_str = "\n".join(
        f"{i + 1}. Q: {c['question']}\n   A: {c['answer']}"
        for i, c in enumerate(candidates)
    )
    prompt = PROMPT_RANK.format(prev_answer=prev_answer, candidates=cand_str)

    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
        top_k=config.DEFAULT_TOP_K,
    )

    numbers = [int(x) for x in re.findall(r'\d+', raw)]
    indices = [n - 1 for n in numbers if 1 <= n <= len(candidates)]
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    for i in range(len(candidates)):
        if i not in seen:
            unique.append(i)
    return unique
