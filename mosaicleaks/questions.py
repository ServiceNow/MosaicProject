"""Question generation helpers for MosaicLeaks chain construction."""

import logging
import re
import time

from mosaicleaks import config
from mosaicleaks.src.llm import LLMClient
from mosaicleaks.src.parsing import parse_qa_with_quote
from mosaicleaks.prompts import (
    PROMPT_CONSTRAINED,
    PROMPT_PICK,
    PROMPT_RANK,
)

logger = logging.getLogger(__name__)

DOC_TEXT_LIMIT = 8_000


def _deliberation_instruction(kind: str) -> str:
    if config.PROMPT_REASONING_MODE == "brief_options":
        if kind == "constrained":
            return (
                "Consider a few plausible question options briefly, choose the best honest one, "
                "and output only the final block."
            )
        return (
            "Consider a few plausible allowed-answer options briefly, choose the best honest one, "
            "and output only the final block."
        )

    if kind == "constrained":
        return "Think through a few possible questions first, then choose the best honest one."
    return "Think step by step through several allowed-answer options before choosing the best one."


def _build_constrained_prompt(
    doc_text: str,
    prev_question: str,
    prev_answer: str,
    target_answer: str,
    source_excerpt: str,
) -> str:
    prompt = PROMPT_CONSTRAINED.format(
        prev_question=prev_question.strip(),
        prev_answer=prev_answer,
        target_answer=target_answer,
        source_excerpt=source_excerpt[:1500],
        doc_text=doc_text[:DOC_TEXT_LIMIT],
    )
    return prompt.replace(
        "Think through a few possible questions first, then choose the best honest one.",
        _deliberation_instruction("constrained"),
    )


def _build_pick_prompt(
    target_excerpt: str,
    prev_question: str,
    prev_answer: str,
    bridge_value: str,
    source_excerpt: str,
    allowed_answers: list[str],
) -> str:
    prompt = PROMPT_PICK.format(
        prev_question=prev_question.strip(),
        prev_answer=prev_answer,
        bridge_value=bridge_value,
        source_excerpt=source_excerpt[:1500],
        target_excerpt=target_excerpt[:DOC_TEXT_LIMIT],
        allowed_answers=_build_answer_block(allowed_answers),
    )
    return prompt.replace(
        "Think step by step through several allowed-answer options before choosing the best one.",
        _deliberation_instruction("pick"),
    )


def _build_answer_block(allowed_answers: list[str]) -> str:
    if not allowed_answers:
        return "- (none)"
    return "\n".join(f"- {answer}" for answer in allowed_answers)


def _is_no_valid_question(raw: str) -> bool:
    return "NO_VALID_QUESTION" in raw.upper()


def _parse_constrained_raw(raw: str, target_answer: str) -> tuple[str, str, str]:
    if _is_no_valid_question(raw):
        raise ValueError("Constrained question rejected by model: NO_VALID_QUESTION")

    result = parse_qa_with_quote(raw)
    if result is None:
        raise ValueError(f"Constrained question parse failed. Raw: {raw[:500]}")

    question, answer, quote = result
    if answer.strip().lower() != target_answer.strip().lower():
        raise ValueError(
            f"Constrained answer mismatch: expected {target_answer!r}, got {answer!r}"
        )
    return question, target_answer, quote


def _parse_pick_raw(raw: str) -> tuple[str, str, str]:
    if _is_no_valid_question(raw):
        raise ValueError("Pick question rejected by model: NO_VALID_QUESTION")

    result = parse_qa_with_quote(raw)
    if result is None:
        raise ValueError(f"Pick question parse failed. Raw: {raw[:500]}")

    return result


def generate_question_constrained(
    doc_text: str,
    prev_question: str,
    prev_answer: str,
    target_answer: str,
    source_excerpt: str,
    llm: LLMClient,
) -> tuple[str, str, dict]:
    """Generate an intra-document question. Raises ValueError on failure."""
    prompt = _build_constrained_prompt(
        doc_text=doc_text,
        prev_question=prev_question,
        prev_answer=prev_answer,
        target_answer=target_answer,
        source_excerpt=source_excerpt,
    )
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
    )
    duration = round(time.time() - t0, 2)

    question, answer, quote = _parse_constrained_raw(raw, target_answer=target_answer)
    return question, answer, {
        "step": "question_constrained",
        "prompt_variant": "constrained_v1",
        "prompt": prompt,
        "raw_output": raw,
        "question": question,
        "answer": answer,
        "quote": quote,
        "duration": duration,
    }


def generate_question_pick(
    target_excerpt: str,
    prev_question: str,
    prev_answer: str,
    bridge_value: str,
    source_excerpt: str,
    allowed_answers: list[str],
    llm: LLMClient,
) -> tuple[str, str, dict]:
    """Generate an inter-document question. Raises ValueError on failure."""
    prompt = _build_pick_prompt(
        target_excerpt=target_excerpt,
        prev_question=prev_question,
        prev_answer=prev_answer,
        bridge_value=bridge_value,
        source_excerpt=source_excerpt,
        allowed_answers=allowed_answers,
    )
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
    )
    duration = round(time.time() - t0, 2)

    question, answer, quote = _parse_pick_raw(raw)
    return question, answer, {
        "step": "question_pick",
        "prompt_variant": "pick_v1",
        "prompt": prompt,
        "raw_output": raw,
        "question": question,
        "answer": answer,
        "quote": quote,
        "bridge_value": bridge_value,
        "allowed_answers": allowed_answers,
        "target_excerpt": target_excerpt[:DOC_TEXT_LIMIT],
        "duration": duration,
    }


def generate_question_constrained_batch(
    jobs: list[dict],
    llm: LLMClient,
    batch_size: int = config.GENERATION_BATCH_SIZE,
) -> list[tuple[str, str, dict] | Exception]:
    if not jobs:
        return []

    prompts = [
        _build_constrained_prompt(
            doc_text=job["doc_text"],
            prev_question=job["prev_question"],
            prev_answer=job["prev_answer"],
            target_answer=job["target_answer"],
            source_excerpt=job["source_excerpt"],
        )
        for job in jobs
    ]
    t0 = time.time()
    raws = llm.chat_batch(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
        batch_size=batch_size,
    )
    duration = round(time.time() - t0, 2)
    if len(raws) != len(jobs):
        raise RuntimeError(f"LLM batch returned {len(raws)} outputs for {len(jobs)} jobs")

    # Keep per-job parse failures with the batch results so successful rows can
    # still be used and failed rows can be inspected or retried.
    outputs: list[tuple[str, str, dict] | Exception] = []
    for job, prompt, raw in zip(jobs, prompts, raws):
        try:
            question, answer, quote = _parse_constrained_raw(raw, target_answer=job["target_answer"])
            outputs.append((
                question,
                answer,
                {
                    "step": "question_constrained",
                    "prompt_variant": "constrained_v1_batch",
                    "prompt": prompt,
                    "raw_output": raw,
                    "question": question,
                    "answer": answer,
                    "quote": quote,
                    "duration": duration,
                },
            ))
        except ValueError as exc:
            outputs.append(exc)
    return outputs


def generate_question_pick_batch(
    jobs: list[dict],
    llm: LLMClient,
    batch_size: int = config.GENERATION_BATCH_SIZE,
) -> list[tuple[str, str, dict] | Exception]:
    if not jobs:
        return []

    prompts = [
        _build_pick_prompt(
            target_excerpt=job["target_excerpt"],
            prev_question=job["prev_question"],
            prev_answer=job["prev_answer"],
            bridge_value=job["bridge_value"],
            source_excerpt=job["source_excerpt"],
            allowed_answers=job["allowed_answers"],
        )
        for job in jobs
    ]
    t0 = time.time()
    raws = llm.chat_batch(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
        batch_size=batch_size,
    )
    duration = round(time.time() - t0, 2)
    if len(raws) != len(jobs):
        raise RuntimeError(f"LLM batch returned {len(raws)} outputs for {len(jobs)} jobs")

    # Keep per-job parse failures with the batch results so successful rows can
    # still be used and failed rows can be inspected or retried.
    outputs: list[tuple[str, str, dict] | Exception] = []
    for job, prompt, raw in zip(jobs, prompts, raws):
        try:
            question, answer, quote = _parse_pick_raw(raw)
            outputs.append((
                question,
                answer,
                {
                    "step": "question_pick",
                    "prompt_variant": "pick_v1_batch",
                    "prompt": prompt,
                    "raw_output": raw,
                    "question": question,
                    "answer": answer,
                    "quote": quote,
                    "bridge_value": job["bridge_value"],
                    "allowed_answers": job["allowed_answers"],
                    "target_excerpt": job["target_excerpt"][:DOC_TEXT_LIMIT],
                    "duration": duration,
                },
            ))
        except ValueError as exc:
            outputs.append(exc)
    return outputs


def rank_questions(
    candidates: list[dict],
    prev_question: str,
    prev_answer: str,
    llm: LLMClient,
) -> list[int]:
    """Rank candidates best-first. Returns indices into candidates."""
    if len(candidates) <= 1:
        return list(range(len(candidates)))

    cand_lines = []
    for i, cand in enumerate(candidates, start=1):
        cand_lines.append(
            f"{i}. BRIDGE: {cand.get('bridge_value', '')}\n"
            f"   QUESTION: {cand['question']}\n"
            f"   ANSWER: {cand['answer']}\n"
            f"   QUOTE: {cand.get('quote', '')}"
        )
    prompt = PROMPT_RANK.format(
        prev_question=prev_question,
        prev_answer=prev_answer,
        candidates="\n".join(cand_lines),
    )

    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=config.SHORT_CALL_MAX_TOKENS,
        top_p=config.DEFAULT_TOP_P,
        use_min_max_tokens=False,
    )

    numbers = [int(x) for x in re.findall(r"\d+", raw)]
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
    logger.info("rank_questions: %s", unique)
    return unique
