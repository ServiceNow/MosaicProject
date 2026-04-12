"""Step 7: Verify complete chain with 4 answerability conditions.

Runs in parallel: no_docs, first_only, last_only, all_docs. Valid when
first three are NOT_ANSWERABLE and all_docs is ANSWERABLE with correct answer.
"""

from __future__ import annotations

import concurrent.futures
import time

from trajectory_generation.src.llm import LLMClient
from trajectory_generation.src.parsing import normalize_answer_text, parse_verification
from trajectory_generation.prompts import build_verification_prompt
from trajectory_generation.src.types import ChainState, HopRecord, VerificationResult
from trajectory_generation import config

_VERIFY_DOC_LIMIT = 6_000
_QUOTE_CONTEXT = 20_000
_VERIFY_MAX_TOKENS = 512


def _context_around_quote(hop: HopRecord) -> str:
    if not hop.quote:
        return hop.doc_text[:_VERIFY_DOC_LIMIT]

    haystack = hop.doc_text.lower()
    norm_haystack = ' '.join(haystack.split())
    parts = [p.strip() for p in hop.quote.split("|") if p.strip()] or [hop.quote]
    idx = -1
    found_len = 0
    use_norm = False
    for part in parts:
        n = part.lower()
        i = haystack.find(n)
        if i == -1:
            norm_part = ' '.join(n.split())
            i = norm_haystack.find(norm_part)
            if i != -1:
                idx, found_len, use_norm = i, len(norm_part), True
                break
        else:
            idx, found_len = i, len(part)
            break
    if idx == -1:
        return hop.doc_text[:_VERIFY_DOC_LIMIT]
    if use_norm:
        start = max(0, idx - _QUOTE_CONTEXT)
        end = min(len(norm_haystack), idx + found_len + _QUOTE_CONTEXT)
        return ' '.join(hop.doc_text.split())[start:end]
    start = max(0, idx - _QUOTE_CONTEXT)
    end = min(len(hop.doc_text), idx + found_len + _QUOTE_CONTEXT)
    return hop.doc_text[start:end]


def _verify_condition(
    condition_name: str, question: str, context: str, llm: LLMClient
) -> tuple[str, bool, str, dict]:
    prompt = build_verification_prompt(context, question)
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=_VERIFY_MAX_TOKENS,
        use_min_max_tokens=False,
    )
    duration = round(time.time() - t0, 2)
    result = parse_verification(raw)

    trace = {
        "step": "step7_verify",
        "condition": condition_name,
        "prompt": prompt,
        "raw_output": raw,
        "parsed": {"answerable": result.answerable, "content": result.content},
        "duration": duration,
    }
    return condition_name, result.answerable, result.content, trace


def verify_chain(state: ChainState, llm: LLMClient) -> tuple[VerificationResult, list[dict]]:
    question = state.global_question
    hops = state.hop_history

    all_contexts = [_context_around_quote(h) for h in hops]
    all_docs_text = "\n\n".join(
        f"=== Document {i+1} ===\n{ctx}" for i, ctx in enumerate(all_contexts)
    )
    first_doc_text = f"=== Document 1 ===\n{all_contexts[0]}" if all_contexts else ""
    last_doc_text = f"=== Document 1 ===\n{all_contexts[-1]}" if all_contexts else ""

    conditions = [
        ("no_docs", ""),
        ("first_only", first_doc_text),
        ("last_only", last_doc_text),
        ("all_docs", all_docs_text),
    ]

    results: dict[str, tuple[bool, str]] = {}
    traces: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_verify_condition, name, question, ctx, llm): name
            for name, ctx in conditions
        }
        for fut in concurrent.futures.as_completed(futures):
            name, answerable, content, trace = fut.result()
            results[name] = (answerable, content)
            traces.append(trace)

    no_docs_answerable, _ = results["no_docs"]
    first_answerable, _ = results["first_only"]
    last_answerable, _ = results["last_only"]
    all_answerable, all_answer = results["all_docs"]
    answer_match = (
        normalize_answer_text(all_answer) == normalize_answer_text(state.global_answer)
        if config.STRICT_VERIFY_ANSWER_MATCH and all_answerable
        else all_answerable
    )

    is_valid = (
        not no_docs_answerable
        and not first_answerable
        and not last_answerable
        and all_answerable
        and answer_match
    )

    verification = VerificationResult(
        no_docs_pass=not no_docs_answerable,
        first_only_pass=not first_answerable,
        last_only_pass=not last_answerable,
        all_docs_pass=all_answerable,
        all_docs_answer=all_answer,
        is_valid=is_valid,
        details={
            name: {"answerable": ans, "content": content}
            for name, (ans, content) in results.items()
        } | {
            "answer_match": answer_match,
            "expected_answer": state.global_answer,
        },
    )
    return verification, traces
