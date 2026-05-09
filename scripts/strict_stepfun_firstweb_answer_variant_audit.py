#!/usr/bin/env python3
"""Strict StepFun audit for first-web-hop exact-match answer variants.

The audit is intentionally non-deterministic in the transformation sense:
StepFun proposes aliases from the gold document context, then a separate
StepFun judge validates each alias against the same gold context. The script
writes proposal rows only; a separate apply step should accept only judged-valid
variants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


DEFAULT_DATASET = Path(
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/"
    "privacy_hopqa_first_web_hop_2026-05-07.jsonl"
)
DEFAULT_CANDIDATES = Path(
    "/home/toolkit/my_code/MosaicProject/runtime_analysis/firstweb_reward_plateau_candidate_chains.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/"
    "stepfun_firstweb_strict_variant_audit_2026-05-09.jsonl"
)
DEFAULT_BASE_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def norm_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text


def dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = norm_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def first_hop(row: dict[str, Any]) -> dict[str, Any]:
    hops = row.get("hops") or []
    return hops[0] if hops and isinstance(hops[0], dict) else {}


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"No JSON object in response: {text[:500]}")


def observed_answers(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    values: list[str] = []
    for key in ("pred_counts", "reader_proposed_counts"):
        for item in row.get(key) or []:
            if isinstance(item, list) and item:
                values.append(str(item[0]))
    return dedupe([value for value in values if value != "<blank>"])


def load_observed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        chain_id = str(row.get("chain_id") or "")
        if chain_id:
            out[chain_id] = row
    return out


def gold_context(hop: dict[str, Any], max_chars: int) -> dict[str, Any]:
    doc_ref = str(hop.get("doc_ref") or "")
    supporting_span = str(hop.get("supporting_span") or "")
    doc_text = ""
    source = "supporting_span_only"
    if doc_ref:
        path = Path(doc_ref)
        if path.exists():
            doc_text = path.read_text(encoding="utf-8", errors="replace")
            source = "gold_document"
    if not doc_text:
        return {
            "doc_ref": doc_ref,
            "context_source": source,
            "supporting_span": supporting_span,
            "gold_document_context": supporting_span,
        }
    if len(doc_text) <= max_chars:
        context = doc_text
    else:
        idx = doc_text.find(supporting_span[:160]) if supporting_span else -1
        if idx == -1:
            context = doc_text[:max_chars]
            source = "gold_document_prefix"
        else:
            start = max(0, idx - max_chars // 2)
            end = min(len(doc_text), start + max_chars)
            start = max(0, end - max_chars)
            context = doc_text[start:end]
            source = "gold_document_window_around_support"
    return {
        "doc_ref": doc_ref,
        "context_source": source,
        "supporting_span": supporting_span,
        "gold_document_context": context,
    }


def current_variants(row: dict[str, Any]) -> list[str]:
    hop = first_hop(row)
    return dedupe(
        [
            row.get("answer"),
            row.get("global_answer"),
            *as_list(row.get("global_answer_variants")),
            *as_list(row.get("global_alternate_valid_answers")),
            hop.get("answer"),
            *as_list(hop.get("accepted_answer_variants")),
            *as_list(hop.get("alternate_valid_answers")),
        ]
    )


def proposal_prompt(row: dict[str, Any], observed: dict[str, Any] | None, max_doc_chars: int) -> str:
    hop = first_hop(row)
    payload = {
        "chain_id": row.get("chain_id"),
        "task_id": row.get("task_id"),
        "company": row.get("company"),
        "question": row.get("question"),
        "gold_answer": row.get("answer"),
        "current_accepted_answer_variants": current_variants(row),
        "observed_model_answers_to_judge": observed_answers(observed)[:32],
        "gold_context": gold_context(hop, max_doc_chars),
    }
    return (
        "You are proposing exact-match aliases for a QA dataset.\n"
        "Use only the supplied gold document context. Do not use outside knowledge.\n"
        "Return exactly one JSON object and no markdown.\n\n"
        "Goal: make exact lowercase string matching robust while avoiding label noise.\n"
        "Rules:\n"
        "- A proposed alias must answer the question exactly and must refer to the same fact as the gold answer.\n"
        "- Prefer short answers under 5 words. Do not include explanations as aliases.\n"
        "- For year questions, normally include only the bare four-digit year. Do not include worded years, 'the year 2022', '2022 CE', or '2022 AD'.\n"
        "- Include full dates or month/day forms only when the question asks for a date, month, or day, or the current accepted variants already include that form.\n"
        "- For non-year count answers under 100 with a clear unit, include digit, number-word, digit-with-unit, and number-word-with-unit forms.\n"
        "- Include aliases that a careful answerer might naturally produce from the gold context: abbreviations, full names, punctuation variants, number words, units, threshold phrasing, and date/month forms.\n"
        "- If an observed model answer is a plausible distractor but is not supported by the gold context for this exact question, mark it invalid.\n"
        "- If the question is ambiguous even with the gold context, propose a minimal rewrite, but do not change the gold answer.\n\n"
        "Return schema:\n"
        "{\n"
        '  "candidate_variants": ["..."],\n'
        '  "observed_answer_validity": {"observed answer": true_or_false},\n'
        '  "question_rewrite": string_or_null,\n'
        '  "rationale": "short explanation",\n'
        '  "confidence": 0.0_to_1.0\n'
        "}\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "JSON:"
    )


def judge_prompt(row: dict[str, Any], candidates: list[str], max_doc_chars: int) -> str:
    hop = first_hop(row)
    payload = {
        "chain_id": row.get("chain_id"),
        "question": row.get("question"),
        "gold_answer": row.get("answer"),
        "candidate_variants": candidates,
        "gold_context": gold_context(hop, max_doc_chars),
    }
    return (
        "You are judging proposed exact-match aliases for a QA dataset.\n"
        "Use only the supplied gold document context. Do not use outside knowledge.\n"
        "Return exactly one JSON object and no markdown.\n\n"
        "A variant is valid only if it is a short answer that means the exact same answer as the gold answer for this exact question.\n"
        "If the question asks who/person and the gold is a surname, a full name from the gold context for that same person is valid.\n"
        "If the gold is an abbreviation, acronym, country code, or punctuation variant, the full form from the gold context is valid when it identifies the same entity.\n"
        "For year questions, reject worded years, 'the year 2022', '2022 CE', and '2022 AD'; the bare four-digit year is the desired minimal answer.\n"
        "For non-year count questions, number-word and unit-bearing variants are valid when they preserve the exact count and unit.\n"
        "The original gold answer itself is valid unless it is not present in the candidate list.\n"
        "Reject variants that are broader, narrower, directionally different, from another time/source, or only plausible from outside context.\n\n"
        "Return schema:\n"
        "{\n"
        '  "valid_variants": ["..."],\n'
        '  "invalid_variants": ["..."],\n'
        '  "confidence": 0.0_to_1.0\n'
        "}\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "JSON:"
    )


async def call_model(client: AsyncOpenAI, args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    if args.endpoint == "chat":
        response = await client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Return exactly one JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        text = response.choices[0].message.content or ""
    else:
        response = await client.completions.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            stop=["\n\nYou are", "\n\nReturn schema:", "\n\nJSON:"],
        )
        text = response.choices[0].text or ""
    if not text.strip():
        raise ValueError("StepFun returned an empty response")
    return parse_json_object(text)


async def audit_one(
    row: dict[str, Any],
    observed: dict[str, Any] | None,
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    proposal = await call_model(client, args, proposal_prompt(row, observed, args.max_doc_chars))
    generated = dedupe(current_variants(row) + as_list(proposal.get("candidate_variants")))
    judge = await call_model(client, args, judge_prompt(row, generated, args.max_doc_chars))
    valid = dedupe(as_list(judge.get("valid_variants")))
    current_keys = {norm_key(value) for value in current_variants(row)}
    return {
        "chain_id": row.get("chain_id"),
        "source_chain_id": row.get("source_chain_id"),
        "task_id": row.get("task_id"),
        "company": row.get("company"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "current_variants": current_variants(row),
        "generated_variants": generated,
        "judged_valid_variants": valid,
        "new_judged_valid_variants": [value for value in valid if norm_key(value) not in current_keys],
        "observed_answers": observed_answers(observed),
        "observed_answer_validity": proposal.get("observed_answer_validity") or {},
        "question_rewrite": proposal.get("question_rewrite"),
        "proposal_rationale": proposal.get("rationale"),
        "proposal_confidence": proposal.get("confidence"),
        "judge_invalid_variants": judge.get("invalid_variants") or {},
        "judge_confidence": judge.get("confidence"),
    }


async def bounded_audit(
    rows: list[dict[str, Any]],
    observed_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    client = AsyncOpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key="EMPTY")
    semaphore = asyncio.Semaphore(args.concurrency)
    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        for row in read_jsonl(args.output):
            chain_id = str(row.get("chain_id") or "")
            if chain_id:
                done_ids.add(chain_id)
    if not args.resume and args.output.exists():
        args.output.unlink()

    async def worker(row: dict[str, Any]) -> None:
        chain_id = str(row.get("chain_id") or "")
        if chain_id in done_ids:
            return
        async with semaphore:
            result = await audit_one(row, observed_rows.get(chain_id), client, args)
            append_jsonl(args.output, result)
            print(
                f"{chain_id}: new_valid={len(result['new_judged_valid_variants'])} "
                f"proposal_conf={result['proposal_confidence']} judge_conf={result['judge_confidence']}",
                flush=True,
            )

    await asyncio.gather(*(worker(row) for row in rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="step-3.5-flash")
    parser.add_argument("--endpoint", choices=["completion", "chat"], default="completion")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-doc-chars", type=int, default=9000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    observed_rows = load_observed(args.candidates)
    asyncio.run(bounded_audit(rows, observed_rows, args))


if __name__ == "__main__":
    main()
