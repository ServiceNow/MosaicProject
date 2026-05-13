#!/usr/bin/env python3
"""Ask StepFun to re-audit first-web-hop answer variants.

This script is intentionally scoped to the one-hop first-web dataset used by
the debugging RL runs. It writes proposal rows by default and can optionally
apply accepted variants/rewrite suggestions to the first-web dataset files.
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
DEFAULT_OUTPUT = Path(
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/"
    "stepfun_firstweb_variant_proposals_2026-05-09.jsonl"
)
DEFAULT_BASE_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000"
DEFAULT_CANDIDATES = Path(
    "/home/toolkit/my_code/MosaicProject/runtime_analysis/firstweb_reward_plateau_candidate_chains.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
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
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
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


def truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ... [truncated]"


def first_hop(row: dict[str, Any]) -> dict[str, Any]:
    hops = row.get("hops") or []
    return hops[0] if hops and isinstance(hops[0], dict) else {}


def load_observed(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        chain_id = str(row.get("chain_id") or "")
        if chain_id:
            out[chain_id] = row
    return out


def observed_answers(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    values: list[str] = []
    for key in ("pred_counts", "reader_proposed_counts"):
        for item in row.get(key) or []:
            if isinstance(item, list) and item:
                values.append(str(item[0]))
            elif isinstance(item, tuple) and item:
                values.append(str(item[0]))
    return dedupe([value for value in values if value != "<blank>"])


def prompt_for(row: dict[str, Any], observed: dict[str, Any] | None) -> str:
    hop = first_hop(row)
    current_variants = dedupe(
        [
            row.get("answer"),
            row.get("global_answer"),
            *as_list(row.get("global_answer_variants")),
            *as_list(row.get("global_alternate_valid_answers")),
            *as_list(hop.get("accepted_answer_variants")),
            *as_list(hop.get("alternate_valid_answers")),
        ]
    )
    observed_values = observed_answers(observed)
    support = hop.get("supporting_span") or ""
    if hop.get("doc_ref"):
        support = f"{support}\nDocument path: {hop.get('doc_ref')}"
    payload = {
        "chain_id": row.get("chain_id"),
        "task_id": row.get("task_id"),
        "company": row.get("company"),
        "question": row.get("question"),
        "gold_answer": row.get("answer"),
        "current_accepted_answer_variants": current_variants,
        "observed_model_or_reader_answers_to_judge": observed_values[:24],
        "supporting_evidence": truncate(support, 2600),
    }
    return (
        "You are auditing exact-match answer aliases for a QA dataset.\n"
        "Return exactly one JSON object. Do not include markdown or prose.\n\n"
        "Goal: produce an exhaustive but conservative list of short strings that should be accepted by naive lowercase exact string match.\n"
        "Rules:\n"
        "- Every accepted string must answer the question exactly and refer to the same fact as the gold answer.\n"
        "- Prefer answers under 5 words. Do not include explanatory sentences.\n"
        "- Include spelling, punctuation, abbreviation, acronym, full-name, and unit-bearing variants when they are exactly valid.\n"
        "- If the gold is a surname but the evidence gives a full name, include both surname and full name.\n"
        "- If the gold is a country abbreviation, include common country-name and punctuation aliases.\n"
        "- If the question asks for a year, do not include a different date/month unless the short year is still present as its own accepted variant.\n"
        "- If the question asks for a day of the month, full dates containing the correct day can be accepted, but wrong months/dates must be rejected.\n"
        "- Numeric thresholds must preserve meaning: '50% or more' and 'at least 50%' are not the same as 'more than 50%'.\n"
        "- Reject observed answers that are merely plausible from another source but not supported by this question/evidence.\n"
        "- If the question is ambiguous or underspecified enough that another answer is naturally plausible, propose a minimal rewrite.\n\n"
        "Return this schema:\n"
        "{\n"
        '  "accepted_answer_variants": ["..."],\n'
        '  "alternate_valid_answers": ["..."],\n'
        '  "observed_answer_validity": {"observed answer": true_or_false},\n'
        '  "question_rewrite": string_or_null,\n'
        '  "issue_types": ["missing_alias" | "ambiguous_question" | "retrieval_distractor" | "no_issue" | "other"],\n'
        '  "rationale": "short explanation",\n'
        '  "confidence": 0.0_to_1.0\n'
        "}\n\n"
        "Audit object:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "JSON:"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in reversed(starts):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"No JSON object found in StepFun output: {text[:500]}")


async def call_stepfun(client: AsyncOpenAI, args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    if args.endpoint == "completion":
        response = await client.completions.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            stop=["\nAudit object:", "\nQuestion:", "\nReturn this schema:"],
        )
        return parse_json_object(response.choices[0].text or "")
    response = await client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": "Return exactly one JSON object in assistant content.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    return parse_json_object(response.choices[0].message.content or "")


def normalize_proposal(row: dict[str, Any], observed: dict[str, Any] | None, raw: dict[str, Any]) -> dict[str, Any]:
    hop = first_hop(row)
    current = dedupe(
        [
            row.get("answer"),
            row.get("global_answer"),
            *as_list(row.get("global_answer_variants")),
            *as_list(row.get("global_alternate_valid_answers")),
            *as_list(hop.get("accepted_answer_variants")),
            *as_list(hop.get("alternate_valid_answers")),
        ]
    )
    accepted = dedupe(current + as_list(raw.get("accepted_answer_variants")) + as_list(raw.get("valid_variants")))
    alternates = dedupe([value for value in as_list(raw.get("alternate_valid_answers")) + accepted if norm_key(value) not in {norm_key(v) for v in current}])
    observed_validity = raw.get("observed_answer_validity")
    if not isinstance(observed_validity, dict):
        observed_validity = {}
    issue_types = as_list(raw.get("issue_types")) or ["no_issue"]
    question_rewrite = raw.get("question_rewrite")
    if question_rewrite is not None:
        question_rewrite = str(question_rewrite).strip() or None
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "chain_id": row.get("chain_id"),
        "source_chain_id": row.get("source_chain_id"),
        "hop_number": 1,
        "task_id": row.get("task_id"),
        "company": row.get("company"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "accepted_answer_variants_before": current,
        "observed_answers": observed_answers(observed),
        "proposal": {
            "accepted_answer_variants": accepted,
            "alternate_valid_answers": alternates,
            "observed_answer_validity": observed_validity,
            "question_rewrite": question_rewrite,
            "issue_types": issue_types,
            "rationale": str(raw.get("rationale") or ""),
            "confidence": confidence,
            "needs_edit": bool(question_rewrite) or len(accepted) > len(current),
        },
        "raw_stepfun": raw,
    }


async def propose(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.dataset)
    observed_by_chain = load_observed(args.observed_candidates)
    if args.chain_id:
        allowed = set(args.chain_id)
        rows = [row for row in rows if str(row.get("chain_id")) in allowed or str(row.get("source_chain_id")) in allowed]
    if args.limit:
        rows = rows[: args.limit]
    if args.dry_run:
        for row in rows:
            print(prompt_for(row, observed_by_chain.get(str(row.get("chain_id")))))
            print("\n" + "=" * 88 + "\n")
        return 0

    completed: set[str] = set()
    if args.resume and args.output.exists():
        for record in read_jsonl(args.output):
            completed.add(str(record.get("chain_id")))
    rows = [row for row in rows if str(row.get("chain_id")) not in completed]
    if not rows:
        print("All proposal rows already complete.")
        return 0

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = AsyncOpenAI(api_key=args.api_key, base_url=base_url)
    mode = "a" if args.resume else "w"
    output = args.output.open(mode, encoding="utf-8")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, args.concurrency))
    done = 0

    async def one(row: dict[str, Any]) -> None:
        nonlocal done
        observed = observed_by_chain.get(str(row.get("chain_id")))
        prompt = prompt_for(row, observed)
        last_error: Exception | None = None
        for attempt in range(1, args.attempts + 1):
            try:
                async with sem:
                    raw = await call_stepfun(client, args, prompt)
                break
            except Exception as exc:
                last_error = exc
                if attempt >= args.attempts:
                    raise
                await asyncio.sleep(min(2 * attempt, 10))
        else:
            raise RuntimeError("StepFun call failed") from last_error
        record = normalize_proposal(row, observed, raw)
        async with lock:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            done += 1
            proposal = record["proposal"]
            print(
                f"[{done}/{len(rows)}] {record['chain_id']} "
                f"needs_edit={proposal['needs_edit']} variants={len(proposal['accepted_answer_variants'])} "
                f"issues={proposal['issue_types']}"
            )

    try:
        await asyncio.gather(*(one(row) for row in rows))
    finally:
        output.close()
    print(f"Wrote proposals to {args.output}")
    return 0


def apply_proposals(args: argparse.Namespace) -> int:
    proposals = {str(row.get("chain_id")): row for row in read_jsonl(args.proposals)}
    paths = [args.dataset]
    if args.apply_splits:
        stem = "privacy_hopqa_first_web_hop_2026-05-07"
        paths.extend(args.dataset.parent / f"{stem}_{split}.jsonl" for split in ("train", "val", "test"))
    for path in paths:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        changed = 0
        for row in rows:
            proposal_row = proposals.get(str(row.get("chain_id")))
            if not proposal_row:
                continue
            proposal = proposal_row.get("proposal") or {}
            accepted = dedupe(as_list(proposal.get("accepted_answer_variants")))
            alternates = dedupe(as_list(proposal.get("alternate_valid_answers")))
            if not accepted:
                continue
            hop = first_hop(row)
            hop["accepted_answer_variants"] = accepted
            hop["alternate_valid_answers"] = alternates
            sources = [{"variant": value, "sources": ["stepfun_firstweb_variant_recheck_2026-05-09"]} for value in accepted]
            hop["answer_variant_sources"] = sources
            row["global_answer_variants"] = accepted
            row["global_alternate_valid_answers"] = alternates
            row["global_answer_variant_sources"] = sources
            row["annotation_ts"] = "2026-05-09_firstweb_variant_recheck"
            rewrite = proposal.get("question_rewrite")
            if rewrite:
                hop["suggested_rewrite"] = rewrite
                hop.setdefault("question_issues", [])
                if "ambiguous_question" not in hop["question_issues"]:
                    hop["question_issues"].append("ambiguous_question")
            changed += 1
        write_jsonl(path, rows)
        print(f"updated {changed} rows in {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prop = sub.add_parser("propose")
    prop.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prop.add_argument("--observed-candidates", type=Path, default=DEFAULT_CANDIDATES)
    prop.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prop.add_argument("--chain-id", action="append")
    prop.add_argument("--limit", type=int, default=0)
    prop.add_argument("--dry-run", action="store_true")
    prop.add_argument("--resume", action="store_true")
    prop.add_argument("--base-url", default=DEFAULT_BASE_URL)
    prop.add_argument("--api-key", default="EMPTY")
    prop.add_argument("--model", default="step-3.5-flash")
    prop.add_argument("--endpoint", choices=["completion", "chat"], default="completion")
    prop.add_argument("--temperature", type=float, default=0.0)
    prop.add_argument("--max-tokens", type=int, default=4096)
    prop.add_argument("--timeout", type=float, default=120.0)
    prop.add_argument("--concurrency", type=int, default=16)
    prop.add_argument("--attempts", type=int, default=3)

    apply = sub.add_parser("apply")
    apply.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    apply.add_argument("--proposals", type=Path, default=DEFAULT_OUTPUT)
    apply.add_argument("--apply-splits", action="store_true")

    args = parser.parse_args()
    if args.command == "apply":
        return apply_proposals(args)
    return asyncio.run(propose(args))


if __name__ == "__main__":
    raise SystemExit(main())
