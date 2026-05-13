#!/usr/bin/env python3
"""Build single-web-hop datasets from Privacy HopQA chains."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset")
DEFAULT_OUTPUT_DIR = Path("/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset")
DEFAULT_SOURCE_BASENAME = "privacy_hopqa_finalized_2026-05-07_answer_variants"
DEFAULT_OUTPUT_BASENAME = "privacy_hopqa_first_web_hop_2026-05-07"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_path(source_dir: Path, source_basename: str, split: str | None) -> Path:
    if split is None:
        path = source_dir / f"{source_basename}.jsonl"
        if path.exists():
            return path
        matches = sorted(source_dir.glob("*.jsonl"))
    else:
        path = source_dir / f"{source_basename}_{split}.jsonl"
        if path.exists():
            return path
        plain_path = source_dir / f"{split}.jsonl"
        if plain_path.exists():
            return plain_path
        matches = sorted(source_dir.glob(f"*_{split}.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not resolve source path for split={split!r} in {source_dir}")
    return matches[0]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def hop_type(row: dict[str, Any], hop: dict[str, Any]) -> str:
    value = str(hop.get("hop_type") or hop.get("type") or hop.get("source_type") or "").upper()
    if value:
        return value[0]
    pattern = str(row.get("pattern") or row.get("hop_pattern") or "").upper()
    return pattern[0] if pattern else ""


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out


def render_question(template: str, prior_hops: list[dict[str, Any]]) -> str:
    rendered = str(template or "")
    for prior in prior_hops:
        hop_number = prior.get("hop_number")
        if hop_number is None:
            continue
        answer = str(prior.get("answer") or "").strip()
        rendered = rendered.replace(f"({hop_number})", answer)
    return " ".join(rendered.split())


def select_web_hop(row: dict[str, Any], selection: str) -> tuple[int, dict[str, Any]] | None:
    hops = row.get("hops") or []
    if not hops:
        return None
    if selection == "chain_first_hop_web":
        first = hops[0]
        if hop_type(row, first) != "W":
            return None
        return 0, first
    if selection == "first_web_hop_anywhere":
        for idx, hop in enumerate(hops):
            if hop_type(row, hop) == "W":
                return idx, hop
        return None
    raise ValueError(f"unknown selection mode: {selection}")


def transform(row: dict[str, Any], index: int, output_basename: str, selection: str) -> dict[str, Any] | None:
    hops = row.get("hops") or []
    selected = select_web_hop(row, selection)
    if selected is None:
        return None
    selected_idx, selected_hop = selected
    prior_hops = hops[:selected_idx]
    first = dict(selected_hop)

    source_chain_id = str(row.get("chain_id") or row.get("id") or row.get("problem_id") or index)
    question_template = str(first.get("question_templated") or first.get("question") or row.get("question") or "")
    question = render_question(question_template, prior_hops)
    if "(" in question and ")" in question and first.get("question"):
        question = render_question(str(first.get("question")), prior_hops)
    answer = str(first.get("answer") or row.get("answer") or row.get("global_answer") or "")
    variants = dedupe(as_list(first.get("accepted_answer_variants")) + as_list(answer))
    alternates = dedupe(as_list(first.get("alternate_valid_answers")))

    first["hop_number"] = 1
    first["hop_type"] = "W"
    first["question"] = question
    first["question_templated"] = question
    first["answer"] = answer
    first["accepted_answer_variants"] = variants
    first["alternate_valid_answers"] = alternates

    out = dict(row)
    out["id"] = f"{output_basename}_{index:04d}"
    out["problem_id"] = f"privacy_hopqa_first_web_hop_{source_chain_id}"
    out["chain_id"] = f"{source_chain_id}_first_web_hop"
    out["source_chain_id"] = source_chain_id
    out["source_problem_id"] = row.get("problem_id")
    out["source_hop_number"] = selected_hop.get("hop_number")
    out["source_question_templated"] = selected_hop.get("question_templated")
    out["source_prior_answers"] = [
        {
            "hop_number": prior.get("hop_number"),
            "answer": prior.get("answer"),
            "question": prior.get("question"),
        }
        for prior in prior_hops
    ]
    out["source_pattern"] = row.get("pattern")
    out["source_n_hops"] = row.get("n_hops") or len(hops)
    out["dataset"] = "privacy_hopqa_first_web_hop"
    out["selection_mode"] = selection
    out["pattern"] = "W"
    out["n_hops"] = 1
    out["n_local_hops"] = 0
    out["n_web_hops"] = 1
    out["n_transitions"] = 0
    out["local_to_web_transitions"] = 0
    out["web_to_local_transitions"] = 0
    out["global_question"] = question
    out["question"] = question
    out["task"] = question
    out["numbered_questions"] = f"1 (final) {question}"
    out["global_answer"] = answer
    out["answer"] = answer
    out["global_answer_variants"] = variants
    out["global_alternate_valid_answers"] = alternates
    out["global_answer_variant_sources"] = first.get("answer_variant_sources") or [
        {"variant": value, "sources": ["first_web_hop_transform"]} for value in variants
    ]
    out["hops"] = [first]
    return out


def build_split(
    rows: list[dict[str, Any]],
    start_index: int,
    output_basename: str,
    selection: str,
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    next_index = start_index
    for row in rows:
        converted = transform(row, next_index, output_basename, selection)
        if converted is None:
            continue
        out.append(converted)
        next_index += 1
    return out, next_index


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    companies = Counter(str(row.get("company") or "unknown") for row in rows)
    return {"count": len(rows), "companies": dict(sorted(companies.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-basename", default=DEFAULT_SOURCE_BASENAME)
    parser.add_argument("--output-basename", default=DEFAULT_OUTPUT_BASENAME)
    parser.add_argument(
        "--selection",
        choices=["chain_first_hop_web", "first_web_hop_anywhere"],
        default="chain_first_hop_web",
        help=(
            "chain_first_hop_web keeps only chains whose first hop is web; "
            "first_web_hop_anywhere keeps each chain's first web hop and fills prior-hop answers into the question."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, next_index = build_split(
        read_jsonl(split_path(args.source_dir, args.source_basename, None)),
        1,
        args.output_basename,
        args.selection,
    )
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        split_rows[split], next_index = build_split(
            read_jsonl(split_path(args.source_dir, args.source_basename, split)),
            next_index,
            args.output_basename,
            args.selection,
        )

    write_jsonl(args.output_dir / f"{args.output_basename}.jsonl", all_rows)
    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{args.output_basename}_{split}.jsonl", rows)

    manifest = {
        "dataset": "privacy_hopqa_first_web_hop",
        "source_dir": str(args.source_dir),
        "source_basename": args.source_basename,
        "output_basename": args.output_basename,
        "selection_mode": args.selection,
        "selection_rule": (
            "Keep only chains whose first hop is web; retain only that first hop."
            if args.selection == "chain_first_hop_web"
            else "Keep each chain's first web hop, fill prior-hop gold answers into the web-hop question, and retain only that hop."
        ),
        "all": summarize(all_rows),
        "splits": {split: summarize(rows) for split, rows in split_rows.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / f"manifest_{args.output_basename}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# Privacy HopQA first-web-hop dataset\n\n"
        "Derived from the finalized Privacy HopQA split. Each example keeps one web hop "
        "as a standalone web QA problem, preserving the canonical train/val/test assignment.\n\n"
        f"- Selection mode: `{args.selection}`\n"
        f"- Source: `{args.source_dir / (args.source_basename + '.jsonl')}`\n"
        f"- Full dataset: `{args.output_dir / (args.output_basename + '.jsonl')}`\n"
        f"- Train: `{args.output_dir / (args.output_basename + '_train.jsonl')}`\n"
        f"- Val: `{args.output_dir / (args.output_basename + '_val.jsonl')}`\n"
        f"- Test: `{args.output_dir / (args.output_basename + '_test.jsonl')}`\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
