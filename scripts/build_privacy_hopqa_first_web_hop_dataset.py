#!/usr/bin/env python3
"""Build a first-hop-only dataset from Privacy HopQA web-first chains."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset")
DEFAULT_OUTPUT_DIR = Path("/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset")
SOURCE_BASENAME = "privacy_hopqa_finalized_2026-05-07_answer_variants"
OUTPUT_BASENAME = "privacy_hopqa_first_web_hop_2026-05-07"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_path(source_dir: Path, split: str | None) -> Path:
    if split is None:
        path = source_dir / f"{SOURCE_BASENAME}.jsonl"
        if path.exists():
            return path
        matches = sorted(source_dir.glob("*.jsonl"))
    else:
        path = source_dir / f"{SOURCE_BASENAME}_{split}.jsonl"
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


def transform(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    hops = row.get("hops") or []
    if not hops:
        return None
    first = dict(hops[0])
    if hop_type(row, first) != "W":
        return None

    source_chain_id = str(row.get("chain_id") or row.get("id") or row.get("problem_id") or index)
    question = str(first.get("question") or row.get("question") or row.get("global_question") or "")
    answer = str(first.get("answer") or row.get("answer") or row.get("global_answer") or "")
    variants = dedupe(as_list(first.get("accepted_answer_variants")) + as_list(answer))
    alternates = dedupe(as_list(first.get("alternate_valid_answers")))

    first["hop_number"] = 1
    first["hop_type"] = "W"
    first["question"] = question
    first["answer"] = answer
    first["accepted_answer_variants"] = variants
    first["alternate_valid_answers"] = alternates

    out = dict(row)
    out["id"] = f"{OUTPUT_BASENAME}_{index:04d}"
    out["problem_id"] = f"privacy_hopqa_first_web_hop_{source_chain_id}"
    out["chain_id"] = f"{source_chain_id}_first_web_hop"
    out["source_chain_id"] = source_chain_id
    out["source_problem_id"] = row.get("problem_id")
    out["source_pattern"] = row.get("pattern")
    out["source_n_hops"] = row.get("n_hops") or len(hops)
    out["dataset"] = "privacy_hopqa_first_web_hop"
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
    out["hops"] = [first]
    return out


def build_split(rows: list[dict[str, Any]], start_index: int) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    next_index = start_index
    for row in rows:
        converted = transform(row, next_index)
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, next_index = build_split(read_jsonl(split_path(args.source_dir, None)), 1)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        split_rows[split], next_index = build_split(read_jsonl(split_path(args.source_dir, split)), next_index)

    write_jsonl(args.output_dir / f"{OUTPUT_BASENAME}.jsonl", all_rows)
    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{OUTPUT_BASENAME}_{split}.jsonl", rows)

    manifest = {
        "dataset": "privacy_hopqa_first_web_hop",
        "source_dir": str(args.source_dir),
        "source_basename": SOURCE_BASENAME,
        "output_basename": OUTPUT_BASENAME,
        "selection_rule": "Keep only chains whose first hop is web; retain only that first hop.",
        "all": summarize(all_rows),
        "splits": {split: summarize(rows) for split, rows in split_rows.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Privacy HopQA first-web-hop dataset\n\n"
        "Derived from the finalized Privacy HopQA split. Each example keeps only the first hop "
        "when that hop is a web hop, preserving the canonical train/val/test assignment.\n\n"
        f"- Source: `{args.source_dir / (SOURCE_BASENAME + '.jsonl')}`\n"
        f"- Full dataset: `{args.output_dir / (OUTPUT_BASENAME + '.jsonl')}`\n"
        f"- Train: `{args.output_dir / (OUTPUT_BASENAME + '_train.jsonl')}`\n"
        f"- Val: `{args.output_dir / (OUTPUT_BASENAME + '_val.jsonl')}`\n"
        f"- Test: `{args.output_dir / (OUTPUT_BASENAME + '_test.jsonl')}`\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
