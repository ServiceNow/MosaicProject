#!/usr/bin/env python3
"""Publish generated chains as a versioned training dataset bundle."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any

from mosaicleaks.src.jsonl import read_jsonl_objects, write_jsonl_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Final chain JSONL to publish.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset bundle directory.")
    parser.add_argument("--dataset-name", default="mosaicleaks", help="Name recorded in manifest.json.")
    parser.add_argument("--version", default=None, help="Dataset version. Defaults to current UTC timestamp.")
    parser.add_argument(
        "--split-key",
        choices=["company", "task_id", "pattern", "chain_id"],
        default="company",
        help="Group key used for deterministic split assignment.",
    )
    parser.add_argument(
        "--test-group",
        action="append",
        default=[],
        help="Exact split-key value to reserve for test. May be passed multiple times.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation fraction over non-test groups.")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test fraction when --test-group is not set.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed.")
    parser.add_argument("--source-run-dir", type=Path, default=None, help="Optional generation run directory.")
    parser.add_argument("--allow-invalid", action="store_true", help="Include rows marked valid=false.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing bundle directory.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_objects(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl_objects(path, rows, sort_keys=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return None
    commit = result.stdout.strip()
    return commit or None


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else {}


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_example(row: dict[str, Any], allow_invalid: bool) -> dict[str, Any]:
    if row.get("valid") is False and not allow_invalid:
        raise ValueError(f"chain {row.get('chain_id', '<missing>')} is marked valid=false")

    meta = metadata(row)
    question = first_nonempty(row.get("global_question"), row.get("question"), row.get("numbered_questions"))
    answer = first_nonempty(row.get("global_answer"), row.get("answer"))
    chain_id = first_nonempty(row.get("chain_id"))
    pattern = first_nonempty(row.get("pattern"), meta.get("seed_pattern"))
    task_id = first_nonempty(row.get("task_id"), meta.get("task_id"), meta.get("seed_task_id"))
    company = first_nonempty(row.get("company"), meta.get("company"))
    hops = row.get("hops")

    if not chain_id:
        raise ValueError("missing chain_id")
    if not question:
        raise ValueError(f"chain {chain_id}: missing question")
    if not answer:
        raise ValueError(f"chain {chain_id}: missing answer")
    if not isinstance(hops, list) or not hops:
        raise ValueError(f"chain {chain_id}: missing hops")

    for idx, hop in enumerate(hops, start=1):
        if not isinstance(hop, dict):
            raise ValueError(f"chain {chain_id}: hop {idx} is not an object")
        for field in ("doc_id", "question", "answer"):
            if not first_nonempty(hop.get(field)):
                raise ValueError(f"chain {chain_id}: hop {idx} missing {field}")

    accepted_answers = row.get("accepted_answers")
    if not isinstance(accepted_answers, list) or not accepted_answers:
        accepted_answers = [answer]
    accepted_answers = [str(value).strip() for value in accepted_answers if str(value).strip()]
    if answer not in accepted_answers:
        accepted_answers.insert(0, answer)

    example = dict(row)
    example["chain_id"] = chain_id
    example["question"] = question
    example["answer"] = answer
    example["accepted_answers"] = accepted_answers
    example["pattern"] = pattern
    if task_id:
        example["task_id"] = task_id
    if company:
        example["company"] = company
    example["metadata"] = {**meta, "task_id": task_id, "company": company, "pattern": pattern}
    return example


def group_value(row: dict[str, Any], split_key: str) -> str:
    value = first_nonempty(row.get(split_key), metadata(row).get(split_key))
    if value:
        return value
    return str(row["chain_id"])


def assign_splits(
    rows: list[dict[str, Any]],
    split_key: str,
    test_groups: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_value(row, split_key)].append(row)

    # Split by groups, not rows, to avoid leaking related company/task context
    # across train, validation, and test files.
    explicit_test = set(test_groups)
    all_groups = sorted(grouped)
    rng = random.Random(seed)
    shuffled = list(all_groups)
    rng.shuffle(shuffled)

    if explicit_test:
        missing = explicit_test - set(grouped)
        if missing:
            raise ValueError(f"--test-group values not present for split key {split_key}: {sorted(missing)}")
        test = sorted(explicit_test)
        remaining = [group for group in shuffled if group not in explicit_test]
    else:
        n_test = max(1, round(len(shuffled) * test_ratio)) if len(shuffled) > 1 else 0
        test = sorted(shuffled[:n_test])
        remaining = shuffled[n_test:]

    n_val = max(1, round(len(remaining) * val_ratio)) if len(remaining) > 1 else 0
    val = sorted(remaining[:n_val])
    train = sorted(remaining[n_val:])

    split_by_group = {group: "train" for group in train}
    split_by_group.update({group: "val" for group in val})
    split_by_group.update({group: "test" for group in test})

    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[split_by_group[group_value(row, split_key)]].append(row)
    return splits


def split_stats(rows: list[dict[str, Any]], split_key: str) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "patterns": dict(Counter(str(row.get("pattern", "")) for row in rows)),
        split_key: dict(Counter(group_value(row, split_key) for row in rows)),
    }


def build_schema() -> dict[str, Any]:
    return {
        "format": "jsonl",
        "required_fields": [
            "chain_id",
            "question",
            "answer",
            "accepted_answers",
            "hops",
            "pattern",
            "metadata",
        ],
        "hop_required_fields": ["doc_id", "question", "answer"],
    }


def main() -> int:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test-ratio must be in [0, 1)")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace bundle files")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(args.input)
    rows = [normalize_example(row, allow_invalid=args.allow_invalid) for row in raw_rows]
    splits = assign_splits(
        rows,
        split_key=args.split_key,
        test_groups=args.test_group,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    write_jsonl(output_dir / "dataset.jsonl", rows)
    for split, split_rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_rows)

    version = args.version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stats = {
        "total": split_stats(rows, args.split_key),
        "splits": {split: split_stats(split_rows, args.split_key) for split, split_rows in splits.items()},
    }
    manifest = {
        "dataset_name": args.dataset_name,
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_input": str(args.input),
        "source_input_sha256": file_sha256(args.input),
        "source_run_dir": str(args.source_run_dir) if args.source_run_dir else "",
        "code_commit": current_commit(),
        "split_key": args.split_key,
        "test_groups": args.test_group,
        "seed": args.seed,
        "files": {
            "dataset": "dataset.jsonl",
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl",
            "manifest": "manifest.json",
            "stats": "dataset_stats.json",
            "schema": "schema.json",
        },
        "counts": {split: len(split_rows) for split, split_rows in splits.items()} | {"dataset": len(rows)},
    }

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "dataset_stats.json", stats)
    write_json(output_dir / "schema.json", build_schema())

    print(json.dumps({"output_dir": str(output_dir), "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    main()
