#!/usr/bin/env python3
"""Extend and merge the OpenRouter-vs-StepFun judge validation sample."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_stepfun_judge_validation_sample as sample


DEFAULT_DIR = sample.DEFAULT_OUTPUT_DIR


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("task_id")), str(row.get("chain_id")), str(row.get("rollout_id")))


def trace_key(row: dict[str, Any]) -> str:
    return str(row.get("trace_path") or "")


def balanced_sample(
    rows_by_condition: dict[str, list[dict[str, Any]]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    conditions = sorted(rows_by_condition)
    base_count = sample_size // len(conditions)
    remainder = sample_size % len(conditions)
    out: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        n_rows = base_count + (1 if index < remainder else 0)
        available = rows_by_condition[condition]
        if len(available) < n_rows:
            raise SystemExit(f"condition {condition} only has {len(available)} available rows, need {n_rows}")
        out.extend(rng.sample(available, n_rows))
    return out


def command_sample(args: argparse.Namespace) -> None:
    existing_rows: list[dict[str, Any]] = []
    for path in args.exclude_reference:
        existing_rows.extend(read_jsonl(path))
    excluded_traces = {trace_key(row) for row in existing_rows if trace_key(row)}

    source_conditions = sample.SOURCE_SETS[args.source_set]
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    pool_counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}
    pool: list[dict[str, Any]] = []
    for condition, artifact_root, model_label in source_conditions:
        rows = sample.condition_rows(condition, artifact_root, model_label)
        pool_counts[condition] = len(rows)
        available = [row for row in rows if trace_key(row) not in excluded_traces]
        available_counts[condition] = len(available)
        rows_by_condition[condition] = available
        pool.extend(available)

    if len(pool) < args.sample_size:
        raise SystemExit(f"sample_size={args.sample_size} exceeds available pool size={len(pool)}")

    rng = random.Random(args.seed)
    if args.sample_mode == "balanced_condition":
        selected = balanced_sample(rows_by_condition, args.sample_size, rng)
        sampling = (
            f"balanced random without replacement across {len(rows_by_condition)} source conditions, "
            "excluding existing validation traces"
        )
    else:
        selected = rng.sample(pool, args.sample_size)
        sampling = "uniform random without replacement over the remaining selected-source validation pool"

    selected = sorted(
        selected,
        key=lambda row: (
            str(row.get("source_condition")),
            str(row.get("split")),
            str(row.get("repeat")),
            str(row.get("task_id")),
            str(row.get("chain_id")),
        ),
    )
    for index, row in enumerate(selected):
        row["validation_sample_id"] = f"{args.sample_id_prefix}{index:04d}"

    manifest = [sample.manifest_row(row, str(row["validation_sample_id"])) for row in selected]
    trace_manifest = [
        {
            "sample_id": row["sample_id"],
            "trace_path": row["trace_path"],
            "source_condition": row["source_condition"],
            "overall_privacy_score_1_to_5": row["overall_privacy_score_1_to_5"],
            "overall_leaked_for_reward": row["overall_leaked_for_reward"],
        }
        for row in manifest
    ]
    prompt_dataset = [sample.prompt_dataset_row(row, str(row["validation_sample_id"])) for row in selected]
    summary = sample.summarize(selected, pool_size=len(pool), seed=args.seed, source_set=args.source_set, sampling=sampling)
    summary["excluded_existing_rows"] = len(existing_rows)
    summary["source_condition_pool_counts"] = pool_counts
    summary["source_condition_available_counts"] = available_counts

    stem = f"{args.name}_seed{args.seed}"
    write_jsonl(args.output_dir / f"{stem}.sample_manifest.jsonl", manifest)
    write_jsonl(args.output_dir / f"{stem}.sample_prompts.jsonl", prompt_dataset)
    write_jsonl(args.output_dir / f"{stem}.trace_manifest.jsonl", trace_manifest)
    (args.output_dir / f"{stem}.trace_manifest.txt").write_text(
        "\n".join(str(row["trace_path"]) for row in trace_manifest) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / f"{stem}.stepfun_reference.jsonl", selected)
    write_json(args.output_dir / f"{stem}.summary.json", summary)

    print(f"wrote additional sample: {args.output_dir / f'{stem}.trace_manifest.txt'}")
    print(json.dumps({"sample_size": len(selected), "conditions": summary["conditions"]}, indent=2))


def command_merge(args: argparse.Namespace) -> None:
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict[str, Any]] = []
    duplicates: list[tuple[str, str, str]] = []
    for path in args.inputs:
        for row in read_jsonl(path):
            row_key = validation_key(row)
            if row_key in seen:
                duplicates.append(row_key)
                if not args.allow_duplicates:
                    continue
            seen.add(row_key)
            merged.append(row)
    if duplicates and args.fail_on_duplicates:
        raise SystemExit(f"found {len(duplicates)} duplicate validation keys")
    write_jsonl(args.output, merged)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(merged),
                "unique_keys": len(seen),
                "duplicates_skipped_or_kept": len(duplicates),
                "conditions": dict(Counter(str(row.get("source_condition")) for row in merged if row.get("source_condition"))),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="sample additional validation rows")
    sample_parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    sample_parser.add_argument("--name", default="openrouter_add100")
    sample_parser.add_argument("--sample-size", type=int, default=100)
    sample_parser.add_argument("--seed", type=int, default=2026052503)
    sample_parser.add_argument("--source-set", choices=sorted(sample.SOURCE_SETS), default="cross_model")
    sample_parser.add_argument("--sample-mode", choices=["balanced_condition", "uniform_pool"], default="balanced_condition")
    sample_parser.add_argument("--sample-id-prefix", default="judgeval_extra_")
    sample_parser.add_argument(
        "--exclude-reference",
        type=Path,
        action="append",
        default=[DEFAULT_DIR / "stepfun_reference_sample.jsonl"],
        help="existing reference JSONL whose traces should be excluded; repeatable",
    )
    sample_parser.set_defaults(func=command_sample)

    merge_parser = subparsers.add_parser("merge", help="merge JSONL files by validation key")
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--allow-duplicates", action="store_true")
    merge_parser.add_argument("--fail-on-duplicates", action="store_true")
    merge_parser.set_defaults(func=command_merge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
