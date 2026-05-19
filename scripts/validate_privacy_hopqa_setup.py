#!/usr/bin/env python3
"""Fail-fast checks for the runbook Privacy HopQA q30 setup."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

DEFAULT_MOSAIC_ROOT = Path("/home/toolkit/my_code/MosaicProject")
DEFAULT_PIPELINERL_ROOT = Path("/home/toolkit/my_code/PipelineRL")
DATASET_BASENAME = "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review"
DEFAULT_SHARED_ROOT = Path("/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513")
EXPECTED_SPLIT_COUNTS = {"train": 559, "val": 98, "test": 344}
EXPECTED_FULL_CHAINS = 1001
EXPECTED_HOPS = 3403
EXPECTED_LOCAL_TASKS = 101
EXPECTED_BROWSECOMP_SHARDS = 4


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chain_id(row: dict) -> str:
    for key in ("chain_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError(f"Row has no chain_id/id keys: {sorted(row)[:12]}")


def check(condition: bool, label: str, failures: list[str], details: str = "") -> None:
    if condition:
        print(f"OK   {label}{(': ' + details) if details else ''}")
    else:
        print(f"FAIL {label}{(': ' + details) if details else ''}")
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic-root", type=Path, default=DEFAULT_MOSAIC_ROOT)
    parser.add_argument("--pipelinerl-root", type=Path, default=DEFAULT_PIPELINERL_ROOT)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    args = parser.parse_args()

    failures: list[str] = []
    dataset_root = args.mosaic_root / "data_files/privacy_hopqa_final_dataset"
    manifest_path = dataset_root / f"manifest_{DATASET_BASENAME}.json"
    full_path = dataset_root / f"{DATASET_BASENAME}.jsonl"
    split_paths = {split: dataset_root / f"{DATASET_BASENAME}_{split}.jsonl" for split in EXPECTED_SPLIT_COUNTS}

    check(args.mosaic_root.exists(), "MosaicProject root exists", failures, str(args.mosaic_root))
    check(args.pipelinerl_root.exists(), "PipelineRL root exists", failures, str(args.pipelinerl_root))
    check(manifest_path.exists(), "dataset manifest exists", failures, str(manifest_path))
    check(full_path.exists(), "full dataset exists", failures, str(full_path))
    for split, path in split_paths.items():
        check(path.exists(), f"{split} split exists", failures, str(path))

    if failures:
        return 1

    manifest = json.loads(manifest_path.read_text())
    full_rows = read_jsonl(full_path)
    check(manifest.get("target_version") == "2026-05-13_q30_final_residual_review", "manifest target version", failures, str(manifest.get("target_version")))
    check(manifest.get("chains") == EXPECTED_FULL_CHAINS, "manifest chain count", failures, str(manifest.get("chains")))
    check(manifest.get("hops") == EXPECTED_HOPS, "manifest hop count", failures, str(manifest.get("hops")))
    check(len(full_rows) == EXPECTED_FULL_CHAINS, "full dataset row count", failures, str(len(full_rows)))

    full_ids = {chain_id(row) for row in full_rows}
    split_ids: dict[str, set[str]] = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        rows = read_jsonl(split_paths[split])
        ids = {chain_id(row) for row in rows}
        split_ids[split] = ids
        check(len(rows) == expected_count, f"{split} row count", failures, str(len(rows)))
        check(len(ids) == len(rows), f"{split} chain ids unique", failures)

    union_ids = set().union(*split_ids.values())
    disjoint = not (split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] or split_ids["val"] & split_ids["test"])
    check(disjoint, "split ids are disjoint", failures)
    check(union_ids == full_ids, "split ids cover full dataset", failures, f"union={len(union_ids)} full={len(full_ids)}")

    artifact_manifest = args.shared_root / "artifact_manifest.json"
    local_root = args.shared_root / "local_indices_goldcoverage_20260519"
    browsecomp_glob = str(args.shared_root / "browsecomp_indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl")
    helper_inputs = args.shared_root / "helper_inputs"
    check(artifact_manifest.exists(), "shared artifact manifest exists", failures, str(artifact_manifest))
    check(local_root.exists(), "shared local index root exists", failures, str(local_root))
    check(helper_inputs.exists(), "shared helper inputs root exists", failures, str(helper_inputs))
    local_tasks = [p for p in local_root.iterdir() if p.is_dir() and p.name.startswith("DR")] if local_root.exists() else []
    browsecomp_shards = glob.glob(browsecomp_glob)
    check(len(local_tasks) == EXPECTED_LOCAL_TASKS, "shared local index task count", failures, str(len(local_tasks)))
    check(len(browsecomp_shards) == EXPECTED_BROWSECOMP_SHARDS, "BrowseComp shard count", failures, str(len(browsecomp_shards)))
    check((args.shared_root / "browsecomp_indexes/qwen3-embedding-4b/bm25_cache.pkl").exists(), "BrowseComp BM25 cache exists", failures)

    required_helper_inputs = [
        "browsecomp_doc_ids_over_10_windows_2026-05-12.txt",
        "docs_curated_web_plus_privacy_hopqa_q24.jsonl",
        "chunks_curated_web_base.jsonl",
        "chunks_curated_web_privacy_hopqa_q19.jsonl",
        "chunks_curated_web_privacy_hopqa_q20.jsonl",
        "chunks_curated_web_privacy_hopqa_q21.jsonl",
        "chunks_curated_web_privacy_hopqa_q24.jsonl",
    ]
    for name in required_helper_inputs:
        check((helper_inputs / name).exists(), f"helper input {name} exists", failures)

    helper_spec = args.mosaic_root / "privacy_helper_2gpu.yaml"
    run_script = args.mosaic_root / "run_privacy_hopqa.sh"
    domain_conf = args.pipelinerl_root / "conf/domain_params/privacy_hopqa.yaml"
    shared_root_text = str(args.shared_root)
    check(shared_root_text in helper_spec.read_text(), "helper spec points at shared root", failures)
    check(shared_root_text in run_script.read_text(), "training wrapper points at shared root", failures)
    check(shared_root_text in domain_conf.read_text(), "PipelineRL config points at shared root", failures)

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrivacy HopQA q30 setup is internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
