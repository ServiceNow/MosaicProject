#!/usr/bin/env python3
"""Extract seed info from chains for smoke-test reproducibility.

Usage:
    python -m trajectory_generation.scripts.extract_bad_seeds \
        --input making_dataset_2/outputs/chains_all.jsonl \
        --chain-ids 17453ea8 b0d08ed6 \
        --output ~/detailed_trajectory_logs/bad_seeds.jsonl

Or extract from all filtered chains that scored high (to test same seeds with new pipeline):
    python -m trajectory_generation.scripts.extract_bad_seeds \
        --input making_dataset_2/outputs/chains_all_filtered.jsonl \
        --output bad_seeds_sample.jsonl \
        --max 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_generation import config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chain-ids", nargs="*", help="Specific chain IDs to extract")
    p.add_argument("--max", type=int, default=None, help="Max chains to extract (first N)")
    args = p.parse_args()

    chains = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                chains.append(json.loads(line))

    if args.chain_ids:
        chain_ids = set(args.chain_ids)
        chains = [c for c in chains if c.get("chain_id") in chain_ids]

    records = []
    for i, chain in enumerate(chains):
        if args.max and len(records) >= args.max:
            break
        hops = chain.get("hops", [])
        if not hops:
            continue
        h0 = hops[0]
        records.append({
            "idx": len(records),
            "chain_id": chain.get("chain_id"),
            "pattern": chain.get("pattern"),
            "task_id": chain.get("metadata", {}).get("task_id"),
            "seed": hash(chain.get("chain_id", "")) % config.MAX_RANDOM_SEED,
            "forced_hop": {
                "question": h0["question"],
                "answer": h0["answer"],
                "doc_id": h0["doc_id"],
            },
        })

    args.output = Path(args.output).expanduser()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} seed records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
