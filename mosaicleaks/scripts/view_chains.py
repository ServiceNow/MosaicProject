#!/usr/bin/env python3
"""Pretty-print chains from chains.jsonl."""

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chains.jsonl")
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    with open(path) as f:
        chains = [json.loads(line) for line in f if line.strip()]

    print(f"Chains: {len(chains)}\n")
    for i, d in enumerate(chains):
        print("=" * 60)
        print(f"Chain {i+1}: {d.get('chain_id')} | {d.get('pattern')} | valid={d.get('valid')}")
        print()
        print("Questions:")
        for line in (d.get("numbered_questions") or "").split("\n"):
            print(" ", line)
        print()
        print("Final answer:", d.get("global_answer"))
        hops = d.get("hops", [])
        if hops:
            print()
            print("Hops:")
            for h in hops:
                q = h.get("question", "")
                print(f"  [{h.get('hop_type')}] Q: {q[:75]}{'...' if len(q) > 75 else ''}")
                print(f"       A: {h.get('answer')}")
        print()
    return 0


if __name__ == "__main__":
    main()
