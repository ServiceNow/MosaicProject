#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def canonical_local_doc_family(doc_id: str) -> str | None:
    if not isinstance(doc_id, str) or not doc_id.startswith("local/"):
        return None
    parts = doc_id.split("/")
    if not parts:
        return None
    last = parts[-1]
    if "." in last:
        last = last.rsplit(".", 1)[0]
    parts[-1] = last
    return "/".join(parts)


def find_variant_reuse(chain: dict) -> dict[str, list[str]]:
    fam_to_raw: dict[str, set[str]] = {}
    for hop in chain.get("hops", []):
        if not isinstance(hop, dict):
            continue
        doc_id = hop.get("doc_id")
        fam = canonical_local_doc_family(doc_id)
        if fam:
            fam_to_raw.setdefault(fam, set()).add(doc_id)
    return {fam: sorted(raws) for fam, raws in fam_to_raw.items() if len(raws) > 1}


def main() -> int:
    p = argparse.ArgumentParser(description="Audit chains for local document variant reuse (.md/.docx/.xlsx/.txt/.jsonl)")
    p.add_argument("chains", type=Path)
    p.add_argument("--output", type=Path, default=None, help="Optional JSONL path for affected chains only")
    args = p.parse_args()

    total = affected = valid_like = affected_valid_like = 0
    out_f = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        with args.chains.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chain = json.loads(line)
                total += 1
                is_valid = bool((chain.get("verification") or {}).get("is_valid") or chain.get("valid"))
                valid_like += 1 if is_valid else 0
                bad = find_variant_reuse(chain)
                if not bad:
                    continue
                affected += 1
                affected_valid_like += 1 if is_valid else 0
                if out_f is not None:
                    payload = {
                        "chain_id": chain.get("chain_id"),
                        "pattern": chain.get("pattern"),
                        "valid_like": is_valid,
                        "metadata": chain.get("metadata", {}),
                        "bad_families": bad,
                        "hop_doc_ids": [h.get("doc_id") for h in chain.get("hops", []) if isinstance(h, dict)],
                    }
                    out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    print(json.dumps({
        "chains": total,
        "valid_like": valid_like,
        "affected": affected,
        "affected_valid_like": affected_valid_like,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
