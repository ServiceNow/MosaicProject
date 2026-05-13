#!/usr/bin/env python3
"""Create the 2026-05-10 span-fixed Privacy HopQA dataset.

The proposal application pass reassigned a small number of gold documents, but
left their supporting_span fields pointing at evidence in the old documents.
This script creates a new versioned dataset with those spans fixed and rebuilds
the canonical train/val/test split files for the same chain split.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/toolkit/my_code/MosaicProject")
DATA_DIR = ROOT / "data_files/privacy_hopqa_final_dataset"
INPUT_DATASET = DATA_DIR / "privacy_hopqa_finalized_2026-05-10.jsonl"
OUTPUT_BASENAME = "privacy_hopqa_finalized_2026-05-10_spanfixed"
OUTPUT_DATASET = DATA_DIR / f"{OUTPUT_BASENAME}.jsonl"
SPANFIX_LOG = DATA_DIR / f"{OUTPUT_BASENAME}.spanfix_log.jsonl"
MANIFEST = DATA_DIR / f"manifest_{OUTPUT_BASENAME}.json"

SPLIT_SOURCE_FILES = {
    "train": DATA_DIR / "train_2026-05-09_contextsafe_answer_variants.jsonl",
    "val": DATA_DIR / "val_2026-05-09_contextsafe_answer_variants.jsonl",
    "test": DATA_DIR / "test_2026-05-09_contextsafe_answer_variants.jsonl",
}

SPAN_OVERRIDES = {
    "fd3a3f26::hop2": (
        "Q4 2022 vs Q4 2021,Lee's Market Delivery Service,5,60000,57142.86"
    ),
    "0ee7622b::hop2": (
        "I wanted to share a quick update regarding the renovations that took place across our stores in Q2 2024. "
        "We successfully renovated twenty-five stores, with an average cost of $200,000 per renovation."
    ),
    "13d66edb::hop2": (
        "Date: 2025-07-01 09:30:00\n"
        "From: Lee Wang <lee.wang@leesmarket.com>\n"
        "To: sofia.patel@leesmarket.com\n"
        "CC: linda.murphy@leesmarket.com\n"
        "Subject: Results of June 2025 Internal Employee Experience Survey\n"
        "Folder: inbox | Status: Read\n\n"
        "Message Body:\n"
        "----------------------------------------\n"
        "Hi Sofia,\n\n"
        "I hope this email finds you well. The results of the recent employee experience survey conducted at our Richmond store are in. "
        "Overall, we received an 82% 'excellent' or 'very good' rating for the store atmosphere. Such a high score is a testament to our team's efforts and the effectiveness of our current cleaning schedule.\n\n"
        "However, we have identified areas for improvement, particularly around staff breakroom amenities. The score was relatively low in this regard. We will be discussing this issue in our next management meeting and figuring out the best course of action.\n\n"
        "Best,\n"
        "Lee Wang\n"
        "============================================================\n\n"
        "Email ID: email_002\n"
        "Date: 2025-07-01 10:45:00\n"
        "From: Sofia Patel <sofia.patel@leesmarket.com>"
    ),
    "cfae3321::hop1": (
        "A recent survey revealed that 80% of our employees feel motivated and engaged in their work, citing the company's commitment "
        "to innovation and employee development as key factors."
    ),
    "73bea4c4::hop2": (
        "Since its inauguration in 2025, the Vaughan store has consistently outperformed initial projections. "
        "Customer footfall has been impressive, averaging 1000 shoppers per day."
    ),
    "7c388d5e::hop3": (
        "Over the past year, we have seen a 20% reduction in documentation errors, a testament to the effectiveness of our rigorous review process."
    ),
    "bff7f93c::hop4": (
        "AUD004,Lee's Market,2024-06-20,Regulatory,Minor issues with employee safety training records,Improved training program,Jason Wong\n"
        "AUD006,Evans' Electronics,2025-02-10,Financial,Several discrepancies in financial statements,Revised financial statements,George Evans\n"
        "AUD003,Carter's Supplies,2025-01-30,Operational,Two minor discrepancies in inventory count,Adjusted inventory tracking system,Emma Carter"
    ),
    "bc18cce3::hop1": (
        "The CRM system utilized by MediConn Solutions boasts a robust suite of features designed to streamline sales operations. "
        "Implemented in Q4 2024, it integrates seamlessly with our existing IT infrastructure"
    ),
    "8687c9b5::hop7": (
        "January 30 2023 – The Department of Health and Human Services issues an Analyst Note reporting that Killnet "
        "\"is actively targeting the health and public health sector \" 202301301200_KillNet Analyst Note_TLPCLEAR hhs gov"
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hop_audit_id(chain: dict[str, Any], hop: dict[str, Any]) -> str:
    return f"{chain['chain_id']}::hop{hop['hop_number']}"


def normalize_for_search(text: str) -> str:
    return " ".join(text.split()).lower()


def span_present(span: str, doc_text: str) -> bool:
    if span in doc_text:
        return True
    return normalize_for_search(span) in normalize_for_search(doc_text)


def split_chain_ids(path: Path) -> set[str]:
    return {str(row["chain_id"]) for row in read_jsonl(path)}


def main() -> int:
    rows = read_jsonl(INPUT_DATASET)
    by_chain_id = {str(row["chain_id"]): row for row in rows}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_rows: list[dict[str, Any]] = []

    seen: set[str] = set()
    for chain in rows:
        for hop in chain.get("hops", []):
            audit_id = hop_audit_id(chain, hop)
            if audit_id not in SPAN_OVERRIDES:
                continue
            span = SPAN_OVERRIDES[audit_id]
            doc_ref = Path(str(hop["doc_ref"]))
            doc_text = doc_ref.read_text(encoding="utf-8", errors="replace")
            if not span_present(span, doc_text):
                raise RuntimeError(f"span override for {audit_id} was not found in {doc_ref}")
            old_span = hop.get("supporting_span")
            hop["supporting_span"] = span
            log_rows.append(
                {
                    "audit_id": audit_id,
                    "updated_at": now,
                    "field": "supporting_span",
                    "old": old_span,
                    "new": span,
                    "doc_ref": str(doc_ref),
                }
            )
            seen.add(audit_id)

    missing = sorted(set(SPAN_OVERRIDES) - seen)
    if missing:
        raise RuntimeError(f"span overrides did not match dataset hops: {missing}")

    write_jsonl(OUTPUT_DATASET, rows)
    write_jsonl(SPANFIX_LOG, log_rows)

    split_counts: dict[str, int] = {}
    split_companies: dict[str, dict[str, int]] = {}
    assigned: set[str] = set()
    for split, source_path in SPLIT_SOURCE_FILES.items():
        ids = split_chain_ids(source_path)
        split_rows = [by_chain_id[chain_id] for chain_id in ids]
        split_rows.sort(key=lambda row: int(row.get("id", 0)))
        write_jsonl(DATA_DIR / f"{OUTPUT_BASENAME}_{split}.jsonl", split_rows)
        split_counts[split] = len(split_rows)
        split_companies[split] = dict(Counter(str(row.get("company")) for row in split_rows))
        assigned.update(ids)

    if assigned != set(by_chain_id):
        missing_ids = sorted(set(by_chain_id) - assigned)[:10]
        extra_ids = sorted(assigned - set(by_chain_id))[:10]
        raise RuntimeError(f"split ids do not cover dataset; missing={missing_ids} extra={extra_ids}")

    manifest = {
        "dataset": "privacy_hopqa_final",
        "version": OUTPUT_BASENAME,
        "source": str(INPUT_DATASET),
        "output": str(OUTPUT_DATASET),
        "spanfix_log": str(SPANFIX_LOG),
        "updated_at": now,
        "spanfix_count": len(log_rows),
        "spanfix_audit_ids": sorted(SPAN_OVERRIDES),
        "splits": {
            split: {
                "path": str(DATA_DIR / f"{OUTPUT_BASENAME}_{split}.jsonl"),
                "count": split_counts[split],
                "companies": split_companies[split],
            }
            for split in ("train", "val", "test")
        },
        "counts": {
            "chains": len(rows),
            "hops": sum(len(row.get("hops", [])) for row in rows),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
