#!/usr/bin/env python3
"""Apply approved Privacy HopQA question/gold decisions for questionfix10.

Inputs are intentionally versioned and immutable. Outputs create a new
variantfix10_questionfix10 dataset family plus manifests and an applied log.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/toolkit/my_code/MosaicProject")
FULL_DIR = ROOT / "data_files/privacy_hopqa_final_dataset"
FIRST_DIR = ROOT / "data_files/privacy_hopqa_first_web_hop_dataset"
RUNTIME_DIR = ROOT / "runtime_analysis/privacy_hopqa_questionfix10_2026-05-12"

SOURCE_VERSION = "2026-05-12_variantfix9_questionfix9"
TARGET_VERSION = "2026-05-12_variantfix10_questionfix10"
DECISIONS = RUNTIME_DIR / "approved_question_gold_decisions_questionfix10_2026-05-12.jsonl"

FULL_INPUTS = {
    "full": FULL_DIR / f"privacy_hopqa_finalized_{SOURCE_VERSION}.jsonl",
    "train": FULL_DIR / f"privacy_hopqa_finalized_{SOURCE_VERSION}_train.jsonl",
    "val": FULL_DIR / f"privacy_hopqa_finalized_{SOURCE_VERSION}_val.jsonl",
    "test": FULL_DIR / f"privacy_hopqa_finalized_{SOURCE_VERSION}_test.jsonl",
}
FULL_OUTPUTS = {
    key: FULL_DIR / f"privacy_hopqa_finalized_{TARGET_VERSION}{'' if key == 'full' else '_' + key}.jsonl"
    for key in FULL_INPUTS
}
FIRST_INPUTS = {
    "full": FIRST_DIR / f"privacy_hopqa_first_web_hop_anywhere_{SOURCE_VERSION}.jsonl",
    "train": FIRST_DIR / f"privacy_hopqa_first_web_hop_anywhere_{SOURCE_VERSION}_train.jsonl",
    "val": FIRST_DIR / f"privacy_hopqa_first_web_hop_anywhere_{SOURCE_VERSION}_val.jsonl",
    "test": FIRST_DIR / f"privacy_hopqa_first_web_hop_anywhere_{SOURCE_VERSION}_test.jsonl",
}
FIRST_OUTPUTS = {
    key: FIRST_DIR / f"privacy_hopqa_first_web_hop_anywhere_{TARGET_VERSION}{'' if key == 'full' else '_' + key}.jsonl"
    for key in FIRST_INPUTS
}

APPLIED_LOG = FULL_DIR / f"privacy_hopqa_finalized_{TARGET_VERSION}.applied_log.jsonl"
MANIFEST = FULL_DIR / f"manifest_privacy_hopqa_finalized_{TARGET_VERSION}.json"
FIRST_MANIFEST = FIRST_DIR / f"manifest_privacy_hopqa_first_web_hop_anywhere_{TARGET_VERSION}.json"

# Explicit templated forms preserve chain placeholders instead of baking realized
# upstream answers into global_question/numbered_questions.
TEMPLATED_OVERRIDES = {
    ("19", 2): "What kind of customer feedback regarding vehicle performance improved by (1)% year over year?",
    ("190", 3): "Who sent the email with subject 'New Store Opening in (2)'?",
    ("225", 2): "What fiscal year is named in the title of MediConn's Virtual Consultations Performance report that documented a (1) reduction in resource utilization?",
    ("379", 2): "What quarter had average daily foot traffic above (1) hundred but below 13 hundred at Lee's Market Suburb outlet?",
    ("425", 2): "In 2023, the year of MediConn's (1) Q2 corporate-client rise, what share of revenue came from corporate clients?",
    ("612", 2): "In H1 2024, the half-year before (1) of customers praised return team efficiency, what was Lee's Market's average response time for refund queries?",
    ("648", 4): "In the Audit Findings Report, what audit date is listed for the row whose audit ID is one less than the audit ID on (3)'s Lee's Market 2024 row?",
    ("659", 3): "What kind of standards across different states does Elexion's cross-border tracking system manage to maintain its (2) compliance rate?",
    ("682", 5): "What is the full name of the sender of the email with subject 'Re: Upcoming Cybersecurity Strategy Meeting' mentioning the (4) strategy meeting?",
    ("719", 2): "Since what year has the average load per truck increased by (1) in the same operational-efficiency period as the fleet's fuel-consumption reduction?",
    ("833", 4): "In the warranty repair downtime section, in what year was Elexion's average downtime described as a 5% improvement, when competitors saw a (3) reduction from OTA diagnostics?",
    ("848", 3): "What was MediConn Solutions' overall eNPS score in (2), the earlier score from which the 2022 +25 score improved?",
    ("71", 3): "In what year was the Wild Species 2020 report on species status in (2) published?",
    ("106", 3): "What was Netflix's 2023 revenue, the year the article says it had over 80 million more than (2) subscribers, rounded to the nearest tenth of a billion dollars?",
    ("141", 3): "Across how many projects was the corpus analyzed by the functional code clone detector with a (2) false positive rate?",
    ("144", 3): "Which state's table row has every elective public officer of the state, petitions requiring signatures from (2) of votes cast in the last election, and a 60-day circulation period?",
    ("522", 5): "Currently, what percent of registered nurses qualify via apprenticeships, compared with the NHS's (4) target for 2028/29?",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def merge_unique(existing: list[Any] | None, additions: list[Any] | None) -> tuple[list[str], list[str]]:
    out = [str(value).strip() for value in (existing or []) if str(value).strip()]
    seen = {norm(value) for value in out}
    added: list[str] = []
    for value in additions or []:
        text = str(value or "").strip()
        if not text:
            continue
        key = norm(text)
        if key in seen:
            continue
        out.append(text)
        added.append(text)
        seen.add(key)
    return out, added


def find_hop(chain: dict[str, Any], hop_number: int) -> dict[str, Any] | None:
    for hop in chain.get("hops") or []:
        if int(hop.get("hop_number") or 0) == hop_number:
            return hop
    return None


def render_question_from_templated(templated: str, chain: dict[str, Any], hop_number: int) -> str:
    rendered = templated
    for hop in chain.get("hops") or []:
        current = int(hop.get("hop_number") or 0)
        if current >= hop_number:
            continue
        rendered = rendered.replace(f"({current})", str(hop.get("answer") or ""))
    return rendered


def regenerate_chain_questions(chain: dict[str, Any]) -> None:
    lines: list[str] = []
    hops = chain.get("hops") or []
    final_n = max((int(h.get("hop_number") or 0) for h in hops), default=0)
    for hop in hops:
        n = int(hop.get("hop_number") or 0)
        prefix = f"{n} (final)" if n == final_n else f"{n})"
        if n == final_n:
            lines.append(f"{prefix} {hop.get('question_templated') or hop.get('question') or ''}")
        else:
            lines.append(f"{prefix} {hop.get('question_templated') or hop.get('question') or ''}")
    text = "\n".join(lines)
    chain["numbered_questions"] = text
    chain["global_question"] = text


def update_variant_sources(container: dict[str, Any], key: str, added: list[str], source_tag: str) -> None:
    if not added:
        return
    source_key = "global_answer_variant_sources" if key == "global" else "answer_variant_sources"
    existing = list(container.get(source_key) or [])
    seen = {norm(item.get("variant")) for item in existing if isinstance(item, dict)}
    for variant in added:
        if norm(variant) in seen:
            continue
        existing.append({"variant": variant, "sources": [source_tag]})
        seen.add(norm(variant))
    container[source_key] = existing


def apply_to_hop(chain: dict[str, Any], hop: dict[str, Any], decision: dict[str, Any], now: str) -> dict[str, Any]:
    cid = str(chain.get("chain_id") or chain.get("source_chain_id") or decision["chain_id"])
    hop_n = int(hop.get("hop_number") or decision["hop_number"])
    changes: list[dict[str, Any]] = []

    approved_question = decision.get("approved_question")
    if approved_question:
        new_templated = TEMPLATED_OVERRIDES.get((cid, hop_n), approved_question)
        new_question = render_question_from_templated(new_templated, chain, hop_n)
        if hop.get("question") != new_question:
            changes.append({"field": "question", "old": hop.get("question"), "new": new_question})
            hop["question"] = new_question
        if hop.get("question_templated") != new_templated:
            changes.append({"field": "question_templated", "old": hop.get("question_templated"), "new": new_templated})
            hop["question_templated"] = new_templated

    approved_answer = str(decision.get("approved_answer") or "").strip()
    if decision.get("decision_type") == "rewrite_question_change_final_answer" and approved_answer:
        if hop.get("answer") != approved_answer:
            changes.append({"field": "answer", "old": hop.get("answer"), "new": approved_answer})
            hop["answer"] = approved_answer

    variants = list(decision.get("approved_variants") or [])
    if approved_answer:
        variants = [approved_answer, *variants]
    merged, added = merge_unique(hop.get("accepted_answer_variants"), variants)
    if added:
        changes.append({"field": "accepted_answer_variants", "added": added})
        hop["accepted_answer_variants"] = merged
        alternate_candidates = [v for v in added if norm(v) != norm(hop.get("answer"))]
        alt_merged, alt_added = merge_unique(hop.get("alternate_valid_answers"), alternate_candidates)
        if alt_added:
            changes.append({"field": "alternate_valid_answers", "added": alt_added})
            hop["alternate_valid_answers"] = alt_merged
        update_variant_sources(hop, "hop", added, "approved_question_gold_decisions_questionfix10_2026-05-12")

    hops = chain.get("hops") or []
    final_n = max((int(h.get("hop_number") or 0) for h in hops), default=0)
    if hop_n == final_n:
        if decision.get("decision_type") == "rewrite_question_change_final_answer" and approved_answer:
            if chain.get("global_answer") != approved_answer:
                changes.append({"field": "global_answer", "old": chain.get("global_answer"), "new": approved_answer})
                chain["global_answer"] = approved_answer
        global_variants = list(hop.get("accepted_answer_variants") or [])
        gv_merged, gv_added = merge_unique(chain.get("global_answer_variants"), global_variants)
        if gv_added:
            changes.append({"field": "global_answer_variants", "added": gv_added})
            chain["global_answer_variants"] = gv_merged
            ga_candidates = [v for v in gv_added if norm(v) != norm(chain.get("global_answer"))]
            ga_merged, ga_added = merge_unique(chain.get("global_alternate_valid_answers"), ga_candidates)
            if ga_added:
                changes.append({"field": "global_alternate_valid_answers", "added": ga_added})
                chain["global_alternate_valid_answers"] = ga_merged
            update_variant_sources(chain, "global", gv_added, "approved_question_gold_decisions_questionfix10_2026-05-12")

    if changes:
        hop.setdefault("audit_resolutions", []).append({
            "audit_source": "approved_question_gold_decisions_questionfix10_2026-05-12",
            "resolution": decision.get("decision_type"),
            "rationale": decision.get("reason"),
            "applied_at": now,
        })
    return {"changes": changes}


def patch_dataset(rows: list[dict[str, Any]], decisions_by_key: dict[tuple[str, int], dict[str, Any]], dataset_kind: str, split: str, now: str) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for chain in rows:
        cid = str(chain.get("chain_id") or chain.get("source_chain_id") or "")
        source_cid = str(chain.get("source_chain_id") or chain.get("chain_id") or "")
        row_id = str(chain.get("id") or "")
        row_changed = False
        for hop in chain.get("hops") or []:
            hop_n = int(hop.get("source_hop_number") or hop.get("hop_number") or chain.get("source_hop_number") or 0)
            candidate_keys = [
                (source_cid, hop_n),
                (cid, hop_n),
                (row_id, hop_n),
            ]
            if dataset_kind == "firstweb":
                source_hop_n = int(chain.get("source_hop_number") or 0)
                candidate_keys.extend([
                    (source_cid, source_hop_n),
                    (cid, source_hop_n),
                    (row_id, source_hop_n),
                ])
            decision = next((decisions_by_key[key] for key in candidate_keys if key in decisions_by_key), None)
            if not decision:
                continue
            result = apply_to_hop(chain, hop, decision, now)
            if result["changes"]:
                row_changed = True
                logs.append({
                    "ts": now,
                    "dataset_kind": dataset_kind,
                    "split": split,
                    "chain_id": source_cid,
                    "row_chain_id": cid,
                    "hop_number": hop_n,
                    "decision_type": decision.get("decision_type"),
                    "changes": result["changes"],
                    "reason": decision.get("reason"),
                })
        if row_changed:
            regenerate_chain_questions(chain)
    return logs


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    decisions = read_jsonl(DECISIONS)
    decisions_by_key = {(str(d["chain_id"]), int(d["hop_number"])): d for d in decisions}
    all_logs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "version": TARGET_VERSION,
        "source_version": SOURCE_VERSION,
        "created_at": now,
        "decisions_file": str(DECISIONS),
        "n_decisions": len(decisions),
        "datasets": {},
    }

    for split, input_path in FULL_INPUTS.items():
        rows = read_jsonl(input_path)
        logs = patch_dataset(rows, decisions_by_key, "full", split, now)
        write_jsonl(FULL_OUTPUTS[split], rows)
        all_logs.extend(logs)
        summary["datasets"][f"full_{split}"] = {
            "input": str(input_path),
            "output": str(FULL_OUTPUTS[split]),
            "rows": len(rows),
            "changed_records": len({(log["row_chain_id"], log["hop_number"]) for log in logs}),
            "change_events": len(logs),
        }

    for split, input_path in FIRST_INPUTS.items():
        rows = read_jsonl(input_path)
        logs = patch_dataset(rows, decisions_by_key, "firstweb", split, now)
        write_jsonl(FIRST_OUTPUTS[split], rows)
        all_logs.extend(logs)
        summary["datasets"][f"firstweb_{split}"] = {
            "input": str(input_path),
            "output": str(FIRST_OUTPUTS[split]),
            "rows": len(rows),
            "changed_records": len({(log["row_chain_id"], log["hop_number"]) for log in logs}),
            "change_events": len(logs),
        }

    write_jsonl(APPLIED_LOG, all_logs)

    full_rows = read_jsonl(FULL_OUTPUTS["full"])
    manifest = deepcopy(summary)
    manifest.update({
        "dataset_file": str(FULL_OUTPUTS["full"]),
        "train_file": str(FULL_OUTPUTS["train"]),
        "val_file": str(FULL_OUTPUTS["val"]),
        "test_file": str(FULL_OUTPUTS["test"]),
        "n_chains": len(full_rows),
        "n_hops": sum(len(row.get("hops") or []) for row in full_rows),
        "applied_log": str(APPLIED_LOG),
        "notes": [
            "Questionfix10 applies targeted ambiguity fixes from q9 changed-hop residuals.",
            "Case-only duplicate variants are removed by lowercase-normalized merge.",
            "No unsupported outside-name aliases were added where not present in source docs.",
        ],
    })
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    first_rows = read_jsonl(FIRST_OUTPUTS["full"])
    first_manifest = deepcopy(summary)
    first_manifest.update({
        "dataset_file": str(FIRST_OUTPUTS["full"]),
        "train_file": str(FIRST_OUTPUTS["train"]),
        "val_file": str(FIRST_OUTPUTS["val"]),
        "test_file": str(FIRST_OUTPUTS["test"]),
        "n_rows": len(first_rows),
        "notes": ["First-web rows synchronized where their source chain/hop matched approved decisions."],
    })
    FIRST_MANIFEST.write_text(json.dumps(first_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
