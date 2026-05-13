#!/usr/bin/env python3
"""Apply targeted q30 question fixes and conservative comparative variants."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DATA_DIR = Path("/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset")
BASE = "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review"
PATHS = [
    DATA_DIR / f"{BASE}.jsonl",
    DATA_DIR / f"{BASE}_train.jsonl",
    DATA_DIR / f"{BASE}_val.jsonl",
    DATA_DIR / f"{BASE}_test.jsonl",
]
LOG_PATH = DATA_DIR / f"{BASE}.targeted_question_variant_fixes_2026-05-13.jsonl"


TARGET_REWRITES = {
    ("DR0007", "d1a45183", 2): {
        "question": "In MediConn's quarterly marketing report where social media engagement increased by 25%, what quarter and year did the report cover?",
        "question_templated": "In MediConn's quarterly marketing report where social media engagement increased by (1), what quarter and year did the report cover?",
        "variants": [
            "Q1 2024",
            "Q1 of 2024",
            "2024 Q1",
            "Q1-2024",
            "Q1/2024",
            "1st quarter of 2024",
            "first quarter of 2024",
            "Quarter 1 of 2024",
            "Quarter 1 2024",
            "1st quarter 2024",
            "first quarter 2024",
            "Q1 2024 Marketing Performance Report",
            "Q1 2024 report",
            "Q1 2024 marketing report",
            "Q1 2024 quarterly marketing report",
        ],
        "note": "Replaces awkward quarter-report wording with a normal research-agent question while preserving the 25% bridge.",
    },
    ("DR0023", "4c2a9010", 1): {
        "question": "After Anthemis closed its $106 million Venture Fund I in March 2018, how many investments were in Anthemis Group's portfolio?",
        "question_templated": "After Anthemis closed its $106 million Venture Fund I in March 2018, how many investments were in Anthemis Group's portfolio?",
        "variants": ["70", "70 investments", "portfolio of 70 investments"],
        "note": "Anchors the question using source facts rather than document-viewer wording; the source states Anthemis had a portfolio of 70 investments after closing the $106 million fund.",
    },
    ("DR0022", "e7bdf883", 2): {
        "question": "In Lee's Market's Q3 2025 catalog feedback, where 85% of employees used feedback channels, which sauce nationality besides Korean was well received?",
        "question_templated": "In Lee's Market's Q3 2025 catalog feedback, where (1) of employees used feedback channels, which sauce nationality besides Korean was well received?",
        "variants": ["Japanese", "Japanese sauces", "Japanese sauce"],
        "note": "Anchors the local feedback context and asks the document-supported entity directly.",
    },
    ("DR0021", "1d523013", 5): {
        "question": "Who published the 2020 ICCESE work by Zong, Qi, and Zong about character expression shaping in animation movies?",
        "question_templated": "Who published the (4) ICCESE work by Zong, Qi, and Zong about character expression shaping in animation movies?",
        "variants": ["Atlantis Press", "Atlantis.Press", "atlantispress"],
        "note": "Specifies author names and subject to avoid ambiguity among multiple proceedings without asking about a title field.",
    },
    ("DR0017", "643e6343", 4): {
        "question": "In Lee's Market's market-share metrics, what year was the Q2 period when sales volume showed a steady 6% increase?",
        "question_templated": "In Lee's Market's market-share metrics, what year was the Q2 period when sales volume showed a steady 6% increase?",
        "variants": ["2025"],
        "note": "Anchors the supporting 6% Q2 sales-volume sentence; document supports 2025, not 2024.",
    },
}


ANSWER_SLOT_COMPARATIVE_PATTERNS = [
    (re.compile(r"^\s*over\s+how\s+many\b", re.IGNORECASE), ["over", "more than"]),
    (re.compile(r"^\s*over\s+what\s+amount\b", re.IGNORECASE), ["over", "more than"]),
    (re.compile(r"^\s*more\s+than\s+what\s+percentage\b", re.IGNORECASE), ["more than", "over"]),
    (re.compile(r"^\s*above\s+what\s+age\b", re.IGNORECASE), ["above", "over", "older than"]),
    (re.compile(r"^\s*at\s+least\s+how\s+many\b", re.IGNORECASE), ["at least"]),
    (re.compile(r"^\s*at\s+least\s+what\b", re.IGNORECASE), ["at least"]),
    (re.compile(r"^\s*below\s+what\s+amount\b", re.IGNORECASE), ["below", "under", "less than"]),
    (re.compile(r"^\s*under\s+what\s+amount\b", re.IGNORECASE), ["under", "less than", "below"]),
    (re.compile(r"^\s*less\s+than\s+what\b", re.IGNORECASE), ["less than", "under", "below"]),
]


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = re.sub(r"\s+", " ", text.lower())
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def numeric_like(answer: str) -> bool:
    return bool(
        re.search(
            r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|percent|pounds?|years?|words?|acres?|dollars?)\b",
            answer,
            re.IGNORECASE,
        )
    )


def compact_number_forms(answer: str) -> list[str]:
    values = [answer]
    if "," in answer:
        values.append(answer.replace(",", ""))
    if answer.startswith("$"):
        values.append(answer[1:])
    if answer.startswith("US$"):
        values.append(answer[3:])
    more = []
    for value in values:
        if "," in value:
            more.append(value.replace(",", ""))
    return dedupe([*values, *more])


def comparative_prefixes(question: str) -> list[str]:
    for pattern, prefixes in ANSWER_SLOT_COMPARATIVE_PATTERNS:
        if pattern.search(question):
            return prefixes
    return []


def comparative_variants(question: str, answer: str, existing: list[str]) -> list[str]:
    prefixes = comparative_prefixes(question)
    if not prefixes or not numeric_like(answer):
        return []
    bases = compact_number_forms(answer)
    for value in existing:
        text = str(value).strip()
        if text and numeric_like(text):
            bases.extend(compact_number_forms(text))
    bases = [base for base in dedupe(bases) if not re.match(r"^(over|more than|at least|above|older than|below|under|less than)\b", base, re.IGNORECASE)]
    variants: list[str] = []
    for prefix in prefixes:
        for base in bases:
            variants.append(f"{prefix} {base}")
    return dedupe(variants)


def percent_answer_variants(question: str, answer: str, existing: list[str]) -> list[str]:
    question_lower = question.lower()
    has_comparative = bool(
        re.search(
            r"\b(over|more than|above|at least|exceed(?:ed|s|ing)?|greater than)\b",
            question_lower,
            re.IGNORECASE,
        )
    )
    has_rate_slot = bool(re.search(r"\brate\b", question_lower))
    if not (has_comparative or has_rate_slot):
        return []

    answer_text = str(answer).strip()
    percent_values: list[str] = []
    number_match = re.search(r"(\d+(?:\.\d+)?)\s*%", answer_text)
    if number_match:
        number = number_match.group(1)
        percent_values.extend([f"{number}%", f"{number} percent", f"{number} per cent"])
    else:
        word_match = re.search(r"\b(\d+(?:\.\d+)?)\s+percent\b", answer_text, re.IGNORECASE)
        if word_match:
            number = word_match.group(1)
            percent_values.extend([f"{number}%", f"{number} percent", f"{number} per cent"])
    if not percent_values:
        return []

    for value in existing:
        text = str(value).strip()
        if re.search(r"^\d+(?:\.\d+)?\s*(?:%|percent|per cent)?$", text, re.IGNORECASE):
            if text not in {"0", "0.0"} and not text.startswith("0."):
                percent_values.append(text)

    metric_phrases: list[str] = []
    question_text = re.sub(r"[^A-Za-z0-9%\\-\\s]", " ", question.lower())
    words = question_text.split()
    if has_rate_slot:
        for idx, word in enumerate(words):
            if word == "rate":
                for width in (1, 2, 3):
                    start = max(0, idx - width)
                    phrase = " ".join(words[start : idx + 1])
                    if phrase and phrase != "rate":
                        metric_phrases.append(phrase)
        for phrase in (
            "completion rate",
            "participation rate",
            "response rate",
            "retention rate",
            "satisfaction rate",
            "compliance rate",
            "usage rate",
            "adoption rate",
            "growth rate",
            "churn rate",
            "turnover rate",
        ):
            if phrase in question_text:
                metric_phrases.append(phrase)

    variants: list[str] = []
    prefixes = ["over", "more than", "above", "at least"]
    for value in dedupe(percent_values):
        for prefix in prefixes:
            variants.append(f"{prefix} {value}")
        for phrase in dedupe(metric_phrases):
            variants.append(f"{value} {phrase}")
            for prefix in prefixes:
                variants.append(f"{prefix} {value} {phrase}")
    return dedupe(variants)


def variant_sources(variants: list[str], source: str) -> list[dict[str, Any]]:
    return [{"variant": variant, "sources": [source]} for variant in variants]


def rebuild_numbered_questions(chain: dict[str, Any]) -> None:
    hops = sorted(chain.get("hops") or [], key=lambda hop: int(hop.get("hop_number") or 0))
    if not hops:
        return
    final_hop = max(int(hop.get("hop_number") or 0) for hop in hops)
    lines: list[str] = []
    for hop in hops:
        hop_number = int(hop.get("hop_number") or 0)
        question = hop.get("question_templated") or hop.get("question") or ""
        lines.append(f"{hop_number} (final) {question}" if hop_number == final_hop else f"{hop_number}) {question}")
    chain["numbered_questions"] = "\n".join(lines)
    chain["global_question"] = chain["numbered_questions"]


def patch_chain(chain: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    chain_changed = False
    for hop in chain.get("hops") or []:
        key = (chain.get("task_id"), chain.get("chain_id"), int(hop.get("hop_number") or 0))
        before = {
            "question": hop.get("question"),
            "question_templated": hop.get("question_templated"),
            "accepted_answer_variants": list(hop.get("accepted_answer_variants") or []),
            "alternate_valid_answers": list(hop.get("alternate_valid_answers") or []),
        }
        target = TARGET_REWRITES.get(key)
        if target:
            hop["question"] = target["question"]
            hop["question_templated"] = target["question_templated"]
            hop["accepted_answer_variants"] = dedupe([*target["variants"], *before["accepted_answer_variants"]])
            hop["alternate_valid_answers"] = dedupe([v for v in hop["accepted_answer_variants"] if v != str(hop.get("answer"))])
            hop["answer_variant_sources"] = variant_sources(hop["accepted_answer_variants"], "targeted_question_variant_fixes_2026-05-13")
            note = target["note"]
            old_note = str(hop.get("answer_variant_notes") or "").strip()
            hop["answer_variant_notes"] = note if not old_note else f"{old_note} {note}"

        existing = list(hop.get("accepted_answer_variants") or [])
        comp = comparative_variants(str(hop.get("question") or ""), str(hop.get("answer") or ""), existing)
        percent_variants = percent_answer_variants(str(hop.get("question") or ""), str(hop.get("answer") or ""), existing)
        new_variants = dedupe([*comp, *percent_variants])
        if new_variants:
            hop["accepted_answer_variants"] = dedupe([*existing, *new_variants])
            hop["alternate_valid_answers"] = dedupe([*(hop.get("alternate_valid_answers") or []), *new_variants])
            hop["answer_variant_sources"] = [
                *(hop.get("answer_variant_sources") or []),
                *variant_sources(comp, "comparative_answer_variant_cleanup_2026-05-13"),
                *variant_sources(percent_variants, "qualified_percent_answer_variant_cleanup_2026-05-13"),
            ]
            old_note = str(hop.get("answer_variant_notes") or "").strip()
            comp_note = "Comparative and rate wording cleanup: accepted variants include qualified answer forms such as 'over X%' where useful for string matching."
            hop["answer_variant_notes"] = comp_note if not old_note else f"{old_note} {comp_note}"

        after = {
            "question": hop.get("question"),
            "question_templated": hop.get("question_templated"),
            "accepted_answer_variants": list(hop.get("accepted_answer_variants") or []),
            "alternate_valid_answers": list(hop.get("alternate_valid_answers") or []),
        }
        if after != before:
            chain_changed = True
            records.append(
                {
                    "task_id": chain.get("task_id"),
                    "chain_id": chain.get("chain_id"),
                    "hop_number": hop.get("hop_number"),
                    "answer": hop.get("answer"),
                    "question_before": before["question"],
                    "question_after": after["question"],
                    "question_templated_before": before["question_templated"],
                    "question_templated_after": after["question_templated"],
                    "accepted_before": before["accepted_answer_variants"],
                    "accepted_after": after["accepted_answer_variants"],
                    "alternate_before": before["alternate_valid_answers"],
                    "alternate_after": after["alternate_valid_answers"],
                    "targeted_rewrite": bool(target),
                    "comparative_variants_added": comp,
                    "qualified_percent_variants_added": percent_variants,
                }
            )
    if chain_changed:
        rebuild_numbered_questions(chain)
    return records


def patch_path(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    out: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        chain = json.loads(line)
        chain_records = patch_chain(chain)
        for record in chain_records:
            record["path"] = str(path)
            record["line_no"] = line_no
        records.extend(chain_records)
        out.append(json.dumps(chain, ensure_ascii=False))
    path.write_text("\n".join(out) + "\n")
    return records


def main() -> None:
    all_records: list[dict[str, Any]] = []
    for path in PATHS:
        records = patch_path(path)
        all_records.extend(records)
        print(f"{path.name}: {len(records)} hop updates")
    LOG_PATH.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in all_records) + ("\n" if all_records else ""))
    print(f"Wrote {LOG_PATH}")
    print(f"Total updates across files: {len(all_records)}")
    print(f"Targeted rewrites across files: {sum(1 for r in all_records if r['targeted_rewrite'])}")
    print(f"Comparative variant updates across files: {sum(1 for r in all_records if r['comparative_variants_added'])}")
    print(f"Qualified percent variant updates across files: {sum(1 for r in all_records if r['qualified_percent_variants_added'])}")
    for record in all_records[:40]:
        print("---")
        print(f"{Path(record['path']).name} {record['task_id']}/{record['chain_id']} hop {record['hop_number']} answer={record['answer']}")
        print(f"before: {record['question_before']}")
        print(f"after : {record['question_after']}")
        if record["comparative_variants_added"]:
            print(f"added variants: {record['comparative_variants_added'][:12]}")


if __name__ == "__main__":
    main()
