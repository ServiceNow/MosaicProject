#!/usr/bin/env python3
"""Clean quarter/year answer wording and variants for the q30 Privacy HopQA set.

Policy:
- If the gold answer is "Q# YYYY" and the question only asks for a quarter,
  rewrite the question to ask for "quarter and year".
- If the question explicitly asks for a quarter within a stated year, keep the
  wording and add bare-quarter aliases such as "Q2" and "second quarter".
- Keep quarter/year variants exhaustive but do not allow bare-quarter aliases
  when the question asks for quarter and year.
"""

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
LOG_PATH = DATA_DIR / f"{BASE}.quarter_cleanup_log.jsonl"

ANSWER_RE = re.compile(r"^\s*(Q[1-4])\s+(20\d{2})\s*$", re.IGNORECASE)
ASKS_QUARTER_YEAR_RE = re.compile(
    r"quarter\s*(?:and|&)\s*year|year\s*(?:and|&)\s*quarter|quarter/year|quarter-year|quarter_?year|quarter of which year|\(quarter and year\)",
    re.IGNORECASE,
)
BARE_QUARTER_RE = re.compile(
    r"^\s*(?:Q[1-4]|(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter|quarter\s+[1-4])\s*$",
    re.IGNORECASE,
)

ORDINAL = {
    "Q1": ("1", "1st", "first"),
    "Q2": ("2", "2nd", "second"),
    "Q3": ("3", "3rd", "third"),
    "Q4": ("4", "4th", "fourth"),
}


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


def qyear_variants(quarter: str, year: str, *, fiscal: bool) -> list[str]:
    quarter = quarter.upper()
    number, ordinal, word = ORDINAL[quarter]
    variants = [
        f"{quarter} {year}",
        f"{quarter} of {year}",
        f"{year} {quarter}",
        f"{quarter}-{year}",
        f"{quarter}/{year}",
        f"{ordinal} quarter of {year}",
        f"{word} quarter of {year}",
        f"Quarter {number} of {year}",
        f"Quarter {number} {year}",
        f"{ordinal} quarter {year}",
        f"{word} quarter {year}",
    ]
    if fiscal:
        variants.extend(
            [
                f"fiscal {quarter} {year}",
                f"fiscal {quarter} of {year}",
                f"FY{year} {quarter}",
                f"{quarter} FY{year}",
                f"fiscal {ordinal} quarter of {year}",
                f"fiscal {word} quarter of {year}",
            ]
        )
    return dedupe(variants)


def bare_quarter_variants(quarter: str) -> list[str]:
    quarter = quarter.upper()
    number, ordinal, word = ORDINAL[quarter]
    return [quarter, f"Quarter {number}", f"{ordinal} quarter", f"{word} quarter"]


def allows_bare_quarter(question: str, year: str) -> bool:
    return bool(re.search(rf"\bquarter\s+(?:of|in|during)\s+{re.escape(year)}\b", question, re.IGNORECASE))


def asks_quarter_and_year(question: str) -> bool:
    return bool(ASKS_QUARTER_YEAR_RE.search(question))


def rewrite_question(question: str) -> str:
    if asks_quarter_and_year(question):
        return question

    custom_patterns = [
        (r"^When is (.+) scheduled\?$", r"For what quarter and year is \1 scheduled?"),
        (r"^What is the slated date for (.+)\?$", r"What quarter and year is slated for \1?"),
        (r"^By when does (.+)$", r"By what quarter and year does \1"),
        (r"^By what target date does (.+)$", r"By what target quarter and year does \1"),
        (
            r"^In which (.+?) quarterly (.+?) is (.+)\?$",
            r"What quarter and year identifies the \1 quarterly \2 where \3?",
        ),
    ]
    for pattern, repl in custom_patterns:
        rewritten, count = re.subn(pattern, repl, question, count=1, flags=re.IGNORECASE)
        if count:
            return re.sub(r"\s+", " ", rewritten).strip()

    replacements = [
        (r"\bwhat\s+period\b", "what quarter and year"),
        (r"\bwhich\s+period\b", "which quarter and year"),
        (r"\bwhat\s+specific\s+quarter\b", "what specific quarter and year"),
        (r"\bspecific\s+quarter\b", "specific quarter and year"),
        (r"\btarget\s+quarter\b", "target quarter and year"),
        (r"\bdeadline\s+quarter\b", "deadline quarter and year"),
        (r"\bstart\s+quarter\b", "start quarter and year"),
        (r"\bstarting\s+quarter\b", "starting quarter and year"),
        (r"\bplanned\s+distribution\s+quarter\b", "planned distribution quarter and year"),
        (r"\breporting\s+quarter\b", "reporting quarter and year"),
        (r"\bcompetitive\s+intelligence\s+quarter\b", "competitive intelligence quarter and year"),
        (r"\bfiscal\s+quarter's\b", "fiscal quarter and year"),
        (r"\bfiscal\s+quarter\b", "fiscal quarter and year"),
        (r"\bquarter's\b", "quarter and year"),
        (r"\bquarter\b", "quarter and year"),
    ]
    rewritten = question
    for pattern, repl in replacements:
        rewritten, count = re.subn(pattern, repl, rewritten, count=1, flags=re.IGNORECASE)
        if count:
            break
    rewritten = re.sub(r"\bquarter and year and year\b", "quarter and year", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return rewritten


def normalize_question_text(question: str) -> str:
    question = question.strip()
    if re.match(r"^(what|which|when|where|who|why|how)\b", question):
        return question[0].upper() + question[1:]
    return question


def rebuild_numbered_questions(chain: dict[str, Any]) -> str:
    hops = sorted(chain.get("hops") or [], key=lambda hop: int(hop.get("hop_number") or 0))
    lines: list[str] = []
    final_hop = max([int(hop.get("hop_number") or 0) for hop in hops] or [0])
    for hop in hops:
        hop_number = int(hop.get("hop_number") or 0)
        question = hop.get("question_templated") or hop.get("question") or ""
        if hop_number == final_hop:
            lines.append(f"{hop_number} (final) {question}")
        else:
            lines.append(f"{hop_number}) {question}")
    return "\n".join(lines)


def rebuild_variant_sources(variants: list[str]) -> list[dict[str, Any]]:
    return [{"variant": variant, "sources": ["quarter_cleanup"]} for variant in variants]


def patch_hop(hop: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    answer = str(hop.get("answer") or "")
    match = ANSWER_RE.match(answer)
    if not match:
        return False, {}

    quarter = match.group(1).upper()
    year = match.group(2)
    question_before = str(hop.get("question") or "")
    templated_before = str(hop.get("question_templated") or question_before)
    fiscal = "fiscal" in question_before.lower() or "fiscal" in templated_before.lower()
    already_qy = asks_quarter_and_year(question_before)
    allow_bare = allows_bare_quarter(question_before, year)
    should_rewrite = not already_qy and not allow_bare

    variants_before = list(hop.get("accepted_answer_variants") or [])
    alternates_before = list(hop.get("alternate_valid_answers") or [])

    base_variants = qyear_variants(quarter, year, fiscal=fiscal)
    existing_non_bare = [str(v).strip() for v in variants_before if not BARE_QUARTER_RE.match(str(v))]
    new_variants = dedupe([answer, *base_variants, *existing_non_bare])
    new_alternates = dedupe([*base_variants[1:], *[str(v).strip() for v in alternates_before if not BARE_QUARTER_RE.match(str(v))]])

    if allow_bare:
        bare = bare_quarter_variants(quarter)
        new_variants = dedupe([*new_variants, *bare])
        new_alternates = dedupe([*new_alternates, *bare])

    question_after = rewrite_question(question_before) if should_rewrite else question_before
    templated_after = rewrite_question(templated_before) if should_rewrite else templated_before
    question_after = normalize_question_text(question_after)
    templated_after = normalize_question_text(templated_after)

    changed = (
        question_after != question_before
        or templated_after != templated_before
        or new_variants != variants_before
        or new_alternates != alternates_before
    )
    if not changed:
        return False, {}

    hop["question"] = question_after
    hop["question_templated"] = templated_after
    hop["accepted_answer_variants"] = new_variants
    hop["alternate_valid_answers"] = new_alternates
    hop["answer_variant_sources"] = rebuild_variant_sources(new_variants)
    previous_note = str(hop.get("answer_variant_notes") or "").strip()
    cleanup_note = (
        "Quarter cleanup: question asks for quarter and year when answer includes year; "
        "bare-quarter aliases allowed only when the year is explicit in the question."
    )
    hop["answer_variant_notes"] = cleanup_note if not previous_note else f"{previous_note} {cleanup_note}"

    return True, {
        "hop_number": hop.get("hop_number"),
        "answer": answer,
        "question_before": question_before,
        "question_after": question_after,
        "question_templated_before": templated_before,
        "question_templated_after": templated_after,
        "accepted_before": variants_before,
        "accepted_after": new_variants,
        "alternate_before": alternates_before,
        "alternate_after": new_alternates,
        "already_quarter_year": already_qy,
        "bare_quarter_allowed": allow_bare,
        "rewrote_question": should_rewrite,
    }


def patch_path(path: Path) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    output_lines: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        chain = json.loads(line)
        chain_changed = False
        chain_changes: list[dict[str, Any]] = []
        for hop in chain.get("hops") or []:
            changed, record = patch_hop(hop)
            if changed:
                chain_changed = True
                chain_changes.append(record)
        if chain_changed:
            numbered = rebuild_numbered_questions(chain)
            chain["numbered_questions"] = numbered
            chain["global_question"] = numbered
            for record in chain_changes:
                record.update(
                    {
                        "path": str(path),
                        "line_no": line_no,
                        "task_id": chain.get("task_id"),
                        "chain_id": chain.get("chain_id"),
                    }
                )
            changes.extend(chain_changes)
        output_lines.append(json.dumps(chain, ensure_ascii=False))
    path.write_text("\n".join(output_lines) + "\n")
    return changes


def main() -> None:
    all_changes: list[dict[str, Any]] = []
    for path in PATHS:
        changes = patch_path(path)
        all_changes.extend(changes)
        print(f"{path.name}: {len(changes)} hop updates")
    LOG_PATH.write_text("\n".join(json.dumps(change, ensure_ascii=False) for change in all_changes) + ("\n" if all_changes else ""))
    print(f"Wrote {LOG_PATH}")
    rewrites = sum(1 for change in all_changes if change["rewrote_question"])
    bare_allowed = sum(1 for change in all_changes if change["bare_quarter_allowed"])
    print(f"Total hop updates across files: {len(all_changes)}")
    print(f"Question rewrites across files: {rewrites}")
    print(f"Bare-quarter alias cases across files: {bare_allowed}")
    for change in all_changes[:30]:
        print("---")
        print(f"{Path(change['path']).name} {change['task_id']}/{change['chain_id']} hop {change['hop_number']} answer {change['answer']}")
        print(f"before: {change['question_before']}")
        print(f"after : {change['question_after']}")


if __name__ == "__main__":
    main()
