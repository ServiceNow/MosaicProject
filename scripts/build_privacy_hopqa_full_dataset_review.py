#!/usr/bin/env python3
"""Build a full Privacy HopQA answer/question review artifact.

This is a reviewer, not an auto-mutator. It joins the current finalized dataset
with the latest StepFun answer-unit proposal file and adds conservative risk
flags that help prioritize human/source-grounded cleanup across every hop.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/toolkit/my_code/MosaicProject")
DATASET_PATH = ROOT / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-07_answer_variants.jsonl"
PROPOSAL_PATH = ROOT / "data_files/privacy_hopqa_final_dataset/stepfun_answer_unit_variant_proposals_2026-05-07_v11.jsonl"
OUT_HTML = ROOT / "runtime_analysis/privacy_hopqa_full_dataset_answer_review_2026-05-09.html"
OUT_CSV = ROOT / "runtime_analysis/privacy_hopqa_full_dataset_answer_review_2026-05-09.csv"
OUT_JSON = ROOT / "runtime_analysis/privacy_hopqa_full_dataset_answer_review_2026-05-09.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def norm_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text


def dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = norm_key(text)
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def truncate(value: Any, limit: int = 900) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ... [truncated]"


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def is_year_question(question: str) -> bool:
    lower = question.lower()
    if (
        "month and year" in lower
        or "quarter and year" in lower
        or "fiscal year" in lower
        or "year-over-year" in lower
        or "year and period" in lower
        or "period of what year" in lower
        or "quarter of which year" in lower
        or "full date" in lower
    ):
        return False
    return bool(re.search(r"\b(in what year|which year|what year|year did|year was|year is)\b", lower))


def is_month_question(question: str) -> bool:
    lower = question.lower()
    if "month and year" in lower:
        return False
    return bool(re.search(r"\b(in what month|which month|what month)\b", lower))


def is_day_question(question: str) -> bool:
    return "day of the month" in question.lower() or bool(re.search(r"\bwhat day\b", question.lower()))


def is_city_question(question: str) -> bool:
    return bool(re.search(r"\b(which|what|in which|in what) city\b", question.lower()))


def looks_like_full_date(value: str) -> bool:
    month = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    return bool(re.search(rf"\b({month})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}\b", value.lower()))


def materially_different(a: Any, b: Any) -> bool:
    return norm_key(a) != norm_key(b)


def question_risk_flags(question: str, answer: str, variants: list[str], hop: dict[str, Any], proposal: dict[str, Any] | None) -> list[str]:
    flags: list[str] = []
    if hop.get("question_well_formed") is False:
        flags.append("question_marked_not_well_formed")
    if hop.get("question_issues"):
        flags.append("question_issues_present")
    if hop.get("suggested_rewrite"):
        flags.append("existing_suggested_rewrite")
    proposal_obj = proposal or {}
    if proposal_obj.get("needs_edit"):
        flags.append("stepfun_needs_edit")
    if proposal_obj.get("question_rewrite") and materially_different(proposal_obj.get("question_rewrite"), question):
        flags.append("stepfun_question_rewrite")
    if proposal_obj.get("rejected_existing_answer_variants"):
        flags.append("stepfun_rejected_existing_variant")

    for variant in variants:
        lower = norm_key(variant)
        if word_count(variant) > 5:
            flags.append("long_answer_variant")
        if is_year_question(question) and not re.fullmatch(r"\d{4}", lower):
            flags.append("year_question_non_year_variant")
        if is_month_question(question) and re.search(r"\d{4}", lower):
            flags.append("month_question_year_or_date_variant")
        if is_day_question(question) and looks_like_full_date(variant):
            flags.append("day_question_full_date_variant")
        if is_city_question(question) and "," in variant:
            flags.append("city_question_location_expansion")
        if lower.startswith(("the year ", "in the year ", "in ")) and not is_month_question(question):
            flags.append("prefix_noise_variant")
        if ":" in variant and re.search(r"\d+\s*:\s*\d+", variant):
            flags.append("ratio_variant")
        if re.search(r"\b(?:ce|ad)\b", lower) and is_year_question(question):
            flags.append("era_suffix_year_variant")
    if is_year_question(question) and not re.fullmatch(r"\d{4}", norm_key(answer)):
        flags.append("year_question_non_year_gold")
    return dedupe(flags)


def proposal_delta(current_variants: list[str], proposal: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not proposal:
        return [], []
    proposed = dedupe(as_list(proposal.get("accepted_answer_variants")) + as_list(proposal.get("alternate_valid_answers")))
    current_keys = {norm_key(value) for value in current_variants}
    proposed_keys = {norm_key(value) for value in proposed}
    additions = [value for value in proposed if norm_key(value) not in current_keys]
    removals = [value for value in current_variants if norm_key(value) not in proposed_keys]
    return additions, removals


def priority(flags: list[str]) -> str:
    severe = {
        "stepfun_question_rewrite",
        "question_marked_not_well_formed",
        "question_issues_present",
        "year_question_non_year_gold",
    }
    medium = {
        "stepfun_needs_edit",
        "day_question_full_date_variant",
        "year_question_non_year_variant",
        "month_question_year_or_date_variant",
        "city_question_location_expansion",
        "ratio_variant",
        "stepfun_rejected_existing_variant",
    }
    flag_set = set(flags)
    if flag_set & severe:
        return "high"
    if flag_set & medium:
        return "medium"
    if flags:
        return "low"
    return "none"


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = read_jsonl(DATASET_PATH)
    proposal_rows = read_jsonl(PROPOSAL_PATH)
    proposals: dict[tuple[str, int], dict[str, Any]] = {}
    for row in proposal_rows:
        chain_id = str(row.get("chain_id") or "")
        hop_number = int(row.get("hop_number") or 0)
        proposals[(chain_id, hop_number)] = row.get("proposal") or {}

    review_rows: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    company_counts: Counter[str] = Counter()
    hop_type_counts: Counter[str] = Counter()
    missing_proposals = 0
    for chain in dataset:
        chain_id = str(chain.get("chain_id") or "")
        for fallback_idx, hop in enumerate(chain.get("hops") or [], start=1):
            hop_number = int(hop.get("hop_number") or fallback_idx)
            proposal = proposals.get((chain_id, hop_number))
            if proposal is None:
                missing_proposals += 1
            answer = str(hop.get("answer") or "")
            variants = dedupe([answer, *as_list(hop.get("accepted_answer_variants")), *as_list(hop.get("alternate_valid_answers"))])
            additions, removals = proposal_delta(variants, proposal)
            flags = question_risk_flags(str(hop.get("question") or ""), answer, variants, hop, proposal)
            if additions:
                flags.append("stepfun_proposed_additions")
            if removals:
                flags.append("stepfun_proposed_removals")
            flags = dedupe(flags)
            row_priority = priority(flags)
            flag_counts.update(flags)
            priority_counts[row_priority] += 1
            company_counts[str(chain.get("company") or "")] += 1
            hop_type_counts[str(hop.get("hop_type") or "")] += 1
            review_rows.append(
                {
                    "chain_id": chain_id,
                    "problem_id": chain.get("problem_id"),
                    "task_id": chain.get("task_id"),
                    "company": chain.get("company"),
                    "pattern": chain.get("pattern"),
                    "hop_number": hop_number,
                    "hop_type": hop.get("hop_type"),
                    "priority": row_priority,
                    "flags": flags,
                    "question": hop.get("question"),
                    "answer": answer,
                    "accepted_variants": variants,
                    "proposal_additions": additions,
                    "proposal_removals": removals,
                    "proposal_rejected_existing": as_list((proposal or {}).get("rejected_existing_answer_variants")),
                    "proposal_question_rewrite": (proposal or {}).get("question_rewrite"),
                    "proposal_rationale": (proposal or {}).get("rationale"),
                    "proposal_confidence": (proposal or {}).get("confidence"),
                    "question_issues": hop.get("question_issues"),
                    "suggested_rewrite": hop.get("suggested_rewrite"),
                    "supporting_span": truncate(hop.get("supporting_span")),
                    "doc_ref": hop.get("doc_ref"),
                }
            )

    summary = {
        "dataset": str(DATASET_PATH),
        "proposal_file": str(PROPOSAL_PATH),
        "chains": len(dataset),
        "hops": len(review_rows),
        "proposal_rows": len(proposal_rows),
        "hops_missing_proposal": missing_proposals,
        "priority_counts": dict(priority_counts),
        "flag_counts": dict(flag_counts.most_common()),
        "company_hop_counts": dict(company_counts),
        "hop_type_counts": dict(hop_type_counts),
    }
    return review_rows, summary


def joined(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "flags",
        "chain_id",
        "task_id",
        "company",
        "pattern",
        "hop_number",
        "hop_type",
        "question",
        "answer",
        "accepted_variants",
        "proposal_additions",
        "proposal_removals",
        "proposal_rejected_existing",
        "proposal_question_rewrite",
        "proposal_rationale",
        "proposal_confidence",
        "question_issues",
        "suggested_rewrite",
        "supporting_span",
        "doc_ref",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: joined(row.get(key)) for key in fieldnames})


def esc(value: Any) -> str:
    return html.escape(str(value or ""))


def chips(values: list[str], kind: str = "") -> str:
    cls = f"chip {kind}".strip()
    return " ".join(f'<span class="{cls}">{esc(value)}</span>' for value in values)


def badge(value: str, kind: str = "") -> str:
    cls = f"badge {kind}".strip()
    return f'<span class="{cls}">{esc(value)}</span>'


def render_rows(rows: list[dict[str, Any]], *, only_flagged: bool) -> str:
    selected = [row for row in rows if row["priority"] != "none"] if only_flagged else rows
    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    selected.sort(key=lambda row: (order.get(row["priority"], 9), str(row["chain_id"]), int(row["hop_number"])))
    body: list[str] = []
    for row in selected:
        body.append(
            "<tr>"
            f'<td>{badge(row["priority"], row["priority"])}</td>'
            f"<td>{chips(row['flags'], 'flag')}</td>"
            f'<td class="mono">{esc(row["chain_id"])}<br>H{esc(row["hop_number"])}</td>'
            f"<td>{esc(row['company'])}<br><span class=\"muted\">{esc(row['pattern'])} / {esc(row['hop_type'])}</span></td>"
            f"<td>{esc(row['question'])}</td>"
            f'<td class="answer">{esc(row["answer"])}</td>'
            f"<td>{chips(row['accepted_variants'])}</td>"
            f"<td>{chips(row['proposal_additions'], 'add')}</td>"
            f"<td>{chips(row['proposal_removals'], 'remove')}</td>"
            f"<td>{chips(row['proposal_rejected_existing'], 'remove')}</td>"
            f'<td class="rewrite">{esc(row["proposal_question_rewrite"] or row["suggested_rewrite"])}</td>'
            f'<td class="note">{esc(row["proposal_rationale"])}</td>'
            f'<td class="support">{esc(row["supporting_span"])}</td>'
            "</tr>"
        )
    return "\n".join(body)


def write_html(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    priority_cards = "".join(
        f'<div class="card"><span class="num">{summary["priority_counts"].get(key, 0)}</span><span>{key} priority</span></div>'
        for key in ["high", "medium", "low", "none"]
    )
    top_flags = list(summary["flag_counts"].items())[:18]
    flag_cards = "".join(f'<span class="flagcount">{esc(flag)}: <b>{count}</b></span>' for flag, count in top_flags)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy HopQA Full Dataset Review</title>
  <style>
    :root {{
      --bg: #eef2e7;
      --paper: #fbfaf1;
      --ink: #1f271f;
      --muted: #65705f;
      --line: #cbd5bd;
      --high: #9a2f1f;
      --medium: #946000;
      --low: #3d6f84;
      --add: #1f6b45;
      --remove: #9a2f1f;
    }}
    body {{ margin: 0; background: linear-gradient(135deg, #eef2e7, #dedcc9); color: var(--ink); font-family: Georgia, "Times New Roman", serif; }}
    header {{ padding: 30px 38px 18px; border-bottom: 1px solid var(--line); background: radial-gradient(circle at top right, #f7e1b8, transparent 30rem); }}
    main {{ padding: 18px 28px 46px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin-top: 30px; }}
    p {{ color: var(--muted); max-width: 1100px; line-height: 1.45; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, minmax(130px, 1fr)); gap: 10px; margin: 16px 0; }}
    .card {{ background: rgba(251,250,241,.9); border: 1px solid var(--line); border-radius: 14px; padding: 12px; box-shadow: 0 2px 9px rgba(41, 53, 37, .08); }}
    .num {{ display: block; font-size: 27px; font-weight: 700; }}
    .toolbar {{ position: sticky; top: 0; z-index: 10; background: rgba(238,242,231,.96); padding: 12px 0; backdrop-filter: blur(8px); }}
    input {{ width: min(860px, 100%); padding: 12px 14px; border-radius: 999px; border: 1px solid var(--line); background: #fffdf5; }}
    table {{ width: 100%; border-collapse: collapse; background: rgba(251,250,241,.94); border: 1px solid var(--line); margin: 12px 0 28px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; vertical-align: top; text-align: left; font-size: 12px; line-height: 1.35; }}
    th {{ position: sticky; top: 58px; background: #e1e7d4; z-index: 5; text-transform: uppercase; letter-spacing: .04em; font-size: 11px; }}
    .mono {{ font-family: "Courier New", monospace; }}
    .muted {{ color: var(--muted); }}
    .answer {{ font-weight: 700; white-space: nowrap; }}
    .support {{ max-width: 360px; color: #3f493d; }}
    .note, .rewrite {{ max-width: 330px; }}
    .chip, .badge, .flagcount {{ display: inline-block; margin: 2px 3px 2px 0; padding: 3px 7px; border-radius: 999px; border: 1px solid var(--line); background: #fffef8; }}
    .chip.flag {{ background: #e9efe0; font-family: "Courier New", monospace; font-size: 11px; }}
    .chip.add {{ background: #dcedd8; color: var(--add); border-color: #b7d4ad; }}
    .chip.remove {{ background: #f5d5cc; color: var(--remove); border-color: #e7b5aa; }}
    .badge {{ font-family: "Courier New", monospace; font-size: 11px; border: 0; }}
    .badge.high {{ background: #f5d5cc; color: var(--high); }}
    .badge.medium {{ background: #ffe3ad; color: var(--medium); }}
    .badge.low {{ background: #d7e8f2; color: var(--low); }}
    .badge.none {{ background: #e1eadb; color: #44613f; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>Privacy HopQA Full Dataset Review</h1>
    <p>Review queue across the main finalized dataset. This page combines current accepted variants, latest StepFun answer-unit proposals, existing question issue metadata, and conservative risk flags. It does not imply every flag is a bug; it is a prioritization surface for source-grounded cleanup.</p>
    <div class="cards">
      <div class="card"><span class="num">{summary['chains']}</span><span>chains</span></div>
      <div class="card"><span class="num">{summary['hops']}</span><span>hops</span></div>
      <div class="card"><span class="num">{summary['proposal_rows']}</span><span>StepFun proposal rows</span></div>
      <div class="card"><span class="num">{summary['hops_missing_proposal']}</span><span>hops missing proposal</span></div>
      {priority_cards}
    </div>
    <p>{flag_cards}</p>
  </header>
  <main>
    <div class="toolbar"><input id="search" placeholder="Filter by chain id, company, flag, question, answer, variant, or support span"></div>
    <h2>Flagged review queue</h2>
    <table data-filterable>
      <thead><tr><th>Priority</th><th>Flags</th><th>Chain/Hop</th><th>Context</th><th>Question</th><th>Gold</th><th>Accepted variants</th><th>StepFun additions</th><th>StepFun removals</th><th>Rejected existing</th><th>Rewrite</th><th>Rationale</th><th>Support</th></tr></thead>
      <tbody>{render_rows(rows, only_flagged=True)}</tbody>
    </table>
    <h2>All hops</h2>
    <table data-filterable>
      <thead><tr><th>Priority</th><th>Flags</th><th>Chain/Hop</th><th>Context</th><th>Question</th><th>Gold</th><th>Accepted variants</th><th>StepFun additions</th><th>StepFun removals</th><th>Rejected existing</th><th>Rewrite</th><th>Rationale</th><th>Support</th></tr></thead>
      <tbody>{render_rows(rows, only_flagged=False)}</tbody>
    </table>
  </main>
  <script>
    const search = document.getElementById('search');
    const rows = Array.from(document.querySelectorAll('tbody tr'));
    search.addEventListener('input', () => {{
      const q = search.value.trim().toLowerCase();
      for (const row of rows) {{
        row.classList.toggle('hidden', q && !row.textContent.toLowerCase().includes(q));
      }}
    }});
  </script>
</body>
</html>
"""
    OUT_HTML.write_text(html_text, encoding="utf-8")


def main() -> None:
    rows, summary = build_rows()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(rows)
    write_html(rows, summary)
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
