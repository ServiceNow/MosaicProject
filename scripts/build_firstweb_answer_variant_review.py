#!/usr/bin/env python3
"""Build a self-contained first-web answer-variant review artifact."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/home/toolkit/my_code/MosaicProject")
DATASET_PATH = ROOT / "data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-07.jsonl"
STRICT_AUDIT_PATH = ROOT / "data_files/privacy_hopqa_first_web_hop_dataset/stepfun_firstweb_strict_variant_audit_2026-05-09.jsonl"
CURATED_PATH = ROOT / "data_files/privacy_hopqa_first_web_hop_dataset/strict_variant_curated_acceptance_2026-05-09.json"
AMBIGUITY_MD_PATH = ROOT / "runtime_analysis/firstweb_ambiguity_audit_2026-05-09.md"
CANDIDATE_PATH = ROOT / "runtime_analysis/firstweb_reward_plateau_candidate_chains.jsonl"
OUT_HTML = ROOT / "runtime_analysis/firstweb_answer_variant_review_2026-05-09.html"
OUT_CSV = ROOT / "runtime_analysis/firstweb_answer_variant_review_2026-05-09.csv"


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


def first_hop(row: dict[str, Any]) -> dict[str, Any]:
    hops = row.get("hops") or []
    return hops[0] if hops and isinstance(hops[0], dict) else {}


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    rows: list[dict[str, str]] = []
    headers: list[str] = []
    for line in lines:
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not headers:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def joined(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value or "")


def truncate(text: Any, limit: int = 1200) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + " ... [truncated]"


def badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="badge {kind}">{html.escape(label)}</span>'


def td(value: Any, cls: str = "") -> str:
    cls_attr = f' class="{cls}"' if cls else ""
    return f"<td{cls_attr}>{html.escape(str(value or ''))}</td>"


def variant_chips(values: list[str], highlight: set[str] | None = None) -> str:
    highlight = highlight or set()
    chips = []
    for value in values:
        key = norm_key(value)
        cls = "chip new" if key in highlight else "chip"
        chips.append(f'<span class="{cls}">{html.escape(value)}</span>')
    return " ".join(chips)


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = read_jsonl(DATASET_PATH)
    by_chain = {str(row.get("chain_id")): row for row in dataset}
    strict_rows = read_jsonl(STRICT_AUDIT_PATH)
    strict_by_chain = {str(row.get("chain_id")): row for row in strict_rows}
    curated = json.loads(CURATED_PATH.read_text(encoding="utf-8"))
    accepted = curated.get("accepted_variants") or {}
    rejected_notes = curated.get("rejected_notes") or {}
    candidate_rows = {str(row.get("chain_id")): row for row in read_jsonl(CANDIDATE_PATH)}
    ambiguity_rows = parse_markdown_table(AMBIGUITY_MD_PATH)

    changed: list[dict[str, Any]] = []
    for chain_id, new_variants in sorted(accepted.items()):
        row = by_chain.get(chain_id, {})
        hop = first_hop(row)
        strict = strict_by_chain.get(chain_id, {})
        candidate = candidate_rows.get(chain_id, {})
        changed.append(
            {
                "chain_id": chain_id,
                "company": row.get("company"),
                "task_id": row.get("task_id"),
                "question": row.get("question"),
                "gold_answer": row.get("answer"),
                "accepted_variants": as_list(hop.get("accepted_answer_variants")),
                "new_variants": as_list(new_variants),
                "supporting_span": truncate(hop.get("supporting_span")),
                "doc_ref": hop.get("doc_ref"),
                "observed_answers": strict.get("observed_answers") or candidate.get("pred_counts") or [],
                "observed_answer_validity": strict.get("observed_answer_validity") or {},
                "rejected_notes": rejected_notes.get(chain_id) or [],
            }
        )

    problems: list[dict[str, Any]] = []
    for item in ambiguity_rows:
        chain_id = item.get("chain_id", "")
        row = by_chain.get(chain_id, {})
        hop = first_hop(row)
        strict = strict_by_chain.get(chain_id, {})
        candidate = candidate_rows.get(chain_id, {})
        problems.append(
            {
                "chain_id": chain_id,
                "diagnosis": item.get("diagnosis"),
                "action": item.get("action"),
                "question": item.get("current question") or row.get("question"),
                "gold_answer": item.get("answer") or row.get("answer"),
                "proposed_fix": item.get("proposed fix"),
                "notes": item.get("notes"),
                "company": row.get("company"),
                "task_id": row.get("task_id"),
                "accepted_variants": as_list(hop.get("accepted_answer_variants")),
                "supporting_span": truncate(hop.get("supporting_span")),
                "doc_ref": hop.get("doc_ref"),
                "observed_answers": strict.get("observed_answers") or candidate.get("pred_counts") or [],
                "observed_answer_validity": strict.get("observed_answer_validity") or {},
            }
        )

    all_rows: list[dict[str, Any]] = []
    changed_ids = set(accepted)
    problem_ids = {row["chain_id"] for row in problems}
    for row in dataset:
        chain_id = str(row.get("chain_id"))
        hop = first_hop(row)
        strict = strict_by_chain.get(chain_id, {})
        flags: list[str] = []
        if chain_id in changed_ids:
            flags.append("variant_changed")
        if chain_id in problem_ids:
            flags.append("problem_queue")
        if strict.get("new_judged_valid_variants"):
            flags.append("stepfun_new_candidate")
        if strict.get("observed_answers"):
            flags.append("observed_wrong_or_mixed")
        if not flags:
            flags.append("no_flag")
        all_rows.append(
            {
                "chain_id": chain_id,
                "company": row.get("company"),
                "task_id": row.get("task_id"),
                "question": row.get("question"),
                "gold_answer": row.get("answer"),
                "flags": flags,
                "accepted_variants": as_list(hop.get("accepted_answer_variants")),
                "supporting_span": truncate(hop.get("supporting_span"), 700),
                "doc_ref": hop.get("doc_ref"),
            }
        )

    summary = {
        "total_rows": len(dataset),
        "changed_rows": len(changed),
        "new_curated_variants": sum(len(row["new_variants"]) for row in changed),
        "problem_queue_rows": len(problems),
        "strict_audit_rows": len(strict_rows),
        "stepfun_rows_with_new_candidates": sum(1 for row in strict_rows if row.get("new_judged_valid_variants")),
    }
    return changed, problems, all_rows, summary


def write_csv(changed: list[dict[str, Any]], problems: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "chain_id",
        "company",
        "task_id",
        "diagnosis",
        "action",
        "flags",
        "question",
        "gold_answer",
        "accepted_variants",
        "new_variants",
        "proposed_fix",
        "rejected_notes",
        "observed_answers",
        "observed_answer_validity",
        "supporting_span",
        "doc_ref",
        "notes",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for section, rows in [("changed_variants", changed), ("problem_queue", problems), ("all_rows", all_rows)]:
            for row in rows:
                writer.writerow({name: joined(row.get(name)) if name != "section" else section for name in fieldnames})


def render_changed_rows(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        new_keys = {norm_key(value) for value in row["new_variants"]}
        body.append(
            "<tr>"
            + td(row["chain_id"], "mono")
            + td(row["company"])
            + td(row["question"])
            + td(row["gold_answer"], "answer")
            + f"<td>{variant_chips(row['accepted_variants'], new_keys)}</td>"
            + f"<td>{variant_chips(row['new_variants'], new_keys)}</td>"
            + td(joined(row["rejected_notes"]), "note")
            + td(joined(row["observed_answer_validity"]), "note")
            + td(row["supporting_span"], "support")
            + "</tr>"
        )
    return "\n".join(body)


def render_problem_rows(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        action = str(row.get("action") or "")
        kind = "danger" if "drop" in action or "rewrite" in action else "ok" if action == "keep" else "warn"
        body.append(
            "<tr>"
            + td(row["chain_id"], "mono")
            + f"<td>{badge(str(row.get('diagnosis') or ''), 'warn')}</td>"
            + f"<td>{badge(action, kind)}</td>"
            + td(row["question"])
            + td(row["gold_answer"], "answer")
            + td(row["proposed_fix"], "rewrite")
            + td(row["notes"], "note")
            + td(joined(row["observed_answer_validity"]), "note")
            + td(row["supporting_span"], "support")
            + "</tr>"
        )
    return "\n".join(body)


def render_all_rows(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        flags = " ".join(badge(flag, "warn" if flag != "no_flag" else "ok") for flag in row["flags"])
        body.append(
            "<tr>"
            + td(row["chain_id"], "mono")
            + td(row["company"])
            + f"<td>{flags}</td>"
            + td(row["question"])
            + td(row["gold_answer"], "answer")
            + f"<td>{variant_chips(row['accepted_variants'])}</td>"
            + td(row["supporting_span"], "support")
            + "</tr>"
        )
    return "\n".join(body)


def write_html(changed: list[dict[str, Any]], problems: list[dict[str, Any]], all_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>First-Web Answer Variant Review</title>
  <style>
    :root {{
      --ink: #17211b;
      --muted: #5f6f65;
      --paper: #f5f0e6;
      --card: #fffaf0;
      --line: #d9cdb7;
      --new: #165c46;
      --warn: #8a5a00;
      --danger: #8f2f21;
      --ok: #2f6d3c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #e4dcc7, transparent 34rem), linear-gradient(120deg, #f7f2e8, #ece1cd);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    header {{
      padding: 32px 42px 20px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: -0.02em; }}
    h2 {{ margin: 28px 0 10px; font-size: 22px; }}
    p {{ color: var(--muted); max-width: 1100px; line-height: 1.45; }}
    main {{ padding: 20px 32px 48px; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, minmax(130px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 14px; box-shadow: 0 1px 8px rgba(59, 44, 23, 0.08); }}
    .num {{ display: block; font-size: 28px; font-weight: 700; }}
    .label {{ color: var(--muted); font-size: 13px; }}
    .toolbar {{ position: sticky; top: 0; z-index: 10; background: rgba(245, 240, 230, 0.95); padding: 12px 0; backdrop-filter: blur(8px); }}
    input {{ width: min(760px, 100%); padding: 12px 14px; border-radius: 999px; border: 1px solid var(--line); background: #fffdf7; color: var(--ink); font-size: 15px; }}
    table {{ width: 100%; border-collapse: collapse; background: rgba(255, 250, 240, 0.88); border: 1px solid var(--line); margin: 12px 0 28px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; vertical-align: top; text-align: left; font-size: 13px; line-height: 1.35; }}
    th {{ position: sticky; top: 58px; background: #efe4cf; z-index: 5; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    tr:hover td {{ background: #fff6df; }}
    .mono {{ font-family: "Courier New", monospace; font-size: 12px; }}
    .answer {{ font-weight: 700; white-space: nowrap; }}
    .support {{ max-width: 420px; color: #39443c; }}
    .note {{ max-width: 360px; color: #514539; }}
    .rewrite {{ max-width: 360px; font-weight: 600; }}
    .chip {{ display: inline-block; margin: 2px 3px 2px 0; padding: 3px 7px; border-radius: 999px; border: 1px solid var(--line); background: #fffdf8; }}
    .chip.new {{ color: white; background: var(--new); border-color: var(--new); }}
    .badge {{ display: inline-block; margin: 1px 2px; padding: 3px 7px; border-radius: 999px; font-size: 11px; font-family: "Courier New", monospace; background: #eee3ce; }}
    .badge.warn {{ background: #ffe3ad; color: var(--warn); }}
    .badge.danger {{ background: #ffd5cc; color: var(--danger); }}
    .badge.ok {{ background: #d7ead4; color: var(--ok); }}
    .policy {{ background: #fff8e8; border-left: 4px solid var(--new); padding: 12px 16px; margin: 14px 0 22px; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>First-Web Answer Variant Review</h1>
    <p>Self-contained audit page for checking answer aliases and remaining problem rows. All accepted aliases are intended to preserve the existing gold answer and be grounded in the gold document context; external sources are not used to define targets.</p>
    <div class="cards">
      <div class="card"><span class="num">{summary['total_rows']}</span><span class="label">first-web rows</span></div>
      <div class="card"><span class="num">{summary['changed_rows']}</span><span class="label">rows changed</span></div>
      <div class="card"><span class="num">{summary['new_curated_variants']}</span><span class="label">new curated variants</span></div>
      <div class="card"><span class="num">{summary['problem_queue_rows']}</span><span class="label">problem queue rows</span></div>
      <div class="card"><span class="num">{summary['strict_audit_rows']}</span><span class="label">StepFun audited rows</span></div>
      <div class="card"><span class="num">{summary['stepfun_rows_with_new_candidates']}</span><span class="label">rows with new candidates</span></div>
    </div>
  </header>
  <main>
    <div class="toolbar">
      <input id="search" placeholder="Filter by chain id, question, answer, variant, diagnosis, or support span">
    </div>
    <section class="policy">
      <strong>Acceptance policy:</strong> accept only short exact-match aliases that answer the exact question with the same semantics as the gold answer and are supported by the gold document context. Reject verbose year aliases, source/time-shifted answers, inverted ratios, broader/narrower entities, and wrong answer types.
    </section>

    <h2>Changed answer variants to verify</h2>
    <table data-filterable>
      <thead><tr><th>Chain</th><th>Company</th><th>Question</th><th>Gold</th><th>Accepted variants</th><th>New variants</th><th>Rejected notes</th><th>Observed answer validity</th><th>Gold support span</th></tr></thead>
      <tbody>{render_changed_rows(changed)}</tbody>
    </table>

    <h2>Remaining potential problems queue</h2>
    <p>These are not all label changes. Rows marked rewrite/drop/source-check need direct gold-document review before changing task text or gold semantics.</p>
    <table data-filterable>
      <thead><tr><th>Chain</th><th>Diagnosis</th><th>Action</th><th>Current question</th><th>Gold</th><th>Proposed fix</th><th>Notes</th><th>Observed answer validity</th><th>Gold support span</th></tr></thead>
      <tbody>{render_problem_rows(problems)}</tbody>
    </table>

    <h2>All first-web rows with flags</h2>
    <table data-filterable>
      <thead><tr><th>Chain</th><th>Company</th><th>Flags</th><th>Question</th><th>Gold</th><th>Accepted variants</th><th>Gold support span</th></tr></thead>
      <tbody>{render_all_rows(all_rows)}</tbody>
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
    changed, problems, all_rows, summary = build_rows()
    write_csv(changed, problems, all_rows)
    write_html(changed, problems, all_rows, summary)
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
