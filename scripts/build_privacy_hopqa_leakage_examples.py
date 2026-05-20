#!/usr/bin/env python3
"""Build a paper-facing gallery of real Privacy HopQA leakage examples."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_mosaic_v11_20260514/"
    "model_runs/qwen3_4b_instruct_2507/tables/privacy_leakage_suite_v2_stepfun_full.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper_assets/privacy_hopqa_prompts/leakage_examples.html"

SELECTED_ROLLOUT_IDS = [
    "DR0028_288c9225_46ec811d",
    "DR0021_662ac292_5f81d4b8",
    "DR0039_15a5e6f7_6c6d8781",
    "DR0048_acaebdac_c8a771af",
    "DR0059_9ecba6eb_e84cc39b",
    "DR0066_b1251225_fee0db2e",
    "DR0017_0dffc549_e72c93c6",
    "DR0035_3d3e2fdd_c388498c",
    "DR0065_51b97c6b_c30922c4",
    "DR0077_e93f7168_7ac9ecdf",
    "DR0099_47983d2e_14e5ff20",
    "DR0099_f774a5d3_579ce00f",
    "DR0065_a5ef6264_f5773a43",
]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def load_rows(source: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rollout_id = row.get("rollout_id")
            if rollout_id:
                rows[rollout_id] = row
    return rows


def list_items(items: list[Any], class_name: str = "") -> str:
    cls = f' class="{class_name}"' if class_name else ""
    if not items:
        return f"<p{cls}>None recorded.</p>"
    return "<ol{}>{}</ol>".format(
        cls,
        "\n".join(f"<li>{esc(item)}</li>" for item in items),
    )


def build_fact_lookup(row: dict[str, Any]) -> dict[str, dict[str, str]]:
    hidden = row.get("hidden_reference_for_judges_only") or {}
    lookup: dict[str, dict[str, str]] = {}
    for fact in hidden.get("local_hop_facts") or []:
        fact_id = fact.get("fact_id")
        if fact_id:
            lookup[fact_id] = {
                "id": fact_id,
                "question": fact.get("question", ""),
                "answer": fact.get("canonical_answer", ""),
                "source": "local hop",
            }
    for qa in hidden.get("private_qas") or []:
        qa_id = qa.get("qa_id")
        if qa_id:
            lookup[qa_id] = {
                "id": qa_id,
                "question": qa.get("question", ""),
                "answer": qa.get("canonical_answer", ""),
                "source": "private QA",
            }
    return lookup


def render_queries(row: dict[str, Any]) -> str:
    queries = [
        query.get("query", "")
        for query in (row.get("model_input") or {}).get("web_queries") or []
    ]
    return list_items(queries, "query-list")


def render_intent(row: dict[str, Any]) -> str:
    output = row.get("intent_adversary_output") or {}
    predictions = output.get("predicted_subquestions") or []
    if not predictions and output.get("predicted_subquestion"):
        predictions = output.get("predicted_subquestion")
    if isinstance(predictions, str):
        predictions = [predictions]
    return list_items(predictions, "prediction-list intent-list")


def render_answers(row: dict[str, Any]) -> str:
    answers = (row.get("answer_adversary_output") or {}).get("answers") or []
    judgments = {
        judgment.get("qa_id"): judgment
        for judgment in (row.get("answer_judge_output") or {}).get("answer_judgments") or []
    }
    if not answers:
        return "<p>No answer predictions recorded.</p>"
    parts = ['<ol class="prediction-list answer-list">']
    for answer in answers:
        qa_id = answer.get("qa_id", "")
        judgment = judgments.get(qa_id, {})
        label = judgment.get("judgment", "not judged")
        confidence = answer.get("confidence", "unknown")
        parts.append(
            '<li><div class="item-head">'
            f'<span class="code">{esc(qa_id)}</span>'
            f'<span class="badge answer">{esc(confidence)}</span>'
            f'<span class="badge {esc(label)}">{esc(label)}</span>'
            "</div>"
            f"<p>{esc(answer.get('answer', ''))}</p></li>"
        )
    parts.append("</ol>")
    return "\n".join(parts)


def render_claims(row: dict[str, Any]) -> str:
    claims = (row.get("claim_adversary_output") or {}).get("claims") or []
    judgments = {
        judgment.get("claim_index"): judgment
        for judgment in (row.get("claim_judge_output") or {}).get("claim_judgments") or []
    }
    if not claims:
        return "<p>No claim predictions recorded.</p>"
    parts = ['<ol class="prediction-list claim-list">']
    for index, claim in enumerate(claims):
        judgment = judgments.get(index, {})
        label = judgment.get("label", "not judged")
        matched = judgment.get("matched_fact_id", "NONE")
        parts.append(
            '<li><div class="item-head">'
            f'<span class="badge {esc(label)}">{esc(label)}</span>'
            f'<span class="code">matched: {esc(matched)}</span>'
            "</div>"
            f"<p>{esc(claim)}</p></li>"
        )
    parts.append("</ol>")
    return "\n".join(parts)


def render_targets(row: dict[str, Any]) -> str:
    fact_lookup = build_fact_lookup(row)
    hidden = row.get("hidden_reference_for_judges_only") or {}
    ids: list[str] = []
    for fact in hidden.get("local_hop_facts") or []:
        fact_id = fact.get("fact_id")
        if fact_id:
            ids.append(fact_id)
    ids.extend((row.get("answer_judge_output") or {}).get("correct_qa_ids") or [])
    ids.extend((row.get("claim_judge_output") or {}).get("matched_fact_ids") or [])

    seen: set[str] = set()
    parts = ['<div class="target-grid">']
    for fact_id in ids:
        if fact_id in seen or fact_id not in fact_lookup:
            continue
        seen.add(fact_id)
        fact = fact_lookup[fact_id]
        parts.append(
            '<div class="target">'
            f'<div class="item-head"><span class="code">{esc(fact["id"])}</span>'
            f'<span class="badge target-kind">{esc(fact["source"])}</span></div>'
            f'<p class="target-q">{esc(fact["question"])}</p>'
            f'<p class="target-a">Answer: <strong>{esc(fact["answer"])}</strong></p>'
            "</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def render_example(row: dict[str, Any], index: int) -> str:
    intent_judge = row.get("intent_judge_output") or {}
    answer_judge = row.get("answer_judge_output") or {}
    claim_judge = row.get("claim_judge_output") or {}
    title = f"{index}. {row.get('company', 'Unknown company')} - {row.get('task_id', '')}"
    rollout_id = row.get("rollout_id", "")
    score = row.get("overall_privacy_score_1_to_5", "")
    pattern = row.get("pattern", "")
    intent_score = intent_judge.get("intent_subquestion_score_1_to_5", "")
    badges = [
        ("intent", f"intent leaked: {intent_judge.get('intent_leaked')}"),
        ("answer", f"answer leaked: {answer_judge.get('answer_leaked')}"),
        ("privacy", f"full-info leaked: {claim_judge.get('full_info_leaked')}"),
        ("neutral", f"score {score}/5"),
        ("neutral", f"intent score {intent_score}/5"),
        ("neutral", f"pattern {pattern}"),
    ]
    badge_html = "\n".join(
        f'<span class="badge {kind}">{esc(text)}</span>' for kind, text in badges
    )
    nav_label = f"{row.get('task_id', '')} {row.get('company', '')}"
    return f"""
    <article class="example-card" id="{esc(rollout_id)}">
      <header class="card-head">
        <div>
          <div class="kicker">Strict leakage example</div>
          <h2>{esc(title)}</h2>
          <p class="row-id">{esc(rollout_id)}</p>
        </div>
        <div class="badge-row">{badge_html}</div>
      </header>
      <div class="example-grid">
        <section class="panel query-panel">
          <h3>Visible web queries</h3>
          {render_queries(row)}
        </section>
        <section class="panel intent-panel">
          <h3>Intent predictions</h3>
          {render_intent(row)}
        </section>
        <section class="panel answer-panel">
          <h3>Answer predictions</h3>
          {render_answers(row)}
        </section>
        <section class="panel privacy-panel">
          <h3>Full-information claims</h3>
          {render_claims(row)}
        </section>
      </div>
      <section class="panel targets-panel">
        <h3>Hidden local targets used by judges</h3>
        {render_targets(row)}
      </section>
    </article>
    <a class="mini-nav-link" href="#{esc(rollout_id)}">{esc(nav_label)}</a>
    """


def render_page(rows: list[dict[str, Any]], source: Path) -> str:
    cards = "\n".join(render_example(row, i + 1) for i, row in enumerate(rows))
    nav_links = "\n".join(
        f'<a href="#{esc(row.get("rollout_id", ""))}">{esc(i + 1)}. {esc(row.get("task_id", ""))}</a>'
        for i, row in enumerate(rows)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy HopQA Leakage Examples</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5e6a7a;
      --rule: #d8dee8;
      --paper: #f7f8f5;
      --panel: #ffffff;
      --query: #2457a6;
      --intent: #9a6500;
      --answer: #166b54;
      --privacy: #a33a2f;
      --soft-query: #edf4ff;
      --soft-intent: #fff6df;
      --soft-answer: #eaf7f1;
      --soft-privacy: #fff0ed;
      --soft-neutral: #f3f5f8;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header.page-head {{
      padding: 34px 42px 24px;
      background: #ffffff;
      border-bottom: 1px solid var(--rule);
    }}
    h1, h2, h3, p {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 34px; line-height: 1.1; }}
    h2 {{ font-size: 22px; line-height: 1.18; }}
    h3 {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    p {{ color: var(--muted); }}
    .kicker {{
      color: var(--answer);
      font-size: 12px;
      font-weight: 850;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin-bottom: 7px;
    }}
    .page-head p {{ max-width: 1020px; margin-top: 10px; }}
    .top-nav, .example-nav {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .top-nav a, .example-nav a {{
      color: #ffffff;
      background: #22324d;
      border-radius: 6px;
      padding: 6px 10px;
      text-decoration: none;
      font-size: 13px;
      font-weight: 800;
    }}
    .example-nav a {{ background: var(--query); }}
    main {{ padding: 26px 42px 48px; }}
    .source-note {{
      max-width: 1180px;
      border: 1px solid var(--rule);
      background: #ffffff;
      border-radius: 8px;
      padding: 13px 15px;
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .source-note code {{
      color: var(--ink);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }}
    .example-card {{
      max-width: 1380px;
      margin: 0 0 28px;
      border: 1px solid var(--rule);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
      box-shadow: 0 8px 22px rgba(28, 40, 66, .06);
    }}
    .card-head {{
      display: flex;
      gap: 16px;
      align-items: flex-start;
      justify-content: space-between;
      padding: 18px 20px;
      background: #fbfcfd;
      border-bottom: 1px solid var(--rule);
    }}
    .row-id {{
      margin-top: 5px;
      font-size: 12px;
      color: var(--muted);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }}
    .badge-row, .item-head {{
      display: flex;
      gap: 6px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 3px 8px;
      border: 1px solid var(--rule);
      background: var(--soft-neutral);
      color: var(--ink);
      font-size: 11px;
      font-weight: 850;
      white-space: nowrap;
    }}
    .badge.intent {{ border-color: #e2c876; background: var(--soft-intent); color: var(--intent); }}
    .badge.answer, .badge.correct {{ border-color: #afd8c7; background: var(--soft-answer); color: var(--answer); }}
    .badge.privacy, .badge.true_private_fact {{ border-color: #e4b5ac; background: var(--soft-privacy); color: var(--privacy); }}
    .badge.partial_private_fact {{ border-color: #dfcf9b; background: var(--soft-intent); color: var(--intent); }}
    .badge.generic_or_public, .badge.unsupported, .badge.incorrect, .badge.not {{
      border-color: var(--rule);
      background: var(--soft-neutral);
      color: var(--muted);
    }}
    .badge.target-kind {{ border-color: #c6ceda; background: #ffffff; color: var(--muted); }}
    .code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
    }}
    .example-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0;
    }}
    .panel {{
      padding: 16px 18px;
      border-bottom: 1px solid var(--rule);
      min-width: 0;
    }}
    .example-grid .panel:nth-child(odd) {{ border-right: 1px solid var(--rule); }}
    .query-panel {{ background: var(--soft-query); }}
    .intent-panel {{ background: var(--soft-intent); }}
    .answer-panel {{ background: var(--soft-answer); }}
    .privacy-panel {{ background: var(--soft-privacy); }}
    .targets-panel {{ background: #ffffff; border-bottom: 0; }}
    ol, ul {{
      margin: 0;
      padding-left: 22px;
    }}
    li {{
      margin: 0 0 8px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }}
    .prediction-list li {{
      background: rgba(255, 255, 255, .72);
      border: 1px solid rgba(215, 221, 231, .9);
      border-radius: 7px;
      padding: 8px 9px;
      list-style-position: outside;
    }}
    .prediction-list p {{
      margin-top: 5px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 720;
    }}
    .target-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 10px;
    }}
    .target {{
      border: 1px dashed #b9c1cc;
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }}
    .target-q {{
      margin-top: 7px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 760;
    }}
    .target-a {{
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
    }}
    .mini-nav-link {{ display: none; }}
    @media (max-width: 900px) {{
      header.page-head, main {{ padding-left: 20px; padding-right: 20px; }}
      .card-head {{ display: block; }}
      .badge-row {{ margin-top: 12px; }}
      .example-grid {{ grid-template-columns: 1fr; }}
      .example-grid .panel:nth-child(odd) {{ border-right: 0; }}
    }}
    @media print {{
      body {{ background: #ffffff; }}
      .top-nav, .example-nav, .source-note {{ display: none; }}
      header.page-head {{ padding: 18px 22px; }}
      main {{ padding: 0; }}
      .example-card {{ break-inside: avoid; box-shadow: none; border-radius: 0; }}
    }}
  </style>
</head>
<body>
  <header class="page-head">
    <div class="kicker">Saved leakage-run examples</div>
    <h1>Real web-query traces and adversary predictions</h1>
    <p>These are selected rows from the saved Privacy HopQA leakage suite where intent leakage, answer leakage, and full-information leakage were all judged positive. Query strings and prediction text are rendered from the saved JSONL run.</p>
    <nav class="top-nav" aria-label="Related pages">
      <a href="index.html">Prompt viewer</a>
      <a href="leakage_diagrams.html">Diagram concepts</a>
    </nav>
    <nav class="example-nav" aria-label="Example jumps">
      {nav_links}
    </nav>
  </header>
  <main>
    <div class="source-note">
      Source artifact: <code>{esc(source)}</code>. Outputs shown here are parsed adversary predictions and judge labels, not paraphrases of the run.
    </div>
    {cards}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows_by_id = load_rows(args.source)
    missing = [row_id for row_id in SELECTED_ROLLOUT_IDS if row_id not in rows_by_id]
    if missing:
        raise SystemExit(f"Missing selected rollout ids: {missing}")
    rows = [rows_by_id[row_id] for row_id in SELECTED_ROLLOUT_IDS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_page(rows, args.source), encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} examples")


if __name__ == "__main__":
    main()
