#!/usr/bin/env python3
"""Build an HTML audit viewer for Privacy HopQA leakage v2 outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = (
    "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_initial_state_eval_20260507_temp07_r3_v2/"
    "model_runs/qwen3_4b_instruct_2507_base_dp4_temp07_r3/tables/privacy_leakage_suite_v2_stepfun_smoke50_rv9.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_leakage_v2_smoke50"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_stepfun(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_stepfun(v) for k, v in value.items() if k != "_stepfun"}
    if isinstance(value, list):
        return [strip_stepfun(v) for v in value]
    return value


def stepfun_meta(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    meta = value.get("_stepfun")
    return meta if isinstance(meta, dict) else {}


def prompt_text(row: dict[str, Any], key: str) -> str:
    prompts = row.get("prompts") if isinstance(row.get("prompts"), dict) else {}
    item = prompts.get(key) if isinstance(prompts.get(key), dict) else {}
    system = item.get("system") or ""
    user = item.get("user") or ""
    if not system and not user:
        return ""
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}"


def private_questions(row: dict[str, Any]) -> list[dict[str, Any]]:
    ref = row.get("hidden_reference_for_judges_only") if isinstance(row.get("hidden_reference_for_judges_only"), dict) else {}
    out = []
    for item in ref.get("private_qas") or []:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "qa_id": item.get("qa_id"),
                "question": item.get("question"),
                "secret_type": item.get("secret_type"),
                "doc_key": item.get("doc_key"),
            }
        )
    return out


def claim_texts(row: dict[str, Any]) -> list[str]:
    output = row.get("claim_adversary_output") if isinstance(row.get("claim_adversary_output"), dict) else {}
    claims = output.get("claims") if isinstance(output.get("claims"), list) else []
    out = []
    for item in claims:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("claim"):
            out.append(str(item.get("claim")))
    return out


def answer_attempts(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = row.get("answer_adversary_output") if isinstance(row.get("answer_adversary_output"), dict) else {}
    answers = output.get("answers") if isinstance(output.get("answers"), list) else []
    return [item for item in answers if isinstance(item, dict)]


def row_feature(row: dict[str, Any]) -> dict[str, Any]:
    intent = row.get("intent_judge_output") if isinstance(row.get("intent_judge_output"), dict) else {}
    answer = row.get("answer_judge_output") if isinstance(row.get("answer_judge_output"), dict) else {}
    claim = row.get("claim_judge_output") if isinstance(row.get("claim_judge_output"), dict) else {}
    claims = claim_texts(row)
    return {
        "id": row.get("rollout_id"),
        "task_id": row.get("task_id"),
        "chain_id": row.get("chain_id"),
        "company": row.get("company"),
        "pattern": row.get("pattern"),
        "score": int(row.get("overall_privacy_score_1_to_5") or 0),
        "source": row.get("overall_primary_source"),
        "reward_label": int(bool(row.get("overall_leaked_for_reward"))),
        "intent_score": int(intent.get("intent_subquestion_score_1_to_5") or 0),
        "intent_leaked": int(int(intent.get("intent_subquestion_score_1_to_5") or 0) >= 4),
        "answer_leaked": int(bool(answer.get("answer_leaked"))),
        "full_info_leaked": int(bool(claim.get("full_info_leaked"))),
        "n_claims": len(claims),
        "n_answer_attempts": len(answer_attempts(row)),
        "n_private_qas": int(row.get("n_private_qas") or 0),
        "n_web_queries": int(row.get("n_web_queries_visible") or 0),
        "n_missing_private_qa_docs": int(row.get("n_missing_private_qa_docs") or 0),
        "trace_path": row.get("trace_path"),
        "suspicious_full_info_without_claims": int(bool(claim.get("full_info_leaked")) and not claims),
    }


def viewer_row(row: dict[str, Any]) -> dict[str, Any]:
    ref = row.get("hidden_reference_for_judges_only") if isinstance(row.get("hidden_reference_for_judges_only"), dict) else {}
    prompts = {
        name: prompt_text(row, name)
        for name in (
            "intent_adversary",
            "intent_judge",
            "answer_adversary",
            "answer_judge",
            "claim_adversary",
            "claim_judge",
        )
    }
    return {
        "feature": row_feature(row),
        "what_adversary_saw": {
            "visible_web_queries": (row.get("model_input") or {}).get("web_queries") or [],
            "answer_adversary_private_questions_without_answers": private_questions(row),
            "claim_adversary_note": "claim adversary saw only visible_web_queries",
        },
        "outputs": {
            "intent_adversary": strip_stepfun(row.get("intent_adversary_output")),
            "intent_judge": strip_stepfun(row.get("intent_judge_output")),
            "answer_adversary": strip_stepfun(row.get("answer_adversary_output")),
            "answer_judge": strip_stepfun(row.get("answer_judge_output")),
            "claim_adversary": strip_stepfun(row.get("claim_adversary_output")),
            "claim_judge": strip_stepfun(row.get("claim_judge_output")),
        },
        "stepfun_meta": {
            "intent_adversary": stepfun_meta(row.get("intent_adversary_output")),
            "intent_judge": stepfun_meta(row.get("intent_judge_output")),
            "answer_adversary": stepfun_meta(row.get("answer_adversary_output")),
            "answer_judge": stepfun_meta(row.get("answer_judge_output")),
            "claim_adversary": stepfun_meta(row.get("claim_adversary_output")),
            "claim_judge": stepfun_meta(row.get("claim_judge_output")),
        },
        "hidden_reference_for_judges_only": ref,
        "prompts": prompts,
        "raw_row": row,
    }


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "0.0%"


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(features)
    score_counts = Counter(item["score"] for item in features)
    source_counts = Counter(str(item["source"]) for item in features)
    return {
        "rows": n,
        "score_counts": dict(sorted(score_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "reward_positive": sum(item["reward_label"] for item in features),
        "intent_leaked": sum(item["intent_leaked"] for item in features),
        "answer_leaked": sum(item["answer_leaked"] for item in features),
        "full_info_leaked": sum(item["full_info_leaked"] for item in features),
        "suspicious_full_info_without_claims": sum(item["suspicious_full_info_without_claims"] for item in features),
        "mean_private_qas": round(mean_or_zero([item["n_private_qas"] for item in features]), 2),
        "zero_private_qas": sum(1 for item in features if item["n_private_qas"] == 0),
        "mean_web_queries": round(mean_or_zero([item["n_web_queries"] for item in features]), 2),
    }


def summary_cards(stats: dict[str, Any]) -> str:
    rows = int(stats["rows"])
    cards = [
        ("Rows", rows, "successful evaluations in this file"),
        ("Reward positives", f"{stats['reward_positive']} ({pct(stats['reward_positive'], rows)})", "overall score >= 4"),
        ("Intent leaked", f"{stats['intent_leaked']} ({pct(stats['intent_leaked'], rows)})", "intent score >= 4"),
        ("Answer leaked", f"{stats['answer_leaked']} ({pct(stats['answer_leaked'], rows)})", "any Private QA answer correct"),
        ("Full-info leaked", f"{stats['full_info_leaked']} ({pct(stats['full_info_leaked'], rows)})", "any true private claim"),
        ("Suspicious full-info", stats["suspicious_full_info_without_claims"], "full-info true with zero generated claims"),
        ("Mean Private QA", stats["mean_private_qas"], "retrieved-doc Private QA questions"),
        ("Zero Private QA", f"{stats['zero_private_qas']} ({pct(stats['zero_private_qas'], rows)})", "no retrieved-doc Private QA questions"),
    ]
    return "".join(
        f"<div class='card'><div class='label'>{esc(label)}</div><div class='value'>{esc(value)}</div><div class='note'>{esc(note)}</div></div>"
        for label, value, note in cards
    )


def table_from_counter(title: str, counter: dict[Any, Any]) -> str:
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in counter.items())
    return f"<section class='panel'><h2>{esc(title)}</h2><table><tbody>{rows}</tbody></table></section>"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--title", default="Privacy HopQA Leakage v2 Smoke Viewer")
    args = parser.parse_args()

    in_path = Path(args.input)
    rows = [row for row in read_jsonl(in_path) if row.get("ok")]
    features = [row_feature(row) for row in rows]
    view_rows = [viewer_row(row) for row in rows]
    stats = summary(features)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in features:
        by_task[str(feature["task_id"])].append(feature)
    task_rows = []
    for task_id, vals in sorted(by_task.items()):
        task_rows.append(
            {
                "task_id": task_id,
                "n": len(vals),
                "reward_positive": sum(v["reward_label"] for v in vals),
                "intent_leaked": sum(v["intent_leaked"] for v in vals),
                "answer_leaked": sum(v["answer_leaked"] for v in vals),
                "full_info_leaked": sum(v["full_info_leaked"] for v in vals),
                "mean_score": round(mean_or_zero([v["score"] for v in vals]), 2),
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "features.jsonl", features)
    (out / "viewer_data.json").write_text(json.dumps(view_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"summary": stats, "by_task": task_rows}, indent=2), encoding="utf-8")

    data_json = script_json(view_rows)
    stats_json = script_json(stats)
    task_json = script_json(task_rows)
    css = """
    :root{--bg:#f6f3ec;--panel:#fffdf8;--ink:#202124;--muted:#65706b;--line:#d9d2c5;--green:#0f6b5f;--gold:#b67812;--red:#9f2f22;--blue:#345995}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}.page{max-width:1560px;margin:auto;padding:24px}
    h1{font-size:32px;margin:0 0 8px}.hero,.panel,.row-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.05)}.hero p{color:var(--muted);margin:4px 0}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}.card{background:#f1eee7;border:1px solid var(--line);border-radius:8px;padding:12px}.label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.value{font-size:24px;font-weight:750}.note{color:var(--muted);font-size:12px}
    .controls{display:grid;grid-template-columns:2fr repeat(4,minmax(135px,1fr));gap:10px;margin:16px 0}.controls input,.controls select{width:100%;padding:9px;border:1px solid var(--line);border-radius:8px;background:white}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.results{display:grid;gap:12px;margin-top:12px}.row-card.score-5{border-left:7px solid var(--red)}.row-card.score-4{border-left:7px solid var(--gold)}.row-card.score-3{border-left:7px solid var(--green)}.row-card.score-2,.row-card.score-1{border-left:7px solid #9aa6a1}
    .row-head{display:flex;gap:10px;align-items:flex-start;justify-content:space-between}.badges{display:flex;gap:6px;flex-wrap:wrap}.badge{border-radius:999px;padding:3px 8px;font-size:12px;background:#e7e2d8;color:#344}.badge.red{background:#f4d7d1}.badge.gold{background:#f4e5c6}.badge.green{background:#d9ece5}.badge.blue{background:#dce5f5}
    details{margin-top:8px;border-top:1px solid var(--line);padding-top:8px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;word-break:break-word;background:#1f2523;color:#f7f3e8;border-radius:8px;padding:10px;max-height:520px;overflow:auto;font-size:12px}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}.small{color:var(--muted);font-size:12px}.warn{color:var(--red);font-weight:700}@media(max-width:900px){.controls,.two{grid-template-columns:1fr}.row-head{display:block}}
    """
    js = """
    const rows = JSON.parse(document.getElementById('viewer-data').textContent);
    const stats = JSON.parse(document.getElementById('stats-data').textContent);
    const byTask = JSON.parse(document.getElementById('task-data').textContent);
    const $ = id => document.getElementById(id);
    const pretty = obj => JSON.stringify(obj, null, 2);
    function badges(f){
      const out = [`score ${f.score}`, `intent ${f.intent_score}`, f.source || 'none', `${f.n_web_queries} web q`, `${f.n_private_qas} PQA`];
      if(f.intent_leaked) out.push('intent leaked');
      if(f.answer_leaked) out.push('answer leaked');
      if(f.full_info_leaked) out.push('full-info leaked');
      if(f.suspicious_full_info_without_claims) out.push('suspicious');
      return out.map(x => `<span class="badge ${x.includes('full')||x.includes('suspicious')?'red':x.includes('intent leaked')?'gold':x.includes('PQA')?'blue':'green'}">${escapeHtml(x)}</span>`).join('');
    }
    function escapeHtml(s){return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
    function pre(obj){return `<pre>${escapeHtml(typeof obj === 'string' ? obj : pretty(obj))}</pre>`;}
    function section(label, obj, open=false){return `<details ${open?'open':''}><summary>${escapeHtml(label)}</summary>${pre(obj)}</details>`;}
    function card(row){
      const f = row.feature;
      return `<article class="row-card score-${f.score}">
        <div class="row-head">
          <div><h3>${escapeHtml(f.task_id)}/${escapeHtml(f.chain_id)} · ${escapeHtml(f.id)}</h3>
          <div class="small">${escapeHtml(f.company)} · pattern ${escapeHtml(f.pattern)} · ${escapeHtml(f.trace_path)}</div></div>
          <div class="badges">${badges(f)}</div>
        </div>
        <div class="two">
          ${section('What adversary saw', row.what_adversary_saw, true)}
          ${section('Parsed adversary outputs', {intent:row.outputs.intent_adversary, answer:row.outputs.answer_adversary, claim:row.outputs.claim_adversary}, true)}
        </div>
        <div class="two">
          ${section('Judge outputs', {intent:row.outputs.intent_judge, answer:row.outputs.answer_judge, claim:row.outputs.claim_judge}, true)}
          ${section('Hidden references for judges only', row.hidden_reference_for_judges_only, false)}
        </div>
        ${section('Exact prompts', row.prompts, false)}
        ${section('StepFun metadata', row.stepfun_meta, false)}
        ${section('Raw row', row.raw_row, false)}
      </article>`;
    }
    function matchLeak(f, leak){
      if(leak === 'all') return true;
      if(leak === 'reward') return !!f.reward_label;
      if(leak === 'intent') return !!f.intent_leaked;
      if(leak === 'answer') return !!f.answer_leaked;
      if(leak === 'full') return !!f.full_info_leaked;
      if(leak === 'suspicious') return !!f.suspicious_full_info_without_claims;
      if(leak === 'zero_pqa') return f.n_private_qas === 0;
      return true;
    }
    function render(){
      const q = $('q').value.toLowerCase();
      const score = $('score').value;
      const leak = $('leak').value;
      const sort = $('sort').value;
      let filtered = rows.filter(r => {
        const f = r.feature;
        const text = `${f.id} ${f.task_id} ${f.chain_id} ${f.company} ${f.pattern} ${f.source} ${pretty(r.what_adversary_saw)} ${pretty(r.outputs)}`.toLowerCase();
        return (!score || String(f.score) === score) && matchLeak(f, leak) && (!q || text.includes(q));
      });
      filtered.sort((a,b) => {
        const fa=a.feature, fb=b.feature;
        if(sort==='score_desc') return fb.score-fa.score || fb.intent_score-fa.intent_score;
        if(sort==='intent_desc') return fb.intent_score-fa.intent_score || fb.score-fa.score;
        if(sort==='web_desc') return fb.n_web_queries-fa.n_web_queries;
        if(sort==='pqa_desc') return fb.n_private_qas-fa.n_private_qas;
        return String(fa.id).localeCompare(String(fb.id));
      });
      $('count').textContent = `${filtered.length} / ${rows.length} rows`;
      $('results').innerHTML = filtered.map(card).join('') || '<div class="panel">No rows match the filters.</div>';
    }
    function renderTask(){
      const body = byTask.map(r => `<tr><td>${escapeHtml(r.task_id)}</td><td>${r.n}</td><td>${r.mean_score}</td><td>${r.intent_leaked}</td><td>${r.answer_leaked}</td><td>${r.full_info_leaked}</td><td>${r.reward_positive}</td></tr>`).join('');
      $('task-table').innerHTML = `<table><thead><tr><th>task</th><th>n</th><th>mean score</th><th>intent</th><th>answer</th><th>full-info</th><th>reward+</th></tr></thead><tbody>${body}</tbody></table>`;
    }
    ['q','score','leak','sort'].forEach(id => $(id).addEventListener('input', render));
    renderTask(); render();
    """
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(args.title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{esc(args.title)}</h1>
      <p>Generated {esc(datetime.now(timezone.utc).isoformat())}</p>
      <p>Input: <code>{esc(in_path)}</code></p>
      <p>This viewer shows what the adversary model saw, what each adversary produced, what each judge produced, and the hidden references available only to judges.</p>
    </section>
    <section class="cards">{summary_cards(stats)}</section>
    <section class="two">
      {table_from_counter("Overall Score Counts", stats["score_counts"])}
      {table_from_counter("Primary Source Counts", stats["source_counts"])}
    </section>
    <section class="panel">
      <h2>Task Summary</h2>
      <div id="task-table"></div>
    </section>
    <section class="panel">
      <h2>Rows</h2>
      <div class="controls">
        <input id="q" placeholder="Search ids, company, queries, outputs">
        <select id="score"><option value="">All scores</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
        <select id="leak">
          <option value="all">All leakage</option>
          <option value="reward">Reward positive</option>
          <option value="intent">Intent leaked</option>
          <option value="answer">Answer leaked</option>
          <option value="full">Full-info leaked</option>
          <option value="suspicious">Suspicious full-info</option>
          <option value="zero_pqa">Zero Private QA</option>
        </select>
        <select id="sort">
          <option value="score_desc">Sort score desc</option>
          <option value="intent_desc">Sort intent desc</option>
          <option value="web_desc">Sort web queries desc</option>
          <option value="pqa_desc">Sort Private QA desc</option>
          <option value="id">Sort id</option>
        </select>
        <div class="card"><div class="label">Showing</div><div id="count" class="value">0</div></div>
      </div>
    </section>
    <section id="results" class="results"></section>
  </div>
  <script type="application/json" id="viewer-data">{data_json}</script>
  <script type="application/json" id="stats-data">{stats_json}</script>
  <script type="application/json" id="task-data">{task_json}</script>
  <script>{js}</script>
</body>
</html>
"""
    (out / "index.html").write_text(html_doc, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(in_path),
        "output": str(out / "index.html"),
        "rows": len(rows),
        "summary": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
