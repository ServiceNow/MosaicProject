#!/usr/bin/env python3
"""Build a readable HTML audit viewer for Privacy HopQA leakage v2 outputs."""

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
    "model_runs/qwen3_4b_instruct_2507_base_dp4_temp07_r3/tables/privacy_leakage_suite_v2_stepfun_mid50_rv10.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_leakage_v2"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_path_key(path: Any) -> str:
    return str(path or "")


def discover_rollout_index(input_path: Path) -> Path | None:
    candidate = input_path.parent / "rollout_index.jsonl"
    if candidate.exists():
        return candidate
    return None


def load_rollout_metrics(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        trace_path = compact_path_key(row.get("trace_path"))
        if trace_path:
            out[trace_path] = row
    return out


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


def prompt_pair(row: dict[str, Any], key: str) -> dict[str, str]:
    prompts = row.get("prompts") if isinstance(row.get("prompts"), dict) else {}
    item = prompts.get(key) if isinstance(prompts.get(key), dict) else {}
    return {"system": str(item.get("system") or ""), "user": str(item.get("user") or "")}


def web_query_texts(row: dict[str, Any]) -> list[str]:
    model_input = row.get("model_input") if isinstance(row.get("model_input"), dict) else {}
    queries = model_input.get("web_queries") if isinstance(model_input.get("web_queries"), list) else []
    out = []
    for item in queries:
        if isinstance(item, dict):
            text = str(item.get("query") or "").strip()
            if text:
                out.append(text)
        elif str(item).strip():
            out.append(str(item).strip())
    return out


def private_questions(row: dict[str, Any], include_answers: bool = False) -> list[dict[str, Any]]:
    ref = row.get("hidden_reference_for_judges_only") if isinstance(row.get("hidden_reference_for_judges_only"), dict) else {}
    out: list[dict[str, Any]] = []
    for item in ref.get("private_qas") or []:
        if not isinstance(item, dict):
            continue
        q = {
            "qa_id": item.get("qa_id"),
            "question": item.get("question"),
            "secret_type": item.get("secret_type"),
            "doc_key": item.get("doc_key"),
        }
        if include_answers:
            q["answer"] = item.get("canonical_answer")
            q["accepted_answers"] = item.get("accepted_answers")
        out.append(q)
    return out


def local_facts(row: dict[str, Any]) -> list[dict[str, Any]]:
    ref = row.get("hidden_reference_for_judges_only") if isinstance(row.get("hidden_reference_for_judges_only"), dict) else {}
    return [item for item in (ref.get("local_hop_facts") or []) if isinstance(item, dict)]


def true_subquestions(row: dict[str, Any]) -> list[dict[str, Any]]:
    ref = row.get("hidden_reference_for_judges_only") if isinstance(row.get("hidden_reference_for_judges_only"), dict) else {}
    return [item for item in (ref.get("true_subquestions") or []) if isinstance(item, dict)]


def retrieved_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    ref = row.get("hidden_reference_for_judges_only") if isinstance(row.get("hidden_reference_for_judges_only"), dict) else {}
    return [item for item in (ref.get("retrieved_local_docs") or []) if isinstance(item, dict)]


def claim_texts(row: dict[str, Any]) -> list[str]:
    output = row.get("claim_adversary_output") if isinstance(row.get("claim_adversary_output"), dict) else {}
    claims = output.get("claims") if isinstance(output.get("claims"), list) else []
    out: list[str] = []
    for item in claims:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict) and item.get("claim"):
            out.append(str(item.get("claim")).strip())
    return out


def answer_attempts(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = row.get("answer_adversary_output") if isinstance(row.get("answer_adversary_output"), dict) else {}
    answers = output.get("answers") if isinstance(output.get("answers"), list) else []
    return [item for item in answers if isinstance(item, dict)]


def row_feature(row: dict[str, Any], rollout_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    intent = row.get("intent_judge_output") if isinstance(row.get("intent_judge_output"), dict) else {}
    answer = row.get("answer_judge_output") if isinstance(row.get("answer_judge_output"), dict) else {}
    claim = row.get("claim_judge_output") if isinstance(row.get("claim_judge_output"), dict) else {}
    metrics = (rollout_metrics or {}).get("metrics") if isinstance((rollout_metrics or {}).get("metrics"), dict) else {}
    claims = claim_texts(row)
    intent_score = int(intent.get("intent_subquestion_score_1_to_5") or 0)
    return {
        "rollout_id": row.get("rollout_id"),
        "task_id": row.get("task_id"),
        "chain_id": row.get("chain_id"),
        "company": row.get("company"),
        "pattern": row.get("pattern"),
        "score": int(row.get("overall_privacy_score_1_to_5") or 0),
        "source": row.get("overall_primary_source"),
        "reward_label": int(bool(row.get("overall_leaked_for_reward"))),
        "intent_score": intent_score,
        "intent_leaked": int(intent_score >= 4),
        "answer_leaked": int(bool(answer.get("answer_leaked"))),
        "full_info_leaked": int(bool(claim.get("full_info_leaked"))),
        "n_claims": len(claims),
        "n_answer_attempts": len(answer_attempts(row)),
        "n_private_qas": int(row.get("n_private_qas") or 0),
        "n_web_queries": int(row.get("n_web_queries_visible") or 0),
        "n_missing_private_qa_docs": int(row.get("n_missing_private_qa_docs") or 0),
        "n_hops": int(row.get("n_hops") or 0),
        "trace_path": row.get("trace_path"),
        "suspicious_full_info_without_claims": int(bool(claim.get("full_info_leaked")) and not claims),
        "hop_accuracy": metrics.get("hop_accuracy", rollout_metrics.get("hop_accuracy") if rollout_metrics else None),
        "final_correct": metrics.get("final_correct", rollout_metrics.get("final_correct") if rollout_metrics else None),
        "chain_complete": metrics.get("chain_complete", rollout_metrics.get("chain_complete") if rollout_metrics else None),
        "correct_hops": metrics.get("correct_hops"),
        "total_hops": metrics.get("total_hops", row.get("n_hops")),
        "hops_resolved": metrics.get("hops_resolved"),
    }


def viewer_row(row: dict[str, Any], rollout_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "feature": row_feature(row, rollout_metrics),
        "web_queries": web_query_texts(row),
        "private_questions_no_answers": private_questions(row, include_answers=False),
        "private_questions_with_answers": private_questions(row, include_answers=True),
        "true_subquestions": true_subquestions(row),
        "local_facts": local_facts(row),
        "retrieved_docs": retrieved_docs(row),
        "missing_private_qa_doc_keys": (row.get("hidden_reference_for_judges_only") or {}).get("missing_private_qa_doc_keys") or [],
        "prompts": {
            name: prompt_pair(row, name)
            for name in (
                "intent_adversary",
                "intent_judge",
                "answer_adversary",
                "answer_judge",
                "claim_adversary",
                "claim_judge",
            )
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
    }


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "0.0%"


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(features)
    accuracy_values = [float(item["hop_accuracy"]) for item in features if item.get("hop_accuracy") is not None]
    return {
        "rows": n,
        "score_counts": dict(sorted(Counter(item["score"] for item in features).items())),
        "source_counts": dict(sorted(Counter(str(item["source"]) for item in features).items())),
        "reward_positive": sum(item["reward_label"] for item in features),
        "intent_leaked": sum(item["intent_leaked"] for item in features),
        "answer_leaked": sum(item["answer_leaked"] for item in features),
        "full_info_leaked": sum(item["full_info_leaked"] for item in features),
        "suspicious_full_info_without_claims": sum(item["suspicious_full_info_without_claims"] for item in features),
        "mean_private_qas": round(mean_or_zero([item["n_private_qas"] for item in features]), 2),
        "zero_private_qas": sum(1 for item in features if item["n_private_qas"] == 0),
        "mean_web_queries": round(mean_or_zero([item["n_web_queries"] for item in features]), 2),
        "mean_hop_accuracy": round(mean_or_zero(accuracy_values), 3) if accuracy_values else None,
        "final_correct": sum(1 for item in features if item.get("final_correct") is True),
        "chain_complete": sum(1 for item in features if item.get("chain_complete") is True),
    }


def summary_cards(stats: dict[str, Any]) -> str:
    rows = int(stats["rows"])
    cards = [
        ("Rows", rows, "successful evaluations in this file"),
        ("Reward positives", f"{stats['reward_positive']} ({pct(stats['reward_positive'], rows)})", "overall score >= 4"),
        ("Intent leaked", f"{stats['intent_leaked']} ({pct(stats['intent_leaked'], rows)})", "intent score >= 4"),
        ("Answer leaked", f"{stats['answer_leaked']} ({pct(stats['answer_leaked'], rows)})", "Private QA proposition judged correct"),
        ("Full-info leaked", f"{stats['full_info_leaked']} ({pct(stats['full_info_leaked'], rows)})", "true private claim"),
        ("Mean hop accuracy", "n/a" if stats["mean_hop_accuracy"] is None else stats["mean_hop_accuracy"], "from rollout_index when available"),
        ("Final correct", f"{stats['final_correct']} ({pct(stats['final_correct'], rows)})", "rollout final answer"),
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
    parser.add_argument("--title", default="Privacy HopQA Leakage v2 Viewer")
    parser.add_argument("--rollout-index", default=None, help="Optional rollout_index.jsonl with accuracy metrics.")
    parser.add_argument(
        "--write-viewer-data-json",
        action="store_true",
        help="Also write the full viewer data payload next to index.html. The HTML embeds it already.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    rollout_index_path = Path(args.rollout_index) if args.rollout_index else discover_rollout_index(in_path)
    rollout_metrics = load_rollout_metrics(rollout_index_path)
    rows = [row for row in read_jsonl(in_path) if row.get("ok")]
    view_rows = [viewer_row(row, rollout_metrics.get(compact_path_key(row.get("trace_path")))) for row in rows]
    features = [row["feature"] for row in view_rows]
    stats = summary(features)

    by_chain: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for feature in features:
        by_chain[(str(feature["task_id"]), str(feature["chain_id"]))].append(feature)
    chain_rows = []
    for (task_id, chain_id), vals in sorted(by_chain.items()):
        acc = [float(v["hop_accuracy"]) for v in vals if v.get("hop_accuracy") is not None]
        chain_rows.append(
            {
                "task_id": task_id,
                "chain_id": chain_id,
                "n": len(vals),
                "mean_leakage_score": round(mean_or_zero([v["score"] for v in vals]), 2),
                "mean_hop_accuracy": None if not acc else round(mean_or_zero(acc), 3),
                "intent_leaked": sum(v["intent_leaked"] for v in vals),
                "answer_leaked": sum(v["answer_leaked"] for v in vals),
                "full_info_leaked": sum(v["full_info_leaked"] for v in vals),
                "reward_positive": sum(v["reward_label"] for v in vals),
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "features.jsonl", features)
    if args.write_viewer_data_json:
        (out / "viewer_data.json").write_text(json.dumps(view_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"summary": stats, "by_chain": chain_rows, "rollout_index": str(rollout_index_path or "")}, indent=2),
        encoding="utf-8",
    )

    data_json = script_json(view_rows)
    stats_json = script_json(stats)
    chain_json = script_json(chain_rows)
    css = """
    :root{--bg:#f6f3ec;--panel:#fffdf8;--ink:#202124;--muted:#65706b;--line:#d9d2c5;--green:#0f6b5f;--gold:#b67812;--red:#9f2f22;--blue:#345995}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}.page{max-width:1600px;margin:auto;padding:24px}
    h1{font-size:32px;margin:0 0 8px}.hero,.panel,.row-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.05)}.hero p{color:var(--muted);margin:4px 0}.hero code{font-size:12px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}.card{background:#f1eee7;border:1px solid var(--line);border-radius:8px;padding:12px}.label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.value{font-size:24px;font-weight:750}.note{color:var(--muted);font-size:12px}
    .controls{display:grid;grid-template-columns:2fr repeat(4,minmax(135px,1fr));gap:10px;margin:16px 0}.controls input,.controls select{width:100%;padding:9px;border:1px solid var(--line);border-radius:8px;background:white}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.results{display:grid;gap:12px;margin-top:12px}.row-card.score-5{border-left:7px solid var(--red)}.row-card.score-4{border-left:7px solid var(--gold)}.row-card.score-3{border-left:7px solid var(--green)}.row-card.score-2,.row-card.score-1{border-left:7px solid #9aa6a1}
    .row-head{display:flex;gap:10px;align-items:flex-start;justify-content:space-between}.badges{display:flex;gap:6px;flex-wrap:wrap}.badge{border-radius:999px;padding:3px 8px;font-size:12px;background:#e7e2d8;color:#344}.badge.red{background:#f4d7d1}.badge.gold{background:#f4e5c6}.badge.green{background:#d9ece5}.badge.blue{background:#dce5f5}
    .tabs{margin-top:12px}.tabbar{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:8px}.tabbar button{border:1px solid var(--line);background:#f4f1ea;border-radius:7px;padding:7px 10px;cursor:pointer}.tabbar button.active{background:var(--blue);color:white;border-color:var(--blue)}.tab{display:none;padding-top:12px}.tab.active{display:block}
    .prompt{white-space:pre-wrap;background:#f8f4eb;border:1px solid var(--line);border-radius:8px;padding:12px;line-height:1.5}.prompt h4{margin:.2rem 0 .4rem}.query-list li,.claim-list li,.subq-list li{margin:.35rem 0}.empty{color:var(--muted);font-style:italic}.small{color:var(--muted);font-size:12px}.warn{color:var(--red);font-weight:700}
    details{margin-top:8px;border-top:1px solid var(--line);padding-top:8px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;word-break:break-word;background:#1f2523;color:#f7f3e8;border-radius:8px;padding:10px;max-height:420px;overflow:auto;font-size:12px}
    table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}th{font-size:12px;color:#344}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-top:10px}.metric{background:#f4f1ea;border:1px solid var(--line);border-radius:8px;padding:8px}.metric b{display:block;font-size:18px}
    @media(max-width:900px){.controls,.two{grid-template-columns:1fr}.row-head{display:block}}
    """
    js = """
    const rows = JSON.parse(document.getElementById('viewer-data').textContent);
    const stats = JSON.parse(document.getElementById('stats-data').textContent);
    const byChain = JSON.parse(document.getElementById('chain-data').textContent);
    const $ = id => document.getElementById(id);
    const pretty = obj => JSON.stringify(obj, null, 2);
    function escapeHtml(s){return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
    function pre(obj){return `<pre>${escapeHtml(typeof obj === 'string' ? obj : pretty(obj))}</pre>`;}
    function fmt(v){return v === null || v === undefined ? 'n/a' : escapeHtml(v);}
    function badges(f){
      const out = [`score ${f.score}`, `intent ${f.intent_score}`, f.source || 'none', `${f.n_web_queries} web q`, `${f.n_private_qas} PQA`];
      if(f.intent_leaked) out.push('intent leaked');
      if(f.answer_leaked) out.push('answer leaked');
      if(f.full_info_leaked) out.push('full-info leaked');
      if(f.suspicious_full_info_without_claims) out.push('suspicious');
      return out.map(x => `<span class="badge ${x.includes('full')||x.includes('suspicious')?'red':x.includes('intent leaked')?'gold':x.includes('PQA')?'blue':'green'}">${escapeHtml(x)}</span>`).join('');
    }
    function table(headers, rows){
      if(!rows || !rows.length) return '<div class="empty">None.</div>';
      return `<table><thead><tr>${headers.map(h=>`<th>${escapeHtml(h[1])}</th>`).join('')}</tr></thead><tbody>` +
        rows.map(r => `<tr>${headers.map(h=>`<td>${escapeHtml(r[h[0]])}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
    }
    function list(items, cls=''){
      if(!items || !items.length) return '<div class="empty">None.</div>';
      return `<ol class="${cls}">${items.map(x=>`<li>${escapeHtml(x)}</li>`).join('')}</ol>`;
    }
    function promptBlock(pair){
      if(!pair || (!pair.system && !pair.user)) return '<div class="empty">This model call was skipped.</div>';
      return `<div class="prompt"><h4>System</h4>${escapeHtml(pair.system || '')}<h4>User</h4>${escapeHtml(pair.user || '')}</div>`;
    }
    function promptOutputTab(title, promptPair, outputHtml, meta){
      return `<section><h3>${escapeHtml(title)}</h3><h4>Exact rendered prompt</h4>${promptBlock(promptPair)}<h4>Parsed model output</h4>${outputHtml}<details><summary>StepFun metadata</summary>${pre(meta || {})}</details></section>`;
    }
    function intentOutput(row){
      const preds = (row.outputs.intent_adversary && row.outputs.intent_adversary.predicted_subquestions) || [];
      const judge = row.outputs.intent_judge || {};
      return `<h5>Predicted subquestions</h5>${list(preds, 'subq-list')}<h5>Judge</h5>` +
        table([['intent_subquestion_score_1_to_5','score'],['score_label','label'],['n_fully_predicted_subquestions','fully predicted'],['n_partially_predicted_subquestions','partially predicted'],['intent_leaked','leaked']], [judge]);
    }
    function answerOutput(row){
      const answers = ((row.outputs.answer_adversary || {}).answers || []);
      const judgments = ((row.outputs.answer_judge || {}).answer_judgments || []);
      const judge = row.outputs.answer_judge || {};
      return `<h5>Answer attempts</h5>${table([['qa_id','qa_id'],['answer','answer'],['confidence','confidence']], answers)}<h5>Judge summary</h5>` +
        table([['answer_leaked','leaked'],['n_answer_attempts','attempts'],['n_correct','correct'],['correct_qa_ids','correct ids']], [judge]) +
        `<h5>Per-answer judgments</h5>${table([['qa_id','qa_id'],['judgment','judgment'],['matched_answer_variant','matched variant']], judgments)}`;
    }
    function claimOutput(row){
      const claims = (row.outputs.claim_adversary && row.outputs.claim_adversary.claims) || [];
      const judgments = ((row.outputs.claim_judge || {}).claim_judgments || []);
      const judge = row.outputs.claim_judge || {};
      return `<h5>Generated claims</h5>${list(claims, 'claim-list')}<h5>Judge summary</h5>` +
        table([['full_info_leaked','leaked'],['n_true_private_claims','true private claims'],['matched_fact_ids','matched ids']], [judge]) +
        `<h5>Per-claim judgments</h5>${table([['claim_index','claim #'],['label','label'],['matched_fact_id','matched fact']], judgments)}`;
    }
    function visibleInputs(row){
      return `<h3>Visible to adversary</h3><p class="small">The adversary sees only these web-query strings for intent and claim prediction.</p>` +
        `<h4>Web queries</h4>${list(row.web_queries, 'query-list')}` +
        `<h4>Private questions shown only to answer adversary</h4>` +
        table([['qa_id','qa_id'],['question','question'],['secret_type','type']], row.private_questions_no_answers);
    }
    function hiddenRefs(row){
      return `<h3>Judge-only hidden references</h3>` +
        `<h4>True subquestions</h4>${table([['hop_number','hop'],['hop_type','type'],['question','question']], row.true_subquestions)}` +
        `<h4>Local-hop facts</h4>${table([['fact_id','fact'],['hop_number','hop'],['question','question'],['canonical_answer','answer'],['doc_id','doc']], row.local_facts)}` +
        `<h4>Private QA references</h4>${table([['qa_id','qa_id'],['question','question'],['answer','answer'],['secret_type','type'],['doc_key','doc']], row.private_questions_with_answers)}` +
        `<h4>Retrieved local documents</h4>${table([['hop_number','hop'],['title','title'],['doc_key','doc key'],['source_kind','source']], row.retrieved_docs)}` +
        `<h4>Missing Private QA doc keys</h4>${list(row.missing_private_qa_doc_keys)}`;
    }
    function metricsGrid(f){
      const metrics = [
        ['leak score', f.score], ['intent score', f.intent_score], ['hop accuracy', f.hop_accuracy], ['final correct', f.final_correct],
        ['chain complete', f.chain_complete], ['correct hops', f.correct_hops], ['resolved hops', f.hops_resolved], ['total hops', f.total_hops]
      ];
      return `<div class="metric-grid">${metrics.map(([k,v])=>`<div class="metric"><span class="label">${escapeHtml(k)}</span><b>${fmt(v)}</b></div>`).join('')}</div>`;
    }
    function tabs(row, idx){
      const tabs = [
        ['visible','Visible inputs', visibleInputs(row)],
        ['intent','Intent prompt/output', promptOutputTab('Intent adversary and judge', row.prompts.intent_adversary, intentOutput(row), row.stepfun_meta.intent_adversary) + '<h4>Intent judge prompt</h4>' + promptBlock(row.prompts.intent_judge)],
        ['answer','Answer prompt/output', promptOutputTab('Answer adversary and judge', row.prompts.answer_adversary, answerOutput(row), row.stepfun_meta.answer_adversary) + '<h4>Answer judge prompt</h4>' + promptBlock(row.prompts.answer_judge)],
        ['claim','Claim prompt/output', promptOutputTab('Claim adversary and judge', row.prompts.claim_adversary, claimOutput(row), row.stepfun_meta.claim_adversary) + '<h4>Claim judge prompt</h4>' + promptBlock(row.prompts.claim_judge)],
        ['hidden','Judge-only references', hiddenRefs(row)],
      ];
      return `<div class="tabs" data-tabs="${idx}"><div class="tabbar">${tabs.map((t,i)=>`<button class="${i===0?'active':''}" data-tab="${t[0]}">${escapeHtml(t[1])}</button>`).join('')}</div>` +
        tabs.map((t,i)=>`<div class="tab ${i===0?'active':''}" data-tab="${t[0]}">${t[2]}</div>`).join('') + '</div>';
    }
    function card(row, idx){
      const f = row.feature;
      return `<article class="row-card score-${f.score}">
        <div class="row-head">
          <div><h3>Task ${escapeHtml(f.task_id)} · chain ${escapeHtml(f.chain_id)} · rollout ${escapeHtml(f.rollout_id)}</h3>
          <div class="small">${escapeHtml(f.company)} · pattern ${escapeHtml(f.pattern)} · ${escapeHtml(f.trace_path)}</div></div>
          <div class="badges">${badges(f)}</div>
        </div>
        ${metricsGrid(f)}
        ${tabs(row, idx)}
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
        const text = `${f.rollout_id} ${f.task_id} ${f.chain_id} ${f.company} ${f.pattern} ${f.source} ${r.web_queries.join(' ')} ${pretty(r.outputs)}`.toLowerCase();
        return (!score || String(f.score) === score) && matchLeak(f, leak) && (!q || text.includes(q));
      });
      filtered.sort((a,b) => {
        const fa=a.feature, fb=b.feature;
        if(sort==='score_desc') return fb.score-fa.score || fb.intent_score-fa.intent_score;
        if(sort==='intent_desc') return fb.intent_score-fa.intent_score || fb.score-fa.score;
        if(sort==='accuracy_desc') return (fb.hop_accuracy ?? -1)-(fa.hop_accuracy ?? -1);
        if(sort==='web_desc') return fb.n_web_queries-fa.n_web_queries;
        if(sort==='pqa_desc') return fb.n_private_qas-fa.n_private_qas;
        return String(fa.rollout_id).localeCompare(String(fb.rollout_id));
      });
      $('count').textContent = `${filtered.length} / ${rows.length} rows`;
      $('results').innerHTML = filtered.map(card).join('') || '<div class="panel">No rows match the filters.</div>';
    }
    function renderChain(){
      const body = byChain.map(r => `<tr><td>${escapeHtml(r.task_id)}</td><td>${escapeHtml(r.chain_id)}</td><td>${r.n}</td><td>${r.mean_leakage_score}</td><td>${fmt(r.mean_hop_accuracy)}</td><td>${r.intent_leaked}</td><td>${r.answer_leaked}</td><td>${r.full_info_leaked}</td><td>${r.reward_positive}</td></tr>`).join('');
      $('chain-table').innerHTML = `<table><thead><tr><th>task</th><th>chain</th><th>n</th><th>mean leakage score</th><th>mean hop accuracy</th><th>intent</th><th>answer</th><th>full-info</th><th>reward+</th></tr></thead><tbody>${body}</tbody></table>`;
    }
    document.addEventListener('click', (ev) => {
      const btn = ev.target.closest('.tabbar button');
      if(!btn) return;
      const root = btn.closest('.tabs');
      root.querySelectorAll('.tabbar button').forEach(x => x.classList.toggle('active', x === btn));
      root.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.dataset.tab === btn.dataset.tab));
    });
    ['q','score','leak','sort'].forEach(id => $(id).addEventListener('input', render));
    renderChain(); render();
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
      <p>Rollout metrics: <code>{esc(rollout_index_path or 'not found')}</code></p>
      <p>Each row shows only the web-query strings visible to the adversary. Prompt tabs show the exact rendered system/user prompts sent for each call.</p>
    </section>
    <section class="cards">{summary_cards(stats)}</section>
    <section class="two">
      {table_from_counter("Overall Score Counts", stats["score_counts"])}
      {table_from_counter("Primary Source Counts", stats["source_counts"])}
    </section>
    <section class="panel">
      <h2>Chain Summary</h2>
      <p class="small">Leakage score is privacy risk, not task accuracy. Hop accuracy and final correctness come from rollout_index when available.</p>
      <div id="chain-table"></div>
    </section>
    <section class="panel">
      <h2>Rows</h2>
      <div class="controls">
        <input id="q" placeholder="Search ids, company, query text, outputs">
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
          <option value="score_desc">Sort leakage score desc</option>
          <option value="intent_desc">Sort intent desc</option>
          <option value="accuracy_desc">Sort accuracy desc</option>
          <option value="web_desc">Sort web queries desc</option>
          <option value="pqa_desc">Sort Private QA desc</option>
          <option value="id">Sort rollout id</option>
        </select>
        <div class="card"><div class="label">Showing</div><div id="count" class="value">0</div></div>
      </div>
    </section>
    <section id="results" class="results"></section>
  </div>
  <script type="application/json" id="viewer-data">{data_json}</script>
  <script type="application/json" id="stats-data">{stats_json}</script>
  <script type="application/json" id="chain-data">{chain_json}</script>
  <script>{js}</script>
</body>
</html>
"""
    (out / "index.html").write_text(html_doc, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(in_path),
        "rollout_index": str(rollout_index_path or ""),
        "output": str(out / "index.html"),
        "rows": len(rows),
        "summary": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
