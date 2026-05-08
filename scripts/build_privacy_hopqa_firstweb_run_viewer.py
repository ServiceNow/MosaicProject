#!/usr/bin/env python3
"""Build a compact static viewer for a live Privacy HopQA first-web training run."""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_RUN_DIR = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_firstweb_qwen3_4b2507_r2_lowconcurrency_20260507202420"
)
DEFAULT_DATASET = Path(
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/"
    "privacy_hopqa_first_web_hop_2026-05-07.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/toolkit/my_code/MosaicProject/runtime_analysis/"
    "privacy_hopqa_firstweb_qwen3_4b2507_live_viewer.html"
)

COMPLETED_RE = re.compile(r"^(?:\[finetune\]:\s*)?(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\.\d+) .*Completed steps (\d+): (\{.*\})")
PROCESSED_RE = re.compile(r"Processed (\d+) samples \(filtered out (\d+)\) in ([\d.]+)s .* total (\d+) samples")
PUBLISHED_RE = re.compile(r"Published (\d+) train samples .* total (\d+) samples")
LLM_SUMMARY_RE = re.compile(r"privacy_hopqa llm call summary: (\{.*\})")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-scan", type=int, default=12000, help="Max rollout directories to inspect")
    parser.add_argument("--max-examples", type=int, default=180, help="Max sampled rollout records embedded")
    parser.add_argument("--log-tail-lines", type=int, default=50000, help="Recent actor lines used for LLM-call summaries")
    return parser.parse_args()


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def trunc(value: Any, n: int = 900) -> str:
    text = "" if value is None else str(value)
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_answer(text: Any) -> str:
    value = "" if text is None else str(text).strip().lower()
    value = re.sub(r"[\s\u00a0]+", " ", value)
    value = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", value)
    value = re.sub(r"[.,;:!?]+$", "", value)
    return value


def read_tail_lines(path: Path, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    # Read from the end to avoid pulling multi-GB logs into memory.
    block_size = 1024 * 1024
    chunks: list[bytes] = []
    lines_seen = 0
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        pos = handle.tell()
        while pos > 0 and lines_seen <= max_lines:
            read_size = min(block_size, pos)
            pos -= read_size
            handle.seek(pos)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            lines_seen += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-max_lines:]


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chain_id = str(row.get("chain_id") or "")
            if chain_id:
                out[chain_id] = row
    return out


def parse_finetune_log(path: Path) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    empty_after_latest = 0
    latest_line_no = -1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    for idx, line in enumerate(lines):
        match = COMPLETED_RE.search(line)
        if not match:
            continue
        ts, step_text, payload_text = match.groups()
        try:
            payload = ast.literal_eval(payload_text)
        except Exception:
            continue
        row = {"step": int(step_text), "timestamp": ts}
        row.update(payload)
        steps.append(row)
        latest_line_no = idx
    if latest_line_no >= 0:
        empty_after_latest = sum("Batch queue is empty" in line for line in lines[latest_line_no + 1 :])
    return {"steps": steps, "empty_after_latest_step": empty_after_latest}


def parse_preprocess_log(path: Path) -> dict[str, Any]:
    processed: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PROCESSED_RE.search(line)
            if not match:
                continue
            kept, filtered, seconds, total = match.groups()
            kept_i = int(kept)
            filtered_i = int(filtered)
            processed.append(
                {
                    "kept": kept_i,
                    "filtered": filtered_i,
                    "seconds": float(seconds),
                    "total": int(total),
                    "filter_rate": filtered_i / max(1, kept_i + filtered_i),
                }
            )
    return {"processed_batches": processed}


def parse_actor_log(path: Path, max_lines: int) -> dict[str, Any]:
    lines = read_tail_lines(path, max_lines)
    published: list[dict[str, int]] = []
    llm_calls: list[dict[str, Any]] = []
    for line in lines:
        pub = PUBLISHED_RE.search(line)
        if pub:
            count, total = pub.groups()
            published.append({"count": int(count), "total": int(total)})
        match = LLM_SUMMARY_RE.search(line)
        if not match:
            continue
        try:
            llm_calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    hop_plan_calls = [call for call in llm_calls if call.get("log_name") == "hop_plan"]
    two_token_plans = [call for call in hop_plan_calls if safe_float(call.get("output_tokens")) == 2]
    by_type: dict[str, int] = {}
    for call in llm_calls:
        name = str(call.get("log_name") or "unknown")
        by_type[name] = by_type.get(name, 0) + 1
    return {
        "published": published,
        "llm_call_count_recent": len(llm_calls),
        "llm_call_by_type_recent": by_type,
        "hop_plan_calls_recent": len(hop_plan_calls),
        "two_token_hop_plans_recent": len(two_token_plans),
        "two_token_hop_plan_rate_recent": len(two_token_plans) / max(1, len(hop_plan_calls)),
    }


def parse_warning_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    retrieval_failures = [line for line in lines if "retrieval action" in line and "failed" in line]
    return {
        "warning_lines": len(lines),
        "retrieval_failure_count": len(retrieval_failures),
        "recent_retrieval_failures": retrieval_failures[-20:],
    }


def variants_for(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    variants: list[str] = []
    for key in ("answer", "global_answer", "global_answer_variants", "global_alternate_valid_answers"):
        value = row.get(key)
        if isinstance(value, list):
            variants.extend(str(item) for item in value)
        elif value not in (None, ""):
            variants.append(str(value))
    for hop in row.get("hops") or []:
        for key in ("answer", "accepted_answer_variants", "alternate_valid_answers"):
            value = hop.get(key)
            if isinstance(value, list):
                variants.extend(str(item) for item in value)
            elif value not in (None, ""):
                variants.append(str(value))
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        norm = normalize_answer(item)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(item)
    return out


def summarize_trace(path: Path, dataset: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    summary = data.get("summary") or {}
    chain_id = str(summary.get("chain_id") or "")
    row = dataset.get(chain_id)
    hop_states = data.get("hop_states") or []
    hop = hop_states[0] if hop_states and isinstance(hop_states[0], dict) else {}
    actions = data.get("actions") or []
    error_records = data.get("error_records") or []
    accepted = variants_for(row)
    predicted = str(hop.get("answer") or "")
    correct = bool(predicted) and normalize_answer(predicted) in {normalize_answer(v) for v in accepted}
    failed_actions = [action for action in actions if action.get("status") == "failed"]
    completed_actions = [action for action in actions if action.get("status") == "completed"]
    web_actions = [action for action in actions if action.get("type") == "web_search"]
    local_actions = [action for action in actions if action.get("type") == "local_document_search"]
    reader_results = hop.get("reader_results") or []
    can_answer_reads = [result for result in reader_results if result.get("can_answer")]
    search_history = hop.get("search_history") or []
    flags: list[str] = []
    if not correct:
        flags.append("answer_mismatch")
    if summary.get("stalled"):
        flags.append("stalled")
    if (safe_float(summary.get("resolved_hops")) or 0) < (safe_float(summary.get("total_hops")) or 1):
        flags.append("unresolved")
    if safe_float(summary.get("total_errors")):
        flags.append("errors")
    if safe_float(summary.get("parse_errors_total")):
        flags.append("parse_errors")
    if failed_actions:
        flags.append("failed_retrieval")
    if not actions or not search_history:
        flags.append("no_search_history")
    if (safe_float(summary.get("iterations_used")) or 0) >= 3:
        flags.append("max_iterations")
    if (safe_float(summary.get("docs_read")) or 0) >= 10:
        flags.append("many_docs_read")
    record = {
        "rollout_id": path.parent.name,
        "trace_path": str(path),
        "mtime": path.stat().st_mtime,
        "task_id": summary.get("task_id"),
        "chain_id": chain_id,
        "company": row.get("company") if row else None,
        "question": row.get("question") if row else hop.get("question"),
        "gold_answer": row.get("answer") if row else None,
        "accepted_answers": accepted[:20],
        "predicted_answer": predicted,
        "correct": correct,
        "calculated_reward": 1.0 if correct else 0.0,
        "status": hop.get("status"),
        "confidence": hop.get("confidence"),
        "reason": trunc(hop.get("resolution_reason"), 700),
        "justification": trunc(hop.get("justification"), 900),
        "duration_s": summary.get("duration_s"),
        "iterations_used": summary.get("iterations_used"),
        "searches_total": summary.get("searches_total"),
        "docs_read": summary.get("docs_read"),
        "llm_calls_total": summary.get("llm_calls_total"),
        "total_tokens": summary.get("total_tokens"),
        "total_errors": summary.get("total_errors"),
        "parse_errors_total": summary.get("parse_errors_total"),
        "actions_total": len(actions),
        "completed_actions": len(completed_actions),
        "failed_actions": len(failed_actions),
        "web_actions": len(web_actions),
        "local_actions": len(local_actions),
        "reader_results": len(reader_results),
        "can_answer_reads": len(can_answer_reads),
        "flags": flags,
        "search_history": [
            {
                "iteration": item.get("iteration"),
                "type": item.get("type"),
                "query": item.get("query"),
                "result_count": item.get("result_count"),
                "top_doc_ids": item.get("top_doc_ids") or [],
                "error": item.get("error"),
            }
            for item in search_history[:12]
        ],
        "actions": [
            {
                "id": action.get("id"),
                "type": action.get("type"),
                "status": action.get("status"),
                "query": (action.get("parameters") or {}).get("query"),
                "results_count": ((action.get("actual_output") or {}).get("results_count")),
                "execution_time": action.get("execution_time"),
                "error": action.get("error"),
            }
            for action in actions[:12]
        ],
        "reader_sample": [
            {
                "doc_id": result.get("doc_id"),
                "source": result.get("source"),
                "title": trunc(result.get("title"), 160),
                "can_answer": result.get("can_answer"),
                "proposed_answer": result.get("proposed_answer"),
                "confidence": result.get("confidence"),
                "justification": trunc(result.get("justification"), 600),
            }
            for result in reader_results[:8]
        ],
        "error_records": error_records[:5],
    }
    return record


def sample_trace_records(run_dir: Path, dataset: dict[str, dict[str, Any]], max_scan: int, max_examples: int) -> dict[str, Any]:
    root = run_dir / "privacy_hopqa_runs"
    records: list[dict[str, Any]] = []
    scanned = 0
    if root.exists():
        for entry in os.scandir(root):
            if scanned >= max_scan:
                break
            scanned += 1
            if not entry.is_dir():
                continue
            trace_path = Path(entry.path) / "trace.json"
            if not trace_path.exists():
                continue
            record = summarize_trace(trace_path, dataset)
            if record:
                records.append(record)
    records.sort(key=lambda item: item.get("mtime") or 0)
    durations = [safe_float(item.get("duration_s")) for item in records]
    durations = [value for value in durations if value is not None]
    p90 = sorted(durations)[int(0.9 * (len(durations) - 1))] if durations else None
    for record in records:
        if p90 is not None and (safe_float(record.get("duration_s")) or 0) >= p90:
            record["flags"].append("slow_p90")
    problem_records = [record for record in records if record.get("flags")]
    non_problem_records = [record for record in records if not record.get("flags")]
    rng = random.Random(17)
    selected: list[dict[str, Any]] = []

    def add_unique(items: list[dict[str, Any]], limit: int) -> None:
        seen = {item["rollout_id"] for item in selected}
        for item in items:
            if len(selected) >= limit:
                return
            rollout_id = item["rollout_id"]
            if rollout_id in seen:
                continue
            selected.append(item)
            seen.add(rollout_id)

    latest_records = sorted(records, key=lambda item: item.get("mtime") or 0, reverse=True)
    latest_problems = sorted(problem_records, key=lambda item: item.get("mtime") or 0, reverse=True)
    mid_start = max(0, len(records) // 2 - 40)
    middle_records = records[mid_start : mid_start + 80]
    random_records = list(records)
    rng.shuffle(random_records)

    add_unique(latest_problems, max_examples // 2)
    add_unique(latest_records, max_examples * 3 // 4)
    add_unique(records[:80], max_examples * 5 // 6)
    add_unique(middle_records, max_examples * 11 // 12)
    add_unique(random_records, max_examples)
    selected = selected[:max_examples]
    answered_count = sum(1 for item in records if str(item.get("predicted_answer") or "").strip())
    summary = {
        "scanned_dirs": scanned,
        "trace_records_loaded": len(records),
        "embedded_records": len(selected),
        "correct_count": sum(1 for item in records if item.get("correct")),
        "incorrect_count": sum(1 for item in records if not item.get("correct")),
        "answered_count": answered_count,
        "blank_count": len(records) - answered_count,
        "problem_count": len(problem_records),
        "mean_duration_s": mean([v for v in durations]) if durations else None,
        "p90_duration_s": p90,
        "mean_docs_read": mean([safe_float(item.get("docs_read")) or 0 for item in records]) if records else None,
        "mean_searches": mean([safe_float(item.get("searches_total")) or 0 for item in records]) if records else None,
        "flag_counts": {},
    }
    flag_counts: dict[str, int] = {}
    for record in records:
        for flag in record.get("flags") or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary["flag_counts"] = dict(sorted(flag_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    answer_audit = [
        {
            "rollout_id": record["rollout_id"],
            "chain_id": record["chain_id"],
            "task_id": record["task_id"],
            "company": record["company"],
            "question": record["question"],
            "predicted_answer": record["predicted_answer"],
            "gold_answer": record["gold_answer"],
            "accepted_answers": record["accepted_answers"],
            "calculated_reward": record["calculated_reward"],
            "correct": record["correct"],
            "flags": record["flags"],
            "duration_s": record["duration_s"],
            "searches_total": record["searches_total"],
            "docs_read": record["docs_read"],
            "mtime": record["mtime"],
        }
        for record in sorted(records, key=lambda item: (item["correct"], -(item.get("mtime") or 0)))
    ]
    return {"summary": summary, "records": selected, "answer_audit": answer_audit}


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_dir: Path = args.run_dir
    dataset = load_dataset(args.dataset)
    finetune = parse_finetune_log(run_dir / "finetune/log/info_0.log")
    preprocess = parse_preprocess_log(run_dir / "preprocess/info.log")
    actor = parse_actor_log(run_dir / "actor/info.log", args.log_tail_lines)
    warnings = parse_warning_log(run_dir / "actor/warning.log")
    traces = sample_trace_records(run_dir, dataset, args.max_scan, args.max_examples)
    latest_step = finetune["steps"][-1] if finetune["steps"] else {}
    recent_processed = preprocess["processed_batches"][-20:]
    recent_kept = sum(item["kept"] for item in recent_processed)
    recent_filtered = sum(item["filtered"] for item in recent_processed)
    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_dir": str(run_dir),
        "dataset_path": str(args.dataset),
        "latest_step": latest_step,
        "finetune_steps": finetune["steps"],
        "empty_batch_retries_after_latest_step": finetune["empty_after_latest_step"],
        "preprocess_batches": preprocess["processed_batches"][-240:],
        "recent_filter_rate": recent_filtered / max(1, recent_kept + recent_filtered),
        "actor": actor,
        "warnings": warnings,
        "traces": traces,
    }
    return payload


def json_for_html(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


def render_html(data: dict[str, Any]) -> str:
    blob = json_for_html(data)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Privacy HopQA First-Web 4B Run Viewer</title>
<style>
:root {{ --bg:#f5f4ef; --panel:#fff; --ink:#1e2430; --muted:#637084; --line:#d8deea; --accent:#1f5da8; --accent-soft:#e7f0fb; --bad:#b42318; --warn:#b56b00; --good:#137333; --shadow:0 14px 38px rgba(30,36,48,.08); }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:var(--ink); background:linear-gradient(180deg,#fbfaf6 0,var(--bg) 46%,#eceff5 100%); }}
.page {{ max-width:1500px; margin:0 auto; padding:28px 30px 56px; }}
.hero {{ background:linear-gradient(135deg,#ffffff 0,#f2f6fd 100%); border:1px solid var(--line); border-radius:24px; padding:24px 28px; box-shadow:var(--shadow); }}
.eyebrow {{ color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-bottom:8px; }}
h1 {{ margin:0 0 8px; font-size:34px; line-height:1.08; letter-spacing:-0.035em; }}
.hero p {{ margin:5px 0; color:var(--muted); }}
.topnav {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }}
.topnav a {{ color:var(--accent); background:var(--accent-soft); border:1px solid #cfe0f6; border-radius:999px; padding:7px 11px; text-decoration:none; font-size:13px; font-weight:700; }}
.stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }}
.stat-card {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:16px; box-shadow:0 8px 24px rgba(30,36,48,.06); }}
.stat-label {{ color:var(--muted); font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }}
.stat-value {{ font-size:27px; font-weight:850; margin-top:7px; letter-spacing:-.025em; }}
.section {{ margin-top:20px; background:rgba(255,255,255,.9); border:1px solid var(--line); border-radius:22px; padding:18px; box-shadow:0 8px 28px rgba(30,36,48,.05); }}
h2 {{ margin:0 0 12px; font-size:23px; letter-spacing:-.02em; }}
h3 {{ margin:16px 0 8px; }}
.small {{ color:var(--muted); font-size:13px; }}
.bad {{ color:var(--bad); font-weight:700; }} .warn {{ color:var(--warn); font-weight:700; }} .good {{ color:var(--good); font-weight:700; }}
.pill {{ display:inline-block; padding:3px 8px; margin:2px; border-radius:999px; background:#eef2f8; color:#39475c; font-size:12px; }}
.pill.bad {{ background:#f9dedb; color:var(--bad); }} .pill.good {{ background:#dff0e5; color:var(--good); }} .pill.warn {{ background:#faead2; color:var(--warn); }}
.plotgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; }}
.plot-card {{ background:#fbfcff; border:1px solid var(--line); border-radius:18px; padding:12px; }}
svg {{ width:100%; height:220px; background:#fff; border:1px solid var(--line); border-radius:14px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:16px; background:#fff; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 7px; text-align:left; vertical-align:top; }}
tr:last-child td {{ border-bottom:0; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
button {{ border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:999px; padding:7px 11px; cursor:pointer; font-weight:700; }}
button.active {{ background:var(--accent); color:white; border-color:var(--accent); }}
pre {{ white-space:pre-wrap; background:#1d231f; color:#f7efdd; border-radius:14px; padding:12px; overflow:auto; max-height:280px; font-size:12px; }}
details {{ border:1px solid var(--line); background:#fbfcff; border-radius:14px; padding:10px 12px; margin:10px 0; }}
summary {{ cursor:pointer; font-weight:700; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:900px) {{ .page {{ padding:16px; }} .two {{ grid-template-columns:1fr; }} .plotgrid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class=\"page\">
<section class=\"hero\">
<div class=\"eyebrow\">Privacy HopQA run viewer</div>
<h1>First-Web 4B Validation Run</h1>
<p>Generated: <span id=\"generated\"></span></p>
<p class=\"small\">Run: <code id=\"runDir\"></code></p>
<nav class=\"topnav\">
<a href=\"#summary\">Summary</a><a href=\"#concernsSec\">Concerns</a><a href=\"#curves\">Training curves</a><a href=\"#quality\">Rollout quality</a><a href=\"#answers\">Answer audit</a><a href=\"#detailsSec\">Details</a>
</nav>
</section>
<section class=\"section\" id=\"summary\"><h2>Summary</h2><div class=\"stat-grid\" id=\"cards\"></div></section>
<section class=\"section\" id=\"concernsSec\"><h2>Concrete concerns</h2><div id=\"concerns\"></div></section>
<section class=\"section\" id=\"curves\"><h2>Training curves</h2><div class=\"plotgrid\" id=\"plots\"></div><p class=\"small\">Reward is the batch mean logged by finetune. Filter rate is filtered / (kept + filtered) from preprocess batches.</p></section>
<section class=\"section\" id=\"quality\"><h2>Sampled rollout quality</h2><div id=\"traceSummary\"></div><div id=\"filters\" style=\"margin:12px 0\"></div><div id=\"rolloutTable\"></div></section>
<section class=\"section\" id=\"answers\"><h2>Answer audit table</h2><p class=\"small\">Calculated reward here is deterministic for this one-hop dataset: normalized predicted answer exact-matches any accepted answer variant => 1, otherwise 0. Use the answer-presence filters to separate blank failures from non-empty wrong answers.</p><div id=\"answerControls\" style=\"margin:12px 0\"></div><div id=\"answerAudit\"></div></section>
<section class=\"section\" id=\"detailsSec\"><h2>Rollout details</h2><div id=\"details\"></div></section>
</div>
<script>
const DATA = {blob};
const fmt = (x, d=3) => x === null || x === undefined || Number.isNaN(Number(x)) ? 'n/a' : Number(x).toLocaleString(undefined, {{maximumFractionDigits:d}});
const pct = x => x === null || x === undefined ? 'n/a' : (100*Number(x)).toFixed(1)+'%';
const esc = s => String(s ?? '').replace(/[&<>\"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
function val(obj, key) {{ return obj && Object.prototype.hasOwnProperty.call(obj,key) ? Number(obj[key]) : null; }}
function card(label, value, cls='') {{ return `<div class=\"stat-card\"><div class=\"stat-label\">${{esc(label)}}</div><div class=\"stat-value ${{cls}}\">${{esc(value)}}</div></div>`; }}
function metric(step, key) {{ return step && step[key] !== undefined ? Number(step[key]) : null; }}
function lineSvg(title, rows, key, color) {{
  const vals = rows.map(r => metric(r,key)).filter(v => Number.isFinite(v));
  if (!vals.length) return `<div class=\"plot-card\"><h3>${{esc(title)}}</h3><svg><text x=\"20\" y=\"40\">No data</text></svg></div>`;
  const points = rows.map((r,i)=>[i, metric(r,key)]).filter(p=>Number.isFinite(p[1]));
  const minX = points[0][0], maxX = points[points.length-1][0];
  const minY = Math.min(...vals), maxY = Math.max(...vals); const pad=28, w=640, h=220;
  const sx = x => pad + (x-minX) / Math.max(1,(maxX-minX)) * (w-2*pad);
  const sy = y => h-pad - (y-minY) / Math.max(1e-9,(maxY-minY)) * (h-2*pad);
  const d = points.map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+','+sy(p[1]).toFixed(1)).join(' ');
  const last = points[points.length-1][1];
  return `<div class=\"plot-card\"><h3>${{esc(title)}} <span class=\"small\">latest ${{fmt(last)}}</span></h3><svg viewBox=\"0 0 ${{w}} ${{h}}\"><path d=\"M${{pad}},${{h-pad}} L${{w-pad}},${{h-pad}} M${{pad}},${{pad}} L${{pad}},${{h-pad}}\" stroke=\"#d8deea\" fill=\"none\"/><path d=\"${{d}}\" stroke=\"${{color}}\" stroke-width=\"3\" fill=\"none\"/><text x=\"${{pad}}\" y=\"18\" font-size=\"12\">max ${{fmt(maxY)}}</text><text x=\"${{pad}}\" y=\"${{h-6}}\" font-size=\"12\">min ${{fmt(minY)}}</text></svg></div>`;
}}
function preprocessSvg() {{
  const rows = DATA.preprocess_batches || [];
  const withRates = rows.map((r,i)=>({{i, rate:Number(r.filter_rate), kept:Number(r.kept), filtered:Number(r.filtered)}}));
  return lineSvg('Zero-advantage filter rate', withRates.map(r=>({{'rate':r.rate}})), 'rate', '#ad741f');
}}
function renderCards() {{
  const s = DATA.latest_step || {{}};
  const traces = DATA.traces.summary || {{}};
  const actor = DATA.actor || {{}};
  document.getElementById('generated').textContent = DATA.generated_at_utc;
  document.getElementById('runDir').textContent = DATA.run_dir;
  document.getElementById('cards').innerHTML = [
    card('latest completed step', fmt(metric(s,'stats/completed_steps'),0)),
    card('trainer samples', fmt(metric(s,'stats/samples'),0)),
    card('trainer tokens', fmt(metric(s,'stats/tokens'),0)),
    card('latest reward', fmt(metric(s,'rl/reward'),3), metric(s,'rl/reward') < .25 ? 'warn' : ''),
    card('recent filter rate', pct(DATA.recent_filter_rate), DATA.recent_filter_rate > .75 ? 'bad' : 'warn'),
    card('recent 2-token hop plans', pct(actor.two_token_hop_plan_rate_recent), actor.two_token_hop_plan_rate_recent > .1 ? 'warn' : ''),
    card('sample answer exact match', pct((traces.correct_count||0)/Math.max(1,traces.trace_records_loaded||0)), ''),
    card('sample answered rate', pct((traces.answered_count||0)/Math.max(1,traces.trace_records_loaded||0)), ''),
    card('blank predictions', fmt(traces.blank_count||0,0), traces.blank_count ? 'warn' : '')
  ].join('');
}}
function renderConcerns() {{
  const s = DATA.latest_step || {{}}; const actor=DATA.actor||{{}}; const warn=DATA.warnings||{{}}; const traces=DATA.traces.summary||{{}};
  const items = [];
  items.push(`<p><b>Trainer starvation:</b> ${{DATA.empty_batch_retries_after_latest_step}} batch-empty retries after latest completed step. Latest step is ${{fmt(metric(s,'stats/completed_steps'),0)}} at ${{esc(s.timestamp)}}.</p>`);
  items.push(`<p><b>Zero-advantage filtering:</b> recent filter rate is <span class=\"${{DATA.recent_filter_rate > .75 ? 'bad' : 'warn'}}\">${{pct(DATA.recent_filter_rate)}}</span>. This means many rollout groups have no within-group learning signal.</p>`);
  items.push(`<p><b>Planner collapse proxy:</b> recent hop-plan calls with exactly 2 output tokens: <span class=\"${{actor.two_token_hop_plan_rate_recent > .1 ? 'warn' : 'good'}}\">${{actor.two_token_hop_plans_recent || 0}} / ${{actor.hop_plan_calls_recent || 0}} (${{pct(actor.two_token_hop_plan_rate_recent)}})</span>.</p>`);
  items.push(`<p><b>Retrieval failures:</b> actor warning log contains ${{warn.retrieval_failure_count || 0}} retrieval-action failures. Recent low-concurrency run should keep this near zero.</p>`);
  items.push(`<p><b>Sampled answer mismatch:</b> ${{traces.incorrect_count || 0}} / ${{traces.trace_records_loaded || 0}} sampled traces did not exact-match accepted answers after normalization.</p>`);
  document.getElementById('concerns').innerHTML = items.join('');
}}
function renderPlots() {{
  const rows = DATA.finetune_steps || [];
  document.getElementById('plots').innerHTML = [
    lineSvg('RL reward by step', rows, 'rl/reward', '#1e6b52'),
    lineSvg('Seconds per step', rows, 'throughput/sec_per_step', '#285c8f'),
    lineSvg('Real tokens/sec', rows, 'throughput/real_tokens_per_sec', '#8b5a22'),
    lineSvg('KL', rows, 'rl/kl', '#7a3d3d'),
    preprocessSvg()
  ].join('');
}}
let currentFilter = 'problem';
let answerFilter = 'wrong';
let answerLimit = 500;
function flagPills(flags) {{ return (flags||[]).map(f=>`<span class=\"pill ${{f.includes('mismatch')||f.includes('failed')?'bad':f.includes('slow')||f.includes('many')||f.includes('max')?'warn':''}}\">${{esc(f)}}</span>`).join(''); }}
function recordRow(r, idx) {{
  const cls = r.correct ? 'good' : 'bad';
  return `<tr><td><button onclick=\"showDetail(${{idx}})\">open</button></td><td>${{esc(r.chain_id)}}<br><span class=\"small\">${{esc(r.task_id)}} / ${{esc(r.company)}} </span></td><td>${{esc(r.question)}}</td><td class=\"${{cls}}\">${{r.correct?'yes':'no'}}</td><td>${{esc(r.predicted_answer)}}<br><span class=\"small\">gold: ${{esc(r.gold_answer)}}</span></td><td>${{fmt(r.searches_total,0)}} / ${{fmt(r.docs_read,0)}}</td><td>${{fmt(r.duration_s,1)}}s</td><td>${{flagPills(r.flags)}}</td></tr>`;
}}
function filteredRecords() {{
  const rows = DATA.traces.records || [];
  if (currentFilter === 'all') return rows;
  if (currentFilter === 'incorrect') return rows.filter(r=>!r.correct);
  if (currentFilter === 'slow') return rows.filter(r=>(r.flags||[]).includes('slow_p90'));
  if (currentFilter === 'planner') return rows.filter(r=>(r.flags||[]).includes('no_search_history') || (r.flags||[]).includes('max_iterations'));
  return rows.filter(r=>(r.flags||[]).length);
}}
function setFilter(f) {{ currentFilter=f; renderTable(); }}
function renderTable() {{
  const rows = filteredRecords();
  document.getElementById('filters').innerHTML = ['problem','incorrect','planner','slow','all'].map(f=>`<button class=\"${{currentFilter===f?'active':''}}\" onclick=\"setFilter('${{f}}')\">${{f}}</button>`).join(' ');
  document.getElementById('rolloutTable').innerHTML = `<div class=\"table-wrap\"><table><thead><tr><th></th><th>chain</th><th>question</th><th>exact</th><th>answer</th><th>searches/docs</th><th>duration</th><th>flags</th></tr></thead><tbody>${{rows.slice(0,80).map((r)=>recordRow(r, (DATA.traces.records||[]).indexOf(r))).join('')}}</tbody></table></div>`;
}}
function renderTraceSummary() {{
  const s = DATA.traces.summary || {{}};
  const flags = Object.entries(s.flag_counts || {{}}).map(([k,v])=>`<span class=\"pill\">${{esc(k)}}: ${{v}}</span>`).join('');
  document.getElementById('traceSummary').innerHTML = `<p>Scanned ${{s.scanned_dirs || 0}} rollout dirs, loaded ${{s.trace_records_loaded || 0}} traces, embedded ${{s.embedded_records || 0}} examples. Mean duration ${{fmt(s.mean_duration_s,1)}}s, p90 ${{fmt(s.p90_duration_s,1)}}s, mean searches ${{fmt(s.mean_searches,2)}}, mean docs read ${{fmt(s.mean_docs_read,2)}}.</p><p>${{flags}}</p>`;
}}
function answerRows() {{
  const rows = DATA.traces.answer_audit || [];
  const hasPrediction = r => String(r.predicted_answer || '').trim().length > 0;
  if (answerFilter === 'wrong') return rows.filter(r=>!r.correct);
  if (answerFilter === 'nonempty_wrong') return rows.filter(r=>!r.correct && hasPrediction(r));
  if (answerFilter === 'blank') return rows.filter(r=>!hasPrediction(r));
  if (answerFilter === 'answered') return rows.filter(r=>hasPrediction(r));
  if (answerFilter === 'right') return rows.filter(r=>r.correct);
  if (answerFilter === 'flagged') return rows.filter(r=>(r.flags||[]).length);
  return rows;
}}
function setAnswerFilter(f) {{ answerFilter=f; renderAnswerAudit(); }}
function setAnswerLimit(n) {{ answerLimit=n; renderAnswerAudit(); }}
function answerFlagPills(r) {{
  const reward = Number(r.calculated_reward || 0);
  const rewardPill = `<span class=\"pill ${{reward === 1 ? 'good' : 'bad'}}\">reward=${{reward.toFixed(0)}}</span>`;
  return rewardPill + flagPills(r.flags || []);
}}
function renderAnswerAudit() {{
  const rows = answerRows();
  const limited = answerLimit === -1 ? rows : rows.slice(0, answerLimit);
  document.getElementById('answerControls').innerHTML =
    [['wrong','wrong'],['nonempty_wrong','nonempty wrong'],['blank','blank'],['answered','answered'],['flagged','flagged'],['right','right'],['all','all']].map(([f,label])=>`<button class=\"${{answerFilter===f?'active':''}}\" onclick=\"setAnswerFilter('${{f}}')\">${{label}}</button>`).join(' ') +
    ' <span class=\"small\"> rows: </span>' +
    [[500,'500'],[2000,'2k'],[-1,'all']].map(([n,label])=>`<button class=\"${{answerLimit===n?'active':''}}\" onclick=\"setAnswerLimit(${{n}})\">${{label}}</button>`).join(' ') +
    ` <span class=\"small\">showing ${{limited.length.toLocaleString()}} / ${{rows.length.toLocaleString()}}</span>`;
  document.getElementById('answerAudit').innerHTML = `<div class=\"table-wrap\"><table><thead><tr><th>reward</th><th>chain</th><th>question</th><th>predicted answer</th><th>gold answer</th><th>accepted variants</th><th>search/docs</th><th>flags</th></tr></thead><tbody>${{
    limited.map(r=>`<tr><td class=\"${{r.correct?'good':'bad'}}\">${{fmt(r.calculated_reward,0)}}</td><td>${{esc(r.chain_id)}}<br><span class=\"small\">${{esc(r.task_id)}} / ${{esc(r.company)}}</span></td><td>${{esc(r.question)}}</td><td>${{esc(r.predicted_answer)}}</td><td>${{esc(r.gold_answer)}}</td><td>${{esc((r.accepted_answers||[]).join(' | '))}}</td><td>${{fmt(r.searches_total,0)}} / ${{fmt(r.docs_read,0)}}<br><span class=\"small\">${{fmt(r.duration_s,1)}}s</span></td><td>${{answerFlagPills(r)}}</td></tr>`).join('')
  }}</tbody></table></div>`;
}}
function showDetail(idx) {{
  const r = (DATA.traces.records || [])[idx]; if (!r) return;
  const searches = (r.search_history||[]).map(s=>`<tr><td>${{esc(s.iteration)}}</td><td>${{esc(s.type)}}</td><td>${{esc(s.query)}}</td><td>${{fmt(s.result_count,0)}}</td><td>${{esc((s.top_doc_ids||[]).join(', '))}}</td></tr>`).join('');
  const readers = (r.reader_sample||[]).map(x=>`<details><summary>${{esc(x.doc_id)}} | ${{esc(x.source)}} | can_answer=${{esc(x.can_answer)}} | proposed=${{esc(x.proposed_answer)}}</summary><p><b>${{esc(x.title)}}</b></p><p>${{esc(x.justification)}}</p></details>`).join('');
  const actions = (r.actions||[]).map(a=>`<tr><td>${{esc(a.id)}}</td><td>${{esc(a.type)}}</td><td>${{esc(a.status)}}</td><td>${{esc(a.query)}}</td><td>${{fmt(a.results_count,0)}}</td><td>${{fmt(a.execution_time,2)}}s</td><td>${{esc(a.error)}}</td></tr>`).join('');
  document.getElementById('details').innerHTML = `<h3>${{esc(r.chain_id)}} <span class=\"small\">${{esc(r.rollout_id)}}</span></h3><p>${{flagPills(r.flags)}}</p><div class=\"two\"><div><p><b>Question:</b> ${{esc(r.question)}}</p><p><b>Predicted:</b> <span class=\"${{r.correct?'good':'bad'}}\">${{esc(r.predicted_answer)}}</span></p><p><b>Gold:</b> ${{esc(r.gold_answer)}}</p><p><b>Accepted variants:</b> ${{esc((r.accepted_answers||[]).join(' | '))}}</p><p><b>Reason:</b> ${{esc(r.reason)}}</p><p><b>Justification:</b> ${{esc(r.justification)}}</p><p class=\"small\">Trace path: ${{esc(r.trace_path)}}</p></div><div><p><b>Status:</b> ${{esc(r.status)}} confidence ${{fmt(r.confidence,2)}}</p><p><b>Iterations/searches/docs/LLM calls:</b> ${{fmt(r.iterations_used,0)}} / ${{fmt(r.searches_total,0)}} / ${{fmt(r.docs_read,0)}} / ${{fmt(r.llm_calls_total,0)}}</p><p><b>Tokens:</b> ${{fmt(r.total_tokens,0)}}; duration ${{fmt(r.duration_s,1)}}s</p></div></div><h3>Search history</h3><div class=\"table-wrap\"><table><thead><tr><th>iter</th><th>type</th><th>query</th><th>results</th><th>top docs</th></tr></thead><tbody>${{searches}}</tbody></table></div><h3>Actions</h3><div class=\"table-wrap\"><table><thead><tr><th>id</th><th>type</th><th>status</th><th>query</th><th>results</th><th>time</th><th>error</th></tr></thead><tbody>${{actions}}</tbody></table></div><h3>Reader sample</h3>${{readers || '<p>No reader sample embedded.</p>'}}`;
  window.scrollTo({{top: document.getElementById('details').offsetTop - 12, behavior:'smooth'}});
}}
renderCards(); renderConcerns(); renderPlots(); renderTraceSummary(); renderTable(); renderAnswerAudit();
showDetail(0);
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    data = make_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {args.output}")
    latest = data.get("latest_step") or {}
    traces = data.get("traces", {}).get("summary", {})
    print(
        "summary step={step} reward={reward} traces={traces_loaded} exact={exact}/{loaded} filter_rate={filter_rate:.3f}".format(
            step=latest.get("stats/completed_steps"),
            reward=latest.get("rl/reward"),
            traces_loaded=traces.get("trace_records_loaded"),
            exact=traces.get("correct_count"),
            loaded=max(1, traces.get("trace_records_loaded") or 0),
            filter_rate=float(data.get("recent_filter_rate") or 0),
        )
    )


if __name__ == "__main__":
    main()
