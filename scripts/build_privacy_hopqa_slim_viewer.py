#!/usr/bin/env python3
"""Build a compact single-file viewer from a privacy_hopqa analysis bundle."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


TEMPORAL_PLOTS = [
    ("temporal_distributions.png", "Early / middle / late distributions"),
    ("temporal_searches_vs_docs.png", "Searches vs docs by window"),
    ("temporal_source_distributions.png", "Web/local source distributions"),
    ("temporal_source_balance.png", "Mean source usage"),
    ("temporal_source_hop_alignment.png", "Hop/source alignment"),
    ("temporal_source_usage.png", "Source usage boxplots"),
    ("temporal_moving_averages.png", "Moving averages"),
]

QUESTION_TYPE_PLOTS = [
    ("question_type_retrieval_alignment.png", "Source retrieval vs source hops"),
    ("question_type_composition_heatmaps.png", "Reward and cost by hop composition"),
    ("question_type_source_fraction.png", "Local-hop fraction vs local retrieval fraction"),
    ("question_type_pattern_performance.png", "Pattern-level performance and cost"),
    ("question_type_local_hop_sensitivity.png", "Local-hop sensitivity"),
    ("question_type_source_mismatch.png", "Wrong-source retrieval checks"),
]

PLOTS = [
    ("rollout_duration_distribution.png", "Rollout duration"),
    ("iteration_distribution.png", "Iterations"),
    ("stage_duration_boxplot.png", "Stage timing"),
    ("reward_distribution.png", "Reward distribution"),
    ("reward_by_hop_length.png", "Reward by hop length"),
    ("llm_calls_distribution.png", "LLM calls per rollout"),
    ("llm_total_tokens_per_rollout.png", "Total LLM tokens"),
    ("total_tokens_by_call_type.png", "Total tokens by call type"),
    ("total_tokens_by_final_iteration.png", "Tokens by final iteration"),
    ("searches_vs_docs_read.png", "Searches vs docs read"),
    ("source_usage_distributions.png", "Source usage"),
    ("prompt_tokens_by_call_type.png", "Prompt tokens"),
    ("output_tokens_by_call_type.png", "Output tokens"),
    ("parse_error_rate_by_stage.png", "Parse errors"),
    ("errors_by_stage.png", "Errors by stage"),
]

TREND_METRICS = [
    ("reward", "Reward", ""),
    ("duration_s", "Duration", "s"),
    ("iterations_used", "Iterations", ""),
    ("llm_calls_total", "LLM calls", ""),
    ("searches_total", "Searches", ""),
    ("web_searches", "Web searches", ""),
    ("local_searches", "Local searches", ""),
    ("docs_read", "Docs read", ""),
    ("web_docs_read", "Web docs read", ""),
    ("local_docs_read", "Local docs read", ""),
    ("total_tokens", "Total tokens", ""),
    ("prompt_tokens", "Prompt tokens", ""),
    ("output_tokens", "Output tokens", ""),
]

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-05.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, help="Directory produced by plot_privacy_hopqa_runtime.py")
    parser.add_argument("--output", help="Output HTML path. Defaults to <analysis-dir>/slim_viewer.html")
    parser.add_argument("--max-samples", type=int, default=8, help="Representative examples to embed")
    parser.add_argument("--max-events-per-sample", type=int, default=16, help="Timeline rows per embedded example")
    parser.add_argument("--time-slice-size", type=int, default=128, help="Rollouts per early/middle/late window")
    parser.add_argument("--moving-average-window", type=int, default=256, help="Rollouts per moving-average point")
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="Final Privacy HopQA JSONL dataset used to enrich examples with gold answers and accepted answer variants.",
    )
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if limit is not None and len(text) > limit:
        return text[:limit].rstrip() + "\n..."
    return text


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return esc(value)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := as_float(row.get(key))) is not None]
    return mean(values) if values else None


def median_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := as_float(row.get(key))) is not None]
    return median(values) if values else None


def values_for(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [value for row in rows if (value := as_float(row.get(key))) is not None]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def load_rollout_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {row.get("rollout_id", ""): row for row in rows if row.get("rollout_id")}


def load_trend_rows(analysis_dir: Path, row_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = read_csv_rows(analysis_dir / "rollout_trends.csv")
    if rows:
        return sorted(rows, key=lambda row: as_float(row.get("completed_at_s")) or 0.0)

    # Older bundles do not have rollout_trends.csv. Fall back to the sampled
    # rows and stat the trace files while the shared filesystem is mounted.
    fallback_rows: list[dict[str, str]] = []
    for row in row_by_id.values():
        copied = dict(row)
        completed_at = as_float(copied.get("completed_at_s"))
        trace_path = copied.get("trace_path")
        if completed_at is None and trace_path:
            path = Path(trace_path)
            if path.exists():
                completed_at = path.stat().st_mtime
        if completed_at is not None:
            copied["completed_at_s"] = str(completed_at)
            fallback_rows.append(copied)
    return sorted(fallback_rows, key=lambda row: as_float(row.get("completed_at_s")) or 0.0)


def load_problem_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    problems: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                problem = json.loads(line)
            except json.JSONDecodeError:
                continue
            chain_id = str(problem.get("chain_id") or "")
            if chain_id:
                problems[chain_id] = problem
    return problems


def unique_texts(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def accepted_answers_for_hop(problem_hop: dict[str, Any] | None, fallback: str | None = None) -> list[str]:
    if not problem_hop:
        return unique_texts([fallback])
    return unique_texts(
        [
            problem_hop.get("answer"),
            *(problem_hop.get("accepted_answer_variants") or []),
            *(problem_hop.get("alternate_valid_answers") or []),
            fallback,
        ]
    )


def accepted_answers_for_final(problem: dict[str, Any] | None, fallback: str | None = None) -> list[str]:
    if not problem:
        return unique_texts([fallback])
    return unique_texts(
        [
            problem.get("global_answer"),
            *(problem.get("global_answer_variants") or []),
            *(problem.get("global_alternate_valid_answers") or []),
            fallback,
        ]
    )


def format_answer_list(values: list[str], limit: int = 420) -> str:
    text = " | ".join(values)
    return truncate_text(text, limit)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def example_row_from_trace(sample_dir: Path, row_by_id: dict[str, dict[str, str]], trace: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = dict(row_by_id.get(sample_dir.name, {}))
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    if summary:
        for key, value in summary.items():
            row.setdefault(key, value)
    row.setdefault("rollout_id", sample_dir.name)
    return row


def choose_sample_dirs(sample_root: Path, row_by_id: dict[str, dict[str, str]], limit: int) -> list[Path]:
    dirs = [path for path in sample_root.iterdir() if path.is_dir()] if sample_root.exists() else []
    if len(dirs) <= limit:
        return sorted(dirs)

    def sort_key(path: Path) -> tuple[float, float, str]:
        row = row_by_id.get(path.name, {})
        try:
            reward = float(row.get("reward") or 0.0)
        except ValueError:
            reward = 0.0
        try:
            duration = float(row.get("duration_s") or 0.0)
        except ValueError:
            duration = 0.0
        return reward, duration, path.name

    ordered = sorted(dirs, key=sort_key)
    if limit <= 1:
        return [ordered[-1]]

    # Evenly sample the ordered reward/duration spectrum so failures, partial
    # successes, and full successes are all likely to be visible.
    indices = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)]
    seen: set[int] = set()
    chosen: list[Path] = []
    for index in indices:
        if index not in seen:
            chosen.append(ordered[index])
            seen.add(index)
    return chosen


def event_detail(event: dict[str, Any]) -> str:
    meta = dict(event.get("meta") or {})
    stage = str(event.get("stage") or "")
    pieces: list[str] = []
    for key in ("hop_number", "iteration", "action_type", "result_count", "candidate_count"):
        if key in meta:
            pieces.append(f"{key}={meta[key]}")
    if stage == "search" and meta.get("query"):
        pieces.append(f"query={str(meta['query'])[:120]}")
    if stage == "doc_choose" and meta.get("selected_doc_ids"):
        pieces.append(f"selected={', '.join(map(str, meta['selected_doc_ids'][:4]))}")
    if stage == "doc_read":
        for key in ("doc_id", "parent_doc_id", "window_index", "window_count", "source", "title"):
            if meta.get(key):
                pieces.append(f"{key}={str(meta[key])[:80]}")
    return "; ".join(pieces)


def summarize_events(events: list[dict[str, Any]], max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_totals: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: float(item.get("start_s") or 0.0)):
        start = float(event.get("start_s") or 0.0)
        end = float(event.get("end_s") or start)
        duration = max(0.0, end - start)
        stage = str(event.get("stage") or "unknown")
        stage_totals[stage] = stage_totals.get(stage, 0.0) + duration
        if len(rows) < max_rows:
            rows.append(
                {
                    "stage": stage,
                    "label": event.get("label") or "",
                    "start_s": start,
                    "duration_s": duration,
                    "detail": event_detail(event),
                }
            )
    totals = [
        {"stage": stage, "duration_s": duration}
        for stage, duration in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    return rows, totals[:8]


def render_stat_cards(summary: dict[str, Any]) -> str:
    cards = [
        ("Rollouts", fmt_num(summary.get("rollout_count"), 0)),
        ("Mean reward", fmt_num(summary.get("mean_reward"), 3)),
        ("Solved", fmt_pct(summary.get("solved_fraction"), 1)),
        ("Rollouts / sec", fmt_num(summary.get("completed_rollouts_per_s"), 3)),
        ("Mean duration", f"{fmt_num(summary.get('mean_duration_s'), 1)}s"),
        ("Median duration", f"{fmt_num(summary.get('median_duration_s'), 1)}s"),
        ("Mean LLM calls", fmt_num(summary.get("mean_llm_calls"), 1)),
        ("Mean prompt / call", fmt_num(summary.get("mean_prompt_tokens"), 1)),
        ("Mean output / call", fmt_num(summary.get("mean_output_tokens"), 1)),
        ("Median iterations", fmt_num(summary.get("median_iterations"), 1)),
        ("Mean web searches", fmt_num(summary.get("mean_web_searches"), 1)),
        ("Mean web docs read", fmt_num(summary.get("mean_web_docs_read"), 1)),
        ("Parse error rate", fmt_pct(summary.get("parse_error_rate"), 2)),
        ("Context overflows", fmt_num(summary.get("context_overflow_errors"), 0)),
    ]
    return "\n".join(
        f"<div class='stat'><div class='label'>{esc(label)}</div><div class='value'>{value}</div></div>"
        for label, value in cards
    )


def render_plots(analysis_dir: Path, plots: list[tuple[str, str]] = PLOTS) -> str:
    cards: list[str] = []
    for filename, title in plots:
        path = analysis_dir / filename
        uri = image_data_uri(path)
        if not uri:
            continue
        cards.append(
            f"""
            <article class="plot">
              <h3>{esc(title)}</h3>
              <button class="plot-open" type="button" aria-label="Open {esc(title)} full size">
                <img src="{uri}" alt="{esc(title)}" loading="lazy" />
              </button>
            </article>
            """
        )
    return "\n".join(cards)


def render_table(rows: list[dict[str, Any]], headers: list[tuple[str, str]], raw_columns: set[str] | None = None) -> str:
    raw_columns = raw_columns or set()
    head = "".join(f"<th>{esc(label)}</th>" for _, label in headers)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{row.get(key, '') if key in raw_columns else esc(row.get(key, ''))}</td>"
            for key, _ in headers
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def truncate_text(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def render_hop_state_table(trace: dict[str, Any], row: dict[str, Any], problem: dict[str, Any] | None) -> str:
    rows: list[dict[str, Any]] = []
    hop_states = {
        int(hop.get("hop_number")): hop
        for hop in trace.get("hop_states") or []
        if isinstance(hop, dict) and str(hop.get("hop_number") or "").isdigit()
    }
    problem_hops = {
        int(hop.get("hop_number")): hop
        for hop in (problem or {}).get("hops", [])
        if isinstance(hop, dict) and str(hop.get("hop_number") or "").isdigit()
    }
    total_hops = as_float(row.get("total_hops") or row.get("hop_count") or (problem or {}).get("n_hops")) or max(
        len(hop_states),
        len(problem_hops),
        1,
    )
    hop_numbers = sorted(set(hop_states) | set(problem_hops))
    for hop_number in hop_numbers:
        hop = hop_states.get(hop_number, {})
        problem_hop = problem_hops.get(hop_number, {})
        correct = parse_bool(row.get(f"hop_{hop_number}_correct"))
        reward_contribution = ""
        if correct is not None:
            reward_contribution = fmt_num((1.0 / total_hops) if correct else 0.0, 3)
        gold = problem_hop.get("answer") or row.get(f"hop_{hop_number}_gold")
        accepted_answers = accepted_answers_for_hop(problem_hop, row.get(f"hop_{hop_number}_gold"))
        rows.append(
            {
                "hop": hop_number,
                "type": problem_hop.get("hop_type", ""),
                "status": hop.get("status", ""),
                "attempts": hop.get("attempts", ""),
                "correct": "" if correct is None else str(correct),
                "reward": reward_contribution,
                "match": fmt_num(row.get(f"hop_{hop_number}_match_score"), 3) if row.get(f"hop_{hop_number}_match_score") not in (None, "") else "",
                "question": truncate_text(problem_hop.get("question") or hop.get("question"), 260),
                "predicted": truncate_text(hop.get("answer") or row.get(f"hop_{hop_number}_answer"), 160),
                "gold": truncate_text(gold, 160),
                "accepted": format_answer_list(accepted_answers),
                "reason": truncate_text(hop.get("resolution_reason") or hop.get("justification"), 260),
            }
        )
    if not rows:
        return "<p class='note'>No hop-state records were copied for this example.</p>"
    return render_table(
        rows,
        [
            ("hop", "Hop"),
            ("type", "Type"),
            ("status", "Status"),
            ("attempts", "Attempts"),
            ("correct", "Correct"),
            ("reward", "Reward contrib"),
            ("match", "Match"),
            ("question", "Question"),
            ("predicted", "Predicted answer"),
            ("gold", "Gold answer"),
            ("accepted", "Accepted answer(s)"),
            ("reason", "Reason"),
        ],
    )


def action_result_preview(action: dict[str, Any], limit: int = 3) -> str:
    output = action.get("actual_output") if isinstance(action.get("actual_output"), dict) else {}
    results = output.get("results") if isinstance(output.get("results"), list) else []
    if not results:
        summary = output.get("summary") or output.get("synthesis") or output.get("error") or ""
        return f"<p class='note'>{esc(truncate_text(summary, 420) or 'No result preview recorded.')}</p>"

    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results[:limit], start=1):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        parent_doc_id = result.get("parent_doc_id") or metadata.get("parent_doc_id") or result.get("doc_id") or result.get("docid")
        window_index = result.get("window_index") or metadata.get("window_index")
        window_count = result.get("window_count") or metadata.get("window_count")
        window_label = f"{window_index}/{window_count}" if window_index else ""
        rows.append(
            {
                "rank": index,
                "doc_id": result.get("doc_id") or result.get("docid"),
                "parent_doc_id": parent_doc_id,
                "window": window_label,
                "source": result.get("source"),
                "title": truncate_text(result.get("title") or result.get("url") or metadata.get("filename"), 170),
                "snippet": truncate_text(result.get("snippet") or result.get("preview") or result.get("content") or result.get("text"), 300),
            }
        )
    return render_table(
        rows,
        [
            ("rank", "#"),
            ("doc_id", "Evidence window"),
            ("parent_doc_id", "Parent doc"),
            ("window", "Window"),
            ("source", "Source"),
            ("title", "Title"),
            ("snippet", "Preview"),
        ],
    )


def render_action_cards(trace: dict[str, Any], max_actions: int = 48) -> str:
    actions = [action for action in trace.get("actions") or [] if isinstance(action, dict)]
    if not actions:
        return "<p class='note'>No action records were copied for this example.</p>"

    actions = sorted(actions, key=lambda item: (int(item.get("iteration") or 0), str(item.get("id") or "")))
    visible_actions = actions[:max_actions]
    max_exec = max([as_float(action.get("execution_time")) or 0.0 for action in visible_actions] + [0.001])
    cards: list[str] = []
    current_iteration: int | None = None
    open_grid = False
    for action in visible_actions:
        iteration = int(action.get("iteration") or 0)
        if current_iteration != iteration:
            if open_grid:
                cards.append("</div></section>")
            current_iteration = iteration
            cards.append(f"<section class='iter-block'><h4>Iteration {iteration}</h4><div class='action-grid'>")
            open_grid = True

        output = action.get("actual_output") if isinstance(action.get("actual_output"), dict) else {}
        parameters = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}
        query = parameters.get("query") or output.get("query") or ""
        exec_s = as_float(action.get("execution_time")) or 0.0
        width_pct = 8.0 + 92.0 * math.sqrt(exec_s / max_exec)
        result_count = output.get("results_count", "")
        action_type = str(action.get("type") or "action")
        source_class = "local-action" if "local" in action_type else "web-action" if "web" in action_type else ""
        cards.append(
            f"""
            <article class="action-card {source_class}">
              <div class="action-bar" style="width:{width_pct:.1f}%"></div>
              <div class="action-head">
                <strong>{esc(action_type)}</strong>
                <span>{fmt_num(exec_s, 2)}s</span>
              </div>
              <p><strong>ID:</strong> {esc(action.get("id"))}</p>
              <p><strong>Query:</strong> {esc(query)}</p>
              <p><strong>Status:</strong> {esc(action.get("status"))} | <strong>Results:</strong> {esc(result_count)}</p>
              <p><strong>Description:</strong> {esc(truncate_text(action.get("description"), 220))}</p>
              <details>
                <summary>Top retrieved docs</summary>
                {action_result_preview(action)}
              </details>
            </article>
            """
        )
    if open_grid:
        cards.append("</div></section>")
    if len(actions) > max_actions:
        cards.append(f"<p class='note'>Showing the first {max_actions} of {len(actions)} action records for this example.</p>")
    return "\n".join(cards)


def render_token_section(analysis_dir: Path, summary: dict[str, Any]) -> str:
    token_rows = read_csv_rows(analysis_dir / "llm_token_summary.csv")
    if not token_rows:
        return "<p class='note'>No per-call token summary was found in this bundle.</p>"

    table_rows: list[dict[str, Any]] = []
    for row in token_rows:
        table_rows.append(
            {
                "call_type": row.get("call_type"),
                "calls": fmt_num(row.get("calls"), 0),
                "median_prompt": fmt_num(row.get("median_prompt_tokens"), 1),
                "median_output": fmt_num(row.get("median_output_tokens"), 1),
                "median_total": fmt_num(row.get("median_total_tokens"), 1),
                "mean_prompt": fmt_num(row.get("mean_prompt_tokens"), 1),
                "mean_output": fmt_num(row.get("mean_output_tokens"), 1),
                "mean_total": fmt_num(row.get("mean_total_tokens"), 1),
            }
        )

    cards = [
        ("Mean prompt / call", fmt_num(summary.get("mean_prompt_tokens"), 1)),
        ("Median prompt / call", fmt_num(summary.get("median_prompt_tokens"), 1)),
        ("Mean output / call", fmt_num(summary.get("mean_output_tokens"), 1)),
        ("Median output / call", fmt_num(summary.get("median_output_tokens"), 1)),
    ]
    card_html = "\n".join(
        f"<div class='mini-stat'><span>{esc(label)}</span><strong>{value}</strong></div>"
        for label, value in cards
    )
    table = render_table(
        table_rows,
        [
            ("call_type", "Call type"),
            ("calls", "Calls"),
            ("median_prompt", "Med prompt"),
            ("median_output", "Med output"),
            ("median_total", "Med total"),
            ("mean_prompt", "Mean prompt"),
            ("mean_output", "Mean output"),
            ("mean_total", "Mean total"),
        ],
    )
    return f"<div class='mini-stats'>{card_html}</div>{table}"


def render_timeline_overview(analysis_dir: Path) -> str:
    rows = read_csv_rows(analysis_dir / "events_sample.csv")
    if not rows:
        return "<p class='note'>No sampled timeline events were found in this bundle.</p>"

    by_stage: dict[str, list[float]] = {}
    for row in rows:
        stage = row.get("stage") or "unknown"
        duration = as_float(row.get("duration_s"))
        if duration is not None:
            by_stage.setdefault(stage, []).append(duration)
    if not by_stage:
        return "<p class='note'>Timeline events did not include usable durations.</p>"

    max_mean = max(mean(values) for values in by_stage.values())
    table_rows: list[dict[str, Any]] = []
    for stage, values in sorted(by_stage.items(), key=lambda item: mean(item[1]), reverse=True):
        stage_mean = mean(values)
        table_rows.append(
            {
                "stage": stage,
                "events": fmt_num(len(values), 0),
                "mean_s": f"{stage_mean:.2f}s",
                "median_s": f"{median(values):.2f}s",
                "bar": f"<span class='bar' style='width:{max(2.0, 100.0 * stage_mean / max_mean):.1f}%'></span>",
            }
        )
    return render_table(
        table_rows,
        [("stage", "Stage"), ("events", "Events"), ("mean_s", "Mean"), ("median_s", "Median"), ("bar", "Mean duration")],
        raw_columns={"bar"},
    )


def timeline_svg(events: list[dict[str, Any]], *, width: int = 900, height: int = 92) -> str:
    if not events:
        return ""
    colors = {
        "hop_plan": "#2f6fed",
        "doc_choose": "#4c78a8",
        "doc_read": "#f58518",
        "search": "#54a24b",
        "hop_resolve": "#b279a2",
        "error": "#b13d2e",
    }
    parsed: list[tuple[float, float, str]] = []
    for event in events:
        start = as_float(event.get("start_s")) or 0.0
        end = as_float(event.get("end_s"))
        duration = as_float(event.get("duration_s"))
        if end is None:
            end = start + (duration or 0.0)
        parsed.append((start, max(start, end), str(event.get("stage") or "unknown")))
    if not parsed:
        return ""
    min_start = min(item[0] for item in parsed)
    max_end = max(item[1] for item in parsed)
    span = max(1e-6, max_end - min_start)
    lane_for: dict[str, int] = {}
    rects: list[str] = []
    for start, end, stage in sorted(parsed, key=lambda item: item[0]):
        if stage not in lane_for:
            lane_for[stage] = len(lane_for)
        lane = lane_for[stage] % 5
        x = 70 + (start - min_start) / span * (width - 100)
        rect_width = max(2.0, (end - start) / span * (width - 100))
        y = 18 + lane * 13
        color = colors.get(stage, "#65706c")
        rects.append(
            f"<rect x='{x:.1f}' y='{y}' width='{rect_width:.1f}' height='9' rx='3' fill='{color}'><title>{esc(stage)} {end - start:.1f}s</title></rect>"
        )
    labels = "".join(
        f"<text x='8' y='{25 + lane * 13}'>{esc(stage[:12])}</text>"
        for stage, lane in sorted(lane_for.items(), key=lambda item: item[1])[:5]
    )
    return (
        f"<svg class='timeline-svg' viewBox='0 0 {width} {height}' role='img' aria-label='Timeline overview'>"
        f"<line x1='70' y1='{height - 18}' x2='{width - 30}' y2='{height - 18}' stroke='#d8dee8'/>"
        f"<text x='70' y='{height - 4}'>0s</text><text x='{width - 82}' y='{height - 4}'>{span:.0f}s</text>"
        f"{labels}{''.join(rects)}</svg>"
    )


def select_time_slices(rows: list[dict[str, str]], size: int) -> list[tuple[str, list[dict[str, str]]]]:
    ordered = [row for row in rows if as_float(row.get("completed_at_s")) is not None]
    if not ordered:
        return []
    ordered = sorted(ordered, key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
    size = max(1, min(size, len(ordered)))
    middle_start = max(0, (len(ordered) // 2) - (size // 2))

    def window(label: str, start: int, end: int) -> tuple[str, list[dict[str, str]]]:
        return (f"{label} ({start}-{end - 1})", ordered[start:end])

    return [
        window("Early", 0, size),
        window("Middle", middle_start, middle_start + size),
        window("Late", len(ordered) - size, len(ordered)),
    ]


def rolling_means(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def apply_plot_style(ax: Any) -> None:
    ax.grid(True, axis="y", color="#d8dee8", alpha=0.65, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


def save_temporal_distribution_plot(path: Path, slices: list[tuple[str, list[dict[str, str]]]]) -> None:
    metrics = [
        ("reward", "Reward"),
        ("searches_total", "Searches"),
        ("docs_read", "Docs read"),
        ("llm_calls_total", "LLM calls"),
        ("total_tokens", "Total tokens"),
        ("duration_s", "Duration (s)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    colors = ["#2f6fed", "#7c8aa5", "#f58518"]
    for ax, (key, title) in zip(axes.flat, metrics):
        data = [values_for(rows, key) for _, rows in slices]
        labels = [name for name, _ in slices]
        box = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)
        for median_line in box["medians"]:
            median_line.set_color("#111827")
            median_line.set_linewidth(1.6)
        for mean_line in box["means"]:
            mean_line.set_markerfacecolor("#111827")
            mean_line.set_markeredgecolor("#111827")
        ax.set_title(title)
        apply_plot_style(ax)
    fig.suptitle("Early / Middle / Late Rollout Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_temporal_source_usage_plot(path: Path, slices: list[tuple[str, list[dict[str, str]]]]) -> None:
    metrics = [
        ("web_searches", "Web searches"),
        ("local_searches", "Local searches"),
        ("web_docs_read", "Web docs read"),
        ("local_docs_read", "Local docs read"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    colors = ["#2f6fed", "#7c8aa5", "#f58518"]
    for ax, (key, title) in zip(axes.flat, metrics):
        data = [values_for(rows, key) for _, rows in slices]
        labels = [name for name, _ in slices]
        box = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)
        for median_line in box["medians"]:
            median_line.set_color("#111827")
            median_line.set_linewidth(1.6)
        ax.set_title(title)
        apply_plot_style(ax)
    fig.suptitle("Source Usage By Time Window", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_temporal_search_docs_plot(path: Path, slices: list[tuple[str, list[dict[str, str]]]]) -> None:
    all_searches = [value for _, rows in slices for value in values_for(rows, "searches_total")]
    all_docs = [value for _, rows in slices for value in values_for(rows, "docs_read")]
    if not all_searches or not all_docs:
        return
    x_max = max(all_searches + [1.0])
    y_max = max(all_docs + [1.0])
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), sharex=True, sharey=True)
    scatter = None
    for ax, (label, rows) in zip(axes, slices):
        points = [
            (searches, docs, as_float(row.get("reward")) or 0.0)
            for row in rows
            if (searches := as_float(row.get("searches_total"))) is not None
            and (docs := as_float(row.get("docs_read"))) is not None
        ]
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        rewards = [item[2] for item in points]
        scatter = ax.scatter(
            xs,
            ys,
            c=rewards,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.set_title(label)
        ax.set_xlabel("Search actions")
        ax.set_xlim(-0.5, x_max + 1.0)
        ax.set_ylim(-0.5, y_max + 1.0)
        apply_plot_style(ax)
    axes[0].set_ylabel("Selected docs read")
    if scatter is not None:
        # Keep the reward scale outside the panels; automatic placement can
        # overlap the late-window subplot when the image is embedded.
        cbar_ax = fig.add_axes([0.935, 0.18, 0.016, 0.64])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("Reward")
    fig.suptitle("Searches vs Docs Read by Time Window", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.91, top=0.82, bottom=0.14, wspace=0.18)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def save_temporal_source_distribution_plot(path: Path, slices: list[tuple[str, list[dict[str, str]]]]) -> None:
    search_values = [value for _, rows in slices for key in ("web_searches", "local_searches") for value in values_for(rows, key)]
    read_values = [value for _, rows in slices for key in ("web_docs_read", "local_docs_read") for value in values_for(rows, key)]
    if not search_values or not read_values:
        return
    search_bins = range(0, int(max(search_values + [1.0])) + 3)
    read_bins = range(0, int(max(read_values + [1.0])) + 3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharey="row")
    for col, (label, rows) in enumerate(slices):
        ax = axes[0, col]
        ax.hist(values_for(rows, "web_searches"), bins=search_bins, alpha=0.62, label="web", color="#2f6fed")
        ax.hist(values_for(rows, "local_searches"), bins=search_bins, alpha=0.62, label="local", color="#f58518")
        ax.set_title(label)
        ax.set_xlabel("Searches / rollout")
        apply_plot_style(ax)
        if col == 0:
            ax.set_ylabel("Rollouts")
            ax.legend()

        ax = axes[1, col]
        ax.hist(values_for(rows, "web_docs_read"), bins=read_bins, alpha=0.62, label="web", color="#2f6fed")
        ax.hist(values_for(rows, "local_docs_read"), bins=read_bins, alpha=0.62, label="local", color="#f58518")
        ax.set_xlabel("Docs read / rollout")
        apply_plot_style(ax)
        if col == 0:
            ax.set_ylabel("Rollouts")
            ax.legend()
    axes[0, 0].text(-0.18, 0.5, "Search actions", transform=axes[0, 0].transAxes, rotation=90, va="center", fontweight="bold")
    axes[1, 0].text(-0.18, 0.5, "Docs read", transform=axes[1, 0].transAxes, rotation=90, va="center", fontweight="bold")
    fig.suptitle("Web vs Local Source Usage Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_temporal_source_balance_plot(path: Path, slices: list[tuple[str, list[dict[str, str]]]]) -> None:
    labels = [label for label, _ in slices]
    x = list(range(len(slices)))
    width = 0.36
    search_web = [mean_field(rows, "web_searches") or 0.0 for _, rows in slices]
    search_local = [mean_field(rows, "local_searches") or 0.0 for _, rows in slices]
    docs_web = [mean_field(rows, "web_docs_read") or 0.0 for _, rows in slices]
    docs_local = [mean_field(rows, "local_docs_read") or 0.0 for _, rows in slices]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
    axes[0].bar([i - width / 2 for i in x], search_web, width, label="web", color="#2f6fed")
    axes[0].bar([i + width / 2 for i in x], search_local, width, label="local", color="#f58518")
    axes[0].set_title("Mean searches / rollout")
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].legend()
    apply_plot_style(axes[0])

    axes[1].bar([i - width / 2 for i in x], docs_web, width, label="web", color="#2f6fed")
    axes[1].bar([i + width / 2 for i in x], docs_local, width, label="local", color="#f58518")
    axes[1].set_title("Mean docs read / rollout")
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].legend()
    apply_plot_style(axes[1])

    fig.suptitle("Mean Source Usage by Time Window", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_temporal_source_hop_alignment_plot(path: Path, slices: list[tuple[str, list[dict[str, str]]]]) -> None:
    if not all(any(key in row for _, rows in slices for row in rows) for key in ("web_hop_count", "local_hop_count")):
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharey=False)
    for col, (label, rows) in enumerate(slices):
        for row_index, (hop_key, usage_key, title) in enumerate(
            [
                ("web_hop_count", "web_searches", "Web hops vs web searches"),
                ("local_hop_count", "local_searches", "Local hops vs local searches"),
            ]
        ):
            ax = axes[row_index, col]
            cell_counts: dict[tuple[int, int], int] = {}
            cell_rewards: dict[tuple[int, int], list[float]] = {}
            for row in rows:
                hop_count = as_float(row.get(hop_key))
                usage = as_float(row.get(usage_key))
                if hop_count is None or usage is None:
                    continue
                coords = (int(hop_count), int(usage))
                cell_counts[coords] = cell_counts.get(coords, 0) + 1
                cell_rewards.setdefault(coords, []).append(as_float(row.get("reward")) or 0.0)
            if cell_counts:
                max_count = max(cell_counts.values())
                coords = sorted(cell_counts)
                xs = [item[0] for item in coords]
                ys = [item[1] for item in coords]
                sizes = [55.0 + 650.0 * math.sqrt(cell_counts[item] / max_count) for item in coords]
                colors = [mean(cell_rewards[item]) for item in coords]
                ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.82, edgecolors="#1f2937", linewidths=0.5)
                for coords_item in sorted(coords, key=lambda item: cell_counts[item], reverse=True)[:3]:
                    ax.text(coords_item[0], coords_item[1], str(cell_counts[coords_item]), ha="center", va="center", fontsize=7, fontweight="bold")
            ax.set_title(label if row_index == 0 else title)
            ax.set_xlabel(hop_key.replace("_", " "))
            ax.set_ylabel(usage_key.replace("_", " "))
            apply_plot_style(ax)
    fig.suptitle("Hop Count vs Matching Source Usage (bubble=count, color=reward)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_temporal_moving_average_plot(path: Path, rows: list[dict[str, str]], window: int) -> None:
    ordered = sorted(rows, key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
    metrics = [
        ("reward", "Reward"),
        ("searches_total", "Searches"),
        ("docs_read", "Docs read"),
        ("llm_calls_total", "LLM calls"),
        ("total_tokens", "Total tokens"),
        ("duration_s", "Duration (s)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 9.2), sharex=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        pairs = [(idx + 1, value) for idx, row in enumerate(ordered) if (value := as_float(row.get(key))) is not None]
        if pairs:
            x_values = [idx for idx, _ in pairs]
            y_values = rolling_means([value for _, value in pairs], max(1, window))
            ax.plot(x_values, y_values, color="#2f6fed", linewidth=1.8)
        ax.set_title(title)
        apply_plot_style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Completed rollout index")
    fig.suptitle(f"Moving Averages Over Time (window={window})", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def write_temporal_plots(analysis_dir: Path, rows: list[dict[str, str]], slices: list[tuple[str, list[dict[str, str]]]], moving_average_window: int) -> None:
    if not rows or not slices:
        return
    save_temporal_distribution_plot(analysis_dir / "temporal_distributions.png", slices)
    save_temporal_search_docs_plot(analysis_dir / "temporal_searches_vs_docs.png", slices)
    save_temporal_source_distribution_plot(analysis_dir / "temporal_source_distributions.png", slices)
    save_temporal_source_balance_plot(analysis_dir / "temporal_source_balance.png", slices)
    save_temporal_source_hop_alignment_plot(analysis_dir / "temporal_source_hop_alignment.png", slices)
    save_temporal_source_usage_plot(analysis_dir / "temporal_source_usage.png", slices)
    save_temporal_moving_average_plot(analysis_dir / "temporal_moving_averages.png", rows, moving_average_window)


def composition_label(row: dict[str, Any]) -> str:
    local_hops = as_float(row.get("local_hop_count"))
    web_hops = as_float(row.get("web_hop_count"))
    if local_hops is None or web_hops is None:
        pattern = str(row.get("hop_pattern") or "")
        if pattern:
            local_hops = float(pattern.count("L"))
            web_hops = float(pattern.count("W"))
    if local_hops is None or web_hops is None:
        return "unknown"
    return f"L{int(local_hops)}/W{int(web_hops)}"


def source_fraction(row: dict[str, Any], numerator_key: str, denominator_keys: tuple[str, str]) -> float | None:
    numerator = as_float(row.get(numerator_key))
    denominator = sum(as_float(row.get(key)) or 0.0 for key in denominator_keys)
    if numerator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def add_count_labels(ax: Any, coords: list[tuple[int, int]], counts: dict[tuple[int, int], int], limit: int = 4) -> None:
    if not coords:
        return
    max_count = max(counts.values())
    for coord in sorted(coords, key=lambda item: counts[item], reverse=True)[:limit]:
        ax.text(
            coord[0],
            coord[1],
            str(counts[coord]),
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if counts[coord] > 0.35 * max_count else "#111827",
            fontweight="bold",
        )


def plot_bubble_cells(ax: Any, rows: list[dict[str, str]], x_key: str, y_key: str, title: str, x_label: str, y_label: str) -> Any:
    counts: dict[tuple[int, int], int] = {}
    rewards: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        x_value = as_float(row.get(x_key))
        y_value = as_float(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        coord = (int(x_value), int(y_value))
        counts[coord] = counts.get(coord, 0) + 1
        rewards[coord].append(as_float(row.get("reward")) or 0.0)
    coords = sorted(counts)
    if not coords:
        ax.set_title(title)
        return None
    max_count = max(counts.values())
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    sizes = [45.0 + 760.0 * math.sqrt(counts[coord] / max_count) for coord in coords]
    colors = [mean(rewards[coord]) for coord in coords]
    scatter = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=colors,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.84,
        edgecolors="#1f2937",
        linewidths=0.55,
    )
    add_count_labels(ax, coords, counts)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.45, max(xs + [0]) + 0.75)
    ax.set_ylim(-0.45, max(ys + [0]) + 1.0)
    ax.set_xticks(range(0, max(xs + [0]) + 1))
    apply_plot_style(ax)
    return scatter


def save_question_type_retrieval_alignment_plot(path: Path, rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    scatter = None
    specs = [
        ("local_hop_count", "local_docs_read", "Local hops vs local docs read", "Local hops in question", "Local docs read"),
        ("local_hop_count", "local_searches", "Local hops vs local searches", "Local hops in question", "Local searches"),
        ("web_hop_count", "web_docs_read", "Web hops vs web docs read", "Web hops in question", "Web docs read"),
        ("web_hop_count", "web_searches", "Web hops vs web searches", "Web hops in question", "Web searches"),
    ]
    for ax, spec in zip(axes.flat, specs):
        scatter = plot_bubble_cells(ax, rows, *spec) or scatter
    if scatter is not None:
        cax = fig.add_axes([0.88, 0.18, 0.018, 0.66])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Mean reward")
    fig.suptitle("Retrieval Alignment With Question Source Requirements", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.08, wspace=0.24, hspace=0.32)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def composition_grid(rows: list[dict[str, str]], metric_key: str, reducer: str = "mean") -> tuple[list[int], list[int], list[list[float | None]], list[list[int]]]:
    local_values = sorted({int(value) for row in rows if (value := as_float(row.get("local_hop_count"))) is not None})
    web_values = sorted({int(value) for row in rows if (value := as_float(row.get("web_hop_count"))) is not None})
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        local_hops = as_float(row.get("local_hop_count"))
        web_hops = as_float(row.get("web_hop_count"))
        metric = as_float(row.get(metric_key))
        if local_hops is None or web_hops is None:
            continue
        coord = (int(local_hops), int(web_hops))
        counts[coord] += 1
        if metric is not None:
            grouped[coord].append(metric)
    values: list[list[float | None]] = []
    count_grid: list[list[int]] = []
    for local_hops in local_values:
        value_row: list[float | None] = []
        count_row: list[int] = []
        for web_hops in web_values:
            bucket = grouped.get((local_hops, web_hops), [])
            if reducer == "count":
                value_row.append(float(counts.get((local_hops, web_hops), 0)))
            else:
                value_row.append(mean(bucket) if bucket else None)
            count_row.append(counts.get((local_hops, web_hops), 0))
        values.append(value_row)
        count_grid.append(count_row)
    return local_values, web_values, values, count_grid


def save_question_type_composition_heatmaps(path: Path, rows: list[dict[str, str]]) -> None:
    specs = [
        ("reward", "Mean reward", "mean", "{:.2f}"),
        ("total_tokens", "Mean total tokens", "mean", "{:.0f}"),
        ("duration_s", "Mean duration (s)", "mean", "{:.0f}"),
        ("reward", "Rollout count", "count", "{:.0f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2))
    for ax, (metric, title, reducer, fmt) in zip(axes.flat, specs):
        local_values, web_values, values, counts = composition_grid(rows, metric, reducer)
        if not local_values or not web_values:
            ax.set_title(title)
            continue
        matrix = [[float("nan") if value is None else value for value in row] for row in values]
        finite_values = [value for row in matrix for value in row if math.isfinite(value)]
        color_threshold = 0.55 * max(finite_values) if finite_values else 0.0
        image = ax.imshow(matrix, cmap="viridis" if reducer != "count" else "Blues", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(web_values)), [str(value) for value in web_values])
        ax.set_yticks(range(len(local_values)), [str(value) for value in local_values])
        ax.set_xlabel("Web hops")
        ax.set_ylabel("Local hops")
        for i, local_hops in enumerate(local_values):
            for j, web_hops in enumerate(web_values):
                value = values[i][j]
                count = counts[i][j]
                if value is None or count == 0:
                    continue
                label = fmt.format(value)
                if reducer != "count":
                    label = f"{label}\nn={count}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if value and value > color_threshold else "#111827")
        cbar = fig.colorbar(image, ax=ax, shrink=0.82)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Performance And Cost By Question Hop Composition", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_question_type_source_fraction_plot(path: Path, rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharex=True, sharey=True)
    specs = [
        ("local_searches", ("local_searches", "web_searches"), "Local search fraction"),
        ("local_docs_read", ("local_docs_read", "web_docs_read"), "Local doc-read fraction"),
    ]
    for ax, (numerator, denominator_keys, title) in zip(axes, specs):
        cell_counts: dict[tuple[float, float], int] = defaultdict(int)
        cell_rewards: dict[tuple[float, float], list[float]] = defaultdict(list)
        for row in rows:
            local_hops = as_float(row.get("local_hop_count"))
            web_hops = as_float(row.get("web_hop_count"))
            if local_hops is None or web_hops is None or local_hops + web_hops <= 0:
                continue
            x_value = round(local_hops / (local_hops + web_hops), 2)
            y_raw = source_fraction(row, numerator, denominator_keys)
            if y_raw is None:
                continue
            y_value = round(y_raw, 2)
            coord = (x_value, y_value)
            cell_counts[coord] += 1
            cell_rewards[coord].append(as_float(row.get("reward")) or 0.0)
        coords = sorted(cell_counts)
        if coords:
            max_count = max(cell_counts.values())
            xs = [coord[0] for coord in coords]
            ys = [coord[1] for coord in coords]
            sizes = [40.0 + 720.0 * math.sqrt(cell_counts[coord] / max_count) for coord in coords]
            colors = [mean(cell_rewards[coord]) for coord in coords]
            scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.84, edgecolors="#1f2937", linewidths=0.5)
            for coord in sorted(coords, key=lambda item: cell_counts[item], reverse=True)[:5]:
                ax.text(coord[0], coord[1], str(cell_counts[coord]), ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2, label="ideal source match")
        ax.set_title(title)
        ax.set_xlabel("Local-hop fraction in question")
        ax.set_ylabel("Local retrieval fraction")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.legend(loc="upper left", frameon=False)
        apply_plot_style(ax)
    if "scatter" in locals():
        cax = fig.add_axes([0.88, 0.18, 0.018, 0.62])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Mean reward")
    fig.suptitle("Does Retrieval Source Mix Track Question Source Mix?", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.84, top=0.84, bottom=0.14, wspace=0.22)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def grouped_by_label(rows: list[dict[str, str]], label_key: str, limit: int = 14) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if label_key == "composition":
            label = composition_label(row)
        else:
            label = str(row.get(label_key) or composition_label(row))
        grouped[label].append(row)
    return sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))[:limit]


def save_question_type_pattern_performance_plot(path: Path, rows: list[dict[str, str]]) -> None:
    groups = grouped_by_label(rows, "hop_pattern", limit=14)
    if not groups:
        return
    labels = [label for label, _ in groups]
    y = list(range(len(groups)))
    metrics = [
        ("reward", "Mean reward", lambda bucket: mean_field(bucket, "reward") or 0.0),
        ("searches_total", "Mean searches", lambda bucket: mean_field(bucket, "searches_total") or 0.0),
        ("docs_read", "Mean docs read", lambda bucket: mean_field(bucket, "docs_read") or 0.0),
        ("total_tokens", "Median total tokens", lambda bucket: median_field(bucket, "total_tokens") or 0.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8), sharey=True)
    for ax, (_, title, fn) in zip(axes.flat, metrics):
        values = [fn(bucket) for _, bucket in groups]
        ax.barh(y, values, color="#2f6fed", alpha=0.78)
        ax.set_title(title)
        ax.set_yticks(y, labels)
        apply_plot_style(ax)
        for idx, value in enumerate(values):
            ax.text(value, idx, f" n={len(groups[idx][1])}", va="center", fontsize=7.8, color="#334155")
    axes.flat[0].invert_yaxis()
    fig.suptitle("Top Hop Patterns: Performance, Retrieval, And Token Cost", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_question_type_local_hop_sensitivity_plot(path: Path, rows: list[dict[str, str]]) -> None:
    local_counts = sorted({int(value) for row in rows if (value := as_float(row.get("local_hop_count"))) is not None})
    if not local_counts:
        return
    metrics = [
        ("reward", "Reward"),
        ("local_docs_read", "Local docs read"),
        ("local_searches", "Local searches"),
        ("web_docs_read", "Web docs read"),
        ("total_tokens", "Total tokens"),
        ("duration_s", "Duration (s)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.8))
    labels = [str(value) for value in local_counts]
    for ax, (key, title) in zip(axes.flat, metrics):
        data = [
            [as_float(row.get(key)) or 0.0 for row in rows if int(as_float(row.get("local_hop_count")) or -1) == local_count]
            for local_count in local_counts
        ]
        box = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("#2f6fed")
            patch.set_alpha(0.18)
            patch.set_edgecolor("#2f6fed")
        for median_line in box["medians"]:
            median_line.set_color("#111827")
            median_line.set_linewidth(1.5)
        ax.set_title(title)
        ax.set_xlabel("Local hops in question")
        apply_plot_style(ax)
    fig.suptitle("Behavior Sensitivity To Number Of Local Hops", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def rate_for(rows: list[dict[str, str]], predicate: Any) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if predicate(row)) / len(rows)


def save_question_type_source_mismatch_plot(path: Path, rows: list[dict[str, str]]) -> None:
    local_counts = sorted({int(value) for row in rows if (value := as_float(row.get("local_hop_count"))) is not None and value > 0})
    web_counts = sorted({int(value) for row in rows if (value := as_float(row.get("web_hop_count"))) is not None and value > 0})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    if local_counts:
        labels = [str(value) for value in local_counts]
        no_search = []
        no_docs = []
        for count in local_counts:
            bucket = [row for row in rows if int(as_float(row.get("local_hop_count")) or -1) == count]
            no_search.append(rate_for(bucket, lambda row: (as_float(row.get("local_searches")) or 0.0) <= 0.0) or 0.0)
            no_docs.append(rate_for(bucket, lambda row: (as_float(row.get("local_docs_read")) or 0.0) <= 0.0) or 0.0)
        x = list(range(len(labels)))
        width = 0.36
        axes[0].bar([i - width / 2 for i in x], no_search, width, label="zero local searches", color="#f58518")
        axes[0].bar([i + width / 2 for i in x], no_docs, width, label="zero local docs read", color="#2f6fed")
        for i, value in enumerate(no_search):
            axes[0].text(i - width / 2, min(0.98, value + 0.015), f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=8)
        for i, value in enumerate(no_docs):
            axes[0].text(i + width / 2, min(0.98, value + 0.015), f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=8)
        axes[0].set_xticks(x, labels)
        axes[0].set_xlabel("Local hops in question")
        axes[0].set_title("Missing local retrieval when local hops exist")
        axes[0].legend(frameon=False)
    if web_counts:
        labels = [str(value) for value in web_counts]
        no_search = []
        no_docs = []
        for count in web_counts:
            bucket = [row for row in rows if int(as_float(row.get("web_hop_count")) or -1) == count]
            no_search.append(rate_for(bucket, lambda row: (as_float(row.get("web_searches")) or 0.0) <= 0.0) or 0.0)
            no_docs.append(rate_for(bucket, lambda row: (as_float(row.get("web_docs_read")) or 0.0) <= 0.0) or 0.0)
        x = list(range(len(labels)))
        width = 0.36
        axes[1].bar([i - width / 2 for i in x], no_search, width, label="zero web searches", color="#f58518")
        axes[1].bar([i + width / 2 for i in x], no_docs, width, label="zero web docs read", color="#2f6fed")
        for i, value in enumerate(no_search):
            axes[1].text(i - width / 2, min(0.98, value + 0.015), f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=8)
        for i, value in enumerate(no_docs):
            axes[1].text(i + width / 2, min(0.98, value + 0.015), f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=8)
        axes[1].set_xticks(x, labels)
        axes[1].set_xlabel("Web hops in question")
        axes[1].set_title("Missing web retrieval when web hops exist")
        axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction of rollouts")
        ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
        apply_plot_style(ax)
    fig.suptitle("Source Mismatch / Missing Retrieval Checks", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_question_type_plots(analysis_dir: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    save_question_type_retrieval_alignment_plot(analysis_dir / "question_type_retrieval_alignment.png", rows)
    save_question_type_composition_heatmaps(analysis_dir / "question_type_composition_heatmaps.png", rows)
    save_question_type_source_fraction_plot(analysis_dir / "question_type_source_fraction.png", rows)
    save_question_type_pattern_performance_plot(analysis_dir / "question_type_pattern_performance.png", rows)
    save_question_type_local_hop_sensitivity_plot(analysis_dir / "question_type_local_hop_sensitivity.png", rows)
    save_question_type_source_mismatch_plot(analysis_dir / "question_type_source_mismatch.png", rows)


def render_question_type_table(rows: list[dict[str, str]]) -> str:
    groups = grouped_by_label(rows, "composition", limit=32)
    table_rows: list[dict[str, Any]] = []
    for label, bucket in groups:
        rewards = values_for(bucket, "reward")
        table_rows.append(
            {
                "type": label,
                "n": fmt_num(len(bucket), 0),
                "reward": fmt_num(mean(rewards) if rewards else None, 3),
                "solved": fmt_pct(sum(1 for value in rewards if value >= 0.999) / len(rewards) if rewards else None, 1),
                "local_docs": fmt_num(mean_field(bucket, "local_docs_read"), 1),
                "web_docs": fmt_num(mean_field(bucket, "web_docs_read"), 1),
                "local_search": fmt_num(mean_field(bucket, "local_searches"), 1),
                "web_search": fmt_num(mean_field(bucket, "web_searches"), 1),
                "tokens": fmt_num(median_field(bucket, "total_tokens"), 0),
                "duration": f"{fmt_num(median_field(bucket, 'duration_s'), 1)}s",
            }
        )
    return render_table(
        table_rows,
        [
            ("type", "Question type"),
            ("n", "n"),
            ("reward", "Mean reward"),
            ("solved", "Solved"),
            ("local_docs", "Mean local docs"),
            ("web_docs", "Mean web docs"),
            ("local_search", "Mean local searches"),
            ("web_search", "Mean web searches"),
            ("tokens", "Median tokens"),
            ("duration", "Median duration"),
        ],
    )


def render_question_type_section(analysis_dir: Path, rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p class='note'>No full-rollout trend rows were found for question-type analysis.</p>"
    write_question_type_plots(analysis_dir, rows)
    return (
        "<p class='note'>These plots group rollouts by the source requirements of the question chain. "
        "They are designed to answer whether the policy increases local retrieval when local hops exist, "
        "whether source mix tracks source mix in the question, and which hop patterns are expensive or low reward.</p>"
        f"<section class='plots'>{render_plots(analysis_dir, QUESTION_TYPE_PLOTS)}</section>"
        f"{render_question_type_table(rows)}"
    )


def render_temporal_table(slices: list[tuple[str, list[dict[str, str]]]]) -> str:
    table_rows: list[dict[str, Any]] = []
    for key, label, suffix in TREND_METRICS:
        digits = 3 if key == "reward" else 1
        for slice_name, slice_rows in slices:
            values = values_for(slice_rows, key)
            table_rows.append(
                {
                    "metric": label,
                    "window": slice_name,
                    "n": fmt_num(len(values), 0),
                    "mean": fmt_metric(mean(values) if values else None, suffix, digits),
                    "median": fmt_metric(median(values) if values else None, suffix, digits),
                    "p10": fmt_metric(percentile(values, 10), suffix, digits),
                    "p90": fmt_metric(percentile(values, 90), suffix, digits),
                    "min": fmt_metric(min(values) if values else None, suffix, digits),
                    "max": fmt_metric(max(values) if values else None, suffix, digits),
                }
            )
    return render_table(
        table_rows,
        [
            ("metric", "Metric"),
            ("window", "Window"),
            ("n", "n"),
            ("mean", "Mean"),
            ("median", "Median"),
            ("p10", "P10"),
            ("p90", "P90"),
            ("min", "Min"),
            ("max", "Max"),
        ],
    )


def fmt_metric(value: float | None, suffix: str, digits: int) -> str:
    text = fmt_num(value, digits)
    if suffix and text != "n/a":
        return f"{text}{suffix}"
    return text


def render_temporal_section(analysis_dir: Path, rows: list[dict[str, str]], window_size: int, moving_average_window: int) -> str:
    slices = select_time_slices(rows, window_size)
    if not slices:
        return "<p class='note'>No chronological rollout trend rows were found.</p>"
    write_temporal_plots(analysis_dir, rows, slices, moving_average_window)
    table = render_temporal_table(slices)
    plot_cards = render_plots(analysis_dir, TEMPORAL_PLOTS)
    return (
        f"<p class='note'>These are chronological {window_size}-rollout windows: first available, centered middle, and most recent. "
        f"Moving-average plots use a {moving_average_window}-rollout window. Click any plot to open it at full size.</p>"
        f"<section class='plots'>{plot_cards}</section>{table}"
    )


def decile_means(rows: list[dict[str, str]], key: str, bins: int = 10) -> list[float]:
    ordered = [row for row in rows if as_float(row.get("completed_at_s")) is not None and as_float(row.get(key)) is not None]
    ordered = sorted(ordered, key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
    if not ordered:
        return []
    values: list[float] = []
    for index in range(bins):
        start = round(index * len(ordered) / bins)
        end = round((index + 1) * len(ordered) / bins)
        chunk = ordered[start:end]
        if chunk:
            values.append(mean(as_float(row.get(key)) or 0.0 for row in chunk))
    return values


def sparkline_svg(values: list[float], *, suffix: str = "", width: int = 230, height: int = 54) -> str:
    if not values:
        return ""
    low = min(values)
    high = max(values)
    span = max(1e-9, high - low)
    points = []
    for index, value in enumerate(values):
        x = 8 + index * (width - 16) / max(1, len(values) - 1)
        y = height - 10 - ((value - low) / span * (height - 22) if span else 0)
        points.append(f"{x:.1f},{y:.1f}")
    start = fmt_num(values[0], 2)
    end = fmt_num(values[-1], 2)
    if suffix:
        start += suffix
        end += suffix
    return (
        f"<svg class='spark-svg' viewBox='0 0 {width} {height}' role='img'>"
        f"<polyline points='{' '.join(points)}' fill='none' stroke='#176c5f' stroke-width='2.5'/>"
        f"<circle cx='{points[-1].split(',')[0]}' cy='{points[-1].split(',')[1]}' r='3' fill='#a35620'/>"
        f"<text x='8' y='{height - 2}'>{esc(start)}</text><text x='{width - 74}' y='{height - 2}'>{esc(end)}</text>"
        "</svg>"
    )


def render_examples(
    sample_dirs: list[Path],
    row_by_id: dict[str, dict[str, str]],
    problem_by_chain: dict[str, dict[str, Any]],
    max_events: int,
) -> str:
    examples: list[str] = []
    for sample_dir in sample_dirs:
        rollout_id = sample_dir.name
        trace = load_json(sample_dir / "trace.json", {})
        row = example_row_from_trace(sample_dir, row_by_id, trace)
        problem = problem_by_chain.get(str(row.get("chain_id") or ""))
        events = load_json(sample_dir / "timeline_events.json", [])
        event_rows, stage_totals = summarize_events(events, max_events)
        report = read_text(sample_dir / "report.md", limit=2600)
        reward = row.get("reward")
        hop_count = row.get("hop_count") or row.get("total_hops") or (problem or {}).get("n_hops")
        title_parts = [rollout_id]
        if reward not in (None, ""):
            title_parts.append(f"reward={fmt_num(reward, 3)}")
        if hop_count not in (None, ""):
            title_parts.append(f"hops={hop_count}")
        title = " · ".join(map(str, title_parts))

        meta: list[tuple[str, Any]] = []
        for label, value in [
            ("Task", row.get("task_id")),
            ("Company", (problem or {}).get("company")),
            ("Pattern", row.get("hop_pattern") or (problem or {}).get("pattern")),
            ("Iterations", row.get("iterations_used")),
            ("Searches", row.get("searches_total")),
            ("Docs read", row.get("docs_read")),
            ("LLM calls", row.get("llm_calls_total")),
            ("Parse errors", row.get("parse_errors_total")),
            ("Predicted final", row.get("predicted_final_answer")),
        ]:
            if value not in (None, ""):
                meta.append((label, value))
        if (duration := as_float(row.get("duration_s"))) is not None:
            meta.append(("Duration", f"{fmt_num(duration, 1)}s"))
        if (tokens := as_float(row.get("total_tokens"))) is not None:
            meta.append(("Total tokens", fmt_num(tokens, 0)))
        final_answers = accepted_answers_for_final(problem, row.get("ground_truth_final_answer"))
        if final_answers:
            meta.append(("Gold final answer(s)", format_answer_list(final_answers, 520)))
        meta_html = "".join(
            f"<div><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"
            for label, value in meta
            if value not in (None, "")
        )
        totals_html = render_table(
            [{"stage": item["stage"], "duration_s": f"{item['duration_s']:.1f}s"} for item in stage_totals],
            [("stage", "Stage"), ("duration_s", "Total time")],
        )
        events_html = render_table(
            [
                {
                    "stage": item["stage"],
                    "start_s": f"{item['start_s']:.1f}s",
                    "duration_s": f"{item['duration_s']:.1f}s",
                    "label": item["label"],
                    "detail": item["detail"],
                }
                for item in event_rows
            ],
            [
                ("stage", "Stage"),
                ("start_s", "Start"),
                ("duration_s", "Dur"),
                ("label", "Label"),
                ("detail", "Detail"),
            ],
        )
        examples.append(
            f"""
            <section class="example">
              <h3>{esc(title)}</h3>
              <div class="meta">{meta_html}</div>
              {timeline_svg(events)}
              <details open>
                <summary>Report preview</summary>
                <pre>{esc(report)}</pre>
              </details>
              <details open>
                <summary>Hop states</summary>
                {render_hop_state_table(trace, row, problem)}
              </details>
              <details open>
                <summary>Action and search timeline</summary>
                {render_action_cards(trace)}
              </details>
              <details>
                <summary>Stage time totals</summary>
                {totals_html}
              </details>
              <details>
                <summary>Timeline preview</summary>
                {events_html}
              </details>
            </section>
            """
        )
    return "\n".join(examples)


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else analysis_dir / "slim_viewer.html"
    summary = load_json(analysis_dir / "summary.json", {})
    row_by_id = load_rollout_rows(analysis_dir / "rollouts_sample.csv")
    trend_rows = load_trend_rows(analysis_dir, row_by_id)
    sample_dirs = choose_sample_dirs(analysis_dir / "sample_artifacts", row_by_id, args.max_samples)
    problem_by_chain = load_problem_index(Path(args.dataset_path).expanduser().resolve())

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Privacy HopQA Slim Viewer</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --ink: #18212b;
      --muted: #637083;
      --card: #ffffff;
      --line: #d8dee8;
      --accent: #2f6fed;
      --accent-2: #4c78a8;
      --accent-soft: #eef4ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #ffffff 0%, var(--bg) 320px);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 36px clamp(20px, 5vw, 72px) 24px;
      border-bottom: 1px solid var(--line);
      background: radial-gradient(circle at top left, #ffffff 0%, #eef4fb 52%, #f8f7f2 100%);
    }}
    h1 {{ margin: 0; font-size: clamp(32px, 4vw, 52px); letter-spacing: -0.035em; }}
    h2 {{ margin: 36px 0 14px; font-size: 26px; }}
    h3 {{ margin: 0 0 10px; font-size: 18px; }}
    h4 {{ margin: 16px 0 10px; font-size: 15px; }}
    p {{ max-width: 76ch; line-height: 1.55; color: var(--muted); }}
    main {{ padding: 0 clamp(20px, 5vw, 72px) 48px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .stat, .plot, .example {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(24, 33, 43, .05);
    }}
    .stat {{ padding: 14px 16px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 4px; font-size: 24px; font-weight: 700; color: var(--accent); }}
    .mini-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 12px 0;
    }}
    .mini-stat {{ padding: 10px 12px; border: 1px solid var(--line); border-radius: 14px; background: #fbfcff; }}
    .mini-stat span {{ display: block; color: var(--muted); font-size: 12px; }}
    .mini-stat strong {{ display: block; margin-top: 4px; color: var(--accent); font-size: 20px; }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .plot {{ padding: 16px; }}
    .plot-open {{
      display: block;
      width: 100%;
      padding: 0;
      border: 0;
      background: transparent;
      cursor: zoom-in;
    }}
    .plot img {{ width: 100%; display: block; border-radius: 12px; background: white; }}
    .plot-modal {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 28px;
      background: rgba(15, 23, 42, .78);
      z-index: 50;
    }}
    .plot-modal.open {{ display: flex; }}
    .plot-modal-inner {{
      width: min(96vw, 1400px);
      max-height: 92vh;
      padding: 14px;
      border-radius: 18px;
      background: #fff;
      box-shadow: 0 28px 80px rgba(0, 0, 0, .35);
    }}
    .plot-modal-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }}
    .plot-modal-title {{ margin: 0; font-size: 16px; font-weight: 700; }}
    .plot-modal-close {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 12px;
      background: #f7f9fc;
      color: var(--ink);
      cursor: pointer;
      font-weight: 700;
    }}
    .plot-modal img {{
      display: block;
      max-width: 100%;
      max-height: calc(92vh - 74px);
      margin: 0 auto;
      border-radius: 12px;
    }}
    .example {{ padding: 18px; margin: 18px 0; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 8px;
      margin: 12px 0 16px;
    }}
    .meta div {{ padding: 8px 10px; border: 1px solid var(--line); border-radius: 12px; background: #fbfcff; }}
    .meta span {{ display: block; color: var(--muted); font-size: 12px; }}
    .meta strong {{ display: block; margin-top: 3px; overflow-wrap: anywhere; }}
    .timeline-svg {{ width: 100%; height: auto; margin: 8px 0 12px; border-radius: 14px; background: #fbfcff; border: 1px solid var(--line); }}
    .timeline-svg text, .spark-svg text {{ fill: var(--muted); font-size: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .bar {{ display: inline-block; min-width: 3px; height: 11px; border-radius: 99px; background: linear-gradient(90deg, var(--accent), #7aa2f7); }}
    .action-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .action-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #fbfcff; overflow: hidden; }}
    .action-card.local-action {{ border-color: #f1b879; background: #fffaf3; }}
    .action-card.web-action {{ border-color: #b8c9ee; }}
    .action-bar {{ height: 7px; margin-bottom: 9px; border-radius: 999px; background: linear-gradient(90deg, var(--accent), #7aa2f7); }}
    .local-action .action-bar {{ background: linear-gradient(90deg, #d97706, #fbbf24); }}
    .action-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }}
    .action-head span {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .action-card p {{ margin: 7px 0; font-size: 13px; line-height: 1.42; color: var(--ink); overflow-wrap: anywhere; }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 700; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      padding: 14px;
      border-radius: 14px;
      background: #f7f9fc;
      color: #18212b;
      border: 1px solid var(--line);
      max-height: 360px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.45;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    td {{ overflow-wrap: anywhere; }}
    .note {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>Privacy HopQA Slim Viewer</h1>
    <p>Single-file snapshot generated {esc(generated_at)} from <code>{esc(analysis_dir.name)}</code>. This is intentionally compact: key aggregate plots plus a small set of embedded examples for qualitative debugging.</p>
    <div class="stats">{render_stat_cards(summary)}</div>
  </header>
  <main>
    <h2>Performance And Behavior</h2>
    <p class="note">Use these plots to quickly check throughput shape, rollout length, reward distribution, source usage, and whether failures are concentrated in a particular stage.</p>
    <section class="plots">{render_plots(analysis_dir)}</section>

    <h2>Question Type Breakdown</h2>
    {render_question_type_section(analysis_dir, trend_rows)}

    <h2>Early / Middle / Late</h2>
    {render_temporal_section(analysis_dir, trend_rows, args.time_slice_size, args.moving_average_window)}

    <h2>Token And Action Costs</h2>
    <p class="note">Per-call token costs are grouped by the logged action type, so this is the quickest way to see whether planning, choosing, reading, or resolving is driving context pressure.</p>
    {render_token_section(analysis_dir, summary)}

    <h2>Timeline Stage Summary</h2>
    <p class="note">This aggregates the sampled timeline events from the full bundle. The embedded examples below include compact per-rollout timeline bars plus event tables.</p>
    {render_timeline_overview(analysis_dir)}

    <h2>Representative Examples</h2>
    <p class="note">Examples are sampled across the reward/duration spectrum. Each includes the report preview, stage-time totals, and a compact timeline preview. The full heavy bundle still has the complete timelines and CSVs if needed.</p>
      {render_examples(sample_dirs, row_by_id, problem_by_chain, args.max_events_per_sample)}
  </main>
  <div class="plot-modal" id="plot-modal" role="dialog" aria-modal="true" aria-label="Full-size plot">
    <div class="plot-modal-inner">
      <div class="plot-modal-bar">
        <p class="plot-modal-title" id="plot-modal-title"></p>
        <button class="plot-modal-close" type="button">Close</button>
      </div>
      <img id="plot-modal-img" alt="" />
    </div>
  </div>
  <script>
    (() => {{
      const modal = document.getElementById("plot-modal");
      const modalImg = document.getElementById("plot-modal-img");
      const modalTitle = document.getElementById("plot-modal-title");
      const closeButton = modal.querySelector(".plot-modal-close");

      function closeModal() {{
        modal.classList.remove("open");
        modalImg.removeAttribute("src");
        modalImg.alt = "";
        modalTitle.textContent = "";
      }}

      document.querySelectorAll(".plot-open").forEach((button) => {{
        button.addEventListener("click", () => {{
          const image = button.querySelector("img");
          if (!image) return;
          modalImg.src = image.src;
          modalImg.alt = image.alt || "Full-size plot";
          modalTitle.textContent = image.alt || "Full-size plot";
          modal.classList.add("open");
        }});
      }});

      closeButton.addEventListener("click", closeModal);
      modal.addEventListener("click", (event) => {{
        if (event.target === modal) closeModal();
      }});
      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape" && modal.classList.contains("open")) closeModal();
      }});
    }})();
  </script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
