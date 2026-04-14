#!/usr/bin/env python3
"""Generate runtime analysis plots for a privacy-agent run directory."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTOR_TS_RE = re.compile(r"\[actor\]: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
LLM_DONE_RE = re.compile(r"LLM request done in ([0-9.]+)s")
PROMPT_SUMMARY_RE = re.compile(r"privacy_agent prompt budget summary: (\{.*\})")
REPORT_TS_RE = re.compile(r"research_report_(\d{8}_\d{6})\.md$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to a privacy-agent output directory")
    parser.add_argument(
        "--output-dir",
        help="Directory to write plots and CSVs into. Defaults to MosaicProject/runtime_analysis/<run_name>",
    )
    return parser.parse_args()


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_actor_timestamp(line: str) -> datetime | None:
    match = ACTOR_TS_RE.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def parse_report_timestamp(path: Path) -> datetime | None:
    match = REPORT_TS_RE.search(path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    try:
        return datetime.utcfromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = max(0.0, min(len(ordered) - 1, (len(ordered) - 1) * (p / 100.0)))
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_actor_info(actor_info_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    llm_requests: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    with actor_info_path.open(errors="ignore") as handle:
        for raw_line in handle:
            timestamp = parse_actor_timestamp(raw_line)
            if not timestamp:
                continue

            llm_match = LLM_DONE_RE.search(raw_line)
            if llm_match:
                llm_requests.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "duration_s": float(llm_match.group(1)),
                    }
                )

            prompt_match = PROMPT_SUMMARY_RE.search(raw_line)
            if prompt_match:
                payload = ast.literal_eval(prompt_match.group(1))
                by_category = payload.get("by_category", {})
                findings = by_category.get("findings_or_summary", {})
                prompt_records.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "log_name": payload.get("log_name", "unknown"),
                        "prompt_tokens": int(payload.get("prompt_tokens", 0)),
                        "prompt_chars": int(payload.get("prompt_chars", 0)),
                        "latest_findings_tokens": int(findings.get("tokens", 0)),
                        "latest_findings_chars": int(findings.get("chars", 0)),
                        "instructions_tokens": int(by_category.get("instructions", {}).get("tokens", 0)),
                        "planning_context_tokens": int(by_category.get("planning_context", {}).get("tokens", 0)),
                        "task_definition_tokens": int(by_category.get("task_definition", {}).get("tokens", 0)),
                        "documents_or_evidence_tokens": int(
                            by_category.get("documents_or_evidence", {}).get("tokens", 0)
                        ),
                        "reporting_context_tokens": int(
                            by_category.get("reporting_context", {}).get("tokens", 0)
                        ),
                        "other_tokens": int(by_category.get("other", {}).get("tokens", 0)),
                    }
                )
    return llm_requests, prompt_records


def discover_report_timestamp(run_dir: Path) -> tuple[bool, datetime | None]:
    workspace = run_dir / "workspace"
    reports = sorted(workspace.glob("research_report_*.md"))
    if not reports:
        return False, None
    timestamps = [ts for ts in (parse_report_timestamp(path) for path in reports) if ts is not None]
    return True, max(timestamps) if timestamps else None


def derive_backend_tool(action: dict[str, Any]) -> str:
    actual_output = action.get("actual_output")
    if isinstance(actual_output, dict):
        for key in ("tool", "source", "analysis_type"):
            value = actual_output.get(key)
            if isinstance(value, str) and value:
                return value
    preferred_tool = action.get("preferred_tool")
    if isinstance(preferred_tool, str) and preferred_tool:
        return preferred_tool
    return str(action.get("type", "unknown"))


def parse_rollout_sessions(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rollouts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []

    runs_root = run_dir / "privacy_agent_runs"
    for session_dir in sorted(runs_root.glob("*/session")):
        run_path = session_dir.parent
        initial_path = session_dir / "action_plan_initial.json"
        final_path = session_dir / "action_plan_final.json"
        if not final_path.exists():
            continue

        initial_payload = json.loads(initial_path.read_text()) if initial_path.exists() else {}
        final_payload = json.loads(final_path.read_text())

        initial_ts = parse_iso_timestamp(initial_payload.get("timestamp"))
        final_ts = parse_iso_timestamp(final_payload.get("timestamp"))
        has_report, report_ts = discover_report_timestamp(run_path)
        start_ts = initial_ts
        end_ts = report_ts or final_ts

        final_actions = final_payload.get("actions", [])
        completion_times = [
            parse_iso_timestamp(action.get("completion_time"))
            for action in final_actions
            if parse_iso_timestamp(action.get("completion_time")) is not None
        ]
        if start_ts is None and completion_times:
            start_ts = min(completion_times)
        if end_ts is None and completion_times:
            end_ts = max(completion_times)

        total_tool_exec_s = sum(
            float(action["execution_time"])
            for action in final_actions
            if action.get("execution_time") is not None and action.get("status") == "completed"
        )
        duration_s = None
        if start_ts and end_ts:
            duration_s = max(0.0, (end_ts - start_ts).total_seconds())

        rollout_record = {
            "rollout_id": run_path.name,
            "has_report": has_report,
            "start_timestamp": start_ts.isoformat() if start_ts else "",
            "end_timestamp": end_ts.isoformat() if end_ts else "",
            "duration_s": duration_s if duration_s is not None else "",
            "current_iteration": int(final_payload.get("current_iteration", 0)),
            "max_iterations": int(final_payload.get("max_iterations", 0)),
            "total_actions": int(final_payload.get("stats", {}).get("total_actions", len(final_actions))),
            "completed_actions": int(final_payload.get("completed_actions", 0)),
            "failed_actions": int(final_payload.get("failed_actions", 0)),
            "pending_actions": int(
                final_payload.get("stats", {}).get(
                    "pending",
                    sum(1 for action in final_actions if action.get("status") == "pending"),
                )
            ),
            "tool_exec_time_s": total_tool_exec_s,
            "non_tool_time_s": max(0.0, duration_s - total_tool_exec_s) if duration_s is not None else "",
        }
        rollouts.append(rollout_record)

        per_iter_completion: dict[int, list[datetime]] = defaultdict(list)
        per_iter_exec_s: dict[int, float] = defaultdict(float)
        per_iter_action_count: dict[int, int] = defaultdict(int)

        for action in final_actions:
            exec_time = action.get("execution_time")
            completion_time = parse_iso_timestamp(action.get("completion_time"))
            iteration_completed = action.get("iteration_completed")
            action_record = {
                "rollout_id": run_path.name,
                "has_report": has_report,
                "action_id": action.get("id", ""),
                "action_type": action.get("type", "unknown"),
                "backend_tool": derive_backend_tool(action),
                "status": action.get("status", "unknown"),
                "execution_time_s": float(exec_time) if exec_time is not None else "",
                "iteration_completed": iteration_completed if iteration_completed is not None else "",
                "completion_time": completion_time.isoformat() if completion_time else "",
                "created_from_research_step": action.get("created_from_research_step", ""),
                "preferred_tool": action.get("preferred_tool", ""),
            }
            actions.append(action_record)

            if completion_time is not None and iteration_completed is not None:
                iter_idx = int(iteration_completed)
                per_iter_completion[iter_idx].append(completion_time)
                per_iter_action_count[iter_idx] += 1
                if exec_time is not None and action.get("status") == "completed":
                    per_iter_exec_s[iter_idx] += float(exec_time)

        if start_ts is not None:
            previous_end = start_ts
            for iter_idx in sorted(per_iter_completion):
                iter_end = max(per_iter_completion[iter_idx])
                wall_time_s = max(0.0, (iter_end - previous_end).total_seconds())
                tool_time_s = per_iter_exec_s.get(iter_idx, 0.0)
                iterations.append(
                    {
                        "rollout_id": run_path.name,
                        "has_report": has_report,
                        "iteration": iter_idx,
                        "action_count": per_iter_action_count.get(iter_idx, 0),
                        "wall_time_s": wall_time_s,
                        "tool_time_s": tool_time_s,
                        "non_tool_time_s": max(0.0, wall_time_s - tool_time_s),
                    }
                )
                previous_end = max(previous_end, iter_end)

    return rollouts, actions, iterations


def aggregate_group_stats(records: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record.get(value_key)
        if value in ("", None):
            continue
        grouped[str(record.get(group_key, "unknown"))].append(float(value))
    return {group: numeric_stats(values) for group, values in sorted(grouped.items())}


def plot_hist_and_ecdf(
    values: list[float],
    path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 50,
) -> None:
    if not values:
        return
    ordered = sorted(values)
    ecdf_y = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(values, bins=bins, color="#4472c4", edgecolor="white")
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")
    axes[1].plot(ordered, ecdf_y, color="#c44e52")
    axes[1].set_title(f"{title} ECDF")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Fraction <= x")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bar(counter: Counter[str], path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    if not counter:
        return
    labels, values = zip(*counter.most_common())
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_boxplot(
    grouped: dict[str, list[float]],
    path: Path,
    *,
    title: str,
    xlabel: str,
    top_n: int = 10,
    use_log_scale: bool = False,
) -> None:
    non_empty = [(label, values) for label, values in grouped.items() if values]
    if not non_empty:
        return
    non_empty.sort(key=lambda item: (len(item[1]), median(item[1])), reverse=True)
    chosen = non_empty[:top_n]
    labels = [label for label, _ in chosen]
    data = [values for _, values in chosen]

    fig, ax = plt.subplots(figsize=(11, max(4.5, len(labels) * 0.55)))
    ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if use_log_scale:
        ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rollout_duration_by_iterations(rollouts: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        duration = rollout.get("duration_s")
        if duration in ("", None):
            continue
        grouped[str(rollout["current_iteration"])].append(float(duration))
    plot_grouped_boxplot(
        grouped,
        path,
        title="Rollout Duration by Final Iteration Count",
        xlabel="Rollout duration (seconds)",
        top_n=12,
        use_log_scale=True,
    )


def plot_iteration_breakdown(iterations: list[dict[str, Any]], path: Path) -> None:
    if not iterations:
        return
    by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in iterations:
        if not row.get("has_report"):
            continue
        by_iteration[int(row["iteration"])].append(row)
    if not by_iteration:
        return

    labels = sorted(by_iteration)
    median_tool = [median([float(row["tool_time_s"]) for row in by_iteration[label]]) for label in labels]
    median_non_tool = [median([float(row["non_tool_time_s"]) for row in by_iteration[label]]) for label in labels]
    counts = [len(by_iteration[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, median_tool, label="Tool execution", color="#4c78a8")
    ax.bar(labels, median_non_tool, bottom=median_tool, label="LLM + orchestration", color="#f58518")
    for idx, count in enumerate(counts):
        ax.text(labels[idx], median_tool[idx] + median_non_tool[idx], str(count), ha="center", va="bottom", fontsize=8)
    ax.set_title("Median Wall Time per Iteration")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("Median time (seconds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_prompt_token_breakdown(prompt_records: list[dict[str, Any]], path: Path) -> None:
    if not prompt_records:
        return
    counts = Counter(record["log_name"] for record in prompt_records)
    labels = [label for label, _ in counts.most_common(8)]
    grouped = {label: [int(record["prompt_tokens"]) for record in prompt_records if record["log_name"] == label] for label in labels}
    median_share = []
    for label in labels:
        shares = []
        for record in prompt_records:
            if record["log_name"] != label:
                continue
            prompt_tokens = int(record["prompt_tokens"])
            findings = int(record["latest_findings_tokens"])
            if prompt_tokens > 0:
                shares.append(findings / prompt_tokens)
        median_share.append(median(shares) if shares else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(4.5, len(labels) * 0.65)))
    axes[0].boxplot([grouped[label] for label in labels], vert=False, tick_labels=labels, showfliers=False)
    axes[0].set_title("Prompt Tokens by LLM Call Type")
    axes[0].set_xlabel("Prompt tokens")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(labels, median_share, color="#e45756")
    axes[1].set_title("Median Latest Findings Share")
    axes[1].set_xlabel("Fraction of prompt tokens")
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    path: Path,
    *,
    run_dir: Path,
    llm_requests: list[dict[str, Any]],
    prompt_records: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    completed_rollouts: list[dict[str, Any]],
) -> None:
    llm_values = [float(row["duration_s"]) for row in llm_requests]
    completed_durations = [float(row["duration_s"]) for row in completed_rollouts if row.get("duration_s") not in ("", None)]
    tool_exec = [float(row["execution_time_s"]) for row in actions if row.get("execution_time_s") not in ("", None)]
    tool_counter = Counter(row["action_type"] for row in actions if row.get("has_report"))
    findings_share = [
        int(row["latest_findings_tokens"]) / int(row["prompt_tokens"])
        for row in prompt_records
        if int(row["prompt_tokens"]) > 0
    ]

    lines = [
        f"# Runtime Analysis for `{run_dir.name}`",
        "",
        "## Notes",
        "- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.",
        "- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.",
        "- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.",
        "",
        "## Headline Stats",
        f"- Completed rollouts analyzed: `{len(completed_rollouts)}`",
        f"- Total completed tool actions analyzed: `{sum(tool_counter.values())}`",
        f"- Total LLM request duration samples: `{len(llm_values)}`",
        f"- Median LLM request latency: `{median(llm_values):.2f}s`" if llm_values else "- Median LLM request latency: n/a",
        f"- Median completed rollout duration: `{median(completed_durations):.2f}s`" if completed_durations else "- Median completed rollout duration: n/a",
        f"- Median tool execution time: `{median(tool_exec):.2f}s`" if tool_exec else "- Median tool execution time: n/a",
        f"- Median Latest Findings share of logged prompts: `{median(findings_share) * 100:.1f}%`"
        if findings_share
        else "- Median Latest Findings share of logged prompts: n/a",
        "",
        "## Top Action Types",
    ]
    for label, count in tool_counter.most_common(8):
        lines.append(f"- `{label}`: `{count}`")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    mosaic_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else mosaic_root / "runtime_analysis" / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_info_path = run_dir / "actor" / "info.log"
    llm_requests, prompt_records = parse_actor_info(actor_info_path)
    rollouts, actions, iterations = parse_rollout_sessions(run_dir)

    completed_rollouts = [row for row in rollouts if row.get("has_report")]
    completed_actions = [row for row in actions if row.get("has_report") and row.get("status") == "completed"]
    completed_iterations = [row for row in iterations if row.get("has_report")]

    write_csv(output_dir / "llm_requests.csv", llm_requests)
    write_csv(output_dir / "prompt_records.csv", prompt_records)
    write_csv(output_dir / "rollouts.csv", rollouts)
    write_csv(output_dir / "actions.csv", actions)
    write_csv(output_dir / "iterations.csv", iterations)

    llm_durations = [float(row["duration_s"]) for row in llm_requests]
    prompt_tokens = [int(row["prompt_tokens"]) for row in prompt_records]
    completed_rollout_durations = [float(row["duration_s"]) for row in completed_rollouts if row.get("duration_s") not in ("", None)]
    actions_per_rollout = [int(row["total_actions"]) for row in completed_rollouts]
    iteration_counts = Counter(int(row["current_iteration"]) for row in completed_rollouts)
    tool_exec_values = [float(row["execution_time_s"]) for row in completed_actions if row.get("execution_time_s") not in ("", None)]
    tool_by_type: dict[str, list[float]] = defaultdict(list)
    tool_by_backend: dict[str, list[float]] = defaultdict(list)
    for row in completed_actions:
        value = row.get("execution_time_s")
        if value in ("", None):
            continue
        tool_by_type[str(row["action_type"])].append(float(value))
        tool_by_backend[str(row["backend_tool"])].append(float(value))

    prompt_counter = Counter(record["log_name"] for record in prompt_records)
    prompt_stats_by_log_name = aggregate_group_stats(prompt_records, "log_name", "prompt_tokens")
    action_stats_by_type = aggregate_group_stats(completed_actions, "action_type", "execution_time_s")
    action_stats_by_backend = aggregate_group_stats(completed_actions, "backend_tool", "execution_time_s")
    rollout_duration_by_iteration = aggregate_group_stats(completed_rollouts, "current_iteration", "duration_s")

    summary = {
        "run_dir": str(run_dir),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "counts": {
            "llm_requests": len(llm_requests),
            "prompt_records": len(prompt_records),
            "rollouts_all": len(rollouts),
            "rollouts_completed_with_report": len(completed_rollouts),
            "actions_completed_with_report": len(completed_actions),
            "iteration_records_completed_with_report": len(completed_iterations),
        },
        "llm_request_latency_s": numeric_stats(llm_durations),
        "logged_prompt_tokens": numeric_stats(prompt_tokens),
        "completed_rollout_duration_s": numeric_stats(completed_rollout_durations),
        "completed_action_execution_s": numeric_stats(tool_exec_values),
        "prompt_count_by_log_name": dict(prompt_counter.most_common()),
        "iteration_count_distribution": dict(sorted(iteration_counts.items())),
        "prompt_tokens_by_log_name": prompt_stats_by_log_name,
        "action_execution_s_by_type": action_stats_by_type,
        "action_execution_s_by_backend": action_stats_by_backend,
        "rollout_duration_s_by_final_iteration": rollout_duration_by_iteration,
        "limitations": [
            "LLM duration logs do not include request IDs, so exact latency attribution to log_name is not possible under concurrency.",
            "Prompt summaries cover large prompts (>= 4096 tokens) or overflow cases, not every single LLM call.",
            "Rollout timing analysis is filtered to runs with a saved report to avoid in-progress session artifacts.",
        ],
    }
    save_json(output_dir / "summary.json", summary)
    write_summary_markdown(
        output_dir / "README.md",
        run_dir=run_dir,
        llm_requests=llm_requests,
        prompt_records=prompt_records,
        rollouts=rollouts,
        actions=completed_actions,
        completed_rollouts=completed_rollouts,
    )

    plot_hist_and_ecdf(
        llm_durations,
        output_dir / "llm_latency_distribution.png",
        title="LLM Request Latency Distribution",
        xlabel="Latency (seconds)",
    )
    plot_prompt_token_breakdown(prompt_records, output_dir / "prompt_tokens_by_log_name.png")
    plot_hist_and_ecdf(
        completed_rollout_durations,
        output_dir / "rollout_duration_distribution.png",
        title="Completed Rollout Duration Distribution",
        xlabel="Duration (seconds)",
    )
    plot_hist_and_ecdf(
        actions_per_rollout,
        output_dir / "actions_per_rollout_distribution.png",
        title="Actions per Completed Rollout",
        xlabel="Action count",
        bins=24,
    )
    plot_bar(
        Counter({str(key): value for key, value in iteration_counts.items()}),
        output_dir / "iteration_count_distribution.png",
        title="Completed Rollout Final Iteration Distribution",
        xlabel="Final iteration count",
        ylabel="Rollouts",
    )
    plot_rollout_duration_by_iterations(completed_rollouts, output_dir / "rollout_duration_by_iteration.png")
    plot_iteration_breakdown(completed_iterations, output_dir / "iteration_time_breakdown.png")
    plot_grouped_boxplot(
        tool_by_type,
        output_dir / "tool_latency_by_action_type.png",
        title="Tool Execution Latency by Action Type",
        xlabel="Execution time (seconds)",
        top_n=10,
        use_log_scale=True,
    )
    plot_grouped_boxplot(
        tool_by_backend,
        output_dir / "tool_latency_by_backend.png",
        title="Tool Execution Latency by Backend Tool",
        xlabel="Execution time (seconds)",
        top_n=10,
        use_log_scale=True,
    )
    plot_bar(
        Counter(row["action_type"] for row in completed_actions),
        output_dir / "tool_counts_by_action_type.png",
        title="Completed Tool Calls by Action Type",
        xlabel="Action type",
        ylabel="Completed tool calls",
    )
    plot_bar(
        Counter(row["backend_tool"] for row in completed_actions),
        output_dir / "tool_counts_by_backend.png",
        title="Completed Tool Calls by Backend Tool",
        xlabel="Backend tool",
        ylabel="Completed tool calls",
    )

    print(f"Wrote runtime analysis to {output_dir}")


if __name__ == "__main__":
    main()
