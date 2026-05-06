#!/usr/bin/env python3
"""Live training dashboard for the privacy_hopqa PipelineRL runs."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import heapq
import html
import json
import math
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover - dashboard should still render in minimal envs.
    sns = None


DEFAULT_HELPER_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-ph-844f0105:8012"
DEFAULT_RUNS = [
    "Qwen3-8B|a1161525-4e2e-4e43-b9bf-d83653ae1d57|/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_final_qwen3_8b_r2_filter_vllm18_schedulerfix_20260506_152928",
    "Qwen3-4B-Instruct-2507|fbe9aa12-9c2b-43e3-9c45-7e6489add3e4|/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_final_qwen3_4b_instruct_2507_r2_filter_vllm18_schedulerfix_20260506_152949",
]

LOG_TS_RE = re.compile(r"\[[^\]]+\]: (?P<ts>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\.\d{3})")
COMPLETED_RE = re.compile(r"Completed steps (?P<step>\d+): (?P<stats>\{.*\})")
PREPROCESS_RE = re.compile(
    r"Processed (?P<processed>\d+) samples \(filtered out (?P<filtered>\d+)\).*?total (?P<total>\d+) samples so far"
)
ZERO_FILTER_RE = re.compile(r"Filtered out (?P<count>\d+) samples from groups with zero advantage")
ACTOR_PUBLISH_RE = re.compile(r"Published (?P<count>\d+) train samples.*?total (?P<total>\d+) samples so far")
ERROR_RE = re.compile(
    r"Traceback|AsyncEngineDead|CUDA out of memory|CUBLAS_STATUS|RuntimeError|Exception|\bERROR\b|ValueError",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    job_id: str
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_schedulerfix_live",
        help="Directory for index.html, plots, CSVs, and snapshot JSONL files.",
    )
    parser.add_argument("--helper-url", default=DEFAULT_HELPER_URL, help="Privacy helper base URL.")
    parser.add_argument("--run", action="append", default=[], help="Run spec: LABEL|JOB_ID|RUN_DIR")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Refresh once and exit.")
    parser.add_argument("--tail-lines", type=int, default=8000, help="EAI log tail lines per run.")
    parser.add_argument("--rolling-window", type=int, default=8, help="Snapshot points for moving averages.")
    parser.add_argument("--new-trace-limit", type=int, default=512, help="New rollout traces to parse per run per refresh.")
    parser.add_argument("--trace-cache-limit", type=int, default=6000, help="Recent rollout traces to retain per run.")
    return parser.parse_args()


def parse_run(spec: str) -> RunSpec:
    parts = spec.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"Run spec must be LABEL|JOB_ID|RUN_DIR, got: {spec}")
    label, job_id, run_dir = parts
    return RunSpec(label=label.strip(), job_id=job_id.strip(), run_dir=Path(run_dir.strip()))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_log_ts(line: str) -> datetime | None:
    match = LOG_TS_RE.search(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("ts"), "%m/%d/%Y %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def fmt_num(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if float(number).is_integer():
        return f"{int(number)}"
    return f"{number:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{100 * number:.{digits}f}%"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rolling(values: list[float | None], window: int) -> list[float | None]:
    out: list[float | None] = []
    for idx in range(len(values)):
        chunk = [value for value in values[max(0, idx - window + 1) : idx + 1] if value is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def fetch_helper_health(helper_url: str) -> dict[str, Any]:
    url = helper_url.rstrip("/") + "/health"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=6) as response:
        return json.loads(response.read().decode("utf-8"))


def snapshot_helper(output_dir: Path, helper_url: str, now: datetime) -> dict[str, Any]:
    row: dict[str, Any] = {"timestamp": iso(now), "time_s": now.timestamp(), "helper_url": helper_url}
    try:
        data = fetch_helper_health(helper_url)
        row["ok"] = bool(data.get("ok"))
        row["embedding_server_ok"] = bool(data.get("embedding_server_ok"))
        row["metrics"] = data.get("metrics", {})
        row["raw"] = data
    except Exception as exc:
        row["ok"] = False
        row["error"] = repr(exc)
        row["metrics"] = {}
    append_jsonl(output_dir / "helper_snapshots.jsonl", row)
    return row


def flatten_helper_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        timestamp = row.get("timestamp")
        time_s = row.get("time_s")
        metrics = row.get("metrics") or {}
        for endpoint, values in metrics.items():
            if not isinstance(values, dict):
                continue
            out = {"timestamp": timestamp, "time_s": time_s, "endpoint": endpoint}
            for key in ("count", "errors", "active", "p50_ms", "p95_ms", "max_ms"):
                out[key] = values.get(key)
            flat.append(out)
    return flat


def parse_finetune_log(run: RunSpec) -> list[dict[str, Any]]:
    path = run.run_dir / "finetune" / "log" / "info_0.log"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = COMPLETED_RE.search(line)
            if not match:
                continue
            try:
                stats = ast.literal_eval(match.group("stats"))
            except (ValueError, SyntaxError):
                stats = {}

            timestamp = parse_log_ts(line)
            row: dict[str, Any] = {
                "label": run.label,
                "job_id": run.job_id,
                "run_dir": str(run.run_dir),
                "step": int(match.group("step")),
            }
            if timestamp:
                row["timestamp"] = iso(timestamp)
                row["time_s"] = timestamp.timestamp()
            for key, value in stats.items():
                number = as_float(value)
                row[key] = number if number is not None else value
            rows.append(row)
    return rows


def run_command(command: list[str], timeout_s: int) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except Exception as exc:
        return f"COMMAND_FAILED {command!r}: {exc!r}"
    return completed.stdout or ""


def parse_eai_tail(text: str, run: RunSpec, now: datetime) -> dict[str, Any]:
    processed_sum = 0
    filtered_sum = 0
    zero_filtered_sum = 0
    actor_published_sum = 0
    latest_accepted_total: int | None = None
    latest_actor_total: int | None = None
    latest_log_ts: datetime | None = None
    errors: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        ts = parse_log_ts(line)
        if ts:
            latest_log_ts = ts

        if match := PREPROCESS_RE.search(line):
            processed_sum += int(match.group("processed"))
            filtered_sum += int(match.group("filtered"))
            latest_accepted_total = int(match.group("total"))

        if match := ZERO_FILTER_RE.search(line):
            zero_filtered_sum += int(match.group("count"))

        if match := ACTOR_PUBLISH_RE.search(line):
            actor_published_sum += int(match.group("count"))
            latest_actor_total = int(match.group("total"))

        if ERROR_RE.search(line):
            errors.append(line[-500:])

    denominator = processed_sum + filtered_sum
    return {
        "timestamp": iso(now),
        "time_s": now.timestamp(),
        "label": run.label,
        "job_id": run.job_id,
        "run_dir": str(run.run_dir),
        "latest_log_timestamp": iso(latest_log_ts) if latest_log_ts else None,
        "recent_processed_samples": processed_sum,
        "recent_filtered_samples": filtered_sum,
        "recent_zero_advantage_filtered_samples": zero_filtered_sum,
        "recent_filter_rate": (filtered_sum / denominator) if denominator else None,
        "latest_accepted_total": latest_accepted_total,
        "recent_actor_published_samples": actor_published_sum,
        "latest_actor_total": latest_actor_total,
        "error_count_in_tail": len(errors),
        "last_errors": errors[-6:],
    }


def snapshot_eai_runs(output_dir: Path, runs: list[RunSpec], tail_lines: int, now: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        text = run_command(["eai", "job", "log", run.job_id, "--tail", str(tail_lines)], timeout_s=45)
        row = parse_eai_tail(text, run, now)
        rows.append(row)
        append_jsonl(output_dir / "run_snapshots.jsonl", row)
    return rows


def newest_rollout_dirs(run: RunSpec, limit: int) -> list[tuple[float, Path]]:
    root = run.run_dir / "privacy_hopqa_runs"
    if not root.exists():
        return []
    candidates: list[tuple[float, Path]] = []
    try:
        for path in root.iterdir():
            if not path.is_dir():
                continue
            summary = path / "timeline_summary.json"
            events = path / "timeline_events.json"
            if not summary.exists() or not events.exists():
                continue
            try:
                mtime = max(summary.stat().st_mtime, events.stat().st_mtime)
            except OSError:
                continue
            candidates.append((mtime, path))
    except OSError:
        return []
    return sorted(heapq.nlargest(limit, candidates, key=lambda item: item[0]), key=lambda item: item[0])


def event_duration_s(event: dict[str, Any]) -> float | None:
    start = as_float(event.get("start_s"))
    end = as_float(event.get("end_s"))
    if start is None or end is None:
        return None
    duration = end - start
    return duration if duration >= 0 else None


def search_backend(event: dict[str, Any]) -> str:
    meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}
    action_type = str(meta.get("action_type") or meta.get("tool") or event.get("label") or "").lower()
    if "local" in action_type:
        return "local"
    if "web" in action_type or "browse" in action_type:
        return "web"
    return "unknown"


def parse_trace_timings(
    runs: list[RunSpec],
    seen_rollouts: set[tuple[str, str]],
    new_trace_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for run in runs:
        root = run.run_dir / "privacy_hopqa_runs"
        if not root.exists():
            continue
        parsed_for_run = 0
        try:
            rollout_dirs = root.iterdir()
        except OSError:
            continue
        for rollout_dir in rollout_dirs:
            if parsed_for_run >= new_trace_limit:
                break
            if not rollout_dir.is_dir():
                continue
            rollout_id = rollout_dir.name
            if (run.label, rollout_id) in seen_rollouts:
                continue
            summary_path = rollout_dir / "timeline_summary.json"
            events_path = rollout_dir / "timeline_events.json"
            trace_path = rollout_dir / "trace.json"
            if not summary_path.exists() or not events_path.exists():
                continue
            try:
                completed_at_s = max(summary_path.stat().st_mtime, events_path.stat().st_mtime)
            except OSError:
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                events = json.loads(events_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(summary, dict) or not isinstance(events, list):
                continue

            base = {
                "label": run.label,
                "job_id": run.job_id,
                "run_dir": str(run.run_dir),
                "rollout_id": rollout_id,
                "task_id": summary.get("task_id"),
                "chain_id": summary.get("chain_id"),
                "completed_at_s": completed_at_s,
                "timestamp": iso(datetime.fromtimestamp(completed_at_s, tz=timezone.utc)),
            }

            stage_values: dict[str, list[float]] = {}
            search_values: dict[str, list[float]] = {"web": [], "local": [], "unknown": [], "all": []}
            action_values: dict[str, list[float]] = {"web_search": [], "local_document_search": [], "all": []}
            for event in events:
                if not isinstance(event, dict):
                    continue
                duration = event_duration_s(event)
                if duration is None:
                    continue
                stage = str(event.get("stage") or "unknown")
                stage_values.setdefault(stage, []).append(duration)
                row = {
                    **base,
                    "stage": stage,
                    "duration_s": duration,
                    "lane": event.get("lane"),
                    "event_label": event.get("label"),
                }
                if stage == "search":
                    backend = search_backend(event)
                    row["search_backend"] = backend
                    search_values.setdefault(backend, []).append(duration)
                    search_values["all"].append(duration)
                stage_rows.append(row)

            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                trace = {}
            actions = trace.get("actions") if isinstance(trace, dict) else None
            if isinstance(actions, list):
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    action_type = str(action.get("type") or "")
                    if action_type not in {"web_search", "local_document_search"}:
                        continue
                    execution_time = as_float(action.get("execution_time"))
                    if execution_time is None:
                        execution_time = as_float(action.get("execution_time_s"))
                    if execution_time is None:
                        continue
                    actual_output = action.get("actual_output") if isinstance(action.get("actual_output"), dict) else {}
                    action_values.setdefault(action_type, []).append(execution_time)
                    action_values["all"].append(execution_time)
                    action_rows.append(
                        {
                            **base,
                            "action_id": action.get("id"),
                            "action_type": action_type,
                            "tool": actual_output.get("tool"),
                            "status": action.get("status"),
                            "execution_time_s": execution_time,
                            "query": actual_output.get("query")
                            or (action.get("parameters") or {}).get("query")
                            if isinstance(action.get("parameters"), dict)
                            else actual_output.get("query"),
                            "success": actual_output.get("success"),
                            "results_count": actual_output.get("results_count"),
                        }
                    )

            rollout_row: dict[str, Any] = {
                **base,
                "duration_s": summary.get("duration_s"),
                "iterations_used": summary.get("iterations_used"),
                "resolved_hops": summary.get("resolved_hops"),
                "total_hops": summary.get("total_hops"),
                "searches_total": summary.get("searches_total"),
                "docs_read": summary.get("docs_read"),
                "llm_calls_total": summary.get("llm_calls_total"),
                "prompt_tokens": summary.get("prompt_tokens"),
                "output_tokens": summary.get("output_tokens"),
                "total_tokens": summary.get("total_tokens"),
                "total_errors": summary.get("total_errors"),
            }
            for stage, values in stage_values.items():
                safe_stage = re.sub(r"[^A-Za-z0-9]+", "_", stage).strip("_")
                rollout_row[f"stage_{safe_stage}_count"] = len(values)
                rollout_row[f"stage_{safe_stage}_mean_s"] = sum(values) / len(values)
                rollout_row[f"stage_{safe_stage}_sum_s"] = sum(values)
            for backend, values in search_values.items():
                rollout_row[f"search_{backend}_count"] = len(values)
                rollout_row[f"search_{backend}_mean_s"] = sum(values) / len(values) if values else None
                rollout_row[f"search_{backend}_sum_s"] = sum(values) if values else None
            for action_type, values in action_values.items():
                safe_action = re.sub(r"[^A-Za-z0-9]+", "_", action_type).strip("_")
                rollout_row[f"helper_action_{safe_action}_count"] = len(values)
                rollout_row[f"helper_action_{safe_action}_mean_s"] = sum(values) / len(values) if values else None
                rollout_row[f"helper_action_{safe_action}_sum_s"] = sum(values) if values else None
            rollout_rows.append(rollout_row)
            parsed_for_run += 1
    return stage_rows, rollout_rows, action_rows


def trim_trace_cache(
    stage_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    cache_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept_ids: set[tuple[str, str]] = set()
    kept_rollouts: list[dict[str, Any]] = []
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    for label in labels:
        rows = [row for row in rollout_rows if row.get("label") == label]
        rows = sorted(rows, key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
        for row in rows[-cache_limit:]:
            rollout_id = str(row.get("rollout_id") or "")
            if not rollout_id:
                continue
            kept_ids.add((label, rollout_id))
            kept_rollouts.append(row)

    def keep(row: dict[str, Any]) -> bool:
        label = str(row.get("label") or "")
        rollout_id = str(row.get("rollout_id") or "")
        return (label, rollout_id) in kept_ids

    kept_stage = [row for row in stage_rows if keep(row)]
    kept_actions = [row for row in action_rows if keep(row)]
    return kept_stage, kept_rollouts, kept_actions


def plot_setup() -> None:
    if sns is not None:
        sns.set_theme(
            context="talk",
            style="whitegrid",
            palette=["#184e77", "#d95d39", "#2a9d8f", "#7f4f24", "#5a189a"],
        )
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def datetimes(rows: list[dict[str, Any]]) -> list[datetime | None]:
    out: list[datetime | None] = []
    for row in rows:
        timestamp = parse_iso(row.get("timestamp"))
        if timestamp is None and row.get("time_s") is not None:
            try:
                timestamp = datetime.fromtimestamp(float(row["time_s"]), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                timestamp = None
        out.append(timestamp)
    return out


def finish_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC time")
    ax.grid(True, alpha=0.25)


def plot_training_progress(output_dir: Path, finetune_rows_by_run: dict[str, list[dict[str, Any]]]) -> str | None:
    if not any(finetune_rows_by_run.values()):
        return None
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, rows in finetune_rows_by_run.items():
        rows = [row for row in rows if row.get("timestamp")]
        if not rows:
            continue
        xs = [dt for dt in datetimes(rows) if dt is not None]
        paired = [(dt, row) for dt, row in zip(datetimes(rows), rows) if dt is not None]
        if not paired:
            continue
        xs = [item[0] for item in paired]
        step = [as_float(item[1].get("step")) for item in paired]
        samples = [as_float(item[1].get("stats/samples")) for item in paired]
        axes[0].plot(xs, step, marker="o", markersize=2.5, linewidth=1.8, label=label)
        axes[1].plot(xs, samples, marker="o", markersize=2.5, linewidth=1.8, label=label)
    axes[0].set_title("Finetune completed steps")
    axes[0].set_ylabel("steps")
    axes[1].set_title("Trained samples")
    axes[1].set_ylabel("samples")
    for ax in axes:
        ax.legend(loc="best")
        finish_time_axis(ax)
    path = output_dir / "training_progress.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_reward_throughput(output_dir: Path, finetune_rows_by_run: dict[str, list[dict[str, Any]]], window: int) -> str | None:
    if not any(finetune_rows_by_run.values()):
        return None
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, rows in finetune_rows_by_run.items():
        paired = [(dt, row) for dt, row in zip(datetimes(rows), rows) if dt is not None]
        if not paired:
            continue
        xs = [item[0] for item in paired]
        rewards = [as_float(item[1].get("rl/reward")) for item in paired]
        throughput = [as_float(item[1].get("throughput/tokens_perGPU_per_sec")) for item in paired]
        axes[0].plot(xs, rolling(rewards, window), linewidth=2.0, label=f"{label} reward")
        axes[1].plot(xs, rolling(throughput, window), linewidth=2.0, label=f"{label} tok/GPU/s")
    axes[0].set_title(f"Reward moving average ({window} points)")
    axes[0].set_ylabel("reward")
    axes[1].set_title(f"Throughput moving average ({window} points)")
    axes[1].set_ylabel("tokens/GPU/sec")
    for ax in axes:
        ax.legend(loc="best")
        finish_time_axis(ax)
    path = output_dir / "reward_throughput_ma.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_filtering(output_dir: Path, snapshot_rows: list[dict[str, Any]], window: int) -> str | None:
    if not snapshot_rows:
        return None
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    labels = sorted({str(row.get("label")) for row in snapshot_rows if row.get("label")})
    plotted = False
    for label in labels:
        rows = [row for row in snapshot_rows if row.get("label") == label]
        paired = [(dt, row) for dt, row in zip(datetimes(rows), rows) if dt is not None]
        if not paired:
            continue
        xs = [item[0] for item in paired]
        filter_rate = [as_float(item[1].get("recent_filter_rate")) for item in paired]
        accepted = [as_float(item[1].get("latest_accepted_total")) for item in paired]
        actor_total = [as_float(item[1].get("latest_actor_total")) for item in paired]
        axes[0].plot(xs, rolling(filter_rate, window), marker="o", markersize=2.5, linewidth=2.0, label=label)
        axes[1].plot(xs, accepted, marker="o", markersize=2.5, linewidth=1.8, label=f"{label} accepted")
        axes[1].plot(xs, actor_total, linestyle="--", linewidth=1.5, label=f"{label} actor published")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title(f"Recent zero-advantage filter rate moving average ({window} snapshots)")
    axes[0].set_ylabel("filtered / generated")
    axes[0].set_ylim(bottom=0, top=1)
    axes[1].set_title("Preprocess accepted total vs actor published total")
    axes[1].set_ylabel("samples")
    for ax in axes:
        ax.legend(loc="best")
        finish_time_axis(ax)
    path = output_dir / "filtering_progress.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_helper_latency(output_dir: Path, helper_rows: list[dict[str, Any]], window: int) -> str | None:
    rows = [row for row in helper_rows if row.get("endpoint") in ("/browsecomp/search", "/local/search")]
    if not rows:
        return None
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    endpoints = ["/browsecomp/search", "/local/search"]
    for endpoint in endpoints:
        endpoint_rows = [row for row in rows if row.get("endpoint") == endpoint]
        paired = [(dt, row) for dt, row in zip(datetimes(endpoint_rows), endpoint_rows) if dt is not None]
        if not paired:
            continue
        xs = [item[0] for item in paired]
        p50 = rolling([as_float(item[1].get("p50_ms")) for item in paired], window)
        p95 = rolling([as_float(item[1].get("p95_ms")) for item in paired], window)
        axes[0].plot(xs, p50, marker="o", markersize=2.2, linewidth=2.0, label=f"{endpoint} p50")
        axes[0].plot(xs, p95, linestyle="--", linewidth=1.8, label=f"{endpoint} p95")

        rates: list[float | None] = []
        prev_count: float | None = None
        prev_time: datetime | None = None
        for dt, item in paired:
            count = as_float(item.get("count"))
            if count is None or prev_count is None or prev_time is None:
                rates.append(None)
            else:
                elapsed_min = max(0.0, (dt - prev_time).total_seconds() / 60.0)
                rates.append(((count - prev_count) / elapsed_min) if elapsed_min > 0 else None)
            prev_count = count
            prev_time = dt
        axes[1].plot(xs, rolling(rates, window), marker="o", markersize=2.2, linewidth=2.0, label=endpoint)
    axes[0].set_title(f"Helper search latency moving average ({window} snapshots)")
    axes[0].set_ylabel("milliseconds")
    axes[1].set_title(f"Helper request rate moving average ({window} snapshots)")
    axes[1].set_ylabel("requests/min")
    for ax in axes:
        ax.legend(loc="best")
        finish_time_axis(ax)
    path = output_dir / "helper_latency_and_rate.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_trace_search_latency(output_dir: Path, rollout_rows: list[dict[str, Any]], window: int) -> str | None:
    if not rollout_rows:
        return None
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    plotted = False
    for label in labels:
        rows = sorted([row for row in rollout_rows if row.get("label") == label], key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
        paired = [
            (datetime.fromtimestamp(float(row["completed_at_s"]), tz=timezone.utc), row)
            for row in rows
            if as_float(row.get("completed_at_s")) is not None
        ]
        if not paired:
            continue
        xs = [item[0] for item in paired]
        web = rolling([as_float(item[1].get("search_web_mean_s")) for item in paired], window)
        local = rolling([as_float(item[1].get("search_local_mean_s")) for item in paired], window)
        all_search = rolling([as_float(item[1].get("search_all_mean_s")) for item in paired], window)
        search_count = rolling([as_float(item[1].get("search_all_count")) for item in paired], window)
        axes[0].plot(xs, all_search, linewidth=2.2, label=f"{label} all search")
        axes[0].plot(xs, web, linestyle="--", linewidth=1.7, label=f"{label} web")
        axes[0].plot(xs, local, linestyle=":", linewidth=2.0, label=f"{label} local")
        axes[1].plot(xs, search_count, linewidth=2.0, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title(f"Trace search-stage latency moving average ({window} rollouts)")
    axes[0].set_ylabel("seconds/search")
    axes[1].set_title(f"Search calls per rollout moving average ({window} rollouts)")
    axes[1].set_ylabel("search events/rollout")
    for ax in axes:
        ax.legend(loc="best")
        finish_time_axis(ax)
    path = output_dir / "trace_search_latency_ma.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_trace_helper_action_latency(output_dir: Path, action_rows: list[dict[str, Any]], window: int) -> str | None:
    if not action_rows:
        return None
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    labels = sorted({str(row.get("label")) for row in action_rows if row.get("label")})
    plotted = False
    for label in labels:
        rows = sorted([row for row in action_rows if row.get("label") == label], key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
        for action_type, line_style in (("web_search", "-"), ("local_document_search", "--")):
            action_specific = [row for row in rows if row.get("action_type") == action_type]
            paired = [
                (datetime.fromtimestamp(float(row["completed_at_s"]), tz=timezone.utc), row)
                for row in action_specific
                if as_float(row.get("completed_at_s")) is not None
            ]
            if not paired:
                continue
            xs = [item[0] for item in paired]
            ys = rolling([as_float(item[1].get("execution_time_s")) for item in paired], window)
            axes[0].plot(xs, ys, linestyle=line_style, linewidth=1.9, label=f"{label} {action_type}")
            plotted = True

        rollout_counts: dict[tuple[float, str], int] = {}
        for row in rows:
            completed_at = as_float(row.get("completed_at_s"))
            action_type = str(row.get("action_type") or "")
            if completed_at is None:
                continue
            rollout_counts[(completed_at, action_type)] = rollout_counts.get((completed_at, action_type), 0) + 1
        for action_type, line_style in (("web_search", "-"), ("local_document_search", "--")):
            pairs = sorted((completed_at, count) for (completed_at, kind), count in rollout_counts.items() if kind == action_type)
            if not pairs:
                continue
            xs = [datetime.fromtimestamp(completed_at, tz=timezone.utc) for completed_at, _ in pairs]
            ys = rolling([float(count) for _, count in pairs], window)
            axes[1].plot(xs, ys, linestyle=line_style, linewidth=1.8, label=f"{label} {action_type}")
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title(f"Trace helper action execution time moving average ({window} calls)")
    axes[0].set_ylabel("seconds/call")
    axes[1].set_title(f"Helper actions per rollout moving average ({window} rollouts)")
    axes[1].set_ylabel("actions/rollout")
    for ax in axes:
        ax.legend(loc="best")
        finish_time_axis(ax)
    path = output_dir / "trace_helper_action_latency_ma.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_trace_stage_duration(output_dir: Path, stage_rows: list[dict[str, Any]], window: int) -> str | None:
    if not stage_rows:
        return None
    wanted = ["hop_plan", "search", "doc_choose", "doc_read", "hop_resolve", "report"]
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    labels = sorted({str(row.get("label")) for row in stage_rows if row.get("label")})
    plotted = False
    for label in labels:
        rows = sorted([row for row in stage_rows if row.get("label") == label], key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
        for stage in wanted:
            stage_specific = [row for row in rows if row.get("stage") == stage]
            if not stage_specific:
                continue
            paired = [
                (datetime.fromtimestamp(float(row["completed_at_s"]), tz=timezone.utc), row)
                for row in stage_specific
                if as_float(row.get("completed_at_s")) is not None
            ]
            if not paired:
                continue
            xs = [item[0] for item in paired]
            ys = rolling([as_float(item[1].get("duration_s")) for item in paired], window)
            target_ax = axes[0] if stage in {"hop_plan", "doc_choose", "doc_read", "hop_resolve", "report"} else axes[1]
            target_ax.plot(xs, ys, linewidth=1.7, label=f"{label} {stage}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title(f"LLM-heavy stage duration moving average ({window} events)")
    axes[0].set_ylabel("seconds/event")
    axes[1].set_title(f"Search/helper stage duration moving average ({window} events)")
    axes[1].set_ylabel("seconds/event")
    for ax in axes:
        ax.legend(loc="best", ncol=2)
        finish_time_axis(ax)
    path = output_dir / "trace_stage_duration_ma.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def latest_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("label") or "")
        if not label:
            continue
        latest[label] = row
    return latest


def latest_helper_metrics(helper_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in helper_rows:
        endpoint = str(row.get("endpoint") or "")
        if endpoint:
            latest[endpoint] = row
    return latest


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def card(title: str, lines: list[tuple[str, Any]]) -> str:
    rows = "\n".join(
        f"<div class='metric'><span>{esc(name)}</span><strong>{esc(value)}</strong></div>" for name, value in lines
    )
    return f"<section class='card'><h3>{esc(title)}</h3>{rows}</section>"


def error_block(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return ""
    errors = snapshot.get("last_errors") or []
    if not errors:
        return "<p class='okline'>No error markers in current EAI log tail.</p>"
    items = "\n".join(f"<li><code>{esc(error)}</code></li>" for error in errors)
    return f"<ul class='errors'>{items}</ul>"


def build_html(
    output_dir: Path,
    runs: list[RunSpec],
    finetune_rows_by_run: dict[str, list[dict[str, Any]]],
    run_snapshots: list[dict[str, Any]],
    helper_flat_rows: list[dict[str, Any]],
    trace_rollout_rows: list[dict[str, Any]],
    trace_action_rows: list[dict[str, Any]],
    plot_names: list[str],
    now: datetime,
    helper_url: str,
) -> None:
    latest_snapshots = latest_by_label(run_snapshots)
    latest_helper = latest_helper_metrics(helper_flat_rows)
    latest_trace = latest_by_label(trace_rollout_rows)

    run_cards: list[str] = []
    for run in runs:
        finetune_latest = finetune_rows_by_run.get(run.label, [])[-1] if finetune_rows_by_run.get(run.label) else {}
        snapshot = latest_snapshots.get(run.label, {})
        trace_latest = latest_trace.get(run.label, {})
        run_cards.append(
            card(
                run.label,
                [
                    ("job", run.job_id),
                    ("completed step", fmt_num(finetune_latest.get("step"))),
                    ("trained samples", fmt_num(finetune_latest.get("stats/samples"))),
                    ("reward", fmt_num(finetune_latest.get("rl/reward"), 3)),
                    ("advantage", fmt_num(finetune_latest.get("rl/advantage"), 3)),
                    ("tokens/GPU/sec", fmt_num(finetune_latest.get("throughput/tokens_perGPU_per_sec"))),
                    ("sec/step", fmt_num(finetune_latest.get("throughput/sec_per_step"))),
                    ("queue batches", fmt_num(finetune_latest.get("stats/queue/batches"))),
                    ("accepted total", fmt_num(snapshot.get("latest_accepted_total"))),
                    ("actor total", fmt_num(snapshot.get("latest_actor_total"))),
                    ("recent filter rate", fmt_pct(snapshot.get("recent_filter_rate"))),
                    ("recent zero-filtered", fmt_num(snapshot.get("recent_zero_advantage_filtered_samples"))),
                    ("trace rollouts parsed", fmt_num(len([row for row in trace_rollout_rows if row.get("label") == run.label]))),
                    ("trace search mean", fmt_num(trace_latest.get("search_all_mean_s"), 3) + "s"),
                    ("trace web search mean", fmt_num(trace_latest.get("search_web_mean_s"), 3) + "s"),
                    ("trace local search mean", fmt_num(trace_latest.get("search_local_mean_s"), 3) + "s"),
                    ("helper action mean", fmt_num(trace_latest.get("helper_action_all_mean_s"), 3) + "s"),
                    ("helper web action mean", fmt_num(trace_latest.get("helper_action_web_search_mean_s"), 3) + "s"),
                    ("helper local action mean", fmt_num(trace_latest.get("helper_action_local_document_search_mean_s"), 3) + "s"),
                    ("tail errors", fmt_num(snapshot.get("error_count_in_tail"))),
                ],
            )
        )

    helper_cards: list[str] = []
    for endpoint in ("/browsecomp/search", "/local/search"):
        row = latest_helper.get(endpoint, {})
        helper_cards.append(
            card(
                endpoint,
                [
                    ("count", fmt_num(row.get("count"))),
                    ("active", fmt_num(row.get("active"))),
                    ("errors", fmt_num(row.get("errors"))),
                    ("p50 ms", fmt_num(row.get("p50_ms"))),
                    ("p95 ms", fmt_num(row.get("p95_ms"))),
                    ("max ms", fmt_num(row.get("max_ms"))),
                ],
            )
        )

    plot_html = "\n".join(
        f"<figure><img src='{esc(name)}' alt='{esc(name)}'><figcaption>{esc(name)}</figcaption></figure>"
        for name in plot_names
        if name
    )

    error_sections = "\n".join(
        f"<section class='panel'><h3>{esc(run.label)} log-tail error markers</h3>{error_block(latest_snapshots.get(run.label))}</section>"
        for run in runs
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>privacy_hopqa live training monitor</title>
  <style>
    :root {{
      --ink: #17212b;
      --muted: #64748b;
      --line: #d8e0e8;
      --card: rgba(255, 255, 255, 0.88);
      --accent: #184e77;
      --accent-2: #d95d39;
      --bg-a: #f5f1e8;
      --bg-b: #dcebf2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 5%, rgba(217, 93, 57, 0.24), transparent 32rem),
        radial-gradient(circle at 92% 6%, rgba(24, 78, 119, 0.22), transparent 28rem),
        linear-gradient(135deg, var(--bg-a), var(--bg-b));
    }}
    header {{
      padding: 34px 5vw 20px;
      border-bottom: 1px solid rgba(23, 33, 43, 0.12);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 4.4rem);
      line-height: 0.95;
      letter-spacing: -0.06em;
    }}
    .subhead {{
      max-width: 980px;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.55;
    }}
    main {{
      padding: 24px 5vw 60px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card, .panel, figure {{
      background: var(--card);
      border: 1px solid rgba(23, 33, 43, 0.10);
      border-radius: 22px;
      box-shadow: 0 18px 55px rgba(23, 33, 43, 0.10);
      backdrop-filter: blur(10px);
    }}
    .card {{
      padding: 18px;
    }}
    .card h3, .panel h3 {{
      margin: 0 0 14px;
      font-size: 1.02rem;
      letter-spacing: -0.02em;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 8px 0;
      border-top: 1px solid rgba(23, 33, 43, 0.08);
      font-size: 0.92rem;
    }}
    .metric span {{
      color: var(--muted);
    }}
    .metric strong {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      padding: 14px;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 14px;
      background: white;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 8px;
    }}
    .panel {{
      margin-top: 18px;
      padding: 18px;
    }}
    .errors {{
      margin: 0;
      padding-left: 20px;
      color: #9f1239;
    }}
    code {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .okline {{
      color: #0f766e;
      margin: 0;
    }}
    a {{
      color: var(--accent);
    }}
    @media (max-width: 760px) {{
      .plots {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>privacy_hopqa live monitor</h1>
    <p class="subhead">
      Last refreshed {esc(iso(now))}. This dashboard tracks both scheduler-fix training runs, recent
      zero-advantage filtering, finetune progress, service-level helper latency sampled from
      <code>{esc(helper_url.rstrip('/') + '/health')}</code>, and per-rollout search-stage timings
      from <code>privacy_hopqa_runs/*/timeline_events.json</code>. It also reads
      <code>trace.json/actions[*].execution_time</code> for the cleanest per-helper-call duration.
      The helper service endpoint still exposes p50/p95/max rather than a true arithmetic mean.
    </p>
  </header>
  <main>
    <h2>Runs</h2>
    <div class="grid">{''.join(run_cards)}</div>
    <h2>Helper</h2>
    <div class="grid">{''.join(helper_cards)}</div>
    <h2>Plots</h2>
    <section class="plots">{plot_html}</section>
    {error_sections}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")

    portable_text = html_text
    for name in plot_names:
        image_path = output_dir / name
        if not image_path.exists():
            continue
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        portable_text = portable_text.replace(f"src='{esc(name)}'", f"src='data:image/png;base64,{data}'")
    portable_text = portable_text.replace("<meta http-equiv=\"refresh\" content=\"60\">\n", "")
    portable_text = portable_text.replace(
        "This dashboard tracks both scheduler-fix training runs, recent",
        "Portable snapshot. This dashboard tracks both scheduler-fix training runs, recent",
    )
    portable_name = "privacy_hopqa_schedulerfix_live_portable.html"
    (output_dir / portable_name).write_text(portable_text, encoding="utf-8")
    (output_dir.parent / portable_name).write_text(portable_text, encoding="utf-8")


def refresh(args: argparse.Namespace, runs: list[RunSpec]) -> None:
    now = utc_now()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_helper(output_dir, args.helper_url, now)
    snapshot_eai_runs(output_dir, runs, args.tail_lines, now)
    cached_stage_rows = read_csv_rows(output_dir / "trace_stage_timings.csv")
    cached_rollout_rows = read_csv_rows(output_dir / "trace_rollout_timings.csv")
    cached_action_rows = read_csv_rows(output_dir / "trace_helper_actions.csv")
    seen_rollouts = {
        (str(row.get("label") or ""), str(row.get("rollout_id") or ""))
        for row in cached_rollout_rows
        if row.get("label") and row.get("rollout_id")
    }
    new_stage_rows, new_rollout_rows, new_action_rows = parse_trace_timings(runs, seen_rollouts, args.new_trace_limit)
    trace_stage_rows, trace_rollout_rows, trace_action_rows = trim_trace_cache(
        cached_stage_rows + new_stage_rows,
        cached_rollout_rows + new_rollout_rows,
        cached_action_rows + new_action_rows,
        args.trace_cache_limit,
    )

    finetune_rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        rows = parse_finetune_log(run)
        finetune_rows_by_run[run.label] = rows
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.label).strip("_").lower()
        write_csv(output_dir / f"finetune_{safe_label}.csv", rows)

    run_snapshots = read_jsonl(output_dir / "run_snapshots.jsonl")
    helper_snapshots = read_jsonl(output_dir / "helper_snapshots.jsonl")
    helper_flat = flatten_helper_snapshots(helper_snapshots)
    write_csv(output_dir / "run_snapshots.csv", run_snapshots)
    write_csv(output_dir / "helper_metrics.csv", helper_flat)
    write_csv(output_dir / "trace_stage_timings.csv", trace_stage_rows)
    write_csv(output_dir / "trace_rollout_timings.csv", trace_rollout_rows)
    write_csv(output_dir / "trace_helper_actions.csv", trace_action_rows)

    plot_names = [
        plot_training_progress(output_dir, finetune_rows_by_run),
        plot_reward_throughput(output_dir, finetune_rows_by_run, args.rolling_window),
        plot_filtering(output_dir, run_snapshots, args.rolling_window),
        plot_helper_latency(output_dir, helper_flat, args.rolling_window),
        plot_trace_helper_action_latency(output_dir, trace_action_rows, args.rolling_window),
        plot_trace_search_latency(output_dir, trace_rollout_rows, args.rolling_window),
        plot_trace_stage_duration(output_dir, trace_stage_rows, args.rolling_window),
    ]
    build_html(
        output_dir=output_dir,
        runs=runs,
        finetune_rows_by_run=finetune_rows_by_run,
        run_snapshots=run_snapshots,
        helper_flat_rows=helper_flat,
        trace_rollout_rows=trace_rollout_rows,
        trace_action_rows=trace_action_rows,
        plot_names=[name for name in plot_names if name],
        now=now,
        helper_url=args.helper_url,
    )


def main() -> None:
    args = parse_args()
    run_specs = args.run or DEFAULT_RUNS
    runs = [parse_run(spec) for spec in run_specs]
    while True:
        refresh(args, runs)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
