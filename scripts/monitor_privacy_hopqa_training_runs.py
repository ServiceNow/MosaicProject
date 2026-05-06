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
import random
import re
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
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
DEFAULT_DATASET_PATH = "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-05.jsonl"
DEFAULT_RUNS = [
    "8B|a1161525-4e2e-4e43-b9bf-d83653ae1d57|/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_final_qwen3_8b_r2_filter_vllm18_schedulerfix_20260506_152928",
    "4B|fbe9aa12-9c2b-43e3-9c45-7e6489add3e4|/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_final_qwen3_4b_instruct_2507_r2_filter_vllm18_schedulerfix_20260506_152949",
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
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Final privacy_hopqa JSONL dataset.")
    parser.add_argument("--samples-per-phase", type=int, default=5, help="Random rollout samples per run per early/middle/late phase.")
    return parser.parse_args()


def parse_run(spec: str) -> RunSpec:
    parts = spec.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"Run spec must be LABEL|JOB_ID|RUN_DIR, got: {spec}")
    label, job_id, run_dir = parts
    return RunSpec(label=short_run_label(label.strip()), job_id=job_id.strip(), run_dir=Path(run_dir.strip()))


def short_run_label(label: str) -> str:
    lowered = label.lower()
    if "4b" in lowered:
        return "4B"
    if "8b" in lowered:
        return "8B"
    return label


def normalize_row_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        if copied.get("label"):
            copied["label"] = short_run_label(str(copied["label"]))
        normalized.append(copied)
    return normalized


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


def load_dataset_index(dataset_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = str(row.get("task_id") or "")
            chain_id = str(row.get("chain_id") or "")
            if not task_id or not chain_id:
                continue
            hop_sources: dict[int, str] = {}
            for hop in row.get("hops") or []:
                if not isinstance(hop, dict):
                    continue
                hop_number = as_int(hop.get("hop_number"))
                doc_id = str(hop.get("doc_id") or "")
                hop_type = str(hop.get("hop_type") or "")
                source = "unknown"
                marker = f"{doc_id} {hop_type}".lower()
                if marker.startswith("local/") or "local" in marker:
                    source = "local"
                elif marker.startswith("web/") or "web" in marker:
                    source = "web"
                if hop_number is not None:
                    hop_sources[hop_number] = source
            pattern = str(row.get("pattern") or "")
            for pos, char in enumerate(pattern, start=1):
                if pos not in hop_sources:
                    if char.upper() == "L":
                        hop_sources[pos] = "local"
                    elif char.upper() == "W":
                        hop_sources[pos] = "web"
            index[(task_id, chain_id)] = {
                "company": row.get("company"),
                "pattern": pattern,
                "n_hops": row.get("n_hops"),
                "n_local_hops": row.get("n_local_hops"),
                "n_web_hops": row.get("n_web_hops"),
                "hop_sources": hop_sources,
            }
    return index


def source_matches(actual: str | None, expected: str | None) -> bool | None:
    if not actual or not expected or actual == "unknown" or expected == "unknown":
        return None
    return actual == expected


def parse_trace_timings(
    runs: list[RunSpec],
    seen_rollouts: set[tuple[str, str]],
    new_trace_limit: int,
    dataset_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
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

            task_id = str(summary.get("task_id") or "")
            chain_id = str(summary.get("chain_id") or "")
            dataset_meta = dataset_index.get((task_id, chain_id), {})
            hop_sources = dataset_meta.get("hop_sources") if isinstance(dataset_meta.get("hop_sources"), dict) else {}
            base = {
                "label": run.label,
                "job_id": run.job_id,
                "run_dir": str(run.run_dir),
                "rollout_id": rollout_id,
                "task_id": task_id,
                "chain_id": chain_id,
                "company": dataset_meta.get("company"),
                "hop_pattern": dataset_meta.get("pattern"),
                "completed_at_s": completed_at_s,
                "timestamp": iso(datetime.fromtimestamp(completed_at_s, tz=timezone.utc)),
            }

            stage_values: dict[str, list[float]] = {}
            search_values: dict[str, list[float]] = {"web": [], "local": [], "unknown": [], "all": []}
            action_values: dict[str, list[float]] = {"web_search": [], "local_document_search": [], "all": []}
            action_hop: dict[str, dict[str, Any]] = {}
            source_call_counts = Counter()
            aligned_source_calls = 0
            known_source_calls = 0
            for event in events:
                if not isinstance(event, dict):
                    continue
                duration = event_duration_s(event)
                if duration is None:
                    continue
                stage = str(event.get("stage") or "unknown")
                meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}
                hop_number = as_int(meta.get("hop_number"))
                iteration = as_int(meta.get("iteration"))
                expected_source = hop_sources.get(hop_number) if hop_number is not None else None
                stage_values.setdefault(stage, []).append(duration)
                row = {
                    **base,
                    "stage": stage,
                    "duration_s": duration,
                    "lane": event.get("lane"),
                    "event_label": event.get("label"),
                    "hop_number": hop_number,
                    "iteration": iteration,
                    "expected_source": expected_source,
                    "event_action_type": meta.get("action_type"),
                    "doc_source": meta.get("source"),
                    "doc_id": meta.get("doc_id"),
                    "result_count": meta.get("result_count"),
                    "planned_action_count": len(meta.get("planned_actions") or [])
                    if isinstance(meta.get("planned_actions"), list)
                    else None,
                }
                if stage == "search":
                    backend = search_backend(event)
                    matched = source_matches(backend, expected_source)
                    row["search_backend"] = backend
                    row["source_matched_expected"] = matched
                    search_values.setdefault(backend, []).append(duration)
                    search_values["all"].append(duration)
                    source_call_counts[(expected_source or "unknown", backend)] += 1
                    if matched is not None:
                        known_source_calls += 1
                        aligned_source_calls += int(matched)
                    action_id = meta.get("action_id")
                    if action_id:
                        action_hop[str(action_id)] = {
                            "hop_number": hop_number,
                            "iteration": iteration,
                            "expected_source": expected_source,
                            "actual_source": backend,
                            "source_matched_expected": matched,
                        }
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
                    results = actual_output.get("results") if isinstance(actual_output.get("results"), list) else []
                    top_results: list[str] = []
                    for result in results[:3]:
                        if isinstance(result, dict):
                            top_results.append(
                                f"{result.get('doc_id') or ''} | {truncate_text(result.get('title') or result.get('snippet'), 180)}"
                            )
                    action_id = str(action.get("id") or "")
                    mapped = action_hop.get(action_id, {})
                    actual_source = "web" if action_type == "web_search" else "local" if action_type == "local_document_search" else "unknown"
                    expected_source = mapped.get("expected_source")
                    matched = source_matches(actual_source, expected_source)
                    action_values.setdefault(action_type, []).append(execution_time)
                    action_values["all"].append(execution_time)
                    action_rows.append(
                        {
                            **base,
                            "action_id": action_id,
                            "action_type": action_type,
                            "actual_source": actual_source,
                            "hop_number": mapped.get("hop_number"),
                            "iteration": action.get("iteration") if action.get("iteration") is not None else mapped.get("iteration"),
                            "expected_source": expected_source,
                            "source_matched_expected": matched,
                            "tool": actual_output.get("tool"),
                            "status": action.get("status"),
                            "execution_time_s": execution_time,
                            "query": actual_output.get("query")
                            or ((action.get("parameters") or {}).get("query") if isinstance(action.get("parameters"), dict) else None),
                            "success": actual_output.get("success"),
                            "results_count": actual_output.get("results_count"),
                            "top_results": " || ".join(top_results),
                        }
                    )

            for kind, values in (
                ("errors_by_stage", summary.get("errors_by_stage")),
                ("errors_by_kind", summary.get("errors_by_kind")),
                ("parse_errors_by_stage", summary.get("parse_errors_by_stage")),
            ):
                if not isinstance(values, dict):
                    continue
                for key, count in values.items():
                    error_rows.append({**base, "error_group": kind, "error_key": key, "count": count})
            trace_errors = trace.get("error_records") if isinstance(trace, dict) else None
            if isinstance(trace_errors, list):
                for error in trace_errors:
                    if isinstance(error, dict):
                        error_rows.append(
                            {
                                **base,
                                "error_group": "error_records",
                                "error_key": error.get("stage") or error.get("kind") or "unknown",
                                "count": 1,
                                "stage": error.get("stage"),
                                "kind": error.get("kind"),
                                "message": error.get("message") or error.get("error"),
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
                "source_alignment_known_calls": known_source_calls,
                "source_alignment_matched_calls": aligned_source_calls,
                "source_alignment_rate": (aligned_source_calls / known_source_calls) if known_source_calls else None,
            }
            for (expected_source, actual_source), count in source_call_counts.items():
                rollout_row[f"search_expected_{expected_source}_actual_{actual_source}_count"] = count
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
    return stage_rows, rollout_rows, action_rows, error_rows


def trim_trace_cache(
    stage_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    cache_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    kept_errors = [row for row in error_rows if keep(row)]
    return kept_stage, kept_rollouts, kept_actions, kept_errors


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


def label_values(rows: list[dict[str, Any]], label: str, key: str) -> list[float]:
    return [value for row in rows if row.get("label") == label and (value := as_float(row.get(key))) is not None]


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def plot_classic_search_doc_counts(
    output_dir: Path, rollout_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]
) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    metrics = [
        ("searches_total", "searches"),
        ("docs_read", "docs read"),
        ("search_web_count", "web searches"),
        ("search_local_count", "local searches"),
    ]
    doc_read_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in stage_rows:
        if row.get("stage") != "doc_read":
            continue
        label = str(row.get("label") or "")
        rollout_id = str(row.get("rollout_id") or "")
        source = str(row.get("doc_source") or "unknown")
        doc_read_counts[(label, rollout_id)][source] += 1
    derived_rollout_rows: list[dict[str, Any]] = []
    for row in rollout_rows:
        label = str(row.get("label") or "")
        rollout_id = str(row.get("rollout_id") or "")
        copied = dict(row)
        copied["doc_read_web_count"] = doc_read_counts[(label, rollout_id)].get("web", 0)
        copied["doc_read_local_count"] = doc_read_counts[(label, rollout_id)].get("local", 0)
        derived_rollout_rows.append(copied)
    metrics.extend([("doc_read_web_count", "web docs read"), ("doc_read_local_count", "local docs read")])

    plot_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.8 / max(1, len(labels))
    xs = list(range(len(metrics)))
    for idx, label in enumerate(labels):
        means = [mean_or_none(label_values(derived_rollout_rows, label, key)) or 0.0 for key, _ in metrics]
        offsets = [x - 0.4 + width / 2 + idx * width for x in xs]
        ax.bar(offsets, means, width=width, label=label)
    ax.set_title("Mean search and document-read counts per rollout")
    ax.set_ylabel("mean count")
    ax.set_xticks(xs)
    ax.set_xticklabels([name for _, name in metrics], rotation=25, ha="right")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_search_doc_counts.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_rollout_distributions(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    metrics = [
        ("duration_s", "rollout duration (s)"),
        ("llm_calls_total", "LLM calls"),
        ("searches_total", "searches"),
        ("docs_read", "docs read"),
    ]
    plot_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flat, metrics):
        data = [label_values(rollout_rows, label, key) for label in labels]
        data = [values for values in data if values]
        used_labels = [label for label in labels if label_values(rollout_rows, label, key)]
        if data:
            ax.boxplot(data, tick_labels=used_labels, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_rollout_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_error_breakdown(output_dir: Path, error_rows: list[dict[str, Any]]) -> str | None:
    if not error_rows:
        return None
    counter: Counter[tuple[str, str]] = Counter()
    for row in error_rows:
        label = str(row.get("label") or "unknown")
        key = f"{row.get('error_group') or 'error'}:{row.get('error_key') or 'unknown'}"
        counter[(label, key)] += as_int(row.get("count")) or 0
    if not counter:
        return None
    labels = sorted({label for label, _ in counter})
    keys = [key for _, key in counter.most_common(12)]
    keys = list(dict.fromkeys(keys))[:12]
    plot_setup()
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(keys))))
    y = list(range(len(keys)))
    width = 0.8 / max(1, len(labels))
    for idx, label in enumerate(labels):
        values = [counter.get((label, key), 0) for key in keys]
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in y]
        ax.barh(offsets, values, height=width, label=label)
    ax.set_title("Error breakdown from trace summaries")
    ax.set_xlabel("count")
    ax.set_yticks(y)
    ax.set_yticklabels(keys)
    ax.legend(loc="best")
    ax.grid(True, axis="x", alpha=0.25)
    path = output_dir / "classic_error_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_parse_and_empty_plan_breakdown(
    output_dir: Path, stage_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]]
) -> str | None:
    labels = sorted(
        {
            str(row.get("label"))
            for row in list(stage_rows) + list(error_rows)
            if row.get("label")
        }
    )
    if not labels:
        return None
    counter: Counter[tuple[str, str]] = Counter()
    for row in stage_rows:
        if row.get("stage") != "hop_plan":
            continue
        label = str(row.get("label") or "unknown")
        planned = as_float(row.get("planned_action_count"))
        if planned == 0:
            counter[(label, "empty hop-plan output accepted as []")] += 1
    for row in error_rows:
        if row.get("error_group") != "parse_errors_by_stage":
            continue
        count = as_int(row.get("count")) or 0
        if count <= 0:
            continue
        label = str(row.get("label") or "unknown")
        stage = str(row.get("error_key") or "unknown")
        counter[(label, f"parse error in {stage}")] += count
    if not counter:
        return None
    keys = [
        "empty hop-plan output accepted as []",
        "parse error in hop_plan",
        "parse error in doc_choose",
        "parse error in doc_read",
        "parse error in hop_resolve",
        "parse error in report",
        "parse error in unknown",
    ]
    remaining = [
        key
        for _, key in counter
        if key not in keys
    ]
    keys = [key for key in keys if any(counter.get((label, key), 0) for label in labels)]
    keys.extend(sorted(set(remaining)))
    plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.48 * len(keys))))
    y = list(range(len(keys)))
    width = 0.8 / max(1, len(labels))
    for idx, label in enumerate(labels):
        values = [counter.get((label, key), 0) for key in keys]
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in y]
        axes[0].barh(offsets, values, height=width, label=label)
    axes[0].set_title("Raw parser/planner failure counts")
    axes[0].set_xlabel("events")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(keys)
    axes[0].legend(loc="best")
    axes[0].grid(True, axis="x", alpha=0.25)

    rate_keys = ["empty hop-plan output accepted as []"]
    rate_labels: list[str] = []
    rates: list[float] = []
    for label in labels:
        plans = [
            row
            for row in stage_rows
            if row.get("label") == label and row.get("stage") == "hop_plan"
        ]
        total = len(plans)
        empty = counter.get((label, rate_keys[0]), 0)
        if total:
            rate_labels.append(f"{label}\n{empty}/{total}")
            rates.append(empty / total)
    axes[1].bar(range(len(rates)), rates, color="#b45f06")
    axes[1].set_title("Empty hop-plan output rate")
    axes[1].set_ylabel("empty hop_plan events / hop_plan events")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(rate_labels)))
    axes[1].set_xticklabels(rate_labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Hop parsing errors and accepted empty planner outputs by model")
    path = output_dir / "classic_parse_and_empty_plan_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_search_source_confusion(output_dir: Path, stage_rows: list[dict[str, Any]]) -> str | None:
    rows = [
        row
        for row in stage_rows
        if row.get("stage") == "search"
        and str(row.get("expected_source") or "unknown") in {"local", "web"}
        and str(row.get("search_backend") or "unknown") in {"local", "web"}
    ]
    if not rows:
        return None
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    categories = [
        ("local", "local", "expected local -> local"),
        ("local", "web", "expected local -> web"),
        ("web", "local", "expected web -> local"),
        ("web", "web", "expected web -> web"),
    ]
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[(str(row.get("label")), str(row.get("expected_source")), str(row.get("search_backend")))] += 1
    plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    xs = list(range(len(categories)))
    width = 0.8 / max(1, len(labels))
    for idx, label in enumerate(labels):
        values = [counts.get((label, expected, actual), 0) for expected, actual, _ in categories]
        offsets = [x - 0.4 + width / 2 + idx * width for x in xs]
        axes[0].bar(offsets, values, width=width, label=label)
    axes[0].set_title("Search source confusion counts")
    axes[0].set_ylabel("search events")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([name for _, _, name in categories], rotation=25, ha="right")
    axes[0].set_yscale("log")
    axes[0].legend(loc="best")
    axes[0].grid(True, axis="y", alpha=0.25)

    frac_labels: list[str] = []
    frac_values: list[float] = []
    colors: list[str] = []
    for label in labels:
        for expected in ("local", "web"):
            expected_total = sum(counts.get((label, expected, actual), 0) for actual in ("local", "web"))
            for actual in ("local", "web"):
                frac_labels.append(f"{label}\nexpected {expected}\nactual {actual}")
                frac_values.append(counts.get((label, expected, actual), 0) / expected_total if expected_total else 0.0)
                colors.append("#1f5da8" if expected == actual else "#b45f06")
    axes[1].bar(range(len(frac_values)), frac_values, color=colors)
    axes[1].set_title("Actual backend share within each expected source")
    axes[1].set_ylabel("fraction of expected-source search events")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(frac_labels)))
    axes[1].set_xticklabels(frac_labels, rotation=30, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Expected hop source vs actual search backend")
    path = output_dir / "classic_search_source_confusion.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_doc_read_source_totals(output_dir: Path, stage_rows: list[dict[str, Any]]) -> str | None:
    rows = [
        row
        for row in stage_rows
        if row.get("stage") == "doc_read" and str(row.get("doc_source") or "unknown") in {"local", "web"}
    ]
    if not rows:
        return None
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    sources = ["local", "web"]
    counts: Counter[tuple[str, str]] = Counter((str(row.get("label")), str(row.get("doc_source"))) for row in rows)
    plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / max(1, len(labels))
    xs = list(range(len(sources)))
    for idx, label in enumerate(labels):
        values = [counts.get((label, source), 0) for source in sources]
        offsets = [x - 0.4 + width / 2 + idx * width for x in xs]
        axes[0].bar(offsets, values, width=width, label=label)
    axes[0].set_title("Document reads by source")
    axes[0].set_ylabel("doc_read events")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["local docs read", "web docs read"])
    axes[0].legend(loc="best")
    axes[0].grid(True, axis="y", alpha=0.25)

    frac_labels: list[str] = []
    frac_values: list[float] = []
    for label in labels:
        total = sum(counts.get((label, source), 0) for source in sources)
        for source in sources:
            frac_labels.append(f"{label}\n{source}")
            frac_values.append(counts.get((label, source), 0) / total if total else 0.0)
    axes[1].bar(range(len(frac_values)), frac_values, color=["#1f5da8", "#14866d"] * len(labels))
    axes[1].set_title("Document-read source share")
    axes[1].set_ylabel("fraction of doc_read events")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(frac_labels)))
    axes[1].set_xticklabels(frac_labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_doc_read_source_totals.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_source_alignment_by_hop(output_dir: Path, stage_rows: list[dict[str, Any]]) -> str | None:
    search_rows = [row for row in stage_rows if row.get("stage") == "search"]
    if not search_rows:
        return None
    labels = sorted({str(row.get("label")) for row in search_rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plotted = False
    for label in labels:
        rows = [row for row in search_rows if row.get("label") == label]
        hop_numbers = sorted({as_int(row.get("hop_number")) for row in rows if as_int(row.get("hop_number")) is not None})
        if not hop_numbers:
            continue
        alignment: list[float | None] = []
        local_share: list[float | None] = []
        expected_local: list[float | None] = []
        for hop_number in hop_numbers:
            hop_rows = [row for row in rows if as_int(row.get("hop_number")) == hop_number]
            known = [row for row in hop_rows if str(row.get("expected_source") or "unknown") != "unknown"]
            matched = [row for row in known if str(row.get("source_matched_expected")).lower() == "true"]
            alignment.append(len(matched) / len(known) if known else None)
            web = sum(1 for row in hop_rows if row.get("search_backend") == "web")
            local = sum(1 for row in hop_rows if row.get("search_backend") == "local")
            local_share.append(local / (local + web) if (local + web) else None)
            expected_local.append(sum(1 for row in known if row.get("expected_source") == "local") / len(known) if known else None)
        axes[0].plot(hop_numbers, alignment, marker="o", linewidth=2.0, label=label)
        axes[1].plot(hop_numbers, local_share, marker="o", linewidth=2.0, label=f"{label} actual local share")
        axes[1].plot(hop_numbers, expected_local, linestyle="--", linewidth=1.5, label=f"{label} expected local share")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title("Executed search calls that used the expected source by hop position")
    axes[0].set_ylabel("fraction matching expected source")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Among executed searches: actual local-search share vs expected-local source share")
    axes[1].set_ylabel("local share")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("hop number within rollout")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    path = output_dir / "classic_source_alignment_by_hop.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_source_calls_by_iteration(output_dir: Path, stage_rows: list[dict[str, Any]]) -> str | None:
    search_rows = [row for row in stage_rows if row.get("stage") == "search"]
    if not search_rows:
        return None
    labels = sorted({str(row.get("label")) for row in search_rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plotted = False
    for label in labels:
        rows = [row for row in search_rows if row.get("label") == label]
        iterations = sorted({as_int(row.get("iteration")) for row in rows if as_int(row.get("iteration")) is not None})
        if not iterations:
            continue
        local_share: list[float | None] = []
        searches_per_rollout: list[float | None] = []
        for iteration in iterations:
            iter_rows = [row for row in rows if as_int(row.get("iteration")) == iteration]
            web = sum(1 for row in iter_rows if row.get("search_backend") == "web")
            local = sum(1 for row in iter_rows if row.get("search_backend") == "local")
            rollout_count = len({row.get("rollout_id") for row in iter_rows if row.get("rollout_id")})
            local_share.append(local / (local + web) if (local + web) else None)
            searches_per_rollout.append((local + web) / rollout_count if rollout_count else None)
        axes[0].plot(iterations, local_share, marker="o", linewidth=2.0, label=label)
        axes[1].plot(iterations, searches_per_rollout, marker="o", linewidth=2.0, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title("Among executed searches: local-search share by rollout iteration")
    axes[0].set_ylabel("local share")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Search calls per active rollout by iteration")
    axes[1].set_ylabel("searches / rollout")
    axes[1].set_xlabel("iteration within rollout")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    path = output_dir / "classic_source_calls_by_iteration.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_stage_duration_boxplot(output_dir: Path, stage_rows: list[dict[str, Any]]) -> str | None:
    rows = [row for row in stage_rows if as_float(row.get("duration_s")) is not None]
    if not rows:
        return None
    stages = ["hop_plan", "search", "doc_choose", "doc_read", "hop_resolve", "report"]
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, max(4, 3.5 * len(labels))), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        data = []
        used_stages = []
        for stage in stages:
            values = [
                as_float(row.get("duration_s"))
                for row in rows
                if row.get("label") == label and row.get("stage") == stage and as_float(row.get("duration_s")) is not None
            ]
            values = [value for value in values if value is not None]
            if values:
                data.append(values)
                used_stages.append(stage)
        if data:
            ax.boxplot(data, tick_labels=used_stages, showfliers=False)
        ax.set_title(f"{label}: stage duration distribution")
        ax.set_ylabel("seconds")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_stage_duration_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_source_usage_distributions(
    output_dir: Path, rollout_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]
) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    doc_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in stage_rows:
        if row.get("stage") == "doc_read":
            doc_counts[(str(row.get("label") or ""), str(row.get("rollout_id") or ""))][str(row.get("doc_source") or "unknown")] += 1
    enriched = []
    for row in rollout_rows:
        copied = dict(row)
        key = (str(row.get("label") or ""), str(row.get("rollout_id") or ""))
        copied["web_docs_read_trace"] = doc_counts[key].get("web", 0)
        copied["local_docs_read_trace"] = doc_counts[key].get("local", 0)
        enriched.append(copied)
    metrics = [
        ("search_web_count", "web searches"),
        ("search_local_count", "local searches"),
        ("web_docs_read_trace", "web docs read"),
        ("local_docs_read_trace", "local docs read"),
    ]
    plot_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flat, metrics):
        data = [label_values(enriched, label, key) for label in labels]
        used = [(label, values) for label, values in zip(labels, data) if values]
        if used:
            ax.boxplot([values for _, values in used], tick_labels=[label for label, _ in used], showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_source_usage_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_token_usage(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    metrics = [
        ("prompt_tokens", "prompt tokens"),
        ("output_tokens", "output tokens"),
        ("total_tokens", "total tokens"),
        ("llm_calls_total", "LLM calls"),
    ]
    plot_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flat, metrics):
        data = [label_values(rollout_rows, label, key) for label in labels]
        used = [(label, values) for label, values in zip(labels, data) if values]
        if used:
            ax.boxplot([values for _, values in used], tick_labels=[label for label, _ in used], showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_token_usage.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_iteration_distribution(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    plot_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in labels:
        values = label_values(rollout_rows, label, "iterations_used")
        if values:
            bins = range(0, int(max(values)) + 2)
            ax.hist(values, bins=bins, alpha=0.55, label=label)
    ax.set_title("Iteration count distribution")
    ax.set_xlabel("iterations used")
    ax.set_ylabel("rollouts")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_iteration_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_searches_vs_docs(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    plot_setup()
    fig, ax = plt.subplots(figsize=(10, 7))
    for label in labels:
        rows = [row for row in rollout_rows if row.get("label") == label]
        xs = [as_float(row.get("searches_total")) for row in rows]
        ys = [as_float(row.get("docs_read")) for row in rows]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if pairs:
            ax.scatter([x for x, _ in pairs], [y for _, y in pairs], s=12, alpha=0.35, label=label)
    ax.set_title("Searches vs documents read per rollout")
    ax.set_xlabel("searches")
    ax.set_ylabel("docs read")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    path = output_dir / "classic_searches_vs_docs_read.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_stage_event_mix(output_dir: Path, stage_rows: list[dict[str, Any]]) -> str | None:
    if not stage_rows:
        return None
    labels = sorted({str(row.get("label")) for row in stage_rows if row.get("label")})
    stages = ["hop_plan", "search", "doc_choose", "doc_read", "hop_resolve", "report"]
    counts: Counter[tuple[str, str]] = Counter((str(row.get("label") or ""), str(row.get("stage") or "")) for row in stage_rows)
    plot_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = list(range(len(stages)))
    width = 0.8 / max(1, len(labels))
    for idx, label in enumerate(labels):
        values = [counts.get((label, stage), 0) for stage in stages]
        offsets = [x - 0.4 + width / 2 + idx * width for x in xs]
        ax.bar(offsets, values, width=width, label=label)
    ax.set_title("Stage event counts in parsed traces")
    ax.set_ylabel("events")
    ax.set_xticks(xs)
    ax.set_xticklabels(stages, rotation=25, ha="right")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_stage_event_mix.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_pattern_alignment(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    rows = [row for row in rollout_rows if row.get("hop_pattern")]
    if not rows:
        return None
    pattern_counts = Counter(str(row.get("hop_pattern")) for row in rows)
    patterns = [pattern for pattern, _ in pattern_counts.most_common(14)]
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    xs = list(range(len(patterns)))
    width = 0.8 / max(1, len(labels))
    for idx, label in enumerate(labels):
        align_values = []
        resolved_values = []
        for pattern in patterns:
            bucket = [row for row in rows if row.get("label") == label and row.get("hop_pattern") == pattern]
            align_values.append(mean_or_none([value for row in bucket if (value := as_float(row.get("source_alignment_rate"))) is not None]) or 0.0)
            resolved_values.append(
                mean_or_none(
                    [
                        (resolved / total)
                        for row in bucket
                        if (resolved := as_float(row.get("resolved_hops"))) is not None
                        and (total := as_float(row.get("total_hops"))) is not None
                        and total > 0
                    ]
                )
                or 0.0
            )
        offsets = [x - 0.4 + width / 2 + idx * width for x in xs]
        axes[0].bar(offsets, align_values, width=width, label=label)
        axes[1].bar(offsets, resolved_values, width=width, label=label)
    axes[0].set_title("Search calls that used the expected source by hop pattern")
    axes[0].set_ylabel("alignment")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Hops marked answered by the agent fraction by hop pattern")
    axes[1].set_ylabel("hops marked answered / total hops")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(patterns, rotation=35, ha="right")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_pattern_alignment.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_company_alignment(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    rows = [row for row in rollout_rows if row.get("company")]
    if not rows:
        return None
    company_counts = Counter(str(row.get("company")) for row in rows)
    companies = [company for company, _ in company_counts.most_common(12)]
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    plot_setup()
    fig, ax = plt.subplots(figsize=(13, 6))
    xs = list(range(len(companies)))
    width = 0.8 / max(1, len(labels))
    for idx, label in enumerate(labels):
        values = []
        for company in companies:
            bucket = [row for row in rows if row.get("label") == label and row.get("company") == company]
            values.append(mean_or_none([value for row in bucket if (value := as_float(row.get("source_alignment_rate"))) is not None]) or 0.0)
        offsets = [x - 0.4 + width / 2 + idx * width for x in xs]
        ax.bar(offsets, values, width=width, label=label)
    ax.set_title("Search calls that used the expected source by company")
    ax.set_ylabel("alignment")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(companies, rotation=30, ha="right")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_company_alignment.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_rollout_duration_by_iteration(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    rows = [row for row in rollout_rows if as_float(row.get("iterations_used")) is not None and as_float(row.get("duration_s")) is not None]
    if not rows:
        return None
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    plot_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in labels:
        buckets: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("label") != label:
                continue
            iteration = as_int(row.get("iterations_used"))
            duration = as_float(row.get("duration_s"))
            if iteration is not None and duration is not None:
                buckets[iteration].append(duration)
        xs = sorted(buckets)
        ys = [mean_or_none(buckets[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    ax.set_title("Mean rollout duration by final iteration")
    ax.set_xlabel("iterations used")
    ax.set_ylabel("seconds")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    path = output_dir / "classic_rollout_duration_by_iteration.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_planning_yield(output_dir: Path, stage_rows: list[dict[str, Any]]) -> str | None:
    plan_rows = [row for row in stage_rows if row.get("stage") == "hop_plan"]
    if not plan_rows:
        return None
    labels = sorted({str(row.get("label")) for row in plan_rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plotted = False
    for label in labels:
        rows = [row for row in plan_rows if row.get("label") == label]
        iterations = sorted({as_int(row.get("iteration")) for row in rows if as_int(row.get("iteration")) is not None})
        empty_rates: list[float | None] = []
        planned_counts: list[float | None] = []
        for iteration in iterations:
            bucket = [row for row in rows if as_int(row.get("iteration")) == iteration]
            counts = [as_float(row.get("planned_action_count")) for row in bucket if as_float(row.get("planned_action_count")) is not None]
            empty_rates.append(sum(1 for value in counts if value == 0) / len(counts) if counts else None)
            planned_counts.append(mean_or_none([value for value in counts if value is not None]))
        axes[0].plot(iterations, empty_rates, marker="o", linewidth=2.0, label=label)
        axes[1].plot(iterations, planned_counts, marker="o", linewidth=2.0, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title("Empty retrieval-plan rate by iteration")
    axes[0].set_ylabel("fraction empty")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Planned retrieval actions by iteration")
    axes[1].set_ylabel("mean planned actions")
    axes[1].set_xlabel("iteration within rollout")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    path = output_dir / "classic_planning_yield_by_iteration.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_resolved_hops(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    rows = [row for row in rollout_rows if as_float(row.get("resolved_hops")) is not None and as_float(row.get("total_hops")) is not None]
    if not rows:
        return None
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label in labels:
        values = [as_float(row.get("resolved_hops")) for row in rows if row.get("label") == label]
        values = [value for value in values if value is not None]
        fracs = [
            resolved / total
            for row in rows
            if row.get("label") == label
            and (resolved := as_float(row.get("resolved_hops"))) is not None
            and (total := as_float(row.get("total_hops"))) is not None
            and total > 0
        ]
        if values:
            axes[0].hist(values, bins=range(0, int(max(values)) + 3), alpha=0.55, label=label)
        if fracs:
            axes[1].hist(fracs, bins=10, range=(0, 1), alpha=0.55, label=label)
    axes[0].set_title("Hops marked answered by the agent per rollout")
    axes[0].set_xlabel("hops marked answered by the agent")
    axes[0].set_ylabel("rollouts")
    axes[1].set_title("Hops marked answered by the agent fraction")
    axes[1].set_xlabel("hops marked answered / total hops")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_resolved_hops_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_temporal_rollout_metrics(output_dir: Path, rollout_rows: list[dict[str, Any]], window: int) -> str | None:
    rows = [row for row in rollout_rows if as_float(row.get("completed_at_s")) is not None]
    if not rows:
        return None
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for label in labels:
        label_rows = sorted([row for row in rows if row.get("label") == label], key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
        xs = [datetime.fromtimestamp(as_float(row.get("completed_at_s")) or 0.0, tz=timezone.utc) for row in label_rows]
        alignment = rolling([as_float(row.get("source_alignment_rate")) for row in label_rows], window)
        searches = rolling([as_float(row.get("searches_total")) for row in label_rows], window)
        llm_calls = rolling([as_float(row.get("llm_calls_total")) for row in label_rows], window)
        axes[0].plot(xs, alignment, linewidth=2.0, label=label)
        axes[1].plot(xs, searches, linewidth=2.0, label=label)
        axes[2].plot(xs, llm_calls, linewidth=2.0, label=label)
    axes[0].set_title(f"Search calls that used the expected source moving average ({window} rollouts)")
    axes[0].set_ylabel("alignment")
    axes[0].set_ylim(0, 1)
    axes[1].set_title(f"Searches per rollout moving average ({window} rollouts)")
    axes[1].set_ylabel("searches")
    axes[2].set_title(f"LLM calls per rollout moving average ({window} rollouts)")
    axes[2].set_ylabel("LLM calls")
    for ax in axes:
        ax.legend(loc="best")
        finish_time_axis(ax)
    path = output_dir / "classic_temporal_rollout_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_early_middle_late(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    rows = [row for row in rollout_rows if as_float(row.get("completed_at_s")) is not None]
    if not rows:
        return None
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    metrics = [
        ("source_alignment_rate", "search calls that used the expected source"),
        ("searches_total", "searches"),
        ("docs_read", "docs read"),
        ("llm_calls_total", "LLM calls"),
    ]
    plot_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flat, metrics):
        xlabels: list[str] = []
        means: list[float] = []
        for label in labels:
            label_rows = sorted([row for row in rows if row.get("label") == label], key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
            n = len(label_rows)
            buckets = [
                ("early", label_rows[: max(1, n // 3)]),
                ("middle", label_rows[max(1, n // 3) : max(2, (2 * n) // 3)]),
                ("late", label_rows[max(2, (2 * n) // 3) :]),
            ]
            for phase, bucket in buckets:
                xlabels.append(f"{label}\n{phase}")
                means.append(mean_or_none([value for row in bucket if (value := as_float(row.get(key))) is not None]) or 0.0)
        ax.bar(range(len(means)), means)
        ax.set_title(title)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_early_middle_late_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_rollout_duration_hist(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    plot_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in labels:
        values = label_values(rollout_rows, label, "duration_s")
        if values:
            ax.hist(values, bins=40, alpha=0.48, label=label)
    ax.set_title("Rollout duration distribution")
    ax.set_xlabel("seconds")
    ax.set_ylabel("rollouts")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_rollout_duration_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_llm_calls_hist(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    plot_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in labels:
        values = label_values(rollout_rows, label, "llm_calls_total")
        if values:
            ax.hist(values, bins=range(0, int(max(values)) + 3), alpha=0.5, label=label)
    ax.set_title("LLM calls per rollout distribution")
    ax.set_xlabel("LLM calls")
    ax.set_ylabel("rollouts")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_llm_calls_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_total_tokens_hist(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    plot_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in labels:
        values = label_values(rollout_rows, label, "total_tokens")
        if values:
            ax.hist(values, bins=40, alpha=0.48, label=label)
    ax.set_title("Total LLM tokens per rollout")
    ax.set_xlabel("prompt + output tokens")
    ax.set_ylabel("rollouts")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_llm_total_tokens_per_rollout.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_total_tokens_by_final_iteration(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    rows = [
        row
        for row in rollout_rows
        if as_int(row.get("iterations_used")) is not None and as_float(row.get("total_tokens")) is not None
    ]
    if not rows:
        return None
    labels = sorted({str(row.get("label")) for row in rows if row.get("label")})
    plot_setup()
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, max(4, 3.5 * len(labels))), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        label_rows = [row for row in rows if row.get("label") == label]
        iterations = sorted({as_int(row.get("iterations_used")) for row in label_rows if as_int(row.get("iterations_used")) is not None})
        data = []
        used = []
        for iteration in iterations:
            values = [
                as_float(row.get("total_tokens"))
                for row in label_rows
                if as_int(row.get("iterations_used")) == iteration and as_float(row.get("total_tokens")) is not None
            ]
            values = [value for value in values if value is not None]
            if values:
                data.append(values)
                used.append(str(iteration))
        if data:
            ax.boxplot(data, tick_labels=used, showfliers=False)
        ax.set_title(f"{label}: total tokens by final iteration")
        ax.set_xlabel("iterations used")
        ax.set_ylabel("tokens")
        ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "classic_total_tokens_by_final_iteration.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_classic_source_usage_scatter(output_dir: Path, rollout_rows: list[dict[str, Any]]) -> str | None:
    if not rollout_rows:
        return None
    labels = sorted({str(row.get("label")) for row in rollout_rows if row.get("label")})
    pairs = [
        ("search_expected_web_actual_web_count", "search_web_count", "expected web hop-searches", "actual web searches"),
        ("search_expected_local_actual_local_count", "search_local_count", "expected local hop-searches", "actual local searches"),
        ("search_expected_web_actual_local_count", "search_local_count", "web hops with local searches", "actual local searches"),
        ("search_expected_local_actual_web_count", "search_web_count", "local hops with web searches", "actual web searches"),
    ]
    plot_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes.flat, pairs):
        for label in labels:
            rows = [row for row in rollout_rows if row.get("label") == label]
            xy = [
                (x, y)
                for row in rows
                if (x := as_float(row.get(xkey))) is not None and (y := as_float(row.get(ykey))) is not None
            ]
            if xy:
                ax.scatter([x for x, _ in xy], [y for _, y in xy], s=12, alpha=0.35, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(loc="best")
    fig.suptitle("Source usage scatter (subset buckets vs rollout totals)")
    path = output_dir / "classic_source_usage_scatter.png"
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


def embedded_image(output_dir: Path, name: str) -> str:
    path = output_dir / name
    if not path.exists():
        return "<div class='missing-note'>plot not available yet</div>"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}' alt='{esc(name)}'>"


def stat_card_html(label: str, value: Any) -> str:
    return f"<div class='stat-card'><div class='stat-label'>{esc(label)}</div><div class='stat-value'>{esc(value)}</div></div>"


def simple_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int = 12) -> str:
    if not rows:
        return "<p class='muted'>No rows yet.</p>"
    head = "".join(f"<th>{esc(title)}</th>" for key, title in columns)
    body_rows = []
    for row in rows[:limit]:
        body_rows.append("<tr>" + "".join(f"<td>{esc(row.get(key, ''))}</td>" for key, _ in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def aggregate_run_summary(
    runs: list[RunSpec],
    rollout_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run in runs:
        label = run.label
        rollouts = [row for row in rollout_rows if row.get("label") == label]
        searches = [row for row in stage_rows if row.get("label") == label and row.get("stage") == "search"]
        local_searches = [row for row in searches if row.get("search_backend") == "local"]
        web_searches = [row for row in searches if row.get("search_backend") == "web"]
        doc_reads = [row for row in stage_rows if row.get("label") == label and row.get("stage") == "doc_read"]
        local_doc_reads = [row for row in doc_reads if row.get("doc_source") == "local"]
        web_doc_reads = [row for row in doc_reads if row.get("doc_source") == "web"]
        hop_plans = [row for row in stage_rows if row.get("label") == label and row.get("stage") == "hop_plan"]
        empty_hop_plans = [row for row in hop_plans if as_float(row.get("planned_action_count")) == 0]
        known = [row for row in searches if str(row.get("expected_source") or "unknown") != "unknown"]
        matched = [row for row in known if str(row.get("source_matched_expected")).lower() == "true"]
        helpers = [row for row in action_rows if row.get("label") == label]
        errors = sum(as_int(row.get("count")) or 0 for row in error_rows if row.get("label") == label)
        summaries.append(
            {
                "run": label,
                "rollouts parsed": len(rollouts),
                "mean duration": fmt_num(mean_or_none(label_values(rollouts, label, "duration_s")), 1),
                "mean searches": fmt_num(mean_or_none(label_values(rollouts, label, "searches_total")), 2),
                "mean docs read": fmt_num(mean_or_none(label_values(rollouts, label, "docs_read")), 2),
                "search calls that used the expected source": fmt_pct(len(matched) / len(known) if known else None),
                "actual local searches": fmt_num(len(local_searches)),
                "actual web searches": fmt_num(len(web_searches)),
                "local docs read": fmt_num(len(local_doc_reads)),
                "web docs read": fmt_num(len(web_doc_reads)),
                "empty hop-plan outputs": fmt_num(len(empty_hop_plans)),
                "empty hop-plan rate": fmt_pct(len(empty_hop_plans) / len(hop_plans) if hop_plans else None),
                "mean helper call": fmt_num(mean_or_none([as_float(row.get("execution_time_s")) for row in helpers if as_float(row.get("execution_time_s")) is not None]), 3),
                "errors": errors,
            }
        )
    return summaries


def top_error_rows(error_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in error_rows:
        counter[(str(row.get("label") or ""), str(row.get("error_group") or ""), str(row.get("error_key") or ""))] += as_int(row.get("count")) or 0
    return [
        {"run": label, "group": group, "error": key, "count": count}
        for (label, group, key), count in counter.most_common(20)
    ]


def build_classic_html(
    output_dir: Path,
    runs: list[RunSpec],
    rollout_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    plot_names: list[str],
    now: datetime,
) -> None:
    summaries = aggregate_run_summary(runs, rollout_rows, stage_rows, action_rows, error_rows)
    all_known = [row for row in stage_rows if row.get("stage") == "search" and str(row.get("expected_source") or "unknown") != "unknown"]
    all_matched = [row for row in all_known if str(row.get("source_matched_expected")).lower() == "true"]
    helper_latencies = [as_float(row.get("execution_time_s")) for row in action_rows if as_float(row.get("execution_time_s")) is not None]
    plot_cards = {
        "classic_source_alignment_by_hop.png": (
            "Search calls that used the expected source by hop",
            "Does the agent use local search on local hops and web search on web hops?",
        ),
        "classic_source_calls_by_iteration.png": (
            "Source calls by rollout iteration",
            "Within a rollout, this tracks local-search share and search volume as iterations advance.",
        ),
        "classic_search_doc_counts.png": (
            "Search/doc counts",
            "Mean retrieval and document-read counts per rollout, split by run and source where available.",
        ),
        "classic_rollout_distributions.png": (
            "Rollout distributions",
            "Runtime, LLM-call, search, and document-read distributions for the sampled trace cache.",
        ),
        "classic_rollout_duration_distribution.png": (
            "Rollout duration distribution",
            "Old-viewer style rollout wall-time histogram.",
        ),
        "classic_llm_calls_distribution.png": (
            "LLM calls per rollout",
            "Old-viewer style distribution of total LLM calls per rollout.",
        ),
        "classic_llm_total_tokens_per_rollout.png": (
            "Total LLM tokens per rollout",
            "Prompt plus output tokens per rollout.",
        ),
        "classic_total_tokens_by_final_iteration.png": (
            "Total tokens by final iteration",
            "Token volume split by how many iterations the agent used.",
        ),
        "classic_source_usage_scatter.png": (
            "Source usage scatter",
            "Subset source buckets versus total actual local/web search volume; use the confusion plot for exact source-mismatch counts.",
        ),
        "classic_search_source_confusion.png": (
            "Expected source vs actual search backend",
            "Explicit confusion counts and fractions for local/web expected hops versus local/web executed searches.",
        ),
        "classic_doc_read_source_totals.png": (
            "Document reads by source",
            "Total local/web doc_read events and shares; this checks whether local documents were actually read.",
        ),
        "classic_stage_duration_boxplot.png": (
            "Stage duration boxplots",
            "Old runtime-viewer style distribution of hop planning, search, doc choice, doc read, resolver, and report timing.",
        ),
        "classic_source_usage_distributions.png": (
            "Source usage distributions",
            "Web/local search and document-read distributions across sampled rollouts.",
        ),
        "classic_token_usage.png": (
            "Token and LLM-call usage",
            "Prompt, output, total token, and LLM-call distributions; useful for comparing why 4B is much cheaper/faster.",
        ),
        "classic_iteration_distribution.png": (
            "Iteration distribution",
            "How many loop iterations rollouts use before completion/stalling.",
        ),
        "classic_searches_vs_docs_read.png": (
            "Searches vs docs read",
            "Scatter plot showing retrieval volume versus document-reading volume per rollout.",
        ),
        "classic_stage_event_mix.png": (
            "Stage event mix",
            "Counts of emitted timeline events by stage for each run.",
        ),
        "classic_pattern_alignment.png": (
            "Pattern-level behavior",
            "Search calls that used the expected source and hops marked answered by the agent fraction by hop pattern.",
        ),
        "classic_company_alignment.png": (
            "Company-level search calls that used the expected source",
            "Whether retrieval source behavior differs by company/domain.",
        ),
        "classic_rollout_duration_by_iteration.png": (
            "Rollout duration by final iteration",
            "Mean rollout wall time as a function of how long the agent kept looping.",
        ),
        "classic_planning_yield_by_iteration.png": (
            "Planning yield by iteration",
            "Empty-plan rate and mean planned retrieval actions; this makes the 4B repeated-planning failure visible.",
        ),
        "classic_resolved_hops_distribution.png": (
            "Hops marked answered by the agent distribution",
            "Resolved means the agent marked a hop as answered/resolved, not necessarily final correctness.",
        ),
        "classic_temporal_rollout_metrics.png": (
            "Temporal moving averages",
            "Search calls that used the expected source, retrieval volume, and LLM-call volume over rollout completion time.",
        ),
        "classic_early_middle_late_metrics.png": (
            "Early / middle / late metrics",
            "Same run split into training-time thirds to catch drift across the run.",
        ),
        "classic_error_breakdown.png": (
            "Error breakdown",
            "Trace summary errors by stage/kind. If empty, there were no emitted error records in the cache.",
        ),
        "classic_parse_and_empty_plan_breakdown.png": (
            "Hop parse errors and empty planner outputs",
            "Parse errors by stage plus accepted empty hop-plan outputs. Empty [] is valid JSON and is not counted as a parser error by the run.",
        ),
        "trace_stage_duration_ma.png": (
            "Stage duration moving averages",
            "Old-viewer style stage timing, but plotted over recent rollout trace events.",
        ),
        "trace_helper_action_latency_ma.png": (
            "Helper action latency",
            "Per-helper-call execution times from trace.json actions, split by local/web action type.",
        ),
        "trace_search_latency_ma.png": (
            "Search-stage latency",
            "Search-stage event durations from timeline_events.json, split by local/web.",
        ),
    }
    plot_html = "\n".join(
        f"<div class='plot-card'><h3>{esc(plot_cards[name][0])}</h3><p class='blurb'>{esc(plot_cards[name][1])}</p>{embedded_image(output_dir, name)}</div>"
        for name in plot_names
        if name in plot_cards
    )
    stats = [
        ("trace rollouts", fmt_num(len(rollout_rows))),
        ("stage events", fmt_num(len(stage_rows))),
        ("helper actions", fmt_num(len(action_rows))),
        ("search calls that used the expected source", fmt_pct(len(all_matched) / len(all_known) if all_known else None)),
        ("mean helper call s", fmt_num(mean_or_none([value for value in helper_latencies if value is not None]), 3)),
        ("trace errors", fmt_num(sum(as_int(row.get("count")) or 0 for row in error_rows))),
    ]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>privacy_hopqa classic live runtime viewer</title>
  <style>
    :root {{
      --bg: #f5f4ef;
      --panel: #ffffff;
      --ink: #1e2430;
      --muted: #5f6b7a;
      --line: #d7dbe2;
      --accent: #1f5da8;
      --accent-soft: #e7f0fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f8f7f2 0%, #eef3f8 100%);
      color: var(--ink);
    }}
    a {{ color: var(--accent); }}
    .page {{ max-width: 1600px; margin: 0 auto; padding: 28px; }}
    .hero {{
      background: radial-gradient(circle at top left, #ffffff 0%, #eef4fb 45%, #f8f7f2 100%);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px 30px;
      margin-bottom: 22px;
      box-shadow: 0 18px 50px rgba(30, 36, 48, 0.06);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 6px 0; color: var(--muted); max-width: 1100px; }}
    .topnav {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
    .topnav a {{
      text-decoration: none; background: var(--accent-soft); color: var(--accent);
      padding: 8px 12px; border-radius: 999px; border: 1px solid #cfe0f6; font-size: 14px;
    }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 22px 0; }}
    .stat-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(30,36,48,.04); }}
    .stat-label {{ font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
    .stat-value {{ font-size: 28px; font-weight: 700; }}
    .section {{ margin: 28px 0; }}
    .section h2 {{ margin: 0 0 12px; font-size: 22px; }}
    .section-note, .muted {{ color: var(--muted); }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .plot-card, .table-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 14px; box-shadow: 0 12px 32px rgba(30,36,48,.05); overflow: auto; }}
    .plot-card h3, .table-card h3 {{ margin: 4px 0 8px; font-size: 18px; }}
    .plot-card .blurb {{ margin: 0 0 12px; color: var(--muted); font-size: 14px; min-height: 40px; }}
    .plot-card img {{ width: 100%; display: block; border-radius: 12px; border: 1px solid var(--line); background: #fff; }}
    .missing-note {{ min-height: 260px; display: grid; place-items: center; border: 1px dashed var(--line); border-radius: 12px; color: var(--muted); background: #f8fafc; text-align: center; padding: 16px; }}
    .table-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f7f9fc; z-index: 1; }}
    code {{ background: #f2f5fa; border: 1px solid var(--line); border-radius: 6px; padding: 1px 5px; font-size: 12px; }}
    ul.raw-links {{ margin: 0; padding-left: 18px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>privacy_hopqa live runtime viewer</h1>
      <p>Generated {esc(iso(now))}. This is the classic-style view over the current 4B and 8B training runs. It uses the same durable rollout trace artifacts as the older runtime viewers: <code>timeline_events.json</code> for stage timing and <code>trace.json/actions[*].execution_time</code> for helper-backed search-call latency.</p>
      <p>The source-alignment plots compare actual local/web search calls against the expected hop source from the final dataset pattern and hop document ids.</p>
      <div class="topnav">
        <a href="#summary">Summary</a>
        <a href="#source">Search calls that used the expected source</a>
        <a href="#runtime">Runtime</a>
        <a href="#errors">Errors</a>
        <a href="#raw">Raw exports</a>
        <a href="privacy_hopqa_schedulerfix_rollout_samples.html">Rollout samples</a>
        <a href="privacy_hopqa_schedulerfix_search_calls.html">Search calls</a>
        {''.join(f'<a href="privacy_hopqa_schedulerfix_{safe_slug(run.label)}_detail.html">{esc(run.label)} detail</a>' for run in runs)}
      </div>
    </section>
    <section id="summary" class="section">
      <h2>Summary</h2>
      <div class="stat-grid">{''.join(stat_card_html(label, value) for label, value in stats)}</div>
      <div class="table-card">
        <h3>Run comparison</h3>
        {simple_table(summaries, [("run", "run"), ("rollouts parsed", "rollouts"), ("mean duration", "mean duration s"), ("mean searches", "mean searches"), ("mean docs read", "mean docs read"), ("search calls that used the expected source", "expected-source search rate"), ("actual local searches", "local searches"), ("actual web searches", "web searches"), ("local docs read", "local docs read"), ("web docs read", "web docs read"), ("empty hop-plan outputs", "empty plans"), ("empty hop-plan rate", "empty plan rate"), ("mean helper call", "mean helper call s"), ("errors", "errors")])}
      </div>
    </section>
    <section id="source" class="section">
      <h2>Runtime, retrieval, source behavior, and error plots</h2>
      <p class="section-note">This restores the broader old-viewer style plot set, with extra source-alignment plots over hop position and rollout iteration.</p>
      <div class="plot-grid">{plot_html}</div>
    </section>
    <section id="errors" class="section">
      <h2>Error details</h2>
      <div class="table-grid">
        <div class="table-card">
          <h3>Top emitted trace errors</h3>
          {simple_table(top_error_rows(error_rows), [("run", "run"), ("group", "group"), ("error", "error"), ("count", "count")], limit=20)}
        </div>
      </div>
    </section>
    <section id="raw" class="section">
      <h2>Raw exports</h2>
      <ul class="raw-links">
        <li><code>trace_stage_timings.csv</code>: every parsed timeline stage event.</li>
        <li><code>trace_rollout_timings.csv</code>: rollout-level timing, search/doc counts, and source-alignment aggregates.</li>
        <li><code>trace_helper_actions.csv</code>: helper-backed web/local search action execution times, queries, and top result summaries.</li>
        <li><code>trace_error_summary.csv</code>: error summaries and emitted trace error records.</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""
    classic_name = "privacy_hopqa_schedulerfix_classic_viewer.html"
    (output_dir / classic_name).write_text(html_text, encoding="utf-8")
    (output_dir.parent / classic_name).write_text(html_text, encoding="utf-8")


def truncate_text(value: Any, limit: int = 900) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def read_json_safe(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return default


def read_text_safe(path: Path, limit: int = 4500) -> str:
    try:
        return truncate_text(path.read_text(encoding="utf-8", errors="ignore"), limit)
    except OSError:
        return ""


def choose_sample_rollouts(
    runs: list[RunSpec], rollout_rows: list[dict[str, Any]], samples_per_phase: int
) -> list[dict[str, Any]]:
    rng = random.Random(20260506)
    selected: list[dict[str, Any]] = []
    for run in runs:
        rows = [
            row
            for row in rollout_rows
            if row.get("label") == run.label and as_float(row.get("completed_at_s")) is not None
        ]
        rows = sorted(rows, key=lambda row: as_float(row.get("completed_at_s")) or 0.0)
        if not rows:
            continue
        n = len(rows)
        phase_specs = [
            ("early", rows[: max(1, n // 3)]),
            ("middle", rows[max(1, n // 3) : max(2, (2 * n) // 3)]),
            ("late", rows[max(2, (2 * n) // 3) :]),
        ]
        for phase, bucket in phase_specs:
            if not bucket:
                continue
            count = min(samples_per_phase, len(bucket))
            for row in sorted(rng.sample(bucket, count), key=lambda item: as_float(item.get("completed_at_s")) or 0.0):
                copied = dict(row)
                copied["phase"] = phase
                selected.append(copied)
    return selected


def phase_rank(sample: dict[str, Any]) -> str:
    return f"{sample.get('label')} / {sample.get('phase')} / {sample.get('rollout_id')}"


def render_timeline_events(events: list[dict[str, Any]], max_events: int = 120) -> str:
    usable = [event for event in events if isinstance(event, dict) and as_float(event.get("start_s")) is not None]
    usable = sorted(usable, key=lambda event: as_float(event.get("start_s")) or 0.0)[:max_events]
    if not usable:
        return "<p class='muted'>No timeline events stored.</p>"
    total = max(as_float(event.get("end_s")) or as_float(event.get("start_s")) or 0.0 for event in usable) or 1.0
    colors = {
        "hop_plan": "#1f5da8",
        "search": "#14866d",
        "doc_choose": "#9467bd",
        "doc_read": "#e17c05",
        "hop_resolve": "#d62728",
        "report": "#0f766e",
    }
    rows = []
    for event in usable:
        start = as_float(event.get("start_s")) or 0.0
        end = as_float(event.get("end_s")) or start
        duration = max(0.0, end - start)
        stage = str(event.get("stage") or "unknown")
        meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}
        left = min(100.0, max(0.0, 100.0 * start / total))
        width = max(0.35, min(100.0 - left, 100.0 * duration / total))
        color = colors.get(stage, "#6b7280")
        label = truncate_text(event.get("label"), 80)
        rows.append(
            "<tr>"
            f"<td><code>{esc(stage)}</code></td>"
            f"<td>{esc(meta.get('hop_number'))}</td>"
            f"<td>{esc(meta.get('iteration'))}</td>"
            f"<td>{fmt_num(duration, 2)}s</td>"
            f"<td class='timeline-cell'><div class='bar-track'><span class='bar' style='left:{left:.3f}%;width:{width:.3f}%;background:{color}' title='{esc(label)}'></span></div></td>"
            f"<td>{esc(label)}</td>"
            "</tr>"
        )
    return (
        "<table class='timeline-table'><thead><tr><th>stage</th><th>hop</th><th>iter</th><th>duration</th><th>timeline</th><th>label</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_actions(actions: list[dict[str, Any]], max_actions: int = 40) -> str:
    if not actions:
        return "<p class='muted'>No actions stored.</p>"
    rows = []
    for action in actions[:max_actions]:
        if not isinstance(action, dict):
            continue
        actual = action.get("actual_output") if isinstance(action.get("actual_output"), dict) else {}
        results = actual.get("results") if isinstance(actual.get("results"), list) else []
        top_results = []
        for result in results[:3]:
            if isinstance(result, dict):
                top_results.append(
                    f"{result.get('doc_id') or ''} {truncate_text(result.get('title') or result.get('snippet'), 120)}"
                )
        query = actual.get("query") or ((action.get("parameters") or {}).get("query") if isinstance(action.get("parameters"), dict) else "")
        rows.append(
            "<tr>"
            f"<td><code>{esc(action.get('type'))}</code></td>"
            f"<td>{esc(action.get('iteration'))}</td>"
            f"<td>{fmt_num(action.get('execution_time'), 3)}s</td>"
            f"<td>{esc(action.get('status'))}</td>"
            f"<td>{esc(truncate_text(query, 180))}</td>"
            f"<td>{esc('; '.join(top_results))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>type</th><th>iter</th><th>time</th><th>status</th><th>query</th><th>top results</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_hop_states(hop_states: list[dict[str, Any]]) -> str:
    if not hop_states:
        return "<p class='muted'>No hop states stored.</p>"
    blocks = []
    for hop in hop_states:
        if not isinstance(hop, dict):
            continue
        readers = hop.get("reader_results") if isinstance(hop.get("reader_results"), list) else []
        reader_rows = []
        for reader in readers[:5]:
            if isinstance(reader, dict):
                reader_rows.append(
                    "<tr>"
                    f"<td>{esc(reader.get('doc_id'))}</td>"
                    f"<td>{esc(reader.get('source'))}</td>"
                    f"<td>{esc(reader.get('can_answer'))}</td>"
                    f"<td>{esc(truncate_text(reader.get('proposed_answer'), 120))}</td>"
                    f"<td>{esc(truncate_text(reader.get('justification'), 240))}</td>"
                    "</tr>"
                )
        reader_table = (
            "<table><thead><tr><th>doc</th><th>source</th><th>can answer</th><th>proposed answer</th><th>reader justification</th></tr></thead>"
            f"<tbody>{''.join(reader_rows)}</tbody></table>"
            if reader_rows
            else "<p class='muted'>No reader results for this hop.</p>"
        )
        blocks.append(
            "<details open>"
            f"<summary>Hop {esc(hop.get('hop_number'))}: {esc(truncate_text(hop.get('question'), 170))}</summary>"
            "<div class='hop-grid'>"
            f"<div><b>Status</b><br>{esc(hop.get('status'))}</div>"
            f"<div><b>Answer</b><br>{esc(truncate_text(hop.get('answer'), 180))}</div>"
            f"<div><b>Confidence</b><br>{esc(hop.get('confidence'))}</div>"
            "</div>"
            f"<p><b>Resolution reason</b><br>{esc(truncate_text(hop.get('resolution_reason'), 700))}</p>"
            f"<p><b>Justification</b><br>{esc(truncate_text(hop.get('justification'), 700))}</p>"
            f"{reader_table}"
            "</details>"
        )
    return "".join(blocks)


def build_rollout_samples_html(
    output_dir: Path,
    runs: list[RunSpec],
    rollout_rows: list[dict[str, Any]],
    samples_per_phase: int,
    now: datetime,
) -> None:
    samples = choose_sample_rollouts(runs, rollout_rows, samples_per_phase)
    nav_items = []
    sample_sections = []
    for idx, sample in enumerate(samples, start=1):
        rollout_id = str(sample.get("rollout_id") or "")
        run_dir = Path(str(sample.get("run_dir") or ""))
        sample_dir = run_dir / "privacy_hopqa_runs" / rollout_id
        trace = read_json_safe(sample_dir / "trace.json", {})
        events = read_json_safe(sample_dir / "timeline_events.json", [])
        summary = trace.get("summary") if isinstance(trace, dict) else {}
        actions = trace.get("actions") if isinstance(trace, dict) else []
        hop_states = trace.get("hop_states") if isinstance(trace, dict) else []
        report_path = Path(str(trace.get("report_path") or "")) if isinstance(trace, dict) else Path("")
        report = read_text_safe(report_path, 4500) if str(report_path) else ""
        if not report:
            report = read_text_safe(sample_dir / "workspace" / "research_report.md", 4500)
        nav_label = f"{idx}. {sample.get('label')} {sample.get('phase')} {rollout_id[:18]}"
        nav_items.append(f"<a href='#sample-{idx}'>{esc(nav_label)}</a>")
        sample_sections.append(
            f"""
            <section class='sample' id='sample-{idx}'>
              <h2>{esc(phase_rank(sample))}</h2>
              <div class='sample-meta-grid'>
                <div><b>task</b><br>{esc(sample.get('task_id'))}</div>
                <div><b>chain</b><br>{esc(sample.get('chain_id'))}</div>
                <div><b>company</b><br>{esc(sample.get('company'))}</div>
                <div><b>pattern</b><br>{esc(sample.get('hop_pattern'))}</div>
                <div><b>duration</b><br>{fmt_num(summary.get('duration_s') if isinstance(summary, dict) else sample.get('duration_s'), 1)}s</div>
                <div><b>hops marked answered by agent</b><br>{esc(summary.get('resolved_hops') if isinstance(summary, dict) else sample.get('resolved_hops'))}/{esc(summary.get('total_hops') if isinstance(summary, dict) else sample.get('total_hops'))}</div>
                <div><b>LLM calls</b><br>{fmt_num(summary.get('llm_calls_total') if isinstance(summary, dict) else sample.get('llm_calls_total'))}</div>
                <div><b>search/doc</b><br>{fmt_num(summary.get('searches_total') if isinstance(summary, dict) else sample.get('searches_total'))} / {fmt_num(summary.get('docs_read') if isinstance(summary, dict) else sample.get('docs_read'))}</div>
              </div>
              <p class='muted'>Raw per-call LLM completions are not stored in these traces. This page shows stored LLM artifacts: hop states, reader decisions/justifications, resolver reasons, planned/search actions, retrieved result snippets, and final report excerpts.</p>
              <h3>Horizontal timing timeline</h3>
              {render_timeline_events(events if isinstance(events, list) else [])}
              <h3>Stored final report excerpt</h3>
              <pre>{esc(report or 'No report excerpt found.')}</pre>
              <h3>Hop states and reader outputs</h3>
              {render_hop_states(hop_states if isinstance(hop_states, list) else [])}
              <h3>Search/action outputs</h3>
              {render_actions(actions if isinstance(actions, list) else [])}
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>privacy_hopqa rollout samples</title>
  <style>
    :root {{ --bg:#f5f4ef; --panel:#fff; --ink:#1e2430; --muted:#5f6b7a; --line:#d7dbe2; --accent:#1f5da8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:linear-gradient(180deg,#f8f7f2,#eef3f8); color:var(--ink); }}
    .layout {{ display:grid; grid-template-columns:310px 1fr; min-height:100vh; }}
    aside {{ position:sticky; top:0; height:100vh; overflow:auto; padding:22px; border-right:1px solid var(--line); background:rgba(255,255,255,.82); }}
    aside h1 {{ font-size:20px; margin:0 0 8px; }}
    aside p {{ color:var(--muted); font-size:13px; line-height:1.45; }}
    aside nav {{ display:grid; gap:7px; margin-top:16px; }}
    aside a {{ color:var(--accent); text-decoration:none; padding:7px 9px; border:1px solid #cfe0f6; border-radius:10px; background:#e7f0fb; font-size:12px; }}
    main {{ padding:24px; max-width:1500px; }}
    .sample {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:18px; margin-bottom:22px; box-shadow:0 12px 32px rgba(30,36,48,.05); }}
    h2 {{ margin:0 0 12px; }}
    h3 {{ margin:22px 0 10px; }}
    .muted {{ color:var(--muted); }}
    .sample-meta-grid, .hop-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:12px 0; }}
    .sample-meta-grid div, .hop-grid div {{ border:1px solid var(--line); border-radius:12px; padding:10px; background:#f8fafc; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th, td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#f7f9fc; position:sticky; top:0; }}
    code {{ background:#f2f5fa; border:1px solid var(--line); border-radius:6px; padding:1px 5px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#f7f9fc; border:1px solid var(--line); border-radius:12px; padding:12px; max-height:360px; overflow:auto; font-size:12px; }}
    details {{ border:1px solid var(--line); border-radius:14px; padding:10px 12px; margin:10px 0; background:#fff; }}
    summary {{ cursor:pointer; font-weight:700; }}
    .timeline-cell {{ min-width:360px; }}
    .bar-track {{ position:relative; height:18px; background:#eef2f7; border-radius:999px; border:1px solid var(--line); overflow:hidden; }}
    .bar {{ position:absolute; top:1px; bottom:1px; border-radius:999px; min-width:2px; }}
    @media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ position:relative; height:auto; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Rollout samples</h1>
      <p>Generated {esc(iso(now))}. Up to {samples_per_phase} random rollouts per run from early, middle, and late portions of the parsed trace cache.</p>
      <p>Boundary formula after sorting rollouts within each run by completion time: early = indices &lt; n/3; middle = n/3 through 2n/3; late = indices &gt;= 2n/3.</p>
      <p><a href="privacy_hopqa_schedulerfix_classic_viewer.html">Back to classic viewer</a></p>
      <nav>{''.join(nav_items)}</nav>
    </aside>
    <main>{''.join(sample_sections) if sample_sections else "<p>No samples available yet.</p>"}</main>
  </div>
</body>
</html>
"""
    name = "privacy_hopqa_schedulerfix_rollout_samples.html"
    (output_dir / name).write_text(html_text, encoding="utf-8")
    (output_dir.parent / name).write_text(html_text, encoding="utf-8")


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def build_model_detail_html(
    output_dir: Path,
    run: RunSpec,
    model_dir: Path,
    plot_names: list[str],
    rollout_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    now: datetime,
) -> str:
    label = run.label
    rollouts = [row for row in rollout_rows if row.get("label") == label]
    stage_count = len([row for row in stage_rows if row.get("label") == label])
    action_count = len([row for row in action_rows if row.get("label") == label])
    plan_rows = [row for row in stage_rows if row.get("label") == label and row.get("stage") == "hop_plan"]
    empty_plan_rate = (
        sum(1 for row in plan_rows if as_float(row.get("planned_action_count")) == 0)
        / len([row for row in plan_rows if as_float(row.get("planned_action_count")) is not None])
        if any(as_float(row.get("planned_action_count")) is not None for row in plan_rows)
        else None
    )
    plot_cards = [
        ("classic_rollout_duration_distribution.png", "Rollout duration distribution"),
        ("classic_llm_calls_distribution.png", "LLM calls per rollout"),
        ("classic_llm_total_tokens_per_rollout.png", "Total LLM tokens per rollout"),
        ("classic_total_tokens_by_final_iteration.png", "Total tokens by final iteration"),
        ("classic_source_usage_scatter.png", "Source usage scatter"),
        ("classic_rollout_distributions.png", "Rollout distributions"),
        ("classic_rollout_duration_by_iteration.png", "Duration by final iteration"),
        ("classic_iteration_distribution.png", "Iteration distribution"),
        ("classic_resolved_hops_distribution.png", "Hops marked answered by the agent distribution"),
        ("classic_search_doc_counts.png", "Search/doc counts"),
        ("classic_searches_vs_docs_read.png", "Searches vs docs read"),
        ("classic_source_usage_distributions.png", "Source usage distributions"),
        ("classic_stage_duration_boxplot.png", "Stage duration boxplots"),
        ("classic_stage_event_mix.png", "Stage event mix"),
        ("classic_token_usage.png", "Token and LLM-call usage"),
        ("classic_planning_yield_by_iteration.png", "Planning yield"),
        ("classic_source_alignment_by_hop.png", "Search calls that used the expected source by hop"),
        ("classic_source_calls_by_iteration.png", "Source calls by iteration"),
        ("classic_pattern_alignment.png", "Pattern-level behavior"),
        ("classic_company_alignment.png", "Company-level behavior"),
        ("classic_temporal_rollout_metrics.png", "Temporal moving averages"),
        ("classic_early_middle_late_metrics.png", "Early/middle/late split"),
        ("classic_error_breakdown.png", "Error breakdown"),
        ("trace_helper_action_latency_ma.png", "Helper action latency"),
        ("trace_search_latency_ma.png", "Search-stage latency"),
        ("trace_stage_duration_ma.png", "Stage duration moving average"),
    ]
    cards = "\n".join(
        f"<div class='plot-card'><h3>{esc(title)}</h3>{embedded_image(model_dir, name)}</div>"
        for name, title in plot_cards
        if name in plot_names
    )
    stats = [
        ("rollouts", fmt_num(len(rollouts))),
        ("stage events", fmt_num(stage_count)),
        ("search actions", fmt_num(action_count)),
        ("mean duration s", fmt_num(mean_or_none(label_values(rollouts, label, "duration_s")), 1)),
        ("mean LLM calls", fmt_num(mean_or_none(label_values(rollouts, label, "llm_calls_total")), 2)),
        ("mean searches", fmt_num(mean_or_none(label_values(rollouts, label, "searches_total")), 2)),
        ("empty plan rate", fmt_pct(empty_plan_rate)),
        ("search calls that used the expected source", fmt_pct(mean_or_none([value for row in rollouts if (value := as_float(row.get("source_alignment_rate"))) is not None]))),
    ]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(label)} privacy_hopqa detail</title>
  <style>
    :root {{ --bg:#f5f4ef; --panel:#fff; --ink:#1e2430; --muted:#5f6b7a; --line:#d7dbe2; --accent:#1f5da8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:linear-gradient(180deg,#f8f7f2,#eef3f8); color:var(--ink); }}
    .page {{ max-width:1600px; margin:0 auto; padding:28px; }}
    .hero, .plot-card, .stat-card {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:0 12px 32px rgba(30,36,48,.05); }}
    .hero {{ padding:24px; margin-bottom:22px; }}
    .hero h1 {{ margin:0 0 8px; }}
    .hero p {{ color:var(--muted); }}
    .nav {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }}
    .nav a {{ color:var(--accent); text-decoration:none; background:#e7f0fb; border:1px solid #cfe0f6; border-radius:999px; padding:7px 11px; }}
    .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:18px 0; }}
    .stat-card {{ padding:14px; }}
    .stat-label {{ color:var(--muted); text-transform:uppercase; font-size:11px; letter-spacing:.06em; }}
    .stat-value {{ font-size:26px; font-weight:700; margin-top:6px; }}
    .plot-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }}
    .plot-card {{ padding:14px; overflow:auto; }}
    .plot-card h3 {{ margin:3px 0 10px; }}
    .plot-card img {{ width:100%; border:1px solid var(--line); border-radius:12px; background:white; }}
    .missing-note {{ min-height:220px; display:grid; place-items:center; color:var(--muted); border:1px dashed var(--line); border-radius:12px; background:#f8fafc; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{esc(label)} detail viewer</h1>
      <p>Generated {esc(iso(now))}. This page contains single-model distributions rather than mixed comparison plots. "Hops marked answered" means the agent marked hop state as answered; it is not final answer correctness. "Search calls using expected source" means web/local search matched the dataset hop source. </p>
      <div class="nav">
        <a href="privacy_hopqa_schedulerfix_classic_viewer.html">Mixed viewer</a>
        <a href="privacy_hopqa_schedulerfix_rollout_samples.html">Rollout samples</a>
        <a href="privacy_hopqa_schedulerfix_search_calls.html">Search calls</a>
      </div>
    </section>
    <div class="stat-grid">{''.join(stat_card_html(k, v) for k, v in stats)}</div>
    <section class="plot-grid">{cards}</section>
  </div>
</body>
</html>
"""
    name = f"privacy_hopqa_schedulerfix_{safe_slug(label)}_detail.html"
    (output_dir / name).write_text(html_text, encoding="utf-8")
    (output_dir.parent / name).write_text(html_text, encoding="utf-8")
    return name


def build_search_calls_html(
    output_dir: Path,
    action_rows: list[dict[str, Any]],
    now: datetime,
    samples_per_phase: int = 100,
) -> None:
    rng = random.Random(20260506)
    all_rows = sorted(
        action_rows,
        key=lambda row: (str(row.get("label") or ""), as_float(row.get("completed_at_s")) or 0.0),
    )
    total = len(all_rows)
    rows: list[dict[str, Any]] = []
    boundary_notes: list[str] = []
    for label in sorted({str(row.get("label") or "") for row in all_rows if row.get("label")}):
        label_rows = [row for row in all_rows if row.get("label") == label]
        n = len(label_rows)
        early_end = n // 3
        middle_end = (2 * n) // 3
        boundary_notes.append(
            f"{label}: n={n:,}; early indices [0,{early_end}), middle [{early_end},{middle_end}), late [{middle_end},{n})."
        )
        phase_buckets = [
            ("early", label_rows[:early_end]),
            ("middle", label_rows[early_end:middle_end]),
            ("late", label_rows[middle_end:]),
        ]
        for phase, bucket in phase_buckets:
            if not bucket:
                continue
            selected = rng.sample(bucket, min(samples_per_phase, len(bucket)))
            for row in selected:
                copied = dict(row)
                copied["sample_phase"] = phase
                rows.append(copied)
    rows = sorted(rows, key=lambda row: (str(row.get("label") or ""), str(row.get("sample_phase") or ""), as_float(row.get("completed_at_s")) or 0.0))
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{esc(row.get('sample_phase'))}</td>"
            f"<td>{esc(row.get('label'))}</td>"
            f"<td>{esc(row.get('rollout_id'))}</td>"
            f"<td>{esc(row.get('action_type'))}</td>"
            f"<td>{esc(row.get('expected_source'))}</td>"
            f"<td>{esc(row.get('actual_source'))}</td>"
            f"<td>{esc(row.get('hop_number'))}</td>"
            f"<td>{esc(row.get('iteration'))}</td>"
            f"<td>{fmt_num(row.get('execution_time_s'), 3)}s</td>"
            f"<td>{esc(row.get('results_count'))}</td>"
            f"<td>{esc(truncate_text(row.get('query'), 240))}</td>"
            f"<td>{esc(truncate_text(row.get('top_results'), 520))}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>privacy_hopqa search calls</title>
  <style>
    :root {{ --bg:#f5f4ef; --panel:#fff; --ink:#1e2430; --muted:#5f6b7a; --line:#d7dbe2; --accent:#1f5da8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:linear-gradient(180deg,#f8f7f2,#eef3f8); color:var(--ink); }}
    .page {{ max-width:1700px; margin:0 auto; padding:24px; }}
    .hero, .table-card {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:0 12px 32px rgba(30,36,48,.05); }}
    .hero {{ padding:22px; margin-bottom:18px; }}
    .hero p {{ color:var(--muted); }}
    .nav {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .nav a {{ color:var(--accent); text-decoration:none; background:#e7f0fb; border:1px solid #cfe0f6; border-radius:999px; padding:7px 11px; }}
    .table-card {{ padding:14px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th, td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#f7f9fc; z-index:1; }}
    code {{ background:#f2f5fa; border:1px solid var(--line); border-radius:6px; padding:1px 5px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Search calls / helper actions</h1>
      <p>Generated {esc(iso(now))}. Showing a deterministic random sample of up to {samples_per_phase} calls per run per phase, out of {total:,} cached search calls. The full cached set is exported in <code>trace_helper_actions.csv</code>.</p>
      <p>Early/middle/late phase formula after sorting calls within each run by rollout completion time: early = indices < n/3; middle = n/3 through 2n/3; late = indices >= 2n/3. Concrete boundaries: {esc(' '.join(boundary_notes))}</p>
      <p>These are the web/local helper-backed search actions stored in <code>trace.json/actions[*]</code>, with query text and top retrieved results/snippets when present. Raw LLM completions are still not stored separately in these traces.</p>
      <div class="nav">
        <a href="privacy_hopqa_schedulerfix_classic_viewer.html">Mixed viewer</a>
        <a href="privacy_hopqa_schedulerfix_rollout_samples.html">Rollout samples</a>
      </div>
    </section>
    <section class="table-card">
      <table>
        <thead><tr><th>phase</th><th>run</th><th>rollout</th><th>action</th><th>expected</th><th>actual</th><th>hop</th><th>iter</th><th>time</th><th>results</th><th>query</th><th>top results</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    name = "privacy_hopqa_schedulerfix_search_calls.html"
    (output_dir / name).write_text(html_text, encoding="utf-8")
    (output_dir.parent / name).write_text(html_text, encoding="utf-8")


def refresh(args: argparse.Namespace, runs: list[RunSpec]) -> None:
    now = utc_now()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_helper(output_dir, args.helper_url, now)
    snapshot_eai_runs(output_dir, runs, args.tail_lines, now)
    cached_stage_rows = normalize_row_labels(read_csv_rows(output_dir / "trace_stage_timings.csv"))
    cached_rollout_rows = normalize_row_labels(read_csv_rows(output_dir / "trace_rollout_timings.csv"))
    cached_action_rows = normalize_row_labels(read_csv_rows(output_dir / "trace_helper_actions.csv"))
    cached_error_rows = normalize_row_labels(read_csv_rows(output_dir / "trace_error_summary.csv"))
    seen_rollouts = {
        (str(row.get("label") or ""), str(row.get("rollout_id") or ""))
        for row in cached_rollout_rows
        if row.get("label") and row.get("rollout_id")
    }
    dataset_index = load_dataset_index(args.dataset_path)
    new_stage_rows, new_rollout_rows, new_action_rows, new_error_rows = parse_trace_timings(
        runs, seen_rollouts, args.new_trace_limit, dataset_index
    )
    trace_stage_rows, trace_rollout_rows, trace_action_rows, trace_error_rows = trim_trace_cache(
        cached_stage_rows + new_stage_rows,
        cached_rollout_rows + new_rollout_rows,
        cached_action_rows + new_action_rows,
        cached_error_rows + new_error_rows,
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
    write_csv(output_dir / "trace_error_summary.csv", trace_error_rows)

    plot_names = [
        plot_training_progress(output_dir, finetune_rows_by_run),
        plot_reward_throughput(output_dir, finetune_rows_by_run, args.rolling_window),
        plot_filtering(output_dir, run_snapshots, args.rolling_window),
        plot_helper_latency(output_dir, helper_flat, args.rolling_window),
        plot_trace_helper_action_latency(output_dir, trace_action_rows, args.rolling_window),
        plot_trace_search_latency(output_dir, trace_rollout_rows, args.rolling_window),
        plot_trace_stage_duration(output_dir, trace_stage_rows, args.rolling_window),
        plot_source_alignment_by_hop(output_dir, trace_stage_rows),
        plot_source_calls_by_iteration(output_dir, trace_stage_rows),
        plot_classic_search_doc_counts(output_dir, trace_rollout_rows, trace_stage_rows),
        plot_classic_rollout_distributions(output_dir, trace_rollout_rows),
        plot_classic_rollout_duration_hist(output_dir, trace_rollout_rows),
        plot_classic_llm_calls_hist(output_dir, trace_rollout_rows),
        plot_classic_total_tokens_hist(output_dir, trace_rollout_rows),
        plot_classic_total_tokens_by_final_iteration(output_dir, trace_rollout_rows),
        plot_classic_source_usage_scatter(output_dir, trace_rollout_rows),
        plot_classic_stage_duration_boxplot(output_dir, trace_stage_rows),
        plot_classic_source_usage_distributions(output_dir, trace_rollout_rows, trace_stage_rows),
        plot_classic_token_usage(output_dir, trace_rollout_rows),
        plot_classic_iteration_distribution(output_dir, trace_rollout_rows),
        plot_classic_searches_vs_docs(output_dir, trace_rollout_rows),
        plot_classic_search_source_confusion(output_dir, trace_stage_rows),
        plot_classic_doc_read_source_totals(output_dir, trace_stage_rows),
        plot_classic_stage_event_mix(output_dir, trace_stage_rows),
        plot_classic_pattern_alignment(output_dir, trace_rollout_rows),
        plot_classic_company_alignment(output_dir, trace_rollout_rows),
        plot_classic_rollout_duration_by_iteration(output_dir, trace_rollout_rows),
        plot_classic_planning_yield(output_dir, trace_stage_rows),
        plot_classic_resolved_hops(output_dir, trace_rollout_rows),
        plot_classic_temporal_rollout_metrics(output_dir, trace_rollout_rows, args.rolling_window),
        plot_classic_early_middle_late(output_dir, trace_rollout_rows),
        plot_classic_error_breakdown(output_dir, trace_error_rows),
        plot_classic_parse_and_empty_plan_breakdown(output_dir, trace_stage_rows, trace_error_rows),
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
    build_classic_html(
        output_dir=output_dir,
        runs=runs,
        rollout_rows=trace_rollout_rows,
        stage_rows=trace_stage_rows,
        action_rows=trace_action_rows,
        error_rows=trace_error_rows,
        plot_names=[name for name in plot_names if name],
        now=now,
    )
    build_rollout_samples_html(
        output_dir=output_dir,
        runs=runs,
        rollout_rows=trace_rollout_rows,
        samples_per_phase=args.samples_per_phase,
        now=now,
    )
    model_pages: list[str] = []
    for run in runs:
        model_dir = output_dir / f"model_{safe_slug(run.label)}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_rollouts = [row for row in trace_rollout_rows if row.get("label") == run.label]
        model_stages = [row for row in trace_stage_rows if row.get("label") == run.label]
        model_actions = [row for row in trace_action_rows if row.get("label") == run.label]
        model_errors = [row for row in trace_error_rows if row.get("label") == run.label]
        model_plot_names = [
            plot_classic_search_doc_counts(model_dir, model_rollouts, model_stages),
            plot_classic_rollout_distributions(model_dir, model_rollouts),
            plot_classic_rollout_duration_hist(model_dir, model_rollouts),
            plot_classic_llm_calls_hist(model_dir, model_rollouts),
            plot_classic_total_tokens_hist(model_dir, model_rollouts),
            plot_classic_total_tokens_by_final_iteration(model_dir, model_rollouts),
            plot_classic_source_usage_scatter(model_dir, model_rollouts),
            plot_classic_stage_duration_boxplot(model_dir, model_stages),
            plot_classic_source_usage_distributions(model_dir, model_rollouts, model_stages),
            plot_classic_token_usage(model_dir, model_rollouts),
            plot_classic_iteration_distribution(model_dir, model_rollouts),
            plot_classic_searches_vs_docs(model_dir, model_rollouts),
            plot_classic_search_source_confusion(model_dir, model_stages),
            plot_classic_doc_read_source_totals(model_dir, model_stages),
            plot_classic_stage_event_mix(model_dir, model_stages),
            plot_classic_pattern_alignment(model_dir, model_rollouts),
            plot_classic_company_alignment(model_dir, model_rollouts),
            plot_classic_rollout_duration_by_iteration(model_dir, model_rollouts),
            plot_classic_planning_yield(model_dir, model_stages),
            plot_classic_resolved_hops(model_dir, model_rollouts),
            plot_classic_temporal_rollout_metrics(model_dir, model_rollouts, args.rolling_window),
            plot_classic_early_middle_late(model_dir, model_rollouts),
            plot_classic_error_breakdown(model_dir, model_errors),
            plot_classic_parse_and_empty_plan_breakdown(model_dir, model_stages, model_errors),
            plot_trace_helper_action_latency(model_dir, model_actions, args.rolling_window),
            plot_trace_search_latency(model_dir, model_rollouts, args.rolling_window),
            plot_trace_stage_duration(model_dir, model_stages, args.rolling_window),
            plot_source_alignment_by_hop(model_dir, model_stages),
            plot_source_calls_by_iteration(model_dir, model_stages),
        ]
        model_pages.append(
            build_model_detail_html(
                output_dir=output_dir,
                run=run,
                model_dir=model_dir,
                plot_names=[name for name in model_plot_names if name],
                rollout_rows=trace_rollout_rows,
                stage_rows=trace_stage_rows,
                action_rows=trace_action_rows,
                error_rows=trace_error_rows,
                now=now,
            )
        )
    build_search_calls_html(output_dir, trace_action_rows, now, samples_per_phase=100)


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
