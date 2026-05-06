#!/usr/bin/env python3
"""Live training dashboard for the privacy_hopqa PipelineRL runs."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
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
    plot_names: list[str],
    now: datetime,
    helper_url: str,
) -> None:
    latest_snapshots = latest_by_label(run_snapshots)
    latest_helper = latest_helper_metrics(helper_flat_rows)

    run_cards: list[str] = []
    for run in runs:
        finetune_latest = finetune_rows_by_run.get(run.label, [])[-1] if finetune_rows_by_run.get(run.label) else {}
        snapshot = latest_snapshots.get(run.label, {})
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
      zero-advantage filtering, finetune progress, and helper search latency sampled from
      <code>{esc(helper_url.rstrip('/') + '/health')}</code>. The helper currently exposes count,
      active, errors, p50, p95, and max latency, not a true arithmetic mean; the latency plot is a
      moving average over sampled p50/p95 snapshots.
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

    plot_names = [
        plot_training_progress(output_dir, finetune_rows_by_run),
        plot_reward_throughput(output_dir, finetune_rows_by_run, args.rolling_window),
        plot_filtering(output_dir, run_snapshots, args.rolling_window),
        plot_helper_latency(output_dir, helper_flat, args.rolling_window),
    ]
    build_html(
        output_dir=output_dir,
        runs=runs,
        finetune_rows_by_run=finetune_rows_by_run,
        run_snapshots=run_snapshots,
        helper_flat_rows=helper_flat,
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
