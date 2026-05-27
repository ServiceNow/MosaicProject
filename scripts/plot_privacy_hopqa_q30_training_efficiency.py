#!/usr/bin/env python3
"""Plot q30 training efficiency by generated samples and filtering."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_q30_live_analysis_viewer as live
from scripts import build_privacy_hopqa_q30_paper_viewer as paper
from scripts.evaluate_privacy_hopqa_leakage_suite_v2 import terminal_model_error_fingerprints


DEFAULT_OUTPUT_DIR = ROOT / "runtime_analysis" / "privacy_hopqa_q30_training_efficiency_20260527"
BASE_LABEL = "qwen3_4b_instruct_2507"
CHECKPOINT_STEPS = [0, 20, 40, 60, 80, 100]

NORMAL_FINAL_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_16kgen_32kctx_helperfe_20260522"
)
MISSING_CKPT_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_missing_ckpts_16kgen_32kctx_helperfe_20260524"
)
OUTCOME_EVAL_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_outcome_hopacc_every20_to100_eval_20260527"
)

SERIES = [
    {
        "family": "outcome_hop_accuracy",
        "display": "Outcome reward",
        "short_display": "Outcome",
        "prefix": "qwen3_4b_outcome_hopacc",
        "train_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_4b_outcome_hopacc_lr1e6_rep8_200s_20260524232644"
        ),
        "color": "#0072B2",
        "marker": "o",
    },
    {
        "family": "task_reward",
        "display": "Task reward",
        "short_display": "Task",
        "prefix": "qwen3_4b_situational_reward",
        "train_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_4b_situational_reward_lr1e6_ga4096_rep4_50s_20260520173055"
        ),
        "color": "#D55E00",
        "marker": "s",
    },
    {
        "family": "privacy_aware_tau05",
        "display": "Task+PA-DR reward",
        "short_display": "PA-DR",
        "prefix": "qwen3_4b_privreward_tau05",
        "train_root": Path(
            "/mnt/llmd/results/exps/alexg/reason/"
            "privacy_hopqa_q30_4b_situational_privreward_tau05_lr1e6_rep8_200s_20260522191644"
        ),
        "color": "#009E73",
        "marker": "^",
    },
]

FINETUNE_TS_RE = re.compile(r"\[finetune\]: (\d\d/\d\d/\d{4} \d\d:\d\d:\d\d\.\d{3})")
START_RE = re.compile(r"Starting training with (\d+) samples, .*? (\d+) samples per step")
COMPLETED_RE = re.compile(r"Completed steps (\d+):")
STATS_SAMPLES_RE = re.compile(r"'stats/samples': '([0-9.]+)'")
PREPROCESS_TS_RE = re.compile(r"\[preprocessor\]: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
PROCESSED_RE = re.compile(
    r"Processed (\d+) samples \(filtered out (\d+)/(\d+)\).*total (\d+) samples so far"
)


def _read_lines(path: Path) -> list[str]:
    return path.read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore").splitlines()


def parse_finetune_timing(root: Path) -> dict[str, Any]:
    starts: list[int] = []
    samples_per_step: int | None = None
    step100_samples: int | None = None
    step100_completed_at: datetime | None = None
    step100_saved_at: datetime | None = None
    current_step: int | None = None

    for line in _read_lines(root / "finetune" / "log" / "info_0.log"):
        match = FINETUNE_TS_RE.search(line)
        if not match:
            continue
        timestamp = datetime.strptime(match.group(1), "%m/%d/%Y %H:%M:%S.%f")

        start_match = START_RE.search(line)
        if start_match:
            starts.append(int(start_match.group(1)))
            samples_per_step = int(start_match.group(2))

        completed_match = COMPLETED_RE.search(line)
        if completed_match:
            current_step = int(completed_match.group(1))
            samples_match = STATS_SAMPLES_RE.search(line)
            if current_step == 100:
                step100_completed_at = timestamp
                if samples_match:
                    step100_samples = int(float(samples_match.group(1)))

        if current_step == 100 and "Make sure to save PEFT adapter" in line:
            step100_saved_at = timestamp

    if samples_per_step is None:
        raise RuntimeError(f"Could not parse samples per step from {root}")
    if step100_samples is None:
        step100_samples = samples_per_step * 100
    if step100_saved_at is None:
        if step100_completed_at is None:
            raise RuntimeError(f"Could not parse checkpoint-100 time from {root}")
        step100_saved_at = step100_completed_at

    starts = sorted(dict.fromkeys(start for start in starts if start < step100_samples))
    if not starts or starts[0] != 0:
        starts.insert(0, 0)

    return {
        "samples_per_step": samples_per_step,
        "step100_samples": step100_samples,
        "step100_saved_at": step100_saved_at,
        "restart_start_samples": starts,
    }


def parse_preprocessor_rows(root: Path, cutoff: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_total: int | None = None
    segment = -1
    for line in _read_lines(root / "preprocess" / "debug.log"):
        ts_match = PREPROCESS_TS_RE.search(line)
        processed_match = PROCESSED_RE.search(line)
        if not ts_match or not processed_match:
            continue
        timestamp = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
        if timestamp > cutoff:
            continue
        accepted, filtered, seen, total = map(int, processed_match.groups())
        if last_total is None or total < last_total:
            segment += 1
        rows.append(
            {
                "timestamp": timestamp,
                "segment": segment,
                "accepted": accepted,
                "filtered": filtered,
                "seen": seen,
                "begin": total - accepted,
                "end": total,
                "total": total,
            }
        )
        last_total = total
    return rows


def allocate_effective_steps(
    rows: list[dict[str, Any]],
    restart_start_samples: list[int],
    target_samples: int,
    samples_per_step: int,
) -> tuple[dict[int, dict[str, float]], list[dict[str, float]]]:
    step_stats = {
        step: {"accepted": 0.0, "filtered": 0.0, "seen": 0.0}
        for step in range(1, 101)
    }
    windows: list[tuple[int, int]] = []
    for index, start in enumerate(restart_start_samples):
        end = restart_start_samples[index + 1] if index + 1 < len(restart_start_samples) else target_samples
        end = min(end, target_samples)
        if start < end:
            windows.append((start, end))

    segment_stats: list[dict[str, float]] = [
        {"accepted": 0.0, "filtered": 0.0, "seen": 0.0, "physical_seen": 0.0}
        for _ in windows
    ]

    for row in rows:
        segment = int(row["segment"])
        if segment >= len(windows):
            continue
        valid_start, valid_end = windows[segment]
        begin = max(int(row["begin"]), valid_start)
        end = min(int(row["end"]), valid_end)
        if end <= begin or row["accepted"] <= 0:
            continue
        row_accepted = float(row["accepted"])
        segment_stats[segment]["physical_seen"] += float(row["seen"])
        for step in range(
            max(1, begin // samples_per_step + 1),
            min(100, (end - 1) // samples_per_step + 1) + 1,
        ):
            step_begin = (step - 1) * samples_per_step
            step_end = step * samples_per_step
            overlap = max(0, min(end, step_end) - max(begin, step_begin))
            if overlap <= 0:
                continue
            fraction = overlap / row_accepted
            accepted = float(overlap)
            filtered = float(row["filtered"]) * fraction
            seen = float(row["seen"]) * fraction
            step_stats[step]["accepted"] += accepted
            step_stats[step]["filtered"] += filtered
            step_stats[step]["seen"] += seen
            segment_stats[segment]["accepted"] += accepted
            segment_stats[segment]["filtered"] += filtered
            segment_stats[segment]["seen"] += seen
    return step_stats, segment_stats


def training_efficiency(series: dict[str, Any]) -> dict[str, Any]:
    root = Path(series["train_root"])
    timing = parse_finetune_timing(root)
    rows = parse_preprocessor_rows(root, timing["step100_saved_at"])
    step_stats, segment_stats = allocate_effective_steps(
        rows,
        timing["restart_start_samples"],
        timing["step100_samples"],
        timing["samples_per_step"],
    )
    physical = {
        "accepted": sum(float(row["accepted"]) for row in rows),
        "filtered": sum(float(row["filtered"]) for row in rows),
        "seen": sum(float(row["seen"]) for row in rows),
    }
    effective = {
        "accepted": sum(step_stats[step]["accepted"] for step in step_stats),
        "filtered": sum(step_stats[step]["filtered"] for step in step_stats),
        "seen": sum(step_stats[step]["seen"] for step in step_stats),
    }
    return {
        "root": str(root),
        "timing": timing,
        "physical": physical,
        "effective": effective,
        "step_stats": step_stats,
        "segment_stats": segment_stats,
    }


def read_model_rows(root: Path, model_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    model_root = root / "model_runs" / model_label
    if not model_root.exists():
        return [], [], 0
    tables = model_root / "tables"
    rollouts = []
    for row in live.read_jsonl(tables / "rollout_index.jsonl"):
        row = dict(row)
        row["model_label"] = str(row.get("model_label") or model_label)
        rollouts.append(row)
    rollout_latest = list(live.latest_by_identity(rollouts).values())
    rollout_latest = [row for row in rollout_latest if live.row_split(row) == "test"]

    raw_leakage = []
    for row in live.read_jsonl(tables / "privacy_leakage_suite_v2_stepfun_full.jsonl"):
        row = dict(row)
        row["model_label"] = str(row.get("model_label") or model_label)
        row["judge"] = "stepfun"
        raw_leakage.append(row)
    leakage = live.copy_rollout_identity_from_trace(raw_leakage, rollouts)
    leakage = live.dedupe_rows_by_latest(leakage, "stepfun")
    ok_rollout_ids = {live.rollout_identity(row) for row in rollout_latest if row.get("ok")}
    leakage = [
        row
        for row in leakage
        if live.row_split(row) == "test" and row.get("ok") and live.rollout_identity(row) in ok_rollout_ids
    ]
    ok_fingerprints = {
        str(row.get("input_fingerprint"))
        for row in raw_leakage
        if row.get("ok") and row.get("input_fingerprint")
    }
    terminal_errors = len(
        terminal_model_error_fingerprints(tables / "privacy_leakage_suite_v2_stepfun_full_errors.jsonl")
        - ok_fingerprints
    )
    return rollout_latest, leakage, terminal_errors


def mean_sem(values: list[float]) -> tuple[float | None, float | None]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None, None
    if len(clean) == 1:
        return mean(clean), 0.0
    return mean(clean), stdev(clean) / math.sqrt(len(clean))


def repeat_grouped_rate(rows: list[dict[str, Any]], value_fn: Any) -> tuple[float | None, float | None, str, int]:
    values: list[float] = []
    by_repeat: dict[str, list[float]] = {}
    for row in rows:
        value = value_fn(row)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        values.append(value)
        repeat = row.get("repeat")
        if repeat is not None:
            by_repeat.setdefault(str(repeat), []).append(value)
    if by_repeat and len(by_repeat) >= 2:
        repeat_means = [mean(items) for items in by_repeat.values() if items]
        center, sem = mean_sem(repeat_means)
        return center, sem, "repeat", len(repeat_means)
    center, sem = mean_sem(values)
    return center, sem, "row", len(values)


def model_label_for(series: dict[str, Any], step: int) -> str:
    if step == 0:
        return BASE_LABEL
    return f"{series['prefix']}_s{step}"


def eval_root_for(series: dict[str, Any], step: int) -> Path:
    if step == 0:
        return NORMAL_FINAL_ROOT
    family = series["family"]
    if family == "outcome_hop_accuracy":
        return OUTCOME_EVAL_ROOT
    if family == "task_reward" and step == 100:
        return NORMAL_FINAL_ROOT
    return MISSING_CKPT_ROOT


def metric_row(series: dict[str, Any], step: int) -> dict[str, Any] | None:
    label = model_label_for(series, step)
    rollouts, leakage, terminal_errors = read_model_rows(eval_root_for(series, step), label)
    if not rollouts:
        return None
    accuracy, accuracy_sem, accuracy_sem_source, accuracy_n = repeat_grouped_rate(
        rollouts,
        live.attempt_task_accuracy,
    )
    strict_chain, strict_chain_sem, strict_chain_sem_source, strict_chain_n = repeat_grouped_rate(
        rollouts,
        lambda row: 1.0 if row.get("strict_chain_success") is True or row.get("strict_chain_success") == 1 else 0.0,
    )
    leakage_rate, leakage_sem, leakage_sem_source, leakage_n = repeat_grouped_rate(
        leakage,
        lambda row: 1.0 if live.reward_positive(row) else 0.0,
    )
    if strict_chain is None or accuracy is None or leakage_rate is None:
        return None
    return {
        "family": series["family"],
        "family_display": series["display"],
        "step": step,
        "model_label": label,
        "eval_root": str(eval_root_for(series, step)),
        "accuracy": accuracy,
        "accuracy_sem": accuracy_sem,
        "accuracy_sem_source": accuracy_sem_source,
        "accuracy_sem_n": accuracy_n,
        "strict_chain_success": strict_chain,
        "strict_chain_success_sem": strict_chain_sem,
        "strict_chain_success_sem_source": strict_chain_sem_source,
        "strict_chain_success_sem_n": strict_chain_n,
        "privacy_leakage_rate": leakage_rate,
        "privacy_leakage_sem": leakage_sem,
        "privacy_leakage_sem_source": leakage_sem_source,
        "privacy_leakage_sem_n": leakage_n,
        "rollout_rows": len(rollouts),
        "stepfun_ok_rows": len(leakage),
        "terminal_model_errors": terminal_errors,
    }


def pct(value: float | None) -> float:
    if value is None:
        return 0.0
    return 100.0 * float(value)


def with_efficiency_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for series in SERIES:
        efficiency = training_efficiency(series)
        step_stats = efficiency["step_stats"]
        cumulative = {"accepted": 0.0, "filtered": 0.0, "seen": 0.0}
        by_step: dict[int, dict[str, float]] = {
            0: {"accepted": 0.0, "filtered": 0.0, "seen": 0.0}
        }
        for step in range(1, 101):
            for key in cumulative:
                cumulative[key] += step_stats[step][key]
            by_step[step] = dict(cumulative)

        for step in CHECKPOINT_STEPS:
            metrics = metric_row(series, step)
            if metrics is None:
                continue
            row = {
                **metrics,
                "effective_accepted_train_samples": by_step[step]["accepted"],
                "effective_filtered_train_samples": by_step[step]["filtered"],
                "effective_generated_train_samples": by_step[step]["seen"],
                "effective_generated_per_accepted": (
                    by_step[step]["seen"] / by_step[step]["accepted"] if by_step[step]["accepted"] else 0.0
                ),
            }
            rows.append(row)

        for lo in [1, 21, 41, 61, 81]:
            hi = lo + 19
            accepted = sum(step_stats[step]["accepted"] for step in range(lo, hi + 1))
            filtered = sum(step_stats[step]["filtered"] for step in range(lo, hi + 1))
            seen = sum(step_stats[step]["seen"] for step in range(lo, hi + 1))
            bin_rows.append(
                {
                    "family": series["family"],
                    "family_display": series["display"],
                    "bin_start_step": lo,
                    "bin_end_step": hi,
                    "accepted_train_samples": accepted,
                    "filtered_train_samples": filtered,
                    "generated_train_samples": seen,
                    "filtered_fraction": filtered / seen if seen else None,
                }
            )

        physical = efficiency["physical"]
        effective = efficiency["effective"]
        summaries.append(
            {
                "family": series["family"],
                "family_display": series["display"],
                "train_root": str(series["train_root"]),
                "step100_accepted_train_samples": effective["accepted"],
                "step100_effective_filtered_train_samples": effective["filtered"],
                "step100_effective_generated_train_samples": effective["seen"],
                "step100_physical_generated_train_samples": physical["seen"],
                "step100_restart_or_ahead_waste_train_samples": physical["seen"] - effective["seen"],
                "step100_effective_generated_per_accepted": effective["seen"] / effective["accepted"],
                "step100_physical_generated_per_accepted": physical["seen"] / effective["accepted"],
                "step100_effective_filtered_fraction": effective["filtered"] / effective["seen"],
                "step100_physical_filtered_fraction": physical["filtered"] / physical["seen"],
                "samples_per_step": efficiency["timing"]["samples_per_step"],
                "restart_start_samples": efficiency["timing"]["restart_start_samples"],
                "step100_saved_at": efficiency["timing"]["step100_saved_at"].isoformat(),
            }
        )
    return rows, summaries, bin_rows


def save_figure(fig: Any, path: Path) -> None:
    paper.save_figure(fig, path)


def annotate_final(ax: Any, x: float, y: float, label: str, color: str, *, dy: float = 0.0) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(6, dy),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.90},
    )


def plot_metric_progress(rows: list[dict[str, Any]], path: Path) -> None:
    paper.paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharex=True)
    metrics = [
        ("strict_chain_success", "Strict Chain Success (%)"),
        ("privacy_leakage_rate", "Privacy Leakage (%)"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for series in SERIES:
            data = sorted(
                [row for row in rows if row["family"] == series["family"]],
                key=lambda row: int(row["step"]),
            )
            xs = [row["effective_generated_train_samples"] / 1000.0 for row in data]
            ys = [pct(row[metric]) for row in data]
            yerr = [pct(row.get(f"{metric}_sem")) for row in data]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=series["marker"],
                markersize=5.2,
                linewidth=1.8,
                capsize=2.8,
                color=series["color"],
                label=series["display"],
            )
            if data:
                annotate_final(ax, xs[-1], ys[-1], series["short_display"], series["color"])
        ax.set_xlabel("Effective generated train samples (thousands)")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        paper.add_minor_grid(ax)
    axes[0].set_ylim(35, 70)
    axes[1].set_ylim(0, 65)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncols=3, frameon=False)
    save_figure(fig, path)


def plot_optimizer_progress(rows: list[dict[str, Any]], path: Path) -> None:
    paper.paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for series in SERIES:
        data = sorted(
            [row for row in rows if row["family"] == series["family"]],
            key=lambda row: int(row["step"]),
        )
        xs = [row["effective_generated_train_samples"] / 1000.0 for row in data]
        ys = [int(row["step"]) for row in data]
        ax.plot(
            xs,
            ys,
            marker=series["marker"],
            markersize=5.4,
            linewidth=2.0,
            color=series["color"],
            label=series["display"],
        )
        if data:
            annotate_final(ax, xs[-1], ys[-1], f"{series['short_display']} 100", series["color"])
    ax.set_xlabel("Effective generated train samples (thousands)")
    ax.set_ylabel("Optimizer step")
    ax.set_ylim(-2, 105)
    ax.set_xlim(0, 1030)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    paper.add_minor_grid(ax)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, path)


def plot_filter_fraction(bin_rows: list[dict[str, Any]], path: Path) -> None:
    paper.paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for series in SERIES:
        data = [row for row in bin_rows if row["family"] == series["family"]]
        xs = [int(row["bin_end_step"]) for row in data]
        ys = [pct(row["filtered_fraction"]) for row in data]
        ax.plot(
            xs,
            ys,
            marker=series["marker"],
            markersize=5.4,
            linewidth=2.0,
            color=series["color"],
            label=series["display"],
        )
        if data:
            annotate_final(ax, xs[-1], ys[-1], series["short_display"], series["color"])
    ax.set_xlabel("Optimizer step window end")
    ax.set_ylabel("Filtered fraction of generated train samples (%)")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 65)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    paper.add_minor_grid(ax)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, path)


def plot_step100_accounting(summaries: list[dict[str, Any]], path: Path) -> None:
    paper.paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    labels = [row["family_display"] for row in summaries]
    accepted = [row["step100_accepted_train_samples"] / 1000.0 for row in summaries]
    filtered = [row["step100_effective_filtered_train_samples"] / 1000.0 for row in summaries]
    waste = [row["step100_restart_or_ahead_waste_train_samples"] / 1000.0 for row in summaries]
    xs = list(range(len(labels)))
    ax.bar(xs, accepted, color="#5F6F7A", label="Accepted by trainer")
    ax.bar(xs, filtered, bottom=accepted, color="#D55E00", label="Filtered zero-advantage")
    ax.bar(
        xs,
        waste,
        bottom=[a + f for a, f in zip(accepted, filtered)],
        color="#CC79A7",
        label="Restart/ahead waste",
    )
    for idx, row in enumerate(summaries):
        total = row["step100_physical_generated_train_samples"] / 1000.0
        ax.text(idx, total + 18, f"{total:.0f}k", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Generated train samples to checkpoint 100 (thousands)")
    ax.set_ylim(0, 1060)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    paper.add_minor_grid(ax, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncols=3, frameon=False)
    save_figure(fig, path)


def samples_to_target(data: list[dict[str, Any]], target: float) -> float | None:
    points = sorted(
        [
            (float(row["effective_generated_train_samples"]), float(row["strict_chain_success"]))
            for row in data
        ],
        key=lambda item: item[0],
    )
    best_so_far: list[tuple[float, float]] = []
    current_best = -math.inf
    for x_value, y_value in points:
        current_best = max(current_best, y_value)
        best_so_far.append((x_value, current_best))
    for (x0, y0), (x1, y1) in zip(best_so_far, best_so_far[1:]):
        if y0 >= target:
            return x0
        if y0 < target <= y1:
            if y1 == y0:
                return x1
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    if best_so_far and best_so_far[-1][1] >= target:
        return best_so_far[-1][0]
    return None


def plot_quality_thresholds(rows: list[dict[str, Any]], path: Path) -> None:
    paper.paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax_curve, ax_bar = axes

    max_x = 0.0
    for series in SERIES:
        data = sorted(
            [row for row in rows if row["family"] == series["family"]],
            key=lambda row: float(row["effective_generated_train_samples"]),
        )
        xs = [float(row["effective_generated_train_samples"]) / 1000.0 for row in data]
        ys = [pct(row["strict_chain_success"]) for row in data]
        best = []
        current_best = -math.inf
        for y_value in ys:
            current_best = max(current_best, y_value)
            best.append(current_best)
        if xs:
            max_x = max(max_x, max(xs))
        ax_curve.plot(
            xs,
            best,
            marker=series["marker"],
            markersize=5.2,
            linewidth=2.0,
            color=series["color"],
            label=series["display"],
        )
        ax_curve.scatter(xs, ys, s=22, color=series["color"], alpha=0.35)
        if xs:
            annotate_final(ax_curve, xs[-1], best[-1], series["short_display"], series["color"])

    ax_curve.set_xlabel("Effective generated train samples (thousands)")
    ax_curve.set_ylabel("Best strict chain success so far (%)")
    ax_curve.set_ylim(47, 62)
    ax_curve.set_xlim(0, 1030)
    ax_curve.xaxis.set_major_locator(MultipleLocator(200))
    ax_curve.xaxis.set_minor_locator(MultipleLocator(100))
    ax_curve.yaxis.set_major_locator(MultipleLocator(5))
    ax_curve.yaxis.set_minor_locator(MultipleLocator(1))
    paper.add_minor_grid(ax_curve)
    ax_curve.legend(loc="lower right", frameon=False)

    targets = [0.55, 0.58, 0.59]
    y_centers = list(range(len(targets)))
    offsets = [-0.24, 0.0, 0.24]
    height = 0.19
    for series_index, series in enumerate(SERIES):
        data = [row for row in rows if row["family"] == series["family"]]
        for target_index, target in enumerate(targets):
            y_pos = y_centers[target_index] + offsets[series_index]
            sample_count = samples_to_target(data, target)
            if sample_count is None:
                capped_x = max_x
                ax_bar.barh(
                    y_pos,
                    capped_x,
                    height=height,
                    color=series["color"],
                    alpha=0.13,
                    edgecolor=series["color"],
                    hatch="//",
                    linewidth=0.8,
                )
                ax_bar.text(
                    min(capped_x + 18, 1010),
                    y_pos,
                    "not reached",
                    va="center",
                    ha="left",
                    fontsize=7.8,
                    color=series["color"],
                )
                continue
            x_value = sample_count / 1000.0
            ax_bar.barh(y_pos, x_value, height=height, color=series["color"], alpha=0.88)
            ax_bar.text(
                x_value + 9,
                y_pos,
                f"{x_value:.0f}k",
                va="center",
                ha="left",
                fontsize=7.8,
                color=series["color"],
            )

    ax_bar.set_xlabel("Effective generated train samples to target (thousands)")
    ax_bar.set_yticks(y_centers)
    ax_bar.set_yticklabels([f"{100 * target:.0f}% strict chain" for target in targets])
    ax_bar.set_xlim(0, 1030)
    ax_bar.xaxis.set_major_locator(MultipleLocator(200))
    ax_bar.xaxis.set_minor_locator(MultipleLocator(100))
    paper.add_minor_grid(ax_bar, axis="x")
    handles = [
        Line2D([0], [0], color=series["color"], lw=6, label=series["short_display"])
        for series in SERIES
    ]
    ax_bar.legend(handles=handles, loc="lower right", frameon=False)
    save_figure(fig, path)


def write_tables(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], bin_rows: list[dict[str, Any]], output_dir: Path) -> None:
    rows_sorted = sorted(rows, key=lambda row: (row["family"], int(row["step"])))
    with (output_dir / "training_efficiency_checkpoint_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows_sorted[0].keys()) if rows_sorted else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (output_dir / "training_efficiency_filter_bins.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(bin_rows[0].keys()) if bin_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(bin_rows)
    (output_dir / "training_efficiency_summary.json").write_text(
        json.dumps({"checkpoint_rows": rows_sorted, "summary_rows": summaries, "filter_bin_rows": bin_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def build(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries, bin_rows = with_efficiency_rows()
    write_tables(rows, summaries, bin_rows, output_dir)
    plot_metric_progress(rows, output_dir / "training_sample_efficiency_metrics.svg")
    plot_optimizer_progress(rows, output_dir / "training_optimizer_progress_vs_generation.svg")
    plot_filter_fraction(bin_rows, output_dir / "training_filter_fraction_by_step.svg")
    plot_step100_accounting(summaries, output_dir / "training_generation_accounting_step100.svg")
    plot_quality_thresholds(rows, output_dir / "training_quality_threshold_samples.svg")
    return {"checkpoint_rows": rows, "summary_rows": summaries, "filter_bin_rows": bin_rows}


def main() -> None:
    payload = build(DEFAULT_OUTPUT_DIR)
    print(f"wrote {DEFAULT_OUTPUT_DIR}")
    print(f"checkpoint_rows={len(payload['checkpoint_rows'])}")
    print(f"summary_rows={len(payload['summary_rows'])}")


if __name__ == "__main__":
    main()
