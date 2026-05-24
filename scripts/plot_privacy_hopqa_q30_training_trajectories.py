#!/usr/bin/env python3
"""Plot q30 checkpoint trajectories for task accuracy vs privacy leakage."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_q30_live_analysis_viewer as live
from scripts.evaluate_privacy_hopqa_leakage_suite_v2 import terminal_model_error_fingerprints


NORMAL_FINAL_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_16kgen_32kctx_helperfe_20260522"
)
MISSING_CKPT_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_missing_ckpts_16kgen_32kctx_helperfe_20260524"
)
PRIVREWARD_S50_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_privreward_tau05_s50_16kgen_32kctx_helperfe_20260523"
)
PRIVACY_PROMPT_FINAL_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_privacy_prompt_16kgen_32kctx_helperfe_20260522"
)
PRIVACY_PROMPT_PRIVREWARD_S50_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_privreward_tau05_s50_privacy_prompt_16kgen_32kctx_helperfe_20260523"
)
PRIVACY_PROMPT_PRIVREWARD_S100_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_privreward_tau05_s100_privacy_prompt_16kgen_32kctx_helperfe_20260524"
)
DEFAULT_OUTPUT_DIR = ROOT / "runtime_analysis" / "privacy_hopqa_q30_training_trajectory_20260524"

BASE_LABEL = "qwen3_4b_instruct_2507"
STEPS = list(range(10, 101, 10))


SERIES = [
    {
        "family": "task_reward",
        "display": "Task reward",
        "prefix": "qwen3_4b_situational_reward",
        "color": "#0072B2",
        "marker": "o",
    },
    {
        "family": "privacy_aware_tau05",
        "display": "Task + PA-DR (tau=0.5)",
        "prefix": "qwen3_4b_privreward_tau05",
        "color": "#D55E00",
        "marker": "s",
    },
]


def model_label_for(series: dict[str, str], step: int) -> str:
    if step == 0:
        return BASE_LABEL
    return f"{series['prefix']}_s{step}"


def normal_root_for(label: str) -> Path:
    if label == BASE_LABEL or label in {"qwen3_4b_situational_reward_s50", "qwen3_4b_situational_reward_s100"}:
        return NORMAL_FINAL_ROOT
    if label == "qwen3_4b_privreward_tau05_s50":
        return PRIVREWARD_S50_ROOT
    return MISSING_CKPT_ROOT


def privacy_prompt_root_for(label: str) -> Path | None:
    if label in {BASE_LABEL, "qwen3_4b_situational_reward_s50", "qwen3_4b_situational_reward_s100"}:
        return PRIVACY_PROMPT_FINAL_ROOT
    if label == "qwen3_4b_privreward_tau05_s50":
        return PRIVACY_PROMPT_PRIVREWARD_S50_ROOT
    if label == "qwen3_4b_privreward_tau05_s100":
        return PRIVACY_PROMPT_PRIVREWARD_S100_ROOT
    return None


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
    terminal_errors = len(terminal_model_error_fingerprints(tables / "privacy_leakage_suite_v2_stepfun_full_errors.jsonl"))
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


def metric_row(condition: str, series: dict[str, str], step: int, root: Path) -> dict[str, Any] | None:
    label = model_label_for(series, step)
    rollouts, leakage, terminal_errors = read_model_rows(root, label)
    if not rollouts:
        return None
    accuracy, accuracy_sem, accuracy_sem_source, accuracy_n = repeat_grouped_rate(
        rollouts,
        live.attempt_task_accuracy,
    )
    leakage_rate, leakage_sem, leakage_sem_source, leakage_n = repeat_grouped_rate(
        leakage,
        lambda row: 1.0 if live.reward_positive(row) else 0.0,
    )
    if accuracy is None or leakage_rate is None:
        return None
    return {
        "condition": condition,
        "family": series["family"],
        "family_display": series["display"],
        "step": step,
        "model_label": label,
        "artifact_root": str(root),
        "accuracy": accuracy,
        "accuracy_sem": accuracy_sem,
        "accuracy_sem_source": accuracy_sem_source,
        "accuracy_sem_n": accuracy_n,
        "privacy_leakage_rate": leakage_rate,
        "privacy_leakage_sem": leakage_sem,
        "privacy_leakage_sem_source": leakage_sem_source,
        "privacy_leakage_sem_n": leakage_n,
        "rollout_rows": len(rollouts),
        "stepfun_ok_rows": len(leakage),
        "terminal_model_errors": terminal_errors,
    }


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series in SERIES:
        for step in [0, *STEPS]:
            root = normal_root_for(model_label_for(series, step))
            row = metric_row("normal_prompt", series, step, root)
            if row:
                rows.append(row)
        for step in [0, 50, 100]:
            label = model_label_for(series, step)
            root = privacy_prompt_root_for(label)
            if root is None:
                continue
            row = metric_row("privacy_prompt", series, step, root)
            if row:
                rows.append(row)
    return rows


def pct(value: float | None) -> float:
    return 100.0 * float(value or 0.0)


def add_arrow(ax: Any, start: dict[str, Any], end: dict[str, Any], color: str) -> None:
    x0, y0 = pct(start["accuracy"]), pct(start["privacy_leakage_rate"])
    x1, y1 = pct(end["accuracy"]), pct(end["privacy_leakage_rate"])
    if abs(x1 - x0) < 0.05 and abs(y1 - y0) < 0.05:
        return
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.5,
        color=color,
        alpha=0.64,
        shrinkA=7,
        shrinkB=7,
    )
    ax.add_patch(arrow)


def label_offset(row: dict[str, Any], index: int) -> tuple[float, float]:
    if row["step"] == 0:
        return 7.0, -1.5
    if row["condition"] == "normal_prompt" and row["family"] == "task_reward":
        offsets = [(2, 7), (8, -5), (-14, 7), (8, 7), (-15, -8), (8, 7), (-14, 7), (8, -8), (-14, 7), (8, 7)]
        return offsets[(index - 1) % len(offsets)]
    if row["condition"] == "normal_prompt" and row["family"] == "privacy_aware_tau05":
        offsets = [(4, -11), (-14, -9), (5, 6), (-16, 5), (6, -9), (-16, -8), (6, 6), (-15, -9), (6, 6), (-14, -9)]
        return offsets[(index - 1) % len(offsets)]
    if row["family"] == "task_reward":
        return 5.0, 7.0
    return 5.0, -11.0


def plot_panel(ax: Any, rows: list[dict[str, Any]], condition: str, title: str, include_all_steps: bool) -> None:
    condition_rows = [row for row in rows if row["condition"] == condition]
    for series in SERIES:
        data = sorted(
            [row for row in condition_rows if row["family"] == series["family"]],
            key=lambda row: row["step"],
        )
        if not data:
            continue
        xs = [pct(row["accuracy"]) for row in data]
        ys = [pct(row["privacy_leakage_rate"]) for row in data]
        xerr = [pct(row.get("accuracy_sem")) for row in data]
        yerr = [pct(row.get("privacy_leakage_sem")) for row in data]
        ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            yerr=yerr,
            fmt=series["marker"],
            color=series["color"],
            ecolor=series["color"],
            elinewidth=0.9,
            capsize=2,
            markersize=5.8,
            markeredgecolor="#1f2421",
            markeredgewidth=0.5,
            alpha=0.95,
            label=series["display"],
        )
        for start, end in zip(data, data[1:]):
            add_arrow(ax, start, end, series["color"])
        for index, (row, x, y) in enumerate(zip(data, xs, ys)):
            label = "base" if row["step"] == 0 else str(row["step"])
            dx, dy = label_offset(row, index)
            ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                color=series["color"],
                bbox={"boxstyle": "round,pad=0.13", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
    subtitle = "base -> every 10 steps -> 100" if include_all_steps else "available privacy-prompt evals: base, 50, 100"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel("Chain progress accuracy (%)")
    ax.grid(alpha=0.25)


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda row: (row["condition"], row["family"], row["step"]))
    (output_dir / "training_trajectory_metrics.json").write_text(
        json.dumps(rows_sorted, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "training_trajectory_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows_sorted[0].keys()) if rows_sorted else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    all_x = [pct(row["accuracy"]) for row in rows]
    all_y = [pct(row["privacy_leakage_rate"]) for row in rows]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = max(2.0, (x_max - x_min) * 0.15)
    y_pad = max(3.0, (y_max - y_min) * 0.15)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), sharex=True, sharey=True)
    plot_panel(axes[0], rows, "normal_prompt", "Normal prompt", include_all_steps=True)
    plot_panel(axes[1], rows, "privacy_prompt", "Privacy prompt", include_all_steps=False)
    axes[0].set_ylabel("StepFun privacy leakage (% score 4-5)")
    for ax in axes:
        ax.set_xlim(max(0, x_min - x_pad), min(100, x_max + x_pad))
        ax.set_ylim(max(0, y_min - y_pad), min(100, y_max + y_pad))
        ax.tick_params(axis="both", labelsize=9)
    handles = [
        Line2D([0], [0], marker=series["marker"], color=series["color"], label=series["display"], markersize=6, linewidth=1.5)
        for series in SERIES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94), frameon=False)
    fig.suptitle("Training trajectory: task performance vs privacy leakage", y=0.995, fontsize=13)
    fig.text(
        0.5,
        -0.01,
        "Each point is one checkpoint on test rollouts. Error bars are SEM over repeat means. "
        "Privacy leakage uses successful StepFun judgments; terminal model-output errors are counted separately in the metrics CSV.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4f5b5f",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90), w_pad=2.2)
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(output_dir / f"training_trajectory_accuracy_privacy_arrows.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    if not rows:
        raise SystemExit("No metric rows collected")
    write_outputs(rows, DEFAULT_OUTPUT_DIR)
    plot(rows, DEFAULT_OUTPUT_DIR)
    print(f"wrote {DEFAULT_OUTPUT_DIR}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
