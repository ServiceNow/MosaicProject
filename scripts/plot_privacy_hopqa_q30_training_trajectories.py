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
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_q30_live_analysis_viewer as live
from scripts import build_privacy_hopqa_q30_paper_viewer as paper
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
DEFAULT_OUTPUT_DIR = ROOT / "runtime_analysis" / "privacy_hopqa_q30_training_trajectory_20260524"

BASE_LABEL = "qwen3_4b_instruct_2507"
NORMAL_STEPS = [20, 40, 60, 80, 100]


SERIES = [
    {
        "family": "task_reward",
        "display": "Task training",
        "prefix": "qwen3_4b_situational_reward",
        "color": "#D55E00",
        "dark_color": "#8A3A00",
        "light_color": "#F0A64A",
        "marker": "s",
    },
    {
        "family": "privacy_aware_tau05",
        "display": "Task+PA-DR training",
        "prefix": "qwen3_4b_privreward_tau05",
        "color": "#009E73",
        "dark_color": "#005C43",
        "light_color": "#62BFA5",
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


def metric_row(condition: str, series: dict[str, str], step: int, root: Path) -> dict[str, Any] | None:
    label = model_label_for(series, step)
    rollouts, leakage, terminal_errors = read_model_rows(root, label)
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


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series in SERIES:
        for step in [0, *NORMAL_STEPS]:
            root = normal_root_for(model_label_for(series, step))
            row = metric_row("normal_prompt", series, step, root)
            if row:
                rows.append(row)
    return rows


def pct(value: float | None) -> float:
    return 100.0 * float(value or 0.0)


def interpolate_color(start_hex: str, end_hex: str, amount: float) -> tuple[float, float, float]:
    amount = min(1.0, max(0.0, amount))
    start = to_rgb(start_hex)
    end = to_rgb(end_hex)
    return tuple(start[channel] + (end[channel] - start[channel]) * amount for channel in range(3))


def step_color(series: dict[str, str], step: int) -> tuple[float, float, float]:
    return interpolate_color(series["dark_color"], series["light_color"], step / 100.0)


def add_arrow(ax: Any, series: dict[str, str], start: dict[str, Any], end: dict[str, Any]) -> None:
    x0, y0 = pct(start["strict_chain_success"]), pct(start["privacy_leakage_rate"])
    x1, y1 = pct(end["strict_chain_success"]), pct(end["privacy_leakage_rate"])
    if abs(x1 - x0) < 0.05 and abs(y1 - y0) < 0.05:
        return
    color = step_color(series, int(end["step"]))
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.35,
        color=color,
        alpha=0.78,
        shrinkA=7,
        shrinkB=7,
    )
    ax.add_patch(arrow)


def plot_panel(ax: Any, rows: list[dict[str, Any]]) -> None:
    base_row = next((row for row in rows if int(row["step"]) == 0), None)
    if base_row:
        base_xy = (pct(base_row["strict_chain_success"]), pct(base_row["privacy_leakage_rate"]))
        err = ax.errorbar(
            base_xy[0],
            base_xy[1],
            xerr=pct(base_row.get("strict_chain_success_sem")),
            yerr=pct(base_row.get("privacy_leakage_sem")),
            fmt="o",
            markersize=6.2,
            markerfacecolor="#1f2421",
            markeredgecolor="#1f2421",
            markeredgewidth=1.35,
            color="#1f2421",
            ecolor="#1f2421",
            elinewidth=1.15,
            capsize=3.0,
            zorder=5,
        )
        for capline in err[1]:
            capline.set_color("#1f2421")
            capline.set_markeredgecolor("#1f2421")
            capline.set_markerfacecolor("#1f2421")
        ax.annotate(
            "Base",
            xy=base_xy,
            xytext=(-7, 7),
            textcoords="offset points",
            fontsize=8.5,
            color="#1f2421",
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.90},
            zorder=6,
        )
    for series in SERIES:
        data = sorted(
            [row for row in rows if row["family"] == series["family"] and int(row["step"]) > 0],
            key=lambda row: row["step"],
        )
        if not data:
            continue
        xs = [pct(row["strict_chain_success"]) for row in data]
        ys = [pct(row["privacy_leakage_rate"]) for row in data]
        xerr = [pct(row.get("strict_chain_success_sem")) for row in data]
        yerr = [pct(row.get("privacy_leakage_sem")) for row in data]
        colors = [step_color(series, int(row["step"])) for row in data]
        for x, y, xerr_value, yerr_value, color in zip(xs, ys, xerr, yerr, colors):
            err = ax.errorbar(
                [x],
                [y],
                xerr=[[xerr_value], [xerr_value]],
                yerr=[[yerr_value], [yerr_value]],
                fmt="none",
                ecolor=color,
                elinewidth=1.05,
                capsize=2.8,
                alpha=0.58,
            )
            for capline in err[1]:
                capline.set_color(color)
                capline.set_markeredgecolor(color)
                capline.set_markerfacecolor(color)
        ax.scatter(
            xs,
            ys,
            c=colors,
            marker=series["marker"],
            s=40,
            edgecolors=series["color"],
            linewidths=1.15,
            alpha=0.96,
            zorder=4,
        )
        chain = ([base_row] if base_row else []) + data
        for start, end in zip(chain, chain[1:]):
            add_arrow(ax, series, start, end)
        final_xy = (xs[-1], ys[-1])
        point_label = "Task" if series["family"] == "task_reward" else "PA-DR"
        ax.annotate(
            point_label,
            xy=final_xy,
            xytext=(6, 1),
            textcoords="offset points",
            fontsize=8.5,
            color=series["color"],
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.90},
            zorder=6,
        )
    ax.set_xlabel("Strict Chain Success (%)")
    ax.set_ylabel("Privacy Leakage (%)")
    ax.set_xlim(40, 65)
    ax.set_ylim(0, 60)
    ax.set_xticks([40, 45, 50, 55, 60, 65])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    paper.add_minor_grid(ax)


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda row: (row["condition"], row["family"], row["step"]))
    (output_dir / "training_trajectory_metrics.json").write_text(
        json.dumps(rows_sorted, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "training_trajectory_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows_sorted[0].keys()) if rows_sorted else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_sorted)


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    paper.paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.5))
    plot_panel(ax, rows)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f2421",
            markeredgecolor="#1f2421",
            label="Base Qwen3 4B",
            markersize=6.2,
        ),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=series["marker"],
            color="none",
            markerfacecolor=step_color(series, 80),
            markeredgecolor=series["color"],
            label=series["display"],
            markersize=6.2,
        )
        for series in SERIES
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="#5F6F7A",
            markerfacecolor="white",
            markeredgecolor="#5F6F7A",
            label="gradient: step 20 -> 100",
            markersize=6.2,
            linewidth=1.25,
        )
    )
    ax.legend(handles=handles, loc="lower center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(output_dir / f"training_trajectory_accuracy_privacy_arrows.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    svg_path = output_dir / "training_trajectory_accuracy_privacy_arrows.svg"
    text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


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
