#!/usr/bin/env python3
"""Build a slim step-100 paper viewer for Privacy HopQA q30 results."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_q30_paper_viewer as base


DEFAULT_OUTPUT_DIR = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_paper_viewer_s100")
DEFAULT_LATEST_OUTPUT = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/latest_privacy_hopqa_q30_paper_viewer_s100.html")
PADR_S100_NORMAL_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_missing_ckpts_16kgen_32kctx_helperfe_20260524"
)
PADR_S100_PROMPT_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_final_eval_privreward_tau05_s100_privacy_prompt_16kgen_32kctx_helperfe_20260524"
)

BASE_LABEL = "qwen3_4b_instruct_2507"
TASK_S50_LABEL = "qwen3_4b_situational_reward_s50"
TASK_S100_LABEL = "qwen3_4b_situational_reward_s100"
PADR_S50_LABEL = "qwen3_4b_privreward_tau05_s50"
PADR_S100_LABEL = "qwen3_4b_privreward_tau05_s100"

SUMMARY_METRICS = {
    "strict_chain": "strict_chain_success_rate",
    "per_hop_accuracy": "mean_hop_accuracy",
    "privacy_leakage": "privacy_leakage_rate",
    "strict_privacy": "strict_privacy_rate",
    "intent": "intent4_rate",
    "answer": "answer_rate",
    "full": "full_info_rate",
    "lenient": "lenient_full_info_rate",
}
BEHAVIOR_METRICS = ["web_queries", "web_reads", "local_searches", "local_reads"]


def collect_metric_row(root: Path, model_label: str, condition: str, display: str) -> dict[str, Any]:
    rows = base.final_situational_metric_rows(root, condition, model_labels=[model_label])
    row = rows[0] if rows else {"model_label": model_label, "condition": condition, "status": "missing"}
    row = dict(row)
    row["model"] = display
    return row


def collect_rows() -> dict[str, dict[str, Any]]:
    return {
        "base_normal": collect_metric_row(
            base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT,
            BASE_LABEL,
            "Normal Prompt",
            "Qwen3 4B",
        ),
        "base_prompt": collect_metric_row(
            base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            BASE_LABEL,
            "Privacy Prompt",
            "Qwen3 4B + privacy prompt",
        ),
        "task_s50_normal": collect_metric_row(
            base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT,
            TASK_S50_LABEL,
            "Normal Prompt",
            "Task step50",
        ),
        "task_s50_prompt": collect_metric_row(
            base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            TASK_S50_LABEL,
            "Privacy Prompt",
            "Task step50 + privacy prompt",
        ),
        "task_s100_normal": collect_metric_row(
            base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT,
            TASK_S100_LABEL,
            "Normal Prompt",
            "Task step100",
        ),
        "task_s100_prompt": collect_metric_row(
            base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            TASK_S100_LABEL,
            "Privacy Prompt",
            "Task step100 + privacy prompt",
        ),
        "padr_s50_normal": collect_metric_row(
            base.DEFAULT_PRIVACY_AWARE_FINAL_EVAL_ARTIFACT_ROOT,
            PADR_S50_LABEL,
            "Normal Prompt",
            "PA-DR step50",
        ),
        "padr_s50_prompt": collect_metric_row(
            base.DEFAULT_PRIVACY_AWARE_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT,
            PADR_S50_LABEL,
            "Privacy Prompt",
            "PA-DR step50 + privacy prompt",
        ),
        "padr_s100_normal": collect_metric_row(
            PADR_S100_NORMAL_ROOT,
            PADR_S100_LABEL,
            "Normal Prompt",
            "PA-DR step100",
        ),
        "padr_s100_prompt": collect_metric_row(
            PADR_S100_PROMPT_ROOT,
            PADR_S100_LABEL,
            "Privacy Prompt",
            "PA-DR step100 + privacy prompt",
        ),
    }


def row_with_raw_metrics(label: str, row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "condition": label,
        "attempts": f"{row.get('attempted_unique', 0)}/{row.get('expected', 1032)}",
        "judged": f"{row.get('privacy_unique', 0)}/{row.get('privacy_expected', 0)}",
        "status": row.get("status", "missing"),
    }
    for output_key, source_key in SUMMARY_METRICS.items():
        out[output_key] = base.pct_pm_text(row, source_key)
        out[f"{output_key}_value"] = base.live.as_float(row.get(source_key))
        out[f"{output_key}_sem"] = base.live.as_float(row.get(f"{source_key}_sem"))
    return out


def main_summary_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Qwen3 4B", rows["base_normal"]),
        ("Qwen3 4B + privacy prompt", rows["base_prompt"]),
        ("Task Training", rows["task_s100_normal"]),
        ("Task Training + privacy prompt", rows["task_s100_prompt"]),
        ("Task+PA-DR Training", rows["padr_s100_normal"]),
        ("Task+PA-DR Training + privacy prompt", rows["padr_s100_prompt"]),
    ]
    return [row_with_raw_metrics(label, row) for label, row in specs]


def checkpoint_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Task step50", rows["task_s50_normal"]),
        ("Task step100", rows["task_s100_normal"]),
        ("Task step50 + privacy prompt", rows["task_s50_prompt"]),
        ("Task step100 + privacy prompt", rows["task_s100_prompt"]),
        ("PA-DR step50", rows["padr_s50_normal"]),
        ("PA-DR step100", rows["padr_s100_normal"]),
        ("PA-DR step50 + privacy prompt", rows["padr_s50_prompt"]),
        ("PA-DR step100 + privacy prompt", rows["padr_s100_prompt"]),
    ]
    return [row_with_raw_metrics(label, row) for label, row in specs]


def status_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Base", rows["base_normal"]),
        ("Base + privacy prompt", rows["base_prompt"]),
        ("Task step100", rows["task_s100_normal"]),
        ("Task step100 + privacy prompt", rows["task_s100_prompt"]),
        ("PA-DR step100", rows["padr_s100_normal"]),
        ("PA-DR step100 + privacy prompt", rows["padr_s100_prompt"]),
    ]
    out: list[dict[str, Any]] = []
    for label, row in specs:
        privacy_unique = int(row.get("privacy_unique") or 0)
        privacy_expected = int(row.get("privacy_expected") or 0)
        out.append(
            {
                "condition": label,
                "attempts": f"{row.get('attempted_unique', 0)}/{row.get('expected', 1032)}",
                "successful_rollouts": row.get("ok_rollout_unique", 0),
                "privacy_judgments": f"{privacy_unique}/{privacy_expected}",
                "missing_privacy_judgments": max(0, privacy_expected - privacy_unique),
                "status": row.get("status", "missing"),
            }
        )
    return out


def behavior_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("Base", "Qwen3 4B", "Normal", rows["base_normal"]),
        ("Base", "Qwen3 4B + privacy prompt", "Privacy Prompt", rows["base_prompt"]),
        ("Task", "Task", "Normal", rows["task_s100_normal"]),
        ("Task", "Task + privacy prompt", "Privacy Prompt", rows["task_s100_prompt"]),
        ("PA-DR", "PA-DR", "Normal", rows["padr_s100_normal"]),
        ("PA-DR", "PA-DR + privacy prompt", "Privacy Prompt", rows["padr_s100_prompt"]),
    ]
    out: list[dict[str, Any]] = []
    for training_state, condition, prompt_state, row in specs:
        item = {
            "training_state": training_state,
            "condition": condition,
            "prompt_state": prompt_state,
            "successful_rollouts": row.get("ok_rollout_unique"),
        }
        for key in BEHAVIOR_METRICS:
            item[key] = row.get(key)
            item[f"{key}_sem"] = row.get(f"{key}_sem")
        out.append(item)
    return out


def latex_training_summary(rows: list[dict[str, Any]]) -> str:
    metric_specs = [
        ("strict_chain", "max"),
        ("per_hop_accuracy", "max"),
        ("privacy_leakage", "min"),
        ("intent", "min"),
        ("answer", "min"),
        ("full", "min"),
    ]
    best: dict[str, float] = {}
    for key, direction in metric_specs:
        values = [value for row in rows if (value := base.live.as_float(row.get(f"{key}_value"))) is not None]
        if values:
            best[key] = max(values) if direction == "max" else min(values)
    labels = [
        r"\qwenthreeb",
        r"\quad + privacy prompt",
        "Task Training",
        r"\quad + privacy prompt",
        "Task+PA-DR Training",
        r"\quad + privacy prompt",
    ]
    body: list[str] = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        if index in {2, 4}:
            body.append(r"\midrule")
        cells = [label]
        for key, _direction in metric_specs:
            value = base.live.as_float(row.get(f"{key}_value"))
            is_best = value is not None and key in best and math.isclose(value, best[key], abs_tol=1e-12)
            cells.append(base.latex_pct_pm_cell(row, key, bold=is_best))
        body.append(" & ".join(cells) + r" \\")
    body_text = "\n".join(body)
    return rf"""\begin{{table*}}[t]
\small
\centering
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
\textbf{{Model}} & \textbf{{\shortstack{{Strict Chain\\Success}}}} & \textbf{{\shortstack{{Per-Hop\\Accuracy}}}} & \textbf{{\shortstack{{Privacy\\Leakage}}}} & \textbf{{\shortstack{{Intent\\Leakage}}}} & \textbf{{\shortstack{{Answer\\Leakage}}}} & \textbf{{\shortstack{{Full-Information\\Leakage}}}} \\
\midrule
{body_text}
\bottomrule
\end{{tabular}}
\caption{{Effect of RL training on \qwenthreeb, evaluated with and without the privacy-aware prompt. This step-100 snapshot uses checkpoint 100 for both task-performance training and PA-DR. Values after $\pm$ are SEMs in percentage points; PA-DR step100 is near-final with a small number of pending StepFun privacy judgments in this snapshot.}}
\label{{tab:training_effect_summary_s100}}
\end{{table*}}
"""


def latex_behavior_summary(rows: list[dict[str, Any]]) -> str:
    labels = [
        r"\qwenthreeb",
        r"\quad + privacy prompt",
        "Task",
        r"\quad + privacy prompt",
        "PA-DR",
        r"\quad + privacy prompt",
    ]
    body: list[str] = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        if index in {2, 4}:
            body.append(r"\midrule")
        cells = [label, *(base.latex_num_pm_cell(row, key) for key in BEHAVIOR_METRICS)]
        body.append(" & ".join(cells) + r" \\")
    body_text = "\n".join(body)
    return rf"""\begin{{table*}}[t]
\small
\centering
\begin{{tabular}}{{lllll}}
\toprule
\textbf{{Model}} & \textbf{{Mean Web Queries}} & \textbf{{Mean Web Reads}} & \textbf{{Mean Local Searches}} & \textbf{{Mean Local Reads}} \\
\midrule
{body_text}
\bottomrule
\end{{tabular}}
\caption{{Effect of step-100 training on query and retrieval behavior for \qwenthreeb. Values are means per successful rollout; values after $\pm$ are SEMs over repeat means.}}
\label{{tab:final_training_behavior_s100}}
\end{{table*}}
"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_training_effect(rows: dict[str, dict[str, Any]], path: Path) -> None:
    specs = [
        ("base_normal", "Base", "#1f2421", "o", True, (-8, 10)),
        ("base_prompt", "Base + prompt", "#1f2421", "D", False, (8, -14)),
        ("task_s100_normal", "Task", "#D55E00", "s", True, (8, 8)),
        ("task_s100_prompt", "Task + prompt", "#D55E00", "D", False, (-8, -14)),
        ("padr_s100_normal", "PA-DR", "#009E73", "s", True, (8, 8)),
        ("padr_s100_prompt", "PA-DR + prompt", "#009E73", "D", False, (8, -16)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.45))
    points: dict[str, tuple[float, float]] = {}
    for key, label, color, marker, filled, label_offset in specs:
        row = rows[key]
        x = base.live.as_float(row.get("strict_chain_success_rate"))
        y = base.live.as_float(row.get("privacy_leakage_rate"))
        if x is None or y is None:
            continue
        xy = (100 * x, 100 * y)
        points[key] = xy
        alpha = 1.0 if row.get("complete") else 0.62
        ax.errorbar(
            xy[0],
            xy[1],
            xerr=100 * (base.live.as_float(row.get("strict_chain_success_rate_sem")) or 0.0),
            yerr=100 * (base.live.as_float(row.get("privacy_leakage_rate_sem")) or 0.0),
            fmt=marker,
            markersize=5.8,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=1.25,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.8,
            alpha=alpha,
            zorder=4,
        )
        point_label = f"{label}*" if not row.get("complete") else label
        ax.annotate(
            point_label,
            xy=xy,
            xytext=label_offset,
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            ha="left" if label_offset[0] >= 0 else "right",
            va="bottom" if label_offset[1] >= 0 else "top",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.84},
            zorder=5,
        )
    arrows = [
        ("base_normal", "task_s100_normal", "#D55E00", "-"),
        ("base_normal", "padr_s100_normal", "#009E73", "-"),
        ("base_normal", "base_prompt", "#1f2421", "--"),
        ("task_s100_normal", "task_s100_prompt", "#D55E00", "--"),
        ("padr_s100_normal", "padr_s100_prompt", "#009E73", "--"),
    ]
    for start_key, end_key, color, linestyle in arrows:
        if start_key not in points or end_key not in points:
            continue
        ax.annotate(
            "",
            xy=points[end_key],
            xytext=points[start_key],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.35 if linestyle == "--" else 1.65,
                "linestyle": linestyle,
                "alpha": 0.9,
                "shrinkA": 6,
                "shrinkB": 6,
            },
            zorder=2,
        )
    ax.set_xlabel("Strict Chain Success (%)")
    ax.set_ylabel("Privacy Leakage (%)")
    ax.set_xlim(40, 65)
    ax.set_ylim(0, 60)
    ax.set_xticks([40, 45, 50, 55, 60, 65])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    base.add_minor_grid(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f2421", markeredgecolor="#1f2421", markersize=6, label="Base"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#D55E00", markeredgecolor="#D55E00", markersize=6, label="Task"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#009E73", markeredgecolor="#009E73", markersize=6, label="PA-DR"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#5F6F7A", markersize=6, label="+ privacy prompt"),
        Line2D([0], [0], color="#5F6F7A", lw=1.6, label="training effect"),
        Line2D([0], [0], color="#5F6F7A", lw=1.25, linestyle="--", label="prompt effect"),
    ]
    ax.legend(handles=handles, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    ax.set_title("Effect of Step-100 Training")
    base.save_figure(fig, path)


def plot_core_leakage_types(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = [
        ("Intent Leakage", "intent"),
        ("Answer Leakage", "answer"),
        ("Full-Information Leakage", "full"),
    ]
    labels = ["Base", "Base + prompt", "Task", "Task + prompt", "PA-DR", "PA-DR + prompt"]
    colors = ["#1f2421", "#7a7f7a", "#D55E00", "#E69F00", "#009E73", "#56B4E9"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = list(range(len(metrics)))
    width = 0.12
    offsets = [(index - (len(rows) - 1) / 2) * width for index in range(len(rows))]
    for row, label, color, offset in zip(rows, labels, colors, offsets):
        values = [100 * (base.live.as_float(row.get(f"{key}_value")) or 0.0) for _name, key in metrics]
        yerr = [100 * (base.live.as_float(row.get(f"{key}_sem")) or 0.0) for _name, key in metrics]
        ax.bar([item + offset for item in x], values, width=width, color=color, label=label, yerr=yerr, error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _key in metrics])
    ax.set_ylabel("% Occurrence")
    ax.set_ylim(0, 75)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(NullLocator())
    base.add_minor_grid(ax, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=3, frameon=False)
    ax.set_title("Core Leakage Types")
    base.save_figure(fig, path)


def plot_behavior(rows: list[dict[str, Any]], path: Path) -> None:
    states = ["Base", "Task", "PA-DR"]
    row_map = {(row.get("training_state"), row.get("prompt_state")): row for row in rows}
    normal_rows = [row_map.get((state, "Normal")) or {} for state in states]
    prompt_rows = [row_map.get((state, "Privacy Prompt")) or {} for state in states]
    metrics = [
        ("Mean Web Queries", "web_queries", "web"),
        ("Mean Web Reads", "web_reads", "web"),
        ("Mean Local Searches", "local_searches", "local"),
        ("Mean Local Reads", "local_reads", "local"),
    ]
    group_max: dict[str, float] = {}
    for _title, key, group in metrics:
        values = [
            value
            for row in [*normal_rows, *prompt_rows]
            if (value := base.live.as_float(row.get(key))) is not None
        ]
        group_max[group] = max(group_max.get(group, 1.0), *values, 1.0) * 1.16
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8))
    x = list(range(len(states)))
    width = 0.34
    for ax, (title, key, group) in zip(axes.flat, metrics):
        normal_vals = [base.live.as_float(row.get(key)) or math.nan for row in normal_rows]
        prompt_vals = [base.live.as_float(row.get(key)) or math.nan for row in prompt_rows]
        ax.bar(
            [item - width / 2 for item in x],
            normal_vals,
            width=width,
            color=base.BASELINE_COLOR,
            label="Normal Prompt",
            yerr=base.count_yerr(normal_rows, key),
            error_kw={"elinewidth": 1.0, "ecolor": "#303633", "capsize": 2.6, "capthick": 0.9},
        )
        ax.bar(
            [item + width / 2 for item in x],
            prompt_vals,
            width=width,
            color=base.PROMPT_COLOR,
            label="Privacy-Aware Prompt",
            yerr=base.count_yerr(prompt_rows, key),
            error_kw={"elinewidth": 1.0, "ecolor": "#303633", "capsize": 2.6, "capthick": 0.9},
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(states)
        ax.xaxis.set_minor_locator(NullLocator())
        base.set_nice_count_y_axis(ax, group_max[group], target_ticks=5)
    axes[0, 0].set_ylabel("Mean Count per Rollout")
    axes[1, 0].set_ylabel("Mean Count per Rollout")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.supxlabel("Training State", y=0.02)
    base.save_figure(fig, path)


def table_rows_for_html(rows: list[dict[str, Any]], include_status: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "condition": row["condition"],
            "strict_chain": row["strict_chain"],
            "per_hop_accuracy": row["per_hop_accuracy"],
            "privacy_leakage": row["privacy_leakage"],
            "intent": row["intent"],
            "answer": row["answer"],
            "full": row["full"],
        }
        if include_status:
            item["judged"] = row["judged"]
            item["status"] = row["status"]
        out.append(item)
    return out


def behavior_rows_for_html(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition": row["condition"],
            "web_queries": base.num_pm_text(row, "web_queries"),
            "web_reads": base.num_pm_text(row, "web_reads"),
            "local_searches": base.num_pm_text(row, "local_searches"),
            "local_reads": base.num_pm_text(row, "local_reads"),
        }
        for row in rows
    ]


def build_viewer(output_dir: Path, latest_outputs: list[Path]) -> None:
    base.paper_style()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    summary = main_summary_rows(rows)
    checkpoints = checkpoint_rows(rows)
    behavior = behavior_rows(rows)
    status = status_rows(rows)

    plot_training_effect(rows, plots_dir / "training_effect_step100.svg")
    plot_core_leakage_types(summary, plots_dir / "core_leakage_types_step100.svg")
    plot_behavior(behavior, plots_dir / "training_behavior_step100.svg")

    write_text(tables_dir / "training_effect_summary_s100.tex", latex_training_summary(summary))
    write_text(tables_dir / "training_behavior_s100.tex", latex_behavior_summary(behavior))
    write_text(
        tables_dir / "checkpoint_choice_s50_s100.tex",
        base.latex_table(
            [
                ("condition", "Condition"),
                ("strict_chain", "Strict Chain"),
                ("per_hop_accuracy", "Per-Hop Accuracy"),
                ("privacy_leakage", "Privacy Leakage"),
                ("intent", "Intent Leakage"),
                ("answer", "Answer Leakage"),
                ("full", "Full-Information Leakage"),
                ("judged", "Judged"),
                ("status", "Status"),
            ],
            table_rows_for_html(checkpoints, include_status=True),
            "Checkpoint choice diagnostic comparing step50 and step100. Values after +/- are SEMs in percentage points.",
            "tab:checkpoint_choice_s50_s100",
        ),
    )
    write_text(
        output_dir / "viewer_data.json",
        json.dumps(
            {
                "status_rows": status,
                "summary_rows": summary,
                "checkpoint_rows": checkpoints,
                "behavior_rows": behavior,
                "artifact_roots": {
                    "standard_normal": str(base.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT),
                    "standard_privacy_prompt": str(base.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT),
                    "padr_s50_normal": str(base.DEFAULT_PRIVACY_AWARE_FINAL_EVAL_ARTIFACT_ROOT),
                    "padr_s50_privacy_prompt": str(base.DEFAULT_PRIVACY_AWARE_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT),
                    "padr_s100_normal": str(PADR_S100_NORMAL_ROOT),
                    "padr_s100_privacy_prompt": str(PADR_S100_PROMPT_ROOT),
                },
            },
            indent=2,
        )
        + "\n",
    )

    status_table = base.table(
        [
            ("condition", "Condition"),
            ("attempts", "Attempts"),
            ("successful_rollouts", "Successful Rollouts"),
            ("privacy_judgments", "Privacy Judgments"),
            ("missing_privacy_judgments", "Missing Privacy Judgments"),
            ("status", "Status"),
        ],
        status,
    )
    summary_table = base.table(
        [
            ("condition", "Condition"),
            ("strict_chain", "Strict Chain"),
            ("per_hop_accuracy", "Per-Hop Accuracy"),
            ("privacy_leakage", "Privacy Leakage"),
            ("intent", "Intent Leakage"),
            ("answer", "Answer Leakage"),
            ("full", "Full-Information Leakage"),
        ],
        table_rows_for_html(summary),
    )
    checkpoint_table = base.table(
        [
            ("condition", "Condition"),
            ("strict_chain", "Strict Chain"),
            ("per_hop_accuracy", "Per-Hop Accuracy"),
            ("privacy_leakage", "Privacy Leakage"),
            ("intent", "Intent Leakage"),
            ("answer", "Answer Leakage"),
            ("full", "Full-Information Leakage"),
            ("judged", "Judged"),
            ("status", "Status"),
        ],
        table_rows_for_html(checkpoints, include_status=True),
    )
    behavior_table = base.table(
        [
            ("condition", "Condition"),
            ("web_queries", "Mean Web Queries"),
            ("web_reads", "Mean Web Reads"),
            ("local_searches", "Mean Local Searches"),
            ("local_reads", "Mean Local Reads"),
        ],
        behavior_rows_for_html(behavior),
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy HopQA Q30 Step100 Paper Viewer</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202522; background: #f7f8f6; }}
    header {{ padding: 28px 34px 16px; border-bottom: 1px solid #d8ddd7; background: #ffffff; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 22px 52px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.15; }}
    h2 {{ margin: 0 0 10px; font-size: 22px; }}
    p, .note {{ color: #52605a; line-height: 1.45; }}
    section {{ margin: 0 0 34px; padding: 22px; background: #ffffff; border: 1px solid #dfe4df; border-radius: 8px; }}
    img {{ width: 100%; max-width: 920px; display: block; margin: 14px auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #dfe4df; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f3f0; font-weight: 650; }}
    .links a {{ display: inline-block; margin: 0 12px 8px 0; color: #0b63a3; text-decoration: none; }}
    .links a:hover {{ text-decoration: underline; }}
    .warning {{ color: #755100; background: #fff7dd; border: 1px solid #ead392; padding: 10px 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Privacy HopQA Q30 Step100 Paper Viewer</h1>
    <div class="note">Slim paper-facing snapshot. The default trained checkpoints are Task step100 and PA-DR step100. PA-DR step100 is near-final: normal prompt is missing 5 StepFun privacy judgments and privacy prompt is missing 1.</div>
  </header>
  <main>
    <section id="status">
      <h2>Status</h2>
      <div class="warning">Use the PA-DR step100 rows as near-final until the missing StepFun judgments are resolved or intentionally counted as failures. The maximum denominator effect is about 0.5 percentage points for normal PA-DR step100 and 0.1 percentage points for prompted PA-DR step100.</div>
      {status_table}
    </section>
    <section id="training-effect">
      <h2>Effect Of Training: Accuracy And Privacy</h2>
      <div class="note">Step100 default comparison on held-out test rollouts. Error bars are SEM over repeat means. Asterisks mark near-final rows with missing StepFun privacy judgments.</div>
      <img src="plots/training_effect_step100.svg" alt="Step100 training effect on strict chain success and privacy leakage">
      <div class="links"><a href="plots/training_effect_step100.svg">SVG</a><a href="plots/training_effect_step100.pdf">PDF</a><a href="tables/training_effect_summary_s100.tex">LaTeX table</a></div>
      {summary_table}
    </section>
    <section id="core-leakage-types">
      <h2>Core Leakage Types</h2>
      <div class="note">Intent, answer, and Full-Information Leakage for the same step100 default conditions.</div>
      <img src="plots/core_leakage_types_step100.svg" alt="Step100 core leakage types">
      <div class="links"><a href="plots/core_leakage_types_step100.svg">SVG</a><a href="plots/core_leakage_types_step100.pdf">PDF</a></div>
    </section>
    <section id="training-behavior">
      <h2>Effect Of Training: Query And Retrieval Behavior</h2>
      <div class="note">Means per successful rollout for the same step100 default conditions.</div>
      <img src="plots/training_behavior_step100.svg" alt="Step100 training effect on query and retrieval behavior">
      <div class="links"><a href="plots/training_behavior_step100.svg">SVG</a><a href="plots/training_behavior_step100.pdf">PDF</a><a href="tables/training_behavior_s100.tex">LaTeX table</a></div>
      {behavior_table}
    </section>
    <section id="checkpoint-choice">
      <h2>Checkpoint Choice: Step50 Vs Step100</h2>
      <div class="note">Compact diagnostic table for selecting the default checkpoint. Step100 is better for PA-DR privacy leakage while preserving roughly comparable task performance.</div>
      <div class="links"><a href="tables/checkpoint_choice_s50_s100.tex">LaTeX table</a><a href="viewer_data.json">JSON data</a></div>
      {checkpoint_table}
    </section>
  </main>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    write_text(index_path, html)
    for latest in latest_outputs:
        latest.parent.mkdir(parents=True, exist_ok=True)
        relative_output = Path(os.path.relpath(output_dir, latest.parent)).as_posix()
        latest_html = html.replace(
            "  <title>Privacy HopQA Q30 Step100 Paper Viewer</title>",
            f"  <title>Privacy HopQA Q30 Step100 Paper Viewer</title>\n  <base href=\"{base.esc(relative_output)}/\">",
            1,
        )
        write_text(latest, latest_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latest-output", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.latest_output:
        args.latest_output = [DEFAULT_LATEST_OUTPUT]
    return args


def main() -> None:
    args = parse_args()
    build_viewer(args.output_dir, args.latest_output)


if __name__ == "__main__":
    main()
