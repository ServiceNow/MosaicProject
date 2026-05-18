#!/usr/bin/env python3
"""Build a compact paper-preview viewer for Privacy HopQA q30 results."""

from __future__ import annotations

import argparse
import html
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_q30_live_analysis_viewer as live


DEFAULT_OUTPUT_DIR = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_paper_viewer")
BASELINE_COLOR = "#5F6F7A"
PROMPT_COLOR = "#0072B2"
HOP_COLOR = "#0072B2"
CHAIN_COLOR = "#009E73"
SEARCH_LOCAL_COLOR = "#0072B2"
SEARCH_WEB_COLOR = "#D55E00"
READ_LOCAL_COLOR = "#009E73"
READ_WEB_COLOR = "#CC79A7"
BASE_MODEL_LABEL = "qwen3_4b_instruct_2507"
TRAINED_MODEL_LABELS = ["rrf_4b_hopacc_s102", "rrf_4b_prefix_step_s109"]
TRAINING_FAMILY_LABELS = [BASE_MODEL_LABEL, *TRAINED_MODEL_LABELS]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pct_text(value: Any, digits: int = 1) -> str:
    number = live.as_float(value)
    if number is None:
        return "n/a"
    return f"{100 * number:.{digits}f}%"


def num_text(value: Any, digits: int = 2) -> str:
    number = live.as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def pp_text(value: Any, digits: int = 1) -> str:
    number = live.as_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.{digits}f}"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def percent_yerr(rows: list[dict[str, Any]], key: str) -> list[list[float]]:
    return live.ci_yerr(rows, key, 100)


def count_yerr(rows: list[dict[str, Any]], key: str) -> list[list[float]]:
    return live.ci_yerr(rows, key, 1)


def annotate_ns(ax: Any, positions: list[float], values: list[float], counts: list[int], percent: bool, y_max: float) -> None:
    for x_pos, value, count in zip(positions, values, counts):
        if not math.isfinite(value):
            ax.text(x_pos, y_max * 0.04, f"n={count}", ha="center", va="bottom", fontsize=8, color="#5d6964")
            continue
        if percent:
            label = f"{value:.0f}%\nn={count}"
        else:
            label = f"{value:.1f}\nn={count}"
        ax.text(x_pos, min(y_max * 0.96, value + y_max * 0.025), label, ha="center", va="bottom", fontsize=8)


def model_rows_for_plot(rows: list[dict[str, Any]], require_prompt: bool = False) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("baseline_ok") or row.get("prompt_ok")]
    if require_prompt:
        selected = [row for row in selected if row.get("baseline_ok") and row.get("prompt_ok")]
    return selected


def is_trained_checkpoint(model_label: Any) -> bool:
    return str(model_label) in set(TRAINED_MODEL_LABELS)


def exclude_trained_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not is_trained_checkpoint(row.get("model_label"))]


def training_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(TRAINING_FAMILY_LABELS)
    return [row for row in rows if str(row.get("model_label")) in wanted]


def plot_privacy_prompt_accuracy_leakage(rows: list[dict[str, Any]], path: Path) -> None:
    rows = model_rows_for_plot(rows, require_prompt=True)
    if not rows:
        return
    labels = [live.model_display(str(row["model_label"])) for row in rows]
    metrics = [
        (
            "Strict Chain Success (%)",
            "baseline_strict_chain_success_rate",
            "prompt_strict_chain_success_rate",
            "baseline_attempts",
            "prompt_attempts",
        ),
        (
            "Privacy Leakage (%)",
            "baseline_reward_rate",
            "prompt_reward_rate",
            "baseline_stepfun",
            "prompt_stepfun",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.85), sharey=True)
    x = list(range(len(rows)))
    width = 0.36
    error_kw = {"elinewidth": 1.2, "ecolor": "#303633", "capsize": 3.5, "capthick": 1.1}
    for ax, (ylabel, baseline_key, prompt_key, baseline_n_key, prompt_n_key) in zip(axes, metrics):
        baseline_counts = [int(row.get(baseline_n_key) or 0) for row in rows]
        prompt_counts = [int(row.get(prompt_n_key) or 0) for row in rows]
        baseline_vals = [
            100 * float(row.get(baseline_key))
            if count and row.get(baseline_key) is not None
            else math.nan
            for row, count in zip(rows, baseline_counts)
        ]
        prompt_vals = [
            100 * float(row.get(prompt_key))
            if count and row.get(prompt_key) is not None
            else math.nan
            for row, count in zip(rows, prompt_counts)
        ]
        baseline_pos = [item - width / 2 for item in x]
        prompt_pos = [item + width / 2 for item in x]
        ax.bar(
            baseline_pos,
            baseline_vals,
            width=width,
            color=BASELINE_COLOR,
            label="Baseline",
            yerr=percent_yerr(rows, baseline_key),
            error_kw=error_kw,
        )
        ax.bar(
            prompt_pos,
            prompt_vals,
            width=width,
            color=PROMPT_COLOR,
            label="Privacy Prompt",
            yerr=percent_yerr(rows, prompt_key),
            error_kw=error_kw,
        )
        annotate_ns(ax, baseline_pos, baseline_vals, baseline_counts, percent=True, y_max=124)
        annotate_ns(ax, prompt_pos, prompt_vals, prompt_counts, percent=True, y_max=124)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 124)
        ax.set_yticks(range(0, 101, 20))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.16))
    save_figure(fig, path)


def plot_privacy_prompt_leakage_types(rows: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if row.get("baseline_stepfun") or row.get("prompt_stepfun")]
    if not rows:
        return
    labels = [live.model_display(str(row["model_label"])) for row in rows]
    metrics = [
        ("Intent Leakage\n(Intent Score >= 4)", "baseline_intent4_rate", "prompt_intent4_rate"),
        ("Answer Leakage", "baseline_answer_rate", "prompt_answer_rate"),
        ("Full-Information Claim", "baseline_full_info_rate", "prompt_full_info_rate"),
        ("Partial Claim", "baseline_lenient_full_info_rate", "prompt_lenient_full_info_rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.65), sharey=True)
    x = list(range(len(rows)))
    width = 0.36
    baseline_counts = [int(row.get("baseline_stepfun") or 0) for row in rows]
    prompt_counts = [int(row.get("prompt_stepfun") or 0) for row in rows]
    baseline_pos = [item - width / 2 for item in x]
    prompt_pos = [item + width / 2 for item in x]
    error_kw = {"elinewidth": 1.1, "ecolor": "#303633", "capsize": 3, "capthick": 1.0}
    for ax, (title, baseline_key, prompt_key) in zip(axes.flat, metrics):
        baseline_vals = [
            100 * float(row.get(baseline_key))
            if row.get(baseline_key) is not None and count
            else math.nan
            for row, count in zip(rows, baseline_counts)
        ]
        prompt_vals = [
            100 * float(row.get(prompt_key))
            if row.get(prompt_key) is not None and count
            else math.nan
            for row, count in zip(rows, prompt_counts)
        ]
        ax.bar(
            baseline_pos,
            baseline_vals,
            width=width,
            color=BASELINE_COLOR,
            label="Baseline",
            yerr=percent_yerr(rows, baseline_key),
            error_kw=error_kw,
        )
        ax.bar(
            prompt_pos,
            prompt_vals,
            width=width,
            color=PROMPT_COLOR,
            label="Privacy Prompt",
            yerr=percent_yerr(rows, prompt_key),
            error_kw=error_kw,
        )
        annotate_ns(ax, baseline_pos, baseline_vals, baseline_counts, percent=True, y_max=124)
        annotate_ns(ax, prompt_pos, prompt_vals, prompt_counts, percent=True, y_max=124)
        ax.set_title(title)
        ax.set_ylabel("StepFun-Judged Rollouts (%)")
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 124)
        ax.set_yticks(range(0, 101, 20))
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    save_figure(fig, path)


def plot_privacy_prompt_query_behavior(rows: list[dict[str, Any]], path: Path) -> None:
    rows = model_rows_for_plot(rows, require_prompt=True)
    if not rows:
        return
    labels = [live.model_display(str(row["model_label"])) for row in rows]
    metrics = [
        ("Visible Web Queries", "baseline_web_queries", "prompt_web_queries"),
        ("Local Document Searches", "baseline_local_searches", "prompt_local_searches"),
        ("Local Document Reads", "baseline_local_doc_reads", "prompt_local_doc_reads"),
        ("Web Document Reads", "baseline_web_doc_reads", "prompt_web_doc_reads"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.9))
    x = list(range(len(rows)))
    width = 0.36
    error_kw = {"elinewidth": 1.1, "ecolor": "#303633", "capsize": 3, "capthick": 1.0}
    for ax, (ylabel, baseline_key, prompt_key) in zip(axes.flat, metrics):
        baseline_vals = [row.get(baseline_key) or 0 for row in rows]
        prompt_vals = [row.get(prompt_key) or 0 for row in rows]
        baseline_counts = [int(row.get("baseline_ok") or 0) for row in rows]
        prompt_counts = [int(row.get("prompt_ok") or 0) for row in rows]
        baseline_pos = [item - width / 2 for item in x]
        prompt_pos = [item + width / 2 for item in x]
        ax.bar(
            baseline_pos,
            baseline_vals,
            width=width,
            color=BASELINE_COLOR,
            label="Baseline",
            yerr=count_yerr(rows, baseline_key),
            error_kw=error_kw,
        )
        ax.bar(
            prompt_pos,
            prompt_vals,
            width=width,
            color=PROMPT_COLOR,
            label="Privacy Prompt",
            yerr=count_yerr(rows, prompt_key),
            error_kw=error_kw,
        )
        y_max = max([*baseline_vals, *prompt_vals, 1.0]) * 1.32
        annotate_ns(ax, baseline_pos, baseline_vals, baseline_counts, percent=False, y_max=y_max)
        annotate_ns(ax, prompt_pos, prompt_vals, prompt_counts, percent=False, y_max=y_max)
        ax.set_ylabel(f"{ylabel}\nMean Count Per Rollout")
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    save_figure(fig, path)


def plot_training_effect(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    rows = {str(row.get("model_label")): row for row in training_family_rows(rows)}
    base = rows.get(BASE_MODEL_LABEL)
    if not base:
        return []
    conditions = [
        ("Baseline Prompt", "baseline_strict_chain_success_rate", "baseline_reward_rate", "baseline_attempts", "baseline_stepfun"),
        ("Privacy Prompt", "prompt_strict_chain_success_rate", "prompt_reward_rate", "prompt_attempts", "prompt_stepfun"),
    ]
    checkpoint_colors = {
        BASE_MODEL_LABEL: "#1f2421",
        "rrf_4b_hopacc_s102": "#D55E00",
        "rrf_4b_prefix_step_s109": "#0072B2",
    }
    effect_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.85), sharey=True)
    all_x: list[float] = []
    all_y: list[float] = []
    for ax, (condition, x_key, y_key, rollout_n_key, judge_n_key) in zip(axes, conditions):
        base_x = live.as_float(base.get(x_key))
        base_y = live.as_float(base.get(y_key))
        if base_x is None or base_y is None:
            continue
        base_xy = (100 * base_x, 100 * base_y)
        for model_label in TRAINING_FAMILY_LABELS:
            row = rows.get(model_label)
            if not row:
                continue
            x_val = live.as_float(row.get(x_key))
            y_val = live.as_float(row.get(y_key))
            if x_val is None or y_val is None:
                continue
            x = 100 * x_val
            y = 100 * y_val
            x_sem = 100 * (live.as_float(row.get(f"{x_key}_sem")) or 0.0)
            y_sem = 100 * (live.as_float(row.get(f"{y_key}_sem")) or 0.0)
            all_x.append(x)
            all_y.append(y)
            color = checkpoint_colors.get(model_label, "#5F6F7A")
            marker = "o" if model_label == BASE_MODEL_LABEL else "s"
            label = live.model_display(model_label)
            ax.errorbar(
                x,
                y,
                xerr=x_sem,
                yerr=y_sem,
                fmt=marker,
                markersize=7.5,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                label=label,
                zorder=3,
            )
            ax.text(x + 0.8, y + 0.8, label.replace("Qwen3 4B ", ""), fontsize=9, color=color)
            if model_label != BASE_MODEL_LABEL:
                ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=base_xy,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4, "shrinkA": 5, "shrinkB": 5},
                    zorder=2,
                )
                effect_rows.append(
                    {
                        "condition": condition,
                        "checkpoint": label,
                        "accuracy_delta_pp": x - base_xy[0],
                        "leakage_delta_pp": y - base_xy[1],
                        "rollouts": int(row.get(rollout_n_key) or 0),
                        "judged": int(row.get(judge_n_key) or 0),
                    }
                )
        ax.set_title(condition)
        ax.set_xlabel("Strict Chain Success (%)")
        ax.grid(True, alpha=0.25)
    y_values = all_y or [0, 100]
    x_values = all_x or [0, 100]
    x_pad = max(4.0, 0.12 * (max(x_values) - min(x_values) or 1.0))
    y_pad = max(4.0, 0.12 * (max(y_values) - min(y_values) or 1.0))
    for ax in axes:
        ax.set_xlim(max(0, min(x_values) - x_pad), min(100, max(x_values) + x_pad))
        ax.set_ylim(max(0, min(y_values) - y_pad), min(100, max(y_values) + y_pad))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_ylabel("Privacy Leakage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.11))
    save_figure(fig, path)
    return effect_rows


def task_performance_summary(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in live.ordered_model_labels(row.get("model_label") for row in rollouts if live.rollout_counts_for_accuracy(row)):
        subset = [row for row in rollouts if row.get("model_label") == model and live.rollout_counts_for_accuracy(row)]
        row = {
            "model_label": model,
            "model": live.model_display(model),
            "n": len(subset),
        }
        live.add_repeat_sem_fields(row, "mean_hop_performance", subset, live.attempt_task_accuracy)
        live.add_repeat_sem_fields(row, "all_chain_correct", subset, live.attempt_all_hops_correct)
        rows.append(row)
    return rows


def plot_task_performance(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = [row["model"] for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    hop_vals = [100 * (row.get("mean_hop_performance") or 0) for row in rows]
    chain_vals = [100 * (row.get("all_chain_correct") or 0) for row in rows]
    ns = [int(row.get("n") or 0) for row in rows]
    fig, ax = plt.subplots(figsize=(10.6, 5.05))
    hop_pos = [item - width / 2 for item in x]
    chain_pos = [item + width / 2 for item in x]
    error_kw = {"elinewidth": 1.2, "ecolor": "#303633", "capsize": 3.5, "capthick": 1.1}
    ax.bar(hop_pos, hop_vals, width=width, color=HOP_COLOR, label="Mean Hop Performance", yerr=percent_yerr(rows, "mean_hop_performance"), error_kw=error_kw)
    ax.bar(chain_pos, chain_vals, width=width, color=CHAIN_COLOR, label="All-Chain Correct", yerr=percent_yerr(rows, "all_chain_correct"), error_kw=error_kw)
    annotate_ns(ax, hop_pos, hop_vals, ns, percent=True, y_max=124)
    annotate_ns(ax, chain_pos, chain_vals, ns, percent=True, y_max=124)
    ax.set_ylabel("Percent of Rollouts")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 124)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.20))
    save_figure(fig, path)


def retrieval_summary(rollouts: list[dict[str, Any]], successful_only: bool) -> list[dict[str, Any]]:
    selected = [row for row in rollouts if row.get("ok")] if successful_only else live.scored_rollouts(rollouts)
    rows: list[dict[str, Any]] = []
    metrics = [
        ("local_document_searches", "local_search_events_total"),
        ("web_searches", "web_search_events_total"),
        ("local_document_reads", "local_doc_reads_total"),
        ("web_document_reads", "web_doc_reads_total"),
    ]
    for model in live.ordered_model_labels(row.get("model_label") for row in selected):
        subset = [row for row in selected if row.get("model_label") == model]
        row = {
            "model_label": model,
            "model": live.model_display(model),
            "n": len(subset),
        }
        for out_key, source_key in metrics:
            live.add_repeat_sem_fields(
                row,
                out_key,
                subset,
                lambda item, metric_key=source_key: int(item.get(metric_key) or 0),
            )
        rows.append(row)
    return rows


def plot_retrieval_behavior(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = [row["model"] for row in rows]
    x = list(range(len(rows)))
    series = [
        ("Local Document Searches", "local_document_searches", SEARCH_LOCAL_COLOR),
        ("Web Searches", "web_searches", SEARCH_WEB_COLOR),
        ("Local Document Reads", "local_document_reads", READ_LOCAL_COLOR),
        ("Web Document Reads", "web_document_reads", READ_WEB_COLOR),
    ]
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    fig, ax = plt.subplots(figsize=(11.4, 5.1))
    y_max = 1.0
    for offset, (label, key, color) in zip(offsets, series):
        vals = [row.get(key) or 0 for row in rows]
        y_max = max(y_max, *vals)
        ax.bar([item + offset for item in x], vals, width=width, color=color, label=label, yerr=count_yerr(rows, key), error_kw={"elinewidth": 1.0, "ecolor": "#303633", "capsize": 2.5})
    y_max *= 1.34
    for xi, row in zip(x, rows):
        ax.text(xi, y_max * 0.95, f"n={int(row.get('n') or 0)}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Count Per Rollout")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.25))
    save_figure(fig, path)


def normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def logistic_regression(X: list[list[float]], y: list[int], clusters: list[Any] | None = None) -> dict[str, Any] | None:
    if len(y) < 8 or len(set(y)) < 2:
        return None
    design = np.asarray(X, dtype=float)
    outcomes = np.asarray(y, dtype=float)
    if design.ndim != 2 or design.shape[0] != outcomes.shape[0]:
        return None
    beta = np.zeros(design.shape[1], dtype=float)
    for _ in range(80):
        eta = np.clip(design @ beta, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(prob * (1.0 - prob), 1e-8, None)
        gradient = design.T @ (outcomes - prob)
        hessian = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    eta = np.clip(design @ beta, -30, 30)
    prob = 1.0 / (1.0 + np.exp(-eta))
    weights = np.clip(prob * (1.0 - prob), 1e-8, None)
    hessian = design.T @ (weights[:, None] * design)
    try:
        hessian_inv = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        hessian_inv = np.linalg.pinv(hessian)
    cov = hessian_inv
    if clusters is not None and len(clusters) == len(y):
        cluster_values = np.asarray([str(item) for item in clusters])
        meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
        for cluster in sorted(set(cluster_values)):
            mask = cluster_values == cluster
            cluster_score = design[mask].T @ (outcomes[mask] - prob[mask])
            meat += np.outer(cluster_score, cluster_score)
        n_clusters = len(set(cluster_values))
        n_obs, n_params = design.shape
        correction = 1.0
        if n_clusters > 1 and n_obs > n_params:
            correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        cov = correction * hessian_inv @ meat @ hessian_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = [normal_two_sided_p(float(item)) if math.isfinite(float(item)) else None for item in z]
    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "p": p_values,
        "n": len(y),
    }


def format_p(value: Any) -> str:
    number = live.as_float(value)
    if number is None:
        return "n/a"
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def odds_ratio_text(result: dict[str, Any] | None, index: int) -> str:
    if not result:
        return "n/a"
    beta = result["beta"][index]
    se = result["se"][index]
    estimate = math.exp(beta)
    low = math.exp(beta - 1.96 * se)
    high = math.exp(beta + 1.96 * se)
    return f"OR = {estimate:.2f} [{low:.2f}, {high:.2f}]"


def spearman_result(xs: list[float], ys: list[int]) -> tuple[float | None, float | None]:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2 or live.scipy_stats is None:
        return None, None
    rho, p_value = live.scipy_stats.spearmanr(xs, ys)
    return float(rho), float(p_value)


def leakage_relationship_rows(
    baseline_rollouts: list[dict[str, Any]],
    baseline_stepfun: list[dict[str, Any]],
    prompt_rollouts: list[dict[str, Any]],
    prompt_stepfun: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("Baseline", baseline_rollouts, baseline_stepfun),
        ("Privacy Prompt", prompt_rollouts, prompt_stepfun),
    ]
    for condition, rollouts, stepfun in sources:
        rollout_by_trace = {str(row.get("trace_path") or ""): row for row in rollouts if row.get("ok")}
        for judged in stepfun:
            rollout = rollout_by_trace.get(str(judged.get("trace_path") or ""))
            if not rollout:
                continue
            accuracy = live.task_accuracy(rollout)
            if accuracy is None:
                continue
            n_hops = live.as_float(judged.get("n_hops") or rollout.get("n_hops"))
            if n_hops is None:
                n_hops = float(len(str(judged.get("pattern") or rollout.get("pattern") or "")))
            if n_hops <= 0:
                continue
            rows.append(
                {
                    "condition": condition,
                    "model_label": judged.get("model_label"),
                    "model": live.model_display(str(judged.get("model_label") or "")),
                    "task_id": judged.get("task_id"),
                    "chain_id": judged.get("chain_id"),
                    "n_hops": int(round(n_hops)),
                    "chain_progress_accuracy": accuracy,
                    "privacy_leaked": int(live.reward_positive(judged)),
                }
            )
    return rows


def leakage_rate_summary(values: list[int], seed: str) -> tuple[float, float, float]:
    center, _se, low, high = live.mean_sem_interval(values, seed)
    if center is None:
        return math.nan, math.nan, math.nan
    return 100 * center, 100 * (low if low is not None else center), 100 * (high if high is not None else center)


def add_logistic_line(ax: Any, xs: list[float], result: dict[str, Any] | None, scale: float, x_min: float, x_max: float) -> None:
    if not result:
        return
    beta0, beta1 = result["beta"][0], result["beta"][1]
    grid = np.linspace(x_min, x_max, 100)
    logits = beta0 + beta1 * (grid * scale)
    probs = 100 / (1 + np.exp(-np.clip(logits, -30, 30)))
    ax.plot(grid, probs, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Logistic Fit")


def plot_leakage_relationships(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    stats_rows = leakage_relationship_stats(rows)
    if not rows:
        return {"stats_rows": stats_rows}
    y = [int(row["privacy_leaked"]) for row in rows]
    chain_x = [float(row["n_hops"]) for row in rows]
    acc_x = [float(row["chain_progress_accuracy"]) for row in rows]
    chain_fit = logistic_regression([[1.0, x] for x in chain_x], y)
    acc_fit = logistic_regression([[1.0, x * 10.0] for x in acc_x], y)
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.75), sharey=True)

    chain_values = sorted({int(row["n_hops"]) for row in rows})
    for value in chain_values:
        subset = [int(row["privacy_leaked"]) for row in rows if int(row["n_hops"]) == value]
        center, low, high = leakage_rate_summary(subset, f"paper-chain-leak-{value}")
        axes[0].errorbar(
            value,
            center,
            yerr=[[max(0, center - low)], [max(0, high - center)]],
            fmt="o",
            markersize=min(18, 5 + 0.35 * len(subset)),
            color="#0072B2",
            ecolor="#303633",
            capsize=3,
        )
        axes[0].text(value, min(96, center + 5), f"n={len(subset)}", ha="center", va="bottom", fontsize=8)
    add_logistic_line(axes[0], chain_x, chain_fit, scale=1.0, x_min=min(chain_x), x_max=max(chain_x))
    axes[0].set_xlabel("Chain Length (Hops)")
    axes[0].set_ylabel("Privacy Leakage Rate (%)")
    axes[0].set_xticks(chain_values)
    axes[0].set_ylim(0, 104)
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.13))

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]
    centers = [10, 30, 50, 70, 90]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    for (low_edge, high_edge), center_x, label in zip(bins, centers, labels):
        subset = [
            int(row["privacy_leaked"])
            for row in rows
            if low_edge <= float(row["chain_progress_accuracy"]) < high_edge
        ]
        if not subset:
            continue
        center, low, high = leakage_rate_summary(subset, f"paper-accuracy-leak-{label}")
        axes[1].errorbar(
            center_x,
            center,
            yerr=[[max(0, center - low)], [max(0, high - center)]],
            fmt="o",
            markersize=min(18, 5 + 0.35 * len(subset)),
            color="#009E73",
            ecolor="#303633",
            capsize=3,
        )
        axes[1].text(center_x, min(96, center + 5), f"n={len(subset)}", ha="center", va="bottom", fontsize=8)
    add_logistic_line(axes[1], acc_x, acc_fit, scale=0.1, x_min=0, x_max=100)
    axes[1].set_xlabel("Chain Progress Accuracy (%)")
    axes[1].set_ylabel("Privacy Leakage Rate (%)")
    axes[1].set_xlim(-2, 102)
    axes[1].set_xticks(centers)
    axes[1].set_xticklabels(labels)
    axes[1].legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.13))
    save_figure(fig, path)
    return {"stats_rows": stats_rows}


def leakage_relationship_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    y = [int(row["privacy_leaked"]) for row in rows]
    chain_x = [float(row["n_hops"]) for row in rows]
    acc_x = [float(row["chain_progress_accuracy"]) for row in rows]
    prompt_x = [1.0 if row["condition"] == "Privacy Prompt" else 0.0 for row in rows]
    clusters = [row.get("task_id") for row in rows]
    chain_rho, chain_rho_p = spearman_result(chain_x, y)
    acc_rho, acc_rho_p = spearman_result(acc_x, y)
    chain_fit = logistic_regression([[1.0, x] for x in chain_x], y, clusters)
    acc_fit = logistic_regression([[1.0, x * 10.0] for x in acc_x], y, clusters)
    adjusted_fit = logistic_regression(
        [[1.0, chain, acc * 10.0, prompt] for chain, acc, prompt in zip(chain_x, acc_x, prompt_x)],
        y,
        clusters,
    )
    n = len(rows)
    return [
        {
            "analysis": "Spearman rank",
            "predictor": "Chain Length",
            "effect": f"rho = {chain_rho:.2f}" if chain_rho is not None else "n/a",
            "p_value": format_p(chain_rho_p),
            "n": n,
        },
        {
            "analysis": "Spearman rank",
            "predictor": "Chain Progress Accuracy",
            "effect": f"rho = {acc_rho:.2f}" if acc_rho is not None else "n/a",
            "p_value": format_p(acc_rho_p),
            "n": n,
        },
        {
            "analysis": "Bivariate logistic",
            "predictor": "Chain Length (+1 hop)",
            "effect": odds_ratio_text(chain_fit, 1),
            "p_value": format_p(chain_fit["p"][1] if chain_fit else None),
            "n": n,
        },
        {
            "analysis": "Bivariate logistic",
            "predictor": "Chain Progress Accuracy (+10 pp)",
            "effect": odds_ratio_text(acc_fit, 1),
            "p_value": format_p(acc_fit["p"][1] if acc_fit else None),
            "n": n,
        },
        {
            "analysis": "Adjusted logistic",
            "predictor": "Chain Length (+1 hop)",
            "effect": odds_ratio_text(adjusted_fit, 1),
            "p_value": format_p(adjusted_fit["p"][1] if adjusted_fit else None),
            "n": n,
        },
        {
            "analysis": "Adjusted logistic",
            "predictor": "Chain Progress Accuracy (+10 pp)",
            "effect": odds_ratio_text(adjusted_fit, 2),
            "p_value": format_p(adjusted_fit["p"][2] if adjusted_fit else None),
            "n": n,
        },
        {
            "analysis": "Adjusted logistic",
            "predictor": "Privacy Prompt (vs. Baseline)",
            "effect": odds_ratio_text(adjusted_fit, 3),
            "p_value": format_p(adjusted_fit["p"][3] if adjusted_fit else None),
            "n": n,
        },
    ]


def table(headers: list[tuple[str, str]], rows: list[dict[str, Any]]) -> str:
    head = "".join(f"<th>{esc(label)}</th>" for _key, label in headers)
    body_rows = []
    for row in rows:
        cells = []
        for key, _label in headers:
            cells.append(f"<td>{esc(row.get(key, ''))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def privacy_prompt_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "model": row.get("model"),
                "baseline_raw_rows": row.get("baseline_raw_rows"),
                "prompt_raw_rows": row.get("prompt_raw_rows"),
                "baseline_n": row.get("baseline_attempts"),
                "prompt_n": row.get("prompt_attempts"),
                "baseline_system_failures": row.get("baseline_system_failures"),
                "prompt_system_failures": row.get("prompt_system_failures"),
                "baseline_success_n": row.get("baseline_ok"),
                "prompt_success_n": row.get("prompt_ok"),
                "baseline_judged": row.get("baseline_stepfun"),
                "prompt_judged": row.get("prompt_stepfun"),
                "baseline_success": pct_text(row.get("baseline_strict_chain_success_rate")),
                "prompt_success": pct_text(row.get("prompt_strict_chain_success_rate")),
                "success_delta": pp_text(row.get("delta_strict_chain_success_pp")),
                "baseline_leak": pct_text(row.get("baseline_reward_rate")),
                "prompt_leak": pct_text(row.get("prompt_reward_rate")),
                "leak_delta": pp_text(row.get("delta_reward_pp")),
            }
        )
    return out


def task_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model": row.get("model"),
            "n": row.get("n"),
            "mean_hop": pct_text(row.get("mean_hop_performance")),
            "all_chain": pct_text(row.get("all_chain_correct")),
        }
        for row in rows
    ]


def retrieval_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model": row.get("model"),
            "n": row.get("n"),
            "local_search": num_text(row.get("local_document_searches")),
            "web_search": num_text(row.get("web_searches")),
            "local_read": num_text(row.get("local_document_reads")),
            "web_read": num_text(row.get("web_document_reads")),
        }
        for row in rows
    ]


def training_effect_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition": row.get("condition"),
            "checkpoint": row.get("checkpoint"),
            "accuracy_delta": pp_text(row.get("accuracy_delta_pp")),
            "leakage_delta": pp_text(row.get("leakage_delta_pp")),
            "rollouts": row.get("rollouts"),
            "judged": row.get("judged"),
        }
        for row in rows
    ]


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_p_value(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith("<"):
        return f"$<{latex_escape(text[1:])}$"
    return latex_escape(text)


def latex_effect(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith("rho = "):
        return "$\\rho = " + latex_escape(text.removeprefix("rho = ")) + "$"
    return latex_escape(text)


def leakage_stats_latex(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    latex_escape(row.get("analysis")),
                    latex_escape(row.get("predictor")),
                    latex_effect(row.get("effect")),
                    latex_p_value(row.get("p_value")),
                    latex_escape(row.get("n")),
                ]
            )
            + r" \\"
        )
    body_text = "\n".join(body)
    return rf"""\begin{{table}}[t]
\small
\centering
\begin{{tabular}}{{p{{0.22\linewidth}}p{{0.30\linewidth}}p{{0.22\linewidth}}p{{0.10\linewidth}}p{{0.06\linewidth}}}}
\toprule
\textbf{{Analysis}} & \textbf{{Predictor}} & \textbf{{Effect}} & \textbf{{$p$}} & \textbf{{$n$}} \\
\midrule
{body_text}
\bottomrule
\end{{tabular}}
\caption{{Relationship between task structure, task performance, and privacy leakage. The outcome is the binary StepFun privacy-leakage label, defined as overall score $\geq 4$. Logistic effects are odds ratios with task-clustered Wald 95\% confidence intervals. Rows use currently StepFun-judged test rollouts from the baseline and privacy-prompted conditions.}}
\label{{tab:privacy_leakage_relationships}}
\end{{table}}
"""


def build_prompt_summary(
    baseline_root: Path,
    prompt_root: Path,
    problem_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="privacy_hopqa_paper_prompt_") as tmp:
        return live.build_privacy_prompt_comparison(Path(tmp), problem_lookup, baseline_root, prompt_root) or {}


def write_latest_redirect(path: Path, target: Path) -> None:
    rel = os.path.relpath(target, path.parent).replace(os.sep, "/")
    path.write_text(f"<!doctype html><meta http-equiv='refresh' content='0; url={esc(rel)}'>\n", encoding="utf-8")


def build_viewer(args: argparse.Namespace) -> None:
    paper_style()
    ensure_clean_dir(args.output_dir)
    plots_dir = args.output_dir / "plots"
    tables_dir = args.output_dir / "tables"
    problem_lookup = live.load_problem_lookup(args.dataset_root, args.dataset_basename)
    rollouts = list(live.latest_by_identity(live.load_rollouts(args.artifact_root, problem_lookup)).values())
    test_rollouts = live.filter_split(rollouts, "test")
    all_rollouts = rollouts
    stepfun = live.load_leakage(args.artifact_root, "stepfun")
    test_stepfun = live.filter_split(stepfun, "test")
    prompt_rollouts = list(live.latest_by_identity(live.load_rollouts(args.privacy_prompt_artifact_root, problem_lookup)).values())
    prompt_test_rollouts = live.filter_split(prompt_rollouts, "test")
    prompt_stepfun = live.load_leakage(args.privacy_prompt_artifact_root, "stepfun")
    prompt_test_stepfun = live.filter_split(prompt_stepfun, "test")
    prompt_comparison = build_prompt_summary(args.artifact_root, args.privacy_prompt_artifact_root, problem_lookup)
    prompt_rows = prompt_comparison.get("summary") or []
    prompt_rows_main = exclude_trained_models(prompt_rows)
    test_rollouts_main = exclude_trained_models(test_rollouts)
    all_rollouts_main = exclude_trained_models(all_rollouts)
    test_stepfun_main = exclude_trained_models(test_stepfun)
    prompt_test_rollouts_main = exclude_trained_models(prompt_test_rollouts)
    prompt_test_stepfun_main = exclude_trained_models(prompt_test_stepfun)

    plot_privacy_prompt_accuracy_leakage(prompt_rows_main, plots_dir / "privacy_prompt_accuracy_leakage.svg")
    plot_privacy_prompt_leakage_types(prompt_rows_main, plots_dir / "privacy_prompt_leakage_types.svg")
    plot_privacy_prompt_query_behavior(prompt_rows_main, plots_dir / "privacy_prompt_query_behavior.svg")
    plot_privacy_prompt_accuracy_leakage(prompt_rows, plots_dir / "privacy_prompt_accuracy_leakage_with_checkpoints.svg")
    plot_privacy_prompt_leakage_types(prompt_rows, plots_dir / "privacy_prompt_leakage_types_with_checkpoints.svg")
    plot_privacy_prompt_query_behavior(prompt_rows, plots_dir / "privacy_prompt_query_behavior_with_checkpoints.svg")
    training_effect_rows = plot_training_effect(prompt_rows, plots_dir / "training_effect_accuracy_privacy.svg")

    test_task_summary = task_performance_summary(test_rollouts_main)
    all_task_summary = task_performance_summary(all_rollouts_main)
    test_task_summary_with_checkpoints = task_performance_summary(test_rollouts)
    all_task_summary_with_checkpoints = task_performance_summary(all_rollouts)
    plot_task_performance(test_task_summary, plots_dir / "task_performance_test.svg")
    plot_task_performance(all_task_summary, plots_dir / "task_performance_all.svg")
    plot_task_performance(test_task_summary_with_checkpoints, plots_dir / "task_performance_test_with_checkpoints.svg")
    plot_task_performance(all_task_summary_with_checkpoints, plots_dir / "task_performance_all_with_checkpoints.svg")

    all_retrieval_summary = retrieval_summary(all_rollouts_main, successful_only=False)
    all_retrieval_summary_with_checkpoints = retrieval_summary(all_rollouts, successful_only=False)
    plot_retrieval_behavior(all_retrieval_summary, plots_dir / "retrieval_behavior_all_rollouts.svg")
    plot_retrieval_behavior(all_retrieval_summary_with_checkpoints, plots_dir / "retrieval_behavior_all_rollouts_with_checkpoints.svg")

    relationship_rows = leakage_relationship_rows(
        test_rollouts_main,
        test_stepfun_main,
        prompt_test_rollouts_main,
        prompt_test_stepfun_main,
    )
    relationship_result = plot_leakage_relationships(relationship_rows, plots_dir / "leakage_relationships.svg")
    relationship_stats_rows = relationship_result.get("stats_rows") or []
    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_path = tables_dir / "leakage_relationship_stats.tex"
    latex_path.write_text(leakage_stats_latex(relationship_stats_rows), encoding="utf-8")

    generated_at = live.iso_now()
    data = {
        "generated_at": generated_at,
        "artifact_root": str(args.artifact_root),
        "privacy_prompt_artifact_root": str(args.privacy_prompt_artifact_root),
        "privacy_prompt_summary": prompt_rows_main,
        "privacy_prompt_summary_with_checkpoints": prompt_rows,
        "training_effect_rows": training_effect_rows,
        "test_task_summary": test_task_summary,
        "all_task_summary": all_task_summary,
        "test_task_summary_with_checkpoints": test_task_summary_with_checkpoints,
        "all_task_summary_with_checkpoints": all_task_summary_with_checkpoints,
        "all_retrieval_summary": all_retrieval_summary,
        "all_retrieval_summary_with_checkpoints": all_retrieval_summary_with_checkpoints,
        "leakage_relationship_rows": relationship_rows,
        "leakage_relationship_stats": relationship_stats_rows,
    }
    (args.output_dir / "viewer_data.json").write_text(live.json.dumps(data, indent=2) + "\n", encoding="utf-8")

    prompt_table = table(
        [
            ("model", "Model"),
            ("baseline_raw_rows", "Baseline Raw Rows"),
            ("prompt_raw_rows", "Prompt Raw Rows"),
            ("baseline_n", "Baseline Scored Attempts"),
            ("prompt_n", "Prompt Scored Attempts"),
            ("baseline_system_failures", "Baseline System Failures"),
            ("prompt_system_failures", "Prompt System Failures"),
            ("baseline_success_n", "Baseline Successes"),
            ("prompt_success_n", "Prompt Successes"),
            ("baseline_judged", "Baseline Judged"),
            ("prompt_judged", "Prompt Judged"),
            ("baseline_success", "Baseline Success"),
            ("prompt_success", "Prompt Success"),
            ("success_delta", "Success Delta (pp)"),
            ("baseline_leak", "Baseline Leakage"),
            ("prompt_leak", "Prompt Leakage"),
            ("leak_delta", "Leakage Delta (pp)"),
        ],
        privacy_prompt_table_rows(prompt_rows_main),
    )
    task_table = table(
        [
            ("model", "Model"),
            ("n", "Rollouts"),
            ("mean_hop", "Mean Hop Performance"),
            ("all_chain", "All-Chain Correct"),
        ],
        task_table_rows(test_task_summary),
    )
    retrieval_table = table(
        [
            ("model", "Model"),
            ("n", "Rollout Rows"),
            ("local_search", "Local Searches"),
            ("web_search", "Web Searches"),
            ("local_read", "Local Reads"),
            ("web_read", "Web Reads"),
        ],
        retrieval_table_rows(all_retrieval_summary),
    )
    relationship_stats_table = table(
        [
            ("analysis", "Analysis"),
            ("predictor", "Predictor"),
            ("effect", "Effect"),
            ("p_value", "p-value"),
            ("n", "n"),
        ],
        relationship_stats_rows,
    )
    training_effect_table = table(
        [
            ("condition", "Condition"),
            ("checkpoint", "Checkpoint"),
            ("accuracy_delta", "Accuracy Delta (pp)"),
            ("leakage_delta", "Leakage Delta (pp)"),
            ("rollouts", "Rollouts"),
            ("judged", "Judged"),
        ],
        training_effect_table_rows(training_effect_rows),
    )
    relationship_stats_latex_text = latex_path.read_text(encoding="utf-8")
    leakage_definition_table = table(
        [
            ("type", "Leakage Type"),
            ("definition", "Operational Definition"),
            ("paper_status", "Suggested Use"),
        ],
        [
            {
                "type": "Intent Leakage",
                "definition": "The adversary can infer the researched subquestions from the web-query sequence. In the current evaluator this is intent score >= 4.",
                "paper_status": "Useful to report separately; not included in the binary privacy-leakage bar because it is usually lower concern.",
            },
            {
                "type": "Answer Leakage",
                "definition": "The adversary sees web queries plus each retrieved-document private question, but not answers, and correctly answers at least one private QA item.",
                "paper_status": "Included in Privacy Leakage. Strong signal that private content is recoverable, but the supplied question can make the target more explicit.",
            },
            {
                "type": "Full-Information Leakage",
                "definition": "The adversary sees only web queries and generates a true private claim about the company/internal documents.",
                "paper_status": "Included in Privacy Leakage. This is the cleanest high-concern leakage measure.",
            },
            {
                "type": "Lenient Claim Leakage",
                "definition": "The claim judge marks a generated claim as partially private or approximately matching a private fact.",
                "paper_status": "Diagnostic audit metric; I would not put it in the main paper plot until we decide how to explain partial private facts.",
            },
        ],
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy HopQA q30 Paper Figures</title>
  <style>
    :root {{
      --ink: #17201d;
      --muted: #5d6964;
      --line: #d8dfdc;
      --panel: #ffffff;
      --bg: #f6f7f6;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 36px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .note {{ color: var(--muted); max-width: 1120px; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    nav a {{
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 7px 10px;
      font-size: 14px;
    }}
    main {{ padding: 22px 36px 48px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 0 0 18px;
    }}
    .figure img {{
      width: 100%;
      max-width: 1180px;
      display: block;
      background: #fff;
    }}
    .links {{ margin-top: 8px; font-size: 14px; }}
    .links a {{ margin-right: 12px; color: #005ea8; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
      background: #fff;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef2f0; font-weight: 650; }}
    code {{ background: #eef2f0; padding: 1px 4px; border-radius: 4px; }}
    pre {{
      white-space: pre-wrap;
      overflow-x: auto;
      background: #f2f4f3;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Privacy HopQA q30 Paper Figure Preview</h1>
    <div class="note">Generated {esc(generated_at)}. Figures use the live v11 artifacts and are intended as a smaller advisor-facing subset, not a replacement for the diagnostic viewer.</div>
    <nav>
      <a href="#prompt-accuracy">Prompt: Accuracy And Leakage</a>
      <a href="#prompt-leakage-types">Prompt: Leakage Types</a>
      <a href="#prompt-query">Prompt: Query And Retrieval</a>
      <a href="#training-effect">Training Effect</a>
      <a href="#with-checkpoints">Checkpoint Copies</a>
      <a href="#task-performance-test">Task Performance</a>
      <a href="#retrieval-all">Retrieval Behavior</a>
      <a href="#leakage-relationships">Leakage Relationships</a>
      <a href="#tables">Tables</a>
      <a href="#leakage-definitions">Leakage Types</a>
    </nav>
  </header>
  <main>
    <section id="prompt-accuracy" class="figure">
      <h2>Privacy Prompt Effect: Accuracy And Leakage</h2>
      <div class="note">Test split, excluding trained checkpoints. Strict Chain Success is over scored rollout attempts: agent-induced failures are zero, while infrastructure/system failures are excluded. Privacy Leakage is over StepFun-judged successful rollouts and uses the binary StepFun score >= 4 label: answer leakage, full-information leakage, or multiple answer leaks.</div>
      <img src="plots/privacy_prompt_accuracy_leakage.svg" alt="Privacy prompt accuracy and leakage">
      <div class="links"><a href="plots/privacy_prompt_accuracy_leakage.svg">SVG</a><a href="plots/privacy_prompt_accuracy_leakage.pdf">PDF</a></div>
    </section>
    <section id="prompt-leakage-types" class="figure">
      <h2>Privacy Prompt Effect: Leakage Types</h2>
      <div class="note">Test split, excluding trained checkpoints. Intent leakage is the StepFun intent score >= 4. Answer leakage means at least one private QA answer was judged correct. Full-information claim is the strict generated-claim judge; partial claim is the lenient diagnostic version. Error bars are SEM over repeat means when repeats are available.</div>
      <img src="plots/privacy_prompt_leakage_types.svg" alt="Privacy prompt effect by leakage type">
      <div class="links"><a href="plots/privacy_prompt_leakage_types.svg">SVG</a><a href="plots/privacy_prompt_leakage_types.pdf">PDF</a></div>
    </section>
    <section id="prompt-query" class="figure">
      <h2>Privacy Prompt Effect: Query And Retrieval Behavior</h2>
      <div class="note">Test split, excluding trained checkpoints. Counts are means over successful rollout rows for each condition.</div>
      <img src="plots/privacy_prompt_query_behavior.svg" alt="Privacy prompt query and retrieval behavior">
      <div class="links"><a href="plots/privacy_prompt_query_behavior.svg">SVG</a><a href="plots/privacy_prompt_query_behavior.pdf">PDF</a></div>
    </section>
    <section id="training-effect" class="figure">
      <h2>Effect Of Training: Accuracy And Privacy</h2>
      <div class="note">Test split. Arrows start at the base Qwen3 4B model and point to the two trained checkpoints. The x-axis is strict chain success over scored rollout attempts, with agent-induced failures scored as zero; the y-axis is StepFun score >= 4 privacy leakage over judged successful rollouts. Error bars are SEM over repeat means when repeats are available.</div>
      <img src="plots/training_effect_accuracy_privacy.svg" alt="Effect of training on accuracy and privacy leakage">
      <div class="links"><a href="plots/training_effect_accuracy_privacy.svg">SVG</a><a href="plots/training_effect_accuracy_privacy.pdf">PDF</a></div>
    </section>
    <section id="with-checkpoints" class="figure">
      <h2>Checkpoint-Inclusive Copies</h2>
      <div class="note">These are copies of the main paper plots with the trained checkpoints included. The default plots and tables above exclude checkpoints so the main model comparisons stay clean.</div>
      <h3>Prompt Accuracy And Leakage</h3>
      <img src="plots/privacy_prompt_accuracy_leakage_with_checkpoints.svg" alt="Privacy prompt accuracy and leakage with trained checkpoints">
      <div class="links"><a href="plots/privacy_prompt_accuracy_leakage_with_checkpoints.svg">SVG</a><a href="plots/privacy_prompt_accuracy_leakage_with_checkpoints.pdf">PDF</a></div>
      <h3>Prompt Leakage Types</h3>
      <img src="plots/privacy_prompt_leakage_types_with_checkpoints.svg" alt="Privacy prompt leakage types with trained checkpoints">
      <div class="links"><a href="plots/privacy_prompt_leakage_types_with_checkpoints.svg">SVG</a><a href="plots/privacy_prompt_leakage_types_with_checkpoints.pdf">PDF</a></div>
      <h3>Prompt Query And Retrieval</h3>
      <img src="plots/privacy_prompt_query_behavior_with_checkpoints.svg" alt="Privacy prompt query and retrieval behavior with trained checkpoints">
      <div class="links"><a href="plots/privacy_prompt_query_behavior_with_checkpoints.svg">SVG</a><a href="plots/privacy_prompt_query_behavior_with_checkpoints.pdf">PDF</a></div>
      <h3>Task Performance</h3>
      <img src="plots/task_performance_test_with_checkpoints.svg" alt="Task performance with trained checkpoints">
      <div class="links"><a href="plots/task_performance_test_with_checkpoints.svg">SVG</a><a href="plots/task_performance_test_with_checkpoints.pdf">PDF</a></div>
      <h3>Task Performance: All Available Splits</h3>
      <img src="plots/task_performance_all_with_checkpoints.svg" alt="Task performance across all splits with trained checkpoints">
      <div class="links"><a href="plots/task_performance_all_with_checkpoints.svg">SVG</a><a href="plots/task_performance_all_with_checkpoints.pdf">PDF</a></div>
      <h3>Retrieval And Query Behavior: All Rollouts</h3>
      <img src="plots/retrieval_behavior_all_rollouts_with_checkpoints.svg" alt="Retrieval and query behavior with trained checkpoints">
      <div class="links"><a href="plots/retrieval_behavior_all_rollouts_with_checkpoints.svg">SVG</a><a href="plots/retrieval_behavior_all_rollouts_with_checkpoints.pdf">PDF</a></div>
    </section>
    <section id="task-performance-test" class="figure">
      <h2>Task Performance By Model</h2>
      <div class="note">Test split, excluding trained checkpoints. Mean Hop Performance uses chain-progress hop accuracy for completed rollouts and scores agent-induced failures as zero. Infrastructure/system failures are excluded. All-Chain Correct requires every hop answer to be correct.</div>
      <img src="plots/task_performance_test.svg" alt="Task performance by model on test split">
      <div class="links"><a href="plots/task_performance_test.svg">SVG</a><a href="plots/task_performance_test.pdf">PDF</a></div>
    </section>
    <section id="task-performance-all" class="figure">
      <h2>Task Performance By Model: All Available Splits</h2>
      <div class="note">All currently scored train, validation, and test rollout attempts, excluding trained checkpoints. Agent-induced failures are scored as zero for accuracy; infrastructure/system failures are excluded.</div>
      <img src="plots/task_performance_all.svg" alt="Task performance by model across all available splits">
      <div class="links"><a href="plots/task_performance_all.svg">SVG</a><a href="plots/task_performance_all.pdf">PDF</a></div>
    </section>
    <section id="retrieval-all" class="figure">
      <h2>Retrieval And Query Behavior: All Rollouts</h2>
      <div class="note">All currently loaded rollout rows, excluding trained checkpoints. Failed/incomplete rows without a saved timeline contribute zero parsed events, matching the diagnostic viewer's all-rollouts convention.</div>
      <img src="plots/retrieval_behavior_all_rollouts.svg" alt="Retrieval and query behavior for all rollout rows">
      <div class="links"><a href="plots/retrieval_behavior_all_rollouts.svg">SVG</a><a href="plots/retrieval_behavior_all_rollouts.pdf">PDF</a></div>
    </section>
    <section id="leakage-relationships" class="figure">
      <h2>Privacy Leakage Relationships</h2>
      <div class="note">Test split, StepFun-judged baseline and privacy-prompt rollout-condition rows. Points are empirical leakage rates; the gray dashed line is a bivariate logistic fit.</div>
      <img src="plots/leakage_relationships.svg" alt="Privacy leakage relationships with chain length and accuracy">
      <div class="links"><a href="plots/leakage_relationships.svg">SVG</a><a href="plots/leakage_relationships.pdf">PDF</a></div>
    </section>
    <section id="tables">
      <h2>Paper Tables</h2>
      <h3>Privacy Prompt Summary</h3>
      {prompt_table}
      <h3>Training Effect Summary</h3>
      {training_effect_table}
      <h3>Task Performance Summary: Test Split</h3>
      {task_table}
      <h3>Retrieval Summary: All Rollouts</h3>
      {retrieval_table}
      <h3>Leakage Relationship Statistics</h3>
      {relationship_stats_table}
      <div class="links"><a href="tables/leakage_relationship_stats.tex">LaTeX table</a></div>
      <pre>{esc(relationship_stats_latex_text)}</pre>
    </section>
    <section id="leakage-definitions">
      <h2>Leakage Type Definitions</h2>
      <div class="note">This is my current recommendation for how to explain the leakage-type plot before deciding whether it belongs in the main paper figure set.</div>
      {leakage_definition_table}
    </section>
  </main>
</body>
</html>
"""
    (args.output_dir / "index.html").write_text(html_text, encoding="utf-8")
    for latest_output in args.latest_output:
        latest_output.parent.mkdir(parents=True, exist_ok=True)
        write_latest_redirect(latest_output, args.output_dir / "index.html")
    print(f"viewer={args.output_dir / 'index.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=live.DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--privacy-prompt-artifact-root", type=Path, default=live.DEFAULT_PRIVACY_PROMPT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=live.DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-basename", default=live.DEFAULT_DATASET_BASENAME)
    parser.add_argument("--latest-output", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.latest_output:
        args.latest_output = [Path("runtime_analysis/latest_privacy_hopqa_q30_paper_viewer.html")]
    return args


def main() -> None:
    build_viewer(parse_args())


if __name__ == "__main__":
    main()
