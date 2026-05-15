#!/usr/bin/env python3
"""Build an aggregate viewer for Privacy HopQA binary reward-model SFT runs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


DEFAULT_EXPERIMENT_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515")
DEFAULT_OUTPUT_DIR = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_binary_reward_sft")
DEFAULT_LATEST = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/latest_privacy_hopqa_q30_binary_reward_sft.html")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.1f}%"


def num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_kind(name: str, summary: dict[str, Any]) -> str:
    if "smoke" in name:
        return "Smoke"
    if summary.get("eval_only"):
        return "Base"
    return "SFT"


def run_display(name: str, summary: dict[str, Any]) -> str:
    variant = str(summary.get("variant") or "")
    variant_name = {
        "queries_only": "Queries Only",
        "company_context": "Company Context",
        "chain_context": "Chain Context",
    }.get(variant, variant)
    kind = run_kind(name, summary)
    extra = ""
    args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
    if not summary.get("eval_only"):
        extra = f" ep{args.get('epochs')} rep{args.get('positive_repeat')}"
        if args.get("max_train_examples"):
            extra += f" n{args.get('max_train_examples')}"
    return f"{variant_name} {kind}{extra}".strip()


def binary_metrics(targets: list[int], probs: list[float], threshold: float) -> dict[str, Any]:
    preds = [1 if prob >= threshold else 0 for prob in probs]
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "rows": len(targets),
        "target_positive": sum(targets),
        "predicted_positive": sum(preds),
        "accuracy": (tp + tn) / len(targets) if targets else None,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def best_threshold(rows: list[dict[str, Any]], objective: str) -> tuple[float, dict[str, Any]]:
    targets = [int(row["target"]) for row in rows]
    probs = [float(row["prob_yes"]) for row in rows]
    candidates = sorted({0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, *probs})
    best: tuple[float, dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = binary_metrics(targets, probs, threshold)
        score = metrics[objective]
        if best is None or score > best[1][objective]:
            best = (threshold, metrics)
    return best or (0.5, binary_metrics(targets, probs, 0.5))


def load_runs(models_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(models_root.glob("*/summary.json")):
        summary = read_json(summary_path, {})
        if not summary:
            continue
        run_dir = summary_path.parent
        val_rows = read_jsonl(run_dir / "predictions_val.jsonl")
        test_rows = read_jsonl(run_dir / "predictions_test.jsonl")
        if not val_rows or not test_rows:
            continue
        val_f1_threshold, val_f1_metrics = best_threshold(val_rows, "f1")
        val_bal_threshold, val_bal_metrics = best_threshold(val_rows, "balanced_accuracy")
        test_targets = [int(row["target"]) for row in test_rows]
        test_probs = [float(row["prob_yes"]) for row in test_rows]
        val_targets = [int(row["target"]) for row in val_rows]
        val_probs = [float(row["prob_yes"]) for row in val_rows]
        item = {
            "name": run_dir.name,
            "display": run_display(run_dir.name, summary),
            "kind": run_kind(run_dir.name, summary),
            "variant": summary.get("variant"),
            "run_dir": str(run_dir),
            "summary": summary,
            "val_threshold_f1": val_f1_threshold,
            "val_threshold_balanced_accuracy": val_bal_threshold,
            "val_best_f1": val_f1_metrics,
            "val_best_balanced_accuracy": val_bal_metrics,
            "test_at_val_f1": binary_metrics(test_targets, test_probs, val_f1_threshold),
            "test_at_val_balanced_accuracy": binary_metrics(test_targets, test_probs, val_bal_threshold),
            "test_roc_auc": roc_auc_score(test_targets, test_probs) if len(set(test_targets)) > 1 else None,
            "test_average_precision": average_precision_score(test_targets, test_probs) if len(set(test_targets)) > 1 else None,
            "val_roc_auc": roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else None,
            "val_average_precision": average_precision_score(val_targets, val_probs) if len(set(val_targets)) > 1 else None,
        }
        runs.append(item)
    return runs


def dataset_tables(manifest: dict[str, Any]) -> str:
    summary = (manifest.get("summary") or {}).get("queries_only") or {}
    by_split = summary.get("by_split") or {}
    rows = []
    for split in ["train", "val", "test"]:
        item = by_split.get(split) or {}
        rows.append(
            f"<tr><td>{esc(split)}</td><td>{esc(item.get('rows'))}</td><td>{esc(item.get('tasks'))}</td>"
            f"<td>{esc(item.get('positive_rows'))}</td><td>{pct(item.get('positive_rate'))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Split</th><th>Rows</th><th>Tasks</th><th>Positive</th><th>Positive Rate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def run_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for run in sorted(runs, key=lambda item: (item["kind"], str(item["variant"]), item["name"])):
        m05 = (run["summary"].get("metrics") or {}).get("test") or {}
        tf1 = run["test_at_val_f1"]
        rows.append(
            "<tr>"
            f"<td>{esc(run['display'])}</td>"
            f"<td>{esc(run['variant'])}</td>"
            f"<td>{esc(run['kind'])}</td>"
            f"<td>{num(run['test_roc_auc'])}</td>"
            f"<td>{num(run['test_average_precision'])}</td>"
            f"<td>{pct(m05.get('accuracy'))}</td>"
            f"<td>{pct(m05.get('precision'))}</td>"
            f"<td>{pct(m05.get('recall'))}</td>"
            f"<td>{pct(m05.get('f1'))}</td>"
            f"<td>{num(run['val_threshold_f1'])}</td>"
            f"<td>{pct(tf1.get('precision'))}</td>"
            f"<td>{pct(tf1.get('recall'))}</td>"
            f"<td>{pct(tf1.get('f1'))}</td>"
            f"<td><code>{esc(run['run_dir'])}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Run</th><th>Variant</th><th>Kind</th><th>Test ROC AUC</th><th>Test AP</th>"
        "<th>Acc @0.5</th><th>Prec @0.5</th><th>Recall @0.5</th><th>F1 @0.5</th>"
        "<th>Val F1 Threshold</th><th>Prec @ValThr</th><th>Recall @ValThr</th><th>F1 @ValThr</th><th>Path</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def plot_runs(runs: list[dict[str, Any]], output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_runs = [run for run in runs if run["kind"] != "Smoke"]
    labels = [run["display"] for run in plot_runs]
    x = np.arange(len(plot_runs))

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    width = 0.34
    ax.bar(x - width / 2, [run["test_roc_auc"] or 0 for run in plot_runs], width=width, label="ROC AUC", color="#0072B2")
    ax.bar(x + width / 2, [run["test_average_precision"] or 0 for run in plot_runs], width=width, label="Average Precision", color="#009E73")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "ranking_metrics.svg")
    fig.savefig(plots / "ranking_metrics.pdf")
    plt.close(fig)

    metrics = ["precision", "recall", "specificity", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    for ax, source, title in [
        (axes[0], "summary", "Threshold 0.5"),
        (axes[1], "test_at_val_f1", "Threshold Selected On Val F1"),
    ]:
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(metrics))
        for metric, offset in zip(metrics, offsets):
            if source == "summary":
                values = [((run["summary"].get("metrics") or {}).get("test") or {}).get(metric) or 0 for run in plot_runs]
            else:
                values = [run[source].get(metric) or 0 for run in plot_runs]
            ax.bar(x + offset, values, width=width, label=metric.replace("_", " ").title())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test Score")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(1.05, 1.22))
    fig.tight_layout()
    fig.savefig(plots / "threshold_metrics.svg")
    fig.savefig(plots / "threshold_metrics.pdf")
    plt.close(fig)

    best = max((run for run in plot_runs if run["kind"] == "SFT"), key=lambda run: run["test_average_precision"] or -1, default=None)
    if best:
        rows = read_jsonl(Path(best["run_dir"]) / "predictions_test.jsonl")
        positives = [float(row["prob_yes"]) for row in rows if row["target"]]
        negatives = [float(row["prob_yes"]) for row in rows if not row["target"]]
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        bins = np.linspace(0, 1, 21)
        ax.hist(negatives, bins=bins, alpha=0.65, label="True No", color="#5F6F7A")
        ax.hist(positives, bins=bins, alpha=0.65, label="True Yes", color="#D55E00")
        ax.axvline(best["val_threshold_f1"], color="#222", linestyle="--", linewidth=1.4, label="Val F1 Threshold")
        ax.set_xlabel("Predicted P(Leakage)")
        ax.set_ylabel("Test Rows")
        ax.set_title(f"Best Run Score Distribution: {best['display']}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plots / "best_score_distribution.svg")
        fig.savefig(plots / "best_score_distribution.pdf")
        plt.close(fig)


def write_latest(path: Path, target: Path) -> None:
    rel = os.path.relpath(target, path.parent).replace(os.sep, "/")
    path.write_text(f"<!doctype html><meta http-equiv='refresh' content='0; url={esc(rel)}'>\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.experiment_root / "models")
    manifest = read_json(args.experiment_root / "dataset" / "manifest.json", {})
    plot_runs(runs, args.output_dir)
    data = {
        "generated_at": iso_now(),
        "experiment_root": str(args.experiment_root),
        "runs": runs,
        "dataset_manifest": manifest,
    }
    (args.output_dir / "viewer_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    note = (
        "The target is StepFun score 4-5. Splits are task_id-grouped for reward-model evaluation, "
        "so no DR task appears in more than one train/val/test split."
    )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Privacy HopQA Binary Reward SFT</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #17201d; }}
section {{ margin-bottom: 28px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
td, th {{ border-bottom: 1px solid #d8dfdc; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f0; }}
img {{ max-width: 1200px; width: 100%; display: block; margin: 12px 0; }}
code {{ background: #eef2f0; padding: 1px 4px; border-radius: 4px; }}
.note {{ color: #5d6964; }}
</style></head><body>
<h1>Privacy HopQA Binary Reward SFT</h1>
<p class="note">Generated {esc(data['generated_at'])}. {esc(note)}</p>
<section><h2>Dataset</h2>{dataset_tables(manifest)}</section>
<section><h2>Run Summary</h2>{run_table(runs)}</section>
<section><h2>Ranking Metrics</h2><img src="plots/ranking_metrics.svg"></section>
<section><h2>Threshold Metrics</h2><img src="plots/threshold_metrics.svg"></section>
<section><h2>Best Score Distribution</h2><img src="plots/best_score_distribution.svg"></section>
</body></html>
"""
    (args.output_dir / "index.html").write_text(html_text, encoding="utf-8")
    write_latest(args.latest_output, args.output_dir / "index.html")
    print(f"viewer={args.output_dir / 'index.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latest-output", type=Path, default=DEFAULT_LATEST)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
