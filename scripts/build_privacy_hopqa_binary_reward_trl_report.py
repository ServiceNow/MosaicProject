#!/usr/bin/env python3
"""Build a compact report for binary privacy reward-model SFT experiments."""

from __future__ import annotations

import argparse
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


DEFAULT_MANUAL_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515")
DEFAULT_TRL_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_trl_20260515")
DEFAULT_OUTPUT_DIR = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_binary_reward_trl_report")
DEFAULT_LATEST = Path("/home/toolkit/my_code/MosaicProject/runtime_analysis/latest_privacy_hopqa_q30_binary_reward_trl_report.html")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def short_variant(value: str) -> str:
    return {
        "queries_only": "Queries Only",
        "company_context": "Company Context",
        "chain_context": "Chain Context",
    }.get(value, value)


def binary_metrics(targets: list[int], probs: list[float], threshold: float) -> dict[str, Any]:
    preds = [1 if prob >= threshold else 0 for prob in probs]
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
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
        "predicted_positive": sum(preds),
    }


def best_threshold(rows: list[dict[str, Any]], objective: str = "f1") -> tuple[float, dict[str, Any]]:
    targets = [int(row["target"]) for row in rows]
    probs = [float(row["prob_yes"]) for row in rows]
    candidates = sorted({0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, *probs})
    best: tuple[float, dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = binary_metrics(targets, probs, threshold)
        if best is None or (metrics.get(objective) or 0.0) > (best[1].get(objective) or 0.0):
            best = (threshold, metrics)
    return best or (0.5, binary_metrics(targets, probs, 0.5))


def wandb_url(run_dir: Path) -> str | None:
    for offline_dir in sorted(run_dir.glob("wandb/wandb/offline-run-*")):
        run_id = offline_dir.name.rsplit("-", 1)[-1]
        if run_id:
            return f"https://wandb.ai/alexgurung/privacy_hopqa_binary_reward_sft/runs/{run_id}"
    return None


def classify_source(summary: dict[str, Any], source_label: str) -> str:
    if summary.get("trainer") == "trl.SFTTrainer" or source_label == "TRL":
        return "TRL SFTTrainer"
    if summary.get("eval_only"):
        return "Base"
    return "Manual Trainer"


def run_name(run_dir: Path, summary: dict[str, Any], source_label: str) -> str:
    trainer = classify_source(summary, source_label)
    variant = short_variant(str(summary.get("variant") or ""))
    args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
    if summary.get("eval_only"):
        return f"{variant} Base"
    suffix = f"ep{args.get('epochs')} rep{args.get('positive_repeat')}"
    if args.get("max_train_examples"):
        suffix += f" n{args.get('max_train_examples')}"
    return f"{variant} {trainer} {suffix}"


def load_runs(root: Path, source_label: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted((root / "models").glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path, {})
        val_rows = read_jsonl(run_dir / "predictions_val.jsonl")
        test_rows = read_jsonl(run_dir / "predictions_test.jsonl")
        if not summary or not val_rows or not test_rows:
            continue
        test_targets = [int(row["target"]) for row in test_rows]
        test_probs = [float(row["prob_yes"]) for row in test_rows]
        val_targets = [int(row["target"]) for row in val_rows]
        val_probs = [float(row["prob_yes"]) for row in val_rows]
        val_f1_threshold, val_f1 = best_threshold(val_rows, "f1")
        run = {
            "name": run_dir.name,
            "display": run_name(run_dir, summary, source_label),
            "source": classify_source(summary, source_label),
            "variant": summary.get("variant"),
            "run_dir": str(run_dir),
            "wandb_url": wandb_url(run_dir),
            "summary": summary,
            "val_roc_auc": roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else None,
            "val_average_precision": average_precision_score(val_targets, val_probs) if len(set(val_targets)) > 1 else None,
            "test_roc_auc": roc_auc_score(test_targets, test_probs) if len(set(test_targets)) > 1 else None,
            "test_average_precision": average_precision_score(test_targets, test_probs) if len(set(test_targets)) > 1 else None,
            "val_threshold_f1": val_f1_threshold,
            "test_at_val_f1": binary_metrics(test_targets, test_probs, val_f1_threshold),
            "test_rows": len(test_rows),
            "test_positive": sum(test_targets),
        }
        runs.append(run)
    return runs


def dataset_table(manifest: dict[str, Any]) -> str:
    split_summary = (((manifest.get("summary") or {}).get("queries_only") or {}).get("by_split") or {})
    rows = []
    for split in ["train", "val", "test"]:
        item = split_summary.get(split) or {}
        rows.append(
            f"<tr><td>{esc(split.title())}</td><td>{esc(item.get('rows'))}</td><td>{esc(item.get('tasks'))}</td>"
            f"<td>{esc(item.get('positive_rows'))}</td><td>{pct(item.get('positive_rate'))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Split</th><th>Rows</th><th>Tasks</th><th>Positive Rows</th><th>Positive Rate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def decisions_table() -> str:
    rows = [
        ("Target", "Binary label is Yes when StepFun overall privacy leakage score is 4 or 5; No for 1-3."),
        ("Input", "Model receives the same rendered privacy-classifier prompt used for the reward model; output is exactly one token, Yes or No."),
        ("Evaluation", "Metrics use next-token probabilities for the exact Yes and No tokens, not sampled generation."),
        ("Splits", "Clean deterministic task-id grouped train/validation/test split."),
        ("Training", "Qwen3-4B-Instruct LoRA SFT, r=16, alpha=32, dropout=0.05, bf16, two epochs unless marked as smoke."),
        ("Efficiency", "TRL SFTTrainer prompt-completion loss, FlashAttention 2, Liger kernel, fused AdamW, gradient checkpointing."),
        ("Logging", "W&B project alexgurung/privacy_hopqa_binary_reward_sft. Early TRL runs were offline then synced; future runner default is online."),
    ]
    return "<table><tbody>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows) + "</tbody></table>"


def variants_table() -> str:
    rows = [
        ("Queries Only", "Only the visible web query sequence. This is the strict deployable setting if the reward model only sees public searches."),
        ("Company Context", "Visible web queries plus company name. This is still cheap at runtime and helps decide whether a query reveals enterprise-private facts."),
        ("Chain Context", "Visible web queries plus company, hop pattern, research question, and subquestions. This is diagnostic/oracle-style context, not the main deployable reward-model setting."),
    ]
    return "<table><tbody>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows) + "</tbody></table>"


def run_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    sorted_runs = sorted(
        runs,
        key=lambda run: (
            {"Base": 0, "Manual Trainer": 1, "TRL SFTTrainer": 2}.get(run["source"], 9),
            str(run["variant"]),
            run["name"],
        ),
    )
    for run in sorted_runs:
        if "smoke" in run["name"]:
            continue
        m05 = (run["summary"].get("metrics") or {}).get("test") or {}
        tv = run["test_at_val_f1"]
        wandb = f"<a href='{esc(run['wandb_url'])}'>W&B</a>" if run.get("wandb_url") else ""
        rows.append(
            "<tr>"
            f"<td>{esc(run['display'])}</td>"
            f"<td>{esc(run['source'])}</td>"
            f"<td>{esc(short_variant(str(run['variant'])))}</td>"
            f"<td>{num(run['test_roc_auc'])}</td>"
            f"<td>{num(run['test_average_precision'])}</td>"
            f"<td>{pct(m05.get('f1'))}</td>"
            f"<td>{pct(m05.get('precision'))}</td>"
            f"<td>{pct(m05.get('recall'))}</td>"
            f"<td>{num(run['val_threshold_f1'])}</td>"
            f"<td>{pct(tv.get('f1'))}</td>"
            f"<td>{pct(tv.get('precision'))}</td>"
            f"<td>{pct(tv.get('recall'))}</td>"
            f"<td>{esc(run['test_positive'])}/{esc(run['test_rows'])}</td>"
            f"<td>{wandb}</td>"
            f"<td><code>{esc(run['run_dir'])}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Run</th><th>Trainer</th><th>Variant</th><th>Test ROC AUC</th><th>Test AP</th>"
        "<th>F1 @0.5</th><th>Precision @0.5</th><th>Recall @0.5</th><th>Val F1 Threshold</th>"
        "<th>F1 @ValThr</th><th>Precision @ValThr</th><th>Recall @ValThr</th><th>Test Positives</th><th>W&B</th><th>Path</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def best_recommendation(runs: list[dict[str, Any]]) -> str:
    full_runs = [run for run in runs if "smoke" not in run["name"] and run["source"] != "Base"]
    deployable = [run for run in full_runs if run["variant"] in {"queries_only", "company_context"}]
    best_practical = max(deployable, key=lambda run: (run["test_average_precision"] or -1, ((run["summary"].get("metrics") or {}).get("test") or {}).get("f1") or -1), default=None)
    best_queries = max([run for run in full_runs if run["variant"] == "queries_only"], key=lambda run: run["test_average_precision"] or -1, default=None)
    if not best_practical:
        return "<p>Not enough completed runs to make a recommendation yet.</p>"
    m05 = (best_practical["summary"].get("metrics") or {}).get("test") or {}
    pieces = [
        f"Use <strong>{esc(best_practical['display'])}</strong> as the current practical reward-model candidate.",
        f"It gives test AP {num(best_practical['test_average_precision'])}, ROC AUC {num(best_practical['test_roc_auc'])}, and F1 {pct(m05.get('f1'))} at the raw 0.5 threshold.",
    ]
    if best_queries and best_queries is not best_practical:
        qm05 = (best_queries["summary"].get("metrics") or {}).get("test") or {}
        pieces.append(
            f"If runtime access must be restricted to queries only, use <strong>{esc(best_queries['display'])}</strong> "
            f"(test AP {num(best_queries['test_average_precision'])}, F1 {pct(qm05.get('f1'))})."
        )
    pieces.append("Use the probability score during agent training; use the validation-selected threshold only for binary reporting.")
    return "<p>" + " ".join(pieces) + "</p>"


def plot_runs(runs: list[dict[str, Any]], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_runs = [run for run in runs if "smoke" not in run["name"] and run["source"] != "Base"]
    plot_runs = sorted(plot_runs, key=lambda run: (str(run["variant"]), run["source"], run["name"]))
    if not plot_runs:
        return
    labels = [run["display"].replace(" Trainer", "").replace("SFTTrainer", "TRL") for run in plot_runs]
    x = np.arange(len(plot_runs))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.bar(x - width / 2, [run["test_roc_auc"] or 0 for run in plot_runs], width, label="ROC AUC", color="#0072B2")
    ax.bar(x + width / 2, [run["test_average_precision"] or 0 for run in plot_runs], width, label="Average Precision", color="#009E73")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(plot_dir / "ranking_metrics.svg")
    fig.savefig(plot_dir / "ranking_metrics.pdf")
    plt.close(fig)

    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for ax, title, source in [(axes[0], "Raw Threshold 0.5", "summary"), (axes[1], "Validation F1 Threshold", "test_at_val_f1")]:
        offsets = np.linspace(-0.22, 0.22, len(metrics))
        for metric, offset in zip(metrics, offsets):
            values = []
            for run in plot_runs:
                if source == "summary":
                    values.append((((run["summary"].get("metrics") or {}).get("test") or {}).get(metric)) or 0)
                else:
                    values.append(run[source].get(metric) or 0)
            ax.bar(x + offset, values, width=0.20, label=metric.title())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test Score")
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(1.05, 1.15))
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_metrics.svg")
    fig.savefig(plot_dir / "threshold_metrics.pdf")
    plt.close(fig)

    best = max(plot_runs, key=lambda run: run["test_average_precision"] or -1)
    rows = read_jsonl(Path(best["run_dir"]) / "predictions_test.jsonl")
    positives = [float(row["prob_yes"]) for row in rows if row["target"]]
    negatives = [float(row["prob_yes"]) for row in rows if not row["target"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(negatives, bins=bins, alpha=0.66, label="True No", color="#5F6F7A")
    ax.hist(positives, bins=bins, alpha=0.66, label="True Yes", color="#D55E00")
    ax.axvline(best["val_threshold_f1"], color="#222", linestyle="--", linewidth=1.4, label="Val F1 Threshold")
    ax.set_xlabel("Predicted P(Leakage)")
    ax.set_ylabel("Test Rows")
    ax.set_title(f"Best Run Score Distribution: {best['display']}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_score_distribution.svg")
    fig.savefig(plot_dir / "best_score_distribution.pdf")
    plt.close(fig)


def write_latest(path: Path, target: Path) -> None:
    rel = os.path.relpath(target, path.parent).replace(os.sep, "/")
    path.write_text(f"<!doctype html><meta http-equiv='refresh' content='0; url={esc(rel)}'>\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.manual_root, "Manual") + load_runs(args.trl_root, "TRL")
    manifest = read_json(args.manual_root / "dataset" / "manifest.json", {})
    plot_runs(runs, args.output_dir)
    data = {
        "generated_at": iso_now(),
        "manual_root": str(args.manual_root),
        "trl_root": str(args.trl_root),
        "runs": runs,
        "dataset_manifest": manifest,
    }
    (args.output_dir / "viewer_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Binary Privacy Reward Model TRL Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #17201d; line-height: 1.42; }}
section {{ margin-bottom: 30px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
td, th {{ border-bottom: 1px solid #d8dfdc; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f0; }}
img {{ max-width: 1220px; width: 100%; display: block; margin: 12px 0; }}
code {{ background: #eef2f0; padding: 1px 4px; border-radius: 4px; }}
.note {{ color: #5d6964; }}
.callout {{ background: #f4f7f5; border-left: 4px solid #009E73; padding: 10px 14px; }}
</style></head><body>
<h1>Binary Privacy Reward Model TRL Report</h1>
<p class="note">Generated {esc(data['generated_at'])}. The label predicts whether StepFun judged a rollout as privacy-leaking at score 4 or 5.</p>
<section class="callout"><h2>Recommendation</h2>{best_recommendation(runs)}</section>
<section><h2>Dataset</h2>{dataset_table(manifest)}</section>
<section><h2>Training Decisions</h2>{decisions_table()}</section>
<section><h2>Prompt Variants</h2>{variants_table()}</section>
<section><h2>Completed Runs</h2>{run_table(runs)}</section>
<section><h2>Ranking Metrics</h2><img src="plots/ranking_metrics.svg"></section>
<section><h2>Threshold Metrics</h2><img src="plots/threshold_metrics.svg"></section>
<section><h2>Score Distribution</h2><img src="plots/best_score_distribution.svg"></section>
</body></html>
"""
    (args.output_dir / "index.html").write_text(html_text, encoding="utf-8")
    write_latest(args.latest_output, args.output_dir / "index.html")
    print(f"viewer={args.output_dir / 'index.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", type=Path, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--trl-root", type=Path, default=DEFAULT_TRL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latest-output", type=Path, default=DEFAULT_LATEST)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
