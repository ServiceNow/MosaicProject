#!/usr/bin/env python3
"""Build a compact report for privacy reward-model SFT experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
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
        "private_fact_context": "Private Facts",
    }.get(value, value)


def compact_variant(value: str | None) -> str:
    return {
        "queries_only": "Queries",
        "company_context": "Company",
        "chain_context": "Chain",
        "private_fact_context": "Private Facts",
    }.get(value or "", value or "Unknown")


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
    candidates: list[Path] = []
    for pattern in ["wandb/wandb/offline-run-*", "wandb/wandb/run-*"]:
        candidates.extend(run_dir.glob(pattern))
    for wandb_dir in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True):
        run_id = wandb_dir.name.rsplit("-", 1)[-1]
        if run_id:
            return f"https://wandb.ai/alexgurung/privacy_hopqa_binary_reward_sft/runs/{run_id}"
    return None


def trainer_eval_loss(run_dir: Path) -> dict[str, Any]:
    losses = [point["eval_loss"] for point in trainer_eval_history(run_dir)]
    runtime = read_json(run_dir / "training_runtime.json", {})
    best_metric = runtime.get("best_metric")
    if best_metric is not None:
        try:
            losses.append(float(best_metric))
        except Exception:
            pass
    if not losses:
        return {}
    out = {"best_eval_loss": min(losses), "last_eval_loss": losses[-1], "eval_loss_points": len(losses)}
    if runtime.get("best_model_checkpoint"):
        out["best_model_checkpoint"] = runtime.get("best_model_checkpoint")
    return out


def trainer_eval_history(run_dir: Path) -> list[dict[str, Any]]:
    points_by_step: dict[int, dict[str, Any]] = {}
    state_paths = []
    root_state = run_dir / "trainer" / "trainer_state.json"
    if root_state.exists():
        state_paths.append(root_state)
    state_paths.extend(sorted((run_dir / "trainer").glob("checkpoint-*/trainer_state.json")))
    seen: set[Path] = set()
    for path in state_paths:
        if path in seen:
            continue
        seen.add(path)
        state = read_json(path, {})
        for item in state.get("log_history") or []:
            if "eval_loss" in item:
                raw_step = item.get("step")
                step = int(raw_step if raw_step is not None else len(points_by_step))
                loss = float(item["eval_loss"])
                points_by_step[step] = {
                    "step": step,
                    "epoch": item.get("epoch"),
                    "eval_loss": loss,
                    "eval_mean_token_accuracy": item.get("eval_mean_token_accuracy"),
                    "eval_entropy": item.get("eval_entropy"),
                }
    return [points_by_step[step] for step in sorted(points_by_step)]


def classify_source(summary: dict[str, Any], source_label: str) -> str:
    if summary.get("trainer") == "trl.SFTTrainer" or source_label == "TRL":
        return "TRL SFTTrainer"
    if summary.get("eval_only"):
        return "Base"
    return "Manual Trainer"


def lora_label(summary: dict[str, Any]) -> str:
    args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
    if summary.get("eval_only"):
        return "n/a"
    target = args.get("lora_target_modules") or "default"
    dropout = args.get("lora_dropout")
    r = args.get("lora_r")
    alpha = args.get("lora_alpha")
    if r is None:
        return "r16 default d0.05"
    return f"r{r} a{alpha} {target} d{dropout}"


def compact_lora_label(summary: dict[str, Any]) -> str:
    args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
    if summary.get("eval_only"):
        return "Base"
    r = args.get("lora_r")
    target = args.get("lora_target_modules") or "default"
    if r is None:
        r = 16
    if target == "all-linear":
        return f"r{r} all-linear"
    return f"r{r}"


def compact_source(value: str) -> str:
    return {
        "Manual Trainer": "Custom SFT",
        "TRL SFTTrainer": "TRL SFT",
        "Base": "Base",
    }.get(value, value)


def source_label(value: str) -> str:
    return compact_source(value)


def repeat_label(summary: dict[str, Any]) -> str:
    args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
    repeat = args.get("positive_repeat")
    if repeat is None:
        return ""
    try:
        repeat_int = int(repeat)
    except Exception:
        return f"Repeat {repeat}"
    if repeat_int <= 1:
        return "No repeat"
    return f"Repeat x{repeat_int}"


def clean_run_label(run: dict[str, Any], multiline: bool = False) -> str:
    summary = run.get("summary") or {}
    variant = compact_variant(run.get("variant"))
    source = compact_source(str(run.get("source") or ""))
    if summary.get("eval_only"):
        parts = [variant, "Base"]
    elif summary.get("label_mode") == "ternary":
        parts = [variant, source, "1-3 target", compact_lora_label(summary)]
    else:
        parts = [variant, source, compact_lora_label(summary), repeat_label(summary)]
    parts = [part for part in parts if part]
    return "\n".join(parts) if multiline else " - ".join(parts)


def run_name(run_dir: Path, summary: dict[str, Any], source_label: str) -> str:
    trainer = classify_source(summary, source_label)
    variant = short_variant(str(summary.get("variant") or ""))
    args = summary.get("args") if isinstance(summary.get("args"), dict) else {}
    if summary.get("eval_only"):
        return f"{variant} Base"
    if summary.get("label_mode") == "ternary":
        suffix = f"ternary ep{args.get('epochs')} tri{args.get('ternary_no_repeat')}-{args.get('ternary_intent_repeat')}-{args.get('ternary_fact_repeat')} {lora_label(summary)}"
    else:
        suffix = f"ep{args.get('epochs')} rep{args.get('positive_repeat')} {lora_label(summary)}"
    if args.get("max_train_examples"):
        suffix += f" n{args.get('max_train_examples')}"
    return f"{variant} {trainer} {suffix}".strip()


def load_binary_runs(root: Path, source_label: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted((root / "models").glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path, {})
        if summary.get("label_mode", "binary") != "binary":
            continue
        val_rows = read_jsonl(run_dir / "predictions_val.jsonl")
        test_rows = read_jsonl(run_dir / "predictions_test.jsonl")
        if not summary or not val_rows or not test_rows or "prob_yes" not in test_rows[0]:
            continue
        test_targets = [int(row["target"]) for row in test_rows]
        test_probs = [float(row["prob_yes"]) for row in test_rows]
        val_targets = [int(row["target"]) for row in val_rows]
        val_probs = [float(row["prob_yes"]) for row in val_rows]
        val_f1_threshold, _ = best_threshold(val_rows, "f1")
        run = {
            "name": run_dir.name,
            "display": run_name(run_dir, summary, source_label),
            "source": classify_source(summary, source_label),
            "variant": summary.get("variant"),
            "run_dir": str(run_dir),
            "wandb_url": wandb_url(run_dir),
            "summary": summary,
            "lora": lora_label(summary),
            "eval_loss": trainer_eval_loss(run_dir),
            "eval_history": trainer_eval_history(run_dir),
            "val_roc_auc": roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else None,
            "val_average_precision": average_precision_score(val_targets, val_probs) if len(set(val_targets)) > 1 else None,
            "test_roc_auc": roc_auc_score(test_targets, test_probs) if len(set(test_targets)) > 1 else None,
            "test_average_precision": average_precision_score(test_targets, test_probs) if len(set(test_targets)) > 1 else None,
            "val_threshold_f1": val_f1_threshold,
            "test_at_val_f1": binary_metrics(test_targets, test_probs, val_f1_threshold),
            "test_rows": len(test_rows),
            "test_positive": sum(test_targets),
        }
        run["short_label"] = clean_run_label(run)
        runs.append(run)
    return runs


def load_ternary_runs(root: Path, source_label: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted((root / "models").glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path, {})
        if summary.get("label_mode") != "ternary":
            continue
        test_rows = read_jsonl(run_dir / "predictions_test.jsonl")
        if not summary or not test_rows:
            continue
        run = {
            "name": run_dir.name,
            "display": run_name(run_dir, summary, source_label),
            "source": classify_source(summary, source_label),
            "variant": summary.get("variant"),
            "run_dir": str(run_dir),
            "wandb_url": wandb_url(run_dir),
            "summary": summary,
            "lora": lora_label(summary),
            "eval_loss": trainer_eval_loss(run_dir),
            "eval_history": trainer_eval_history(run_dir),
        }
        run["short_label"] = clean_run_label(run)
        runs.append(run)
    return runs


def dataset_table(manifest: dict[str, Any]) -> str:
    summary = (manifest.get("summary") or {}).get("queries_only") or {}
    split_summary = summary.get("by_split") or {}
    rows = []
    for split in ["train", "val", "test"]:
        item = split_summary.get(split) or {}
        rows.append(
            f"<tr><td>{esc(split.title())}</td><td>{esc(item.get('rows'))}</td><td>{esc(item.get('tasks'))}</td>"
            f"<td>{esc(item.get('positive_rows'))}</td><td>{pct(item.get('positive_rate'))}</td></tr>"
        )
    baseline = (
        "<p class='note'>Random ranking AUC is 0.500. The test positive rate is "
        f"{pct((split_summary.get('test') or {}).get('positive_rate'))}; an always-No classifier has zero recall and zero F1. "
        "Run tables report the row counts from each run's own prediction files, so older pre-rebuild runs may show different counts.</p>"
    )
    return (
        "<table><thead><tr><th>Split</th><th>Rows</th><th>Tasks</th><th>Fact-Leak Positive Rows</th><th>Positive Rate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{baseline}"
    )


def decisions_table() -> str:
    rows = [
        ("Binary Target", "Yes when StepFun overall privacy leakage score is 4 or 5; No for 1-3."),
        ("Ternary Target", "A = no leakage, B = intent-only leakage, C = answer/full-information fact leakage."),
        ("Trainer Labels", "Custom SFT is the older hand-written Hugging Face Trainer loop; TRL SFT uses TRL's SFTTrainer. Both train the causal LM on label tokens, not a separate classification head."),
        ("Binary Score", "P(leakage) is softmax(Yes, No) from the next-token logits after the prompt."),
        ("Threshold", "The 0.5 threshold is the Yes-vs-No argmax. The report also shows a validation-selected threshold chosen to maximize validation F1."),
        ("Evaluation", "Metrics use next-token probabilities for exact label tokens, not sampled generation."),
        ("Primary Metrics", "ROC AUC for ranking plus precision, recall, and F1 at threshold 0.5 and a validation-selected F1 threshold."),
        ("Splits", "Clean deterministic task-id grouped train/validation/test split."),
        ("Efficiency", "TRL SFTTrainer prompt-completion loss, FlashAttention 2, Liger kernel, fused AdamW, bf16, gradient checkpointing."),
        ("Logging", "W&B project alexgurung/privacy_hopqa_binary_reward_sft; current TRL runs stream online and log eval_loss."),
    ]
    return "<table><tbody>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows) + "</tbody></table>"


def variants_table() -> str:
    rows = [
        ("Queries Only", "Only the visible web query sequence. This is the strict deployable setting if the reward model only sees public searches."),
        ("Company Context", "Visible web queries plus company name. This is still cheap at runtime and helps decide whether a query reveals enterprise-private facts."),
        ("Chain Context", "Visible web queries plus company, hop pattern, research question, and subquestions. Diagnostic/oracle-style context, not the main deployable setting."),
        ("Private Facts", "Visible web queries plus company and the full retrieved-doc Private QA/local-hop fact list. This is an oracle/contextual reward model, useful for testing whether explicit private-fact context improves precision."),
    ]
    return "<table><tbody>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows) + "</tbody></table>"


def binary_run_table(runs: list[dict[str, Any]]) -> str:
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
        args = run["summary"].get("args") if isinstance(run["summary"].get("args"), dict) else {}
        wandb = f"<a href='{esc(run['wandb_url'])}'>W&B</a>" if run.get("wandb_url") else ""
        eval_loss = run.get("eval_loss") or {}
        rows.append(
            "<tr>"
            f"<td>{esc(run.get('short_label') or clean_run_label(run))}</td>"
            f"<td>{esc(source_label(str(run['source'])))}</td>"
            f"<td>{esc(short_variant(str(run['variant'])))}</td>"
            f"<td>{esc(args.get('positive_repeat'))}</td>"
            f"<td>{esc(run['lora'])}</td>"
            f"<td>{num(eval_loss.get('best_eval_loss'))}</td>"
            f"<td>{num(run['test_roc_auc'])}</td>"
            f"<td>{pct(m05.get('precision'))}</td>"
            f"<td>{pct(m05.get('recall'))}</td>"
            f"<td>{pct(m05.get('f1'))}</td>"
            f"<td>{num(run['val_threshold_f1'])}</td>"
            f"<td>{pct(tv.get('precision'))}</td>"
            f"<td>{pct(tv.get('recall'))}</td>"
            f"<td>{pct(tv.get('f1'))}</td>"
            f"<td>{esc(m05.get('tp'))}/{esc(m05.get('fp'))}/{esc(m05.get('fn'))}</td>"
            f"<td>{esc(run['test_positive'])}/{esc(run['test_rows'])}</td>"
            f"<td>{wandb}</td>"
            f"<td><code>{esc(run['run_dir'])}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Test</th><th>Trainer</th><th>Variant</th><th>Positive Repeat</th><th>LoRA</th>"
        "<th>Best Val Loss</th><th>Test ROC AUC</th><th>Precision @0.5</th><th>Recall @0.5</th><th>F1 @0.5</th>"
        "<th>Val F1 Threshold</th><th>Precision @ValThr</th><th>Recall @ValThr</th><th>F1 @ValThr</th>"
        "<th>TP/FP/FN @0.5</th><th>Test Positives</th><th>W&B</th><th>Path</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def ternary_table(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "<p class='note'>No ternary runs yet.</p>"
    rows = []
    for run in sorted(runs, key=lambda item: item["name"]):
        test = (run["summary"].get("metrics") or {}).get("test") or {}
        eval_loss = run.get("eval_loss") or {}
        wandb = f"<a href='{esc(run['wandb_url'])}'>W&B</a>" if run.get("wandb_url") else ""
        rows.append(
            "<tr>"
            f"<td>{esc(run.get('short_label') or clean_run_label(run))}</td>"
            f"<td>{num(eval_loss.get('best_eval_loss'))}</td>"
            f"<td>{pct(test.get('accuracy'))}</td>"
            f"<td>{pct(test.get('macro_f1'))}</td>"
            f"<td>{num(test.get('macro_ovr_roc_auc'))}</td>"
            f"<td>{pct(test.get('fact_leakage_precision'))}</td>"
            f"<td>{pct(test.get('fact_leakage_recall'))}</td>"
            f"<td>{pct(test.get('fact_leakage_f1'))}</td>"
            f"<td>{num(test.get('fact_vs_rest_roc_auc'))}</td>"
            f"<td><code>{esc(test.get('confusion_matrix'))}</code></td>"
            f"<td>{wandb}</td>"
            f"<td><code>{esc(run['run_dir'])}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Test</th><th>Best Val Loss</th><th>Accuracy</th><th>Macro F1</th><th>Macro OVR AUC</th>"
        "<th>Fact Precision</th><th>Fact Recall</th><th>Fact F1</th><th>Fact-vs-Rest AUC</th><th>Confusion A/B/C</th><th>W&B</th><th>Path</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def best_recommendation(binary_runs: list[dict[str, Any]], ternary_runs: list[dict[str, Any]]) -> str:
    full_runs = [run for run in binary_runs if "smoke" not in run["name"] and run["source"] != "Base"]
    deployable = [run for run in full_runs if run["variant"] in {"queries_only", "company_context"}]
    oracle_context = [run for run in full_runs if run["variant"] in {"private_fact_context", "chain_context"}]
    best_f1 = max(deployable, key=lambda run: (((run["summary"].get("metrics") or {}).get("test") or {}).get("f1") or -1), default=None)
    best_auc = max(deployable, key=lambda run: run["test_roc_auc"] or -1, default=None)
    pieces = []
    if best_f1:
        test = (best_f1["summary"].get("metrics") or {}).get("test") or {}
        pieces.append(
            f"Best raw-threshold practical F1 is <strong>{esc(best_f1.get('short_label') or clean_run_label(best_f1))}</strong>: "
            f"ROC AUC {num(best_f1['test_roc_auc'])}, precision {pct(test.get('precision'))}, "
            f"recall {pct(test.get('recall'))}, F1 {pct(test.get('f1'))}."
        )
    if best_auc and best_auc is not best_f1:
        test = (best_auc["summary"].get("metrics") or {}).get("test") or {}
        pieces.append(
            f"Best practical ranking AUC is <strong>{esc(best_auc.get('short_label') or clean_run_label(best_auc))}</strong>: "
            f"AUC {num(best_auc['test_roc_auc'])}, F1 {pct(test.get('f1'))}."
        )
    best_oracle = max(oracle_context, key=lambda run: ((run["summary"].get("metrics") or {}).get("test") or {}).get("f1") or -1, default=None)
    if best_oracle:
        test = (best_oracle["summary"].get("metrics") or {}).get("test") or {}
        pieces.append(
            f"Best oracle/contextual run is <strong>{esc(best_oracle.get('short_label') or clean_run_label(best_oracle))}</strong>: "
            f"AUC {num(best_oracle['test_roc_auc'])}, precision {pct(test.get('precision'))}, "
            f"recall {pct(test.get('recall'))}, F1 {pct(test.get('f1'))}."
        )
    trl_runs = [run for run in deployable if run["source"] == "TRL SFTTrainer"]
    if trl_runs:
        best_trl = max(trl_runs, key=lambda run: ((run["summary"].get("metrics") or {}).get("test") or {}).get("f1") or -1)
        if best_trl is not best_f1:
            test = (best_trl["summary"].get("metrics") or {}).get("test") or {}
            pieces.append(
                f"Best TRL SFT run is <strong>{esc(best_trl.get('short_label') or clean_run_label(best_trl))}</strong>: "
                f"AUC {num(best_trl['test_roc_auc'])}, precision {pct(test.get('precision'))}, "
                f"recall {pct(test.get('recall'))}, F1 {pct(test.get('f1'))}."
            )
    pieces.append("The ablation says positive repeat is important for recall/calibration; r=128 all-linear did not clearly improve over r=16 default LoRA.")
    if ternary_runs:
        ternary = max(ternary_runs, key=lambda run: ((run["summary"].get("metrics") or {}).get("test") or {}).get("macro_f1") or -1)
        test = (ternary["summary"].get("metrics") or {}).get("test") or {}
        pieces.append(
            f"The ternary setup is useful diagnostically, but fact-leak F1 is only {pct(test.get('fact_leakage_f1'))} "
            f"with recall {pct(test.get('fact_leakage_recall'))}, below the binary reward model."
        )
    return "<p>" + " ".join(pieces) + "</p>"


def trained_binary_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variant_order = {"queries_only": 0, "company_context": 1, "private_fact_context": 2, "chain_context": 3}
    source_order = {"Manual Trainer": 0, "TRL SFTTrainer": 1}
    return sorted(
        [run for run in runs if "smoke" not in run["name"] and run["source"] != "Base"],
        key=lambda run: (
            variant_order.get(str(run.get("variant")), 9),
            source_order.get(str(run.get("source")), 9),
            str(run.get("name")),
        ),
    )


def binary_matrix_from_metrics(metrics: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [int(metrics.get("tn") or 0), int(metrics.get("fp") or 0)],
            [int(metrics.get("fn") or 0), int(metrics.get("tp") or 0)],
        ],
        dtype=int,
    )


def draw_confusion(ax: Any, matrix: np.ndarray, labels: list[str], title: str) -> Any:
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_pct = np.divide(matrix, row_totals, out=np.zeros(matrix.shape, dtype=float), where=row_totals != 0)
    image = ax.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = row_pct[row_idx, col_idx]
            color = "white" if value >= 0.55 else "#17201d"
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]}\n{100 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    return image


def plot_binary_confusion_grid(plot_runs: list[dict[str, Any]], output_dir: Path, source: str, file_stem: str) -> None:
    if not plot_runs:
        return
    ncols = 3
    nrows = int(np.ceil(len(plot_runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.4, 3.25 * nrows), squeeze=False)
    image = None
    for ax, run in zip(axes.ravel(), plot_runs):
        metrics = ((run["summary"].get("metrics") or {}).get("test") or {}) if source == "raw" else run["test_at_val_f1"]
        image = draw_confusion(
            ax,
            binary_matrix_from_metrics(metrics),
            ["No", "Leak"],
            clean_run_label(run, multiline=True),
        )
    for ax in axes.ravel()[len(plot_runs) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="Percent of true label")
    fig.subplots_adjust(left=0.06, right=0.90, top=0.96, bottom=0.05, hspace=0.74, wspace=0.36)
    plot_dir = output_dir / "plots"
    fig.savefig(plot_dir / f"{file_stem}.svg")
    fig.savefig(plot_dir / f"{file_stem}.pdf")
    plt.close(fig)


def plot_ternary_runs(runs: list[dict[str, Any]], output_dir: Path) -> None:
    plot_runs = [run for run in runs if "smoke" not in run["name"]]
    if not plot_runs:
        return
    ncols = min(2, len(plot_runs))
    nrows = int(np.ceil(len(plot_runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.5 * nrows), squeeze=False)
    image = None
    for ax, run in zip(axes.ravel(), plot_runs):
        test = (run["summary"].get("metrics") or {}).get("test") or {}
        matrix = np.array(test.get("confusion_matrix") or [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
        image = draw_confusion(ax, matrix, ["No", "Intent", "Fact"], clean_run_label(run, multiline=True))
    for ax in axes.ravel()[len(plot_runs) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="Percent of true label")
    fig.subplots_adjust(left=0.08, right=0.90, top=0.92, bottom=0.08, hspace=0.55, wspace=0.34)
    plot_dir = output_dir / "plots"
    fig.savefig(plot_dir / "ternary_confusion.svg")
    fig.savefig(plot_dir / "ternary_confusion.pdf")
    plt.close(fig)


def plot_binary_runs(runs: list[dict[str, Any]], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in plot_dir.glob("*"):
        if stale_plot.is_file():
            stale_plot.unlink()
    plot_runs = trained_binary_runs(runs)
    if not plot_runs:
        return
    metric_runs = sorted(
        plot_runs,
        key=lambda run: (((run["summary"].get("metrics") or {}).get("test") or {}).get("f1") or -1),
    )
    labels = [clean_run_label(run) for run in metric_runs]
    y = np.arange(len(metric_runs))
    height = 0.34

    fig, ax = plt.subplots(figsize=(10.8, max(5.4, 0.46 * len(metric_runs) + 1.4)))
    ax.barh(y - height / 2, [run["test_roc_auc"] or 0 for run in metric_runs], height, label="ROC AUC", color="#0072B2")
    ax.barh(
        y + height / 2,
        [((run["summary"].get("metrics") or {}).get("test") or {}).get("f1") or 0 for run in metric_runs],
        height,
        label="F1 at 0.5",
        color="#009E73",
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test Score")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "auc_f1_metrics.svg")
    fig.savefig(plot_dir / "auc_f1_metrics.pdf")
    plt.close(fig)

    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, max(5.6, 0.46 * len(metric_runs) + 1.6)), sharey=True)
    for ax, title, source in [(axes[0], "Raw Threshold 0.5", "summary"), (axes[1], "Validation F1 Threshold", "test_at_val_f1")]:
        offsets = np.linspace(-0.22, 0.22, len(metrics))
        for metric, offset in zip(metrics, offsets):
            values = []
            for run in metric_runs:
                if source == "summary":
                    values.append((((run["summary"].get("metrics") or {}).get("test") or {}).get(metric)) or 0)
                else:
                    values.append(run[source].get(metric) or 0)
            ax.barh(y + offset, values, height=0.20, label=metric.title())
        ax.set_title(title)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Test Score")
        ax.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].legend(frameon=False, ncols=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "precision_recall_f1.svg")
    fig.savefig(plot_dir / "precision_recall_f1.pdf")
    plt.close(fig)

    history_runs = [run for run in plot_runs if run.get("eval_history")]
    if history_runs:
        fig, ax = plt.subplots(figsize=(10.6, 5.6))
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(history_runs))))
        for color, run in zip(colors, history_runs):
            history = run.get("eval_history") or []
            steps = [int(point["step"]) for point in history]
            losses = [float(point["eval_loss"]) for point in history]
            ax.plot(steps, losses, marker="o", markersize=3.2, linewidth=1.5, color=color, label=clean_run_label(run))
            if losses:
                best_idx = int(np.argmin(losses))
                ax.scatter([steps[best_idx]], [losses[best_idx]], color=color, s=42, edgecolor="#17201d", linewidth=0.7, zorder=3)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Validation Loss")
        ax.set_title("TRL Validation Loss During Training")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=7.5, ncols=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(rect=(0, 0, 0.76, 1))
        fig.savefig(plot_dir / "eval_loss_curves.svg")
        fig.savefig(plot_dir / "eval_loss_curves.pdf")
        plt.close(fig)

    plot_binary_confusion_grid(plot_runs, output_dir, "raw", "binary_confusion_0_5")
    plot_binary_confusion_grid(plot_runs, output_dir, "val_threshold", "binary_confusion_val_threshold")

    best = max(plot_runs, key=lambda run: (((run["summary"].get("metrics") or {}).get("test") or {}).get("f1") or -1))
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
    ax.set_title(f"Best F1 Score Distribution: {clean_run_label(best)}")
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
    binary_runs = load_binary_runs(args.manual_root, "Manual") + load_binary_runs(args.trl_root, "TRL")
    ternary_runs = load_ternary_runs(args.trl_root, "TRL")
    manifest = read_json(args.manual_root / "dataset" / "manifest.json", {})
    plot_binary_runs(binary_runs, args.output_dir)
    plot_ternary_runs(ternary_runs, args.output_dir)
    data = {
        "generated_at": iso_now(),
        "manual_root": str(args.manual_root),
        "trl_root": str(args.trl_root),
        "binary_runs": binary_runs,
        "ternary_runs": ternary_runs,
        "dataset_manifest": manifest,
    }
    (args.output_dir / "viewer_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Privacy Reward Model TRL Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #17201d; line-height: 1.42; }}
section {{ margin-bottom: 30px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
td, th {{ border-bottom: 1px solid #d8dfdc; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f0; }}
img {{ max-width: 1220px; width: 100%; display: block; margin: 12px 0; }}
code {{ background: #eef2f0; padding: 1px 4px; border-radius: 4px; white-space: pre-wrap; }}
.note {{ color: #5d6964; }}
.callout {{ background: #f4f7f5; border-left: 4px solid #009E73; padding: 10px 14px; }}
</style></head><body>
<h1>Privacy Reward Model TRL Report</h1>
<p class="note">Generated {esc(data['generated_at'])}. Binary label predicts whether StepFun judged a rollout as privacy-leaking at score 4 or 5.</p>
<section class="callout"><h2>Recommendation</h2>{best_recommendation(binary_runs, ternary_runs)}</section>
<section><h2>Dataset</h2>{dataset_table(manifest)}</section>
<section><h2>Training Decisions</h2>{decisions_table()}</section>
<section><h2>Prompt Variants</h2>{variants_table()}</section>
<section><h2>Binary Reward Tests</h2>{binary_run_table(binary_runs)}</section>
<section><h2>1-3 Reward Tests</h2>{ternary_table(ternary_runs)}</section>
<section><h2>Binary Reward: AUC And F1</h2><img src="plots/auc_f1_metrics.svg"></section>
<section><h2>Binary Reward: Precision, Recall, F1</h2><img src="plots/precision_recall_f1.svg"></section>
<section><h2>TRL Validation Loss Curves</h2><p class="note">These are the repeated training-time validation-loss evaluations from TRL trainer state. In W&amp;B this curve is logged as <code>eval/loss</code>; final classifier metrics such as <code>eval/test/roc_auc</code> are logged once after training.</p><img src="plots/eval_loss_curves.svg"></section>
<section><h2>Binary Reward: Confusion Matrices At 0.5</h2><p class="note">Rows are true labels, columns are predicted labels. Each cell shows count and percent of the true-label row.</p><img src="plots/binary_confusion_0_5.svg"></section>
<section><h2>Binary Reward: Confusion Matrices At Validation Threshold</h2><p class="note">The validation threshold is selected independently for each run to maximize validation F1.</p><img src="plots/binary_confusion_val_threshold.svg"></section>
<section><h2>1-3 Reward: Confusion Matrix</h2><p class="note">A = No leakage, B = intent leakage, C = fact leakage. The plot uses readable class names.</p><img src="plots/ternary_confusion.svg"></section>
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
