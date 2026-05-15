#!/usr/bin/env python3
"""Train/evaluate a small LoRA SFT binary privacy reward model.

The model is trained as a one-token Yes/No classifier. Evaluation uses the
next-token logits for "Yes" and "No" rather than free-form generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET_DIR = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515/dataset")
DEFAULT_OUTPUT_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515/models")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(dataset_dir: Path, variant: str, split: str) -> list[dict[str, Any]]:
    return read_jsonl(dataset_dir / "variants" / variant / f"{split}.jsonl")


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def prompt_text(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": str(row["system"])},
        {"role": "user", "content": str(row["prompt"])},
    ]
    return apply_chat_template(tokenizer, messages, add_generation_prompt=True)


def full_text(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": str(row["system"])},
        {"role": "user", "content": str(row["prompt"])},
        {"role": "assistant", "content": str(row["completion"])},
    ]
    return apply_chat_template(tokenizer, messages, add_generation_prompt=False)


class CompletionOnlyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        prompt_ids = self.tokenizer(prompt_text(self.tokenizer, row), add_special_tokens=False)["input_ids"]
        ids = self.tokenizer(full_text(self.tokenizer, row), add_special_tokens=False)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(ids)) + ids[min(len(prompt_ids), len(ids)) :]
        if len(labels) < len(ids):
            labels.extend([-100] * (len(ids) - len(labels)))
        overflow = max(0, len(ids) - self.max_length)
        ids = ids[overflow:]
        labels = labels[overflow:]
        if not any(label != -100 for label in labels):
            labels[-1] = ids[-1]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in features)
    input_ids, attention_mask, labels = [], [], []
    for item in features:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def expanded_train_rows(rows: list[dict[str, Any]], positive_repeat: int, max_train_examples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    if max_train_examples:
        positives = [row for row in rows if row["label"]]
        negatives = [row for row in rows if not row["label"]]
        target_pos = min(len(positives), max(1, max_train_examples // 3))
        target_neg = max_train_examples - target_pos
        rows = positives[:target_pos] + negatives[:target_neg]
        rng.shuffle(rows)
    out = []
    for row in rows:
        repeats = positive_repeat if row["label"] else 1
        out.extend([row] * repeats)
    rng.shuffle(out)
    return out


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def label_token_ids(tokenizer: Any) -> tuple[int, int]:
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"expected one-token labels, got Yes={yes_ids} No={no_ids}")
    return yes_ids[0], no_ids[0]


def load_model(base_model: str, adapter: Path | None, for_training: bool) -> Any:
    kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    if not for_training:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    if for_training:
        model.config.use_cache = False
        if torch.cuda.is_available():
            model.to("cuda")
    model.eval()
    return model


def add_lora(model: Any, args: argparse.Namespace) -> Any:
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def binary_metrics(targets: list[int], probs: list[float]) -> dict[str, Any]:
    preds = [1 if prob >= 0.5 else 0 for prob in probs]
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / len(targets) if targets else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else None
    auprc = average_precision_score(targets, probs) if len(set(targets)) > 1 else None
    brier = float(np.mean((np.asarray(probs) - np.asarray(targets)) ** 2)) if targets else None
    return {
        "rows": len(targets),
        "target_positive": int(sum(targets)),
        "predicted_positive": int(sum(preds)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": f1,
        "roc_auc": auc,
        "average_precision": auprc,
        "brier": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@torch.no_grad()
def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    yes_id, no_id = label_token_ids(tokenizer)
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    targets: list[int] = []
    probs: list[float] = []
    predictions: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [prompt_text(tokenizer, row) for row in batch_rows]
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = model(**tokens)
        logits = outputs.logits[:, -1, [yes_id, no_id]].float()
        batch_probs = torch.softmax(logits, dim=-1)[:, 0].detach().cpu().tolist()
        for row, prob in zip(batch_rows, batch_probs):
            target = 1 if row["label"] else 0
            pred = 1 if prob >= 0.5 else 0
            targets.append(target)
            probs.append(float(prob))
            predictions.append(
                {
                    "id": row["id"],
                    "variant": row["variant"],
                    "reward_split": row["reward_split"],
                    "target": target,
                    "prob_yes": float(prob),
                    "predicted": pred,
                    "correct": pred == target,
                    "score": row.get("score"),
                    "condition": row.get("condition"),
                    "source_model_label": row.get("source_model_label"),
                    "source_split": row.get("source_split"),
                    "task_id": row.get("task_id"),
                    "chain_id": row.get("chain_id"),
                    "rollout_id": row.get("rollout_id"),
                    "n_web_queries": row.get("n_web_queries"),
                    "web_queries": row.get("web_queries"),
                }
            )
    return binary_metrics(targets, probs), predictions


def plot_eval(output_dir: Path, split_metrics: dict[str, dict[str, Any]], predictions_by_split: dict[str, list[dict[str, Any]]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision"]
    splits = [split for split in ["val", "test", "train"] if split in split_metrics]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(metric_names))
    width = 0.24
    for idx, split in enumerate(splits):
        vals = [split_metrics[split].get(name) or 0.0 for name in metric_names]
        ax.bar(x + (idx - (len(splits) - 1) / 2) * width, vals, width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", " ").title() for name in metric_names], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "metrics.svg")
    fig.savefig(plot_dir / "metrics.pdf")
    plt.close(fig)

    for split, rows in predictions_by_split.items():
        if not rows:
            continue
        targets = [row["target"] for row in rows]
        probs = [row["prob_yes"] for row in rows]
        preds = [row["predicted"] for row in rows]
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred No", "Pred Yes"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True No", "True Yes"])
        ax.set_title(f"{split.title()} Confusion Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{split}_confusion.svg")
        plt.close(fig)

        if len(set(targets)) > 1:
            fpr, tpr, _ = roc_curve(targets, probs)
            precision, recall, _ = precision_recall_curve(targets, probs)
            fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
            axes[0].plot(fpr, tpr, color="#0072B2")
            axes[0].plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1)
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].set_title(f"ROC AUC = {roc_auc_score(targets, probs):.3f}")
            axes[1].plot(recall, precision, color="#009E73")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title(f"Average Precision = {average_precision_score(targets, probs):.3f}")
            for axis in axes:
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1.05)
                axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{split}_roc_pr.svg")
            fig.savefig(plot_dir / f"{split}_roc_pr.pdf")
            plt.close(fig)


def html_report(output_dir: Path, summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    rows = []
    for split, metrics in summary["metrics"].items():
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{fmt(value)}</td>"
                for value in [
                    split,
                    metrics.get("rows"),
                    metrics.get("target_positive"),
                    metrics.get("predicted_positive"),
                    metrics.get("accuracy"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("specificity"),
                    metrics.get("f1"),
                    metrics.get("roc_auc"),
                    metrics.get("average_precision"),
                ]
            )
            + "</tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Privacy HopQA Binary Reward Model</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #17201d; }}
table {{ border-collapse: collapse; margin: 16px 0; }}
td, th {{ border-bottom: 1px solid #d8dfdc; padding: 7px 10px; text-align: left; }}
th {{ background: #eef2f0; }}
img {{ max-width: 1100px; width: 100%; display: block; margin: 16px 0; }}
code {{ background: #eef2f0; padding: 1px 4px; border-radius: 4px; }}
</style></head><body>
<h1>Privacy HopQA Binary Reward Model</h1>
<p>Generated {summary["generated_at"]}. Variant: <code>{summary["variant"]}</code>. Base model: <code>{summary["base_model"]}</code>.</p>
<table><thead><tr><th>Split</th><th>Rows</th><th>Target Yes</th><th>Pred Yes</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>ROC AUC</th><th>Avg Precision</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<h2>Metrics</h2><img src="plots/metrics.svg">
<h2>Validation ROC/PR</h2><img src="plots/val_roc_pr.svg">
<h2>Test ROC/PR</h2><img src="plots/test_roc_pr.svg">
</body></html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def train(args: argparse.Namespace) -> Path | None:
    if args.eval_only:
        return args.adapter_path
    tokenizer = load_tokenizer(args.base_model)
    train_rows = expanded_train_rows(
        load_split(args.dataset_dir, args.variant, "train"),
        args.positive_repeat,
        args.max_train_examples,
        args.seed,
    )
    model = load_model(args.base_model, None, for_training=True)
    model = add_lora(model, args)
    dataset = CompletionOnlyDataset(train_rows, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: collate(features, tokenizer.pad_token_id),
    )
    started = time.time()
    trainer.train()
    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (args.output_dir / "training_runtime.json").write_text(
        json.dumps(
            {
                "started_at": iso_now(),
                "train_rows_after_repeat": len(train_rows),
                "raw_train_rows": len(load_split(args.dataset_dir, args.variant, "train")),
                "elapsed_s": round(time.time() - started, 2),
                "positive_repeat": args.positive_repeat,
                "max_train_examples": args.max_train_examples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return adapter_dir


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = train(args)
    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.base_model, adapter, for_training=False)
    split_metrics: dict[str, dict[str, Any]] = {}
    predictions_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in args.eval_splits.split(","):
        rows = load_split(args.dataset_dir, args.variant, split)
        if args.max_eval_examples:
            rows = rows[: args.max_eval_examples]
        metrics, predictions = evaluate_rows(model, tokenizer, rows, args.eval_batch_size, args.max_length)
        split_metrics[split] = metrics
        predictions_by_split[split] = predictions
        write_jsonl(args.output_dir / f"predictions_{split}.jsonl", predictions)
    summary = {
        "generated_at": iso_now(),
        "base_model": args.base_model,
        "adapter": str(adapter) if adapter else None,
        "dataset_dir": str(args.dataset_dir),
        "variant": args.variant,
        "eval_only": args.eval_only,
        "metrics": split_metrics,
        "args": json_safe(vars(args)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_eval(args.output_dir, split_metrics, predictions_by_split)
    html_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"viewer={args.output_dir / 'index.html'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--variant", default="queries_only")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--positive-repeat", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()
    if args.output_dir is None:
        mode = "base_eval" if args.eval_only and args.adapter_path is None else "sft"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.output_root / f"{args.variant}_{mode}_{stamp}"
    return args


if __name__ == "__main__":
    run(parse_args())
