"""Train the privacy reward model: a LoRA adapter that answers Yes or No.

The target is one token. TRL's SFTTrainer supplies a completion-only loss so the
gradient lands on the label and not on the prompt, and evaluation never generates
text: it reads the next-token logits at the two label tokens and softmaxes over
just those, which makes a held-out pass one forward per batch.

    mosaicleaks-reward-train --dataset-dir rm_data/private_fact_context \\
        --base-model Qwen/Qwen3-4B-Instruct-2507 --output-dir runs/rm

Positive labels are rare (roughly one rollout in ten leaks), so --positive-repeat
oversamples them in the training split only; val and test keep their true rate.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

LABELS = ("Yes", "No")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for label in LABELS:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"label {label!r} is not a single token for this tokenizer: {encoded}")
        ids.append(encoded[0])
    return ids


def oversample_positives(rows: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    if repeat <= 1:
        return rows
    return rows + [row for row in rows if row["label"] for _ in range(repeat - 1)]


@torch.no_grad()
def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int, max_length: int) -> dict[str, Any]:
    """Accuracy, precision, recall and F1 at a 0.5 threshold on prob(Yes)."""
    yes_id, no_id = label_token_ids(tokenizer)
    model.eval()
    device = next(model.parameters()).device
    # Left padding puts the label position at index -1 for every row in the batch.
    # It must not leak into training, where the collator expects right padding.
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    probs: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length, add_special_tokens=False)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        logits = model(**tokens).logits[:, -1, [yes_id, no_id]].float()
        probs.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())
    tokenizer.padding_side = previous_padding

    targets = [bool(row["label"]) for row in rows]
    predictions = [p >= 0.5 for p in probs]
    tp = sum(1 for t, p in zip(targets, predictions) if t and p)
    fp = sum(1 for t, p in zip(targets, predictions) if not t and p)
    fn = sum(1 for t, p in zip(targets, predictions) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows),
        "positive_rate": round(sum(targets) / len(rows), 4) if rows else 0.0,
        "accuracy": round(sum(t == p for t, p in zip(targets, predictions)) / len(rows), 4) if rows else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Train the privacy reward model.")
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Variant directory holding train/val/test.jsonl.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--positive-repeat", type=int, default=2,
                   help="Oversample positive examples in train only; 1 disables.")

    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-target-modules", default="all-linear")
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()

    train_rows = read_jsonl(args.dataset_dir / "train.jsonl")
    val_rows = read_jsonl(args.dataset_dir / "val.jsonl")
    test_rows = read_jsonl(args.dataset_dir / "test.jsonl")
    if not train_rows:
        raise SystemExit(f"{args.dataset_dir}/train.jsonl is empty")
    train_rows = oversample_positives(train_rows, args.positive_repeat)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([{"messages": r["messages"]} for r in train_rows]),
        eval_dataset=Dataset.from_list([{"messages": r["messages"]} for r in val_rows]) if val_rows else None,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        ),
        args=SFTConfig(
            output_dir=str(args.output_dir),
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.max_length,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            seed=args.seed,
            report_to=[],
            completion_only_loss=True,
        ),
    )

    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # A split can come out empty on a small corpus. Reporting its zeros as though
    # they were measured is worse than saying nothing.
    results = {
        split: evaluate(trainer.model, tokenizer, rows, args.eval_batch_size, args.max_length)
        for split, rows in (("val", val_rows), ("test", test_rows))
        if rows
    }
    for split, rows in (("val", val_rows), ("test", test_rows)):
        if not rows:
            print(f"{split}.jsonl is empty, skipping its evaluation")
    (args.output_dir / "eval.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'split':<6} {'n':>6} {'pos':>7} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7}")
    for split, stats in results.items():
        print(f"{split:<6} {stats['n']:>6} {stats['positive_rate']:>7.1%} {stats['accuracy']:>7.3f} "
              f"{stats['precision']:>7.3f} {stats['recall']:>7.3f} {stats['f1']:>7.3f}")
    print(f"\nadapter -> {adapter_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
