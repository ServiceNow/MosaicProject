#!/usr/bin/env python3
"""Train/evaluate the binary privacy reward model with TRL SFTTrainer.

The target is the StepFun privacy label: overall privacy score 4-5 maps to
``Yes`` and 1-3 maps to ``No``. Training is still a one-token classifier: TRL
is used only to build the completion-only SFT loss, while evaluation reads the
next-token logits for the exact ``Yes`` and ``No`` tokens.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from train_privacy_hopqa_binary_reward_model import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET_DIR,
    evaluate_rows,
    expanded_train_rows,
    html_report,
    json_safe,
    load_split,
    load_tokenizer,
    plot_eval,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_trl_20260515/models")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_sft_row(row: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
    """Return TRL's prompt-completion conversational format."""
    out = {
        "prompt": [
            {"role": "system", "content": str(row["system"])},
            {"role": "user", "content": str(row["prompt"])},
        ],
        "completion": [{"role": "assistant", "content": str(row["completion"])}],
    }
    # Qwen3 supports this chat-template kwarg; disabling thinking keeps the
    # trained policy aligned with the one-token reward-model use case.
    if not enable_thinking:
        out["chat_template_kwargs"] = {"enable_thinking": False}
    return out


def make_sft_dataset(rows: list[dict[str, Any]], enable_thinking: bool) -> Dataset:
    return Dataset.from_list([make_sft_row(row, enable_thinking) for row in rows])


def lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def parse_report_to(value: str) -> list[str]:
    value = value.strip()
    if not value or value.lower() in {"none", "no", "false", "off"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_wandb(args: argparse.Namespace) -> None:
    reports = parse_report_to(args.report_to)
    if "wandb" not in reports:
        return
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_DIR", str(args.output_dir / "wandb"))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)


def log_eval_to_wandb(output_dir: Path, summary: dict[str, Any]) -> None:
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    payload: dict[str, Any] = {}
    for split, metrics in summary.get("metrics", {}).items():
        for metric_name, value in metrics.items():
            if isinstance(value, int | float) and value is not None:
                payload[f"eval/{split}/{metric_name}"] = value
    if payload:
        wandb.log(payload)
    for file_name in ["summary.json", "training_runtime.json", "predictions_val.jsonl", "predictions_test.jsonl"]:
        file_path = output_dir / file_name
        if file_path.exists():
            wandb.save(str(file_path), base_path=str(output_dir), policy="now")


def model_kwargs(args: argparse.Namespace, for_training: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    if not for_training:
        kwargs["device_map"] = "auto"
    return kwargs


def load_causal_model(args: argparse.Namespace, adapter: Path | None, for_training: bool) -> Any:
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs(args, for_training=for_training))
    except Exception:
        if args.attn_implementation != "flash_attention_2" or not args.allow_attention_fallback:
            raise
        fallback_args = argparse.Namespace(**vars(args))
        fallback_args.attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            fallback_args.base_model,
            **model_kwargs(fallback_args, for_training=for_training),
        )
        args.attn_implementation = "sdpa"
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    if for_training:
        model.config.use_cache = False
        if torch.cuda.is_available():
            model.to("cuda")
    model.eval()
    return model


def train(args: argparse.Namespace) -> Path | None:
    if args.eval_only:
        return args.adapter_path

    tokenizer = load_tokenizer(args.base_model)
    raw_train_rows = load_split(args.dataset_dir, args.variant, "train")
    train_rows = expanded_train_rows(raw_train_rows, args.positive_repeat, args.max_train_examples, args.seed)
    train_dataset = make_sft_dataset(train_rows, enable_thinking=args.enable_thinking)

    model = load_causal_model(args, adapter=None, for_training=True)
    configure_wandb(args)
    trainer_args = SFTConfig(
        output_dir=str(args.output_dir / "trainer"),
        run_name=args.run_name or args.output_dir.name,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=args.packing,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=True,
        report_to=parse_report_to(args.report_to),
        remove_unused_columns=True,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        optim=args.optim,
        use_liger_kernel=args.use_liger_kernel,
        include_tokens_per_second=True,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(args),
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
                "raw_train_rows": len(raw_train_rows),
                "elapsed_s": round(time.time() - started, 2),
                "positive_repeat": args.positive_repeat,
                "max_train_examples": args.max_train_examples,
                "trainer": "trl.SFTTrainer",
                "attn_implementation": args.attn_implementation,
                "use_liger_kernel": args.use_liger_kernel,
                "report_to": parse_report_to(args.report_to),
                "wandb_mode": args.wandb_mode,
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
    model = load_causal_model(args, adapter=adapter, for_training=False)
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
        "trainer": "trl.SFTTrainer",
        "metrics": split_metrics,
        "args": json_safe(vars(args)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log_eval_to_wandb(args.output_dir, summary)
    plot_eval(args.output_dir, split_metrics, predictions_by_split)
    html_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"viewer={args.output_dir / 'index.html'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--variant", default="queries_only", choices=["queries_only", "company_context", "chain_context"])
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
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
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-liger-kernel", action="store_true", default=True)
    parser.add_argument("--no-liger-kernel", dest="use_liger_kernel", action="store_false")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["auto", "sdpa", "flash_attention_2"])
    parser.add_argument("--allow-attention-fallback", action="store_true", default=True)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="privacy_hopqa_binary_reward_sft")
    parser.add_argument("--wandb-mode", default="online", choices=["offline", "online", "disabled"])
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()
    if args.output_dir is None:
        mode = "base_eval" if args.eval_only and args.adapter_path is None else "trl_sft"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"ep{args.epochs:g}_rep{args.positive_repeat}"
        args.output_dir = args.output_root / f"{args.variant}_{mode}_{suffix}_{stamp}"
    return args


if __name__ == "__main__":
    random.seed(20260515)
    run(parse_args())
