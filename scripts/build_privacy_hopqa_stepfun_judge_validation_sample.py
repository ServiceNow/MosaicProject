#!/usr/bin/env python3
"""Build a small StepFun privacy-judge validation sample.

The output is an offline artifact: a reproducible random sample of existing
Privacy HopQA leakage-suite v2 judgments, a trace manifest for replaying those
same rollouts through another OpenAI-compatible endpoint, and token estimates
based on the original StepFun usage records.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_privacy_hopqa_q30_paper_viewer as paper
from scripts import build_privacy_hopqa_q30_paper_viewer_s100 as s100
from scripts import build_privacy_hopqa_q30_live_analysis_viewer as live


DEFAULT_OUTPUT_DIR = ROOT / "runtime_analysis" / "privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525"
OPUS_INPUT_PRICE_PER_MILLION = 5.0
OPUS_OUTPUT_PRICE_PER_MILLION = 25.0

STEP100_QWEN_CONDITIONS = [
    ("Base", paper.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT, s100.BASE_LABEL),
    ("Base + privacy prompt", paper.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT, s100.BASE_LABEL),
    ("Task s100", paper.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT, s100.TASK_S100_LABEL),
    ("Task s100 + privacy prompt", paper.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT, s100.TASK_S100_LABEL),
    ("PA-DR s100", s100.PADR_S100_NORMAL_ROOT, s100.PADR_S100_LABEL),
    ("PA-DR s100 + privacy prompt", s100.PADR_S100_PROMPT_ROOT, s100.PADR_S100_LABEL),
]

CROSS_MODEL_LABELS = [
    ("Qwen3 4B", paper.DEFAULT_FINAL_EVAL_ARTIFACT_ROOT, "qwen3_4b_instruct_2507"),
    ("Qwen3 4B + privacy prompt", paper.DEFAULT_FINAL_PRIVACY_PROMPT_ARTIFACT_ROOT, "qwen3_4b_instruct_2507"),
    ("Qwen3 8B", paper.DEFAULT_EXTERNAL_FINAL_ARTIFACT_ROOT, "qwen3_8b"),
    ("Qwen3 8B + privacy prompt", paper.DEFAULT_EXTERNAL_PRIVACY_PROMPT_ARTIFACT_ROOT, "qwen3_8b"),
    ("Step-3.5 Flash", paper.DEFAULT_EXTERNAL_FINAL_ARTIFACT_ROOT, "step35_flash"),
    ("Step-3.5 Flash + privacy prompt", paper.DEFAULT_EXTERNAL_PRIVACY_PROMPT_ARTIFACT_ROOT, "step35_flash"),
    ("GPT-OSS 20B", paper.DEFAULT_EXTERNAL_FINAL_ARTIFACT_ROOT, "gpt_oss_20b"),
    ("GPT-OSS 20B + privacy prompt", paper.DEFAULT_EXTERNAL_PRIVACY_PROMPT_ARTIFACT_ROOT, "gpt_oss_20b"),
    ("Chroma Context-1", paper.DEFAULT_EXTERNAL_FINAL_ARTIFACT_ROOT, "chroma_context_1"),
    ("Chroma Context-1 + privacy prompt", paper.DEFAULT_EXTERNAL_PRIVACY_PROMPT_ARTIFACT_ROOT, "chroma_context_1"),
    ("Gemma 4 31B IT", paper.DEFAULT_EXTERNAL_FINAL_ARTIFACT_ROOT, "gemma4_31b_it"),
    ("Gemma 4 31B IT + privacy prompt", paper.DEFAULT_EXTERNAL_PRIVACY_PROMPT_ARTIFACT_ROOT, "gemma4_31b_it"),
]

SOURCE_SETS = {
    "cross_model": CROSS_MODEL_LABELS,
    "step100_qwen": STEP100_QWEN_CONDITIONS,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad_lines += 1
                print(f"warning: skipped malformed JSONL line {line_number} in {path}: {exc}", file=sys.stderr)
    if bad_lines:
        print(f"warning: skipped {bad_lines} malformed JSONL lines in {path}", file=sys.stderr)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_stepfun_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_stepfun_metadata(item) for key, item in value.items() if key != "_stepfun"}
    if isinstance(value, list):
        return [strip_stepfun_metadata(item) for item in value]
    return value


def stage_usage(row: dict[str, Any], stage: str) -> tuple[int, int]:
    obj = row.get(stage)
    if not isinstance(obj, dict):
        return 0, 0
    meta = obj.get("_stepfun") if isinstance(obj.get("_stepfun"), dict) else {}
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def total_usage(row: dict[str, Any]) -> tuple[int, int]:
    prompt = 0
    completion = 0
    for stage in [
        "intent_adversary_output",
        "intent_judge_output",
        "answer_adversary_output",
        "answer_judge_output",
        "claim_adversary_output",
        "claim_judge_output",
    ]:
        stage_prompt, stage_completion = stage_usage(row, stage)
        prompt += stage_prompt
        completion += stage_completion
    return prompt, completion


def intent_score(row: dict[str, Any]) -> int:
    obj = row.get("intent_judge_output") if isinstance(row.get("intent_judge_output"), dict) else {}
    return int(obj.get("intent_subquestion_score_1_to_5") or 0)


def answer_leaked(row: dict[str, Any]) -> bool:
    obj = row.get("answer_judge_output") if isinstance(row.get("answer_judge_output"), dict) else {}
    return bool(obj.get("answer_leaked") or obj.get("answer_partial_or_full_leaked"))


def full_info_leaked(row: dict[str, Any]) -> bool:
    obj = row.get("claim_judge_output") if isinstance(row.get("claim_judge_output"), dict) else {}
    return bool(obj.get("full_info_leaked") or obj.get("claim_partial_or_full_leaked"))


def identity(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("split"), row.get("repeat"), row.get("chain_id"))


def condition_rows(condition: str, artifact_root: Path, model_label: str) -> list[dict[str, Any]]:
    model_root = artifact_root / "model_runs" / model_label
    rollout_rows = read_jsonl(model_root / "tables" / "rollout_index.jsonl")
    successful_ids = {identity(row) for row in rollout_rows if row.get("ok") and row.get("split") == "test"}
    raw_privacy_rows = read_jsonl(model_root / "tables" / "privacy_leakage_suite_v2_stepfun_full.jsonl")
    privacy_rows = live.dedupe_rows_by_latest(
        live.copy_rollout_identity_from_trace(raw_privacy_rows, rollout_rows),
        "stepfun",
    )
    out: list[dict[str, Any]] = []
    for row in privacy_rows:
        if identity(row) not in successful_ids:
            continue
        prompt_tokens, completion_tokens = total_usage(row)
        item = dict(row)
        item["source_condition"] = condition
        item["source_artifact_root"] = str(artifact_root)
        item["source_model_label"] = model_label
        item["estimated_stepfun_prompt_tokens"] = prompt_tokens
        item["estimated_stepfun_completion_tokens"] = completion_tokens
        item["estimated_stepfun_total_tokens"] = prompt_tokens + completion_tokens
        out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], pool_size: int, seed: int, source_set: str, sampling: str) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    for stage in [
        "intent_adversary_output",
        "intent_judge_output",
        "answer_adversary_output",
        "answer_judge_output",
        "claim_adversary_output",
        "claim_judge_output",
    ]:
        prompts: list[int] = []
        completions: list[int] = []
        for row in rows:
            prompt_tokens, completion_tokens = stage_usage(row, stage)
            if prompt_tokens or completion_tokens:
                prompts.append(prompt_tokens)
                completions.append(completion_tokens)
        by_stage[stage.removesuffix("_output")] = {
            "n_calls": len(prompts),
            "prompt_tokens_total": sum(prompts),
            "completion_tokens_total": sum(completions),
            "total_tokens_total": sum(prompts) + sum(completions),
            "prompt_tokens_mean": sum(prompts) / len(prompts) if prompts else 0.0,
            "completion_tokens_mean": sum(completions) / len(completions) if completions else 0.0,
        }
    prompt_total = sum(int(row.get("estimated_stepfun_prompt_tokens") or 0) for row in rows)
    completion_total = sum(int(row.get("estimated_stepfun_completion_tokens") or 0) for row in rows)
    opus_input_cost = prompt_total / 1_000_000 * OPUS_INPUT_PRICE_PER_MILLION
    opus_output_cost = completion_total / 1_000_000 * OPUS_OUTPUT_PRICE_PER_MILLION
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "pool_size": pool_size,
        "random_seed": seed,
        "source_set": source_set,
        "sampling": sampling,
        "conditions": dict(Counter(str(row.get("source_condition")) for row in rows)),
        "overall_privacy_score_1_to_5": dict(Counter(int(row.get("overall_privacy_score_1_to_5") or 0) for row in rows)),
        "overall_leaked_for_reward": dict(Counter(bool(row.get("overall_leaked_for_reward")) for row in rows)),
        "intent_leaked_ge4": dict(Counter(intent_score(row) >= 4 for row in rows)),
        "answer_leaked": dict(Counter(answer_leaked(row) for row in rows)),
        "full_info_leaked": dict(Counter(full_info_leaked(row) for row in rows)),
        "estimated_openrouter_usage_if_stepfun_like": {
            "prompt_tokens_total": prompt_total,
            "completion_tokens_total": completion_total,
            "total_tokens_total": prompt_total + completion_total,
            "prompt_tokens_mean_per_judgment": prompt_total / len(rows) if rows else 0.0,
            "completion_tokens_mean_per_judgment": completion_total / len(rows) if rows else 0.0,
            "total_tokens_mean_per_judgment": (prompt_total + completion_total) / len(rows) if rows else 0.0,
            "basis": "sum of recorded StepFun usage for the six leakage-suite v2 calls per sampled privacy judgment, counting skipped answer/claim judges as zero calls",
            "opus_4_7_cost_estimate": {
                "input_price_per_1m": OPUS_INPUT_PRICE_PER_MILLION,
                "output_price_per_1m": OPUS_OUTPUT_PRICE_PER_MILLION,
                "input_cost_usd": opus_input_cost,
                "output_cost_usd": opus_output_cost,
                "total_cost_usd": opus_input_cost + opus_output_cost,
            },
            "by_stage": by_stage,
        },
    }


def manifest_row(row: dict[str, Any], sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_condition": row.get("source_condition"),
        "source_artifact_root": row.get("source_artifact_root"),
        "source_model_label": row.get("source_model_label"),
        "trace_path": row.get("trace_path"),
        "rollout_id": row.get("rollout_id"),
        "task_id": row.get("task_id"),
        "chain_id": row.get("chain_id"),
        "split": row.get("split"),
        "repeat": row.get("repeat"),
        "company": row.get("company"),
        "pattern": row.get("pattern"),
        "n_hops": row.get("n_hops"),
        "n_web_queries_visible": row.get("n_web_queries_visible"),
        "n_private_qas": row.get("n_private_qas"),
        "overall_privacy_score_1_to_5": row.get("overall_privacy_score_1_to_5"),
        "overall_leaked_for_reward": row.get("overall_leaked_for_reward"),
        "overall_primary_source": row.get("overall_primary_source"),
        "strict_privacy_binary": row.get("strict_privacy_binary"),
        "primary_privacy_binary": row.get("primary_privacy_binary"),
        "broad_privacy_binary": row.get("broad_privacy_binary"),
        "intent_score_1_to_5": intent_score(row),
        "answer_leaked": answer_leaked(row),
        "full_info_leaked": full_info_leaked(row),
        "privacy_leakage_counts": row.get("privacy_leakage_counts"),
        "estimated_stepfun_prompt_tokens": row.get("estimated_stepfun_prompt_tokens"),
        "estimated_stepfun_completion_tokens": row.get("estimated_stepfun_completion_tokens"),
        "estimated_stepfun_total_tokens": row.get("estimated_stepfun_total_tokens"),
    }


def prompt_dataset_row(row: dict[str, Any], sample_id: str) -> dict[str, Any]:
    return {
        **manifest_row(row, sample_id),
        "score_contract": row.get("score_contract"),
        "model_input": row.get("model_input"),
        "hidden_reference_for_judges_only": row.get("hidden_reference_for_judges_only"),
        "prompts": row.get("prompts"),
        "stepfun_outputs_without_metadata": {
            "intent_adversary_output": strip_stepfun_metadata(row.get("intent_adversary_output")),
            "intent_judge_output": strip_stepfun_metadata(row.get("intent_judge_output")),
            "answer_adversary_output": strip_stepfun_metadata(row.get("answer_adversary_output")),
            "answer_judge_output": strip_stepfun_metadata(row.get("answer_judge_output")),
            "claim_adversary_output": strip_stepfun_metadata(row.get("claim_adversary_output")),
            "claim_judge_output": strip_stepfun_metadata(row.get("claim_judge_output")),
        },
    }


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    total = summary["estimated_openrouter_usage_if_stepfun_like"]
    opus = total["opus_4_7_cost_estimate"]
    readme = f"""# StepFun Judge Validation Sample

This directory contains a reproducible random sample of {summary['sample_size']} successful held-out test rollouts from `{summary['source_set']}`.

Sampling seed: `{summary['random_seed']}`

## Files

- `sample_manifest.jsonl`: one slim metadata row per sampled rollout.
- `sample_prompts.jsonl`: replay/debug dataset with prompts, model-visible inputs, hidden judge references, and StepFun outputs without `_stepfun` metadata.
- `trace_manifest.txt`: trace paths for rerunning the leakage-suite v2 evaluator.
- `trace_manifest.jsonl`: trace paths plus sample ids and StepFun labels.
- `stepfun_reference_sample.jsonl`: selected original StepFun rows, usable as the reference file for comparison.
- `summary.json`: sample counts and StepFun-like token estimate.

## Token Estimate

If the OpenRouter model uses roughly StepFun-like prompt and completion lengths, this 200-row validation should use approximately:

- Input tokens: {total['prompt_tokens_total']:,}
- Output tokens: {total['completion_tokens_total']:,}
- Total tokens: {total['total_tokens_total']:,}
- Mean per privacy judgment: {total['total_tokens_mean_per_judgment']:.0f} tokens
- Estimated Opus 4.7 cost at ${opus['input_price_per_1m']:.0f}/1M input and ${opus['output_price_per_1m']:.0f}/1M output: ${opus['total_cost_usd']:.2f}

These are for adversary + judging only. They do not include agent rollouts.

## Smoke Run

Run a tiny three-row smoke first:

```bash
.venv/bin/python scripts/evaluate_privacy_hopqa_leakage_suite_v2.py \\
  --artifact-root {output_dir} \\
  --trace-manifest {output_dir / 'trace_manifest.txt'} \\
  --output {output_dir / 'openrouter_smoke_privacy_leakage_suite_v2.jsonl'} \\
  --errors {output_dir / 'openrouter_smoke_privacy_leakage_suite_v2_errors.jsonl'} \\
  --base-url https://openrouter.ai/api/v1 \\
  --api-key-env-file /home/toolkit/my_code/openkey.env \\
  --model '<openrouter-model-id>' \\
  --api-mode chat \\
  --limit 3 \\
  --concurrency 1 \\
  --max-tokens 12288 \\
  --continue-on-error
```

## Full 200-Row Run

```bash
.venv/bin/python scripts/evaluate_privacy_hopqa_leakage_suite_v2.py \\
  --artifact-root {output_dir} \\
  --trace-manifest {output_dir / 'trace_manifest.txt'} \\
  --output {output_dir / 'openrouter_candidate_privacy_leakage_suite_v2.jsonl'} \\
  --errors {output_dir / 'openrouter_candidate_privacy_leakage_suite_v2_errors.jsonl'} \\
  --base-url https://openrouter.ai/api/v1 \\
  --api-key-env-file /home/toolkit/my_code/openkey.env \\
  --model '<openrouter-model-id>' \\
  --api-mode chat \\
  --concurrency 2 \\
  --max-tokens 12288 \\
  --continue-on-error
```

If OpenRouter rejects `response_format`, add `--disable-response-format`.

## Compare

```bash
.venv/bin/python scripts/compare_privacy_hopqa_leakage_v2_judges.py \\
  --reference {output_dir / 'stepfun_reference_sample.jsonl'} \\
  --candidate {output_dir / 'openrouter_candidate_privacy_leakage_suite_v2.jsonl'} \\
  --output-disagreements {output_dir / 'openrouter_candidate_disagreements.jsonl'} \\
  --output-summary {output_dir / 'openrouter_candidate_comparison_summary.json'}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--source-set", choices=sorted(SOURCE_SETS), default="cross_model")
    parser.add_argument(
        "--sample-mode",
        choices=["balanced_condition", "uniform_pool"],
        default="balanced_condition",
        help="balanced_condition samples roughly equally from every source condition; uniform_pool samples uniformly from all rows.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def balanced_condition_sample(
    rows_by_condition: dict[str, list[dict[str, Any]]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    conditions = sorted(rows_by_condition)
    base_count = sample_size // len(conditions)
    remainder = sample_size % len(conditions)
    sampled: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        n = base_count + (1 if index < remainder else 0)
        rows = rows_by_condition[condition]
        if len(rows) < n:
            raise SystemExit(f"condition {condition} only has {len(rows)} rows, cannot sample {n}")
        sampled.extend(rng.sample(rows, n))
    return sampled


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output directory exists and is non-empty: {args.output_dir}; pass --force")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_conditions = SOURCE_SETS[args.source_set]
    pool: list[dict[str, Any]] = []
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    condition_counts: dict[str, int] = {}
    for condition, artifact_root, model_label in source_conditions:
        rows = condition_rows(condition, artifact_root, model_label)
        condition_counts[condition] = len(rows)
        rows_by_condition[condition] = rows
        pool.extend(rows)
    if len(pool) < args.sample_size:
        raise SystemExit(f"sample_size={args.sample_size} exceeds pool_size={len(pool)}")

    rng = random.Random(args.seed)
    if args.sample_mode == "balanced_condition":
        sampled = balanced_condition_sample(rows_by_condition, args.sample_size, rng)
        sampling = (
            f"balanced random without replacement across {len(rows_by_condition)} source conditions; "
            "each condition contributes floor/ceil(sample_size / n_conditions) rows"
        )
    else:
        sampled = rng.sample(pool, args.sample_size)
        sampling = "uniform random without replacement over all successful held-out test StepFun judgments in the selected source set"
    sampled = sorted(
        sampled,
        key=lambda row: (
            str(row.get("source_condition")),
            str(row.get("split")),
            str(row.get("repeat")),
            str(row.get("task_id")),
            str(row.get("chain_id")),
        ),
    )
    for index, row in enumerate(sampled):
        row["validation_sample_id"] = f"judgeval_{index:04d}"

    manifest = [manifest_row(row, str(row["validation_sample_id"])) for row in sampled]
    prompt_dataset = [prompt_dataset_row(row, str(row["validation_sample_id"])) for row in sampled]
    trace_manifest = [
        {
            "sample_id": row["sample_id"],
            "trace_path": row["trace_path"],
            "source_condition": row["source_condition"],
            "overall_privacy_score_1_to_5": row["overall_privacy_score_1_to_5"],
            "overall_leaked_for_reward": row["overall_leaked_for_reward"],
        }
        for row in manifest
    ]
    summary = summarize(sampled, pool_size=len(pool), seed=args.seed, source_set=args.source_set, sampling=sampling)
    summary["source_condition_pool_counts"] = condition_counts

    write_jsonl(args.output_dir / "sample_manifest.jsonl", manifest)
    write_jsonl(args.output_dir / "sample_prompts.jsonl", prompt_dataset)
    write_jsonl(args.output_dir / "trace_manifest.jsonl", trace_manifest)
    (args.output_dir / "trace_manifest.txt").write_text(
        "\n".join(str(row["trace_path"]) for row in trace_manifest) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "stepfun_reference_sample.jsonl", sampled)
    write_json(args.output_dir / "summary.json", summary)
    write_readme(args.output_dir, summary)
    print(f"sample_dir={args.output_dir}")
    print(json.dumps(summary["estimated_openrouter_usage_if_stepfun_like"], indent=2))


if __name__ == "__main__":
    main()
