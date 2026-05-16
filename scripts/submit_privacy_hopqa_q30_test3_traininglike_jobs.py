#!/usr/bin/env python3
"""Submit clean Privacy HopQA q30 eval jobs.

These runs are intended for paper-quality model comparisons:
- configurable splits and repeat count
- no whole-case retries
- one planner retry attempt
- training-like generation/context budgets for non-StepFun agents
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from submit_privacy_hopqa_q30_eval_jobs import (
    ACCOUNT,
    BASE_MODELS,
    DATA,
    GEMMA4_SETUP_COMMAND,
    GEMMA4_VENV,
    HOME_DATA,
    IMAGE,
    PYTHON,
    RESULTS,
    STEPFUN_URL,
    TRANSFORMERS_DATA,
    WORKDIR,
    shell_join,
)


RUN_TAG = "test3_traininglike_20260515"
BASELINE_ROOT = Path(f"/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_{RUN_TAG}")
PRIVACY_PROMPT_ROOT = Path(f"/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_{RUN_TAG}_privacy_prompt")
HELPER_URL = "http://dns-24e3447c-506e-4b21-92df-156e18db5087-ph-q30-v11-200642:8012"


MODELS: list[dict[str, Any]] = [
    {
        "model_label": "qwen3_4b_instruct_2507",
        "served_model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "model_path": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "concurrency": 20,
    },
    {
        "model_label": "rrf_4b_hopacc_s102",
        "served_model_name": "rrf_4b_hopacc_s102",
        "model_path": "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_hopacc_20260514202306/finetune/current",
        "tokenizer_name": "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_hopacc_20260514202306/finetune/current",
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "concurrency": 20,
    },
    {
        "model_label": "rrf_4b_prefix_step_s109",
        "served_model_name": "rrf_4b_prefix_step_s109",
        "model_path": "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_prefix_step_20260514201431/finetune/current",
        "tokenizer_name": "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_prefix_step_20260514201431/finetune/current",
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "concurrency": 20,
    },
    {
        "model_label": "qwen3_8b",
        "served_model_name": "Qwen/Qwen3-8B",
        "model_path": "/mnt/llmd/base_models/Qwen3-8B",
        "tokenizer_name": "/mnt/llmd/base_models/Qwen3-8B",
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "concurrency": 16,
    },
    {
        "model_label": "gpt_oss_20b",
        "served_model_name": "openai/gpt-oss-20b",
        "model_path": "/mnt/llmd/base_models/gpt-oss-20b",
        "tokenizer_name": "/mnt/llmd/base_models/gpt-oss-20b",
        "gpu": 1,
        "cpu": 48,
        "mem": 512,
        "tp": 1,
        "concurrency": 12,
        "request_api_mode": "completion_template",
        "request_params_json": {"include_reasoning": False},
        "trust_remote_code": True,
    },
    {
        "model_label": "chroma_context_1",
        "served_model_name": "chromadb/context-1",
        "model_path": "victor/context-1",
        "tokenizer_name": "victor/context-1",
        "gpu": 1,
        "cpu": 48,
        "mem": 512,
        "tp": 1,
        "concurrency": 10,
        "request_api_mode": "completion_template",
        "request_params_json": {"include_reasoning": False},
        "trust_remote_code": True,
    },
    {
        "model_label": "gemma4_31b_it",
        "served_model_name": "google/gemma-4-31B-it",
        "model_path": "google/gemma-4-31B-it",
        "tokenizer_name": "google/gemma-4-31B-it",
        "gpu": 2,
        "cpu": 64,
        "mem": 768,
        "tp": 2,
        "concurrency": 10,
        "trust_remote_code": True,
        "python": f"{GEMMA4_VENV}/bin/python",
        "setup_command": GEMMA4_SETUP_COMMAND,
    },
    {
        "model_label": "step35_flash",
        "served_model_name": "step-3.5-flash",
        "tokenizer_name": "/mnt/llmd/base_models/Step-3.5-Flash",
        "base_url": STEPFUN_URL,
        "gpu": 0,
        "cpu": 48,
        "mem": 160,
        "concurrency": 16,
        "max_model_len": 65536,
        "max_tokens": 16000,
        "stage_max_tokens": 16000,
        "generation_context_limit_tokens": 65536,
        "generation_context_margin_tokens": 512,
        "request_api_mode": "completion_template",
        "request_params_json": {"stop": ["<|im_end|>"]},
    },
]


def base_submit_cmd(name: str, cpu: int, mem: int, gpu: int, bid: int, account: str) -> list[str]:
    cmd = [
        "eai",
        "job",
        "submit",
        "--account",
        account,
        "--name",
        name,
        "--image",
        IMAGE,
        "--preemptable",
        "--restartable",
        "--bid",
        str(bid),
        "--cpu",
        str(cpu),
        "--mem",
        str(mem),
        "--data",
        HOME_DATA,
        "--data",
        TRANSFORMERS_DATA,
        "--data",
        DATA,
        "--data",
        RESULTS,
        "--data",
        BASE_MODELS,
        "--env",
        "HOME=/home/toolkit",
        "--env",
        "HF_HOME=/transformers_cache",
        "--env",
        "HF_DATASETS_CACHE=/transformers_cache",
        "--env",
        "PYTHONPATH=/home/toolkit/my_code/PipelineRL:/home/toolkit/my_code/MosaicProject",
        "--workdir",
        WORKDIR,
    ]
    if gpu > 0:
        cmd.extend(["--gpu", str(gpu), "--gpu-mem", "80"])
    return cmd


def rollout_command(
    model: dict[str, Any],
    artifact_root: Path,
    splits: list[str],
    repeats: int,
    privacy_prompt: bool,
) -> str:
    max_model_len = int(model.get("max_model_len", 16384))
    max_tokens = int(model.get("max_tokens", 4096))
    stage_max_tokens = int(model.get("stage_max_tokens", max_tokens))
    generation_context_limit_tokens = int(model.get("generation_context_limit_tokens", 16384))
    generation_context_margin_tokens = int(model.get("generation_context_margin_tokens", 128))
    parts = [
        model.get("python", PYTHON),
        "scripts/run_privacy_hopqa_q30_rollout_worker.py",
        "--artifact-root",
        str(artifact_root),
        "--model-label",
        model["model_label"],
        "--served-model-name",
        model["served_model_name"],
        "--tokenizer-name",
        model["tokenizer_name"],
        "--splits",
        *splits,
        "--repeats",
        str(repeats),
        "--helper-url",
        HELPER_URL,
        "--rollout-concurrency",
        str(model["concurrency"]),
        "--tensor-parallel-size",
        str(model.get("tp", max(1, int(model.get("gpu", 1))))),
        "--max-model-len",
        str(max_model_len),
        "--max-tokens",
        str(max_tokens),
        "--stage-max-tokens",
        str(stage_max_tokens),
        "--generation-context-limit-tokens",
        str(generation_context_limit_tokens),
        "--generation-context-margin-tokens",
        str(generation_context_margin_tokens),
        "--hop-plan-attempts",
        "1",
        "--case-attempts",
        "1",
        "--case-retry-delay-s",
        "0",
        "--max-parallel-doc-reads",
        "10",
        "--temperature",
        "1.0",
        "--choose-top-k",
        "3",
        "--retrieval-top-k",
        "5",
        "--max-candidate-cards",
        "20",
        "--max-parallel-retrieval-actions",
        "3",
    ]
    if model.get("request_api_mode"):
        parts.extend(["--request-api-mode", model["request_api_mode"]])
    if model.get("request_params_json"):
        parts.extend(["--request-params-json", json.dumps(model["request_params_json"], separators=(",", ":"))])
    if model.get("base_url"):
        parts.extend(["--base-url", model["base_url"]])
    else:
        parts.extend(["--model-path", model["model_path"]])
    if model.get("trust_remote_code"):
        parts.append("--trust-remote-code")
    if privacy_prompt:
        parts.append("--planner-privacy-prompt")
    command = shell_join(parts)
    setup_command = model.get("setup_command")
    if setup_command:
        command = f"{setup_command} && {command}"
    return command


def submit(command: list[str], dry_run: bool) -> None:
    print(shell_join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def write_manifest(
    root: Path,
    selected: list[dict[str, Any]],
    run_tag: str,
    splits: list[str],
    repeats: int,
    privacy_prompt: bool,
    stepfun_url: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    manifest_path = root / "job_manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    model_map = {
        model.get("model_label"): model
        for model in existing.get("models", [])
        if isinstance(model, dict) and model.get("model_label")
    }
    for model in selected:
        model_map[model["model_label"]] = model
    submissions = list(existing.get("submissions", []))
    submissions.append(
        {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "model_labels": [model["model_label"] for model in selected],
            "splits": splits,
            "repeats": repeats,
            "planner_privacy_prompt": privacy_prompt,
        }
    )
    manifest = {
        "artifact_root": str(root),
        "run_tag": run_tag,
        "splits": splits,
        "repeats": repeats,
        "case_attempts": 1,
        "hop_plan_attempts": 1,
        "planner_privacy_prompt": privacy_prompt,
        "non_stepfun_generation": {
            "max_model_len": 16384,
            "max_tokens": 4096,
            "stage_max_tokens": 4096,
            "generation_context_limit_tokens": 16384,
            "generation_context_margin_tokens": 128,
        },
        "stepfun_generation": {
            "max_model_len": 65536,
            "max_tokens": 16000,
            "stage_max_tokens": 16000,
            "generation_context_limit_tokens": 65536,
            "generation_context_margin_tokens": 512,
        },
        "models": list(model_map.values()),
        "submissions": submissions,
        "helper_url": HELPER_URL,
        "stepfun_url": stepfun_url,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--privacy-prompt-root", type=Path)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--condition", choices=["baseline", "privacy_prompt"], action="append", default=[])
    parser.add_argument("--only-model-label", action="append", default=[])
    parser.add_argument("--skip-model-label", action="append", default=[])
    parser.add_argument("--job-name-suffix", default="")
    parser.add_argument("--stepfun-base-url", default="")
    parser.add_argument("--account", default=ACCOUNT)
    parser.add_argument("--bid", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = args.condition or ["baseline", "privacy_prompt"]
    only = set(args.only_model_label)
    skip = set(args.skip_model_label)
    selected = [
        dict(model)
        for model in MODELS
        if (not only or model["model_label"] in only) and model["model_label"] not in skip
    ]
    if args.stepfun_base_url:
        for model in selected:
            if model["model_label"] == "step35_flash":
                # Keep the slow StepFun agent rollouts on the dedicated agent
                # server; the default StepFun URL is used by older judge runs.
                model["base_url"] = args.stepfun_base_url
    baseline_root = args.baseline_root or Path(f"/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_{args.run_tag}")
    privacy_prompt_root = args.privacy_prompt_root or Path(
        f"/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_{args.run_tag}_privacy_prompt"
    )
    for condition in conditions:
        privacy_prompt = condition == "privacy_prompt"
        root = privacy_prompt_root if privacy_prompt else baseline_root
        write_manifest(
            root,
            selected,
            args.run_tag,
            list(args.splits),
            int(args.repeats),
            privacy_prompt=privacy_prompt,
            stepfun_url=args.stepfun_base_url or STEPFUN_URL,
        )
        for model in selected:
            suffix = args.job_name_suffix
            if suffix and not suffix.startswith("_"):
                suffix = "_" + suffix
            name = f"privacy_hopqa_q30_{args.run_tag}_{condition}_{model['model_label']}{suffix}"
            command = rollout_command(
                model,
                root,
                list(args.splits),
                int(args.repeats),
                privacy_prompt=privacy_prompt,
            )
            submit(
                base_submit_cmd(
                    name,
                    int(model["cpu"]),
                    int(model["mem"]),
                    int(model.get("gpu", 0)),
                    args.bid,
                    args.account,
                )
                + ["--", "/bin/bash", "-lc", command],
                args.dry_run,
            )


if __name__ == "__main__":
    main()
