#!/usr/bin/env python3
"""Submit final-harness Privacy HopQA q30 rollouts for comparison models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import submit_privacy_hopqa_q30_final_eval_runner as final
from submit_privacy_hopqa_q30_eval_jobs import shell_join


RUN_TAG = "external_final_rollouts_16kgen_32kctx_helperfe_20260522"
STEPFUN_AGENT_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-sfjudgeq30new0521:8000"
GEMMA4_VENV = "/tmp/gemma4_vllm_env"
GEMMA4_SETUP_COMMAND = shell_join(
    [
        "/bin/bash",
        "-lc",
        (
            "set -euo pipefail; "
            f"if [ ! -x {GEMMA4_VENV}/bin/python ]; then "
            f"rm -rf {GEMMA4_VENV}; "
            f"{final.PYTHON} -m venv --system-site-packages {GEMMA4_VENV}; "
            f"{GEMMA4_VENV}/bin/python -m pip install --no-cache-dir --no-deps 'vllm==0.19.1'; "
            f"{GEMMA4_VENV}/bin/python -m pip install --no-cache-dir "
            "'flashinfer-cubin==0.6.6' 'compressed-tensors==0.15.0.1'; "
            f"{GEMMA4_VENV}/bin/python -m pip install --no-cache-dir -U "
            "'git+https://github.com/huggingface/transformers.git'; "
            "fi; "
            f"{GEMMA4_VENV}/bin/python - <<'PY'\n"
            "import importlib.util\n"
            "import transformers, vllm\n"
            "from transformers import AutoConfig\n"
            "assert importlib.util.find_spec('vllm.model_executor.models.gemma4') is not None\n"
            "AutoConfig.from_pretrained('google/gemma-4-31B-it', trust_remote_code=True)\n"
            "print('gemma4 overlay', transformers.__version__, vllm.__version__)\n"
            "PY"
        ),
    ]
)


MODELS: list[dict[str, Any]] = [
    {
        "model_label": "rrf_4b_hopacc_s102",
        "served_model_name": "rrf_4b_hopacc_s102",
        "model_path": "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_hopacc_20260514202306/finetune/current",
        "tokenizer_name": "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_hopacc_20260514202306/finetune/current",
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "rollout_concurrency": 8,
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
        "rollout_concurrency": 8,
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
        "rollout_concurrency": 8,
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
        "rollout_concurrency": 6,
        "temperature": 0.7,
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
        "rollout_concurrency": 6,
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
        "rollout_concurrency": 4,
        "trust_remote_code": True,
        "python": f"{GEMMA4_VENV}/bin/python",
        "setup_command": GEMMA4_SETUP_COMMAND,
        "vllm_extra_args": [
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "4096",
            "--generation-config",
            "vllm",
            "--skip-mm-profiling",
        ],
    },
    {
        "model_label": "step35_flash",
        "served_model_name": "step-3.5-flash",
        "tokenizer_name": "/mnt/llmd/base_models/Step-3.5-Flash",
        "base_url": STEPFUN_AGENT_URL,
        "gpu": 0,
        "cpu": 48,
        "mem": 160,
        "tp": 1,
        "rollout_concurrency": 8,
        "temperature": 0.0,
        "request_api_mode": "completion_template_direct",
        "request_params_json": {"stop": ["<|im_end|>"]},
    },
]


def model_by_label() -> dict[str, dict[str, Any]]:
    return {str(model["model_label"]): dict(model) for model in MODELS}


def rollout_command(
    model: dict[str, Any],
    artifact_root: Path,
    dataset_root: Path,
    dataset_basename: str,
    helper_url: str,
    rollout_args: dict[str, Any],
) -> str:
    parts: list[Any] = [
        model.get("python", final.PYTHON),
        "scripts/run_privacy_hopqa_q30_rollout_worker.py",
        "--artifact-root",
        artifact_root,
        "--model-label",
        model["model_label"],
        "--served-model-name",
        model["served_model_name"],
        "--tokenizer-name",
        model["tokenizer_name"],
        "--dataset-root",
        dataset_root,
        "--dataset-basename",
        dataset_basename,
        "--splits",
        *rollout_args["splits"],
        "--repeats",
        rollout_args["repeats"],
        "--helper-url",
        helper_url,
        "--helper-timeout-s",
        rollout_args["helper_timeout_s"],
        "--rollout-concurrency",
        model.get("rollout_concurrency", rollout_args["rollout_concurrency"]),
        "--tensor-parallel-size",
        model.get("tp", 1),
        "--max-model-len",
        model.get("max_model_len", rollout_args["max_model_len"]),
        "--max-tokens",
        model.get("max_tokens", rollout_args["max_tokens"]),
        "--stage-max-tokens",
        model.get("stage_max_tokens", rollout_args["stage_max_tokens"]),
        "--generation-context-limit-tokens",
        model.get("generation_context_limit_tokens", rollout_args["generation_context_limit_tokens"]),
        "--generation-context-margin-tokens",
        model.get("generation_context_margin_tokens", rollout_args["generation_context_margin_tokens"]),
        "--max-iterations",
        rollout_args["max_iterations"],
        "--hop-plan-attempts",
        rollout_args["hop_plan_attempts"],
        "--case-attempts",
        rollout_args["case_attempts"],
        "--case-retry-delay-s",
        rollout_args["case_retry_delay_s"],
        "--temperature",
        model.get("temperature", rollout_args["temperature"]),
        "--choose-top-k",
        rollout_args["choose_top_k"],
        "--retrieval-top-k",
        rollout_args["retrieval_top_k"],
        "--max-candidate-cards",
        rollout_args["max_candidate_cards"],
        "--max-parallel-retrieval-actions",
        rollout_args["max_parallel_retrieval_actions"],
        "--max-parallel-doc-reads",
        rollout_args["max_parallel_doc_reads"],
        "--request-api-mode",
        model.get("request_api_mode", "chat"),
    ]
    if model.get("base_url"):
        parts.extend(["--base-url", model["base_url"], "--model-path", ""])
    else:
        parts.extend(["--model-path", model["model_path"]])
        for extra_arg in model.get("vllm_extra_args", rollout_args["vllm_extra_args"]):
            if str(extra_arg).startswith("-"):
                parts.append(f"--vllm-extra-arg={extra_arg}")
            else:
                parts.extend(["--vllm-extra-arg", extra_arg])
    if model.get("request_params_json"):
        parts.extend(["--request-params-json", json.dumps(model["request_params_json"], separators=(",", ":"))])
    if model.get("trust_remote_code"):
        parts.append("--trust-remote-code")
    if rollout_args.get("max_examples"):
        parts.extend(["--max-examples", rollout_args["max_examples"]])
    for repeat_id in rollout_args.get("repeat_ids", []):
        parts.extend(["--repeat-id", repeat_id])
    if rollout_args.get("row_shard_count", 1) > 1:
        parts.extend(["--row-shard-index", rollout_args.get("row_shard_index", 0)])
        parts.extend(["--row-shard-count", rollout_args["row_shard_count"]])
    if rollout_args.get("planner_privacy_prompt"):
        parts.append("--planner-privacy-prompt")

    command = final.shell_join(parts)
    setup_command = model.get("setup_command")
    if setup_command:
        command = f"{setup_command} && {command}"
    return command


def write_manifest(
    artifact_root: Path,
    dataset_view: Path,
    selected_models: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "artifact_root": str(artifact_root),
        "dataset_root": str(args.dataset_root),
        "dataset_basename": args.dataset_basename,
        "dataset_view": str(dataset_view),
        "helper_url": args.helper_url,
        "stepfun_agent_url": args.stepfun_agent_url,
        "privacy_eval": "stepfun_monitor_submitted" if args.submit_privacy_eval else "not_submitted_by_this_runner",
        "rollout_args": final.COMMON_ROLLOUT_ARGS
        | {
            "splits": args.splits,
            "repeats": args.repeats,
            "max_examples": args.max_examples,
            "planner_privacy_prompt": args.planner_privacy_prompt,
        },
        "models": selected_models,
    }
    (artifact_root / "external_final_rollouts_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def submit(command: list[str], dry_run: bool) -> None:
    print(final.shell_join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=final.DATASET_ROOT)
    parser.add_argument("--dataset-basename", default=final.DATASET_BASENAME)
    parser.add_argument("--helper-url", default=final.HELPER_URL)
    parser.add_argument("--stepfun-base-url", default=final.STEPFUN_URL)
    parser.add_argument("--stepfun-agent-url", default=STEPFUN_AGENT_URL)
    parser.add_argument(
        "--stepfun-request-api-mode",
        choices=["chat", "completion_template", "completion_template_direct"],
        default="",
    )
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--planner-privacy-prompt", action="store_true")
    parser.add_argument("--only-model-label", action="append", default=[])
    parser.add_argument("--skip-model-label", action="append", default=[])
    parser.add_argument("--repeat-shards", action="store_true", help="Submit one rollout job per repeat per model.")
    parser.add_argument(
        "--row-shards-per-repeat",
        type=int,
        default=1,
        help="Submit this many deterministic row shards inside each repeat.",
    )
    parser.add_argument("--job-name-suffix", default="")
    parser.add_argument("--submit-privacy-eval", action="store_true")
    parser.add_argument("--privacy-interval-s", type=int, default=600)
    parser.add_argument("--stepfun-concurrency", type=int, default=96)
    parser.add_argument("--rollout-concurrency", type=int)
    parser.add_argument("--vllm-max-num-seqs", type=int)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--viewer-output-dir", type=Path)
    parser.add_argument("--account", default=final.ACCOUNT)
    parser.add_argument("--bid", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root or Path(f"/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_{args.run_tag}")
    viewer_output_dir = args.viewer_output_dir or Path(final.WORKDIR) / "runtime_analysis" / f"privacy_hopqa_q30_{args.run_tag}"
    available = model_by_label()
    selected_labels = list(args.only_model_label) or list(available)
    skip_labels = set(args.skip_model_label)
    missing = sorted(set(selected_labels) - set(available))
    if missing:
        raise ValueError(f"Unknown model labels: {missing}")

    selected_models = [dict(available[label]) for label in selected_labels if label not in skip_labels]
    for model in selected_models:
        if model["model_label"] == "step35_flash":
            model["base_url"] = args.stepfun_agent_url
            if args.stepfun_request_api_mode:
                model["request_api_mode"] = args.stepfun_request_api_mode
                if args.stepfun_request_api_mode == "chat":
                    model.pop("request_params_json", None)
    if not selected_models:
        raise ValueError("No models selected")

    rollout_args = final.COMMON_ROLLOUT_ARGS | {
        "repeats": args.repeats,
        "splits": args.splits,
        "max_examples": args.max_examples,
        "planner_privacy_prompt": args.planner_privacy_prompt,
    }
    if args.rollout_concurrency is not None:
        rollout_args["rollout_concurrency"] = args.rollout_concurrency
    if args.vllm_max_num_seqs is not None or args.vllm_max_num_batched_tokens is not None:
        rollout_args["vllm_extra_args"] = [
            "--max-num-seqs",
            str(args.vllm_max_num_seqs or 8),
            "--max-num-batched-tokens",
            str(args.vllm_max_num_batched_tokens or 4096),
            "--generation-config",
            "vllm",
        ]
    dataset_view = final.ensure_dataset_view(artifact_root, args.dataset_root, args.dataset_basename)
    write_manifest(artifact_root, dataset_view, selected_models, args)

    suffix = args.job_name_suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    for model in selected_models:
        repeat_ids = range(args.repeats) if args.repeat_shards else [None]
        for repeat_id in repeat_ids:
            for row_shard_index in range(args.row_shards_per_repeat):
                repeat_suffix = f"_r{repeat_id}" if repeat_id is not None else ""
                row_suffix = (
                    f"_rs{row_shard_index}of{args.row_shards_per_repeat}"
                    if args.row_shards_per_repeat > 1
                    else ""
                )
                name = f"privacy_hopqa_q30_{args.run_tag}_{model['model_label']}{repeat_suffix}{row_suffix}{suffix}"
                command_rollout_args = dict(rollout_args)
                if repeat_id is not None:
                    command_rollout_args["repeat_ids"] = [repeat_id]
                if args.row_shards_per_repeat > 1:
                    command_rollout_args["row_shard_index"] = row_shard_index
                    command_rollout_args["row_shard_count"] = args.row_shards_per_repeat
                command = rollout_command(
                    model,
                    artifact_root,
                    args.dataset_root,
                    args.dataset_basename,
                    args.helper_url,
                    command_rollout_args,
                )
                submit_cmd = final.base_submit_cmd(
                    name=name,
                    cpu=int(model["cpu"]),
                    mem=int(model["mem"]),
                    gpu=int(model.get("gpu", 0)),
                    bid=args.bid,
                    account=args.account,
                )
                submit_cmd.extend(["--", "/bin/bash", "-lc", command])
                submit(submit_cmd, args.dry_run)

    if args.submit_privacy_eval:
        command = final.privacy_eval_command(
            artifact_root=artifact_root,
            dataset_view=dataset_view,
            stepfun_url=args.stepfun_base_url,
            interval_s=args.privacy_interval_s,
            stepfun_concurrency=args.stepfun_concurrency,
            viewer_output_dir=viewer_output_dir,
            latest_viewer_output=Path(final.WORKDIR)
            / "runtime_analysis"
            / "latest_privacy_hopqa_q30_external_final_rollouts.html",
        )
        submit_cmd = final.base_submit_cmd(
            name=f"privacy_hopqa_q30_{args.run_tag}_stepfun_monitor{suffix}",
            cpu=48,
            mem=160,
            gpu=0,
            bid=args.bid,
            account=args.account,
        )
        submit_cmd.extend(["--", "/bin/bash", "-lc", command])
        submit(submit_cmd, args.dry_run)


if __name__ == "__main__":
    main()
