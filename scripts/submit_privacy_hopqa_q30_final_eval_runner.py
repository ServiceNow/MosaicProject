#!/usr/bin/env python3
"""Submit the final Privacy HopQA q30 trained-model eval.

This runner exists to remove the ambiguity between rollout dataset basenames
and scorer dataset roots. Rollouts use the current q30 basename explicitly,
while StepFun scoring reads an artifact-local dataset_view whose train/val/test
files point at that same basename.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


RUN_TAG = "final_eval_16kgen_32kctx_helperfe_20260522"
ARTIFACT_ROOT = Path(f"/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_{RUN_TAG}")

IMAGE = "registry.toolkit-sp.yul201.service-now.com/snow.research.tapes/interactive-toolkit:throughput"
ACCOUNT = "snow.research.adea"
HOME_DATA = "snow.home.alex_gurung:/home/toolkit:rw"
TRANSFORMERS_DATA = "snow.home.alex_gurung:/transformers_cache:rw"
DATA = "snow.research.tapes.data:/mnt/llmd/data:ro"
RESULTS = "snow.research.rlar.results:/mnt/llmd/results:rw"
BASE_MODELS = "snow.research.tapes.base_models:/mnt/llmd/base_models:ro"
WORKDIR = "/home/toolkit/my_code/MosaicProject"
PYTHON = "/home/toolkit/.conda/envs/pipeline-rl/bin/python"

DATASET_ROOT = Path("/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset")
DATASET_BASENAME = "privacy_hopqa_finalized_2026-05-20_q30_hume_safe_variants"
HELPER_URL = "http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30fev221410:8012"
STEPFUN_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-sfjudgeq30new0521:8000"

SITUATIONAL_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_4b_situational_reward_lr1e6_ga4096_rep4_50s_20260520173055"
)
PRIVREWARD_TAU05_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/"
    "privacy_hopqa_q30_4b_situational_privreward_tau05_lr1e6_rep8_200s_20260522191644"
)
SITUATIONAL_LIVE_ACTOR_URL = "http://dns-7583e47f-7f4f-4369-bd78-28f115897b3e-phrl22d2baf9-3:8080"
SITUATIONAL_LIVE_ACTOR_MODEL = str(SITUATIONAL_ROOT / "finetune" / "current")

COMMON_ROLLOUT_ARGS: dict[str, Any] = {
    "splits": ["test"],
    "repeats": 3,
    "rollout_concurrency": 8,
    "max_model_len": 32768,
    "max_tokens": 16384,
    "stage_max_tokens": 16384,
    "generation_context_limit_tokens": 32768,
    "generation_context_margin_tokens": 128,
    "max_iterations": 14,
    "hop_plan_attempts": 1,
    "case_attempts": 3,
    "case_retry_delay_s": 10,
    "temperature": 1.0,
    "choose_top_k": 3,
    "retrieval_top_k": 5,
    "max_candidate_cards": 20,
    "max_parallel_retrieval_actions": 3,
    "max_parallel_doc_reads": 5,
    "helper_timeout_s": 180,
    "vllm_extra_args": [
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--generation-config",
        "vllm",
    ],
}


def immutable_checkpoint_model(label_prefix: str, run_root: Path, step: int) -> dict[str, Any]:
    checkpoint = run_root / "finetune" / "intermediate" / str(step)
    model_label = f"{label_prefix}_s{step}"
    return {
        "model_label": model_label,
        "served_model_name": model_label,
        "model_path": str(checkpoint),
        "tokenizer_name": str(checkpoint),
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "latest_step_hint": step,
        "include_by_default": False,
    }


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
    },
    {
        "model_label": "qwen3_4b_situational_reward_s50",
        "served_model_name": "qwen3_4b_situational_reward_s50",
        "model_path": str(SITUATIONAL_ROOT / "finetune" / "intermediate" / "50"),
        "tokenizer_name": str(SITUATIONAL_ROOT / "finetune" / "intermediate" / "50"),
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
    },
    {
        # Historical label from the first submission. Those rows were generated
        # from the HF checkpoint saved after completed step 90; keep backfills
        # pinned so finetune/current cannot advance underneath the same label.
        "model_label": "qwen3_4b_situational_reward_current_s80",
        "served_model_name": "qwen3_4b_situational_reward_current_s80",
        "model_path": str(SITUATIONAL_ROOT / "finetune" / "intermediate" / "90"),
        "tokenizer_name": str(SITUATIONAL_ROOT / "finetune" / "intermediate" / "90"),
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "latest_step_hint": 90,
        "include_by_default": False,
    },
    {
        # Latest completed immutable checkpoint from the situational run.
        "model_label": "qwen3_4b_situational_reward_s100",
        "served_model_name": "qwen3_4b_situational_reward_s100",
        "model_path": str(SITUATIONAL_ROOT / "finetune" / "intermediate" / "100"),
        "tokenizer_name": str(SITUATIONAL_ROOT / "finetune" / "intermediate" / "100"),
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "latest_step_hint": 100,
    },
    {
        # Immutable step-50 checkpoint from the privacy-reward tau=0.5 run.
        "model_label": "qwen3_4b_privreward_tau05_s50",
        "served_model_name": "qwen3_4b_privreward_tau05_s50",
        "model_path": str(PRIVREWARD_TAU05_ROOT / "finetune" / "intermediate" / "50"),
        "tokenizer_name": str(PRIVREWARD_TAU05_ROOT / "finetune" / "intermediate" / "50"),
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "latest_step_hint": 50,
        "include_by_default": False,
    },
    *[
        immutable_checkpoint_model("qwen3_4b_situational_reward", SITUATIONAL_ROOT, step)
        for step in (10, 20, 30, 40, 60, 70, 80, 90)
    ],
    *[
        immutable_checkpoint_model("qwen3_4b_privreward_tau05", PRIVREWARD_TAU05_ROOT, step)
        for step in (10, 20, 30, 40, 60, 70, 80, 90, 100, 110)
    ],
    {
        # Non-immutable exploratory endpoint: actor vLLM has received weight
        # updates after step 90 even though no HF checkpoint has been exported.
        "model_label": "qwen3_4b_situational_reward_live_s98_actor",
        "served_model_name": SITUATIONAL_LIVE_ACTOR_MODEL,
        "model_path": "",
        "tokenizer_name": SITUATIONAL_LIVE_ACTOR_MODEL,
        "base_url": SITUATIONAL_LIVE_ACTOR_URL,
        "gpu": 0,
        "cpu": 16,
        "mem": 128,
        "tp": 1,
        "latest_step_hint": "live_actor_after_step_98_non_immutable",
        "include_by_default": False,
    },
]


def shell_join(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def ensure_dataset_view(artifact_root: Path, dataset_root: Path, dataset_basename: str) -> Path:
    view = artifact_root / "dataset_view"
    view.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "created_at": iso_now(),
        "dataset_root": str(dataset_root),
        "dataset_basename": dataset_basename,
        "files": {},
        "note": "Scorer-safe view: evaluate_privacy_hopqa_leakage_suite_v2.py reads train/val/test.jsonl.",
    }
    for split in ("train", "val", "test"):
        source = dataset_root / f"{dataset_basename}_{split}.jsonl"
        target = view / f"{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists() or target.is_symlink():
            try:
                if target.resolve() == source.resolve():
                    manifest["files"][split] = {"source": str(source), "target": str(target)}
                    continue
                target.unlink()
            except FileNotFoundError:
                pass
        try:
            target.symlink_to(source)
        except FileExistsError:
            if target.resolve() != source.resolve():
                raise
        manifest["files"][split] = {"source": str(source), "target": str(target)}
    (view / "dataset_view_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return view


def rollout_command(
    model: dict[str, Any],
    artifact_root: Path,
    dataset_root: Path,
    dataset_basename: str,
    helper_url: str,
    rollout_args: dict[str, Any],
) -> str:
    parts: list[Any] = [
        PYTHON,
        "scripts/run_privacy_hopqa_q30_rollout_worker.py",
        "--artifact-root",
        artifact_root,
        "--model-label",
        model["model_label"],
        "--served-model-name",
        model["served_model_name"],
        "--model-path",
        model["model_path"],
        "--tokenizer-name",
        model["tokenizer_name"],
    ]
    if model.get("base_url"):
        parts.extend(["--base-url", model["base_url"]])
    parts.extend([
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
        rollout_args["rollout_concurrency"],
        "--tensor-parallel-size",
        model.get("tp", 1),
        "--max-model-len",
        rollout_args["max_model_len"],
        "--max-tokens",
        rollout_args["max_tokens"],
        "--stage-max-tokens",
        rollout_args["stage_max_tokens"],
        "--generation-context-limit-tokens",
        rollout_args["generation_context_limit_tokens"],
        "--generation-context-margin-tokens",
        rollout_args["generation_context_margin_tokens"],
        "--max-iterations",
        rollout_args["max_iterations"],
        "--hop-plan-attempts",
        rollout_args["hop_plan_attempts"],
        "--case-attempts",
        rollout_args["case_attempts"],
        "--case-retry-delay-s",
        rollout_args["case_retry_delay_s"],
        "--temperature",
        rollout_args["temperature"],
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
    ])
    for extra_arg in rollout_args["vllm_extra_args"]:
        if str(extra_arg).startswith("-"):
            parts.append(f"--vllm-extra-arg={extra_arg}")
        else:
            parts.extend(["--vllm-extra-arg", extra_arg])
    for repeat_id in rollout_args.get("repeat_ids", []):
        parts.extend(["--repeat-id", repeat_id])
    if rollout_args.get("row_shard_count", 1) > 1:
        parts.extend(["--row-shard-index", rollout_args.get("row_shard_index", 0)])
        parts.extend(["--row-shard-count", rollout_args["row_shard_count"]])
    if rollout_args.get("planner_privacy_prompt"):
        parts.append("--planner-privacy-prompt")
    return shell_join(parts)


def privacy_eval_command(
    artifact_root: Path,
    dataset_view: Path,
    stepfun_url: str,
    interval_s: int,
    stepfun_concurrency: int,
    viewer_output_dir: Path,
    only_model_labels: list[str] | None = None,
    skip_viewer: bool = False,
    latest_viewer_output: Path | None = None,
) -> str:
    if latest_viewer_output is None:
        latest_viewer_output = Path(WORKDIR) / "runtime_analysis" / "latest_privacy_hopqa_q30_final_eval.html"
    parts: list[Any] = [
        PYTHON,
        "scripts/run_privacy_hopqa_q30_privacy_eval_monitor.py",
        "--artifact-root",
        artifact_root,
        "--dataset-root",
        dataset_view,
        "--stepfun-base-url",
        stepfun_url,
        "--interval-s",
        interval_s,
        "--skip-qwen-judge",
        "--skip-binary-predictor",
        "--stepfun-concurrency",
        stepfun_concurrency,
        "--judge-max-tokens",
        "16384",
        "--viewer-output-dir",
        viewer_output_dir,
        "--latest-viewer-output",
        latest_viewer_output,
        "--viewer-skip-privacy-prompt-comparison",
    ]
    for model_label in only_model_labels or []:
        parts.extend(["--only-model-label", model_label])
    if skip_viewer:
        parts.append("--skip-viewer")
    return shell_join(parts)


def write_manifest(
    artifact_root: Path,
    dataset_view: Path,
    selected_models: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created_at": iso_now(),
        "run_tag": args.run_tag,
        "artifact_root": str(artifact_root),
        "dataset_root": str(args.dataset_root),
        "dataset_basename": args.dataset_basename,
        "dataset_view": str(dataset_view),
        "helper_url": args.helper_url,
        "stepfun_url": args.stepfun_base_url,
        "rollout_args": COMMON_ROLLOUT_ARGS
        | {
            "repeats": args.repeats,
            "splits": args.splits,
            "planner_privacy_prompt": args.planner_privacy_prompt,
        },
        "models": selected_models,
        "scoring": {
            "judge": "step-3.5-flash",
            "stepfun_concurrency": args.stepfun_concurrency,
            "privacy_eval_interval_s": args.privacy_interval_s,
            "qwen_judge": False,
            "binary_predictor": False,
        },
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "final_eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def submit(command: list[str], dry_run: bool) -> None:
    print(shell_join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--dataset-basename", default=DATASET_BASENAME)
    parser.add_argument("--helper-url", default=HELPER_URL)
    parser.add_argument("--stepfun-base-url", default=STEPFUN_URL)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only-model-label", action="append", default=[])
    parser.add_argument("--repeat-shards", action="store_true", help="Submit one rollout job per repeat per model.")
    parser.add_argument("--planner-privacy-prompt", action="store_true")
    parser.add_argument(
        "--row-shards-per-repeat",
        type=int,
        default=1,
        help="Submit this many deterministic row shards inside each repeat.",
    )
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--skip-privacy-eval", action="store_true")
    parser.add_argument("--privacy-interval-s", type=int, default=600)
    parser.add_argument("--stepfun-concurrency", type=int, default=96)
    parser.add_argument("--privacy-skip-viewer", action="store_true")
    parser.add_argument("--rollout-concurrency", type=int)
    parser.add_argument("--vllm-max-num-seqs", type=int)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--viewer-output-dir", type=Path)
    parser.add_argument("--job-name-suffix", default="")
    parser.add_argument("--account", default=ACCOUNT)
    parser.add_argument("--bid", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root or Path(f"/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_{args.run_tag}")
    viewer_output_dir = args.viewer_output_dir or Path(WORKDIR) / "runtime_analysis" / f"privacy_hopqa_q30_{args.run_tag}"
    selected_labels = set(args.only_model_label)
    if selected_labels:
        selected_models = [dict(model) for model in MODELS if model["model_label"] in selected_labels]
    else:
        selected_models = [dict(model) for model in MODELS if model.get("include_by_default", True)]
    if not selected_models:
        raise ValueError(f"No models selected from labels: {sorted(selected_labels)}")

    rollout_args = COMMON_ROLLOUT_ARGS | {
        "repeats": args.repeats,
        "splits": args.splits,
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
    dataset_view = ensure_dataset_view(artifact_root, args.dataset_root, args.dataset_basename)
    write_manifest(artifact_root, dataset_view, selected_models, args)

    if not args.skip_rollouts:
        for model in selected_models:
            repeat_ids = range(args.repeats) if args.repeat_shards else [None]
            for repeat_id in repeat_ids:
                row_shards = range(args.row_shards_per_repeat)
                for row_shard_index in row_shards:
                    suffix = args.job_name_suffix
                    if suffix and not suffix.startswith("_"):
                        suffix = "_" + suffix
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
                    submit_cmd = base_submit_cmd(
                        name=name,
                        cpu=int(model["cpu"]),
                        mem=int(model["mem"]),
                        gpu=int(model.get("gpu", 0)),
                        bid=args.bid,
                        account=args.account,
                    )
                    submit_cmd.extend(["--", "/bin/bash", "-lc", command])
                    submit(submit_cmd, args.dry_run)

    if not args.skip_privacy_eval:
        suffix = args.job_name_suffix
        if suffix and not suffix.startswith("_"):
            suffix = "_" + suffix
        command = privacy_eval_command(
            artifact_root,
            dataset_view,
            args.stepfun_base_url,
            args.privacy_interval_s,
            args.stepfun_concurrency,
            viewer_output_dir,
            selected_labels,
            args.privacy_skip_viewer,
        )
        submit_cmd = base_submit_cmd(
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
