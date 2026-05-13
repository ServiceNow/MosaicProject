#!/usr/bin/env python3
"""Submit Privacy HopQA q30 rollout and privacy-eval jobs to EAI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


ARTIFACT_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_full_regen_20260513")
IMAGE = "registry.toolkit-sp.yul201.service-now.com/snow.research.tapes/interactive-toolkit:throughput"
ACCOUNT = "snow.research.adea"
HOME_DATA = "snow.home.alex_gurung:/home/toolkit:rw"
TRANSFORMERS_DATA = "snow.home.alex_gurung:/transformers_cache:rw"
DATA = "snow.research.tapes.data:/mnt/llmd/data:ro"
RESULTS = "snow.research.rlar.results:/mnt/llmd/results:rw"
BASE_MODELS = "snow.research.tapes.base_models:/mnt/llmd/base_models:ro"
WORKDIR = "/home/toolkit/my_code/MosaicProject"
PYTHON = "/home/toolkit/.conda/envs/pipeline-rl/bin/python"
HELPER_URL = "http://dns-24e3447c-506e-4b21-92df-156e18db5087-qrb140411:8012"
STEPFUN_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000"
GEMMA4_VENV = "/tmp/gemma4_vllm_env"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


GEMMA4_SETUP_COMMAND = shell_join(
    [
        "/bin/bash",
        "-lc",
        (
            "set -euo pipefail; "
            f"if [ ! -x {shlex.quote(GEMMA4_VENV)}/bin/python ]; then "
            f"rm -rf {shlex.quote(GEMMA4_VENV)}; "
            f"{shlex.quote(PYTHON)} -m venv --system-site-packages {shlex.quote(GEMMA4_VENV)}; "
            f"{shlex.quote(GEMMA4_VENV)}/bin/python -m pip install --no-cache-dir --no-deps 'vllm==0.19.1'; "
            f"{shlex.quote(GEMMA4_VENV)}/bin/python -m pip install --no-cache-dir "
            "'flashinfer-cubin==0.6.6' 'compressed-tensors==0.15.0.1'; "
            f"{shlex.quote(GEMMA4_VENV)}/bin/python -m pip install --no-cache-dir -U "
            "'git+https://github.com/huggingface/transformers.git'; "
            "fi; "
            f"{shlex.quote(GEMMA4_VENV)}/bin/python - <<'PY'\n"
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


ROLLOUT_JOBS: list[dict[str, Any]] = [
    {
        "name": "privacy_hopqa_q30_eval_rollouts_qwen3_4b_20260513",
        "model_label": "qwen3_4b_instruct_2507",
        "served_model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "model_path": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
        "splits": ["train", "val", "test"],
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "concurrency": 20,
        "max_model_len": 32768,
        "max_parallel_doc_reads": 10,
        "hop_plan_attempts": 3,
        "case_attempts": 5,
    },
    {
        "name": "privacy_hopqa_q30_eval_rollouts_qwen3_8b_20260513",
        "model_label": "qwen3_8b",
        "served_model_name": "Qwen/Qwen3-8B",
        "model_path": "/mnt/llmd/base_models/Qwen3-8B",
        "tokenizer_name": "/mnt/llmd/base_models/Qwen3-8B",
        "splits": ["train", "val", "test"],
        "gpu": 1,
        "cpu": 48,
        "mem": 384,
        "tp": 1,
        "concurrency": 16,
        "max_model_len": 32768,
        "max_parallel_doc_reads": 10,
        "hop_plan_attempts": 3,
        "case_attempts": 5,
    },
    {
        "name": "privacy_hopqa_q30_eval_rollouts_stepfun35_flash_20260513",
        "model_label": "step35_flash",
        "served_model_name": "step-3.5-flash",
        "tokenizer_name": "/mnt/llmd/base_models/Step-3.5-Flash",
        "base_url": STEPFUN_URL,
        "splits": ["train", "val", "test"],
        "gpu": 0,
        "cpu": 32,
        "mem": 128,
        "concurrency": 12,
        "max_model_len": 65536,
        "max_parallel_doc_reads": 10,
        "hop_plan_attempts": 3,
        "case_attempts": 5,
    },
    {
        "name": "privacy_hopqa_q30_eval_rollouts_gpt_oss_20b_20260513",
        "model_label": "gpt_oss_20b",
        "served_model_name": "openai/gpt-oss-20b",
        "model_path": "/mnt/llmd/base_models/gpt-oss-20b",
        "tokenizer_name": "/mnt/llmd/base_models/gpt-oss-20b",
        "splits": ["train", "val", "test"],
        "gpu": 1,
        "cpu": 48,
        "mem": 512,
        "tp": 1,
        "concurrency": 12,
        "max_model_len": 32768,
        "max_parallel_doc_reads": 10,
        "hop_plan_attempts": 3,
        "case_attempts": 5,
        "temperature": 0.7,
        "request_api_mode": "completion_template",
        "request_params_json": {"include_reasoning": False},
        "trust_remote_code": True,
    },
    {
        "name": "privacy_hopqa_q30_eval_rollouts_gemma4_31b_it_20260513",
        "model_label": "gemma4_31b_it",
        "served_model_name": "google/gemma-4-31B-it",
        "model_path": "google/gemma-4-31B-it",
        "tokenizer_name": "google/gemma-4-31B-it",
        "splits": ["train", "val", "test"],
        "gpu": 2,
        "cpu": 64,
        "mem": 768,
        "tp": 2,
        "concurrency": 10,
        "max_model_len": 32768,
        "max_parallel_doc_reads": 10,
        "hop_plan_attempts": 3,
        "case_attempts": 5,
        "trust_remote_code": True,
        "python": f"{GEMMA4_VENV}/bin/python",
        "setup_command": GEMMA4_SETUP_COMMAND,
    },
]


PRIVACY_EVAL_JOB = {
    "name": "privacy_hopqa_q30_eval_privacy_judges_20260513",
    "gpu": 1,
    "cpu": 48,
    "mem": 384,
}
def base_submit_cmd(name: str, cpu: int, mem: int, gpu: int, bid: int) -> list[str]:
    cmd = [
        "eai",
        "job",
        "submit",
        "--account",
        ACCOUNT,
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


def rollout_worker_command(job: dict[str, Any], artifact_root: Path, helper_url: str) -> str:
    parts = [
        job.get("python", PYTHON),
        "scripts/run_privacy_hopqa_q30_rollout_worker.py",
        "--artifact-root",
        str(artifact_root),
        "--model-label",
        job["model_label"],
        "--served-model-name",
        job["served_model_name"],
        "--tokenizer-name",
        job["tokenizer_name"],
        "--splits",
        *job["splits"],
        "--helper-url",
        helper_url,
        "--rollout-concurrency",
        str(job["concurrency"]),
        "--tensor-parallel-size",
        str(job.get("tp", max(1, job.get("gpu", 1)))),
        "--max-model-len",
        str(job["max_model_len"]),
        "--max-tokens",
        "16000",
        "--stage-max-tokens",
        "16000",
        "--hop-plan-attempts",
        str(job.get("hop_plan_attempts", 3)),
        "--case-attempts",
        str(job.get("case_attempts", 5)),
        "--case-retry-delay-s",
        str(job.get("case_retry_delay_s", 5.0)),
        "--max-parallel-doc-reads",
        str(job.get("max_parallel_doc_reads", 10)),
        "--temperature",
        str(job.get("temperature", 1.0)),
    ]
    if job.get("request_api_mode"):
        parts.extend(["--request-api-mode", job["request_api_mode"]])
    if job.get("request_params_json"):
        parts.extend(["--request-params-json", json.dumps(job["request_params_json"], separators=(",", ":"))])
    if job.get("base_url"):
        parts.extend(["--base-url", job["base_url"]])
    else:
        parts.extend(["--model-path", job["model_path"]])
    if job.get("trust_remote_code"):
        parts.append("--trust-remote-code")
    if job.get("planner_privacy_prompt"):
        parts.append("--planner-privacy-prompt")
    if job.get("max_examples"):
        parts.extend(["--max-examples", str(job["max_examples"])])
    return shell_join(parts)


def privacy_eval_command(
    artifact_root: Path,
    interval_s: int,
    once: bool,
    push_viewer: bool,
    skip_qwen_judge: bool,
    skip_binary_predictor: bool,
    skip_viewer: bool,
) -> str:
    parts = [
        PYTHON,
        "scripts/run_privacy_hopqa_q30_privacy_eval_monitor.py",
        "--artifact-root",
        str(artifact_root),
        "--stepfun-base-url",
        STEPFUN_URL,
        "--qwen-model-path",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--qwen-served-model-name",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--interval-s",
        str(interval_s),
    ]
    if once:
        parts.append("--once")
    if push_viewer:
        parts.append("--push-viewer")
    if skip_qwen_judge:
        parts.append("--skip-qwen-judge")
    if skip_binary_predictor:
        parts.append("--skip-binary-predictor")
    if skip_viewer:
        parts.append("--skip-viewer")
    return shell_join(parts)


def submit(command: list[str], dry_run: bool) -> None:
    print(shell_join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--bid", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--skip-privacy-eval", action="store_true")
    parser.add_argument("--only-model-label", action="append", default=[])
    parser.add_argument("--job-name-suffix", default="", help="Suffix appended to submitted EAI job names for reruns.")
    parser.add_argument(
        "--force-privacy-eval-with-filter",
        action="store_true",
        help="Submit the privacy-eval monitor even when --only-model-label is used.",
    )
    parser.add_argument("--privacy-once", action="store_true")
    parser.add_argument("--privacy-interval-s", type=int, default=1800)
    parser.add_argument("--skip-qwen-judge", action="store_true")
    parser.add_argument("--skip-binary-predictor", action="store_true")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--push-viewer", action="store_true")
    parser.add_argument("--helper-url", default=HELPER_URL)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=[])
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--planner-privacy-prompt", action="store_true")
    parser.add_argument("--case-attempts", type=int, default=5)
    parser.add_argument("--case-retry-delay-s", type=float, default=5.0)
    parser.add_argument("--hop-plan-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_root": str(args.artifact_root),
        "rollout_jobs": ROLLOUT_JOBS,
        "privacy_eval_job": PRIVACY_EVAL_JOB,
        "helper_url": args.helper_url,
        "stepfun_url": STEPFUN_URL,
    }
    manifest_path = args.artifact_root / "job_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if not args.skip_rollouts:
        selected_labels = set(args.only_model_label)
        for base_job in ROLLOUT_JOBS:
            job = dict(base_job)
            if selected_labels and job["model_label"] not in selected_labels:
                continue
            if args.splits:
                job["splits"] = args.splits
            if args.max_examples:
                job["max_examples"] = args.max_examples
            if args.planner_privacy_prompt:
                job["planner_privacy_prompt"] = True
            job["case_attempts"] = args.case_attempts
            job["case_retry_delay_s"] = args.case_retry_delay_s
            job["hop_plan_attempts"] = args.hop_plan_attempts
            job_command = rollout_worker_command(job, args.artifact_root, args.helper_url)
            if job.get("setup_command"):
                job_command = f"{job['setup_command']} && {job_command}"
            submit_cmd = base_submit_cmd(job["name"] + args.job_name_suffix, job["cpu"], job["mem"], job["gpu"], args.bid)
            submit_cmd.extend(["--", "/bin/bash", "-lc", job_command])
            submit(submit_cmd, args.dry_run)

    should_submit_privacy_eval = not args.skip_privacy_eval and (
        not args.only_model_label or args.force_privacy_eval_with_filter
    )
    if should_submit_privacy_eval:
        job = PRIVACY_EVAL_JOB
        privacy_eval_gpu = 0 if args.skip_qwen_judge and args.skip_binary_predictor else job["gpu"]
        job_command = privacy_eval_command(
            args.artifact_root,
            args.privacy_interval_s,
            args.privacy_once,
            args.push_viewer,
            args.skip_qwen_judge,
            args.skip_binary_predictor,
            args.skip_viewer,
        )
        submit_cmd = base_submit_cmd(job["name"] + args.job_name_suffix, job["cpu"], job["mem"], privacy_eval_gpu, args.bid)
        submit_cmd.extend(["--", "/bin/bash", "-lc", job_command])
        submit(submit_cmd, args.dry_run)


if __name__ == "__main__":
    main()
