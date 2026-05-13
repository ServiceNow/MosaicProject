#!/usr/bin/env python3
"""Run one Privacy HopQA q30 rollout shard inside an EAI job.

The worker is intentionally resumable: it writes into a fixed model-run
artifact root and delegates row-level skipping to
`run_privacy_hopqa_initial_state_eval.py`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_full_regen_20260513")
DEFAULT_DATASET_ROOT = ROOT / "data_files" / "privacy_hopqa_final_dataset"
DEFAULT_DATASET_BASENAME = "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review"
DEFAULT_HELPER_URL = "http://dns-24e3447c-506e-4b21-92df-156e18db5087-q4:8012"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wait_for_models(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = base_url.rstrip("/") + "/v1/models"
    last_error = ""
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=10) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # pragma: no cover - diagnostic path.
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def start_vllm(args: argparse.Namespace, model_run_root: Path) -> subprocess.Popen[bytes] | None:
    if args.base_url:
        print(f"using external model endpoint {args.base_url}", flush=True)
        return None
    log_path = model_run_root / "logs" / "vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.vllm_port),
        "--model",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.vllm_extra_arg:
        cmd.extend(args.vllm_extra_arg)

    print("starting vllm:", " ".join(cmd), flush=True)
    print("vllm log:", log_path, flush=True)
    log_handle = log_path.open("ab")
    process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
    return process


def terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def run_rollouts(args: argparse.Namespace, model_run_root: Path, base_url: str) -> int:
    script = ROOT / "scripts" / "run_privacy_hopqa_initial_state_eval.py"
    cmd = [
        sys.executable,
        str(script),
        "--output-root",
        str(model_run_root),
        "--model-label",
        args.model_label,
        "--model",
        args.served_model_name,
        "--base-url",
        base_url.rstrip("/"),
        "--tokenizer-name",
        args.tokenizer_name or args.model_path or args.served_model_name,
        "--dataset-root",
        str(args.dataset_root),
        "--dataset-basename",
        args.dataset_basename,
        "--splits",
        *args.splits,
        "--helper-url",
        args.helper_url,
        "--helper-timeout-s",
        str(args.helper_timeout_s),
        "--concurrency",
        str(args.rollout_concurrency),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--request-api-mode",
        args.request_api_mode,
        "--generation-context-limit-tokens",
        str(args.generation_context_limit_tokens),
        "--generation-context-margin-tokens",
        str(args.generation_context_margin_tokens),
        "--hop-plan-max-tokens",
        str(args.stage_max_tokens),
        "--doc-choose-max-tokens",
        str(args.stage_max_tokens),
        "--doc-read-max-tokens",
        str(args.stage_max_tokens),
        "--hop-resolve-max-tokens",
        str(args.stage_max_tokens),
        "--max-iterations",
        str(args.max_iterations),
        "--hop-plan-attempts",
        str(args.hop_plan_attempts),
        "--case-attempts",
        str(args.case_attempts),
        "--case-retry-delay-s",
        str(args.case_retry_delay_s),
        "--choose-top-k",
        str(args.choose_top_k),
        "--retrieval-top-k",
        str(args.retrieval_top_k),
        "--max-candidate-cards",
        str(args.max_candidate_cards),
        "--max-parallel-retrieval-actions",
        str(args.max_parallel_retrieval_actions),
        "--max-parallel-doc-reads",
        str(args.max_parallel_doc_reads),
        "--capture-mode",
        "all_calls",
        "--log-prompts",
        "--continue-on-error",
    ]
    if args.request_params_json:
        cmd.extend(["--request-params-json", args.request_params_json])
    if args.planner_privacy_prompt:
        cmd.append("--planner-privacy-prompt")
    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])

    log_path = model_run_root / "logs" / "rollout_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("running rollouts:", " ".join(cmd), flush=True)
    print("rollout log:", log_path, flush=True)
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-basename", default=DEFAULT_DATASET_BASENAME)
    parser.add_argument("--helper-url", default=os.environ.get("PRIVACY_AGENT_HELPER_URL", DEFAULT_HELPER_URL))
    parser.add_argument("--helper-timeout-s", type=float, default=180.0)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vllm-extra-arg", action="append", default=[])
    parser.add_argument("--vllm-timeout-s", type=float, default=1800.0)
    parser.add_argument("--rollout-concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--request-api-mode", choices=["chat", "completion_template"], default="chat")
    parser.add_argument("--request-params-json", default="")
    parser.add_argument("--stage-max-tokens", type=int, default=16000)
    parser.add_argument("--generation-context-limit-tokens", type=int, default=32000)
    parser.add_argument("--generation-context-margin-tokens", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=21)
    parser.add_argument("--hop-plan-attempts", type=int, default=1)
    parser.add_argument("--case-attempts", type=int, default=1)
    parser.add_argument("--case-retry-delay-s", type=float, default=5.0)
    parser.add_argument("--planner-privacy-prompt", action="store_true")
    parser.add_argument("--choose-top-k", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--max-candidate-cards", type=int, default=20)
    parser.add_argument("--max-parallel-retrieval-actions", type=int, default=3)
    parser.add_argument("--max-parallel-doc-reads", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_run_root = args.artifact_root / "model_runs" / args.model_label
    model_run_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at": iso_now(),
        "worker": "run_privacy_hopqa_q30_rollout_worker.py",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "vLLM is launched without max-num-seqs so endpoint concurrency is controlled by vLLM defaults.",
    }
    write_json(model_run_root / "config" / "worker_metadata.json", metadata)

    vllm_process: subprocess.Popen[bytes] | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; terminating child processes", flush=True)
        terminate(vllm_process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        vllm_process = start_vllm(args, model_run_root)
        base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
        wait_for_models(base_url, args.vllm_timeout_s)
        code = run_rollouts(args, model_run_root, base_url)
        if code:
            raise SystemExit(code)
    finally:
        terminate(vllm_process)


if __name__ == "__main__":
    main()
