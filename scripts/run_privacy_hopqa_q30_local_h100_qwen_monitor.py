#!/usr/bin/env python3
"""Use the local H100 for Qwen privacy-judge/binary-predictor backfill.

This intentionally writes to *_local_h100.jsonl files so it can run alongside
the EAI monitors without two processes appending to the same JSONL.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_test3_traininglike_20260515"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_test3_traininglike_20260515_privacy_prompt"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_all1_traininglike_20260515"),
    Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_all1_traininglike_20260515_privacy_prompt"),
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        except Exception as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def start_vllm(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "local_h100_qwen4b_vllm.log"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--model",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        "1",
    ]
    print("starting local qwen vllm:", " ".join(cmd), flush=True)
    print("local qwen vllm log:", log_path, flush=True)
    return subprocess.Popen(cmd, stdout=log_path.open("ab"), stderr=subprocess.STDOUT, cwd=ROOT)


def terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def run_cmd(cmd: list[str]) -> int:
    print("running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def model_roots(artifact_root: Path) -> list[Path]:
    root = artifact_root / "model_runs"
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "tables" / "rollout_index.jsonl").exists())


def run_pass(args: argparse.Namespace, qwen_base_url: str) -> None:
    for artifact_root in args.artifact_root:
        for model_root in model_roots(artifact_root):
            tables = model_root / "tables"
            qwen_output = tables / "privacy_leakage_suite_v2_qwen3_4b_local_h100.jsonl"
            qwen_errors = tables / "privacy_leakage_suite_v2_qwen3_4b_local_h100_errors.jsonl"
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_privacy_hopqa_leakage_suite_v2.py"),
                    "--artifact-root",
                    str(model_root),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--private-qa-doc-scope",
                    "read",
                    "--max-private-questions",
                    "200",
                    "--continue-on-error",
                    "--output",
                    str(qwen_output),
                    "--errors",
                    str(qwen_errors),
                    "--base-url",
                    qwen_base_url,
                    "--model",
                    args.served_model_name,
                    "--api-mode",
                    "chat",
                    "--concurrency",
                    str(args.qwen_concurrency),
                    "--timeout",
                    "600",
                    "--max-tokens",
                    str(args.judge_max_tokens),
                ]
            )
            stepfun_output = tables / "privacy_leakage_suite_v2_stepfun_full.jsonl"
            if not stepfun_output.exists():
                continue
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_privacy_hopqa_binary_leakage_predictor.py"),
                    "--input",
                    str(stepfun_output),
                    "--output",
                    str(tables / "privacy_binary_leakage_predictor_qwen3_4b_local_h100.jsonl"),
                    "--errors",
                    str(tables / "privacy_binary_leakage_predictor_qwen3_4b_local_h100_errors.jsonl"),
                    "--summary",
                    str(tables / "privacy_binary_leakage_predictor_qwen3_4b_local_h100_summary.json"),
                    "--base-url",
                    qwen_base_url,
                    "--model",
                    args.served_model_name,
                    "--api-mode",
                    "chat",
                    "--concurrency",
                    str(args.binary_concurrency),
                    "--max-tokens",
                    "2048",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data_files" / "privacy_hopqa_final_dataset")
    parser.add_argument("--log-dir", type=Path, default=Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_local_h100_qwen_monitor/logs"))
    parser.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--qwen-concurrency", type=int, default=32)
    parser.add_argument("--binary-concurrency", type=int, default=32)
    parser.add_argument("--judge-max-tokens", type=int, default=16000)
    parser.add_argument("--interval-s", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.artifact_root:
        args.artifact_root = list(DEFAULT_ROOTS)
    return args


def main() -> None:
    args = parse_args()
    vllm_process: subprocess.Popen[bytes] | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        terminate(vllm_process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        vllm_process = start_vllm(args)
        qwen_base_url = f"http://127.0.0.1:{args.port}"
        wait_for_models(qwen_base_url, 1800)
        while True:
            print(f"local h100 qwen monitor pass started {iso_now()}", flush=True)
            run_pass(args, qwen_base_url)
            if args.once:
                break
            print(f"local h100 qwen monitor sleeping {args.interval_s}s", flush=True)
            time.sleep(args.interval_s)
    finally:
        terminate(vllm_process)


if __name__ == "__main__":
    main()
