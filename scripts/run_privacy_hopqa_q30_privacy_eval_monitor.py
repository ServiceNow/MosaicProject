#!/usr/bin/env python3
"""Continuously evaluate Privacy HopQA q30 rollout privacy leakage."""

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
DEFAULT_STEPFUN_URL = "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000"
VIEWER_DIR = ROOT / "runtime_analysis" / "privacy_hopqa_q30_live_analysis"
LATEST_VIEWERS = [
    ROOT / "runtime_analysis" / "latest_privacy_hopqa_q30_live_analysis.html",
    ROOT / "runtime_analysis" / "privacy_hopqa_q30_live_analysis_latest.html",
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
        except Exception as exc:  # pragma: no cover - diagnostic path.
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def discover_model_roots(artifact_root: Path) -> list[Path]:
    roots = []
    model_runs = artifact_root / "model_runs"
    if not model_runs.exists():
        return []
    for path in sorted(model_runs.iterdir()):
        if path.is_dir() and (path / "tables" / "rollout_index.jsonl").exists():
            roots.append(path)
    return roots


def count_ok_rollouts(model_root: Path) -> int:
    return sum(1 for row in read_jsonl(model_root / "tables" / "rollout_index.jsonl") if row.get("ok"))


def start_qwen_vllm(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    log_path = args.artifact_root / "logs" / "qwen3_4b_privacy_judge_vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.qwen_port),
        "--model",
        args.qwen_model_path,
        "--served-model-name",
        args.qwen_served_model_name,
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        str(args.qwen_max_model_len),
        "--tensor-parallel-size",
        "1",
    ]
    print("starting qwen privacy judge vllm:", " ".join(cmd), flush=True)
    print("qwen judge vllm log:", log_path, flush=True)
    log_handle = log_path.open("ab")
    return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)


def terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def run_cmd(cmd: list[str], cwd: Path = ROOT, tolerate_failure: bool = True) -> int:
    print("running:", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=cwd).returncode
    if code and not tolerate_failure:
        raise subprocess.CalledProcessError(code, cmd)
    if code:
        print(f"command failed with exit code {code}; continuing", flush=True)
    return code


def run_privacy_evals_for_model(args: argparse.Namespace, model_root: Path, qwen_base_url: str) -> None:
    if count_ok_rollouts(model_root) == 0:
        print(f"skip {model_root.name}: no successful rollouts yet", flush=True)
        return
    tables = model_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    stepfun_output = tables / "privacy_leakage_suite_v2_stepfun_full.jsonl"
    qwen_output = tables / "privacy_leakage_suite_v2_qwen3_4b_full.jsonl"
    binary_output = tables / "privacy_binary_leakage_predictor_qwen3_4b.jsonl"

    common = [
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
    ]
    # Run the fast Qwen judge first so comparison plots update while the
    # StepFun ground-truth judge, which is much slower, continues to catch up.
    run_cmd(
        common
        + [
            "--output",
            str(qwen_output),
            "--errors",
            str(tables / "privacy_leakage_suite_v2_qwen3_4b_full_errors.jsonl"),
            "--base-url",
            qwen_base_url,
            "--model",
            args.qwen_served_model_name,
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
    run_cmd(
        common
        + [
            "--output",
            str(stepfun_output),
            "--errors",
            str(tables / "privacy_leakage_suite_v2_stepfun_full_errors.jsonl"),
            "--base-url",
            args.stepfun_base_url,
            "--model",
            "step-3.5-flash",
            "--api-mode",
            "completion_template",
            "--concurrency",
            str(args.stepfun_concurrency),
            "--timeout",
            "600",
            "--max-tokens",
            str(args.judge_max_tokens),
        ]
    )
    if stepfun_output.exists():
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_privacy_hopqa_binary_leakage_predictor.py"),
                "--input",
                str(stepfun_output),
                "--output",
                str(binary_output),
                "--errors",
                str(tables / "privacy_binary_leakage_predictor_qwen3_4b_errors.jsonl"),
                "--summary",
                str(tables / "privacy_binary_leakage_predictor_qwen3_4b_summary.json"),
                "--base-url",
                qwen_base_url,
                "--model",
                args.qwen_served_model_name,
                "--api-mode",
                "chat",
                "--concurrency",
                str(args.binary_concurrency),
                "--max-tokens",
                "2048",
            ]
        )


def write_eai_snapshot() -> None:
    snapshot = VIEWER_DIR / "eai_jobs_snapshot.jsonl"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "eai",
        "job",
        "ls",
        "--account",
        "snow.research.adea",
        "--state",
        "all",
        "--limit",
        "100",
        "--name",
        "privacy_hopqa_q30",
        "--fields",
        "id,state,name,created,updatedOn,runs.exitCode,resources.gpu,resources.replicas",
        "--format",
        "json",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        snapshot.write_text(json.dumps({"timestamp": iso_now(), "error": str(exc)}) + "\n", encoding="utf-8")
        return
    snapshot.write_text(result.stdout, encoding="utf-8")


def build_viewer(args: argparse.Namespace) -> None:
    write_eai_snapshot()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_privacy_hopqa_q30_live_analysis_viewer.py"),
        "--artifact-root",
        str(args.artifact_root),
        "--output-dir",
        str(VIEWER_DIR),
    ]
    for latest_viewer in LATEST_VIEWERS:
        cmd.extend(["--latest-output", str(latest_viewer)])
    run_cmd(cmd)


def push_viewer_update() -> None:
    paths = [
        str(VIEWER_DIR),
        *(str(latest_viewer) for latest_viewer in LATEST_VIEWERS),
        str(ROOT / "docs" / "privacy_hopqa_q30_eval_runbook_2026-05-13.md"),
    ]
    run_cmd(["git", "add", *paths], cwd=ROOT, tolerate_failure=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
    if diff == 0:
        print("viewer push skipped: no staged changes", flush=True)
        return
    message = f"Update Privacy HopQA q30 live analysis viewer ({iso_now()})"
    if run_cmd(["git", "commit", "-m", message], cwd=ROOT, tolerate_failure=True) == 0:
        run_cmd(["git", "push"], cwd=ROOT, tolerate_failure=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data_files" / "privacy_hopqa_final_dataset")
    parser.add_argument("--stepfun-base-url", default=DEFAULT_STEPFUN_URL)
    parser.add_argument("--qwen-model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--qwen-served-model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--qwen-port", type=int, default=8000)
    parser.add_argument("--qwen-max-model-len", type=int, default=32768)
    parser.add_argument("--qwen-timeout-s", type=float, default=1800.0)
    parser.add_argument("--judge-max-tokens", type=int, default=16000)
    parser.add_argument("--stepfun-concurrency", type=int, default=48)
    parser.add_argument("--qwen-concurrency", type=int, default=24)
    parser.add_argument("--binary-concurrency", type=int, default=32)
    parser.add_argument("--interval-s", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--push-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen_process: subprocess.Popen[bytes] | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; terminating qwen judge", flush=True)
        terminate(qwen_process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        qwen_process = start_qwen_vllm(args)
        qwen_base_url = f"http://127.0.0.1:{args.qwen_port}"
        wait_for_models(qwen_base_url, args.qwen_timeout_s)
        while True:
            print(f"privacy eval monitor pass started {iso_now()}", flush=True)
            for model_root in discover_model_roots(args.artifact_root):
                run_privacy_evals_for_model(args, model_root, qwen_base_url)
            build_viewer(args)
            if args.push_viewer:
                push_viewer_update()
            if args.once:
                break
            print(f"privacy eval monitor sleeping {args.interval_s}s", flush=True)
            time.sleep(args.interval_s)
    finally:
        terminate(qwen_process)


if __name__ == "__main__":
    main()
