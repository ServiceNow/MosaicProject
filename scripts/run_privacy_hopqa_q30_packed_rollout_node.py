#!/usr/bin/env python3
"""Run Privacy HopQA q30 rollout shards packed onto one multi-GPU EAI node."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunningTask:
    task: dict[str, Any]
    task_id: str
    gpu_slot: int
    port: int
    process: subprocess.Popen[bytes]
    log_handle: Any
    log_path: Path


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe[:180] or "task"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            if "command" not in task:
                raise ValueError(f"{path}:{line_number} missing command")
            task.setdefault("task_id", f"{path.stem}_{line_number}")
            tasks.append(task)
    return tasks


def visible_devices(gpus_per_node: int) -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if configured:
        devices = [part.strip() for part in configured.split(",") if part.strip()]
        if len(devices) >= gpus_per_node:
            return devices[:gpus_per_node]
    return [str(index) for index in range(gpus_per_node)]


def command_with_port(command: str, port: int) -> list[str]:
    parts = shlex.split(command)
    filtered: list[str] = []
    skip_next = False
    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part == "--vllm-port":
            skip_next = True
            continue
        if part.startswith("--vllm-port="):
            continue
        filtered.append(part)
    filtered.extend(["--vllm-port", str(port)])
    return filtered


def start_task(
    task: dict[str, Any],
    gpu_slot: int,
    device: str,
    port: int,
    log_root: Path,
    node_index: int,
) -> RunningTask:
    task_id = str(task["task_id"])
    command = command_with_port(str(task["command"]), port)
    node_log_root = log_root / f"node_{node_index:02d}"
    node_log_root.mkdir(parents=True, exist_ok=True)
    log_path = node_log_root / f"gpu{gpu_slot}_{sanitize(task_id)}.log"
    log_handle = log_path.open("ab")
    header = {
        "event": "start_task",
        "time": iso_now(),
        "task_id": task_id,
        "gpu_slot": gpu_slot,
        "cuda_visible_devices": device,
        "port": port,
        "command": command,
    }
    log_handle.write((json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"))
    log_handle.flush()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(
        f"{iso_now()} started {task_id} on gpu_slot={gpu_slot} device={device} port={port} pid={process.pid}",
        flush=True,
    )
    return RunningTask(task, task_id, gpu_slot, port, process, log_handle, log_path)


def terminate_running(running: dict[int, RunningTask]) -> None:
    for item in running.values():
        if item.process.poll() is None:
            item.process.terminate()
    deadline = time.time() + 45
    for item in running.values():
        while item.process.poll() is None and time.time() < deadline:
            time.sleep(1)
        if item.process.poll() is None:
            item.process.kill()
        item.log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=8000)
    parser.add_argument("--poll-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.task_file)
    devices = visible_devices(args.gpus_per_node)
    pending = list(tasks)
    running: dict[int, RunningTask] = {}
    failures: list[dict[str, Any]] = []
    stopping = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        print(f"{iso_now()} received signal {signum}; terminating running tasks", flush=True)
        terminate_running(running)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    args.log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": iso_now(),
        "node_index": args.node_index,
        "task_file": str(args.task_file),
        "task_count": len(tasks),
        "gpus_per_node": args.gpus_per_node,
        "devices": devices,
    }
    (args.log_root / f"node_{args.node_index:02d}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"{iso_now()} packed node {args.node_index} starting {len(tasks)} tasks across {args.gpus_per_node} GPUs",
        flush=True,
    )
    try:
        while pending or running:
            for gpu_slot in range(args.gpus_per_node):
                if not pending or gpu_slot in running:
                    continue
                port = args.port_base + gpu_slot
                task = pending.pop(0)
                running[gpu_slot] = start_task(
                    task,
                    gpu_slot,
                    devices[gpu_slot],
                    port,
                    args.log_root,
                    args.node_index,
                )

            time.sleep(args.poll_s)

            for gpu_slot, item in list(running.items()):
                code = item.process.poll()
                if code is None:
                    continue
                item.log_handle.write(
                    (
                        json.dumps(
                            {
                                "event": "finish_task",
                                "time": iso_now(),
                                "task_id": item.task_id,
                                "returncode": code,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                item.log_handle.close()
                running.pop(gpu_slot)
                if code:
                    failures.append(
                        {
                            "task_id": item.task_id,
                            "returncode": code,
                            "log_path": str(item.log_path),
                        }
                    )
                print(
                    f"{iso_now()} finished {item.task_id} gpu_slot={gpu_slot} returncode={code}",
                    flush=True,
                )

            if stopping:
                break
    finally:
        terminate_running(running)

    summary = {
        "finished_at": iso_now(),
        "node_index": args.node_index,
        "task_count": len(tasks),
        "failure_count": len(failures),
        "failures": failures,
    }
    summary_path = args.log_root / f"node_{args.node_index:02d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{iso_now()} packed node {args.node_index} complete failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
