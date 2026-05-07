#!/usr/bin/env python3
"""Run base-model Privacy HopQA rollouts for the initial-state artifact.

This script intentionally reuses PipelineRL's privacy_hopqa rollout function
rather than approximating the domain. It runs one model over one or more
canonical splits and appends compact records to tables/rollout_index.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "data_files" / "privacy_hopqa_final_dataset"
DEFAULT_PIPELINERL_ROOT = Path(os.environ.get("PIPELINERL_ROOT", "/home/toolkit/my_code/PipelineRL"))
DEFAULT_TASK_DATA_ROOT = Path("/home/toolkit/nice_code/drbench/drbench/data/tasks")
DEFAULT_LOCAL_INDEX_ROOT = Path("/mnt/llmd/results/exps/alexg/privacy_agent/local_indices_fixeddeps_20260415")
RUNNER_VERSION = 3


@dataclass(frozen=True)
class RunCase:
    split_name: str
    split_output_dir: Path
    problem: dict[str, Any]
    repeat: int


@dataclass
class RunStats:
    queued: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0

    def record(self, ok: bool, skipped: bool = False) -> None:
        if skipped:
            self.skipped += 1
        elif ok:
            self.ok += 1
        else:
            self.failed += 1


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("_").lower() or "model"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed(index_path: Path, model_label: str) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    if not index_path.exists():
        return completed
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model_label") != model_label or not row.get("ok"):
                continue
            completed.add((str(row.get("split")), str(row.get("chain_id")), int(row.get("repeat", 0))))
    return completed


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(rows)
    if args.chain_id:
        wanted = {str(value) for value in args.chain_id}
        selected = [row for row in selected if str(row.get("chain_id")) in wanted]
    if args.task_id:
        wanted = {str(value) for value in args.task_id}
        selected = [row for row in selected if str(row.get("task_id")) in wanted]
    if args.max_examples:
        selected = selected[: args.max_examples]
    return selected


def ensure_pipelinerl_imports(pipelinerl_root: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(pipelinerl_root))
    from omegaconf import OmegaConf
    from pipelinerl.domains.privacy_hopqa.rollouts import generate_privacy_hopqa_rollout
    from pipelinerl.llm import TrainableLLM

    return OmegaConf, TrainableLLM, generate_privacy_hopqa_rollout


def build_cfg(args: argparse.Namespace, split_output_dir: Path, OmegaConf: Any) -> Any:
    helper_url = args.helper_url or os.environ.get("PRIVACY_AGENT_HELPER_URL")
    if not helper_url and not args.local_tools:
        raise ValueError("Set --helper-url or PRIVACY_AGENT_HELPER_URL, or pass --local-tools.")
    dataset_root = Path(args.dataset_root).expanduser()
    return OmegaConf.create(
        {
            "output_dir": str(split_output_dir),
            "privacy_hopqa": {
                "final_dataset_path": str(dataset_root / "privacy_hopqa_finalized_2026-05-05.jsonl"),
                "final_train_dataset_path": str(dataset_root / "train.jsonl"),
                "final_val_dataset_path": str(dataset_root / "val.jsonl"),
                "final_test_dataset_path": str(dataset_root / "test.jsonl"),
                "task_data_root": str(Path(args.task_data_root).expanduser()),
                "local_index_root": str(Path(args.local_index_root).expanduser()),
                "local_index_mmap_mode": "r",
                "helper_service_url": helper_url,
                "helper_timeout_s": args.helper_timeout_s,
                "use_remote_local_search": not args.local_tools,
                "use_remote_browsecomp": not args.local_tools,
                "capture_mode": args.capture_mode,
                "answer_match_f1_threshold": args.answer_match_f1_threshold,
                "max_iterations": args.max_iterations,
                "hop_plan_attempts": args.hop_plan_attempts,
                "enable_generic_retrieval_fallback": args.enable_generic_retrieval_fallback,
                "max_parallel_retrieval_actions": args.max_parallel_retrieval_actions,
                "retrieval_top_k": args.retrieval_top_k,
                "choose_top_k": args.choose_top_k,
                "max_candidate_cards": args.max_candidate_cards,
                "chooser_excerpt_chars": args.chooser_excerpt_chars,
                "reader_excerpt_chars": args.reader_excerpt_chars,
                "max_history_entries": args.max_history_entries,
                "local_search_threshold": args.local_search_threshold,
                "semantic_threshold": args.semantic_threshold,
                "browsecomp_enabled": True,
                "browsecomp_device": args.browsecomp_device,
                "browsecomp_index_glob": args.browsecomp_index_glob,
                "browsecomp_model_name": args.embedding_model,
                "browsecomp_corpus": args.browsecomp_corpus,
                "browsecomp_top_k": args.retrieval_top_k,
                "browsecomp_max_chars": args.browsecomp_max_chars,
                "llm_provider": "vllm",
                "embedding_provider": args.embedding_provider,
                "embedding_model": args.embedding_model,
                "log_prompts": args.log_prompts,
                "verbose": args.verbose,
                "no_web": False,
                "parallel_searches": not args.serial_searches,
                "max_parallel_doc_reads": args.max_parallel_doc_reads,
            },
        }
    )


def new_trace_for(split_output_dir: Path, task_id: str, chain_id: str, before: set[str]) -> Path:
    root = split_output_dir / "privacy_hopqa_runs"
    if not root.exists():
        raise RuntimeError(f"rollout root does not exist: {root}")
    prefix = f"{task_id}_{chain_id}_"
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and path.name not in before and (path / "trace.json").exists()
    ]
    if len(candidates) != 1:
        names = [path.name for path in sorted(candidates)[:8]]
        raise RuntimeError(
            f"expected exactly one new trace for task={task_id} chain={chain_id}; "
            f"found {len(candidates)} candidates={names}"
        )
    return candidates[0] / "trace.json"


async def run_case(
    case: RunCase,
    args: argparse.Namespace,
    llm: Any,
    cfg: Any,
    generate_privacy_hopqa_rollout: Any,
    session: Any,
    sem: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    index_path: Path,
    error_path: Path,
) -> bool:
    split_name = case.split_name
    split_output_dir = case.split_output_dir
    problem = case.problem
    repeat = case.repeat
    task_id = str(problem.get("task_id") or "")
    chain_id = str(problem.get("chain_id") or "")
    root = split_output_dir / "privacy_hopqa_runs"
    started = time.time()
    async with sem:
        try:
            before = {path.name for path in root.iterdir()} if root.exists() else set()
            result = await generate_privacy_hopqa_rollout(cfg, llm, problem, session)
            metrics = result.metrics.model_dump() if hasattr(result.metrics, "model_dump") else dict(result.metrics)
            trace_path = new_trace_for(split_output_dir, task_id, chain_id, before)
            row = {
                "timestamp": iso_now(),
                "ok": True,
                "runner_version": RUNNER_VERSION,
                "model_label": args.model_label,
                "model": args.model,
                "split": split_name,
                "repeat": repeat,
                "task_id": task_id,
                "chain_id": chain_id,
                "company": problem.get("company"),
                "pattern": problem.get("pattern"),
                "n_hops": problem.get("n_hops"),
                "latency_s": round(time.time() - started, 3),
                "reward": metrics.get("reward"),
                "hop_accuracy": metrics.get("hop_accuracy"),
                "final_correct": metrics.get("final_correct"),
                "chain_complete": metrics.get("chain_complete"),
                "training_text_count": len(result.training_texts),
                "metrics": metrics,
                "trace_path": str(trace_path),
            }
        except Exception as exc:
            row = {
                "timestamp": iso_now(),
                "ok": False,
                "runner_version": RUNNER_VERSION,
                "model_label": args.model_label,
                "model": args.model,
                "split": split_name,
                "repeat": repeat,
                "task_id": task_id,
                "chain_id": chain_id,
                "company": problem.get("company"),
                "pattern": problem.get("pattern"),
                "n_hops": problem.get("n_hops"),
                "latency_s": round(time.time() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    async with write_lock:
        append_jsonl(index_path, row)
        if not row.get("ok"):
            append_jsonl(error_path, row)
        status = "ok" if row.get("ok") else "fail"
        print(
            f"{status} model={args.model_label} split={split_name} repeat={repeat} "
            f"chain={chain_id} latency={row.get('latency_s')}s",
            flush=True,
        )
    return bool(row.get("ok"))


async def run_split_cases(
    args: argparse.Namespace,
    cases: list[RunCase],
    llm: Any,
    cfg: Any,
    generate_privacy_hopqa_rollout: Any,
    session: Any,
    index_path: Path,
    error_path: Path,
) -> RunStats:
    stats = RunStats(queued=len(cases))
    if not cases:
        return stats
    queue: asyncio.Queue[RunCase] = asyncio.Queue()
    for case in cases:
        queue.put_nowait(case)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    case_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def case_lock(case: RunCase) -> asyncio.Lock:
        key = (case.split_name, str(case.problem.get("chain_id")))
        lock = case_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            case_locks[key] = lock
        return lock

    async def worker(worker_id: int) -> None:
        while True:
            try:
                case = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                async with case_lock(case):
                    ok = await run_case(
                        case,
                        args,
                        llm,
                        cfg,
                        generate_privacy_hopqa_rollout,
                        session,
                        sem,
                        write_lock,
                        index_path,
                        error_path,
                    )
                stats.record(ok)
            finally:
                queue.task_done()

    worker_count = max(1, min(args.concurrency, len(cases)))
    workers = [asyncio.create_task(worker(i)) for i in range(worker_count)]
    await queue.join()
    await asyncio.gather(*workers)
    return stats


async def run_all(args: argparse.Namespace) -> None:
    import aiohttp

    pipelinerl_root = Path(args.pipelinerl_root).expanduser()
    OmegaConf, TrainableLLM, generate_privacy_hopqa_rollout = ensure_pipelinerl_imports(pipelinerl_root)

    output_root = Path(args.output_root).expanduser()
    model_label = slugify(args.model_label)
    args.model_label = model_label
    tables_dir = output_root / "tables"
    index_path = tables_dir / "rollout_index.jsonl"
    error_path = tables_dir / "rollout_errors.jsonl"
    provenance_path = output_root / "config" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    if not provenance_path.exists() or args.force_provenance:
        provenance = {
            "created_at": iso_now(),
            "artifact_type": "privacy_hopqa_initial_state_eval",
            "runner_version": RUNNER_VERSION,
            "dataset_root": str(Path(args.dataset_root).expanduser()),
            "pipelinerl_root": str(pipelinerl_root),
            "task_data_root": str(Path(args.task_data_root).expanduser()),
            "local_index_root": str(Path(args.local_index_root).expanduser()),
            "helper_url": args.helper_url or os.environ.get("PRIVACY_AGENT_HELPER_URL"),
            "models": [{"label": model_label, "model": args.model, "base_url": args.base_url}],
            "splits": args.splits,
            "runner_args": vars(args),
        }
        provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    llm = TrainableLLM(
        model_name=args.model,
        tokenizer_name=args.tokenizer_name or args.model,
        base_url=args.base_url.rstrip("/"),
        parameters={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
    )

    completed = set() if args.force else load_completed(index_path, model_label)
    total_stats = RunStats()
    async with aiohttp.ClientSession() as session:
        for split_name in args.splits:
            split_path = Path(args.dataset_root).expanduser() / f"{split_name}.jsonl"
            rows = select_rows(read_jsonl(split_path), args)
            split_output_dir = output_root / "rollouts" / model_label / split_name
            split_output_dir.mkdir(parents=True, exist_ok=True)
            cfg = build_cfg(args, split_output_dir, OmegaConf)
            cases: list[RunCase] = []
            for repeat in range(args.repeats):
                for row in rows:
                    key = (split_name, str(row.get("chain_id")), repeat)
                    if key not in completed:
                        cases.append(RunCase(split_name, split_output_dir, row, repeat))
            print(f"model={model_label} split={split_name} queued={len(cases)} rows={len(rows)}", flush=True)
            split_stats = await run_split_cases(
                args,
                cases,
                llm,
                cfg,
                generate_privacy_hopqa_rollout,
                session,
                index_path,
                error_path,
            )
            total_stats.queued += split_stats.queued
            total_stats.ok += split_stats.ok
            total_stats.failed += split_stats.failed
            print(
                f"split_done model={model_label} split={split_name} "
                f"queued={split_stats.queued} ok={split_stats.ok} failed={split_stats.failed}",
                flush=True,
            )
    print(
        f"all_done model={model_label} queued={total_stats.queued} ok={total_stats.ok} failed={total_stats.failed}",
        flush=True,
    )
    if total_stats.failed and not args.continue_on_error:
        raise RuntimeError(
            f"{total_stats.failed} rollout(s) failed. See {error_path}. "
            "Pass --continue-on-error to allow partial artifacts."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL without /v1.")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--pipelinerl-root", default=str(DEFAULT_PIPELINERL_ROOT))
    parser.add_argument("--task-data-root", default=str(DEFAULT_TASK_DATA_ROOT))
    parser.add_argument("--local-index-root", default=str(DEFAULT_LOCAL_INDEX_ROOT))
    parser.add_argument("--helper-url", default=os.environ.get("PRIVACY_AGENT_HELPER_URL"))
    parser.add_argument("--helper-timeout-s", type=float, default=30.0)
    parser.add_argument("--local-tools", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--chain-id", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--capture-mode", default="all_calls")
    parser.add_argument("--answer-match-f1-threshold", type=float, default=0.75)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--hop-plan-attempts", type=int, default=1)
    parser.add_argument("--enable-generic-retrieval-fallback", action="store_true")
    parser.add_argument("--max-parallel-retrieval-actions", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--choose-top-k", type=int, default=3)
    parser.add_argument("--max-candidate-cards", type=int, default=20)
    parser.add_argument("--chooser-excerpt-chars", type=int, default=120)
    parser.add_argument("--reader-excerpt-chars", type=int, default=2500)
    parser.add_argument("--max-history-entries", type=int, default=10)
    parser.add_argument("--local-search-threshold", type=float, default=0.0)
    parser.add_argument("--semantic-threshold", type=float, default=0.7)
    parser.add_argument("--browsecomp-device", default="cpu")
    parser.add_argument("--browsecomp-index-glob", default=".privacy_agent/browsecomp/indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl")
    parser.add_argument("--browsecomp-corpus", default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--browsecomp-max-chars", type=int, default=8000)
    parser.add_argument("--embedding-provider", default="huggingface")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--max-parallel-doc-reads", type=int, default=3)
    parser.add_argument("--serial-searches", action="store_true")
    parser.add_argument("--log-prompts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-provenance", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_all(parse_args()))


if __name__ == "__main__":
    main()
