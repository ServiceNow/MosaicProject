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
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "data_files" / "privacy_hopqa_final_dataset"
DEFAULT_DATASET_BASENAME = "privacy_hopqa_finalized_2026-05-11"
DEFAULT_PIPELINERL_ROOT = Path(os.environ.get("PIPELINERL_ROOT", "/home/toolkit/my_code/PipelineRL"))
DEFAULT_TASK_DATA_ROOT = Path("/home/toolkit/nice_code/drbench/drbench/data/tasks")
DEFAULT_LOCAL_INDEX_ROOT = Path("/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/local_indices_goldcoverage_20260519")
RUNNER_VERSION = 4


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
    system_failed: int = 0

    def record(self, status: str) -> None:
        if status == "skipped":
            self.skipped += 1
        elif status == "system_failed":
            self.system_failed += 1
        elif status == "ok":
            self.ok += 1
        else:
            self.failed += 1


def classify_failure(error_type: str, error: str) -> str:
    """Classify rollout failures for scoring and resume.

    Agent failures are model/protocol failures, such as malformed JSON, and
    count as accuracy errors. Infrastructure and artifact failures are retryable
    and should not consume a repeat.
    """
    message = (error or "").lower()
    if error_type == "PrivacyHopQALLMInfrastructureError" or "llm infrastructure failure" in message:
        return "infrastructure"
    if any(
        marker in message
        for marker in [
            "cannot connect to host",
            "connection timeout",
            "server disconnected",
            "asyncenginedeaderror",
            "engine background task failed",
        ]
    ):
        return "infrastructure"
    if "expected exactly one new trace" in message:
        return "system"
    if error_type in {"PrivacyHopQAProtocolError", "JSONDecodeError"}:
        return "agent"
    if any(marker in message for marker in ["protocol error", "no valid json", "resolver_wipe"]):
        return "agent"
    return "system"


def failure_counts_for_accuracy(row: dict[str, Any]) -> bool:
    if row.get("ok"):
        return True
    kind = str(row.get("failure_kind") or "").strip().lower()
    if not kind:
        kind = classify_failure(str(row.get("error_type") or ""), str(row.get("error") or ""))
    return kind == "agent"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_openai_base_url(base_url: str) -> str:
    """Return the endpoint root used before `/v1/...` API paths.

    The rollout transport builds OpenAI-compatible paths itself. Hosted APIs
    such as OpenRouter commonly document the base URL with `/v1` included, so
    normalize that form here instead of requiring every launch command to know
    the harness convention.
    """
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized


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
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def training_text_token_totals(training_texts: list[Any]) -> dict[str, int]:
    """Aggregate token usage from PipelineRL training-text records."""
    prompt_tokens = 0
    output_tokens = 0
    finished = 0
    length_finished = 0
    for item in training_texts:
        if hasattr(item, "prompt_tokens"):
            prompt_tokens += int(getattr(item, "prompt_tokens") or 0)
            output_tokens += int(getattr(item, "output_tokens") or 0)
            is_finished = bool(getattr(item, "finished", False))
        elif isinstance(item, dict):
            prompt_tokens += int(item.get("prompt_tokens") or 0)
            output_tokens += int(item.get("output_tokens") or 0)
            is_finished = bool(item.get("finished", False))
        else:
            is_finished = False
        if is_finished:
            finished += 1
        else:
            length_finished += 1
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "finished_llm_calls": finished,
        "length_finished_llm_calls": length_finished,
    }


def load_completed(index_path: Path, model_label: str, include_failed: bool = False) -> set[tuple[str, str, int]]:
    del include_failed  # Agent-induced failures are always terminal; system failures remain retryable.
    completed: set[tuple[str, str, int]] = set()
    if not index_path.exists():
        return completed
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model_label") != model_label:
                continue
            if not row.get("ok"):
                if not failure_counts_for_accuracy(row):
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
    if args.row_shard_count > 1:
        selected = [
            row
            for idx, row in enumerate(selected)
            if idx % args.row_shard_count == args.row_shard_index
        ]
    if args.max_examples:
        selected = selected[: args.max_examples]
    return selected


def ensure_pipelinerl_imports(pipelinerl_root: Path) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    sys.path.insert(0, str(pipelinerl_root))
    from omegaconf import OmegaConf
    from pipelinerl.domains.privacy_hopqa.rollouts import generate_privacy_hopqa_rollout
    from pipelinerl.llm import LLMEvent, LLMOutput, LLMStream, Prompt, TrainableLLM

    return OmegaConf, LLMEvent, LLMOutput, LLMStream, Prompt, TrainableLLM, generate_privacy_hopqa_rollout


def build_cfg(args: argparse.Namespace, split_output_dir: Path, OmegaConf: Any) -> Any:
    helper_url = args.helper_url or os.environ.get("PRIVACY_AGENT_HELPER_URL")
    if not helper_url and not args.local_tools:
        raise ValueError("Set --helper-url or PRIVACY_AGENT_HELPER_URL, or pass --local-tools.")
    dataset_root = Path(args.dataset_root).expanduser()
    dataset_basename = args.dataset_basename
    return OmegaConf.create(
        {
            "output_dir": str(split_output_dir),
            "privacy_hopqa": {
                "final_dataset_path": str(dataset_root / f"{dataset_basename}.jsonl"),
                "final_train_dataset_path": str(dataset_root / f"{dataset_basename}_train.jsonl"),
                "final_val_dataset_path": str(dataset_root / f"{dataset_basename}_val.jsonl"),
                "final_test_dataset_path": str(dataset_root / f"{dataset_basename}_test.jsonl"),
                "task_data_root": str(Path(args.task_data_root).expanduser()),
                "local_index_root": str(Path(args.local_index_root).expanduser()),
                "local_index_mmap_mode": "r",
                "helper_service_url": helper_url,
                "helper_timeout_s": args.helper_timeout_s,
                "use_remote_local_search": not args.local_tools,
                "use_remote_browsecomp": not args.local_tools,
                "capture_mode": args.capture_mode,
                "generation_context_limit_tokens": args.generation_context_limit_tokens,
                "generation_context_margin_tokens": args.generation_context_margin_tokens,
                "hop_plan_max_tokens": args.hop_plan_max_tokens,
                "doc_choose_max_tokens": args.doc_choose_max_tokens,
                "doc_read_max_tokens": args.doc_read_max_tokens,
                "hop_resolve_max_tokens": args.hop_resolve_max_tokens,
                "answer_match_f1_threshold": args.answer_match_f1_threshold,
                "max_iterations": args.max_iterations,
                "hop_plan_attempts": args.hop_plan_attempts,
                "planner_privacy_prompt": args.planner_privacy_prompt,
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
    attempt_path: Path,
    system_error_path: Path,
) -> str:
    split_name = case.split_name
    split_output_dir = case.split_output_dir
    problem = case.problem
    repeat = case.repeat
    task_id = str(problem.get("task_id") or "")
    chain_id = str(problem.get("chain_id") or "")
    root = split_output_dir / "privacy_hopqa_runs"
    started = time.time()
    case_attempts = max(1, int(args.case_attempts))
    attempt_rows: list[dict[str, Any]] = []
    async with sem:
        for attempt in range(1, case_attempts + 1):
            attempt_started = time.time()
            try:
                before = {path.name for path in root.iterdir()} if root.exists() else set()
                result = await generate_privacy_hopqa_rollout(cfg, llm, problem, session)
                metrics = result.metrics.model_dump() if hasattr(result.metrics, "model_dump") else dict(result.metrics)
                token_totals = training_text_token_totals(result.training_texts)
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
                    "attempt_latency_s": round(time.time() - attempt_started, 3),
                    "case_attempt": attempt,
                    "case_attempts": case_attempts,
                    "reward": metrics.get("reward"),
                    "hop_accuracy": metrics.get("hop_accuracy"),
                    "raw_hop_accuracy": metrics.get("raw_hop_accuracy", metrics.get("hop_accuracy")),
                    "conditional_hop_accuracy": metrics.get("conditional_hop_accuracy"),
                    "conditional_correct_hops": metrics.get("conditional_correct_hops"),
                    "evaluable_hops": metrics.get("evaluable_hops"),
                    "blocked_hops": metrics.get("blocked_hops"),
                    "prefix_correct_hops": metrics.get("prefix_correct_hops"),
                    "prefix_hop_accuracy": metrics.get("prefix_hop_accuracy"),
                    "first_incorrect_hop": metrics.get("first_incorrect_hop"),
                    "strict_chain_success": metrics.get("strict_chain_success"),
                    "final_correct": metrics.get("final_correct"),
                    "chain_complete": metrics.get("chain_complete"),
                    "training_text_count": len(result.training_texts),
                    "prompt_tokens": token_totals["prompt_tokens"],
                    "output_tokens": token_totals["output_tokens"],
                    "total_tokens": token_totals["total_tokens"],
                    "finished_llm_calls": token_totals["finished_llm_calls"],
                    "length_finished_llm_calls": token_totals["length_finished_llm_calls"],
                    "metrics": metrics,
                    "trace_path": str(trace_path),
                }
                break
            except Exception as exc:
                failure_kind = classify_failure(type(exc).__name__, str(exc))
                terminal_failure = failure_kind == "agent" or attempt >= case_attempts
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
                    "attempt_latency_s": round(time.time() - attempt_started, 3),
                    "case_attempt": attempt,
                    "case_attempts": case_attempts,
                    "attempts_exhausted": terminal_failure,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failure_kind": failure_kind,
                    "counts_for_accuracy": failure_kind == "agent",
                }
                attempt_rows.append({**row, "final_index_row": terminal_failure})
                if failure_kind != "agent" and attempt < case_attempts:
                    delay = min(120.0, max(0.0, args.case_retry_delay_s) * (2 ** (attempt - 1)))
                    print(
                        f"retry model={args.model_label} split={split_name} repeat={repeat} "
                        f"chain={chain_id} attempt={attempt}/{case_attempts} delay={delay:.1f}s "
                        f"error={type(exc).__name__}",
                        flush=True,
                    )
                    if delay:
                        await asyncio.sleep(delay)
    async with write_lock:
        for attempt_row in attempt_rows:
            append_jsonl(attempt_path, attempt_row)
        if row.get("ok") or failure_counts_for_accuracy(row):
            append_jsonl(index_path, row)
            if not row.get("ok"):
                append_jsonl(error_path, row)
            status = "ok" if row.get("ok") else "agent_failed"
        else:
            append_jsonl(system_error_path, row)
            status = "system_failed"
        print(
            f"{status} model={args.model_label} split={split_name} repeat={repeat} "
            f"chain={chain_id} attempt={row.get('case_attempt')}/{row.get('case_attempts')} "
            f"latency={row.get('latency_s')}s",
            flush=True,
        )
    return status


async def run_split_cases(
    args: argparse.Namespace,
    cases: list[RunCase],
    llm: Any,
    cfg: Any,
    generate_privacy_hopqa_rollout: Any,
    session: Any,
    index_path: Path,
    error_path: Path,
    attempt_path: Path,
    system_error_path: Path,
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
                        attempt_path,
                        system_error_path,
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

    if args.api_key_env:
        api_token = os.environ.get(args.api_key_env, "")
        if not api_token:
            raise ValueError(f"--api-key-env was set to {args.api_key_env!r}, but that environment variable is empty")
        os.environ["PIPELINERL_LLM_TOKEN"] = api_token
    args.base_url = normalize_openai_base_url(args.base_url)

    pipelinerl_root = Path(args.pipelinerl_root).expanduser()
    (
        OmegaConf,
        LLMEvent,
        LLMOutput,
        LLMStream,
        Prompt,
        TrainableLLM,
        generate_privacy_hopqa_rollout,
    ) = ensure_pipelinerl_imports(pipelinerl_root)

    output_root = Path(args.output_root).expanduser()
    model_label = slugify(args.model_label)
    args.model_label = model_label
    tables_dir = output_root / "tables"
    index_path = tables_dir / "rollout_index.jsonl"
    error_path = tables_dir / "rollout_errors.jsonl"
    attempt_path = tables_dir / "rollout_attempts.jsonl"
    system_error_path = tables_dir / "rollout_system_errors.jsonl"
    provenance_path = output_root / "config" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    if not provenance_path.exists() or args.force_provenance:
        provenance = {
            "created_at": iso_now(),
            "artifact_type": "privacy_hopqa_initial_state_eval",
            "runner_version": RUNNER_VERSION,
            "dataset_root": str(Path(args.dataset_root).expanduser()),
            "dataset_basename": args.dataset_basename,
            "pipelinerl_root": str(pipelinerl_root),
            "task_data_root": str(Path(args.task_data_root).expanduser()),
            "local_index_root": str(Path(args.local_index_root).expanduser()),
            "helper_url": args.helper_url or os.environ.get("PRIVACY_AGENT_HELPER_URL"),
            "models": [{"label": model_label, "model": args.model, "base_url": args.base_url}],
            "splits": args.splits,
            "runner_args": vars(args),
        }
        provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    request_parameters: dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.request_params_json:
        extra_request_parameters = json.loads(args.request_params_json)
        if not isinstance(extra_request_parameters, dict):
            raise ValueError("--request-params-json must decode to a JSON object")
        request_parameters.update(extra_request_parameters)

    class CompletionTemplateLLM(TrainableLLM):
        """Use /v1/completions with a locally rendered chat template.

        GPT-OSS can fail inside vLLM's chat/Harmony response parser even when the
        model generated usable text. The completions endpoint returns raw text
        and avoids that parser while preserving the same chat template.
        """

        completion_template_direct: bool = False

        async def agenerate(self, prompt: Any, session: aiohttp.ClientSession, **kwargs: Any) -> Any:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers |= {"Authorization": f"Bearer {self.api_token}"}
            self.load_tokenizer()
            chat_kwargs = dict(self.chat_template_kwargs) if self.chat_template_kwargs else {}
            direct_generation_prompt = bool(getattr(self, "completion_template_direct", False))
            rendered_prompt = self.tokenizer.apply_chat_template(
                prompt.messages,
                tokenize=False,
                add_generation_prompt=not direct_generation_prompt,
                **chat_kwargs,
            )
            if direct_generation_prompt:
                rendered_prompt = rendered_prompt.rstrip() + "\n<|im_start|>assistant\n"

            def _plain(value: Any) -> Any:
                return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

            params = {"model": self.model_name, "prompt": rendered_prompt}
            for key, value in self.parameters.items():
                params[key] = _plain(value)
            for key, value in kwargs.items():
                params[key] = _plain(value)
            params.pop("include_reasoning", None)
            params.pop("chat_template_kwargs", None)

            async with self._semaphore:  # type: ignore[attr-defined]
                retry_count = 0
                while True:
                    try:
                        async with session.post(
                            url=f"{self.base_url}/v1/completions",
                            json=params,
                            headers=headers,
                            ssl=False,
                        ) as response:
                            if not response.ok:
                                response.raise_for_status()
                            data = await response.json()
                            break
                    except asyncio.TimeoutError as exc:
                        retry_count += 1
                        if retry_count > self.max_retries:
                            raise exc
                        await asyncio.sleep(self.base_delay * (2 ** (retry_count - 1)))
                    except aiohttp.ClientError as exc:
                        retry_count += 1
                        if retry_count > self.max_retries:
                            raise exc
                        await asyncio.sleep(self.base_delay * (2 ** (retry_count - 1)))

            content = data["choices"][0].get("text") or ""
            prompt_tokens = int(data.get("usage", {}).get("prompt_tokens") or 0)
            completion_tokens = int(data.get("usage", {}).get("completion_tokens") or 0)
            output = LLMOutput(content=content)
            if not isinstance(prompt, Prompt):
                prompt = Prompt(**prompt.model_dump(exclude_none=True))
            llm_call = self.log_output(
                prompt,
                output,
                prompt_length_tokens=prompt_tokens,
                output_length_tokens=completion_tokens,
                count_tokens=False,
            )
            assert llm_call is not None, "llm_call is None"

            def _gen():
                yield LLMEvent(llm_call=llm_call, output=output)

            return LLMStream(generator=_gen(), prompt=prompt)

    use_completion_template = args.request_api_mode in {"completion_template", "completion_template_direct"}
    llm_cls = CompletionTemplateLLM if use_completion_template else TrainableLLM
    llm = llm_cls(
        model_name=args.model,
        tokenizer_name=args.tokenizer_name or args.model,
        base_url=args.base_url.rstrip("/"),
        parameters=request_parameters,
    )
    if args.request_api_mode == "completion_template_direct":
        llm.completion_template_direct = True
    if use_completion_template:
        from pipelinerl.domains.privacy_hopqa import llm_adapter as privacy_hopqa_llm_adapter

        async def completion_template_async_generate(
            llm: Any,
            prompt: Any,
            session: aiohttp.ClientSession,
            *,
            parameters_override: dict[str, Any] | None = None,
        ) -> Any:
            # PrivacyHopQALLMAdapter normally calls PipelineRL's async helper,
            # which is hardwired to /v1/chat/completions. Some models need the
            # completions endpoint to avoid chat response parsers or reasoning-only
            # assistant prefill in their chat templates.
            stream = await llm.agenerate(prompt, session, **(parameters_override or {}))
            return stream.get_llm_call()

        privacy_hopqa_llm_adapter.llm_async_generate = completion_template_async_generate

    completed = set() if args.force else load_completed(index_path, model_label, include_failed=args.skip_failed)
    if args.repeat_id:
        repeat_ids = sorted(set(args.repeat_id))
        invalid = [repeat_id for repeat_id in repeat_ids if repeat_id < 0 or repeat_id >= args.repeats]
        if invalid:
            raise ValueError(f"--repeat-id values must be in [0, {args.repeats - 1}], got {invalid}")
    else:
        repeat_ids = list(range(args.repeats))
    if args.row_shard_count < 1:
        raise ValueError("--row-shard-count must be >= 1")
    if args.row_shard_index < 0 or args.row_shard_index >= args.row_shard_count:
        raise ValueError(
            f"--row-shard-index must be in [0, {args.row_shard_count - 1}], got {args.row_shard_index}"
        )
    total_stats = RunStats()
    async with aiohttp.ClientSession() as session:
        for split_name in args.splits:
            split_path = Path(args.dataset_root).expanduser() / f"{args.dataset_basename}_{split_name}.jsonl"
            rows = select_rows(read_jsonl(split_path), args)
            for repeat in repeat_ids:
                if args.repeat_id:
                    split_output_dir = output_root / "rollouts" / model_label / split_name / f"repeat_{repeat}"
                else:
                    split_output_dir = output_root / "rollouts" / model_label / split_name
                split_output_dir.mkdir(parents=True, exist_ok=True)
                cfg = build_cfg(args, split_output_dir, OmegaConf)
                cases: list[RunCase] = []
                for row in rows:
                    key = (split_name, str(row.get("chain_id")), repeat)
                    if key not in completed:
                        cases.append(RunCase(split_name, split_output_dir, row, repeat))
                print(
                    f"model={model_label} split={split_name} repeat={repeat} "
                    f"row_shard={args.row_shard_index}/{args.row_shard_count} "
                    f"queued={len(cases)} rows={len(rows)}",
                    flush=True,
                )
                split_stats = await run_split_cases(
                    args,
                    cases,
                    llm,
                    cfg,
                    generate_privacy_hopqa_rollout,
                    session,
                    index_path,
                    error_path,
                    attempt_path,
                    system_error_path,
                )
                total_stats.queued += split_stats.queued
                total_stats.ok += split_stats.ok
                total_stats.failed += split_stats.failed
                total_stats.skipped += split_stats.skipped
                total_stats.system_failed += split_stats.system_failed
                print(
                    f"split_done model={model_label} split={split_name} repeat={repeat} "
                    f"queued={split_stats.queued} ok={split_stats.ok} failed={split_stats.failed} "
                    f"system_failed={split_stats.system_failed}",
                    flush=True,
                )
    print(
        f"all_done model={model_label} queued={total_stats.queued} ok={total_stats.ok} "
        f"failed={total_stats.failed} system_failed={total_stats.system_failed}",
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
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Name of an environment variable containing the API key. The value is not written to command logs.",
    )
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-basename", default=DEFAULT_DATASET_BASENAME)
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
    parser.add_argument("--repeat-id", action="append", type=int, default=[])
    parser.add_argument("--row-shard-index", type=int, default=0)
    parser.add_argument("--row-shard-count", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-attempts", type=int, default=1)
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Deprecated compatibility flag. Agent-induced failed rows are always terminal; infrastructure/system failures remain retryable.",
    )
    parser.add_argument("--case-retry-delay-s", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--request-params-json",
        default="",
        help="Extra JSON object merged into each OpenAI-compatible chat completion request.",
    )
    parser.add_argument(
        "--request-api-mode",
        choices=["chat", "completion_template", "completion_template_direct"],
        default="chat",
        help="Use chat completions normally, or render chat locally and call /v1/completions.",
    )
    parser.add_argument("--capture-mode", default="all_calls")
    parser.add_argument("--generation-context-limit-tokens", type=int, default=32768)
    parser.add_argument("--generation-context-margin-tokens", type=int, default=128)
    parser.add_argument("--hop-plan-max-tokens", type=int, default=16384)
    parser.add_argument("--doc-choose-max-tokens", type=int, default=16384)
    parser.add_argument("--doc-read-max-tokens", type=int, default=16384)
    parser.add_argument("--hop-resolve-max-tokens", type=int, default=16384)
    parser.add_argument("--answer-match-f1-threshold", type=float, default=0.75)
    parser.add_argument("--max-iterations", type=int, default=21)
    parser.add_argument("--hop-plan-attempts", type=int, default=1)
    parser.add_argument("--planner-privacy-prompt", action="store_true")
    parser.add_argument("--enable-generic-retrieval-fallback", action="store_true")
    parser.add_argument("--max-parallel-retrieval-actions", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--choose-top-k", type=int, default=3)
    parser.add_argument("--max-candidate-cards", type=int, default=20)
    parser.add_argument("--chooser-excerpt-chars", type=int, default=500)
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
    parser.add_argument("--max-parallel-doc-reads", type=int, default=6)
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
