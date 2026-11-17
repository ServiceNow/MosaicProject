"""Score agent trajectories for privacy leakage.

An external observer sees only the agent's web search queries. For each
trajectory this runs three adversaries against those queries alone, then three
judges that compare what the adversary recovered against the gold private facts:

    intent   what was the agent researching?
    answer   what are the answers to the private questions?
    claim    what private facts can you state about the company?

The judge outputs collapse into one 1-5 score (see scoring). Results are
appended to scores.jsonl, failures to errors.jsonl; re-running skips work
already recorded, so an interrupted run resumes.

Serve the judge locally with vLLM:

    mosaicleaks-privacy-eval --traces-root RUN_DIR --dataset data.jsonl \\
        --private-qa-inventory secrets.jsonl --output-dir scores/ \\
        --base-url http://127.0.0.1:8000/v1 --model my-judge

or route it through OpenRouter:

    export MOSAICLEAKS_PRIVACY_API_KEY=sk-or-...
    mosaicleaks-privacy-eval ... --base-url https://openrouter.ai/api/v1 \\
        --model anthropic/claude-opus-4.7
"""

import argparse
import asyncio
import fcntl
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mosaicleaks.privacy import config, inputs, normalize, prompts, scoring
from mosaicleaks.privacy.client import JudgeClient
from mosaicleaks.privacy.parsing import parse_adversary, parse_judge, strip_metadata

@dataclass
class Case:
    """One trajectory, prepared for scoring."""

    trace_path: Path
    ident: dict[str, str]
    dataset_row: dict[str, Any]
    queries: list[dict[str, Any]]
    subquestions: list[dict[str, Any]]
    docs: list[dict[str, Any]]
    private_qas: list[dict[str, Any]]
    local_facts: list[dict[str, Any]]
    missing_doc_keys: list[str]
    qas_truncated: bool
    n_hops: int
    max_items: int


@dataclass
class Verdicts:
    """What the three adversaries produced and the three judges made of it."""

    adversaries: dict[str, dict[str, Any] | None]
    judges: dict[str, dict[str, Any] | None]
    prompts: dict[str, str]
    errors: dict[str, dict[str, Any]]

    def complete(self) -> bool:
        return all(v is not None for v in {**self.adversaries, **self.judges}.values())


@dataclass
class RunConfig:
    """Everything that identifies a scoring run and its inputs."""

    model: str
    api_mode: str
    private_qa_doc_scope: str
    max_private_questions: int
    validation_attempts: int
    force: bool


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row, locked, so concurrent workers can share an output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def already_scored(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["rollout_id"]) for row in inputs.read_jsonl(path) if row.get("ok")}


def check_manifest(output_dir: Path, run_config: dict[str, Any]) -> None:
    """Refuse to resume into a directory scored under different settings.

    Appending rows scored with a different model or document scope to an existing
    scores.jsonl would silently produce a mixed, meaningless table.
    """
    path = output_dir / "run_config.json"
    if not path.exists():
        path.write_text(json.dumps(run_config, indent=2, sort_keys=True))
        return
    previous = json.loads(path.read_text())
    if previous != run_config:
        changed = [k for k in set(previous) | set(run_config) if previous.get(k) != run_config.get(k)]
        raise SystemExit(
            f"{output_dir} was scored with different settings ({', '.join(sorted(changed))}). "
            f"Use a new --output-dir."
        )


async def _judge(
    client: JudgeClient,
    cfg: RunConfig,
    prompt: str,
    stage: str,
    validate,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Call a judge, retrying while its response fails schema validation.

    A judge that returns well-formed JSON with impossible counts is worse than
    one that fails to parse, because the bad numbers would flow into the score.
    """
    last_error: dict[str, Any] = {}
    last_judge: dict[str, Any] | None = None
    for attempt in range(1, cfg.validation_attempts + 1):
        judge, err = await client.complete_json(prompt, lambda text: parse_judge(text, stage))
        if judge is None:
            last_error = {**err, "validation_attempt": attempt}
            continue
        error = validate(judge)
        if not error:
            return judge, err
        last_judge = judge
        last_error = {
            **err,
            "validation_attempt": attempt,
            "validation_error": error,
            "invalid_judge_output": strip_metadata(judge),
        }
    return last_judge, last_error


def prepare(
    trace_path: Path,
    cfg: RunConfig,
    dataset_index: dict[tuple[str, str], dict[str, Any]],
    qa_index: dict[str, list[dict[str, Any]]],
) -> Case:
    """Read one trajectory and gather everything the adversaries and judges need."""
    trace = inputs.read_json(trace_path)
    ident = inputs.trace_identity(trace_path, trace)
    dataset_row = dataset_index[(ident["task_id"], ident["chain_id"])]
    inputs.validate_trace_against_dataset(trace, dataset_row)

    subquestions = inputs.true_subquestions(dataset_row)
    n_hops = int(dataset_row.get("n_hops") or len(subquestions) or 1)
    docs = inputs.retrieved_local_docs(trace, cfg.private_qa_doc_scope)
    private_qas, missing_doc_keys, qas_truncated = inputs.private_qas_for_docs(
        qa_index, docs, cfg.max_private_questions
    )
    return Case(
        trace_path=trace_path,
        ident=ident,
        dataset_row=dataset_row,
        queries=inputs.query_rows(trace, "web"),
        subquestions=subquestions,
        docs=docs,
        private_qas=private_qas,
        local_facts=inputs.local_hop_facts(dataset_row),
        missing_doc_keys=missing_doc_keys,
        qas_truncated=qas_truncated,
        n_hops=n_hops,
        # The adversary gets twice the hop count of guesses: enough room to be
        # right without letting it carpet-bomb every possible subquestion.
        max_items=max(1, 2 * n_hops),
    )


async def run_adversaries(client: JudgeClient, case: Case) -> Verdicts:
    """Three reconstructions from the visible web queries alone."""
    prompt_text = {
        "intent": prompts.prompt_intent_adversary(case.queries, case.max_items),
        "answer": prompts.prompt_answer_adversary(case.queries, case.private_qas),
        "claim": prompts.prompt_claim_adversary(case.queries, case.max_items),
    }
    keys = {"intent": "predicted_subquestions", "answer": "answers", "claim": "claims"}
    results = await asyncio.gather(*[
        client.complete_json(prompt_text[stage], lambda t, k=keys[stage]: parse_adversary(t, k))
        for stage in ("intent", "answer", "claim")
    ])

    outputs, errors = {}, {}
    for stage, (parsed, error) in zip(("intent", "answer", "claim"), results):
        outputs[stage], errors[f"{stage}_adversary"] = parsed, error
    if outputs["intent"] is not None:
        outputs["intent"] = normalize.normalize_intent_output(outputs["intent"])
    if outputs["answer"] is not None:
        outputs["answer"] = normalize.normalize_answer_output(outputs["answer"])
    if outputs["claim"] is not None:
        outputs["claim"] = normalize.normalize_claim_output(outputs["claim"], case.max_items)

    return Verdicts(
        adversaries=outputs,
        judges={"intent": None, "answer": None, "claim": None},
        prompts={f"{stage}_adversary": prompt_text[stage] for stage in prompt_text},
        errors=errors,
    )


async def run_judges(client: JudgeClient, cfg: RunConfig, case: Case, verdicts: Verdicts) -> None:
    """Grade each reconstruction against the gold facts, in place on `verdicts`."""
    if any(v is None for v in verdicts.adversaries.values()):
        for stage in ("intent", "answer", "claim"):
            verdicts.errors[f"{stage}_judge"] = {"error": "skipped_due_adversary_failure"}
            verdicts.prompts[f"{stage}_judge"] = ""
        return

    intent_out = verdicts.adversaries["intent"]
    answer_out = verdicts.adversaries["answer"]
    claim_out = verdicts.adversaries["claim"]
    valid_fact_ids = normalize.private_fact_id_set(case.local_facts, case.private_qas)

    # An adversary that produced nothing needs no judge call, so those stages get
    # an empty verdict without going near the network.
    prompt_text = {
        "intent": prompts.prompt_intent_judge(case.subquestions, intent_out),
        "answer": prompts.prompt_answer_judge(case.private_qas, answer_out) if answer_out.get("answers") else "",
        "claim": prompts.prompt_claim_judge(case.local_facts, case.private_qas, claim_out) if claim_out.get("claims") else "",
    }
    checks = {
        "intent": normalize.intent_judge_validation_error,
        "answer": lambda j: normalize.answer_judge_validation_error(j, case.private_qas, answer_out),
        "claim": lambda j: normalize.claim_judge_validation_error(j, claim_out, valid_fact_ids),
    }

    called = [stage for stage in ("intent", "answer", "claim") if prompt_text[stage]]
    results = dict(zip(called, await asyncio.gather(*[
        _judge(client, cfg, prompt_text[stage], f"{stage}_judge", checks[stage]) for stage in called
    ])))
    results.setdefault("answer", (normalize.no_answer_judge(len(case.private_qas)), {}))
    results.setdefault("claim", (normalize.no_claim_judge(), {}))

    for stage in ("intent", "answer", "claim"):
        judge, error = results[stage]
        if judge is not None and stage == "answer":
            judge = normalize.normalize_answer_judge_output(judge, case.private_qas, answer_out)
        if judge is not None and stage == "claim":
            judge = normalize.normalize_claim_judge_output(judge, valid_fact_ids)
        # Re-check after normalisation: a judge can pass on its raw output and
        # still be inconsistent once its counts are reconciled against the adversary.
        if judge is not None and prompt_text[stage]:
            failure = checks[stage](judge)
            if failure:
                judge, error = None, {**error, "validation_error": failure}
        verdicts.judges[stage] = judge
        verdicts.errors[f"{stage}_judge"] = error
        verdicts.prompts[f"{stage}_judge"] = prompt_text[stage]


def score_row(cfg: RunConfig, case: Case, verdicts: Verdicts) -> dict[str, Any]:
    """The row written to scores.jsonl for one fully judged trajectory."""
    overall = scoring.derive_overall(
        verdicts.judges["intent"], verdicts.judges["answer"], verdicts.judges["claim"]
    )
    return {
        **_common_fields(cfg, case),
        "ok": True,
        "model_input": {
            "visible_to_adversary": "web_search queries only",
            "web_queries": case.queries,
            "private_qa_questions_visible_to_answer_adversary": True,
            "private_qa_answers_visible_to_answer_adversary": False,
        },
        "hidden_reference_for_judges_only": {
            "true_subquestions": case.subquestions,
            "retrieved_local_docs": case.docs,
            "missing_private_qa_doc_keys": case.missing_doc_keys[:100],
            "private_qas": case.private_qas,
            "local_hop_facts": case.local_facts,
        },
        "prompts": {
            name: {"system": config.SYSTEM_PROMPT, "user": text}
            for name, text in verdicts.prompts.items()
        },
        "intent_adversary_output": verdicts.adversaries["intent"],
        "intent_judge_output": verdicts.judges["intent"],
        "answer_adversary_output": verdicts.adversaries["answer"],
        "answer_judge_output": verdicts.judges["answer"],
        "claim_adversary_output": verdicts.adversaries["claim"],
        "claim_judge_output": verdicts.judges["claim"],
        **overall,
        "reward_model_target": {
            "label": int(overall["overall_leaked_for_reward"]),
            "label_name": "primary_privacy_leakage_full_or_partial_answer_or_claim",
            "source_model": cfg.model,
        },
    }


def _common_fields(cfg: RunConfig, case: Case) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "model": cfg.model,
        "api_mode": cfg.api_mode,
        "trace_path": str(case.trace_path),
        **case.ident,
        "split": case.dataset_row.get("_split"),
        "company": case.dataset_row.get("company"),
        "pattern": case.dataset_row.get("pattern"),
        "n_hops": case.n_hops,
        "n_web_queries_visible": len(case.queries),
        "n_retrieved_local_docs": len(case.docs),
        "n_private_qas": len(case.private_qas),
        "private_qas_truncated": case.qas_truncated,
        "n_missing_private_qa_docs": len(case.missing_doc_keys),
        "private_qa_doc_scope": cfg.private_qa_doc_scope,
    }


async def score_trace(
    client: JudgeClient,
    cfg: RunConfig,
    trace_path: Path,
    dataset_index: dict[tuple[str, str], dict[str, Any]],
    qa_index: dict[str, list[dict[str, Any]]],
    lock: asyncio.Lock,
    done: set[str],
    scores_path: Path,
    errors_path: Path,
) -> bool:
    try:
        case = prepare(trace_path, cfg, dataset_index, qa_index)
    except Exception as exc:
        async with lock:
            append_jsonl(errors_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ok": False,
                "trace_path": str(trace_path),
                "error": f"prepare:{type(exc).__name__}: {exc}",
            })
        return False

    if case.ident["rollout_id"] in done and not cfg.force:
        return True

    verdicts = await run_adversaries(client, case)
    await run_judges(client, cfg, case, verdicts)

    async with lock:
        if not verdicts.complete():
            append_jsonl(errors_path, {**_common_fields(cfg, case), **verdicts.errors})
            print(f"fail  {case.ident['rollout_id']}", flush=True)
            return False

        row = score_row(cfg, case, verdicts)
        append_jsonl(scores_path, row)
        done.add(case.ident["rollout_id"])
        print(f"ok    {case.ident['rollout_id']} score={row['overall_privacy_score_1_to_5']} "
              f"intent={verdicts.judges['intent'].get('intent_subquestion_score_1_to_5')} "
              f"pqa={len(case.private_qas)}", flush=True)
        return True


def collect_traces(args: argparse.Namespace) -> list[Path]:
    if args.trace_manifest:
        paths = inputs.read_trace_manifest(args.trace_manifest)
    else:
        paths = inputs.trace_paths_from_artifact(args.traces_root)
    if args.limit and len(paths) > args.limit:
        rng = random.Random(args.seed)
        paths = sorted(rng.sample(paths, args.limit))
    return paths


async def run(args: argparse.Namespace) -> int:
    dataset_index = inputs.load_dataset_index(args.dataset, args.dataset_basename)
    qa_index = inputs.load_private_qa_index(args.private_qa_inventory)
    trace_paths = collect_traces(args)

    output_dir = args.output_dir
    scores_path = output_dir / "scores.jsonl"
    errors_path = output_dir / "errors.jsonl"
    done = already_scored(scores_path)
    check_manifest(output_dir, {
        "model": args.model,
        "api_mode": args.api_mode,
        "private_qa_doc_scope": args.private_qa_doc_scope,
        "max_private_questions": args.max_private_questions,
        "dataset": str(args.dataset),
        "dataset_basename": args.dataset_basename,
        "private_qa_inventory": str(args.private_qa_inventory),
    })

    cfg = RunConfig(
        model=args.model,
        api_mode=args.api_mode,
        private_qa_doc_scope=args.private_qa_doc_scope,
        max_private_questions=args.max_private_questions,
        validation_attempts=args.validation_attempts,
        force=args.force,
    )
    client = JudgeClient(
        model=args.model,
        base_url=args.base_url,
        api_key=os.getenv(config.JUDGE_API_KEY_ENV, "EMPTY"),
        api_mode=args.api_mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_attempts=args.attempts,
        concurrency=args.concurrency,
        use_response_format=not args.no_response_format,
    )

    print(
        f"traces={len(trace_paths)} already_scored={len(done)} model={args.model} "
        f"base_url={args.base_url} api_mode={args.api_mode} concurrency={args.concurrency}",
        flush=True,
    )

    lock = asyncio.Lock()
    results = await asyncio.gather(*[
        score_trace(client, cfg, path, dataset_index, qa_index, lock, done, scores_path, errors_path)
        for path in trace_paths
    ], return_exceptions=True)

    ok = sum(1 for r in results if r is True)
    crashed = [(path, r) for path, r in zip(trace_paths, results) if isinstance(r, BaseException)]
    for path, exc in crashed:
        append_jsonl(errors_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "trace_path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        })

    print(f"\nscored {ok}/{len(results)} -> {scores_path}", flush=True)
    if crashed:
        print(f"{len(crashed)} crashed, written to {errors_path}", flush=True)
    return 0 if ok == len(results) else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Score agent trajectories for privacy leakage.")

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces-root", type=Path, help="Run directory to walk for trace.json files.")
    source.add_argument("--trace-manifest", type=Path, help="File listing trace.json paths, one per line.")

    p.add_argument("--dataset", type=Path, required=True,
                   help="Published dataset bundle directory holding train/val/test.jsonl.")
    p.add_argument("--dataset-basename", default="",
                   help="For a frozen release whose splits are named {basename}_{split}.jsonl.")
    p.add_argument("--private-qa-inventory", type=Path, required=True, help="Private QA inventory JSONL.")
    p.add_argument("--output-dir", type=Path, required=True, help="Destination for scores.jsonl and errors.jsonl.")

    p.add_argument("--model", default=config.JUDGE_MODEL)
    p.add_argument("--base-url", default=config.JUDGE_BASE_URL,
                   help="OpenAI-compatible endpoint: a local vLLM server or https://openrouter.ai/api/v1.")
    p.add_argument("--api-mode", choices=["chat", "completion_template"], default="chat",
                   help="completion_template applies ChatML by hand; needed to reproduce the reported runs.")
    p.add_argument("--no-response-format", action="store_true",
                   help="Skip response_format=json_object for endpoints that reject it.")

    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    p.add_argument("--timeout", type=float, default=config.REQUEST_TIMEOUT)
    p.add_argument("--attempts", type=int, default=config.MAX_ATTEMPTS, help="Retries per request.")
    p.add_argument("--validation-attempts", type=int, default=config.VALIDATION_ATTEMPTS,
                   help="Retries when a judge fails validation.")
    p.add_argument("--concurrency", type=int, default=config.CONCURRENCY)

    p.add_argument("--private-qa-doc-scope", choices=["read", "retrieved"], default="read",
                   help="Hold the agent against private facts in documents it read, or in all it retrieved.")
    p.add_argument("--max-private-questions", type=int, default=config.MAX_PRIVATE_QUESTIONS)
    p.add_argument("--limit", type=int, default=0, help="Score a random sample of this many inputs.")
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--force", action="store_true", help="Re-score traces already present in scores.jsonl.")

    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
