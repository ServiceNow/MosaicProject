#!/usr/bin/env python3
"""Browse privacy_hopqa answer verification results in a terminal UI.

The fast path reads the rollouts_sample.csv produced by plot_privacy_hopqa_runtime.py.
It can also scan a full privacy_hopqa run directory if a curated chains JSONL is
available for the ground-truth answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ANALYSIS_ROOT = REPO_ROOT / "runtime_analysis"
DEFAULT_RUN_DIR = Path(
    "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_qwen3_8b_async_roll32_retry3_20260429_205615"
)
DEFAULT_CURATED_CHAINS = Path("/home/toolkit/nice_code/drbench/chain_review/curated_chains.jsonl")
DEFAULT_ANNOTATION_NAME = "privacy_hopqa_human_judgements.jsonl"
DEFAULT_CODEX_SAMPLE_NAME = "codex_answer_judgements_200.jsonl"

NO_ANSWER_RE = re.compile(
    r"\b(not[_ ]?found|not answerable|cannot be (?:answered|determined)|not specified|available information)\b",
    re.IGNORECASE,
)


@dataclass
class AnswerExample:
    example_id: str
    rollout_id: str
    task_id: str
    chain_id: str
    answer_kind: str
    hop: int | None
    question: str
    predicted: str
    truth: str
    raw_f1: float
    match_score: float
    verifier_correct: bool
    rollout_reward: float
    rollout_binary_reward: int
    final_correct: bool | None
    trace_path: str
    report_path: str
    source_path: str
    codex_judgement: str = ""
    codex_notes: str = ""
    human_judgement: str = ""
    human_notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help=(
            "Analysis directory, rollouts CSV, or full privacy_hopqa run directory. "
            "Defaults to the newest MosaicProject runtime_analysis/privacy_hopqa* bundle."
        ),
    )
    parser.add_argument(
        "--curated-chains",
        type=Path,
        default=DEFAULT_CURATED_CHAINS,
        help="Curated chains JSONL used when scanning a full run directory.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help=f"Judgement JSONL path. Defaults to <analysis-dir>/{DEFAULT_ANNOTATION_NAME}.",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "final", "hops"],
        default="all",
        help="Which answer comparisons to show or sample.",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="Maximum rollout rows/traces to load before expanding to answer examples.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of examples to export with --make-codex-sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Random seed for sample export.",
    )
    parser.add_argument(
        "--make-codex-sample",
        type=Path,
        nargs="?",
        const=Path("__DEFAULT__"),
        help=(
            "Write a seeded JSONL sample with Codex judgement labels, then exit. "
            f"Default output is <analysis-dir>/{DEFAULT_CODEX_SAMPLE_NAME}."
        ),
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print aggregate verifier/judgement stats and exit instead of launching the TUI.",
    )
    parser.add_argument(
        "--table-window-size",
        type=int,
        default=60,
        help="Maximum rows rendered in the table pane at once. Smaller values are faster.",
    )
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Show source evidence panels immediately instead of toggling them with 'e'.",
    )
    return parser.parse_args()


def normalize_answer(value: str | None) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"\$\s*", "", text)
    text = re.sub(r"(\d(?:\.\d+)?)\s*b\b", r"\1 billion", text)
    text = re.sub(r"(\d(?:\.\d+)?)\s*m\b", r"\1 million", text)
    text = re.sub(r"(\d(?:\.\d+)?)\s*k\b", r"\1 thousand", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1_score(predicted: str | None, truth: str | None) -> float:
    pred = normalize_answer(predicted)
    gold = normalize_answer(truth)
    if not pred or not gold:
        return float(pred == gold)
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answers_match(predicted: str | None, truth: str | None, f1_threshold: float = 0.75) -> tuple[bool, float]:
    pred = normalize_answer(predicted)
    gold = normalize_answer(truth)
    if not pred or not gold:
        score = float(pred == gold)
        return bool(score), score
    if pred == gold or gold in pred or pred in gold:
        return True, 1.0
    score = token_f1_score(pred, gold)
    return score >= f1_threshold, score


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_latest_analysis_dir(root: Path = DEFAULT_RUNTIME_ANALYSIS_ROOT) -> Path:
    candidates = [
        path
        for path in root.glob("privacy_hopqa*")
        if path.is_dir() and (path / "rollouts_sample.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No privacy_hopqa analysis bundles with rollouts_sample.csv under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_data_path(path: Path | None) -> tuple[str, Path]:
    if path is None:
        analysis_dir = find_latest_analysis_dir()
        return "analysis_csv", analysis_dir / "rollouts_sample.csv"

    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".csv":
        return "analysis_csv", path
    if path.is_dir() and (path / "rollouts_sample.csv").is_file():
        return "analysis_csv", path / "rollouts_sample.csv"
    if path.is_dir() and (path / "privacy_hopqa_runs").is_dir():
        return "run_dir", path
    raise FileNotFoundError(
        f"Could not identify {path} as a rollouts CSV, analysis directory, or privacy_hopqa run directory"
    )


def default_annotation_path(data_kind: str, data_path: Path) -> Path:
    if data_kind == "analysis_csv":
        return data_path.parent / DEFAULT_ANNOTATION_NAME
    return data_path / DEFAULT_ANNOTATION_NAME


def default_codex_sample_path(data_kind: str, data_path: Path) -> Path:
    if data_kind == "analysis_csv":
        return data_path.parent / DEFAULT_CODEX_SAMPLE_NAME
    return data_path / DEFAULT_CODEX_SAMPLE_NAME


def load_curated_chains(path: Path) -> dict[str, dict[str, Any]]:
    by_chain: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path.expanduser()):
        chain = dict(row.get("original_chain") or row)
        chain_id = str(row.get("chain_id") or chain.get("chain_id") or "")
        if chain_id:
            by_chain[chain_id] = {
                "task_id": str(row.get("task_id") or (chain.get("metadata") or {}).get("task_id") or ""),
                "chain_id": chain_id,
                "global_question": str(chain.get("global_question") or chain.get("numbered_questions") or ""),
                "global_answer": str(chain.get("global_answer") or ""),
                "hops": list(chain.get("hops") or []),
            }
    return by_chain


def question_for(curated: dict[str, Any] | None, hop_num: int | None, final: bool = False) -> str:
    if not curated:
        return ""
    if final:
        return str(curated.get("global_question") or "")
    for hop in curated.get("hops") or []:
        if int(hop.get("hop_number") or -1) == hop_num:
            return str(hop.get("question") or "")
    return ""


def filter_scope(examples: list[AnswerExample], scope: str) -> list[AnswerExample]:
    if scope == "final":
        return [example for example in examples if example.answer_kind == "final"]
    if scope == "hops":
        return [example for example in examples if example.answer_kind == "hop"]
    return examples


def load_examples_from_csv(
    csv_path: Path,
    *,
    curated_chains: dict[str, dict[str, Any]] | None = None,
    max_rollouts: int | None = None,
) -> list[AnswerExample]:
    rows = read_csv_rows(csv_path)
    if max_rollouts is not None:
        rows = rows[:max_rollouts]
    examples: list[AnswerExample] = []
    for row in rows:
        rollout_id = str(row.get("rollout_id") or "")
        if not rollout_id:
            continue
        task_id = str(row.get("task_id") or "")
        chain_id = str(row.get("chain_id") or "")
        curated = (curated_chains or {}).get(chain_id)
        rollout_reward = as_float(row.get("reward"))
        rollout_binary_reward = int(rollout_reward >= 1.0)
        final_correct = as_bool(row.get("final_correct"))
        total_hops = int(as_float(row.get("total_hops") or row.get("hop_count") or 0))
        if total_hops <= 0:
            hop_nums = []
            for key in row:
                match = re.match(r"hop_(\d+)_answer$", key)
                if match:
                    hop_nums.append(int(match.group(1)))
            total_hops = max(hop_nums, default=0)

        for hop_num in range(1, total_hops + 1):
            predicted = str(row.get(f"hop_{hop_num}_answer") or "")
            truth = str(row.get(f"hop_{hop_num}_gold") or "")
            raw_f1 = token_f1_score(predicted, truth)
            match_score = as_float(row.get(f"hop_{hop_num}_match_score"), default=raw_f1)
            correct = as_bool(row.get(f"hop_{hop_num}_correct"))
            if correct is None:
                correct = answers_match(predicted, truth)[0]
            examples.append(
                AnswerExample(
                    example_id=f"{rollout_id}:hop{hop_num}",
                    rollout_id=rollout_id,
                    task_id=task_id,
                    chain_id=chain_id,
                    answer_kind="hop",
                    hop=hop_num,
                    question=question_for(curated, hop_num),
                    predicted=predicted,
                    truth=truth,
                    raw_f1=raw_f1,
                    match_score=match_score,
                    verifier_correct=bool(correct),
                    rollout_reward=rollout_reward,
                    rollout_binary_reward=rollout_binary_reward,
                    final_correct=final_correct,
                    trace_path=str(row.get("trace_path") or ""),
                    report_path=str(row.get("report_path") or ""),
                    source_path=str(csv_path),
                )
            )

        predicted_final = str(row.get("predicted_final_answer") or row.get(f"hop_{total_hops}_answer") or "")
        truth_final = str(row.get("ground_truth_final_answer") or "")
        if predicted_final or truth_final:
            raw_f1 = token_f1_score(predicted_final, truth_final)
            verifier_final_correct, final_match_score = answers_match(predicted_final, truth_final)
            if final_correct is not None:
                verifier_final_correct = final_correct
            examples.append(
                AnswerExample(
                    example_id=f"{rollout_id}:final",
                    rollout_id=rollout_id,
                    task_id=task_id,
                    chain_id=chain_id,
                    answer_kind="final",
                    hop=None,
                    question=question_for(curated, None, final=True),
                    predicted=predicted_final,
                    truth=truth_final,
                    raw_f1=raw_f1,
                    match_score=final_match_score,
                    verifier_correct=bool(verifier_final_correct),
                    rollout_reward=rollout_reward,
                    rollout_binary_reward=rollout_binary_reward,
                    final_correct=final_correct,
                    trace_path=str(row.get("trace_path") or ""),
                    report_path=str(row.get("report_path") or ""),
                    source_path=str(csv_path),
                )
            )
    return examples


def load_examples_from_run_dir(
    run_dir: Path,
    *,
    curated_chains: dict[str, dict[str, Any]],
    max_rollouts: int | None = None,
) -> list[AnswerExample]:
    trace_paths = sorted((run_dir / "privacy_hopqa_runs").glob("*/trace.json"))
    if max_rollouts is not None:
        trace_paths = trace_paths[:max_rollouts]

    examples: list[AnswerExample] = []
    for trace_path in trace_paths:
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = dict(trace.get("summary") or {})
        rollout_id = trace_path.parent.name
        task_id = str(summary.get("task_id") or rollout_id.split("_", 1)[0])
        chain_id = str(summary.get("chain_id") or "")
        if not chain_id:
            parts = rollout_id.split("_")
            chain_id = parts[1] if len(parts) >= 2 else ""
        curated = curated_chains.get(chain_id)
        if not curated:
            continue

        hop_truth = {
            int(hop.get("hop_number")): str(hop.get("answer") or "")
            for hop in curated.get("hops") or []
            if hop.get("hop_number") is not None
        }
        hop_states = sorted(trace.get("hop_states") or [], key=lambda item: int(item.get("hop_number") or 0))
        total_hops = max(len(hop_truth), 1)
        correct_count = 0
        hop_examples: list[AnswerExample] = []
        for hop_state in hop_states:
            hop_num = int(hop_state.get("hop_number") or 0)
            if hop_num <= 0:
                continue
            predicted = str(hop_state.get("answer") or "")
            truth = hop_truth.get(hop_num, "")
            correct, match_score = answers_match(predicted, truth)
            correct_count += int(correct)
            raw_f1 = token_f1_score(predicted, truth)
            hop_examples.append(
                AnswerExample(
                    example_id=f"{rollout_id}:hop{hop_num}",
                    rollout_id=rollout_id,
                    task_id=task_id,
                    chain_id=chain_id,
                    answer_kind="hop",
                    hop=hop_num,
                    question=question_for(curated, hop_num),
                    predicted=predicted,
                    truth=truth,
                    raw_f1=raw_f1,
                    match_score=match_score,
                    verifier_correct=correct,
                    rollout_reward=0.0,
                    rollout_binary_reward=0,
                    final_correct=None,
                    trace_path=str(trace_path),
                    report_path=str(trace.get("report_path") or ""),
                    source_path=str(run_dir),
                )
            )

        rollout_reward = correct_count / total_hops
        rollout_binary_reward = int(rollout_reward >= 1.0)
        final_predicted = hop_examples[-1].predicted if hop_examples else ""
        final_truth = str(curated.get("global_answer") or "")
        final_correct, final_match_score = answers_match(final_predicted, final_truth)
        final_raw_f1 = token_f1_score(final_predicted, final_truth)

        for example in hop_examples:
            example.rollout_reward = rollout_reward
            example.rollout_binary_reward = rollout_binary_reward
            example.final_correct = final_correct
            examples.append(example)

        examples.append(
            AnswerExample(
                example_id=f"{rollout_id}:final",
                rollout_id=rollout_id,
                task_id=task_id,
                chain_id=chain_id,
                answer_kind="final",
                hop=None,
                question=question_for(curated, None, final=True),
                predicted=final_predicted,
                truth=final_truth,
                raw_f1=final_raw_f1,
                match_score=final_match_score,
                verifier_correct=final_correct,
                rollout_reward=rollout_reward,
                rollout_binary_reward=rollout_binary_reward,
                final_correct=final_correct,
                trace_path=str(trace_path),
                report_path=str(trace.get("report_path") or ""),
                source_path=str(run_dir),
            )
        )
    return examples


def load_judgements(path: Path) -> dict[str, dict[str, Any]]:
    judgements: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        example_id = str(row.get("example_id") or "")
        if example_id:
            judgements[example_id] = row
    return judgements


def apply_judgements(examples: list[AnswerExample], judgement_rows: dict[str, dict[str, Any]]) -> None:
    for example in examples:
        row = judgement_rows.get(example.example_id)
        if not row:
            continue
        example.codex_judgement = str(row.get("codex_judgement") or example.codex_judgement or "")
        example.codex_notes = str(row.get("codex_notes") or example.codex_notes or "")
        example.human_judgement = str(row.get("human_judgement") or example.human_judgement or "")
        example.human_notes = str(row.get("human_notes") or example.human_notes or "")


def extract_numbers(text: str) -> list[float]:
    clean = text.lower().replace(",", "")
    values: list[float] = []
    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?\s*(?:billion|million|thousand|bn|m|k|%)?", clean):
        token = match.group(0).strip()
        number_match = re.match(r"[-+]?\d+(?:\.\d+)?", token)
        if not number_match:
            continue
        number = float(number_match.group(0))
        if any(unit in token for unit in ("billion", "bn")):
            number *= 1_000_000_000
        elif re.search(r"\bm\b", token) or "million" in token:
            number *= 1_000_000
        elif re.search(r"\bk\b", token) or "thousand" in token:
            number *= 1_000
        values.append(number)
    return values


def codex_judge_pair(predicted: str, truth: str, *, raw_f1: float, verifier_correct: bool) -> tuple[str, str]:
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(truth)

    if not pred_norm and not truth_norm:
        return "correct", "both empty"
    if not pred_norm or pred_norm in {"not_found", "not found"}:
        return "incorrect", "prediction is empty or NOT_FOUND"
    if truth_norm and NO_ANSWER_RE.search(predicted) and not NO_ANSWER_RE.search(truth):
        return "incorrect", "prediction says unanswerable but gold has an answer"
    if pred_norm == truth_norm:
        return "correct", "normalized exact match"
    if truth_norm and (truth_norm in pred_norm or pred_norm in truth_norm):
        return "correct", "normalized substring match"

    pred_numbers = extract_numbers(predicted)
    truth_numbers = extract_numbers(truth)
    if pred_numbers and truth_numbers:
        best_rel = min(
            abs(pred - gold) / max(abs(gold), 1.0)
            for pred in pred_numbers
            for gold in truth_numbers
        )
        if best_rel <= 0.01:
            return "correct", "numeric value within 1%"
        if best_rel <= 0.05:
            return "partial", "numeric value within 5%"

    if verifier_correct:
        return "correct", "verifier marked correct"
    if raw_f1 >= 0.75:
        return "correct", "raw token F1 >= 0.75"
    if raw_f1 >= 0.45:
        return "partial", "raw token F1 between 0.45 and 0.75"
    return "incorrect", "low lexical overlap"


def make_codex_sample(examples: list[AnswerExample], output_path: Path, *, size: int, seed: int) -> None:
    rng = random.Random(seed)
    candidates = list(examples)
    if len(candidates) > size:
        candidates = rng.sample(candidates, size)
    candidates.sort(key=lambda example: (example.task_id, example.chain_id, example.rollout_id, example.answer_kind, example.hop or 999))
    rows: list[dict[str, Any]] = []
    for example in candidates:
        judgement, notes = codex_judge_pair(
            example.predicted,
            example.truth,
            raw_f1=example.raw_f1,
            verifier_correct=example.verifier_correct,
        )
        row = asdict(example)
        row.update(
            {
                "codex_judgement": judgement,
                "codex_notes": notes,
                "human_judgement": "",
                "human_notes": "",
                "sample_seed": seed,
            }
        )
        rows.append(row)
    write_jsonl(output_path, rows)


def judgement_value(label: str) -> float | None:
    normalized = label.strip().lower()
    if normalized in {"correct", "yes", "y", "1", "true"}:
        return 1.0
    if normalized in {"partial", "partly", "0.5"}:
        return 0.5
    if normalized in {"incorrect", "wrong", "no", "n", "0", "false"}:
        return 0.0
    return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def build_summary(examples: list[AnswerExample]) -> dict[str, Any]:
    labelled = [
        (example, judgement_value(example.human_judgement or example.codex_judgement))
        for example in examples
    ]
    labelled = [(example, value) for example, value in labelled if value is not None]
    raw_f1_x = [example.raw_f1 for example, _ in labelled]
    match_x = [example.match_score for example, _ in labelled]
    verifier_x = [float(example.verifier_correct) for example, _ in labelled]
    y = [float(value) for _, value in labelled]
    verifier_disagree = [
        example
        for example, value in labelled
        if float(example.verifier_correct) != (1.0 if value >= 0.75 else 0.0)
    ]
    return {
        "examples": len(examples),
        "labelled": len(labelled),
        "mean_raw_f1": statistics.mean([example.raw_f1 for example in examples]) if examples else None,
        "mean_match_score": statistics.mean([example.match_score for example in examples]) if examples else None,
        "verifier_correct_rate": statistics.mean([float(example.verifier_correct) for example in examples]) if examples else None,
        "raw_f1_corr": pearson(raw_f1_x, y),
        "match_score_corr": pearson(match_x, y),
        "verifier_binary_corr": pearson(verifier_x, y),
        "verifier_disagreements": len(verifier_disagree),
    }


def format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def print_summary(examples: list[AnswerExample]) -> None:
    summary = build_summary(examples)
    print(f"examples: {summary['examples']}")
    print(f"labelled: {summary['labelled']}")
    print(f"mean_raw_f1: {format_optional_float(summary['mean_raw_f1'])}")
    print(f"mean_match_score: {format_optional_float(summary['mean_match_score'])}")
    print(f"verifier_correct_rate: {format_optional_float(summary['verifier_correct_rate'])}")
    print(f"corr(raw_f1, judgement): {format_optional_float(summary['raw_f1_corr'])}")
    print(f"corr(match_score, judgement): {format_optional_float(summary['match_score_corr'])}")
    print(f"corr(verifier_correct, judgement): {format_optional_float(summary['verifier_binary_corr'])}")
    print(f"verifier_disagreements: {summary['verifier_disagreements']}")


def truncate_text(value: Any, limit: int = 1200) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def trace_source_context(item: AnswerExample, trace: dict[str, Any] | None) -> str:
    if not item.trace_path:
        return "No trace_path recorded for this row."
    if trace is None:
        return f"Could not load trace: {item.trace_path}"

    hop_states = sorted(trace.get("hop_states") or [], key=lambda row: int(row.get("hop_number") or 0))
    if not hop_states:
        return f"Trace has no hop_states: {item.trace_path}"

    if item.answer_kind == "final":
        hop_state = hop_states[-1]
    else:
        hop_state = next(
            (row for row in hop_states if int(row.get("hop_number") or 0) == int(item.hop or 0)),
            None,
        )
        if hop_state is None:
            return f"No hop_state found for hop {item.hop} in {item.trace_path}"

    selected_doc_ids = [str(doc_id) for doc_id in hop_state.get("selected_doc_ids") or []]
    lines = [
        f"trace: {item.trace_path}",
        f"hop: {hop_state.get('hop_number')}  status: {hop_state.get('status')}  confidence: {hop_state.get('confidence')}",
        f"model_answer: {truncate_text(hop_state.get('answer'), 700)}",
    ]
    if hop_state.get("justification"):
        lines.append(f"justification: {truncate_text(hop_state.get('justification'), 1200)}")
    if hop_state.get("resolution_reason"):
        lines.append(f"resolution: {truncate_text(hop_state.get('resolution_reason'), 1000)}")
    if selected_doc_ids:
        lines.append("selected_doc_ids: " + ", ".join(selected_doc_ids))

    reader_results = list(hop_state.get("reader_results") or [])
    if reader_results:
        lines.append("\nREADER RESULTS")
        for result in reader_results[:8]:
            doc_id = str(result.get("doc_id") or "")
            lines.append(
                "\n".join(
                    [
                        f"- {doc_id} | {result.get('source')} | can_answer={result.get('can_answer')} | confidence={result.get('confidence')}",
                        f"  title: {truncate_text(result.get('title'), 250)}",
                        f"  locator: {truncate_text(result.get('locator'), 500)}",
                        f"  proposed_answer: {truncate_text(result.get('proposed_answer'), 500)}",
                        f"  justification: {truncate_text(result.get('justification'), 1200)}",
                    ]
                )
            )
            if result.get("missing_information"):
                lines.append(f"  missing: {truncate_text(result.get('missing_information'), 700)}")

    candidate_cards = list(hop_state.get("candidate_cards") or [])
    selected_cards = [
        card for card in candidate_cards if not selected_doc_ids or str(card.get("doc_id") or "") in selected_doc_ids
    ]
    if not selected_cards:
        selected_cards = candidate_cards[:5]
    if selected_cards:
        lines.append("\nCANDIDATE SNIPPETS")
        for card in selected_cards[:8]:
            lines.append(
                "\n".join(
                    [
                        f"- {card.get('doc_id')} | {card.get('source')} | rank={card.get('best_rank')}",
                        f"  title: {truncate_text(card.get('title'), 350)}",
                        f"  queries: {truncate_text(', '.join(str(query) for query in card.get('top_queries') or []), 500)}",
                        f"  snippet: {truncate_text(card.get('snippet'), 1000)}",
                    ]
                )
            )

    search_history = list(hop_state.get("search_history") or [])
    if search_history:
        lines.append("\nSEARCH HISTORY")
        for search in search_history[:8]:
            lines.append(
                f"- {search.get('type')} | {truncate_text(search.get('query'), 350)} | top={search.get('top_doc_ids')}"
            )

    return "\n".join(lines)


def load_examples(args: argparse.Namespace) -> tuple[list[AnswerExample], str, Path, Path]:
    data_kind, data_path = resolve_data_path(args.data)
    annotation_path = args.annotations.expanduser() if args.annotations else default_annotation_path(data_kind, data_path)
    curated_chains = load_curated_chains(args.curated_chains) if args.curated_chains.exists() else {}

    if data_kind == "analysis_csv":
        examples = load_examples_from_csv(
            data_path,
            curated_chains=curated_chains,
            max_rollouts=args.max_rollouts,
        )
    else:
        if not curated_chains:
            raise FileNotFoundError(
                f"Need --curated-chains to scan full run directory; default was not found: {args.curated_chains}"
            )
        examples = load_examples_from_run_dir(
            data_path,
            curated_chains=curated_chains,
            max_rollouts=args.max_rollouts,
        )

    examples = filter_scope(examples, args.scope)
    codex_sample_path = default_codex_sample_path(data_kind, data_path)
    if codex_sample_path != annotation_path:
        apply_judgements(examples, load_judgements(codex_sample_path))
    apply_judgements(examples, load_judgements(annotation_path))
    return examples, data_kind, data_path, annotation_path


def run_tui(
    examples: list[AnswerExample],
    annotation_path: Path,
    *,
    table_window_size: int,
    show_source: bool,
) -> None:
    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from textual import on
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, Input, Static
    except ModuleNotFoundError as exc:
        missing = exc.name or "textual"
        print(f"Missing dependency: {missing}", file=sys.stderr)
        print("Install the viewer dependencies, for example:", file=sys.stderr)
        print("  python -m pip install textual rich", file=sys.stderr)
        raise SystemExit(2) from exc

    class VerificationBrowser(App[None]):
        ENABLE_COMMAND_PALETTE = False
        CSS = """
        Screen {
            layout: vertical;
        }
        #body {
            height: 1fr;
        }
        #left {
            width: 64%;
            min-width: 70;
            border: solid $accent;
        }
        #right {
            width: 1fr;
            border: solid $accent;
        }
        #jump {
            display: none;
            height: 3;
        }
        #table {
            height: 1fr;
            overflow: hidden;
        }
        #detail {
            height: 1fr;
            overflow-y: auto;
        }
        """
        BINDINGS = [
            Binding("q", "exit_now", "Quit", priority=True),
            Binding("g", "focus_jump", "Go row", priority=True),
            Binding("escape", "cancel_jump", "Cancel", priority=True),
            Binding("home", "first_row", "First", priority=True),
            Binding("end", "last_row", "Last", priority=True),
            Binding("down", "next_row", "Down", priority=True),
            Binding("up", "previous_row", "Up", priority=True),
            ("j", "next_row", "Down"),
            ("k", "previous_row", "Up"),
            Binding("pagedown", "next_page", "Next page", priority=True),
            Binding("pageup", "previous_page", "Prev page", priority=True),
            Binding("e", "toggle_source", "Source", priority=True),
            ("1", "mark('correct')", "Correct"),
            ("2", "mark('partial')", "Partial"),
            ("3", "mark('incorrect')", "Incorrect"),
            ("0", "mark('')", "Clear"),
            ("s", "save", "Save"),
        ]

        def __init__(
            self,
            items: list[AnswerExample],
            judgement_path: Path,
            *,
            window_size: int,
            show_source_panel: bool,
        ) -> None:
            super().__init__()
            self.title = "Privacy Hop QA Verification Browser"
            self.items = items
            self.filtered = list(items)
            self.judgement_path = judgement_path
            self.current_id = items[0].example_id if items else ""
            self.current_index = 0
            self.window_start = 0
            self.window_size = window_size
            self.show_source_panel = show_source_panel
            self.item_by_id = {item.example_id: item for item in items}
            self.filtered_index_by_id = {item.example_id: idx for idx, item in enumerate(self.filtered)}
            self.trace_cache: dict[str, dict[str, Any] | None] = {}

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="body"):
                with Vertical(id="left"):
                    yield Static(id="table")
                    yield Input(placeholder="row number, then Enter", id="jump", disabled=True)
                with Vertical(id="right"):
                    yield Static(id="detail")
            yield Footer()

        def on_mount(self) -> None:
            self.set_focus(None)
            self.refresh_table()
            self.refresh_detail()

        def current_item(self) -> AnswerExample | None:
            item = self.item_by_id.get(self.current_id)
            if item is not None and self.current_id in self.filtered_index_by_id:
                return item
            if self.filtered:
                self.current_index = min(self.current_index, len(self.filtered) - 1)
                item = self.filtered[self.current_index]
                self.current_id = item.example_id
                return item
            return None

        def rebuild_filtered_index(self) -> None:
            self.filtered_index_by_id = {item.example_id: idx for idx, item in enumerate(self.filtered)}

        def ensure_window(self) -> None:
            if not self.filtered:
                self.current_index = 0
                self.window_start = 0
                return
            self.current_index = max(0, min(self.current_index, len(self.filtered) - 1))
            if len(self.filtered) <= self.window_size:
                self.window_start = 0
                return
            if self.current_index < self.window_start:
                self.window_start = max(0, self.current_index - self.window_size // 2)
            elif self.current_index >= self.window_start + self.window_size:
                self.window_start = max(0, self.current_index - self.window_size // 2)
            self.window_start = min(self.window_start, max(0, len(self.filtered) - self.window_size))

        @staticmethod
        def table_cell(value: Any, limit: int) -> str:
            return truncate_text(value, limit).replace("\n", " ")

        def refresh_table(self) -> None:
            table_widget = self.query_one("#table", Static)
            self.ensure_window()
            window_end = min(len(self.filtered), self.window_start + self.window_size)
            title = (
                f"Rows {self.window_start + 1:,}-{window_end:,} of {len(self.filtered):,} "
                "| g jump | e evidence | 1/2/3 judge | q quit"
            )
            table = Table(title=title, expand=True, show_lines=False)
            table.add_column("#", justify="right", width=6, no_wrap=True)
            table.add_column("kind", width=5, no_wrap=True)
            table.add_column("f1", justify="right", width=5, no_wrap=True)
            table.add_column("ok", width=3, no_wrap=True)
            table.add_column("question", ratio=3, overflow="ellipsis", no_wrap=True)
            table.add_column("predicted", ratio=2, overflow="ellipsis", no_wrap=True)
            table.add_column("true", ratio=2, overflow="ellipsis", no_wrap=True)
            table.add_column("judge", width=9, overflow="ellipsis", no_wrap=True)

            for absolute_index, item in enumerate(self.filtered[self.window_start:window_end], start=self.window_start):
                judge = item.human_judgement or item.codex_judgement
                table.add_row(
                    str(absolute_index + 1),
                    f"h{item.hop}" if item.answer_kind == "hop" else "final",
                    f"{item.raw_f1:.2f}",
                    "Y" if item.verifier_correct else "N",
                    self.table_cell(item.question, 90),
                    self.table_cell(item.predicted or "<empty>", 70),
                    self.table_cell(item.truth or "<empty>", 70),
                    judge,
                    style="reverse bold" if absolute_index == self.current_index else "",
                )
            table_widget.update(table)

        def refresh_detail(self) -> None:
            detail = self.query_one("#detail", Static)
            item = self.current_item()
            if item is None:
                detail.update(Panel("No examples loaded"))
                return

            metrics = Table.grid(expand=True)
            metrics.add_column(ratio=1)
            metrics.add_column(ratio=2)
            rows = [
                ("row", f"{self.current_index + 1:,} / {len(self.filtered):,}"),
                ("example", item.example_id),
                ("task / chain", f"{item.task_id} / {item.chain_id}"),
                ("kind", f"hop {item.hop}" if item.answer_kind == "hop" else "final"),
                ("raw_f1", f"{item.raw_f1:.3f}"),
                ("match_score", f"{item.match_score:.3f}"),
                ("verifier_correct", str(item.verifier_correct)),
                ("rollout_reward", f"{item.rollout_reward:.3f}"),
                ("rollout_binary_reward", str(item.rollout_binary_reward)),
                ("final_correct", str(item.final_correct)),
                ("human_judgement", item.human_judgement or ""),
                ("codex_judgement", item.codex_judgement or ""),
                ("notes", item.human_notes or item.codex_notes or ""),
                ("trace_path", item.trace_path),
                ("report_path", item.report_path),
            ]
            for key, value in rows:
                metrics.add_row(Text(key, style="bold"), str(value))

            body = Table.grid(expand=True)
            body.add_column(ratio=1)
            body.add_row(Panel(metrics, title="Scores", border_style="cyan"))
            if item.question:
                body.add_row(Panel(item.question, title="Question", border_style="blue"))
            body.add_row(Panel(item.predicted or "<empty>", title="Predicted Answer", border_style="yellow"))
            body.add_row(Panel(item.truth or "<empty>", title="True Answer", border_style="green"))
            if self.show_source_panel:
                body.add_row(Panel(self.source_context(item), title="Source Evidence", border_style="magenta"))
            else:
                body.add_row(Panel("Press e to show selected docs, reader outputs, snippets, and locators.", title="Source Evidence", border_style="magenta"))
            detail.update(body)

        def source_context(self, item: AnswerExample) -> str:
            if not item.trace_path:
                return "No trace_path recorded for this row."
            trace = self.trace_cache.get(item.trace_path)
            if item.trace_path not in self.trace_cache:
                try:
                    trace = json.loads(Path(item.trace_path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    trace = None
                self.trace_cache[item.trace_path] = trace
            return trace_source_context(item, trace)

        @on(Input.Submitted, "#jump")
        def jump_submitted(self, event: Input.Submitted) -> None:
            value = event.value.strip()
            jump = self.query_one("#jump", Input)
            jump.display = False
            jump.disabled = True
            self.set_focus(None)
            if not value:
                return
            try:
                row_number = int(value)
            except ValueError:
                self.notify(f"not a row number: {value}", severity="warning")
                return
            self.go_to_index(row_number - 1)

        def go_to_index(self, index: int) -> None:
            if not self.filtered:
                return
            self.current_index = max(0, min(index, len(self.filtered) - 1))
            self.current_id = self.filtered[self.current_index].example_id
            self.refresh_table()
            self.refresh_detail()

        def action_focus_jump(self) -> None:
            jump = self.query_one("#jump", Input)
            jump.disabled = False
            jump.display = True
            jump.value = ""
            jump.focus()

        def action_cancel_jump(self) -> None:
            jump = self.query_one("#jump", Input)
            jump.display = False
            jump.disabled = True
            self.set_focus(None)

        def action_exit_now(self) -> None:
            self.exit()

        def action_first_row(self) -> None:
            self.go_to_index(0)

        def action_last_row(self) -> None:
            self.go_to_index(len(self.filtered) - 1)

        def action_next_row(self) -> None:
            self.go_to_index(self.current_index + 1)

        def action_previous_row(self) -> None:
            self.go_to_index(self.current_index - 1)

        def action_next_page(self) -> None:
            self.go_to_index(self.current_index + self.window_size)

        def action_previous_page(self) -> None:
            self.go_to_index(self.current_index - self.window_size)

        def action_toggle_source(self) -> None:
            self.show_source_panel = not self.show_source_panel
            self.refresh_detail()

        def action_mark(self, label: str) -> None:
            item = self.item_by_id.get(self.current_id)
            if not item:
                return
            item.human_judgement = label
            if label:
                item.human_notes = "annotated in Textual browser"
            else:
                item.human_notes = ""
            self.refresh_table()
            self.refresh_detail()
            self.save_annotations()

        def action_save(self) -> None:
            self.save_annotations()

        def save_annotations(self) -> None:
            rows = []
            for item in self.items:
                if item.human_judgement or item.human_notes or item.codex_judgement or item.codex_notes:
                    rows.append(asdict(item))
            write_jsonl(self.judgement_path, rows)
            self.notify(f"saved {len(rows)} judgements to {self.judgement_path}")

    VerificationBrowser(
        examples,
        annotation_path,
        window_size=table_window_size,
        show_source_panel=show_source,
    ).run(mouse=False)


def main() -> None:
    args = parse_args()
    examples, data_kind, data_path, annotation_path = load_examples(args)

    if args.make_codex_sample is not None:
        output_path = default_codex_sample_path(data_kind, data_path) if args.make_codex_sample == Path("__DEFAULT__") else args.make_codex_sample
        make_codex_sample(examples, output_path.expanduser(), size=args.sample_size, seed=args.seed)
        print(f"wrote {min(len(examples), args.sample_size)} examples to {output_path}")
        return

    if args.print_summary:
        print(f"data: {data_path}")
        print(f"annotations: {annotation_path}")
        print_summary(examples)
        return

    run_tui(
        examples,
        annotation_path,
        table_window_size=max(20, int(args.table_window_size)),
        show_source=args.show_source,
    )


if __name__ == "__main__":
    main()
