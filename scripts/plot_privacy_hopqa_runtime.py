#!/usr/bin/env python3
"""Generate runtime analysis artifacts for a privacy_hopqa run directory."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ANSWER_RE = re.compile(r"^ANSWER_(\w+):\s*(.+)$")
JUSTIFICATION_RE = re.compile(r"^JUSTIFICATION_(\w+):\s*(.+)$")
ACTOR_TS_RE = re.compile(r"\[actor\]: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
ROLLOUT_FINISHED_RE = re.compile(r"rollouts finished so far: (\d+)")

STAGE_COLORS = {
    "start": "#6b7280",
    "hop_plan": "#2f6fed",
    "search": "#14866d",
    "doc_choose": "#9467bd",
    "doc_read": "#e17c05",
    "hop_resolve": "#d62728",
    "report": "#4c78a8",
    "complete": "#0f766e",
    "incomplete": "#991b1b",
}

ERROR_KIND_COLORS = {
    "parse_error": "#b91c1c",
    "context_overflow": "#ea580c",
    "other": "#6b7280",
}

FLOW_STAGE_ORDER = [
    "start",
    "hop_plan",
    "search",
    "doc_choose",
    "doc_read",
    "hop_resolve",
    "report",
    "complete",
    "incomplete",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to a privacy_hopqa output directory")
    parser.add_argument(
        "--output-dir",
        help="Directory to write plots and HTML into. Defaults to MosaicProject/runtime_analysis/<run_name>",
    )
    return parser.parse_args()


def parse_actor_timestamp(line: str) -> datetime | None:
    match = ACTOR_TS_RE.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def compute_completed_rollout_rate(actor_info_path: Path) -> dict[str, float | int | None]:
    if not actor_info_path.exists():
        return {
            "first_finished": None,
            "last_finished": None,
            "elapsed_s": None,
            "completed_rollouts_per_s": None,
            "seconds_per_completed_rollout": None,
        }
    samples: list[tuple[datetime, int]] = []
    with actor_info_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            timestamp = parse_actor_timestamp(raw_line)
            if not timestamp:
                continue
            match = ROLLOUT_FINISHED_RE.search(raw_line)
            if not match:
                continue
            finished = int(match.group(1))
            if not samples or samples[-1][1] != finished:
                samples.append((timestamp, finished))
    if len(samples) < 2:
        finished = samples[-1][1] if samples else None
        return {
            "first_finished": finished,
            "last_finished": finished,
            "elapsed_s": None,
            "completed_rollouts_per_s": None,
            "seconds_per_completed_rollout": None,
        }
    first_ts, first_finished = samples[0]
    last_ts, last_finished = samples[-1]
    elapsed_s = max(0.0, (last_ts - first_ts).total_seconds())
    finished_delta = max(0, last_finished - first_finished)
    completed_rollouts_per_s = (finished_delta / elapsed_s) if elapsed_s > 0 else None
    seconds_per_completed_rollout = (elapsed_s / finished_delta) if finished_delta > 0 else None
    return {
        "first_finished": first_finished,
        "last_finished": last_finished,
        "elapsed_s": elapsed_s,
        "completed_rollouts_per_s": completed_rollouts_per_s,
        "seconds_per_completed_rollout": seconds_per_completed_rollout,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = max(0.0, min(len(ordered) - 1, (len(ordered) - 1) * (p / 100.0)))
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_rows_for_export(
    rows: list[dict[str, Any]],
    limit: int,
    *,
    group_key: str | None = None,
    seed: int = 7,
) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)

    indexed_rows = list(enumerate(rows))
    rng = random.Random(seed)

    def sample_bucket(bucket: list[tuple[int, dict[str, Any]]], quota: int) -> list[tuple[int, dict[str, Any]]]:
        if quota <= 0:
            return []
        if len(bucket) <= quota:
            return list(bucket)
        if quota == 1:
            return [bucket[0]]
        if quota == 2:
            return [bucket[0], bucket[-1]]
        interior = bucket[1:-1]
        need = quota - 2
        chosen = [bucket[0], bucket[-1]]
        if interior:
            chosen.extend(interior if len(interior) <= need else rng.sample(interior, need))
        return chosen

    if not group_key:
        head = min(100, limit // 4)
        tail = min(100, max(0, limit - head))
        middle_quota = max(0, limit - head - tail)
        middle = indexed_rows[head : max(head, len(indexed_rows) - tail)]
        sampled = indexed_rows[:head] + (
            middle if len(middle) <= middle_quota else rng.sample(middle, middle_quota)
        ) + indexed_rows[len(indexed_rows) - tail :]
        sampled.sort(key=lambda item: item[0])
        return [row for _, row in sampled[:limit]]

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in indexed_rows:
        grouped[str(item[1].get(group_key) or "unknown")].append(item)

    total = len(indexed_rows)
    quotas: dict[str, int] = {}
    for group, bucket in grouped.items():
        quotas[group] = min(len(bucket), max(1, int(round(limit * len(bucket) / total))))

    quota_total = sum(quotas.values())
    groups_by_size = sorted(grouped.keys(), key=lambda key: (len(grouped[key]), key), reverse=True)
    while quota_total > limit:
        for group in groups_by_size:
            if quota_total <= limit:
                break
            if quotas[group] > 1:
                quotas[group] -= 1
                quota_total -= 1
    while quota_total < limit:
        grew = False
        for group in groups_by_size:
            if quota_total >= limit:
                break
            if quotas[group] < len(grouped[group]):
                quotas[group] += 1
                quota_total += 1
                grew = True
        if not grew:
            break

    sampled_items: list[tuple[int, dict[str, Any]]] = []
    for group in sorted(grouped):
        sampled_items.extend(sample_bucket(grouped[group], quotas[group]))
    sampled_items.sort(key=lambda item: item[0])
    return [row for _, row in sampled_items[:limit]]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_TIMELINE_RENDERER = None


def load_timeline_renderer():
    global _TIMELINE_RENDERER
    if _TIMELINE_RENDERER is not None:
        return _TIMELINE_RENDERER
    module_path = Path(__file__).resolve().parents[2] / "PipelineRL" / "pipelinerl" / "domains" / "privacy_hopqa" / "timeline.py"
    spec = importlib.util.spec_from_file_location("privacy_hopqa_timeline_renderer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load timeline renderer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TIMELINE_RENDERER = module.render_timeline_html
    return _TIMELINE_RENDERER


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
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = pred_counts & gold_counts
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


def score_chain_answers(problem: dict[str, Any], answers: dict[str, str], f1_threshold: float = 0.75) -> dict[str, Any]:
    hops = list(problem.get("hops") or [])
    per_hop: list[dict[str, Any]] = []
    correct_count = 0
    for hop in hops:
        hop_num = str(hop["hop_number"])
        predicted = answers.get(hop_num, "")
        truth = hop.get("answer", "")
        correct, match_score = answers_match(predicted, truth, f1_threshold=f1_threshold)
        correct_count += int(correct)
        per_hop.append(
            {
                "hop": int(hop_num),
                "agent_answer": predicted,
                "ground_truth": truth,
                "correct": correct,
                "match_score": match_score,
                "doc_id": hop.get("doc_id"),
            }
        )
    total_hops = max(len(hops), 1)
    final_key = "FINAL" if "FINAL" in answers else str(len(hops))
    final_correct, final_match_score = answers_match(
        answers.get(final_key, ""),
        problem.get("global_answer", ""),
        f1_threshold=f1_threshold,
    )
    return {
        "per_hop": per_hop,
        "correct_hops": correct_count,
        "total_hops": len(hops),
        "reward": correct_count / total_hops,
        "final_correct": final_correct,
        "final_match_score": final_match_score,
    }


def parse_dataset_names(run_dir: Path) -> list[str]:
    config_path = run_dir / "conf" / "exp_config.yaml"
    if not config_path.exists():
        return []
    dataset_names: list[str] = []
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() not in {"train_dataset_names", "test_dataset_names"}:
            continue
        raw_value = value.strip()
        if not raw_value:
            continue
        if raw_value.startswith("[") and raw_value.endswith("]"):
            try:
                parsed = json.loads(raw_value.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = []
            for item in parsed:
                if item:
                    dataset_names.append(str(item))
        else:
            dataset_names.append(raw_value)
    deduped: list[str] = []
    seen = set()
    for item in dataset_names:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def load_problem_lookup(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset_name in parse_dataset_names(run_dir):
        dataset_path = Path(dataset_name).expanduser()
        if not dataset_path.exists():
            continue
        for row in load_jsonl(dataset_path):
            task_id = str(row.get("task_id") or "")
            chain_id = str(row.get("chain_id") or "")
            if task_id and chain_id:
                lookup[(task_id, chain_id)] = row
    return lookup


def parse_report_answers(report_text: str) -> tuple[dict[str, str], dict[str, str]]:
    answers: dict[str, str] = {}
    justifications: dict[str, str] = {}
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        answer_match = ANSWER_RE.match(line)
        if answer_match:
            answers[answer_match.group(1)] = answer_match.group(2).strip()
            continue
        justification_match = JUSTIFICATION_RE.match(line)
        if justification_match:
            justifications[justification_match.group(1)] = justification_match.group(2).strip()
    return answers, justifications


def safe_uri(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        return Path(path_str).resolve().as_uri()
    except Exception:
        return ""


def compress_stage_sequence(events: list[dict[str, Any]], *, complete: bool) -> list[str]:
    ordered = sorted(events, key=lambda item: (float(item.get("start_s") or 0.0), str(item.get("lane") or "")))
    sequence = ["start"]
    for event in ordered:
        stage = str(event.get("stage") or "")
        if not stage or stage == "error":
            continue
        if sequence[-1] != stage:
            sequence.append(stage)
    sequence.append("complete" if complete else "incomplete")
    return sequence


def load_rollouts(run_dir: Path, problem_lookup: dict[tuple[str, str], dict[str, Any]]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[list[str]],
]:
    rollouts: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    sequences: list[list[str]] = []

    runs_root = run_dir / "privacy_hopqa_runs"
    for trace_path in sorted(runs_root.glob("*/trace.json")):
        trace = load_json(trace_path)
        rollout_dir = trace_path.parent
        rollout_id = rollout_dir.name
        summary = dict(trace.get("summary") or {})
        task_id = str(summary.get("task_id") or "")
        chain_id = str(summary.get("chain_id") or "")
        report_path = str(trace.get("report_path") or "")
        timeline_paths = dict(trace.get("timeline_paths") or {})
        timeline_html_path = str(timeline_paths.get("html_path") or (rollout_dir / "timeline.html"))
        events_path = Path(str(timeline_paths.get("events_path") or (rollout_dir / "timeline_events.json")))
        events = load_json(events_path) if events_path.exists() else []
        trace_errors = list(trace.get("error_records") or [])
        problem = problem_lookup.get((task_id, chain_id))
        report_text = ""
        answers: dict[str, str] = {}
        justifications: dict[str, str] = {}
        score: dict[str, Any] | None = None
        if report_path and Path(report_path).exists():
            report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
            answers, justifications = parse_report_answers(report_text)
            if problem is not None:
                score = score_chain_answers(problem, answers)

        pattern = ""
        hop_count = int(summary.get("total_hops") or 0)
        local_hop_count = 0
        web_hop_count = 0
        if problem is not None:
            pattern = str(problem.get("pattern") or "")
            if not pattern:
                pattern = "".join(str(hop.get("hop_type") or hop.get("t") or "?") for hop in (problem.get("hops") or []))
            if not hop_count:
                hop_count = int(problem.get("n_hops") or len(problem.get("hops") or []))
            for hop in problem.get("hops") or []:
                hop_type = str(hop.get("hop_type") or hop.get("t") or "").upper()
                if hop_type == "L":
                    local_hop_count += 1
                elif hop_type == "W":
                    web_hop_count += 1

        web_searches = 0
        local_searches = 0
        for action in trace.get("actions") or []:
            action_type = str(action.get("type") or "")
            if action_type == "web_search":
                web_searches += 1
            elif action_type == "local_document_search":
                local_searches += 1

        web_docs_read = 0
        local_docs_read = 0
        for hop_state in trace.get("hop_states") or []:
            for reader_result in hop_state.get("reader_results") or []:
                source = str(reader_result.get("source") or "").lower()
                if source == "web":
                    web_docs_read += 1
                elif source == "local":
                    local_docs_read += 1

        rollout_row = {
            "rollout_id": rollout_id,
            "task_id": task_id,
            "chain_id": chain_id,
            "hop_pattern": pattern,
            "hop_count": hop_count,
            "local_hop_count": local_hop_count,
            "web_hop_count": web_hop_count,
            "duration_s": float(summary.get("duration_s") or 0.0),
            "iterations_used": int(summary.get("iterations_used") or 0),
            "resolved_hops": int(summary.get("resolved_hops") or 0),
            "total_hops": int(summary.get("total_hops") or 0),
            "searches_total": int(summary.get("searches_total") or 0),
            "web_searches": web_searches,
            "local_searches": local_searches,
            "docs_read": int(summary.get("docs_read") or 0),
            "web_docs_read": web_docs_read,
            "local_docs_read": local_docs_read,
            "llm_calls_total": int(summary.get("llm_calls_total") or 0),
            "prompt_tokens": int(summary.get("prompt_tokens") or 0),
            "output_tokens": int(summary.get("output_tokens") or 0),
            "total_tokens": int(summary.get("total_tokens") or 0),
            "total_errors": int(summary.get("total_errors") or 0),
            "parse_errors_total": int(summary.get("parse_errors_total") or 0),
            "context_overflow_errors": int(summary.get("context_overflow_errors") or 0),
            "stalled": bool(summary.get("stalled")),
            "report_path": report_path,
            "report_uri": safe_uri(report_path),
            "timeline_html": timeline_html_path,
            "timeline_uri": safe_uri(timeline_html_path),
            "events_path": str(events_path),
            "events_uri": safe_uri(str(events_path)),
            "trace_path": str(trace_path),
            "trace_uri": safe_uri(str(trace_path)),
            "reward": score["reward"] if score is not None else "",
            "correct_hops": score["correct_hops"] if score is not None else "",
            "final_correct": bool(score["final_correct"]) if score is not None else "",
            "predicted_final_answer": answers.get("FINAL", ""),
            "ground_truth_final_answer": (problem or {}).get("global_answer", ""),
        }
        if score is not None:
            for hop_result in score.get("per_hop") or []:
                hop_num = int(hop_result.get("hop") or 0)
                rollout_row[f"hop_{hop_num}_correct"] = bool(hop_result.get("correct"))
                rollout_row[f"hop_{hop_num}_match_score"] = float(hop_result.get("match_score") or 0.0)
                rollout_row[f"hop_{hop_num}_answer"] = str(hop_result.get("agent_answer") or "")
                rollout_row[f"hop_{hop_num}_gold"] = str(hop_result.get("ground_truth") or "")
        rollouts.append(rollout_row)

        for error in trace_errors:
            error_rows.append(
                {
                    "rollout_id": rollout_id,
                    "task_id": task_id,
                    "chain_id": chain_id,
                    "stage": str(error.get("stage") or ""),
                    "kind": str(error.get("kind") or ""),
                    "hop_number": error.get("hop_number"),
                    "iteration": error.get("iteration"),
                    "message": str(error.get("message") or ""),
                    "elapsed_s": float(error.get("elapsed_s") or 0.0),
                }
            )

        for event in events:
            start_s = float(event.get("start_s") or 0.0)
            end_s = float(event.get("end_s") or start_s)
            event_rows.append(
                {
                    "rollout_id": rollout_id,
                    "task_id": task_id,
                    "chain_id": chain_id,
                    "lane": str(event.get("lane") or ""),
                    "stage": str(event.get("stage") or ""),
                    "label": str(event.get("label") or ""),
                    "start_s": start_s,
                    "duration_s": max(0.0, end_s - start_s),
                }
            )

        complete = bool(score["correct_hops"] == score["total_hops"]) if score is not None else (
            int(summary.get("resolved_hops") or 0) >= int(summary.get("total_hops") or 0) > 0
        )
        sequences.append(compress_stage_sequence(events, complete=complete))

    return rollouts, error_rows, event_rows, sequences


def load_llm_call_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actor_log = run_dir / "actor" / "info.log"
    marker = "privacy_hopqa llm call summary: "
    if not actor_log.exists():
        return rows
    with actor_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if marker not in raw_line:
                continue
            try:
                payload = json.loads(raw_line.split(marker, 1)[1].strip())
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "rollout_id": str(payload.get("rollout_id") or ""),
                    "task_id": str(payload.get("task_id") or ""),
                    "chain_id": str(payload.get("chain_id") or ""),
                    "log_name": str(payload.get("log_name") or ""),
                    "captured": bool(payload.get("captured")),
                    "prompt_tokens": int(payload.get("prompt_tokens") or 0),
                    "output_tokens": int(payload.get("output_tokens") or 0),
                    "total_tokens": int(payload.get("total_tokens") or 0),
                }
            )
    return rows


def plot_hist(values: list[float], path: Path, *, title: str, xlabel: str, bins: int = 30) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bar(counter: Counter[str], path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    if not counter:
        return
    labels_and_values = list(counter.items())

    def _numeric_key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    labels_and_values.sort(key=lambda item: _numeric_key(str(item[0])))
    labels, values = zip(*labels_and_values)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.5))
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_boxplot(grouped: dict[str, list[float]], path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    non_empty = [(label, values) for label, values in grouped.items() if values]
    if not non_empty:
        return

    def _sort_key(label: str) -> tuple[int, float | str]:
        if label in FLOW_STAGE_ORDER:
            return (0, FLOW_STAGE_ORDER.index(label))
        try:
            return (1, float(label))
        except ValueError:
            return (2, label)

    non_empty.sort(key=lambda item: _sort_key(item[0]))
    labels = [label for label, _ in non_empty]
    data = [values for _, values in non_empty]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.55)))
    ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_search_docs_scatter(rollouts: list[dict[str, Any]], path: Path) -> None:
    if not rollouts:
        return
    xs = [float(row["searches_total"]) for row in rollouts]
    ys = [float(row["docs_read"]) for row in rollouts]
    colors = [float(row["reward"]) if row.get("reward") not in ("", None) else 0.0 for row in rollouts]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.set_title("Searches vs Docs Read")
    ax.set_xlabel("Search actions")
    ax.set_ylabel("Selected docs read")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Reward")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_source_usage_scatter(rollouts: list[dict[str, Any]], path: Path) -> None:
    if not rollouts:
        return
    combos = [
        ("local_hop_count", "local_searches", "Local hops vs local searches"),
        ("local_hop_count", "web_searches", "Local hops vs web searches"),
        ("web_hop_count", "local_searches", "Web hops vs local searches"),
        ("web_hop_count", "web_searches", "Web hops vs web searches"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    scatter_obj = None
    for ax, (x_key, y_key, title) in zip(axes.flatten(), combos):
        cell_rewards: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rollouts:
            x = int(float(row.get(x_key) or 0.0))
            y = int(float(row.get(y_key) or 0.0))
            reward = float(row["reward"]) if row.get("reward") not in ("", None) else 0.0
            cell_rewards[(x, y)].append(reward)

        ordered_cells = sorted(cell_rewards.items(), key=lambda item: (item[0][0], item[0][1]))
        xs = [coords[0] for coords, _ in ordered_cells]
        ys = [coords[1] for coords, _ in ordered_cells]
        counts = [len(rewards) for _, rewards in ordered_cells]
        mean_rewards = [mean(rewards) for _, rewards in ordered_cells]
        max_count = max(counts) if counts else 1
        sizes = [55.0 + 845.0 * math.sqrt(count / max_count) for count in counts]

        scatter_obj = ax.scatter(
            xs,
            ys,
            s=sizes,
            c=mean_rewards,
            cmap="viridis",
            alpha=0.82,
            edgecolors="#1f2937",
            linewidths=0.55,
        )
        ax.set_title(title)
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel(y_key.replace("_", " "))
        ax.grid(alpha=0.25)
        x_max = max(xs + [0])
        y_max = max(ys + [0])
        ax.set_xticks(range(0, x_max + 1))
        ax.set_yticks(range(0, y_max + 1))
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_ylim(-0.5, y_max + 0.5)

        for (x, y), count in sorted(zip(zip(xs, ys), counts), key=lambda item: item[1], reverse=True)[:3]:
            ax.text(
                x,
                y,
                str(count),
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if count > (0.35 * max_count) else "#111827",
                fontweight="bold",
            )
    if scatter_obj is not None:
        cbar = fig.colorbar(scatter_obj, ax=axes.ravel().tolist(), shrink=0.92)
        cbar.set_label("Mean reward at this (hop, usage) cell")
    legend_sizes = [1, 10, 100]
    max_count_overall = max(
        (
            len_rewards
            for (x_key, y_key, _) in combos
            for len_rewards in (
                Counter(
                    (
                        int(float(row.get(x_key) or 0.0)),
                        int(float(row.get(y_key) or 0.0)),
                    )
                    for row in rollouts
                ).values()
            )
        ),
        default=1,
    )
    legend_handles = []
    for count in legend_sizes:
        scaled = min(count, max_count_overall)
        size = 55.0 + 845.0 * math.sqrt(scaled / max_count_overall)
        legend_handles.append(
            plt.scatter([], [], s=size, color="#94a3b8", edgecolors="#1f2937", linewidths=0.55, alpha=0.82)
        )
    fig.legend(
        legend_handles,
        [f"{min(count, max_count_overall)} rollouts" for count in legend_sizes],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Source Usage by Hop Type (bubble size = rollout count, color = mean reward)", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.11, wspace=0.24, hspace=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_source_usage_distributions(rollouts: list[dict[str, Any]], path: Path) -> None:
    if not rollouts:
        return
    web_searches = [float(row.get("web_searches") or 0.0) for row in rollouts]
    local_searches = [float(row.get("local_searches") or 0.0) for row in rollouts]
    web_reads = [float(row.get("web_docs_read") or 0.0) for row in rollouts]
    local_reads = [float(row.get("local_docs_read") or 0.0) for row in rollouts]

    max_search = max(web_searches + local_searches + [1.0])
    max_read = max(web_reads + local_reads + [1.0])
    search_bins = min(30, max(10, int(max_search) + 1))
    read_bins = min(30, max(10, int(max_read) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(web_searches, bins=search_bins, alpha=0.6, label="web_searches", color="#1f77b4")
    axes[0].hist(local_searches, bins=search_bins, alpha=0.6, label="local_searches", color="#ff7f0e")
    axes[0].set_title("Search Count Distribution by Source")
    axes[0].set_xlabel("Searches per rollout")
    axes[0].set_ylabel("Rollouts")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].hist(web_reads, bins=read_bins, alpha=0.6, label="web_docs_read", color="#2ca02c")
    axes[1].hist(local_reads, bins=read_bins, alpha=0.6, label="local_docs_read", color="#d62728")
    axes[1].set_title("Doc Read Distribution by Source")
    axes[1].set_xlabel("Docs read per rollout")
    axes[1].set_ylabel("Rollouts")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_bars(error_rows: list[dict[str, Any]], path: Path) -> None:
    if not error_rows:
        return
    stages = sorted({str(row["stage"]) for row in error_rows}, key=lambda stage: FLOW_STAGE_ORDER.index(stage) if stage in FLOW_STAGE_ORDER else 999)
    kinds = ["parse_error", "context_overflow", "other"]
    counts = {stage: Counter(str(row["kind"]) for row in error_rows if str(row["stage"]) == stage) for stage in stages}
    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.0), 4.8))
    bottoms = [0] * len(stages)
    for kind in kinds:
        values = [counts[stage].get(kind, 0) for stage in stages]
        ax.bar(stages, values, bottom=bottoms, color=ERROR_KIND_COLORS[kind], label=kind)
        bottoms = [bottoms[idx] + values[idx] for idx in range(len(values))]
    ax.set_title("Error Counts by Stage")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Error count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_named_values(
    values: dict[str, float],
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    percentage: bool = False,
) -> None:
    items = [(label, value) for label, value in values.items() if value is not None]
    if not items:
        return

    def _numeric_key(label: str) -> tuple[int, float | str]:
        try:
            return (0, float(label))
        except ValueError:
            return (1, label)

    items.sort(key=lambda item: _numeric_key(str(item[0])))
    labels = [str(label) for label, _ in items]
    vals = [float(value) for _, value in items]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    ax.bar(range(len(labels)), vals, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{100 * value:.0f}%"))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_by_group(
    rows: list[dict[str, Any]],
    value_key: str,
    group_key: str,
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n: int | None = None,
) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        group = str(row.get(group_key) or "")
        if not group or value in ("", None):
            continue
        grouped[group].append(float(value))
    if top_n is not None and len(grouped) > top_n:
        counts = Counter({key: len(values) for key, values in grouped.items()})
        keep = {key for key, _count in counts.most_common(top_n)}
        grouped = {key: value for key, value in grouped.items() if key in keep}
    plot_grouped_boxplot(grouped, path, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_rollout_roofline(events: list[dict[str, Any]], path: Path, *, title: str) -> None:
    if not events:
        return

    def base_lane(event: dict[str, Any]) -> str:
        stage = str(event.get("stage") or "")
        lane = str(event.get("lane") or "")
        if stage == "hop_plan":
            return "hop plan"
        if stage == "doc_choose":
            return "doc choose"
        if stage == "hop_resolve":
            return "hop resolve"
        if stage == "report":
            return "final report"
        if stage == "search":
            if "local" in lane:
                return "local retrieval"
            return "web retrieval"
        if stage == "doc_read":
            if lane.startswith("read:local"):
                return "local doc read"
            return "web doc read"
        return stage or "other"

    grouped_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in sorted(events, key=lambda row: (float(row.get("start_s") or 0.0), float(row.get("end_s") or row.get("start_s") or 0.0))):
        grouped_events[base_lane(event)].append(event)

    display_events: list[tuple[str, dict[str, Any]]] = []
    lane_order: list[str] = []
    preferred_order = [
        "hop plan",
        "web retrieval",
        "local retrieval",
        "doc choose",
        "web doc read",
        "local doc read",
        "hop resolve",
        "final report",
    ]
    for base in preferred_order + sorted(key for key in grouped_events if key not in preferred_order):
        if base not in grouped_events:
            continue
        lanes_end: list[float] = []
        for event in grouped_events[base]:
            start_s = float(event.get("start_s") or 0.0)
            end_s = float(event.get("end_s") or start_s)
            slot = 0
            while slot < len(lanes_end) and start_s < lanes_end[slot]:
                slot += 1
            if slot == len(lanes_end):
                lanes_end.append(end_s)
            else:
                lanes_end[slot] = end_s
            lane_name = base if slot == 0 else f"{base} #{slot + 1}"
            if lane_name not in lane_order:
                lane_order.append(lane_name)
            display_events.append((lane_name, event))

    lane_index = {lane: idx for idx, lane in enumerate(lane_order)}
    fig_height = max(3.6, len(lane_order) * 0.38 + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))

    for lane, event in sorted(display_events, key=lambda item: (float(item[1].get("start_s") or 0.0), lane_index.get(item[0], 0))):
        if lane not in lane_index:
            continue
        stage = str(event.get("stage") or "")
        start_s = float(event.get("start_s") or 0.0)
        end_s = float(event.get("end_s") or start_s)
        duration = max(0.0, end_s - start_s)
        y = lane_index[lane]
        ax.barh(
            y,
            duration,
            left=start_s,
            height=0.72,
            color=STAGE_COLORS.get(stage, "#94a3b8"),
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_title(title)
    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_ylabel("Lane")
    ax.set_yticks(range(len(lane_order)))
    ax.set_yticklabels([lane[:44] + ("..." if len(lane) > 44 else "") for lane in lane_order], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_representative_rollouts(rollouts: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    rng = random.Random(7)
    eligible = [row for row in rollouts if row.get("reward") not in ("", None)]
    if len(eligible) <= limit:
        return sorted(eligible, key=lambda row: str(row.get("rollout_id") or ""))

    def reward_bucket(value: float) -> str:
        if value >= 0.999:
            return "solved"
        if value > 0.0:
            return "partial"
        return "zero"

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        bucket = reward_bucket(float(row.get("reward") or 0.0))
        hop_count = int(row.get("hop_count") or row.get("total_hops") or 0)
        grouped[(bucket, hop_count)].append(row)

    sampled: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        candidates = grouped[key]
        sampled.append(rng.choice(candidates))
        if len(sampled) >= limit:
            break

    if len(sampled) < limit:
        sampled_ids = {str(row.get("rollout_id") or "") for row in sampled}
        remaining = [row for row in eligible if str(row.get("rollout_id") or "") not in sampled_ids]
        rng.shuffle(remaining)
        sampled.extend(remaining[: max(0, limit - len(sampled))])

    sampled = sampled[:limit]
    sampled.sort(key=lambda row: (float(row.get("reward") or 0.0), int(row.get("hop_count") or 0), str(row.get("rollout_id") or "")))
    return sampled


def copy_sample_artifacts(output_dir: Path, sampled_rows: list[dict[str, Any]]) -> None:
    sample_root = output_dir / "sample_artifacts"
    if sample_root.exists():
        shutil.rmtree(sample_root)
    sample_root.mkdir(parents=True, exist_ok=True)
    render_timeline_html = load_timeline_renderer()
    for row in sampled_rows:
        rollout_id = str(row.get("rollout_id") or "")
        if not rollout_id:
            continue
        target_dir = sample_root / rollout_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in [
            ("report_path", "report.md"),
            ("trace_path", "trace.json"),
            ("events_path", "timeline_events.json"),
        ]:
            source = str(row.get(key) or "")
            if not source:
                continue
            source_path = Path(source)
            if source_path.exists():
                shutil.copy2(source_path, target_dir / filename)
                row[f"{key}_sample_rel"] = f"sample_artifacts/{rollout_id}/{filename}"
        rollout_dir = Path(str(row.get("trace_path") or "")).parent
        prompt_dir = rollout_dir / "prompts"
        if prompt_dir.exists():
            prompt_files = sorted(prompt_dir.glob("*.txt"))
            if prompt_files:
                target_prompt_dir = target_dir / "prompts"
                target_prompt_dir.mkdir(parents=True, exist_ok=True)
                for prompt_file in prompt_files:
                    shutil.copy2(prompt_file, target_prompt_dir / prompt_file.name)
                row["prompt_dir_sample_rel"] = f"sample_artifacts/{rollout_id}/prompts"
        summary_path = rollout_dir / "timeline_summary.json"
        events_path = rollout_dir / "timeline_events.json"
        trace_path = rollout_dir / "trace.json"
        if summary_path.exists() and events_path.exists() and trace_path.exists():
            summary = load_json(summary_path)
            events = load_json(events_path)
            trace = load_json(trace_path)
            report_text = ""
            report_path = str(trace.get("report_path") or "")
            if report_path and Path(report_path).exists():
                report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
            rendered_timeline = render_timeline_html(
                summary,
                events,
                trace=trace,
                report_text=report_text,
                prompt_dir=prompt_dir if prompt_dir.exists() else None,
            )
            (target_dir / "timeline.html").write_text(rendered_timeline, encoding="utf-8")
            row["timeline_html_sample_rel"] = f"sample_artifacts/{rollout_id}/timeline.html"
        roofline_path = row.get("roofline_path")
        if roofline_path:
            row["roofline_rel"] = str(Path(roofline_path).relative_to(output_dir))


def render_stage_flow_svg(sequences: list[list[str]]) -> str:
    if not sequences:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='120'></svg>"

    node_counts: Counter[tuple[int, str]] = Counter()
    edge_counts: Counter[tuple[tuple[int, str], tuple[int, str]]] = Counter()
    max_step = 0
    for sequence in sequences:
        max_step = max(max_step, len(sequence) - 1)
        for idx, stage in enumerate(sequence):
            node_counts[(idx, stage)] += 1
            if idx + 1 < len(sequence):
                edge_counts[((idx, stage), (idx + 1, sequence[idx + 1]))] += 1

    node_width = 120
    node_height = 28
    col_gap = 140
    row_gap = 56
    margin_x = 50
    margin_y = 40
    width = margin_x * 2 + max_step * col_gap + node_width
    height = margin_y * 2 + (len(FLOW_STAGE_ORDER) - 1) * row_gap + node_height
    stage_y = {stage: margin_y + idx * row_gap for idx, stage in enumerate(FLOW_STAGE_ORDER)}

    def node_pos(step: int, stage: str) -> tuple[float, float]:
        return margin_x + step * col_gap, stage_y.get(stage, margin_y)

    max_edge_count = max(edge_counts.values()) if edge_counts else 1
    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}</style>",
    ]

    for (src, dst), count in sorted(edge_counts.items(), key=lambda item: item[1]):
        src_x, src_y = node_pos(*src)
        dst_x, dst_y = node_pos(*dst)
        x1 = src_x + node_width
        y1 = src_y + node_height / 2
        x2 = dst_x
        y2 = dst_y + node_height / 2
        mid_x = (x1 + x2) / 2
        stroke = 2.0 + 10.0 * (count / max_edge_count)
        color = STAGE_COLORS.get(src[1], "#94a3b8")
        svg_parts.append(
            f"<path d='M{x1},{y1} C{mid_x},{y1} {mid_x},{y2} {x2},{y2}' "
            f"stroke='{color}' stroke-width='{stroke:.2f}' fill='none' stroke-opacity='0.45' />"
        )

    for (step, stage), count in sorted(node_counts.items()):
        x, y = node_pos(step, stage)
        fill = STAGE_COLORS.get(stage, "#94a3b8")
        label = html.escape(stage)
        svg_parts.append(
            f"<rect x='{x}' y='{y}' width='{node_width}' height='{node_height}' rx='8' ry='8' "
            f"fill='{fill}' fill-opacity='0.92' stroke='white' stroke-width='1' />"
        )
        svg_parts.append(f"<text x='{x + 8}' y='{y + 12}' font-size='11' fill='white'>{label}</text>")
        svg_parts.append(f"<text x='{x + 8}' y='{y + 23}' font-size='11' fill='white'>n={count}</text>")

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def html_table(rows: list[list[str]], headers: list[str]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def display_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _truncate(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_decimal(value: Any, decimals: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def preview_table(rows: list[dict[str, Any]], limit: int = 12) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    keys = sorted({key for row in rows[:limit] for key in row.keys()})
    preview_rows: list[list[str]] = []
    for row in rows[:limit]:
        cells = [html.escape(_truncate(display_value(row.get(key, "")), 140)) for key in keys]
        preview_rows.append(cells)
    return html_table(preview_rows, headers=keys)


def mapping_table(mapping: dict[str, Any], *, key_name: str, value_name: str) -> str:
    rows = [[html.escape(str(key)), html.escape(display_value(value))] for key, value in mapping.items()]
    return html_table(rows, headers=[key_name, value_name])


def plot_card(
    title: str,
    filename: str,
    *,
    blurb: str,
    available: bool,
    missing_note: str | None = None,
) -> str:
    body = (
        f"<a href='{html.escape(filename)}'><img src='{html.escape(filename)}' alt='{html.escape(title)}' /></a>"
        if available
        else f"<div class='missing-note'>{html.escape(missing_note or 'Plot unavailable for this run.')}</div>"
    )
    return (
        "<div class='plot-card'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<p class='blurb'>{html.escape(blurb)}</p>"
        f"{body}"
        "</div>"
    )


def render_index_html(
    output_dir: Path,
    run_name: str,
    overview: dict[str, Any],
    plots: list[str],
    sampled_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    llm_call_rows: list[dict[str, Any]],
    hop_accuracy_rows: list[list[str]],
    parse_rate_rows: list[list[str]],
    llm_token_rows: list[list[str]],
) -> str:
    available = set(plots)

    def has_plot(name: str) -> bool:
        return name in available and (output_dir / name).exists()

    cards = [
        ("Rollouts", str(overview.get("rollout_count", 0))),
        ("Mean Reward", f"{float(overview.get('mean_reward') or 0.0):.3f}"),
        ("Solved Rate", f"{100.0 * float(overview.get('solved_fraction') or 0.0):.1f}%"),
        ("Mean Rollout Duration", f"{float(overview.get('mean_duration_s') or 0.0):.1f}s"),
        ("Median Rollout Duration", f"{float(overview.get('median_duration_s') or 0.0):.1f}s"),
        (
            "Completed Rollouts / s",
            format_decimal(overview.get("completed_rollouts_per_s"), decimals=3)
            if overview.get("completed_rollouts_per_s") is not None
            else "n/a",
        ),
        ("Mean LLM Calls", f"{float(overview.get('mean_llm_calls') or 0.0):.1f}"),
        ("Median Iterations", f"{float(overview.get('median_iterations') or 0.0):.1f}"),
        ("Median LLM Calls", f"{float(overview.get('median_llm_calls') or 0.0):.1f}"),
        ("Mean Prompt Tokens", f"{float(overview.get('mean_prompt_tokens') or 0.0):.1f}"),
        ("Median Prompt Tokens", f"{float(overview.get('median_prompt_tokens') or 0.0):.1f}"),
        ("Mean Output Tokens", f"{float(overview.get('mean_output_tokens') or 0.0):.1f}"),
        ("Median Output Tokens", f"{float(overview.get('median_output_tokens') or 0.0):.1f}"),
        ("Parse Errors", str(int(overview.get("parse_errors_total") or 0))),
        ("Parse Error Rate", f"{100.0 * float(overview.get('parse_error_rate') or 0.0):.2f}%"),
        ("Context Overflows", str(int(overview.get("context_overflow_errors") or 0))),
    ]
    card_html = "".join(
        "<div class='stat-card'>"
        f"<div class='stat-label'>{html.escape(label)}</div>"
        f"<div class='stat-value'>{html.escape(value)}</div>"
        "</div>"
        for label, value in cards
    )

    overview_plots = "\n".join(
        [
            plot_card(
                "Rollout Duration Distribution",
                "rollout_duration_distribution.png",
                blurb="Completed rollout time across the sampled run. This is the fastest overall throughput sanity check.",
                available=has_plot("rollout_duration_distribution.png"),
            ),
            plot_card(
                "Iteration Distribution",
                "iteration_distribution.png",
                blurb="How many search-resolve loops the agent needed before terminating each rollout.",
                available=has_plot("iteration_distribution.png"),
            ),
            plot_card(
                "LLM Calls per Rollout",
                "llm_calls_distribution.png",
                blurb="Total number of `hop_plan`, `doc_choose`, `doc_read`, and `hop_resolve` calls per rollout.",
                available=has_plot("llm_calls_distribution.png"),
            ),
            plot_card(
                "Searches vs Docs Read",
                "searches_vs_docs_read.png",
                blurb="Search fanout versus how many candidate docs were actually selected for close reading.",
                available=has_plot("searches_vs_docs_read.png"),
            ),
            plot_card(
                "Source Usage Scatter",
                "source_usage_scatter.png",
                blurb="Four pairings of local/web hop counts against local/web search counts. This makes it obvious when the agent is using the wrong source for a chain.",
                available=has_plot("source_usage_scatter.png"),
            ),
            plot_card(
                "Source Usage Distributions",
                "source_usage_distributions.png",
                blurb="Overlaid per-rollout distributions for web vs local searches and web vs local document reads.",
                available=has_plot("source_usage_distributions.png"),
            ),
        ]
    )

    context_plots = "\n".join(
        [
            plot_card(
                "Rollout Duration by Final Iteration",
                "rollout_duration_by_final_iteration.png",
                blurb="Completed rollout time split by how many iterations the hop-QA loop ended up using.",
                available=has_plot("rollout_duration_by_final_iteration.png"),
            ),
            plot_card(
                "Stage Duration Distribution",
                "stage_duration_boxplot.png",
                blurb="Per-stage wall time distribution. This is the fastest way to see whether planning, choosing, reading, or resolving is dominating.",
                available=has_plot("stage_duration_boxplot.png"),
            ),
            plot_card(
                "Total Tokens by Final Iteration",
                "total_tokens_by_final_iteration.png",
                blurb="Aggregate prompt plus output token volume split by the final number of iterations.",
                available=has_plot("total_tokens_by_final_iteration.png"),
            ),
        ]
    )

    llm_token_plots = "\n".join(
        [
            plot_card(
                "Prompt Tokens by Call Type",
                "prompt_tokens_by_call_type.png",
                blurb="Prompt-side token distribution by `hop_plan`, `doc_choose`, `doc_read`, and `hop_resolve`.",
                available=has_plot("prompt_tokens_by_call_type.png"),
            ),
            plot_card(
                "Output Tokens by Call Type",
                "output_tokens_by_call_type.png",
                blurb="Response-token distribution by LLM call type. This is the quickest read on generation-heavy stages.",
                available=has_plot("output_tokens_by_call_type.png"),
            ),
            plot_card(
                "Total Tokens by Call Type",
                "total_tokens_by_call_type.png",
                blurb="Combined prompt plus response token distribution by call type.",
                available=has_plot("total_tokens_by_call_type.png"),
            ),
            plot_card(
                "LLM Total Tokens per Rollout",
                "llm_total_tokens_per_rollout.png",
                blurb="Total LLM token volume per rollout across the full hop chain.",
                available=has_plot("llm_total_tokens_per_rollout.png"),
            ),
        ]
    )

    reward_plots = "\n".join(
        [
            plot_card(
                "Reward Distribution",
                "reward_distribution.png",
                blurb="Reward counts recomputed from saved reports and dataset ground truth.",
                available=has_plot("reward_distribution.png"),
            ),
            plot_card(
                "Reward by Hop Length",
                "reward_by_hop_length.png",
                blurb="How reward changes as the chain gets longer.",
                available=has_plot("reward_by_hop_length.png"),
            ),
            plot_card(
                "Reward by Hop Pattern",
                "reward_by_hop_pattern.png",
                blurb="Reward split by hop-type pattern like `LLL` or mixed local/web chains.",
                available=has_plot("reward_by_hop_pattern.png"),
            ),
            plot_card(
                "Per-Hop Position Accuracy",
                "hop_position_accuracy.png",
                blurb="Percent of first hops, second hops, and later hops solved correctly.",
                available=has_plot("hop_position_accuracy.png"),
            ),
        ]
    )

    error_plots = "\n".join(
        [
            plot_card(
                "Parse Error Rate by Stage",
                "parse_error_rate_by_stage.png",
                blurb="Parse failures normalized by the number of times that stage could have failed.",
                available=has_plot("parse_error_rate_by_stage.png"),
            ),
            plot_card(
                "Error Counts by Stage",
                "errors_by_stage.png",
                blurb="Absolute error volume by stage, split into parse errors, context overflows, and other failures.",
                available=has_plot("errors_by_stage.png"),
            ),
            plot_card(
                "Error Kinds",
                "errors_by_kind.png",
                blurb="Overall error-kind distribution for the run snapshot.",
                available=has_plot("errors_by_kind.png"),
            ),
        ]
    )

    sampled_table_rows: list[list[str]] = []
    roofline_cards: list[str] = []
    for row in sampled_rows:
        links = []
        for key, label in [
            ("report_path_sample_rel", "report"),
            ("timeline_html_sample_rel", "timeline"),
            ("trace_path_sample_rel", "trace"),
            ("events_path_sample_rel", "events"),
            ("prompt_dir_sample_rel", "prompts"),
        ]:
            if row.get(key):
                links.append(f"<a href='{html.escape(str(row[key]))}'>{label}</a>")
        roofline_img = (
            f"<a href='{html.escape(str(row['roofline_rel']))}'><img src='{html.escape(str(row['roofline_rel']))}' alt='{html.escape(str(row.get('rollout_id') or ''))}' /></a>"
            if row.get("roofline_rel")
            else "<div class='missing-note'>Roofline unavailable.</div>"
        )
        roofline_cards.append(
            "<div class='sample-card'>"
            f"<div class='sample-title'>{html.escape(str(row.get('rollout_id') or ''))}</div>"
            f"<div class='sample-meta'>reward={html.escape(format_decimal(row.get('reward')))}, hops={html.escape(display_value(row.get('hop_count') if row.get('hop_count') not in (None, '') else row.get('total_hops')))}, pattern={html.escape(display_value(row.get('hop_pattern') or ''))}, parse={html.escape(display_value(row.get('parse_errors_total')))}</div>"
            f"{roofline_img}"
            f"<div class='sample-links'>{' | '.join(links)}</div>"
            "</div>"
        )
        sampled_table_rows.append(
            [
                html.escape(str(row.get("rollout_id") or "")),
                html.escape(str(row.get("task_id") or "")),
                html.escape(str(row.get("chain_id") or "")),
                html.escape(str(row.get("hop_pattern") or "")),
                html.escape(display_value(row.get("hop_count") if row.get("hop_count") not in (None, "") else row.get("total_hops"))),
                html.escape(format_decimal(row.get("reward"))),
                html.escape(str(row.get("predicted_final_answer") or "")),
                html.escape(str(row.get("ground_truth_final_answer") or "")),
                "<br/>".join(links),
            ]
        )
    sampled_table = html_table(
        sampled_table_rows,
        ["Rollout", "Task", "Chain", "Pattern", "Hops", "Reward", "Predicted Final", "Gold Final", "Artifacts"],
    )
    sampled_by_rollout_id = {str(row.get("rollout_id") or ""): row for row in sampled_rows}

    rollout_rows_sorted = sorted(
        rollout_rows,
        key=lambda row: (float(row.get("reward") or 0.0), -float(row.get("parse_errors_total") or 0.0), float(row.get("duration_s") or 0.0)),
        reverse=True,
    )
    rollout_table_rows: list[list[str]] = []
    for row in rollout_rows_sorted[:120]:
        sample_row = sampled_by_rollout_id.get(str(row.get("rollout_id") or ""))
        links = []
        if sample_row and sample_row.get("report_path_sample_rel"):
            links.append(f"<a href='{html.escape(str(sample_row['report_path_sample_rel']))}'>report</a>")
        elif row.get("report_uri"):
            links.append(f"<a href='{html.escape(str(row['report_uri']))}'>report</a>")
        if sample_row and sample_row.get("timeline_html_sample_rel"):
            links.append(f"<a href='{html.escape(str(sample_row['timeline_html_sample_rel']))}'>timeline</a>")
        elif row.get("timeline_uri"):
            links.append(f"<a href='{html.escape(str(row['timeline_uri']))}'>timeline</a>")
        if sample_row and sample_row.get("trace_path_sample_rel"):
            links.append(f"<a href='{html.escape(str(sample_row['trace_path_sample_rel']))}'>trace</a>")
        elif row.get("trace_uri"):
            links.append(f"<a href='{html.escape(str(row['trace_uri']))}'>trace</a>")
        rollout_table_rows.append(
            [
                html.escape(str(row.get("rollout_id") or "")),
                html.escape(str(row.get("hop_pattern") or "")),
                html.escape(format_decimal(row.get("reward"))),
                html.escape(display_value(row.get("iterations_used"))),
                html.escape(display_value(row.get("searches_total"))),
                html.escape(display_value(row.get("docs_read"))),
                html.escape(display_value(row.get("parse_errors_total"))),
                html.escape(display_value(row.get("context_overflow_errors"))),
                html.escape(format_decimal(row.get("duration_s"), decimals=3)),
                "<br/>".join(links),
            ]
        )
    rollout_table = html_table(
        rollout_table_rows,
        ["Rollout", "Pattern", "Reward", "Iters", "Searches", "Docs", "Parse Err", "Ctx Oflow", "Duration (s)", "Artifacts"],
    )

    counts = {
        "Rollouts": overview.get("rollout_count", 0),
        "Mean reward": format_decimal(overview.get("mean_reward")),
        "Solved fraction": f"{100.0 * float(overview.get('solved_fraction') or 0.0):.1f}%",
        "Completed rollouts / s": (
            format_decimal(overview.get("completed_rollouts_per_s"))
            if overview.get("completed_rollouts_per_s") is not None
            else "n/a"
        ),
        "Parse errors": int(overview.get("parse_errors_total") or 0),
        "Parse error rate": f"{100.0 * float(overview.get('parse_error_rate') or 0.0):.2f}%",
        "Context overflows": int(overview.get("context_overflow_errors") or 0),
    }
    source_usage_checks = {
        "No-web-hop rollouts with web searches": f"{100.0 * float(overview.get('no_web_hops_with_web_search_fraction') or 0.0):.1f}%",
        "No-local-hop rollouts with local searches": f"{100.0 * float(overview.get('no_local_hops_with_local_search_fraction') or 0.0):.1f}%",
        "Mean web searches / rollout": format_decimal(overview.get("mean_web_searches")),
        "Mean local searches / rollout": format_decimal(overview.get("mean_local_searches")),
        "Mean web docs read / rollout": format_decimal(overview.get("mean_web_docs_read")),
        "Mean local docs read / rollout": format_decimal(overview.get("mean_local_docs_read")),
    }
    duration_stats = {key: format_decimal(value) for key, value in (overview.get("duration_stats") or {}).items()}
    reward_stats = {key: format_decimal(value) for key, value in (overview.get("reward_stats") or {}).items()}
    llm_stats = {key: format_decimal(value) for key, value in (overview.get("llm_call_stats") or {}).items()}

    llm_token_table = html_table(
        llm_token_rows,
        ["Call Type", "Calls", "Median Prompt", "Median Output", "Median Total", "Mean Total"],
    )
    hop_accuracy_table = html_table(hop_accuracy_rows, ["Hop", "Accuracy", "Solved / Total"])
    parse_rate_table = html_table(parse_rate_rows, ["Stage", "Parse Errors", "Attempts", "Rate"])

    reward_plumbing_html = """
      <div class='mapping-list'>
        <div><strong>Rollout reward:</strong> hop accuracy, computed as <code>correct_hops / total_hops</code>.</div>
        <div><strong>Capture mode:</strong> training currently uses planning traces only, so <code>hop_plan</code>, <code>doc_choose</code>, and <code>hop_resolve</code> are trained; <code>doc_read</code> is not.</div>
        <div><strong>Group size:</strong> the actor runs 8 attempts for the same problem under one RL comparison group.</div>
        <div><strong>Advantage shape:</strong> every captured trace from one rollout gets the same scalar rollout reward, and therefore the same leave-one-out advantage relative to the other attempts in its group.</div>
        <div><strong>Current limitation:</strong> this run's persisted trace bundle does not include <code>group_id</code> and <code>rollout_index</code>, so the viewer cannot reconstruct an exact per-group leave-one-out table after the fact.</div>
      </div>
    """

    raw_links = [
        "summary.json",
        "rollouts_sample.csv",
        "errors.csv",
        "events_sample.csv",
        "llm_calls_sample.csv",
        "hop_position_accuracy.csv",
        "parse_error_rates.csv",
        "llm_token_summary.csv",
    ]
    raw_links_html = "\n".join(
        f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>"
        for name in raw_links
        if (output_dir / name).exists()
    )

    caveats = [
        "Representative traces are a 12-rollout sample chosen for qualitative debugging across different outcomes and hop lengths; they are not aggregate statistics for the whole run.",
        "Current runs only show raw prompt text in the timeline when prompt logging is enabled. Otherwise the timeline falls back to structured trace state and tool outputs.",
        "The raw CSV links in this bundle are capped samples rather than full dumps, to keep the exported viewer lightweight and git-friendly.",
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Runtime Analysis - {html.escape(run_name)}</title>
  <style>
    :root {{
      --bg: #f5f4ef;
      --panel: #ffffff;
      --ink: #1e2430;
      --muted: #5f6b7a;
      --line: #d7dbe2;
      --accent: #1f5da8;
      --accent-soft: #e7f0fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f8f7f2 0%, #eef3f8 100%);
      color: var(--ink);
    }}
    a {{ color: var(--accent); }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: radial-gradient(circle at top left, #ffffff 0%, #eef4fb 45%, #f8f7f2 100%);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px 30px;
      margin-bottom: 22px;
      box-shadow: 0 18px 50px rgba(30, 36, 48, 0.06);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 6px 0; color: var(--muted); max-width: 1000px; }}
    .topnav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .topnav a {{
      text-decoration: none;
      background: var(--accent-soft);
      color: var(--accent);
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #cfe0f6;
      font-size: 14px;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}
    .stat-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: 0 8px 24px rgba(30, 36, 48, 0.04);
    }}
    .stat-label {{
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .stat-value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .section {{
      margin: 28px 0;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .section p.section-note {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .plot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 12px 32px rgba(30, 36, 48, 0.05);
    }}
    .plot-card h3 {{
      margin: 4px 0 8px;
      font-size: 18px;
    }}
    .plot-card .blurb {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
      min-height: 40px;
    }}
    .plot-card img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .missing-note {{
      min-height: 260px;
      display: grid;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 12px;
      color: var(--muted);
      background: #f8fafc;
      text-align: center;
      padding: 16px;
    }}
    .table-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .table-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 28px rgba(30, 36, 48, 0.05);
      overflow: auto;
    }}
    .table-card.scrollable {{
      max-height: 640px;
      display: flex;
      flex-direction: column;
    }}
    .table-scroll {{
      overflow: auto;
      min-height: 0;
      flex: 1 1 auto;
    }}
    .table-card h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .mapping-list {{
      display: grid;
      gap: 10px;
      color: var(--ink);
      font-size: 14px;
      line-height: 1.5;
    }}
    .mapping-list code {{
      background: #f2f5fa;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 1px 5px;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f7f9fc;
      z-index: 1;
    }}
    details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      margin-top: 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    ul.raw-links {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    .limitations li {{ margin-bottom: 8px; }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    .sample-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 12px 32px rgba(30, 36, 48, 0.05);
    }}
    .sample-card img {{
      width: 100%;
      display: block;
      margin: 10px 0;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .sample-title {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .sample-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .sample-links {{
      font-size: 13px;
      color: var(--muted);
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f7f9fc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      font-size: 12px;
      max-height: 420px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Runtime Analysis Viewer</h1>
      <p><strong>Run:</strong> {html.escape(run_name)}</p>
      <p>This viewer summarizes `privacy_hopqa` runtime behavior from saved rollout traces, token logs, and sampled artifacts. It mirrors the older privacy-agent viewer more closely, but keeps the hop-specific diagnostics and representative trace viewers at the bottom.</p>
      <p><strong>Loop:</strong> `hop_plan` picks up to 4 searches, retrieval runs, `doc_choose` selects up to 3 docs, `doc_read` inspects them, `hop_resolve` decides whether the current hop is answered, and the loop repeats until the chain is resolved or the budget is exhausted.</p>
      <div class="topnav">
        <a href="#overview">Overview</a>
        <a href="#context">Context + Iterations</a>
        <a href="#llm-tokens">LLM Tokens</a>
        <a href="#rewards">Rewards</a>
        <a href="#errors">Errors</a>
        <a href="#tables">Tables</a>
        <a href="#raw">Raw Files</a>
        <a href="#traces">Representative Traces</a>
      </div>
      <div class="stat-grid">
        {card_html}
      </div>
    </section>

    <section class="section" id="overview">
      <h2>Overview Plots</h2>
      <p class="section-note">These are the quickest plots to check for throughput shape, rollout duration, and how much search fanout is happening before document reading.</p>
      <div class="plot-grid">
        {overview_plots}
      </div>
    </section>

    <section class="section" id="context">
      <h2>Context + Iteration Behavior</h2>
      <p class="section-note">This section is usually the most useful for explaining why the run feels slow. It focuses on stage latency and how extra iterations add wall time and token usage.</p>
      <div class="plot-grid">
        {context_plots}
      </div>
    </section>

    <section class="section" id="llm-tokens">
      <h2>LLM Token Usage</h2>
      <p class="section-note">These views use the explicit `privacy_hopqa llm call summary` lines emitted by the async rollout adapter. They help separate prompt-heavy from generation-heavy stages.</p>
      <div class="plot-grid">
        {llm_token_plots}
      </div>
    </section>

    <section class="section" id="rewards">
      <h2>Reward Spot Checks</h2>
      <p class="section-note">These views recompute reward from the saved report answers against the dataset ground truth, then show representative sampled reports and artifacts for manual inspection.</p>
      <div class="plot-grid">
        {reward_plots}
      </div>
      <div class="table-grid">
        <section class="table-card">
          <h3>Per-Hop Accuracy</h3>
          {hop_accuracy_table}
        </section>
        <section class="table-card">
          <h3>Reward Plumbing</h3>
          {reward_plumbing_html}
        </section>
        <section class="table-card">
          <h3>Sampled Report Checks</h3>
          {sampled_table}
        </section>
      </div>
    </section>

    <section class="section" id="errors">
      <h2>Error Behavior</h2>
      <p class="section-note">Parse-error rate is normalized by the number of LLM calls at that stage, so it reflects how often the stage fails when it actually has a chance to fail.</p>
      <div class="plot-grid">
        {error_plots}
      </div>
      <div class="table-grid">
        <section class="table-card">
          <h3>Parse Error Rates</h3>
          {parse_rate_table}
        </section>
        <section class="table-card">
          <h3>LLM Token Usage by Call Type</h3>
          {llm_token_table}
        </section>
      </div>
    </section>

    <section class="section" id="tables">
      <h2>Summary Tables</h2>
      <div class="table-grid">
        <section class="table-card">
          <h3>Counts</h3>
          {mapping_table(counts, key_name="Metric", value_name="Value")}
        </section>
        <section class="table-card">
          <h3>Source Usage Checks</h3>
          {mapping_table(source_usage_checks, key_name="Metric", value_name="Value")}
        </section>
        <section class="table-card">
          <h3>Reward</h3>
          {mapping_table(reward_stats, key_name="Statistic", value_name="Reward")}
        </section>
        <section class="table-card">
          <h3>Duration</h3>
          {mapping_table(duration_stats, key_name="Statistic", value_name="Seconds")}
        </section>
        <section class="table-card">
          <h3>LLM Calls</h3>
          {mapping_table(llm_stats, key_name="Statistic", value_name="Calls")}
        </section>
        <section class="table-card scrollable">
          <h3>Rollouts</h3>
          <div class="table-scroll">
            {rollout_table}
          </div>
        </section>
      </div>

      <details>
        <summary>Preview: rollouts.csv</summary>
        {preview_table(rollout_rows)}
      </details>
      <details>
        <summary>Preview: errors.csv</summary>
        {preview_table(error_rows)}
      </details>
      <details>
        <summary>Preview: llm_calls.csv</summary>
        {preview_table(llm_call_rows)}
      </details>
    </section>

    <section class="section" id="raw">
      <h2>Raw Files + Caveats</h2>
      <div class="table-grid">
        <section class="table-card">
          <h3>Raw Outputs</h3>
          <ul class="raw-links">
            {raw_links_html}
          </ul>
        </section>
        <section class="table-card">
          <h3>Caveats</h3>
          <ul class="limitations">
            {''.join(f"<li>{html.escape(item)}</li>" for item in caveats)}
          </ul>
        </section>
      </div>
    </section>

    <section class="section" id="traces">
      <h2>Representative Traces</h2>
      <p class="section-note">These rooflines are intentionally at the bottom so the main summary reads first. Each card links to a sampled report, interactive timeline, trace JSON, and event log for manual debugging.</p>
      <div class="samples">
        {''.join(roofline_cards)}
      </div>
    </section>
  </div>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "runtime_analysis" / run_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_lookup = load_problem_lookup(run_dir)
    rollouts, error_rows, event_rows, sequences = load_rollouts(run_dir, problem_lookup)
    llm_call_rows = load_llm_call_rows(run_dir)
    rollout_rate = compute_completed_rollout_rate(run_dir / "actor" / "info.log")

    write_csv(output_dir / "rollouts_sample.csv", sample_rows_for_export(rollouts, 2500, group_key="iterations_used"))
    write_csv(output_dir / "errors.csv", error_rows)
    write_csv(output_dir / "events_sample.csv", sample_rows_for_export(event_rows, 4000, group_key="stage"))
    write_csv(output_dir / "llm_calls_sample.csv", sample_rows_for_export(llm_call_rows, 4000, group_key="log_name"))

    rewards = [float(row["reward"]) for row in rollouts if row.get("reward") not in ("", None)]
    durations = [float(row["duration_s"]) for row in rollouts if row.get("duration_s") not in ("", None)]
    iterations = [int(row["iterations_used"]) for row in rollouts if row.get("iterations_used") not in ("", None)]
    llm_calls = [int(row["llm_calls_total"]) for row in rollouts if row.get("llm_calls_total") not in ("", None)]
    rollout_total_tokens = [int(row["total_tokens"]) for row in rollouts if row.get("total_tokens") not in ("", None)]

    stage_durations: dict[str, list[float]] = defaultdict(list)
    for row in event_rows:
        stage = str(row.get("stage") or "")
        if stage == "error":
            continue
        stage_durations[stage].append(float(row.get("duration_s") or 0.0))

    reward_counter = Counter(f"{value:.2f}" for value in rewards)
    iteration_counter = Counter(str(value) for value in iterations)
    error_kind_counter = Counter(str(row.get("kind") or "other") for row in error_rows)
    parse_errors_by_stage = Counter(str(row.get("stage") or "") for row in error_rows if str(row.get("kind") or "") == "parse_error")
    llm_calls_by_stage = Counter(str(row.get("log_name") or "") for row in llm_call_rows)
    prompt_tokens_by_stage: dict[str, list[float]] = defaultdict(list)
    output_tokens_by_stage: dict[str, list[float]] = defaultdict(list)
    total_tokens_by_stage: dict[str, list[float]] = defaultdict(list)
    for row in llm_call_rows:
        stage = str(row.get("log_name") or "")
        prompt_tokens_by_stage[stage].append(float(row.get("prompt_tokens") or 0.0))
        output_tokens_by_stage[stage].append(float(row.get("output_tokens") or 0.0))
        total_tokens_by_stage[stage].append(float(row.get("total_tokens") or 0.0))

    plots: list[str] = []
    reward_plot = output_dir / "reward_distribution.png"
    plot_bar(reward_counter, reward_plot, title="Reward Distribution", xlabel="Reward", ylabel="Rollouts")
    if reward_plot.exists():
        plots.append(reward_plot.name)

    duration_plot = output_dir / "rollout_duration_distribution.png"
    plot_hist(durations, duration_plot, title="Rollout Duration Distribution", xlabel="Duration (seconds)")
    if duration_plot.exists():
        plots.append(duration_plot.name)

    iteration_plot = output_dir / "iteration_distribution.png"
    plot_bar(iteration_counter, iteration_plot, title="Iteration Distribution", xlabel="Iterations used", ylabel="Rollouts")
    if iteration_plot.exists():
        plots.append(iteration_plot.name)

    llm_calls_plot = output_dir / "llm_calls_distribution.png"
    plot_hist(llm_calls, llm_calls_plot, title="LLM Calls per Rollout", xlabel="LLM calls")
    if llm_calls_plot.exists():
        plots.append(llm_calls_plot.name)

    duration_by_iteration_plot = output_dir / "rollout_duration_by_final_iteration.png"
    plot_metric_by_group(
        rollouts,
        "duration_s",
        "iterations_used",
        duration_by_iteration_plot,
        title="Rollout Duration by Final Iteration Count",
        xlabel="Duration (seconds)",
        ylabel="Final iteration count",
    )
    if duration_by_iteration_plot.exists():
        plots.append(duration_by_iteration_plot.name)

    searches_docs_plot = output_dir / "searches_vs_docs_read.png"
    plot_search_docs_scatter(rollouts, searches_docs_plot)
    if searches_docs_plot.exists():
        plots.append(searches_docs_plot.name)

    source_usage_scatter_plot = output_dir / "source_usage_scatter.png"
    plot_source_usage_scatter(rollouts, source_usage_scatter_plot)
    if source_usage_scatter_plot.exists():
        plots.append(source_usage_scatter_plot.name)

    source_usage_dist_plot = output_dir / "source_usage_distributions.png"
    plot_source_usage_distributions(rollouts, source_usage_dist_plot)
    if source_usage_dist_plot.exists():
        plots.append(source_usage_dist_plot.name)

    stage_duration_plot = output_dir / "stage_duration_boxplot.png"
    plot_grouped_boxplot(stage_durations, stage_duration_plot, title="Stage Duration Distribution", xlabel="Duration (seconds)", ylabel="Stage")
    if stage_duration_plot.exists():
        plots.append(stage_duration_plot.name)

    error_plot = output_dir / "errors_by_stage.png"
    plot_error_bars(error_rows, error_plot)
    if error_plot.exists():
        plots.append(error_plot.name)

    kind_plot = output_dir / "errors_by_kind.png"
    plot_bar(error_kind_counter, kind_plot, title="Error Kinds", xlabel="Error kind", ylabel="Count")
    if kind_plot.exists():
        plots.append(kind_plot.name)

    reward_by_hops_plot = output_dir / "reward_by_hop_length.png"
    plot_metric_by_group(
        rollouts,
        "reward",
        "hop_count",
        reward_by_hops_plot,
        title="Reward by Hop Length",
        xlabel="Reward",
        ylabel="Hop count",
    )
    if reward_by_hops_plot.exists():
        plots.append(reward_by_hops_plot.name)

    reward_by_pattern_plot = output_dir / "reward_by_hop_pattern.png"
    plot_metric_by_group(
        rollouts,
        "reward",
        "hop_pattern",
        reward_by_pattern_plot,
        title="Reward by Hop Pattern",
        xlabel="Reward",
        ylabel="Hop pattern",
        top_n=12,
    )
    if reward_by_pattern_plot.exists():
        plots.append(reward_by_pattern_plot.name)

    hop_accuracy_values: dict[str, float] = {}
    hop_accuracy_rows: list[list[str]] = []
    max_hop = max(
        [
            int(key.split("_")[1])
            for row in rollouts
            for key in row.keys()
            if key.startswith("hop_") and key.endswith("_correct")
        ]
        or [0]
    )
    for hop_num in range(1, max_hop + 1):
        key = f"hop_{hop_num}_correct"
        values = [bool(row[key]) for row in rollouts if key in row]
        if not values:
            continue
        solved = sum(int(value) for value in values)
        accuracy = solved / len(values)
        hop_accuracy_values[str(hop_num)] = accuracy
        hop_accuracy_rows.append([str(hop_num), f"{100.0 * accuracy:.1f}%", f"{solved} / {len(values)}"])

    hop_accuracy_plot = output_dir / "hop_position_accuracy.png"
    plot_named_values(
        hop_accuracy_values,
        hop_accuracy_plot,
        title="Per-Hop Position Accuracy",
        xlabel="Hop position",
        ylabel="Accuracy",
        percentage=True,
    )
    if hop_accuracy_plot.exists():
        plots.append(hop_accuracy_plot.name)

    parse_rate_values: dict[str, float] = {}
    parse_rate_rows: list[list[str]] = []
    for stage in sorted(set(parse_errors_by_stage) | set(llm_calls_by_stage), key=lambda value: FLOW_STAGE_ORDER.index(value) if value in FLOW_STAGE_ORDER else 999):
        attempts = int(llm_calls_by_stage.get(stage, 0))
        if attempts <= 0:
            continue
        parse_errors = int(parse_errors_by_stage.get(stage, 0))
        rate = parse_errors / attempts
        parse_rate_values[stage] = rate
        parse_rate_rows.append([stage, str(parse_errors), str(attempts), f"{100.0 * rate:.2f}%"])

    parse_rate_plot = output_dir / "parse_error_rate_by_stage.png"
    plot_named_values(
        parse_rate_values,
        parse_rate_plot,
        title="Parse Error Rate by Stage",
        xlabel="Stage",
        ylabel="Parse error rate",
        percentage=True,
    )
    if parse_rate_plot.exists():
        plots.append(parse_rate_plot.name)

    total_tokens_rollout_plot = output_dir / "llm_total_tokens_per_rollout.png"
    plot_hist(
        [float(value) for value in rollout_total_tokens],
        total_tokens_rollout_plot,
        title="LLM Total Tokens per Rollout",
        xlabel="Total tokens",
    )
    if total_tokens_rollout_plot.exists():
        plots.append(total_tokens_rollout_plot.name)

    prompt_tokens_plot = output_dir / "prompt_tokens_by_call_type.png"
    plot_grouped_boxplot(
        prompt_tokens_by_stage,
        prompt_tokens_plot,
        title="Prompt Tokens by LLM Call Type",
        xlabel="Prompt tokens",
        ylabel="Call type",
    )
    if prompt_tokens_plot.exists():
        plots.append(prompt_tokens_plot.name)

    output_tokens_plot = output_dir / "output_tokens_by_call_type.png"
    plot_grouped_boxplot(
        output_tokens_by_stage,
        output_tokens_plot,
        title="Output Tokens by LLM Call Type",
        xlabel="Output tokens",
        ylabel="Call type",
    )
    if output_tokens_plot.exists():
        plots.append(output_tokens_plot.name)

    total_tokens_plot = output_dir / "total_tokens_by_call_type.png"
    plot_grouped_boxplot(
        total_tokens_by_stage,
        total_tokens_plot,
        title="Total Tokens by LLM Call Type",
        xlabel="Total tokens",
        ylabel="Call type",
    )
    if total_tokens_plot.exists():
        plots.append(total_tokens_plot.name)

    total_tokens_by_iteration_plot = output_dir / "total_tokens_by_final_iteration.png"
    plot_metric_by_group(
        rollouts,
        "total_tokens",
        "iterations_used",
        total_tokens_by_iteration_plot,
        title="Total Tokens by Final Iteration Count",
        xlabel="Total tokens",
        ylabel="Final iteration count",
    )
    if total_tokens_by_iteration_plot.exists():
        plots.append(total_tokens_by_iteration_plot.name)

    stage_flow_path = output_dir / "stage_flow.svg"
    if stage_flow_path.exists():
        stage_flow_path.unlink()

    sampled_rows = select_representative_rollouts(rollouts, limit=12)
    roofline_dir = output_dir / "sample_rooflines"
    roofline_dir.mkdir(parents=True, exist_ok=True)
    for row in sampled_rows:
        events_path = str(row.get("events_path") or "")
        if not events_path:
            continue
        source = Path(events_path)
        if not source.exists():
            continue
        events = load_json(source)
        roofline_path = roofline_dir / f"{row['rollout_id']}.png"
        plot_rollout_roofline(
            events,
            roofline_path,
            title=f"{row['rollout_id']} ({row.get('hop_pattern') or ''}, reward={row.get('reward')})",
        )
        if roofline_path.exists():
            row["roofline_path"] = str(roofline_path)
    copy_sample_artifacts(output_dir, sampled_rows)

    llm_token_rows: list[list[str]] = []
    for stage in sorted(total_tokens_by_stage.keys(), key=lambda value: FLOW_STAGE_ORDER.index(value) if value in FLOW_STAGE_ORDER else 999):
        totals = total_tokens_by_stage.get(stage) or []
        prompts = prompt_tokens_by_stage.get(stage) or []
        outputs = output_tokens_by_stage.get(stage) or []
        if not totals:
            continue
        llm_token_rows.append(
            [
                stage,
                str(len(totals)),
                f"{median(prompts):.1f}",
                f"{median(outputs):.1f}",
                f"{median(totals):.1f}",
                f"{mean(totals):.1f}",
            ]
        )

    overview = {
        "rollout_count": len(rollouts),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "solved_fraction": (
            sum(1 for row in rollouts if float(row.get("reward") or 0.0) >= 0.999) / len(rollouts)
            if rollouts
            else 0.0
        ),
        "mean_duration_s": mean(durations) if durations else 0.0,
        "median_duration_s": median(durations) if durations else 0.0,
        "median_iterations": median(iterations) if iterations else 0.0,
        "mean_llm_calls": mean(llm_calls) if llm_calls else 0.0,
        "median_llm_calls": median(llm_calls) if llm_calls else 0.0,
        "mean_prompt_tokens": mean([float(row.get("prompt_tokens") or 0.0) for row in llm_call_rows]) if llm_call_rows else 0.0,
        "median_prompt_tokens": median([float(row.get("prompt_tokens") or 0.0) for row in llm_call_rows]) if llm_call_rows else 0.0,
        "mean_output_tokens": mean([float(row.get("output_tokens") or 0.0) for row in llm_call_rows]) if llm_call_rows else 0.0,
        "median_output_tokens": median([float(row.get("output_tokens") or 0.0) for row in llm_call_rows]) if llm_call_rows else 0.0,
        "completed_rollouts_per_s": rollout_rate.get("completed_rollouts_per_s"),
        "seconds_per_completed_rollout": rollout_rate.get("seconds_per_completed_rollout"),
        "parse_errors_total": sum(int(row.get("kind") == "parse_error") for row in error_rows),
        "parse_error_rate": (
            sum(int(row.get("kind") == "parse_error") for row in error_rows) / len(llm_call_rows)
            if llm_call_rows
            else 0.0
        ),
        "context_overflow_errors": sum(int(row.get("kind") == "context_overflow") for row in error_rows),
        "mean_web_searches": mean([float(row.get("web_searches") or 0.0) for row in rollouts]) if rollouts else 0.0,
        "mean_local_searches": mean([float(row.get("local_searches") or 0.0) for row in rollouts]) if rollouts else 0.0,
        "mean_web_docs_read": mean([float(row.get("web_docs_read") or 0.0) for row in rollouts]) if rollouts else 0.0,
        "mean_local_docs_read": mean([float(row.get("local_docs_read") or 0.0) for row in rollouts]) if rollouts else 0.0,
        "no_web_hops_with_web_search_fraction": (
            sum(1 for row in rollouts if float(row.get("web_hop_count") or 0.0) == 0 and float(row.get("web_searches") or 0.0) > 0)
            / max(1, sum(1 for row in rollouts if float(row.get("web_hop_count") or 0.0) == 0))
        ) if rollouts else 0.0,
        "no_local_hops_with_local_search_fraction": (
            sum(1 for row in rollouts if float(row.get("local_hop_count") or 0.0) == 0 and float(row.get("local_searches") or 0.0) > 0)
            / max(1, sum(1 for row in rollouts if float(row.get("local_hop_count") or 0.0) == 0))
        ) if rollouts else 0.0,
        "reward_stats": numeric_stats(rewards),
        "duration_stats": numeric_stats(durations),
        "llm_call_stats": numeric_stats([float(value) for value in llm_calls]),
    }
    save_json(output_dir / "summary.json", overview)
    write_csv(
        output_dir / "hop_position_accuracy.csv",
        [{"hop": row[0], "accuracy": row[1], "solved_total": row[2]} for row in hop_accuracy_rows],
    )
    write_csv(
        output_dir / "parse_error_rates.csv",
        [{"stage": row[0], "parse_errors": row[1], "attempts": row[2], "rate": row[3]} for row in parse_rate_rows],
    )
    write_csv(
        output_dir / "llm_token_summary.csv",
        [
            {
                "call_type": row[0],
                "calls": row[1],
                "median_prompt_tokens": row[2],
                "median_output_tokens": row[3],
                "median_total_tokens": row[4],
                "mean_total_tokens": row[5],
            }
            for row in llm_token_rows
        ],
    )

    html_doc = render_index_html(
        output_dir,
        run_dir.name,
        overview,
        plots,
        sampled_rows,
        rollouts,
        error_rows,
        llm_call_rows,
        hop_accuracy_rows,
        parse_rate_rows,
        llm_token_rows,
    )
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    main()
