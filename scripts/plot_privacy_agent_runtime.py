#!/usr/bin/env python3
"""Generate runtime analysis plots for a privacy-agent run directory."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTOR_TS_RE = re.compile(r"\[actor\]: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
LLM_DONE_RE = re.compile(r"LLM request done in ([0-9.]+)s")
PROMPT_SUMMARY_RE = re.compile(r"privacy_agent prompt budget summary: (\{.*\})")
LLM_CALL_SUMMARY_RE = re.compile(r"privacy_agent llm call summary: (\{.*\})")
REPORT_TS_RE = re.compile(r"research_report_(\d{8}_\d{6})\.md$")
ANSWER_RE = re.compile(r"^ANSWER_(\w+):\s*(.+)$")
JUSTIFICATION_RE = re.compile(r"^JUSTIFICATION_(\w+):\s*(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to a privacy-agent output directory")
    parser.add_argument(
        "--output-dir",
        help="Directory to write plots and CSVs into. Defaults to MosaicProject/runtime_analysis/<run_name>",
    )
    return parser.parse_args()


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_actor_timestamp(line: str) -> datetime | None:
    match = ACTOR_TS_RE.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def parse_report_timestamp(path: Path) -> datetime | None:
    match = REPORT_TS_RE.search(path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    try:
        return datetime.utcfromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


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
        "p99": percentile(values, 99),
        "max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


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
    for raw_line in config_path.read_text().splitlines():
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
                parsed = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
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


def parse_actor_info(
    actor_info_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    llm_requests: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    llm_call_records: list[dict[str, Any]] = []
    with actor_info_path.open(errors="ignore") as handle:
        for raw_line in handle:
            timestamp = parse_actor_timestamp(raw_line)
            if not timestamp:
                continue

            llm_match = LLM_DONE_RE.search(raw_line)
            if llm_match:
                llm_requests.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "duration_s": float(llm_match.group(1)),
                    }
                )

            prompt_match = PROMPT_SUMMARY_RE.search(raw_line)
            if prompt_match:
                payload = ast.literal_eval(prompt_match.group(1))
                by_category = payload.get("by_category", {})
                findings = by_category.get("findings_or_summary", {})
                prompt_records.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "log_name": payload.get("log_name", "unknown"),
                        "prompt_tokens": int(payload.get("prompt_tokens", 0)),
                        "prompt_chars": int(payload.get("prompt_chars", 0)),
                        "latest_findings_tokens": int(findings.get("tokens", 0)),
                        "latest_findings_chars": int(findings.get("chars", 0)),
                        "instructions_tokens": int(by_category.get("instructions", {}).get("tokens", 0)),
                        "planning_context_tokens": int(by_category.get("planning_context", {}).get("tokens", 0)),
                        "task_definition_tokens": int(by_category.get("task_definition", {}).get("tokens", 0)),
                        "documents_or_evidence_tokens": int(
                            by_category.get("documents_or_evidence", {}).get("tokens", 0)
                        ),
                        "reporting_context_tokens": int(
                            by_category.get("reporting_context", {}).get("tokens", 0)
                        ),
                        "other_tokens": int(by_category.get("other", {}).get("tokens", 0)),
                    }
                )

            call_match = LLM_CALL_SUMMARY_RE.search(raw_line)
            if call_match:
                try:
                    payload = json.loads(call_match.group(1))
                except json.JSONDecodeError:
                    continue
                llm_call_records.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "rollout_id": str(payload.get("rollout_id") or ""),
                        "task_id": str(payload.get("task_id") or ""),
                        "chain_id": str(payload.get("chain_id") or ""),
                        "log_name": str(payload.get("log_name") or "unknown"),
                        "prompt_tokens": int(payload.get("prompt_tokens") or 0),
                        "output_tokens": int(payload.get("output_tokens") or 0),
                        "total_tokens": int(payload.get("total_tokens") or 0),
                        "capture_mode": str(payload.get("capture_mode") or ""),
                        "captured": bool(payload.get("captured")),
                        "requested_model": str(payload.get("requested_model") or ""),
                    }
                )
    return llm_requests, prompt_records, llm_call_records


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
            match = re.search(r"rollouts finished so far: (\d+)", raw_line)
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


def discover_report_artifact(run_dir: Path) -> tuple[bool, Path | None, datetime | None]:
    workspace = run_dir / "workspace"
    reports = sorted(workspace.glob("research_report_*.md"))
    if not reports:
        return False, None, None
    scored_reports: list[tuple[datetime, Path]] = []
    for report_path in reports:
        report_ts = parse_report_timestamp(report_path)
        if report_ts is not None:
            scored_reports.append((report_ts, report_path))
    if not scored_reports:
        latest_report = reports[-1]
        return True, latest_report, parse_report_timestamp(latest_report)
    latest_ts, latest_report = max(scored_reports, key=lambda item: item[0])
    return True, latest_report, latest_ts


def derive_backend_tool(action: dict[str, Any]) -> str:
    actual_output = action.get("actual_output")
    if isinstance(actual_output, dict):
        for key in ("tool", "source", "analysis_type"):
            value = actual_output.get(key)
            if isinstance(value, str) and value:
                return value
    preferred_tool = action.get("preferred_tool")
    if isinstance(preferred_tool, str) and preferred_tool:
        return preferred_tool
    return str(action.get("type", "unknown"))


def parse_rollout_sessions(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rollouts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []

    runs_root = run_dir / "privacy_agent_runs"
    for session_dir in sorted(runs_root.glob("*/session")):
        run_path = session_dir.parent
        initial_path = session_dir / "action_plan_initial.json"
        final_path = session_dir / "action_plan_final.json"
        if not final_path.exists():
            continue

        initial_payload = json.loads(initial_path.read_text()) if initial_path.exists() else {}
        final_payload = json.loads(final_path.read_text())

        initial_ts = parse_iso_timestamp(initial_payload.get("timestamp"))
        final_ts = parse_iso_timestamp(final_payload.get("timestamp"))
        has_report, report_path, report_ts = discover_report_artifact(run_path)
        start_ts = initial_ts
        end_ts = report_ts or final_ts

        final_actions = final_payload.get("actions", [])
        completion_times = [
            parse_iso_timestamp(action.get("completion_time"))
            for action in final_actions
            if parse_iso_timestamp(action.get("completion_time")) is not None
        ]
        if start_ts is None and completion_times:
            start_ts = min(completion_times)
        if end_ts is None and completion_times:
            end_ts = max(completion_times)

        total_tool_exec_s = sum(
            float(action["execution_time"])
            for action in final_actions
            if action.get("execution_time") is not None and action.get("status") == "completed"
        )
        duration_s = None
        if start_ts and end_ts:
            duration_s = max(0.0, (end_ts - start_ts).total_seconds())

        rollout_record = {
            "rollout_id": run_path.name,
            "has_report": has_report,
            "report_path": str(report_path) if report_path else "",
            "start_timestamp": start_ts.isoformat() if start_ts else "",
            "end_timestamp": end_ts.isoformat() if end_ts else "",
            "duration_s": duration_s if duration_s is not None else "",
            "current_iteration": int(final_payload.get("current_iteration", 0)),
            "max_iterations": int(final_payload.get("max_iterations", 0)),
            "total_actions": int(final_payload.get("stats", {}).get("total_actions", len(final_actions))),
            "completed_actions": int(final_payload.get("completed_actions", 0)),
            "failed_actions": int(final_payload.get("failed_actions", 0)),
            "pending_actions": int(
                final_payload.get("stats", {}).get(
                    "pending",
                    sum(1 for action in final_actions if action.get("status") == "pending"),
                )
            ),
            "tool_exec_time_s": total_tool_exec_s,
            "non_tool_time_s": max(0.0, duration_s - total_tool_exec_s) if duration_s is not None else "",
        }
        rollouts.append(rollout_record)

        per_iter_completion: dict[int, list[datetime]] = defaultdict(list)
        per_iter_exec_s: dict[int, float] = defaultdict(float)
        per_iter_action_count: dict[int, int] = defaultdict(int)

        for action in final_actions:
            exec_time = action.get("execution_time")
            completion_time = parse_iso_timestamp(action.get("completion_time"))
            iteration_completed = action.get("iteration_completed")
            action_record = {
                "rollout_id": run_path.name,
                "has_report": has_report,
                "action_id": action.get("id", ""),
                "action_type": action.get("type", "unknown"),
                "backend_tool": derive_backend_tool(action),
                "status": action.get("status", "unknown"),
                "execution_time_s": float(exec_time) if exec_time is not None else "",
                "iteration_completed": iteration_completed if iteration_completed is not None else "",
                "completion_time": completion_time.isoformat() if completion_time else "",
                "created_from_research_step": action.get("created_from_research_step", ""),
                "preferred_tool": action.get("preferred_tool", ""),
            }
            actions.append(action_record)

            if completion_time is not None and iteration_completed is not None:
                iter_idx = int(iteration_completed)
                per_iter_completion[iter_idx].append(completion_time)
                per_iter_action_count[iter_idx] += 1
                if exec_time is not None and action.get("status") == "completed":
                    per_iter_exec_s[iter_idx] += float(exec_time)

        if start_ts is not None:
            previous_end = start_ts
            for iter_idx in sorted(per_iter_completion):
                iter_end = max(per_iter_completion[iter_idx])
                wall_time_s = max(0.0, (iter_end - previous_end).total_seconds())
                tool_time_s = per_iter_exec_s.get(iter_idx, 0.0)
                iterations.append(
                    {
                        "rollout_id": run_path.name,
                        "has_report": has_report,
                        "iteration": iter_idx,
                        "action_count": per_iter_action_count.get(iter_idx, 0),
                        "wall_time_s": wall_time_s,
                        "tool_time_s": tool_time_s,
                        "non_tool_time_s": max(0.0, wall_time_s - tool_time_s),
                    }
                )
                previous_end = max(previous_end, iter_end)

    return rollouts, actions, iterations


def evaluate_rollout_reports(
    run_dir: Path,
    rollouts: list[dict[str, Any]],
    *,
    f1_threshold: float = 0.75,
    sample_size: int = 8,
    sample_seed: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    problem_lookup = load_problem_lookup(run_dir)
    evaluation_rows: list[dict[str, Any]] = []

    for rollout in rollouts:
        report_path_str = str(rollout.get("report_path") or "")
        if not rollout.get("has_report") or not report_path_str:
            continue
        report_path = Path(report_path_str)
        if not report_path.exists():
            continue
        rollout_id = str(rollout.get("rollout_id") or "")
        parts = rollout_id.rsplit("_", 2)
        if len(parts) != 3:
            continue
        task_id, chain_id, _ = parts
        problem = problem_lookup.get((task_id, chain_id))
        if problem is None:
            continue

        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        answers, justifications = parse_report_answers(report_text)
        score = score_chain_answers(problem, answers, f1_threshold=f1_threshold)
        evaluation_rows.append(
            {
                "rollout_id": rollout_id,
                "task_id": task_id,
                "chain_id": chain_id,
                "report_path": report_path_str,
                "reward": score["reward"],
                "correct_hops": score["correct_hops"],
                "total_hops": score["total_hops"],
                "final_correct": bool(score["final_correct"]),
                "final_match_score": score["final_match_score"],
                "iterations": rollout.get("current_iteration", ""),
                "total_actions": rollout.get("total_actions", ""),
                "duration_s": rollout.get("duration_s", ""),
                "predicted_final_answer": answers.get("FINAL", ""),
                "ground_truth_final_answer": problem.get("global_answer", ""),
                "predicted_answers_json": json.dumps(answers, sort_keys=True),
                "ground_truth_hops_json": json.dumps(problem.get("hops", []), sort_keys=True),
                "per_hop_json": json.dumps(score["per_hop"], sort_keys=True),
                "report_preview": report_text[:2000],
                "task_question_preview": str(problem.get("numbered_questions", ""))[:1000],
                "justifications_json": json.dumps(justifications, sort_keys=True),
            }
        )

    rng = random.Random(sample_seed)
    if len(evaluation_rows) <= sample_size:
        sampled_rows = list(evaluation_rows)
    else:
        sampled_rows = rng.sample(evaluation_rows, sample_size)
    sampled_rows.sort(key=lambda row: str(row["rollout_id"]))
    return evaluation_rows, sampled_rows


def aggregate_group_stats(records: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record.get(value_key)
        if value in ("", None):
            continue
        grouped[str(record.get(group_key, "unknown"))].append(float(value))
    return {group: numeric_stats(values) for group, values in sorted(grouped.items())}


def aggregate_llm_token_usage(records: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "call_count": 0,
            "captured_call_count": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    for record in records:
        key = str(record.get(group_key) or "unknown")
        grouped[key]["call_count"] += 1
        grouped[key]["captured_call_count"] += int(bool(record.get("captured")))
        grouped[key]["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
        grouped[key]["output_tokens"] += int(record.get("output_tokens") or 0)
        grouped[key]["total_tokens"] += int(record.get("total_tokens") or 0)

    result: dict[str, dict[str, Any]] = {}
    for key, stats in grouped.items():
        call_count = max(1, int(stats["call_count"]))
        result[key] = {
            **stats,
            "mean_prompt_tokens": stats["prompt_tokens"] / call_count,
            "mean_output_tokens": stats["output_tokens"] / call_count,
            "mean_total_tokens": stats["total_tokens"] / call_count,
        }
    return dict(
        sorted(
            result.items(),
            key=lambda item: (item[1]["total_tokens"], item[1]["call_count"], item[0]),
            reverse=True,
        )
    )


def plot_hist_and_ecdf(
    values: list[float],
    path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 50,
) -> None:
    if not values:
        return
    ordered = sorted(values)
    ecdf_y = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(values, bins=bins, color="#4472c4", edgecolor="white")
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")
    axes[1].plot(ordered, ecdf_y, color="#c44e52")
    axes[1].set_title(f"{title} ECDF")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Fraction <= x")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bar(counter: Counter[str], path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    if not counter:
        return
    def _numeric_key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    labels_and_values = list(counter.items())
    if xlabel.lower().startswith("final iteration") or xlabel.lower().startswith("reward"):
        labels_and_values.sort(key=lambda item: _numeric_key(str(item[0])))
    else:
        labels_and_values.sort(key=lambda item: item[1], reverse=True)
    labels, values = zip(*labels_and_values)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_boxplot(
    grouped: dict[str, list[float]],
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n: int = 10,
    use_log_scale: bool = False,
) -> None:
    non_empty = [(label, values) for label, values in grouped.items() if values]
    if not non_empty:
        return
    non_empty.sort(key=lambda item: (len(item[1]), median(item[1])), reverse=True)
    chosen = non_empty[:top_n]
    labels = [label for label, _ in chosen]
    data = [values for _, values in chosen]

    fig, ax = plt.subplots(figsize=(11, max(4.5, len(labels) * 0.55)))
    ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if use_log_scale:
        ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rollout_duration_by_iterations(rollouts: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        duration = rollout.get("duration_s")
        if duration in ("", None):
            continue
        grouped[str(rollout["current_iteration"])].append(float(duration))
    plot_grouped_boxplot(
        grouped,
        path,
        title="Rollout Duration by Final Iteration Count",
        xlabel="Rollout duration (seconds)",
        ylabel="Final iteration count",
        top_n=12,
        use_log_scale=False,
    )


def plot_iteration_breakdown(iterations: list[dict[str, Any]], path: Path) -> None:
    if not iterations:
        return
    by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in iterations:
        if not row.get("has_report"):
            continue
        by_iteration[int(row["iteration"])].append(row)
    if not by_iteration:
        return

    labels = sorted(by_iteration)
    median_tool = [median([float(row["tool_time_s"]) for row in by_iteration[label]]) for label in labels]
    median_non_tool = [median([float(row["non_tool_time_s"]) for row in by_iteration[label]]) for label in labels]
    counts = [len(by_iteration[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, median_tool, label="Tool execution", color="#4c78a8")
    ax.bar(labels, median_non_tool, bottom=median_tool, label="LLM + orchestration", color="#f58518")
    for idx, count in enumerate(counts):
        ax.text(labels[idx], median_tool[idx] + median_non_tool[idx], str(count), ha="center", va="bottom", fontsize=8)
    ax.set_title("Median Wall Time per Iteration")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("Median time (seconds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_prompt_token_breakdown(prompt_records: list[dict[str, Any]], path: Path) -> None:
    if not prompt_records:
        return
    counts = Counter(record["log_name"] for record in prompt_records)
    labels = [label for label, _ in counts.most_common(8)]
    grouped = {
        label: [int(record["prompt_tokens"]) for record in prompt_records if record["log_name"] == label]
        for label in labels
    }
    median_breakdown = {
        "Latest Findings": [],
        "Documents/Evidence": [],
        "Task Definition": [],
        "Planning Context": [],
        "Instructions": [],
        "Other": [],
    }
    for label in labels:
        rows = [record for record in prompt_records if record["log_name"] == label]
        median_breakdown["Latest Findings"].append(
            median([int(row["latest_findings_tokens"]) for row in rows]) if rows else 0.0
        )
        median_breakdown["Documents/Evidence"].append(
            median([int(row["documents_or_evidence_tokens"]) for row in rows]) if rows else 0.0
        )
        median_breakdown["Task Definition"].append(
            median([int(row["task_definition_tokens"]) for row in rows]) if rows else 0.0
        )
        median_breakdown["Planning Context"].append(
            median([int(row["planning_context_tokens"]) for row in rows]) if rows else 0.0
        )
        median_breakdown["Instructions"].append(
            median([int(row["instructions_tokens"]) for row in rows]) if rows else 0.0
        )
        median_breakdown["Other"].append(
            median(
                [
                    int(row["reporting_context_tokens"]) + int(row["other_tokens"])
                    for row in rows
                ]
            )
            if rows
            else 0.0
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, max(4.5, len(labels) * 0.65)))
    axes[0].boxplot([grouped[label] for label in labels], vert=False, tick_labels=labels, showfliers=False)
    axes[0].set_title("Prompt Tokens by LLM Call Type")
    axes[0].set_xlabel("Prompt tokens")
    axes[0].set_ylabel("LLM call type")
    axes[0].grid(axis="x", alpha=0.25)

    left = [0.0] * len(labels)
    colors = {
        "Latest Findings": "#e45756",
        "Documents/Evidence": "#4c78a8",
        "Task Definition": "#72b7b2",
        "Planning Context": "#54a24b",
        "Instructions": "#f58518",
        "Other": "#9d755d",
    }
    for category, values in median_breakdown.items():
        axes[1].barh(labels, values, left=left, label=category, color=colors[category])
        left = [left[idx] + values[idx] for idx in range(len(labels))]
    axes[1].set_title("Median Prompt Token Breakdown")
    axes[1].set_xlabel("Median tokens")
    axes[1].set_ylabel("LLM call type")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stacked_token_totals(
    aggregated: dict[str, dict[str, Any]],
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n: int = 12,
) -> None:
    if not aggregated:
        return
    items = list(aggregated.items())[:top_n]
    labels = [label for label, _ in items]
    prompt_values = [int(stats.get("prompt_tokens", 0)) for _, stats in items]
    output_values = [int(stats.get("output_tokens", 0)) for _, stats in items]

    fig, ax = plt.subplots(figsize=(11, max(4.5, len(labels) * 0.55)))
    ax.barh(labels, prompt_values, color="#4c78a8", label="Prompt tokens")
    ax.barh(labels, output_values, left=prompt_values, color="#f58518", label="Response tokens")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def llm_token_usage_table(mapping: dict[str, dict[str, Any]], *, key_name: str, limit: int | None = None) -> str:
    headers = [
        key_name,
        "call_count",
        "captured_call_count",
        "prompt_tokens",
        "response_tokens",
        "total_tokens",
        "mean_prompt_tokens",
        "mean_response_tokens",
    ]
    items = list(mapping.items())
    if limit is not None:
        items = items[:limit]
    rows = []
    for key, stats in items:
        rows.append(
            [
                key,
                stats.get("call_count", 0),
                stats.get("captured_call_count", 0),
                stats.get("prompt_tokens", 0),
                stats.get("output_tokens", 0),
                stats.get("total_tokens", 0),
                stats.get("mean_prompt_tokens"),
                stats.get("mean_output_tokens"),
            ]
        )
    return html_table(headers, rows)


def write_summary_markdown(
    path: Path,
    *,
    run_dir: Path,
    llm_requests: list[dict[str, Any]],
    prompt_records: list[dict[str, Any]],
    llm_call_records: list[dict[str, Any]],
    reward_records: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    completed_rollouts: list[dict[str, Any]],
) -> None:
    llm_values = [float(row["duration_s"]) for row in llm_requests]
    llm_output_values = [int(row["output_tokens"]) for row in llm_call_records]
    llm_total_values = [int(row["total_tokens"]) for row in llm_call_records]
    reward_values = [float(row["reward"]) for row in reward_records]
    completed_durations = [float(row["duration_s"]) for row in completed_rollouts if row.get("duration_s") not in ("", None)]
    tool_exec = [float(row["execution_time_s"]) for row in actions if row.get("execution_time_s") not in ("", None)]
    tool_counter = Counter(row["action_type"] for row in actions if row.get("has_report"))
    llm_counter = Counter(row["log_name"] for row in llm_call_records)
    findings_share = [
        int(row["latest_findings_tokens"]) / int(row["prompt_tokens"])
        for row in prompt_records
        if int(row["prompt_tokens"]) > 0
    ]

    lines = [
        f"# Runtime Analysis for `{run_dir.name}`",
        "",
        "## Notes",
        "- Tool timing is read from `session/action_plan_final.json` and grouped only over rollouts with a saved report.",
        "- Exact per-`log_name` LLM latency is not recoverable from `actor/info.log` because request durations are logged without request IDs under concurrency.",
        "- LLM plots therefore show overall latency plus prompt-size distributions by `log_name`.",
        "",
        "## Headline Stats",
        f"- Completed rollouts analyzed: `{len(completed_rollouts)}`",
        f"- Total completed tool actions analyzed: `{sum(tool_counter.values())}`",
        f"- Total LLM request duration samples: `{len(llm_values)}`",
        f"- Total LLM call summaries: `{len(llm_call_records)}`",
        f"- Evaluated reports with rewards: `{len(reward_records)}`",
        f"- Median LLM request latency: `{median(llm_values):.2f}s`" if llm_values else "- Median LLM request latency: n/a",
        f"- Median LLM response tokens: `{median(llm_output_values):.0f}`"
        if llm_output_values
        else "- Median LLM response tokens: n/a",
        f"- Median LLM total tokens: `{median(llm_total_values):.0f}`"
        if llm_total_values
        else "- Median LLM total tokens: n/a",
        f"- Mean rollout reward: `{mean(reward_values):.3f}`" if reward_values else "- Mean rollout reward: n/a",
        f"- Median rollout reward: `{median(reward_values):.3f}`" if reward_values else "- Median rollout reward: n/a",
        f"- Median completed rollout duration: `{median(completed_durations):.2f}s`" if completed_durations else "- Median completed rollout duration: n/a",
        f"- Median tool execution time: `{median(tool_exec):.2f}s`" if tool_exec else "- Median tool execution time: n/a",
        f"- Median Latest Findings share of logged prompts: `{median(findings_share) * 100:.1f}%`"
        if findings_share
        else "- Median Latest Findings share of logged prompts: n/a",
        "",
        "## Top LLM Call Types",
    ]
    for label, count in llm_counter.most_common(8):
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(
        [
            "",
        "## Top Action Types",
        ]
    )
    for label, count in tool_counter.most_common(8):
        lines.append(f"- `{label}`: `{count}`")
    lines.append("")
    path.write_text("\n".join(lines))


def format_scalar(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isfinite(value):
            if abs(value) >= 1000:
                return f"{value:,.2f}"
            if abs(value) >= 10:
                return f"{value:.2f}"
            return f"{value:.3f}"
        return str(value)
    return str(value)


def html_table(headers: list[str], rows: list[list[Any]], *, table_class: str = "data-table") -> str:
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(format_scalar(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "\n".join(body_rows) if body_rows else f"<tr><td colspan='{len(headers)}'>No data</td></tr>"
    return (
        f"<table class='{table_class}'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def mapping_table(mapping: dict[str, Any], *, key_name: str, value_name: str) -> str:
    rows = [[key, value] for key, value in mapping.items()]
    return html_table([key_name, value_name], rows)


def stats_mapping_table(mapping: dict[str, dict[str, Any]], *, key_name: str) -> str:
    headers = [key_name, "count", "mean", "median", "p90", "p95", "p99", "max"]
    rows = []
    for key, stats in mapping.items():
        rows.append(
            [
                key,
                stats.get("count", 0),
                stats.get("mean"),
                stats.get("median"),
                stats.get("p90"),
                stats.get("p95"),
                stats.get("p99"),
                stats.get("max"),
            ]
        )
    return html_table(headers, rows)


def preview_table(records: list[dict[str, Any]], *, limit: int = 15) -> str:
    rows = records[:limit]
    if not rows:
        return "<p>No rows.</p>"
    headers = list(rows[0].keys())
    body = [[row.get(header, "") for header in headers] for row in rows]
    return html_table(headers, body, table_class="data-table preview-table")


def render_spot_check_samples(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "<p>No evaluated rollout samples available.</p>"

    cards: list[str] = []
    for sample in samples:
        try:
            per_hop = json.loads(sample.get("per_hop_json", "[]"))
        except json.JSONDecodeError:
            per_hop = []
        hop_rows = [
            [
                row.get("hop"),
                row.get("agent_answer", ""),
                row.get("ground_truth", ""),
                row.get("correct", False),
                row.get("match_score"),
            ]
            for row in per_hop
        ]
        hop_table = html_table(
            ["Hop", "Predicted", "Ground Truth", "Correct", "Match"],
            hop_rows,
            table_class="data-table preview-table",
        )
        cards.append(
            f"""
            <details class="sample-card">
              <summary>{html.escape(sample['rollout_id'])} | reward={format_scalar(sample['reward'])} | hops={sample['correct_hops']}/{sample['total_hops']}</summary>
              <p><strong>Task:</strong> {html.escape(str(sample.get('task_id', '')))}</p>
              <p><strong>Iterations:</strong> {html.escape(format_scalar(sample.get('iterations')))} | <strong>Actions:</strong> {html.escape(format_scalar(sample.get('total_actions')))} | <strong>Duration:</strong> {html.escape(format_scalar(sample.get('duration_s')))} s</p>
              <p><strong>Predicted final answer:</strong> {html.escape(str(sample.get('predicted_final_answer', '')))}</p>
              <p><strong>Ground-truth final answer:</strong> {html.escape(str(sample.get('ground_truth_final_answer', '')))}</p>
              <p><strong>Question preview:</strong> {html.escape(str(sample.get('task_question_preview', '')))}</p>
              {hop_table}
              <p><strong>Report path:</strong> {html.escape(str(sample.get('report_path', '')))}</p>
              <pre>{html.escape(str(sample.get('report_preview', '')))}</pre>
            </details>
            """
        )
    return "\n".join(cards)


def pick_legacy_timeline_examples(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rollouts if row.get("has_report")]
    if not completed:
        return []
    single_iter = [
        row for row in completed if int(float(row.get("current_iteration") or 0)) <= 1 and float(row.get("duration_s") or 0.0) > 0.0
    ]
    multi_iter = [row for row in completed if int(float(row.get("current_iteration") or 0)) >= 4]
    chosen: list[dict[str, Any]] = []
    if single_iter:
        chosen.append(min(single_iter, key=lambda row: float(row.get("duration_s") or 0.0)))
    if multi_iter:
        chosen.append(min(multi_iter, key=lambda row: (abs(int(float(row.get("current_iteration") or 0)) - 6), float(row.get("duration_s") or 0.0))))
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in chosen:
        rollout_id = str(row.get("rollout_id") or "")
        if rollout_id and rollout_id not in seen:
            deduped.append(row)
            seen.add(rollout_id)
    return deduped[:2]


def build_legacy_timeline_html(
    *,
    rollout_row: dict[str, Any],
    reward_row: dict[str, Any] | None,
    action_plan: dict[str, Any],
) -> str:
    rollout_id = str(rollout_row.get("rollout_id") or "")
    actions = list(action_plan.get("actions") or [])
    actions.sort(key=lambda item: (int(item.get("iteration_completed") or 0), str(item.get("completion_time") or ""), str(item.get("id") or "")))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[int(action.get("iteration_completed") or 0)].append(action)

    max_exec = max([float(action.get("execution_time") or 0.0) for action in actions] + [0.001])
    sections: list[str] = []
    for iteration in sorted(grouped):
        cards: list[str] = []
        for action in grouped[iteration]:
            output = action.get("actual_output") if isinstance(action.get("actual_output"), dict) else {}
            query = str((action.get("parameters") or {}).get("query") or "")
            summary = str(output.get("summary") or output.get("synthesis") or "")
            results_count = output.get("results_count", "")
            exec_s = float(action.get("execution_time") or 0.0)
            width_pct = 12.0 + 88.0 * math.sqrt(exec_s / max_exec)
            cards.append(
                f"""
                <div class="action-card">
                  <div class="bar" style="width:{width_pct:.1f}%"></div>
                  <div class="action-head">
                    <strong>{html.escape(str(action.get('type') or 'action'))}</strong>
                    <span>{html.escape(format_scalar(exec_s))} s</span>
                  </div>
                  <p><strong>ID:</strong> {html.escape(str(action.get('id') or ''))}</p>
                  <p><strong>Description:</strong> {html.escape(str(action.get('description') or ''))}</p>
                  <p><strong>Query:</strong> {html.escape(query)}</p>
                  <p><strong>Status:</strong> {html.escape(str(action.get('status') or ''))} | <strong>Results:</strong> {html.escape(str(results_count))}</p>
                  <details>
                    <summary>Output preview</summary>
                    <pre>{html.escape(summary or json.dumps(output, indent=2)[:2500])}</pre>
                  </details>
                </div>
                """
            )
        sections.append(
            f"""
            <section class="iter-block">
              <h2>Iteration {iteration}</h2>
              <div class="action-grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    report_path = str(rollout_row.get("report_path") or "")
    report_preview = ""
    if report_path and Path(report_path).exists():
        report_preview = Path(report_path).read_text(encoding="utf-8", errors="ignore")[:3000]

    reward_text = "n/a"
    if reward_row is not None:
        reward_text = f"{format_scalar(reward_row.get('reward'))} ({reward_row.get('correct_hops')}/{reward_row.get('total_hops')} hops)"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Legacy Timeline - {html.escape(rollout_id)}</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f6f7fb; color:#18212b; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    .hero, .iter-block, .table-card {{ background:#fff; border:1px solid #d8dee8; border-radius:18px; padding:18px 20px; box-shadow:0 10px 30px rgba(24,33,43,0.05); margin-bottom:18px; }}
    .hero h1 {{ margin:0 0 8px; font-size:28px; }}
    .hero p {{ margin:6px 0; color:#516172; }}
    .action-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }}
    .action-card {{ border:1px solid #d8dee8; border-radius:14px; padding:14px; background:#fbfcff; position:relative; overflow:hidden; }}
    .bar {{ height:8px; background:linear-gradient(90deg,#2f6fed,#4c78a8); border-radius:999px; margin-bottom:10px; }}
    .action-head {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; }}
    .action-card p {{ margin:8px 0; font-size:14px; }}
    details {{ margin-top:10px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#f7f9fc; border:1px solid #d8dee8; border-radius:12px; padding:10px; max-height:360px; overflow:auto; font-size:12px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Legacy Privacy-Agent Timeline</h1>
      <p><strong>Rollout:</strong> {html.escape(rollout_id)}</p>
      <p><strong>Reward:</strong> {html.escape(reward_text)}</p>
      <p><strong>Iterations:</strong> {html.escape(format_scalar(rollout_row.get('current_iteration')))} | <strong>Total actions:</strong> {html.escape(format_scalar(rollout_row.get('total_actions')))} | <strong>Duration:</strong> {html.escape(format_scalar(rollout_row.get('duration_s')))} s</p>
      <p><strong>Research query:</strong> {html.escape(str(action_plan.get('research_query') or ''))}</p>
    </section>
    {''.join(sections)}
    <section class="table-card">
      <h2>Report Preview</h2>
      <pre>{html.escape(report_preview)}</pre>
    </section>
  </div>
</body>
</html>"""


def generate_legacy_timelines(
    output_dir: Path,
    rollouts: list[dict[str, Any]],
    reward_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    reward_lookup = {str(row.get("rollout_id") or ""): row for row in reward_records}
    timeline_dir = output_dir / "legacy_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    links: list[dict[str, str]] = []
    for rollout in pick_legacy_timeline_examples(rollouts):
        rollout_id = str(rollout.get("rollout_id") or "")
        report_path = Path(str(rollout.get("report_path") or ""))
        plan_path = report_path.parents[1] / "session" / "action_plan_final.json"
        if not plan_path.exists():
            continue
        action_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        html_doc = build_legacy_timeline_html(
            rollout_row=rollout,
            reward_row=reward_lookup.get(rollout_id),
            action_plan=action_plan,
        )
        out_path = timeline_dir / f"{rollout_id}.html"
        out_path.write_text(html_doc, encoding="utf-8")
        links.append(
            {
                "rollout_id": rollout_id,
                "filename": out_path.name,
                "iterations": format_scalar(rollout.get("current_iteration")),
                "duration_s": format_scalar(rollout.get("duration_s")),
            }
        )
    return links


def plot_card(
    title: str,
    filename: str,
    *,
    blurb: str = "",
    available: bool = True,
    missing_note: str = "Plot not available for this run.",
) -> str:
    if not available:
        return f"""
        <section class="plot-card">
          <h3>{html.escape(title)}</h3>
          <p class="blurb">{html.escape(blurb)}</p>
          <p class="section-note">{html.escape(missing_note)}</p>
        </section>
        """
    return f"""
    <section class="plot-card">
      <h3>{html.escape(title)}</h3>
      <p class="blurb">{html.escape(blurb)}</p>
      <a href="{html.escape(filename)}"><img src="{html.escape(filename)}" alt="{html.escape(title)}"></a>
    </section>
    """


def write_html_viewer(
    path: Path,
    *,
    run_dir: Path,
    summary: dict[str, Any],
    llm_requests: list[dict[str, Any]],
    prompt_records: list[dict[str, Any]],
    llm_call_records: list[dict[str, Any]],
    reward_records: list[dict[str, Any]],
    sampled_reward_records: list[dict[str, Any]],
    legacy_timeline_links: list[dict[str, str]],
    rollouts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    iterations: list[dict[str, Any]],
) -> None:
    plot_dir = path.parent

    def has_plot(filename: str) -> bool:
        return (plot_dir / filename).exists()

    counts = dict(summary["counts"])
    counts["completed_rollouts_per_s"] = format_scalar(summary.get("completed_rollouts_per_s"))
    llm_stats = summary["llm_request_latency_s"]
    prompt_stats = summary["logged_prompt_tokens"]
    response_token_stats = summary["logged_response_tokens"]
    total_token_stats = summary["logged_total_tokens"]
    reward_stats = summary["rollout_reward"]
    rollout_stats = summary["completed_rollout_duration_s"]
    action_stats = summary["completed_action_execution_s"]
    llm_calls_per_rollout_stats = summary.get("llm_calls_per_completed_rollout", {"count": 0})

    latest_findings_share = summary.get("median_latest_findings_share", 0.0)
    top_prompt_counts = dict(list(summary["prompt_count_by_log_name"].items())[:12])
    top_llm_token_stats = dict(list(summary["llm_token_usage_by_log_name"].items())[:12])
    top_rollout_token_stats = dict(list(summary["llm_token_usage_by_rollout"].items())[:12])
    reward_distribution = summary["reward_distribution"]
    top_action_stats = dict(
        sorted(
            summary["action_execution_s_by_type"].items(),
            key=lambda item: item[1].get("count", 0),
            reverse=True,
        )[:12]
    )
    top_backend_stats = dict(
        sorted(
            summary["action_execution_s_by_backend"].items(),
            key=lambda item: item[1].get("count", 0),
            reverse=True,
        )[:12]
    )

    cards = [
        ("Completed rollouts", counts["rollouts_completed_with_report"]),
        ("Completed tool actions", counts["actions_completed_with_report"]),
        ("LLM requests", counts["llm_requests"]),
        ("LLM call summaries", counts["llm_call_records"]),
        ("Prompt summaries", counts["prompt_records"]),
        ("Evaluated rewards", counts["reward_records"]),
        ("Mean LLM calls / rollout", format_scalar(llm_calls_per_rollout_stats.get("mean"))),
        ("Median LLM calls / rollout", format_scalar(llm_calls_per_rollout_stats.get("median"))),
        ("Mean prompt tokens", format_scalar(prompt_stats.get("mean"))),
        ("Median prompt tokens", format_scalar(prompt_stats.get("median"))),
        ("Mean response tokens", format_scalar(response_token_stats.get("mean"))),
        ("Median LLM latency", f"{format_scalar(llm_stats.get('median'))} s"),
        ("Median response tokens", format_scalar(response_token_stats.get("median"))),
        ("Mean reward", format_scalar(reward_stats.get("mean"))),
        ("Median reward", format_scalar(reward_stats.get("median"))),
        ("Mean rollout duration", f"{format_scalar(rollout_stats.get('mean'))} s"),
        ("Median rollout duration", f"{format_scalar(rollout_stats.get('median'))} s"),
        ("Completed rollouts / s", format_scalar(summary.get("completed_rollouts_per_s"))),
        ("Median tool execution", f"{format_scalar(action_stats.get('median'))} s"),
        ("Median findings token share", f"{latest_findings_share * 100:.1f}%"),
    ]
    card_html = "\n".join(
        f"<div class='stat-card'><div class='stat-label'>{html.escape(label)}</div><div class='stat-value'>{html.escape(format_scalar(value))}</div></div>"
        for label, value in cards
    )

    overview_plots = "\n".join(
        [
            plot_card(
                "LLM Latency Distribution",
                "llm_latency_distribution.png",
                blurb="Overall LLM request latency. Exact latency by log_name is not recoverable from the concurrent actor logs.",
                available=has_plot("llm_latency_distribution.png"),
            ),
            plot_card(
                "Rollout Duration Distribution",
                "rollout_duration_distribution.png",
                blurb="Completed rollout wall-clock durations, filtered to rollouts with a final report artifact.",
                available=has_plot("rollout_duration_distribution.png"),
            ),
            plot_card(
                "Iteration Count Distribution",
                "iteration_count_distribution.png",
                blurb="Final iteration counts across completed rollouts.",
                available=has_plot("iteration_count_distribution.png"),
            ),
            plot_card(
                "Actions Per Rollout",
                "actions_per_rollout_distribution.png",
                blurb="How many tool actions each completed rollout executed before producing a report.",
                available=has_plot("actions_per_rollout_distribution.png"),
            ),
        ]
    )
    context_plots = "\n".join(
        [
            plot_card(
                "Prompt Tokens by LLM Call Type",
                "prompt_tokens_by_log_name.png",
                blurb="Large adaptive prompts dominate; the second panel now shows a median token breakdown by prompt section instead of a fraction.",
                available=has_plot("prompt_tokens_by_log_name.png"),
            ),
            plot_card(
                "Rollout Duration by Final Iteration",
                "rollout_duration_by_iteration.png",
                blurb="Completed rollout time split by how many iterations the agent ended up using.",
                available=has_plot("rollout_duration_by_iteration.png"),
            ),
            plot_card(
                "Median Time per Iteration",
                "iteration_time_breakdown.png",
                blurb="Median wall time per iteration, split into tool execution versus residual LLM and orchestration time.",
                available=has_plot("iteration_time_breakdown.png"),
            ),
        ]
    )
    reward_plots = "\n".join(
        [
            plot_card(
                "Rollout Reward Distribution",
                "reward_distribution.png",
                blurb="Reward counts recomputed from saved reports and dataset ground truth.",
                available=has_plot("reward_distribution.png"),
            ),
        ]
    )
    llm_token_plots = "\n".join(
        [
            plot_card(
                "Response Token Distribution",
                "llm_response_token_distribution.png",
                blurb="Overall response-token distribution across logged privacy-agent LLM calls.",
                available=has_plot("llm_response_token_distribution.png"),
                missing_note="This run does not have per-call LLM token logs, so response-token plots are unavailable.",
            ),
            plot_card(
                "Response Tokens by LLM Call Type",
                "llm_output_tokens_by_log_name.png",
                blurb="Per-call response-token distribution grouped by log_name.",
                available=has_plot("llm_output_tokens_by_log_name.png"),
                missing_note="This run does not have per-call LLM token logs, so response-token plots are unavailable.",
            ),
            plot_card(
                "Token Totals by LLM Call Type",
                "llm_token_totals_by_log_name.png",
                blurb="Aggregate prompt versus response token volume for the most expensive call types.",
                available=has_plot("llm_token_totals_by_log_name.png"),
                missing_note="This run does not have per-call LLM token logs, so token-total plots are unavailable.",
            ),
            plot_card(
                "Token Totals by Rollout",
                "llm_token_totals_by_rollout.png",
                blurb="Aggregate prompt versus response token volume for the most token-heavy completed rollouts.",
                available=has_plot("llm_token_totals_by_rollout.png"),
                missing_note="This run does not have per-call LLM token logs, so token-total plots are unavailable.",
            ),
        ]
    )
    tool_plots = "\n".join(
        [
            plot_card(
                "Tool Latency by Action Type",
                "tool_latency_by_action_type.png",
                blurb="Distribution of per-action execution time using action categories from the final action plan.",
                available=has_plot("tool_latency_by_action_type.png"),
            ),
            plot_card(
                "Tool Latency by Backend",
                "tool_latency_by_backend.png",
                blurb="Distribution of execution time grouped by the backend tool actually used.",
                available=has_plot("tool_latency_by_backend.png"),
            ),
            plot_card(
                "Tool Counts by Action Type",
                "tool_counts_by_action_type.png",
                blurb="Volume of completed tool calls by action type.",
                available=has_plot("tool_counts_by_action_type.png"),
            ),
            plot_card(
                "Tool Counts by Backend",
                "tool_counts_by_backend.png",
                blurb="Volume of completed tool calls by backend tool.",
                available=has_plot("tool_counts_by_backend.png"),
            ),
        ]
    )

    raw_links = [
        "summary.json",
        "README.md",
        "rollouts_sample.csv",
        "actions_sample.csv",
        "iterations_sample.csv",
        "llm_requests_sample.csv",
        "llm_call_records_sample.csv",
        "llm_tokens_by_rollout.csv",
        "llm_tokens_by_log_name.csv",
        "prompt_records_sample.csv",
        "reward_records_sample.csv",
    ]
    raw_links_html = "\n".join(
        f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>"
        for name in raw_links
        if (plot_dir / name).exists()
    )
    legacy_timelines_html = (
        "<p>No reconstructed legacy timelines available.</p>"
        if not legacy_timeline_links
        else "<ul class='raw-links'>"
        + "".join(
            f"<li><a href='legacy_timelines/{html.escape(link['filename'])}'>{html.escape(link['rollout_id'])}</a> "
            f"(iterations={html.escape(link['iterations'])}, duration={html.escape(link['duration_s'])} s)</li>"
            for link in legacy_timeline_links
        )
        + "</ul>"
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Runtime Analysis - {html.escape(run_dir.name)}</title>
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
    .table-card h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      position: sticky;
      top: 0;
      background: #f7f9fc;
      z-index: 1;
    }}
    .preview-table {{
      font-size: 12px;
    }}
    details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      margin-top: 12px;
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
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Runtime Analysis Viewer</h1>
      <p><strong>Run:</strong> {html.escape(run_dir.name)}</p>
      <p>This viewer is generated from the saved actor logs and per-rollout session artifacts. Tool timings are exact at the action level; LLM timings are exact overall but only prompt sizes are attributable to specific call types because the actor logs request durations without request IDs under concurrency.</p>
      <div class="topnav">
        <a href="#overview">Overview</a>
        <a href="#context">Context + Iterations</a>
        <a href="#llm-tokens">LLM Tokens</a>
        <a href="#rewards">Rewards</a>
        <a href="#tools">Tools</a>
        <a href="#tables">Tables</a>
        <a href="#raw">Raw Files</a>
      </div>
      <div class="stat-grid">
        {card_html}
      </div>
    </section>

    <section class="section" id="overview">
      <h2>Overview Plots</h2>
      <p class="section-note">These are the quickest plots to check for overall throughput shape, rollout duration, and how often the agent keeps iterating.</p>
      <div class="plot-grid">
        {overview_plots}
      </div>
    </section>

    <section class="section" id="context">
      <h2>Context + Iteration Behavior</h2>
      <p class="section-note">This section is usually the most useful for explaining why the run feels slow. It focuses on prompt size and how each extra iteration adds wall time.</p>
      <div class="plot-grid">
        {context_plots}
      </div>
    </section>

    <section class="section" id="llm-tokens">
      <h2>LLM Token Usage</h2>
      <p class="section-note">These views use the explicit `privacy_agent llm call summary` lines emitted by the rollout adapter. They show response tokens and aggregate token volume by call type and rollout.</p>
      <div class="plot-grid">
        {llm_token_plots}
      </div>
    </section>

    <section class="section" id="rewards">
      <h2>Reward Spot Checks</h2>
      <p class="section-note">These views recompute reward from the saved report answers against the dataset ground truth, then show a reproducible random sample of rollout reports for manual inspection.</p>
      <div class="plot-grid">
        {reward_plots}
      </div>
      <div class="table-grid">
        <section class="table-card">
          <h3>Rollout Reward Stats</h3>
          {mapping_table(reward_stats, key_name="Statistic", value_name="Reward")}
        </section>
        <section class="table-card">
          <h3>Reward Distribution</h3>
          {mapping_table(reward_distribution, key_name="Reward", value_name="Rollouts")}
        </section>
        <section class="table-card">
          <h3>Representative Legacy Timelines</h3>
          <p>These are reconstructed from saved action-plan artifacts for two old DRBench-style rollouts. They are less exact than the new hop-QA traces, but still useful for debugging iteration flow.</p>
          {legacy_timelines_html}
        </section>
      </div>
      {render_spot_check_samples(sampled_reward_records)}
    </section>

    <section class="section" id="tools">
      <h2>Tool Timing</h2>
      <p class="section-note">These plots use per-action execution times stored in each rollout&apos;s final action plan, so this breakdown is much more precise than the text logs alone.</p>
      <div class="plot-grid">
        {tool_plots}
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
          <h3>LLM Request Latency</h3>
          {mapping_table(llm_stats, key_name="Statistic", value_name="Seconds")}
        </section>
        <section class="table-card">
          <h3>Logged Prompt Tokens</h3>
          {mapping_table(prompt_stats, key_name="Statistic", value_name="Tokens")}
        </section>
        <section class="table-card">
          <h3>Logged Response Tokens</h3>
          {mapping_table(response_token_stats, key_name="Statistic", value_name="Tokens")}
        </section>
        <section class="table-card">
          <h3>Logged Total Tokens</h3>
          {mapping_table(total_token_stats, key_name="Statistic", value_name="Tokens")}
        </section>
        <section class="table-card">
          <h3>Rollout Reward</h3>
          {mapping_table(reward_stats, key_name="Statistic", value_name="Reward")}
        </section>
        <section class="table-card">
          <h3>Completed Rollout Duration</h3>
          {mapping_table(rollout_stats, key_name="Statistic", value_name="Seconds")}
        </section>
        <section class="table-card">
          <h3>LLM Token Usage by Call Type</h3>
          {llm_token_usage_table(top_llm_token_stats, key_name="Log name", limit=12)}
        </section>
        <section class="table-card">
          <h3>LLM Token Usage by Rollout</h3>
          {llm_token_usage_table(top_rollout_token_stats, key_name="Rollout", limit=12)}
        </section>
        <section class="table-card">
          <h3>Completed Action Execution</h3>
          {mapping_table(action_stats, key_name="Statistic", value_name="Seconds")}
        </section>
        <section class="table-card">
          <h3>Iteration Count Distribution</h3>
          {mapping_table(summary["iteration_count_distribution"], key_name="Final iteration", value_name="Rollouts")}
        </section>
        <section class="table-card">
          <h3>Prompt Count by Log Name</h3>
          {mapping_table(top_prompt_counts, key_name="Log name", value_name="Prompt summaries")}
        </section>
        <section class="table-card">
          <h3>Action Timing by Type</h3>
          {stats_mapping_table(top_action_stats, key_name="Action type")}
        </section>
        <section class="table-card">
          <h3>Action Timing by Backend Tool</h3>
          {stats_mapping_table(top_backend_stats, key_name="Backend")}
        </section>
      </div>

      <details>
        <summary>Preview: rollouts.csv</summary>
        {preview_table(rollouts)}
      </details>
      <details>
        <summary>Preview: actions.csv</summary>
        {preview_table(actions)}
      </details>
      <details>
        <summary>Preview: iterations.csv</summary>
        {preview_table(iterations)}
      </details>
      <details>
        <summary>Preview: prompt_records.csv</summary>
        {preview_table(prompt_records)}
      </details>
      <details>
        <summary>Preview: llm_call_records.csv</summary>
        {preview_table(llm_call_records)}
      </details>
      <details>
        <summary>Preview: llm_requests.csv</summary>
        {preview_table(llm_requests)}
      </details>
      <details>
        <summary>Preview: reward_records.csv</summary>
        {preview_table(reward_records)}
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
          <h3>Limitations</h3>
          <ul class="limitations">
            {''.join(f"<li>{html.escape(item)}</li>" for item in summary.get("limitations", []))}
          </ul>
        </section>
      </div>
    </section>
  </div>
</body>
</html>
"""
    path.write_text(html_doc)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    mosaic_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else mosaic_root / "runtime_analysis" / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_info_path = run_dir / "actor" / "info.log"
    llm_requests, prompt_records, llm_call_records = parse_actor_info(actor_info_path)
    rollout_rate = compute_completed_rollout_rate(actor_info_path)
    rollouts, actions, iterations = parse_rollout_sessions(run_dir)

    rollout_lookup = {str(row["rollout_id"]): row for row in rollouts}
    llm_call_records_enriched: list[dict[str, Any]] = []
    for row in llm_call_records:
        rollout = rollout_lookup.get(str(row.get("rollout_id") or ""), {})
        llm_call_records_enriched.append(
            {
                **row,
                "has_report": bool(rollout.get("has_report")),
                "rollout_duration_s": rollout.get("duration_s", ""),
                "rollout_iteration_count": rollout.get("current_iteration", ""),
            }
        )
    llm_call_records = llm_call_records_enriched

    reward_records, sampled_reward_records = evaluate_rollout_reports(run_dir, rollouts)
    legacy_timeline_links = generate_legacy_timelines(output_dir, rollouts, reward_records)

    completed_rollouts = [row for row in rollouts if row.get("has_report")]
    completed_actions = [row for row in actions if row.get("has_report") and row.get("status") == "completed"]
    completed_iterations = [row for row in iterations if row.get("has_report")]
    completed_llm_call_records = [row for row in llm_call_records if row.get("has_report")]
    llm_call_records_for_rollout_analysis = completed_llm_call_records or llm_call_records

    write_csv(output_dir / "llm_requests_sample.csv", sample_rows_for_export(llm_requests, 3000))
    write_csv(output_dir / "prompt_records_sample.csv", sample_rows_for_export(prompt_records, 3000, group_key="log_name"))
    write_csv(output_dir / "llm_call_records_sample.csv", sample_rows_for_export(llm_call_records, 3000, group_key="log_name"))
    write_csv(output_dir / "reward_records_sample.csv", sample_rows_for_export(reward_records, 300, group_key="reward"))
    write_csv(output_dir / "rollouts_sample.csv", sample_rows_for_export(rollouts, 2500, group_key="current_iteration"))
    write_csv(output_dir / "actions_sample.csv", sample_rows_for_export(actions, 4000, group_key="action_type"))
    write_csv(output_dir / "iterations_sample.csv", sample_rows_for_export(iterations, 3000, group_key="iteration"))

    llm_durations = [float(row["duration_s"]) for row in llm_requests]
    prompt_tokens = [int(row["prompt_tokens"]) for row in prompt_records]
    response_tokens = [int(row["output_tokens"]) for row in llm_call_records]
    total_tokens = [int(row["total_tokens"]) for row in llm_call_records]
    reward_values = [float(row["reward"]) for row in reward_records]
    completed_rollout_durations = [float(row["duration_s"]) for row in completed_rollouts if row.get("duration_s") not in ("", None)]
    actions_per_rollout = [int(row["total_actions"]) for row in completed_rollouts]
    iteration_counts = Counter(int(row["current_iteration"]) for row in completed_rollouts)
    tool_exec_values = [float(row["execution_time_s"]) for row in completed_actions if row.get("execution_time_s") not in ("", None)]
    tool_by_type: dict[str, list[float]] = defaultdict(list)
    tool_by_backend: dict[str, list[float]] = defaultdict(list)
    findings_shares = [
        int(record["latest_findings_tokens"]) / int(record["prompt_tokens"])
        for record in prompt_records
        if int(record["prompt_tokens"]) > 0
    ]
    for row in completed_actions:
        value = row.get("execution_time_s")
        if value in ("", None):
            continue
        tool_by_type[str(row["action_type"])].append(float(value))
        tool_by_backend[str(row["backend_tool"])].append(float(value))

    prompt_counter = Counter(record["log_name"] for record in prompt_records)
    llm_call_counter = Counter(record["log_name"] for record in llm_call_records)
    reward_counter = Counter(f"{float(record['reward']):.2f}" for record in reward_records)
    prompt_stats_by_log_name = aggregate_group_stats(prompt_records, "log_name", "prompt_tokens")
    llm_output_tokens_by_log_name = aggregate_group_stats(llm_call_records, "log_name", "output_tokens")
    llm_token_usage_by_log_name = aggregate_llm_token_usage(llm_call_records, "log_name")
    llm_token_usage_by_rollout = aggregate_llm_token_usage(llm_call_records_for_rollout_analysis, "rollout_id")
    action_stats_by_type = aggregate_group_stats(completed_actions, "action_type", "execution_time_s")
    action_stats_by_backend = aggregate_group_stats(completed_actions, "backend_tool", "execution_time_s")
    rollout_duration_by_iteration = aggregate_group_stats(completed_rollouts, "current_iteration", "duration_s")

    llm_tokens_by_rollout_rows = []
    for rollout_id, stats in llm_token_usage_by_rollout.items():
        rollout = rollout_lookup.get(rollout_id, {})
        llm_tokens_by_rollout_rows.append(
            {
                "rollout_id": rollout_id,
                "has_report": bool(rollout.get("has_report")),
                "current_iteration": rollout.get("current_iteration", ""),
                "duration_s": rollout.get("duration_s", ""),
                **stats,
            }
        )
    llm_tokens_by_log_name_rows = [
        {"log_name": log_name, **stats}
        for log_name, stats in llm_token_usage_by_log_name.items()
    ]

    write_csv(output_dir / "llm_tokens_by_rollout.csv", llm_tokens_by_rollout_rows)
    write_csv(output_dir / "llm_tokens_by_log_name.csv", llm_tokens_by_log_name_rows)

    summary = {
        "run_dir": str(run_dir),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "completed_rollouts_per_s": rollout_rate.get("completed_rollouts_per_s"),
        "seconds_per_completed_rollout": rollout_rate.get("seconds_per_completed_rollout"),
        "counts": {
            "llm_requests": len(llm_requests),
            "llm_call_records": len(llm_call_records),
            "prompt_records": len(prompt_records),
            "reward_records": len(reward_records),
            "rollouts_all": len(rollouts),
            "rollouts_completed_with_report": len(completed_rollouts),
            "actions_completed_with_report": len(completed_actions),
            "iteration_records_completed_with_report": len(completed_iterations),
        },
        "llm_request_latency_s": numeric_stats(llm_durations),
        "logged_prompt_tokens": numeric_stats(prompt_tokens),
        "logged_response_tokens": numeric_stats(response_tokens),
        "logged_total_tokens": numeric_stats(total_tokens),
        "llm_calls_per_completed_rollout": numeric_stats(
            [float(stats["call_count"]) for stats in llm_token_usage_by_rollout.values()]
        ),
        "rollout_reward": numeric_stats(reward_values),
        "completed_rollout_duration_s": numeric_stats(completed_rollout_durations),
        "completed_action_execution_s": numeric_stats(tool_exec_values),
        "median_latest_findings_share": median(findings_shares) if findings_shares else 0.0,
        "prompt_count_by_log_name": dict(prompt_counter.most_common()),
        "llm_call_count_by_log_name": dict(llm_call_counter.most_common()),
        "reward_distribution": dict(sorted(reward_counter.items(), key=lambda item: float(item[0]))),
        "iteration_count_distribution": dict(sorted(iteration_counts.items())),
        "prompt_tokens_by_log_name": prompt_stats_by_log_name,
        "output_tokens_by_log_name": llm_output_tokens_by_log_name,
        "llm_token_usage_by_log_name": llm_token_usage_by_log_name,
        "llm_token_usage_by_rollout": llm_token_usage_by_rollout,
        "action_execution_s_by_type": action_stats_by_type,
        "action_execution_s_by_backend": action_stats_by_backend,
        "rollout_duration_s_by_final_iteration": rollout_duration_by_iteration,
        "limitations": [
            "LLM duration logs do not include request IDs, so exact latency attribution to log_name is not possible under concurrency.",
            "Prompt summaries cover large prompts (>= 4096 tokens) or overflow cases, not every single LLM call.",
            "Prompt/response token by rollout and by call type requires the newer `privacy_agent llm call summary` log lines; older runs will show empty sections.",
            "Rollout timing analysis is filtered to runs with a saved report to avoid in-progress session artifacts.",
            "The raw CSV links in this bundle are capped samples rather than full dumps, to keep the exported viewer lightweight and git-friendly.",
        ],
    }
    save_json(output_dir / "summary.json", summary)
    write_summary_markdown(
        output_dir / "README.md",
        run_dir=run_dir,
        llm_requests=llm_requests,
        prompt_records=prompt_records,
        llm_call_records=llm_call_records,
        reward_records=reward_records,
        rollouts=rollouts,
        actions=completed_actions,
        completed_rollouts=completed_rollouts,
    )

    plot_hist_and_ecdf(
        llm_durations,
        output_dir / "llm_latency_distribution.png",
        title="LLM Request Latency Distribution",
        xlabel="Latency (seconds)",
    )
    plot_prompt_token_breakdown(prompt_records, output_dir / "prompt_tokens_by_log_name.png")
    plot_hist_and_ecdf(
        response_tokens,
        output_dir / "llm_response_token_distribution.png",
        title="LLM Response Token Distribution",
        xlabel="Response tokens",
    )
    plot_bar(
        reward_counter,
        output_dir / "reward_distribution.png",
        title="Completed Rollout Reward Distribution",
        xlabel="Reward",
        ylabel="Rollouts",
    )
    plot_hist_and_ecdf(
        completed_rollout_durations,
        output_dir / "rollout_duration_distribution.png",
        title="Completed Rollout Duration Distribution",
        xlabel="Duration (seconds)",
    )
    plot_hist_and_ecdf(
        actions_per_rollout,
        output_dir / "actions_per_rollout_distribution.png",
        title="Actions per Completed Rollout",
        xlabel="Action count",
        bins=24,
    )
    plot_bar(
        Counter({str(key): value for key, value in iteration_counts.items()}),
        output_dir / "iteration_count_distribution.png",
        title="Completed Rollout Final Iteration Distribution",
        xlabel="Final iteration count",
        ylabel="Rollouts",
    )
    plot_rollout_duration_by_iterations(completed_rollouts, output_dir / "rollout_duration_by_iteration.png")
    plot_iteration_breakdown(completed_iterations, output_dir / "iteration_time_breakdown.png")
    plot_grouped_boxplot(
        defaultdict(list, {
            str(row["log_name"]): [int(item["output_tokens"]) for item in llm_call_records if item["log_name"] == row["log_name"]]
            for row in [{"log_name": name} for name, _ in llm_call_counter.most_common(10)]
        }),
        output_dir / "llm_output_tokens_by_log_name.png",
        title="LLM Response Tokens by Call Type",
        xlabel="Response tokens",
        ylabel="LLM call type",
        top_n=10,
        use_log_scale=False,
    )
    plot_stacked_token_totals(
        llm_token_usage_by_log_name,
        output_dir / "llm_token_totals_by_log_name.png",
        title="Aggregate LLM Token Volume by Call Type",
        xlabel="Token count",
        ylabel="LLM call type",
        top_n=12,
    )
    plot_stacked_token_totals(
        llm_token_usage_by_rollout,
        output_dir / "llm_token_totals_by_rollout.png",
        title="Aggregate LLM Token Volume by Rollout",
        xlabel="Token count",
        ylabel="Rollout",
        top_n=15,
    )
    plot_grouped_boxplot(
        tool_by_type,
        output_dir / "tool_latency_by_action_type.png",
        title="Tool Execution Latency by Action Type",
        xlabel="Execution time (seconds)",
        ylabel="Action type",
        top_n=10,
        use_log_scale=False,
    )
    plot_grouped_boxplot(
        tool_by_backend,
        output_dir / "tool_latency_by_backend.png",
        title="Tool Execution Latency by Backend Tool",
        xlabel="Execution time (seconds)",
        ylabel="Backend tool",
        top_n=10,
        use_log_scale=False,
    )
    plot_bar(
        Counter(row["action_type"] for row in completed_actions),
        output_dir / "tool_counts_by_action_type.png",
        title="Completed Tool Calls by Action Type",
        xlabel="Action type",
        ylabel="Completed tool calls",
    )
    plot_bar(
        Counter(row["backend_tool"] for row in completed_actions),
        output_dir / "tool_counts_by_backend.png",
        title="Completed Tool Calls by Backend Tool",
        xlabel="Backend tool",
        ylabel="Completed tool calls",
    )
    write_html_viewer(
        output_dir / "index.html",
        run_dir=run_dir,
        summary=summary,
        llm_requests=llm_requests,
        prompt_records=prompt_records,
        llm_call_records=llm_call_records,
        reward_records=reward_records,
        sampled_reward_records=sampled_reward_records,
        legacy_timeline_links=legacy_timeline_links,
        rollouts=rollouts,
        actions=actions,
        iterations=iterations,
    )

    print(f"Wrote runtime analysis to {output_dir}")


if __name__ == "__main__":
    main()
