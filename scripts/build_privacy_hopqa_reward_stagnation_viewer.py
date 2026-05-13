#!/usr/bin/env python3
"""Build an offline viewer for Privacy HopQA reward stagnation debugging.

The viewer is intentionally self-contained. It embeds a bounded sample of trace
summaries plus aggregate log-derived diagnostics, so it can be copied locally
without needing access to the original run directories.
"""

from __future__ import annotations

import html
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MOSAIC_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = MOSAIC_ROOT / "runtime_analysis" / "privacy_hopqa_q30_reward_stagnation_2026-05-13"
OUT_HTML = OUT_DIR / "index.html"

RUNS = {
    "4B": Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_full_planning_docread10_qwen3_4b_instruct_2507_20260513033026"),
    "8B": Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_full_planning_docread10_qwen3_8b_20260513033026"),
}

MAX_TRACES_PER_MODEL = int(os.environ.get("PRIVACY_HOPQA_VIEWER_MAX_TRACES", "1800"))
MAX_INVALID_EXAMPLES = 60
MAX_ANSWER_EXAMPLES = 120
MAX_GROUPS = 16
MAX_GROUP_ROLLOUTS = 8
MAX_LOG_TAIL_BYTES = int(os.environ.get("PRIVACY_HOPQA_VIEWER_LOG_TAIL_BYTES", "75000000"))
RANDOM_SEED = 1337


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def truncate(value: Any, limit: int = 240) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def parse_run_dir_name(name: str) -> tuple[str, str, str]:
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], "_".join(parts[2:])
    return "", "", name


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
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def accepted_answers_for_hop(hop: dict[str, Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for key in ("canonical_answer", "accepted_answer_variants", "alternate_valid_answers"):
        raw = hop.get(key)
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if item is None:
                continue
            text = str(item).strip()
            norm = normalize_answer(text)
            if text and norm not in seen:
                seen.add(norm)
                values.append(text)
    return values


def best_answer_score(predicted: str | None, accepted: list[str]) -> tuple[float, str]:
    if not accepted:
        return (0.0, "")
    scores = [(token_f1_score(predicted, truth), truth) for truth in accepted]
    return max(scores, key=lambda item: item[0])


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def parse_config(path: Path) -> dict[str, str]:
    config_path = path / "conf" / "exp_config.yaml"
    wanted = {
        "attempts",
        "train_batch_size",
        "llm_max_rollouts",
        "capture_mode",
        "answer_match_f1_threshold",
        "answer_match_mode",
        "reward_mode",
        "hop_plan_attempts",
        "max_parallel_doc_reads",
        "filter_zero_advantage_groups",
        "wandb_project_name",
        "wandb_name",
        "final_dataset_path",
        "final_train_dataset_path",
        "final_val_dataset_path",
        "final_test_dataset_path",
    }
    out: dict[str, str] = {}
    if not config_path.exists():
        return out
    for line in config_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key in wanted and key not in out:
            out[key] = value.strip().strip("'\"")
    return out


def iter_trace_paths(run_root: Path) -> list[Path]:
    runs_dir = run_root / "privacy_hopqa_runs"
    if not runs_dir.exists():
        return []
    paths: list[Path] = []
    with os.scandir(runs_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            trace = Path(entry.path) / "trace.json"
            if trace.is_file():
                paths.append(trace)
    return paths


def reservoir_sample(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    rng = random.Random(seed)
    sample: list[Path] = []
    for idx, path in enumerate(paths):
        if idx < limit:
            sample.append(path)
        else:
            repl = rng.randint(0, idx)
            if repl < limit:
                sample[repl] = path
    sample.sort(key=lambda p: p.parent.name)
    return sample


def simplify_search_history(search_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simplified = []
    for event in search_history[:10]:
        simplified.append(
            {
                "iteration": event.get("iteration"),
                "type": event.get("type"),
                "query": event.get("query", ""),
                "result_count": event.get("result_count"),
                "top_parent_doc_ids": list(event.get("top_parent_doc_ids") or [])[:5],
                "top_doc_ids": list(event.get("top_doc_ids") or [])[:5],
            }
        )
    return simplified


def simplify_reader_results(reader_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simplified = []
    for result in reader_results[:12]:
        simplified.append(
            {
                "doc_id": result.get("doc_id", ""),
                "parent_doc_id": result.get("parent_doc_id", ""),
                "source": result.get("source", ""),
                "title": truncate(result.get("title", ""), 140),
                "window": (
                    f"{result.get('window_index')}/{result.get('window_count')}"
                    if result.get("window_index") is not None
                    else ""
                ),
                "can_answer": bool(result.get("can_answer")),
                "proposed_answer": result.get("proposed_answer", ""),
                "confidence": result.get("confidence"),
                "justification": truncate(result.get("justification", ""), 360),
                "missing_information": truncate(result.get("missing_information", ""), 220),
            }
        )
    return simplified


def load_trace_summary(model: str, trace_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(trace_path.read_text(errors="replace"))
    except Exception as exc:
        return {
            "model": model,
            "run_id": trace_path.parent.name,
            "trace_path": str(trace_path),
            "load_error": str(exc),
        }

    task_id, chain_from_dir, rollout_id = parse_run_dir_name(trace_path.parent.name)
    summary = data.get("summary") or {}
    chain_id = str(summary.get("chain_id") or chain_from_dir)
    hop_states = list(data.get("hop_states") or [])
    error_records = list(data.get("error_records") or [])
    hops: list[dict[str, Any]] = []
    correct_hops = 0
    answered_hops = 0
    mismatch_hops = 0
    no_answer_hops = 0
    reader_can_answer_wrong_final = 0

    errors_by_hop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for err in error_records:
        try:
            hop_num = int(err.get("hop_number") or 0)
        except Exception:
            hop_num = 0
        if hop_num:
            errors_by_hop[hop_num].append(err)

    for hop in hop_states:
        hop_number = int(hop.get("hop_number") or 0)
        accepted = accepted_answers_for_hop(hop)
        answer = hop.get("answer", "")
        matched = bool(hop.get("matched_accepted_variant"))
        best_score, best_truth = best_answer_score(answer, accepted)
        if matched:
            correct_hops += 1
        if str(answer).strip() and str(answer).strip().upper() != "NOT_FOUND":
            answered_hops += 1
            if not matched:
                mismatch_hops += 1
        else:
            no_answer_hops += 1

        reader_results = list(hop.get("reader_results") or [])
        any_reader_can_answer = any(bool(r.get("can_answer")) for r in reader_results)
        if any_reader_can_answer and not matched:
            reader_can_answer_wrong_final += 1

        hops.append(
            {
                "hop_number": hop_number,
                "hop_type": hop.get("hop_type", ""),
                "question": hop.get("question", ""),
                "canonical_answer": hop.get("canonical_answer", ""),
                "answer": answer,
                "answer_for_context": hop.get("answer_for_context", ""),
                "matched": matched,
                "status": hop.get("status", ""),
                "resolution_reason": hop.get("resolution_reason", ""),
                "confidence": hop.get("confidence"),
                "best_f1": best_score,
                "best_f1_answer": best_truth,
                "accepted_answers": accepted[:40],
                "accepted_answer_count": len(accepted),
                "gold_doc_id": hop.get("gold_doc_id", ""),
                "gold_doc_ref": hop.get("gold_doc_ref", ""),
                "gold_parent_retrieved": bool(hop.get("gold_parent_retrieved")),
                "gold_parent_read": bool(hop.get("gold_parent_read")),
                "attempts": hop.get("attempts"),
                "selected_doc_ids": list(hop.get("selected_doc_ids") or [])[:20],
                "candidate_doc_count": len(hop.get("candidate_doc_ids") or []),
                "search_history": simplify_search_history(list(hop.get("search_history") or [])),
                "reader_results": simplify_reader_results(reader_results),
                "reader_can_answer_count": sum(1 for r in reader_results if r.get("can_answer")),
                "reader_proposed_answers": [
                    r.get("proposed_answer", "")
                    for r in reader_results
                    if r.get("can_answer") and str(r.get("proposed_answer", "")).strip()
                ][:8],
                "parse_error_stages": [
                    str(err.get("stage", "unknown")) for err in errors_by_hop.get(hop_number, []) if err.get("kind") == "parse_error"
                ],
            }
        )

    total_hops = len(hops)
    binary_reward = 1.0 if total_hops and correct_hops == total_hops else 0.0
    fractional_reward = correct_hops / total_hops if total_hops else 0.0
    parse_errors = [err for err in error_records if err.get("kind") == "parse_error"]
    parse_errors_by_stage = Counter(str(err.get("stage", "unknown")) for err in parse_errors)

    return {
        "model": model,
        "run_id": trace_path.parent.name,
        "task_id": str(summary.get("task_id") or task_id),
        "chain_id": chain_id,
        "rollout_id": rollout_id,
        "trace_path": str(trace_path),
        "timeline_path": str(trace_path.parent / "timeline.html"),
        "report_path": str(trace_path.parent / "workspace" / "research_report_1778661860.md"),
        "duration_s": summary.get("duration_s"),
        "llm_calls_total": summary.get("llm_calls_total"),
        "total_hops": total_hops,
        "correct_hops": correct_hops,
        "answered_hops": answered_hops,
        "mismatch_hops": mismatch_hops,
        "no_answer_hops": no_answer_hops,
        "reader_can_answer_wrong_final": reader_can_answer_wrong_final,
        "binary_reward": binary_reward,
        "fractional_reward": fractional_reward,
        "parse_errors_total": len(parse_errors),
        "parse_errors_by_stage": dict(parse_errors_by_stage),
        "errors_by_stage": summary.get("errors_by_stage") or {},
        "errors_by_kind": summary.get("errors_by_kind") or {},
        "retrieval_summary": summary.get("retrieval_summary") or {},
        "hops": hops,
    }


def parse_warning_log(run_root: Path) -> dict[str, Any]:
    path = run_root / "actor" / "warning.log"
    out = {
        "path": str(path),
        "exists": path.exists(),
        "total_invalid_json_warnings": 0,
        "stage_counts": {},
        "examples": [],
    }
    if not path.exists():
        return out
    stage_counts: Counter[str] = Counter()
    examples: list[str] = []
    stage_patterns = [
        ("hop_plan", re.compile(r"hop planner failed")),
        ("doc_choose", re.compile(r"doc chooser failed")),
        ("doc_read", re.compile(r"doc read failed")),
        ("hop_resolve", re.compile(r"hop resolver failed")),
    ]
    with path.open(errors="replace") as handle:
        for line in handle:
            if "No valid JSON found" not in line:
                continue
            out["total_invalid_json_warnings"] += 1
            stage = "unknown"
            for candidate, pattern in stage_patterns:
                if pattern.search(line):
                    stage = candidate
                    break
            stage_counts[stage] += 1
            if len(examples) < 25:
                examples.append(line.strip())
    out["stage_counts"] = dict(stage_counts)
    out["examples"] = examples
    return out


def read_tail_text(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def parse_actor_info_tail(run_root: Path) -> dict[str, Any]:
    path = run_root / "actor" / "info.log"
    text = read_tail_text(path, MAX_LOG_TAIL_BYTES)
    call_marker = "privacy_hopqa llm call summary:"
    prompt_marker = "privacy_hopqa prompt summary:"
    calls_by_log: dict[str, dict[str, list[float] | int]] = defaultdict(lambda: {"prompt_tokens": [], "output_tokens": [], "total_tokens": [], "captured_count": 0, "count": 0})
    prompts_by_log: dict[str, list[float]] = defaultdict(list)
    total_call_summaries = 0
    total_prompt_summaries = 0

    for line in text.splitlines():
        if call_marker in line:
            payload = line.split(call_marker, 1)[1].strip()
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            log_name = str(obj.get("log_name") or "unknown")
            bucket = calls_by_log[log_name]
            bucket["count"] = int(bucket["count"]) + 1
            total_call_summaries += 1
            if obj.get("captured"):
                bucket["captured_count"] = int(bucket["captured_count"]) + 1
            for key in ("prompt_tokens", "output_tokens", "total_tokens"):
                value = obj.get(key)
                if isinstance(value, (int, float)):
                    bucket[key].append(float(value))  # type: ignore[index]
        elif prompt_marker in line:
            payload = line.split(prompt_marker, 1)[1].strip()
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            log_name = str(obj.get("log_name") or "unknown")
            value = obj.get("prompt_tokens")
            if isinstance(value, (int, float)):
                prompts_by_log[log_name].append(float(value))
                total_prompt_summaries += 1

    call_rows = []
    for log_name, raw in sorted(calls_by_log.items()):
        output_tokens = raw["output_tokens"]  # type: ignore[assignment]
        prompt_tokens = raw["prompt_tokens"]  # type: ignore[assignment]
        total_tokens = raw["total_tokens"]  # type: ignore[assignment]
        output_max = max(output_tokens) if output_tokens else None  # type: ignore[arg-type]
        near_observed_max = (
            sum(1 for value in output_tokens if output_max is not None and value >= 0.98 * output_max)
            if output_tokens
            else 0
        )
        call_rows.append(
            {
                "log_name": log_name,
                "count": int(raw["count"]),
                "captured_count": int(raw["captured_count"]),
                "output_mean": mean(output_tokens),  # type: ignore[arg-type]
                "output_p50": quantile(output_tokens, 0.5),  # type: ignore[arg-type]
                "output_p95": quantile(output_tokens, 0.95),  # type: ignore[arg-type]
                "output_max": output_max,
                "output_near_observed_max": near_observed_max,
                "output_le_5": sum(1 for value in output_tokens if value <= 5),  # type: ignore[union-attr]
                "output_near_4k": sum(1 for value in output_tokens if value >= 3900),  # type: ignore[union-attr]
                "prompt_mean": mean(prompt_tokens),  # type: ignore[arg-type]
                "prompt_p95": quantile(prompt_tokens, 0.95),  # type: ignore[arg-type]
                "prompt_max": max(prompt_tokens) if prompt_tokens else None,  # type: ignore[arg-type]
                "total_p95": quantile(total_tokens, 0.95),  # type: ignore[arg-type]
            }
        )
    prompt_rows = [
        {
            "log_name": log_name,
            "count": len(values),
            "prompt_mean": mean(values),
            "prompt_p95": quantile(values, 0.95),
            "prompt_max": max(values) if values else None,
        }
        for log_name, values in sorted(prompts_by_log.items())
    ]
    return {
        "path": str(path),
        "tail_bytes": min(path.stat().st_size if path.exists() else 0, MAX_LOG_TAIL_BYTES),
        "total_call_summaries": total_call_summaries,
        "total_prompt_summaries": total_prompt_summaries,
        "call_rows": call_rows,
        "prompt_rows": prompt_rows,
    }


@dataclass
class ModelAnalysis:
    model: str
    root: Path
    config: dict[str, str]
    all_trace_count: int
    sampled_trace_count: int
    traces: list[dict[str, Any]]
    warning_log: dict[str, Any]
    actor_info_tail: dict[str, Any]


def analyze_model(model: str, root: Path) -> ModelAnalysis:
    trace_paths = iter_trace_paths(root)
    stable_model_seed = RANDOM_SEED + sum(ord(ch) for ch in model)
    sample = reservoir_sample(trace_paths, MAX_TRACES_PER_MODEL, stable_model_seed)
    traces = []
    for path in sample:
        loaded = load_trace_summary(model, path)
        if loaded:
            traces.append(loaded)
    return ModelAnalysis(
        model=model,
        root=root,
        config=parse_config(root),
        all_trace_count=len(trace_paths),
        sampled_trace_count=len(traces),
        traces=traces,
        warning_log=parse_warning_log(root),
        actor_info_tail=parse_actor_info_tail(root),
    )


def aggregate_model(analysis: ModelAnalysis) -> dict[str, Any]:
    traces = [t for t in analysis.traces if not t.get("load_error")]
    binary = [float(t["binary_reward"]) for t in traces]
    fractional = [float(t["fractional_reward"]) for t in traces]
    parse_errors = [float(t["parse_errors_total"]) for t in traces]
    total_hops = sum(int(t["total_hops"]) for t in traces)
    correct_hops = sum(int(t["correct_hops"]) for t in traces)
    answer_mismatch_hops = sum(int(t["mismatch_hops"]) for t in traces)
    reader_can_answer_wrong_final = sum(int(t["reader_can_answer_wrong_final"]) for t in traces)
    stage_counts: Counter[str] = Counter()
    for trace in traces:
        stage_counts.update(trace.get("parse_errors_by_stage") or {})
    groups = group_rollouts(traces)
    group_stats = summarize_groups(groups)
    return {
        "model": analysis.model,
        "root": str(analysis.root),
        "all_trace_count": analysis.all_trace_count,
        "sampled_trace_count": analysis.sampled_trace_count,
        "config": analysis.config,
        "mean_binary_reward": mean(binary),
        "mean_fractional_reward": mean(fractional),
        "sample_hop_accuracy": (correct_hops / total_hops) if total_hops else None,
        "total_hops": total_hops,
        "correct_hops": correct_hops,
        "answer_mismatch_hops": answer_mismatch_hops,
        "reader_can_answer_wrong_final": reader_can_answer_wrong_final,
        "mean_parse_errors": mean(parse_errors),
        "parse_errors_total": int(sum(parse_errors)),
        "parse_error_stage_counts_sample": dict(stage_counts),
        "warning_log": analysis.warning_log,
        "actor_info_tail": analysis.actor_info_tail,
        "group_stats": group_stats,
    }


def group_rollouts(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        key = f"{trace.get('task_id')}::{trace.get('chain_id')}"
        groups[key].append(trace)
    return groups


def summarize_groups(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    total = 0
    mixed_binary = 0
    partial_with_full = 0
    all_binary_zero_fractional_var = 0
    fractional_variance = 0
    for traces in groups.values():
        if len(traces) < 2:
            continue
        total += 1
        binary_values = {float(t["binary_reward"]) for t in traces}
        frac_values = [float(t["fractional_reward"]) for t in traces]
        if len(binary_values) > 1:
            mixed_binary += 1
        if any(t["binary_reward"] == 1.0 for t in traces) and any(0.0 < float(t["fractional_reward"]) < 1.0 for t in traces):
            partial_with_full += 1
        if all(t["binary_reward"] == 0.0 for t in traces) and max(frac_values) > min(frac_values):
            all_binary_zero_fractional_var += 1
        if max(frac_values) > min(frac_values):
            fractional_variance += 1
    return {
        "groups_with_2plus_rollouts": total,
        "mixed_binary_reward_groups": mixed_binary,
        "partial_with_full_success_groups": partial_with_full,
        "all_binary_zero_but_fractional_varies_groups": all_binary_zero_fractional_var,
        "fractional_varies_groups": fractional_variance,
    }


def collect_invalid_examples(analyses: list[ModelAnalysis]) -> list[dict[str, Any]]:
    examples = []
    for analysis in analyses:
        count = 0
        for trace in analysis.traces:
            if trace.get("load_error"):
                continue
            hops_by_num = {hop["hop_number"]: hop for hop in trace.get("hops", [])}
            raw = json.loads(Path(trace["trace_path"]).read_text(errors="replace"))
            for err in raw.get("error_records") or []:
                if err.get("kind") != "parse_error":
                    continue
                hop = hops_by_num.get(int(err.get("hop_number") or 0), {})
                examples.append(
                    {
                        "model": analysis.model,
                        "run_id": trace["run_id"],
                        "task_id": trace["task_id"],
                        "chain_id": trace["chain_id"],
                        "hop_number": err.get("hop_number"),
                        "iteration": err.get("iteration"),
                        "stage": err.get("stage"),
                        "message": err.get("message"),
                        "elapsed_s": err.get("elapsed_s"),
                        "hop_correct": hop.get("matched"),
                        "question": hop.get("question", ""),
                        "answer": hop.get("answer", ""),
                        "canonical_answer": hop.get("canonical_answer", ""),
                        "trace_path": trace["trace_path"],
                    }
                )
                count += 1
                if count >= MAX_INVALID_EXAMPLES:
                    break
            if count >= MAX_INVALID_EXAMPLES:
                break
    return examples


def collect_answer_noise_examples(analyses: list[ModelAnalysis]) -> list[dict[str, Any]]:
    examples = []
    for analysis in analyses:
        count = 0
        for trace in analysis.traces:
            if trace.get("load_error"):
                continue
            for hop in trace.get("hops", []):
                answer = str(hop.get("answer") or "").strip()
                if not answer or answer.upper() == "NOT_FOUND" or hop.get("matched"):
                    continue
                examples.append(
                    {
                        "model": analysis.model,
                        "run_id": trace["run_id"],
                        "task_id": trace["task_id"],
                        "chain_id": trace["chain_id"],
                        "hop_number": hop.get("hop_number"),
                        "question": hop.get("question", ""),
                        "answer": answer,
                        "canonical_answer": hop.get("canonical_answer", ""),
                        "best_f1": hop.get("best_f1"),
                        "best_f1_answer": hop.get("best_f1_answer"),
                        "accepted_answer_count": hop.get("accepted_answer_count"),
                        "accepted_answers": hop.get("accepted_answers", [])[:24],
                        "reader_proposed_answers": hop.get("reader_proposed_answers", []),
                        "reader_can_answer_count": hop.get("reader_can_answer_count"),
                        "status": hop.get("status"),
                        "resolution_reason": hop.get("resolution_reason"),
                        "trace_path": trace["trace_path"],
                    }
                )
                count += 1
                if count >= MAX_ANSWER_EXAMPLES:
                    break
            if count >= MAX_ANSWER_EXAMPLES:
                break
    examples.sort(key=lambda x: (x["model"], -(x.get("best_f1") or 0), x["task_id"], x["chain_id"]))
    return examples


def collect_group_examples(analyses: list[ModelAnalysis]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for analysis in analyses:
        traces = [t for t in analysis.traces if not t.get("load_error")]
        groups = group_rollouts(traces)
        scored = []
        for key, group in groups.items():
            if len(group) < 2:
                continue
            fracs = [float(t["fractional_reward"]) for t in group]
            bins = [float(t["binary_reward"]) for t in group]
            has_full = any(v == 1.0 for v in bins)
            has_partial = any(0.0 < v < 1.0 for v in fracs)
            all_zero_binary_frac_var = all(v == 0.0 for v in bins) and max(fracs) > min(fracs)
            score = (
                (100 if has_full and has_partial else 0)
                + (80 if all_zero_binary_frac_var else 0)
                + int((max(fracs) - min(fracs)) * 50)
                + min(len(group), 8)
            )
            if score <= 0:
                continue
            scored.append((score, key, group))
        scored.sort(key=lambda item: item[0], reverse=True)
        for _, key, group in scored[:MAX_GROUPS]:
            compact_rollouts = []
            for trace in sorted(group, key=lambda t: (-float(t["binary_reward"]), -float(t["fractional_reward"]), t["run_id"]))[:MAX_GROUP_ROLLOUTS]:
                compact_rollouts.append(
                    {
                        "run_id": trace["run_id"],
                        "task_id": trace["task_id"],
                        "chain_id": trace["chain_id"],
                        "trace_path": trace["trace_path"],
                        "timeline_path": trace["timeline_path"],
                        "binary_reward": trace["binary_reward"],
                        "fractional_reward": trace["fractional_reward"],
                        "correct_hops": trace["correct_hops"],
                        "total_hops": trace["total_hops"],
                        "parse_errors_total": trace["parse_errors_total"],
                        "duration_s": trace.get("duration_s"),
                        "hops": trace.get("hops", []),
                    }
                )
            selected.append(
                {
                    "model": analysis.model,
                    "group_key": key,
                    "rollout_count_in_sample": len(group),
                    "fractional_min": min(fracs),
                    "fractional_max": max(fracs),
                    "binary_values": sorted(set(bins)),
                    "rollouts": compact_rollouts,
                }
            )
    selected.sort(key=lambda g: (g["model"], -(g["fractional_max"] - g["fractional_min"]), g["group_key"]))
    return selected[: MAX_GROUPS * len(analyses)]


def table(rows: list[list[str]], headers: list[str]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def bar(value: float | None, max_value: float | None = None, label: str | None = None) -> str:
    if value is None:
        pct = 0
        label = label or "n/a"
    else:
        denom = max_value if max_value not in (None, 0) else 1.0
        pct = max(0, min(100, 100 * value / denom))
        label = label or fmt_num(value)
    return f"<div class='bar'><span style='width:{pct:.1f}%'></span><b>{esc(label)}</b></div>"


def render_summary(aggregates: list[dict[str, Any]]) -> str:
    rows = []
    for agg in aggregates:
        rows.append(
            [
                f"<b>{esc(agg['model'])}</b>",
                esc(agg["all_trace_count"]),
                esc(agg["sampled_trace_count"]),
                bar(agg["mean_binary_reward"], 1.0, fmt_num(agg["mean_binary_reward"])),
                bar(agg["mean_fractional_reward"], 1.0, fmt_num(agg["mean_fractional_reward"])),
                esc(f"{agg['correct_hops']:,}/{agg['total_hops']:,}"),
                esc(agg["answer_mismatch_hops"]),
                esc(agg["reader_can_answer_wrong_final"]),
                esc(agg["parse_errors_total"]),
                esc(fmt_num(agg["mean_parse_errors"])),
            ]
        )
    return table(
        rows,
        [
            "Model",
            "Trace dirs found",
            "Trace sample",
            "Actual binary reward",
            "Counterfactual hop accuracy",
            "Correct hops",
            "Answered but mismatched hops",
            "Reader could answer but final wrong",
            "Parse errors in sample",
            "Parse errors / rollout",
        ],
    )


def render_config(aggregates: list[dict[str, Any]]) -> str:
    fields = [
        "reward_mode",
        "answer_match_mode",
        "answer_match_f1_threshold",
        "capture_mode",
        "attempts",
        "filter_zero_advantage_groups",
        "llm_max_rollouts",
        "max_parallel_doc_reads",
        "hop_plan_attempts",
        "wandb_project_name",
        "final_train_dataset_path",
    ]
    rows = []
    for field in fields:
        rows.append([esc(field), *[esc(agg["config"].get(field, "")) for agg in aggregates]])
    return table(rows, ["Setting", *[agg["model"] for agg in aggregates]])


def render_stage_counts(aggregates: list[dict[str, Any]]) -> str:
    stages = sorted(
        {
            stage
            for agg in aggregates
            for stage in set(agg["parse_error_stage_counts_sample"]) | set(agg["warning_log"].get("stage_counts", {}))
        }
    )
    rows = []
    max_sample = max([agg["parse_errors_total"] for agg in aggregates] + [1])
    max_warn = max([agg["warning_log"].get("total_invalid_json_warnings", 0) for agg in aggregates] + [1])
    for stage in stages:
        row = [esc(stage)]
        for agg in aggregates:
            sample_count = agg["parse_error_stage_counts_sample"].get(stage, 0)
            warn_count = agg["warning_log"].get("stage_counts", {}).get(stage, 0)
            row.append(
                f"<div class='two-bars'>"
                f"<label>sample</label>{bar(sample_count, max_sample, str(sample_count))}"
                f"<label>full warning log</label>{bar(warn_count, max_warn, str(warn_count))}"
                f"</div>"
            )
        rows.append(row)
    return table(rows, ["Stage", *[agg["model"] for agg in aggregates]])


def render_token_tables(aggregates: list[dict[str, Any]]) -> str:
    parts = []
    for agg in aggregates:
        rows = []
        for row in agg["actor_info_tail"].get("call_rows", []):
            rows.append(
                [
                    esc(row["log_name"]),
                    esc(row["count"]),
                    esc(row["captured_count"]),
                    esc(fmt_num(row["output_mean"], 1)),
                    esc(fmt_num(row["output_p95"], 1)),
                    esc(fmt_num(row["output_max"], 0)),
                    esc(row["output_near_observed_max"]),
                    esc(row["output_le_5"]),
                    esc(row["output_near_4k"]),
                    esc(fmt_num(row["prompt_p95"], 1)),
                    esc(fmt_num(row["prompt_max"], 0)),
                ]
            )
        parts.append(
            f"<h3>{esc(agg['model'])}: LLM call token summary from actor/info.log tail</h3>"
            + table(
                rows,
                [
                    "Call type",
                    "Calls parsed",
                    "Captured for training",
                    "Output mean",
                    "Output p95",
                    "Output max",
                    "Output near observed max",
                    "Output <=5 tokens",
                    "Output >=3900 tokens",
                    "Prompt p95",
                    "Prompt max",
                ],
            )
        )
    return "".join(parts)


def render_invalid_examples(examples: list[dict[str, Any]]) -> str:
    rows = []
    for ex in examples[:120]:
        rows.append(
            [
                esc(ex["model"]),
                esc(ex["stage"]),
                esc(ex["hop_number"]),
                esc(ex["iteration"]),
                esc(ex["run_id"]),
                esc(ex["message"]),
                esc("yes" if ex.get("hop_correct") else "no"),
                esc(truncate(ex.get("question"), 140)),
                esc(truncate(ex.get("answer"), 80)),
                f"<code>{esc(ex['trace_path'])}</code>",
            ]
        )
    return table(
        rows,
        ["Model", "Stage", "Hop", "Iter", "Rollout", "Parse error", "Hop correct?", "Question", "Answer", "Trace"],
    )


def render_answer_noise_examples(examples: list[dict[str, Any]]) -> str:
    chunks = []
    for ex in examples[:160]:
        variants = "".join(f"<li>{esc(v)}</li>" for v in ex.get("accepted_answers", [])[:24])
        reader_answers = "".join(f"<li>{esc(v)}</li>" for v in ex.get("reader_proposed_answers", [])[:10]) or "<li class='muted'>none recorded</li>"
        chunks.append(
            f"""
            <details class="example">
              <summary><b>{esc(ex['model'])}</b> {esc(ex['task_id'])}/{esc(ex['chain_id'])} hop {esc(ex['hop_number'])}
                - model answered <code>{esc(ex['answer'])}</code>, gold <code>{esc(ex['canonical_answer'])}</code>,
                best F1 {esc(fmt_num(ex.get('best_f1'), 3))}</summary>
              <div class="grid2">
                <div>
                  <h4>Question</h4>
                  <p>{esc(ex['question'])}</p>
                  <h4>Runtime status</h4>
                  <p>Status: <code>{esc(ex.get('status'))}</code>; reason: <code>{esc(ex.get('resolution_reason'))}</code>; reader-can-answer count: <code>{esc(ex.get('reader_can_answer_count'))}</code></p>
                  <h4>Trace</h4>
                  <p><code>{esc(ex['trace_path'])}</code></p>
                </div>
                <div>
                  <h4>Accepted answers ({esc(ex.get('accepted_answer_count'))})</h4>
                  <ul>{variants}</ul>
                  <h4>Reader proposed answers</h4>
                  <ul>{reader_answers}</ul>
                </div>
              </div>
            </details>
            """
        )
    return "\n".join(chunks)


def render_hop(hop: dict[str, Any]) -> str:
    correct_class = "ok" if hop.get("matched") else "bad"
    search_items = []
    for event in hop.get("search_history", []):
        top_docs = ", ".join(event.get("top_parent_doc_ids") or event.get("top_doc_ids") or [])
        search_items.append(
            f"<li><b>{esc(event.get('type'))}</b> iter {esc(event.get('iteration'))}: {esc(event.get('query'))}"
            f"<br><span class='muted'>top parents: {esc(truncate(top_docs, 260))}</span></li>"
        )
    if not search_items:
        search_items.append("<li class='muted'>no retrieval actions recorded</li>")
    reader_items = []
    for result in hop.get("reader_results", [])[:8]:
        cls = "ok" if result.get("can_answer") else "muted"
        reader_items.append(
            f"<li class='{cls}'><b>{esc(result.get('doc_id'))}</b> {esc(result.get('window'))}"
            f" - can_answer={esc(result.get('can_answer'))}, answer=<code>{esc(result.get('proposed_answer'))}</code>, conf={esc(result.get('confidence'))}"
            f"<br><span>{esc(result.get('justification'))}</span></li>"
        )
    if not reader_items:
        reader_items.append("<li class='muted'>no reader results recorded</li>")
    parse = ", ".join(hop.get("parse_error_stages") or [])
    return f"""
      <div class="hop {correct_class}">
        <h4>Hop {esc(hop.get('hop_number'))}: {'correct' if hop.get('matched') else 'wrong'} - {esc(hop.get('hop_type'))}</h4>
        <p><b>Q:</b> {esc(hop.get('question'))}</p>
        <p><b>Answer:</b> <code>{esc(hop.get('answer'))}</code>; <b>gold:</b> <code>{esc(hop.get('canonical_answer'))}</code>; <b>best F1:</b> {esc(fmt_num(hop.get('best_f1'), 3))}; <b>accepted count:</b> {esc(hop.get('accepted_answer_count'))}</p>
        <p><b>Status:</b> <code>{esc(hop.get('status'))}</code>; <b>reason:</b> <code>{esc(hop.get('resolution_reason'))}</code>; <b>parse stages:</b> <code>{esc(parse or 'none')}</code></p>
        <div class="grid2">
          <div><h5>Planning/search actions for this hop</h5><ul>{''.join(search_items)}</ul></div>
          <div><h5>Reader outputs</h5><ul>{''.join(reader_items)}</ul></div>
        </div>
      </div>
    """


def render_group_examples(groups: list[dict[str, Any]]) -> str:
    parts = []
    for group in groups:
        rollouts = []
        for rollout in group.get("rollouts", []):
            hops = "".join(render_hop(hop) for hop in rollout.get("hops", []))
            rollouts.append(
                f"""
                <details class="rollout">
                  <summary><b>{esc(rollout['run_id'])}</b> binary={esc(rollout['binary_reward'])}
                    hop_accuracy={esc(fmt_num(rollout['fractional_reward'], 3))}
                    correct={esc(rollout['correct_hops'])}/{esc(rollout['total_hops'])}
                    parse_errors={esc(rollout['parse_errors_total'])}
                    duration={esc(fmt_num(rollout.get('duration_s'), 1))}s</summary>
                  <p><b>Trace:</b> <code>{esc(rollout['trace_path'])}</code></p>
                  <p><b>Timeline:</b> <code>{esc(rollout['timeline_path'])}</code></p>
                  {hops}
                </details>
                """
            )
        parts.append(
            f"""
            <details class="group">
              <summary><b>{esc(group['model'])}</b> group {esc(group['group_key'])}
                - {esc(group['rollout_count_in_sample'])} sampled rollouts,
                hop-accuracy range {esc(fmt_num(group['fractional_min'], 3))} to {esc(fmt_num(group['fractional_max'], 3))},
                binary values {esc(group['binary_values'])}</summary>
              {''.join(rollouts)}
            </details>
            """
        )
    return "\n".join(parts)


def render_group_stats(aggregates: list[dict[str, Any]]) -> str:
    rows = []
    for agg in aggregates:
        stats = agg["group_stats"]
        rows.append(
            [
                esc(agg["model"]),
                esc(stats["groups_with_2plus_rollouts"]),
                esc(stats["mixed_binary_reward_groups"]),
                esc(stats["partial_with_full_success_groups"]),
                esc(stats["all_binary_zero_but_fractional_varies_groups"]),
                esc(stats["fractional_varies_groups"]),
            ]
        )
    return table(
        rows,
        [
            "Model",
            "Groups with >=2 sampled rollouts",
            "Groups with mixed binary reward",
            "Groups where partial rollouts compete with full success",
            "All-binary-zero groups rescued by fractional reward",
            "Groups with any fractional variation",
        ],
    )


def render_html(aggregates: list[dict[str, Any]], invalid_examples: list[dict[str, Any]], answer_examples: list[dict[str, Any]], group_examples: list[dict[str, Any]]) -> str:
    css = """
    :root {
      --bg: #f4efe6;
      --ink: #17211c;
      --muted: #5d6a60;
      --paper: #fffaf0;
      --line: #d9cdb8;
      --accent: #0b6b57;
      --accent2: #b45f06;
      --bad: #b42318;
      --ok: #137333;
      --shadow: 0 14px 40px rgba(40, 29, 10, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(11, 107, 87, 0.16), transparent 34rem),
        linear-gradient(135deg, #f8f1e3 0%, #efe3cd 100%);
    }
    header {
      padding: 46px 56px 30px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 250, 240, 0.78);
      backdrop-filter: blur(8px);
    }
    h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: -0.02em; }
    h2 { margin: 34px 0 12px; font-size: 23px; }
    h3 { margin: 24px 0 10px; font-size: 18px; }
    h4 { margin: 14px 0 6px; }
    h5 { margin: 8px 0 4px; font-size: 14px; }
    main { padding: 28px 56px 70px; }
    section {
      background: rgba(255, 250, 240, 0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      margin: 18px 0;
      box-shadow: var(--shadow);
    }
    .note {
      border-left: 5px solid var(--accent);
      padding: 12px 15px;
      background: rgba(11, 107, 87, 0.08);
      border-radius: 10px;
      margin: 14px 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: var(--paper);
      border-radius: 12px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }
    th { background: #eadcc5; font-weight: 700; }
    code {
      font-family: "SFMono-Regular", "Cascadia Code", Consolas, monospace;
      font-size: 12px;
      background: rgba(23, 33, 28, 0.08);
      padding: 2px 4px;
      border-radius: 4px;
      word-break: break-word;
    }
    .bar {
      position: relative;
      height: 22px;
      background: #eadcc5;
      border-radius: 999px;
      overflow: hidden;
      min-width: 120px;
    }
    .bar span {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      background: linear-gradient(90deg, var(--accent), #54a38f);
    }
    .bar b {
      position: relative;
      display: block;
      padding: 3px 8px;
      font-size: 12px;
    }
    .two-bars {
      display: grid;
      gap: 4px;
    }
    .two-bars label {
      color: var(--muted);
      font-size: 12px;
    }
    details {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      margin: 10px 0;
    }
    summary { cursor: pointer; }
    .grid2 {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
    }
    .hop {
      border-left: 6px solid var(--bad);
      background: #fff7ef;
      padding: 10px 12px;
      margin: 10px 0;
      border-radius: 10px;
    }
    .hop.ok { border-left-color: var(--ok); background: #f4fff5; }
    .bad { color: var(--bad); }
    .ok { color: var(--ok); }
    .muted { color: var(--muted); }
    li { margin: 4px 0; }
    .toc a {
      color: var(--accent);
      display: inline-block;
      margin: 0 14px 8px 0;
      font-weight: 700;
    }
    @media (max-width: 900px) {
      header, main { padding-left: 18px; padding-right: 18px; }
      .grid2 { grid-template-columns: 1fr; }
      table { font-size: 12px; }
    }
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy HopQA reward stagnation viewer</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>Privacy HopQA reward stagnation viewer</h1>
    <p>Generated {esc(generated)}. This is a sampled, self-contained diagnostic for the active q30 full-dataset planning-only runs.</p>
    <div class="toc">
      <a href="#summary">Summary</a>
      <a href="#json">Invalid JSON</a>
      <a href="#answers">Answer checking noise</a>
      <a href="#advantages">Advantage/downweighting</a>
    </div>
  </header>
  <main>
    <section id="summary">
      <h2>Run configuration and reward comparison</h2>
      <div class="note">
        Current runs use <code>reward_mode: all_hops_correct</code>. The binary reward is 1 only when every hop is correct.
        The counterfactual hop-accuracy reward shown here is <code>correct_hops / total_hops</code>, which is the obvious alternate test for reducing noise from useful partial rollouts.
      </div>
      {render_summary(aggregates)}
      <h3>Relevant config</h3>
      {render_config(aggregates)}
    </section>

    <section id="json">
      <h2>Hypothesis 1: invalid JSON / format preference / truncation</h2>
      <div class="note">
        Raw invalid model responses are not persisted in the trace artifacts; this viewer can show hop, stage, iteration, question context, and token summaries.
        If we need exact malformed text, we should add parse-failure response capture in the adapter or parser.
      </div>
      <h3>Parse errors by stage</h3>
      {render_stage_counts(aggregates)}
      <h3>Token summaries: truncation check</h3>
      <p class="muted">If invalid JSON were caused by stage-level output truncation, output token counts should pile up at the observed per-call cap. The <code>Output near observed max</code> column is the useful warning sign here; <code>Output &gt;=3900</code> is only a generic long-output check.</p>
      {render_token_tables(aggregates)}
      <h3>Invalid-JSON examples from sampled traces</h3>
      {render_invalid_examples(invalid_examples)}
    </section>

    <section id="answers">
      <h2>Hypothesis 2: answer-checking noise / missing variants</h2>
      <div class="note">
        These are hops where the agent produced a non-empty answer but the runtime scorer did not mark it accepted.
        Rows with high best-token-F1 or reader-proposed answers that look semantically equivalent are the strongest missing-variant candidates.
      </div>
      {render_answer_noise_examples(answer_examples)}
    </section>

    <section id="advantages">
      <h2>Hypothesis 3: useful planning stages downweighted by binary group reward</h2>
      <div class="note">
        With binary reward, a 2/3-hop rollout gets reward 0. If another attempt in the same group gets 3/3, the 2/3 rollout is explicitly lower advantage even though early-hop planning may have been useful.
        If all attempts in a group get binary 0, <code>filter_zero_advantage_groups</code> can remove the group entirely; the counterfactual fractional columns show where hop-accuracy reward would still create signal.
      </div>
      <h3>Group-level counterfactual summary</h3>
      {render_group_stats(aggregates)}
      <h3>Same-chain rollout examples</h3>
      {render_group_examples(group_examples)}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    analyses = [analyze_model(model, root) for model, root in RUNS.items()]
    aggregates = [aggregate_model(analysis) for analysis in analyses]
    invalid_examples = collect_invalid_examples(analyses)
    answer_examples = collect_answer_noise_examples(analyses)
    group_examples = collect_group_examples(analyses)
    html_text = render_html(aggregates, invalid_examples, answer_examples, group_examples)
    OUT_HTML.write_text(html_text)
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "aggregates": aggregates,
                "invalid_example_count": len(invalid_examples),
                "answer_noise_example_count": len(answer_examples),
                "group_example_count": len(group_examples),
                "output_html": str(OUT_HTML),
            },
            indent=2,
        )
    )
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {summary_path}")
    for agg in aggregates:
        print(
            f"{agg['model']}: traces found={agg['all_trace_count']} sampled={agg['sampled_trace_count']} "
            f"binary={fmt_num(agg['mean_binary_reward'])} fractional={fmt_num(agg['mean_fractional_reward'])} "
            f"parse_errors_sample={agg['parse_errors_total']} warnings={agg['warning_log'].get('total_invalid_json_warnings')}"
        )


if __name__ == "__main__":
    main()
