#!/usr/bin/env python3
"""Build a lightweight Privacy HopQA training filtering / advantage viewer.

The expensive actor stream records include token arrays, so this script intentionally
avoids json.loads on actor samples. It scans each top-level sample object and extracts
only the metadata needed to estimate LOO non-zero-advantage coverage.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "runtime_analysis" / "privacy_hopqa_training_filter_viewer_2026-05-14"
DEFAULT_RUNS = [
    (
        "4B prefix-step",
        Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122"),
        True,
    ),
    (
        "4B hop-accuracy",
        Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_reward_hopacc_4b_20260513140609"),
        False,
    ),
    (
        "4B binary",
        Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_reward_binary_4b_20260513140609"),
        False,
    ),
]

PROCESSED_RE = re.compile(
    r"Processed\s+(?P<kept>\d+)\s+samples\s+\(filtered out\s+(?P<filtered>\d+)\).*?total\s+(?P<total>\d+)\s+samples so far(?:.*?model version\s+(?P<version>\d+))?",
)
FILTERED_ONLY_RE = re.compile(
    r"Filtered out\s+(?P<filtered>\d+)\s+samples.*?total\s+(?P<total>\d+)\s+samples so far.*?model version\s+(?P<version>\d+)",
)

PATTERNS = {
    "group_id": re.compile(r'"group_id"\s*:\s*"([^"]+)"'),
    "step_index": re.compile(r'"step_index"\s*:\s*(\d+)'),
    "reward": re.compile(r'"reward"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'),
    "model_version": re.compile(r'"model_version"\s*:\s*(\d+)'),
    "training_reward_mode": re.compile(r'"training_reward_mode"\s*:\s*"([^"]*)"'),
    "chain_id": re.compile(r'"chain_id"\s*:\s*"([^"]*)"'),
    "task_id": re.compile(r'"task_id"\s*:\s*"([^"]*)"'),
    "company": re.compile(r'"company"\s*:\s*"([^"]*)"'),
    "pattern": re.compile(r'"pattern"\s*:\s*"([^"]*)"'),
    "chain_length": re.compile(r'"chain_length"\s*:\s*(\d+)'),
    "problem_id": re.compile(r'"problem_id"\s*:\s*"([^"]*)"'),
    "hop_number": re.compile(r'"hop_number"\s*:\s*(\d+)'),
    "iteration": re.compile(r'"iteration"\s*:\s*(\d+)'),
    "log_name": re.compile(r'"log_name"\s*:\s*"([^"]*)"'),
    "outcome_reward": re.compile(r'"outcome_reward"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'),
    "prefix_correct_hops": re.compile(r'"prefix_correct_hops"\s*:\s*(\d+)'),
    "gold_parent_retrieved": re.compile(r'"gold_parent_retrieved"\s*:\s*(true|false)'),
    "gold_parent_read": re.compile(r'"gold_parent_read"\s*:\s*(true|false)'),
}


def pct(numer: float, denom: float) -> float:
    return 0.0 if denom <= 0 else 100.0 * numer / denom


def esc(value: object) -> str:
    return html.escape(str(value))


def extract(pattern_name: str, text: str, start: int, end: int) -> str | None:
    match = PATTERNS[pattern_name].search(text, start, end)
    return match.group(1) if match else None


def extract_float(pattern_name: str, text: str, start: int, end: int, default: float = 0.0) -> float:
    value = extract(pattern_name, text, start, end)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def extract_int(pattern_name: str, text: str, start: int, end: int, default: int = 0) -> int:
    value = extract(pattern_name, text, start, end)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def extract_bool(pattern_name: str, text: str, start: int, end: int) -> bool | None:
    value = extract(pattern_name, text, start, end)
    if value is None:
        return None
    return value == "true"


def iter_object_ranges(line: str) -> Iterable[tuple[int, int]]:
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                yield start, index + 1
                start = None


@dataclass
class SampleRecord:
    run_label: str
    chain_key: str
    group_key: str
    reward: float
    model_version: int
    task_id: str
    chain_id: str
    company: str
    pattern: str
    chain_length: int
    problem_id: str
    hop_number: int
    iteration: int
    log_name: str
    outcome_reward: float
    prefix_correct_hops: int
    gold_parent_retrieved: bool | None
    gold_parent_read: bool | None


@dataclass
class RunSummary:
    label: str
    path: str
    actor_scanned: bool
    warnings: list[str] = field(default_factory=list)
    filter_events: list[dict] = field(default_factory=list)
    kept_from_log: int = 0
    filtered_from_log: int = 0
    stats_kept: int = 0
    stats_filtered: int = 0
    actor_samples: int = 0
    group_keys: int = 0
    nonzero_samples: int = 0
    nonzero_group_keys: int = 0
    mixed_group_keys: int = 0
    model_versions: dict[str, dict] = field(default_factory=dict)
    chains: list[dict] = field(default_factory=list)
    never_nonzero_chains: list[dict] = field(default_factory=list)
    retrieval_impossible_candidates: list[dict] = field(default_factory=list)
    targeted_trace_probe: list[dict] = field(default_factory=list)

    @property
    def filtered_fraction(self) -> float:
        total = self.kept_from_log + self.filtered_from_log
        if total <= 0:
            total = self.stats_kept + self.stats_filtered
            filtered = self.stats_filtered
        else:
            filtered = self.filtered_from_log
        return pct(filtered, total)

    @property
    def nonzero_fraction(self) -> float:
        return pct(self.nonzero_samples, self.actor_samples)


def parse_preprocess_log(run_dir: Path) -> tuple[list[dict], int, int]:
    log_path = run_dir / "preprocess" / "info.log"
    events: list[dict] = []
    kept_total = 0
    filtered_total = 0
    if not log_path.exists():
        return events, 0, 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            processed = PROCESSED_RE.search(line)
            if processed:
                kept = int(processed.group("kept"))
                filtered = int(processed.group("filtered"))
                version = int(processed.group("version") or 0)
                kept_total += kept
                filtered_total += filtered
                events.append(
                    {
                        "line": line_no,
                        "model_version": version,
                        "kept": kept,
                        "filtered": filtered,
                        "kept_cumulative": kept_total,
                        "filtered_cumulative": filtered_total,
                        "filtered_fraction": pct(filtered, kept + filtered),
                    }
                )
                continue
            filtered_only = FILTERED_ONLY_RE.search(line)
            if filtered_only:
                # These messages are diagnostics for zero-advantage groups; the
                # subsequent "Processed ... (filtered out N)" line carries the
                # accounting total, so counting both would double-count.
                events.append(
                    {
                        "line": line_no,
                        "model_version": int(filtered_only.group("version") or 0),
                        "kept": 0,
                        "filtered": 0,
                        "kept_cumulative": kept_total,
                        "filtered_cumulative": filtered_total,
                        "filtered_fraction": None,
                    }
                )
    return events, kept_total, filtered_total


def parse_preprocessor_stats(run_dir: Path) -> tuple[int, int]:
    stats_path = run_dir / "streams" / "preprocessor_stats" / "0" / "0" / "0.jsonl"
    kept = 0
    filtered = 0
    if not stats_path.exists():
        return kept, filtered
    with stats_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kept = int(row.get("preprocessor/total_published_samples", row.get("preprocessor/published_samples", kept)) or kept)
            filtered = int(row.get("preprocessor/total_filtered_out_samples", row.get("preprocessor/filtered_out_samples", filtered)) or filtered)
    return kept, filtered


def actor_paths(run_dir: Path) -> list[Path]:
    base = run_dir / "streams" / "actor" / "0"
    if not base.exists():
        return []
    paths: list[Path] = []
    for child in sorted(base.iterdir()):
        path = child / "0.jsonl"
        if path.exists():
            paths.append(path)
    return paths


def scan_actor_samples(run_label: str, run_dir: Path, max_lines: int) -> tuple[list[SampleRecord], list[str]]:
    records: list[SampleRecord] = []
    warnings: list[str] = []
    paths = actor_paths(run_dir)
    if not paths:
        return records, ["No actor stream found"]
    lines_seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                if max_lines and lines_seen >= max_lines:
                    warnings.append(f"Actor scan capped at {max_lines} lines")
                    return records, warnings
                lines_seen += 1
                ranges = list(iter_object_ranges(line))
                for start, end in ranges:
                    group_id = extract("group_id", line, start, end)
                    if not group_id:
                        continue
                    reward = extract_float("reward", line, start, end, math.nan)
                    if math.isnan(reward):
                        continue
                    step_index = extract_int("step_index", line, start, end, 0)
                    mode = extract("training_reward_mode", line, start, end) or "outcome"
                    group_key = f"{group_id}::step={step_index}" if mode == "prefix_progress" else group_id
                    chain_id = extract("chain_id", line, start, end) or "unknown_chain"
                    task_id = extract("task_id", line, start, end) or "unknown_task"
                    chain_key = f"{task_id}::{chain_id}"
                    records.append(
                        SampleRecord(
                            run_label=run_label,
                            chain_key=chain_key,
                            group_key=group_key,
                            reward=reward,
                            model_version=extract_int("model_version", line, start, end, 0),
                            task_id=task_id,
                            chain_id=chain_id,
                            company=extract("company", line, start, end) or "unknown",
                            pattern=extract("pattern", line, start, end) or "unknown",
                            chain_length=extract_int("chain_length", line, start, end, 0),
                            problem_id=extract("problem_id", line, start, end) or "",
                            hop_number=extract_int("hop_number", line, start, end, 0),
                            iteration=extract_int("iteration", line, start, end, 0),
                            log_name=extract("log_name", line, start, end) or "",
                            outcome_reward=extract_float("outcome_reward", line, start, end, reward),
                            prefix_correct_hops=extract_int("prefix_correct_hops", line, start, end, 0),
                            gold_parent_retrieved=extract_bool("gold_parent_retrieved", line, start, end),
                            gold_parent_read=extract_bool("gold_parent_read", line, start, end),
                        )
                    )
    return records, warnings


def scan_trace_stats(run_dir: Path, max_traces: int) -> tuple[dict[str, dict], list[str]]:
    if max_traces <= 0:
        return {}, ["Trace enrichment skipped; pass --max-traces N to include retrieval/read rates"]
    trace_dir = run_dir / "privacy_hopqa_runs"
    if not trace_dir.exists():
        return {}, ["No trace directory found"]
    stats: dict[str, dict] = {}
    warnings: list[str] = []
    count = 0
    for path in sorted(trace_dir.glob("*/trace.json")):
        if count >= max_traces:
            warnings.append(f"Trace scan capped at {max_traces} files")
            break
        count += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        task_id = (PATTERNS["task_id"].search(text).group(1) if PATTERNS["task_id"].search(text) else "unknown_task")
        chain_id = (PATTERNS["chain_id"].search(text).group(1) if PATTERNS["chain_id"].search(text) else "unknown_chain")
        chain_key = f"{task_id}::{chain_id}"
        row = stats.setdefault(
            chain_key,
            {
                "trace_rollouts": 0,
                "trace_outcome_rewards": [],
                "trace_gold_retrieved_values": [],
                "trace_gold_read_values": [],
                "trace_docs_read": [],
                "trace_duration_s": [],
                "trace_examples": [],
            },
        )
        total_hops_match = re.search(r'"total_hops"\s*:\s*(\d+)', text)
        total_hops = int(total_hops_match.group(1)) if total_hops_match else max(1, text.count('"hop_number"'))
        correct_hops = len(re.findall(r'"matched_accepted_variant"\s*:\s*true', text))
        outcome = correct_hops / total_hops if total_hops else 0.0
        row["trace_rollouts"] += 1
        row["trace_outcome_rewards"].append(outcome)
        docs_read_match = re.search(r'"docs_read"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        duration_match = re.search(r'"duration_s"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        row["trace_docs_read"].append(float(docs_read_match.group(1)) if docs_read_match else 0.0)
        row["trace_duration_s"].append(float(duration_match.group(1)) if duration_match else 0.0)
        for value in re.findall(r'"gold_parent_retrieved"\s*:\s*(true|false)', text):
            row["trace_gold_retrieved_values"].append(1.0 if value == "true" else 0.0)
        for value in re.findall(r'"gold_parent_read"\s*:\s*(true|false)', text):
            row["trace_gold_read_values"].append(1.0 if value == "true" else 0.0)
        if len(row["trace_examples"]) < 3:
            row["trace_examples"].append(str(path))
    return stats, warnings


def aggregate_records(summary: RunSummary, records: list[SampleRecord], trace_stats: dict[str, dict] | None = None) -> None:
    by_group: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_group[record.group_key].append(record.reward)
    nonzero_by_sample: list[bool] = []
    nonzero_groups: set[str] = set()
    mixed_groups = 0
    for record in records:
        rewards = by_group[record.group_key]
        if len(rewards) <= 1:
            nonzero = False
        else:
            other_mean = (sum(rewards) - record.reward) / (len(rewards) - 1)
            nonzero = abs(record.reward - other_mean) > 1e-9
        nonzero_by_sample.append(nonzero)
        if nonzero:
            nonzero_groups.add(record.group_key)
    for rewards in by_group.values():
        if len(set(round(value, 8) for value in rewards)) > 1:
            mixed_groups += 1

    summary.actor_samples = len(records)
    summary.group_keys = len(by_group)
    summary.nonzero_samples = sum(1 for value in nonzero_by_sample if value)
    summary.nonzero_group_keys = len(nonzero_groups)
    summary.mixed_group_keys = mixed_groups

    version_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"samples": 0, "nonzero": 0, "reward_sum": 0.0})
    chain_stats: dict[str, dict] = {}
    for record, nonzero in zip(records, nonzero_by_sample, strict=False):
        version = version_stats[record.model_version]
        version["samples"] += 1
        version["nonzero"] += 1 if nonzero else 0
        version["reward_sum"] += record.reward
        stats = chain_stats.setdefault(
            record.chain_key,
            {
                "chain_key": record.chain_key,
                "task_id": record.task_id,
                "chain_id": record.chain_id,
                "company": record.company,
                "pattern": record.pattern,
                "chain_length": record.chain_length,
                "problem_id": record.problem_id,
                "samples": 0,
                "nonzero_samples": 0,
                "groups": set(),
                "nonzero_groups": set(),
                "reward_values": [],
                "outcome_rewards": [],
                "prefix_correct_values": [],
                "gold_retrieved_values": [],
                "gold_read_values": [],
                "examples": [],
            },
        )
        stats["samples"] += 1
        stats["nonzero_samples"] += 1 if nonzero else 0
        stats["groups"].add(record.group_key)
        if nonzero:
            stats["nonzero_groups"].add(record.group_key)
        stats["reward_values"].append(record.reward)
        stats["outcome_rewards"].append(record.outcome_reward)
        stats["prefix_correct_values"].append(record.prefix_correct_hops)
        if record.gold_parent_retrieved is not None:
            stats["gold_retrieved_values"].append(1.0 if record.gold_parent_retrieved else 0.0)
        if record.gold_parent_read is not None:
            stats["gold_read_values"].append(1.0 if record.gold_parent_read else 0.0)
        if len(stats["examples"]) < 3:
            stats["examples"].append(
                {
                    "hop": record.hop_number,
                    "iteration": record.iteration,
                    "log_name": record.log_name,
                    "reward": record.reward,
                    "outcome_reward": record.outcome_reward,
                    "prefix_correct_hops": record.prefix_correct_hops,
                    "nonzero_advantage": nonzero,
                }
            )

    summary.model_versions = {
        str(version): {
            "samples": int(stats["samples"]),
            "nonzero_samples": int(stats["nonzero"]),
            "nonzero_fraction": pct(stats["nonzero"], stats["samples"]),
            "mean_reward": stats["reward_sum"] / stats["samples"] if stats["samples"] else 0.0,
        }
        for version, stats in sorted(version_stats.items())
    }

    chains: list[dict] = []
    trace_stats = trace_stats or {}
    for stats in chain_stats.values():
        trace = trace_stats.get(stats["chain_key"], {})
        trace_outcomes = trace.get("trace_outcome_rewards") or []
        trace_gold_retrieved = trace.get("trace_gold_retrieved_values") or []
        trace_gold_read = trace.get("trace_gold_read_values") or []
        row = {
            "chain_key": stats["chain_key"],
            "task_id": stats["task_id"],
            "chain_id": stats["chain_id"],
            "company": stats["company"],
            "pattern": stats["pattern"],
            "chain_length": stats["chain_length"],
            "problem_id": stats["problem_id"],
            "samples": stats["samples"],
            "nonzero_samples": stats["nonzero_samples"],
            "nonzero_fraction": pct(stats["nonzero_samples"], stats["samples"]),
            "groups": len(stats["groups"]),
            "nonzero_groups": len(stats["nonzero_groups"]),
            "mean_training_reward": mean(stats["reward_values"]) if stats["reward_values"] else 0.0,
            "max_training_reward": max(stats["reward_values"]) if stats["reward_values"] else 0.0,
            "mean_outcome_reward": mean(stats["outcome_rewards"]) if stats["outcome_rewards"] else 0.0,
            "max_outcome_reward": max(stats["outcome_rewards"]) if stats["outcome_rewards"] else 0.0,
            "mean_prefix_correct_hops": mean(stats["prefix_correct_values"]) if stats["prefix_correct_values"] else 0.0,
            "gold_retrieved_rate": mean(stats["gold_retrieved_values"]) if stats["gold_retrieved_values"] else None,
            "gold_read_rate": mean(stats["gold_read_values"]) if stats["gold_read_values"] else None,
            "trace_rollouts": trace.get("trace_rollouts", 0),
            "trace_mean_outcome_reward": mean(trace_outcomes) if trace_outcomes else None,
            "trace_max_outcome_reward": max(trace_outcomes) if trace_outcomes else None,
            "trace_gold_retrieved_rate": mean(trace_gold_retrieved) if trace_gold_retrieved else None,
            "trace_gold_read_rate": mean(trace_gold_read) if trace_gold_read else None,
            "trace_mean_docs_read": mean(trace.get("trace_docs_read") or []) if trace.get("trace_docs_read") else None,
            "trace_mean_duration_s": mean(trace.get("trace_duration_s") or []) if trace.get("trace_duration_s") else None,
            "trace_examples": trace.get("trace_examples", []),
            "examples": stats["examples"],
        }
        if row["gold_retrieved_rate"] is None:
            row["gold_retrieved_rate"] = row["trace_gold_retrieved_rate"]
        if row["gold_read_rate"] is None:
            row["gold_read_rate"] = row["trace_gold_read_rate"]
        if row["max_outcome_reward"] == 0.0 and row["trace_max_outcome_reward"] is not None:
            row["max_outcome_reward"] = row["trace_max_outcome_reward"]
        if row["mean_outcome_reward"] == 0.0 and row["trace_mean_outcome_reward"] is not None:
            row["mean_outcome_reward"] = row["trace_mean_outcome_reward"]
        chains.append(row)
    chains.sort(key=lambda row: (row["nonzero_samples"], row["max_outcome_reward"], -row["samples"]))
    summary.chains = chains
    summary.never_nonzero_chains = [row for row in chains if row["samples"] >= 16 and row["nonzero_samples"] == 0][:300]
    summary.retrieval_impossible_candidates = [
        row
        for row in chains
        if row["samples"] >= 16
        and row["max_outcome_reward"] < 0.34
        and (row["gold_retrieved_rate"] is None or row["gold_retrieved_rate"] < 0.05)
    ][:300]


def rg_match_count(pattern: str, files: list[Path]) -> int:
    if not files:
        return 0
    result = subprocess.run(
        ["rg", "-l", pattern, *[str(path) for path in files]],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def probe_top_never_nonzero_traces(run_dir: Path, chains: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    trace_dir = run_dir / "privacy_hopqa_runs"
    if not trace_dir.exists():
        return []
    rows: list[dict] = []
    for chain in chains[:limit]:
        prefix = f"{chain['task_id']}_{chain['chain_id']}"
        files = sorted(trace_dir.glob(f"{prefix}_*/trace.json"))
        rows.append(
            {
                "task_id": chain["task_id"],
                "chain_id": chain["chain_id"],
                "company": chain["company"],
                "pattern": chain["pattern"],
                "actor_samples": chain["samples"],
                "trace_files": len(files),
                "traces_with_gold_retrieved": rg_match_count(r'"gold_parent_retrieved"\s*:\s*true', files),
                "traces_with_gold_read": rg_match_count(r'"gold_parent_read"\s*:\s*true', files),
                "traces_with_any_hop_accepted": rg_match_count(r'"matched_accepted_variant"\s*:\s*true', files),
                "problem_id": chain["problem_id"],
            }
        )
    return rows


def build_svg_bar(rows: list[dict], value_keys: list[str], labels: list[str], colors: list[str], title: str) -> str:
    width = 960
    height = 260
    pad_l = 70
    pad_b = 56
    pad_t = 32
    inner_w = width - pad_l - 30
    inner_h = height - pad_t - pad_b
    if not rows:
        return f"<div class='empty'>No data for {esc(title)}</div>"
    max_total = max(sum(float(row.get(key, 0) or 0) for key in value_keys) for row in rows) or 1.0
    bar_w = max(16, inner_w / max(len(rows), 1) * 0.7)
    gap = inner_w / max(len(rows), 1)
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img'><title>{esc(title)}</title>"]
    parts.append(f"<text x='{pad_l}' y='22' class='chart-title'>{esc(title)}</text>")
    parts.append(f"<line x1='{pad_l}' y1='{pad_t + inner_h}' x2='{pad_l + inner_w}' y2='{pad_t + inner_h}' class='axis'/>")
    for idx, row in enumerate(rows):
        x = pad_l + idx * gap + (gap - bar_w) / 2
        y_cursor = pad_t + inner_h
        total = 0.0
        for key, color in zip(value_keys, colors, strict=False):
            value = float(row.get(key, 0) or 0)
            total += value
            h = inner_h * value / max_total
            y_cursor -= h
            parts.append(f"<rect x='{x:.1f}' y='{y_cursor:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'><title>{esc(row.get('label', idx))}: {key}={value:.0f}</title></rect>")
        label = str(row.get("label", idx))
        parts.append(f"<text x='{x + bar_w / 2:.1f}' y='{height - 28}' text-anchor='middle' class='tick'>{esc(label[-12:])}</text>")
        parts.append(f"<text x='{x + bar_w / 2:.1f}' y='{max(16, y_cursor - 4):.1f}' text-anchor='middle' class='tiny'>{total:.0f}</text>")
    legend_x = pad_l
    for label, color in zip(labels, colors, strict=False):
        parts.append(f"<rect x='{legend_x}' y='{height - 16}' width='10' height='10' fill='{color}'/>")
        parts.append(f"<text x='{legend_x + 14}' y='{height - 7}' class='legend'>{esc(label)}</text>")
        legend_x += 130
    parts.append("</svg>")
    return "".join(parts)


def build_svg_line(rows: list[dict], x_key: str, y_key: str, title: str, suffix: str = "%") -> str:
    width = 960
    height = 260
    pad_l = 70
    pad_b = 42
    pad_t = 32
    inner_w = width - pad_l - 30
    inner_h = height - pad_t - pad_b
    rows = [row for row in rows if row.get(y_key) is not None]
    if len(rows) < 2:
        return f"<div class='empty'>Not enough data for {esc(title)}</div>"
    xs = list(range(len(rows)))
    ys = [float(row[y_key]) for row in rows]
    y_min = min(0.0, min(ys))
    y_max = max(1.0, max(ys))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    def sx(i: int) -> float:
        return pad_l + inner_w * i / max(len(rows) - 1, 1)
    def sy(value: float) -> float:
        return pad_t + inner_h - inner_h * (value - y_min) / (y_max - y_min)
    points = " ".join(f"{sx(i):.1f},{sy(value):.1f}" for i, value in zip(xs, ys, strict=False))
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img'><title>{esc(title)}</title>"]
    parts.append(f"<text x='{pad_l}' y='22' class='chart-title'>{esc(title)}</text>")
    parts.append(f"<line x1='{pad_l}' y1='{pad_t + inner_h}' x2='{pad_l + inner_w}' y2='{pad_t + inner_h}' class='axis'/>")
    parts.append(f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + inner_h}' class='axis'/>")
    parts.append(f"<polyline fill='none' stroke='#1f6f5b' stroke-width='2.5' points='{points}'/>")
    for i, value in enumerate(ys[:: max(1, len(ys) // 50)]):
        actual_i = i * max(1, len(ys) // 50)
        parts.append(f"<circle cx='{sx(actual_i):.1f}' cy='{sy(value):.1f}' r='2.5' fill='#1f6f5b'><title>{esc(rows[actual_i].get(x_key, actual_i))}: {value:.1f}{suffix}</title></circle>")
    parts.append(f"<text x='{pad_l}' y='{pad_t - 6}' class='tiny'>{y_max:.1f}{esc(suffix)}</text>")
    parts.append(f"<text x='{pad_l}' y='{pad_t + inner_h + 14}' class='tiny'>{y_min:.1f}{esc(suffix)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def table(headers: list[str], rows: list[list[object]], limit: int = 200) -> str:
    shown = rows[:limit]
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>")
    more = ""
    if len(rows) > limit:
        more = f"<p class='muted'>Showing {limit} of {len(rows)} rows.</p>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def build_html(data: dict) -> str:
    run_sections = []
    for run in data["runs"]:
        model_rows = [
            {
                "label": version,
                "nonzero": stats["nonzero_samples"],
                "zero": stats["samples"] - stats["nonzero_samples"],
            }
            for version, stats in run["model_versions"].items()
        ]
        never_rows = [
            [
                row["task_id"], row["chain_id"], row["company"], row["pattern"], row["chain_length"], row["samples"],
                row["trace_rollouts"], f"{row['max_outcome_reward']:.3f}", f"{row['mean_outcome_reward']:.3f}",
                "n/a" if row["gold_retrieved_rate"] is None else f"{100 * row['gold_retrieved_rate']:.1f}%",
                "n/a" if row["gold_read_rate"] is None else f"{100 * row['gold_read_rate']:.1f}%",
                row["problem_id"],
            ]
            for row in run["never_nonzero_chains"]
        ]
        impossible_rows = [
            [
                row["task_id"], row["chain_id"], row["company"], row["pattern"], row["samples"],
                row["trace_rollouts"], f"{row['max_outcome_reward']:.3f}",
                "n/a" if row["gold_retrieved_rate"] is None else f"{100 * row['gold_retrieved_rate']:.1f}%",
                "n/a" if row["gold_read_rate"] is None else f"{100 * row['gold_read_rate']:.1f}%",
                row["problem_id"],
            ]
            for row in run["retrieval_impossible_candidates"]
        ]
        probe_rows = [
            [
                row["task_id"], row["chain_id"], row["company"], row["pattern"], row["actor_samples"], row["trace_files"],
                row["traces_with_gold_retrieved"], row["traces_with_gold_read"], row["traces_with_any_hop_accepted"], row["problem_id"],
            ]
            for row in run.get("targeted_trace_probe", [])
        ]
        warnings = "".join(f"<li>{esc(warning)}</li>" for warning in run["warnings"])
        run_sections.append(
            f"""
<section class='run'>
  <h2>{esc(run['label'])}</h2>
  <p class='path'>{esc(run['path'])}</p>
  <div class='cards'>
    <div class='card'><div class='num'>{run['actor_samples']:,}</div><div>actor samples scanned</div></div>
    <div class='card'><div class='num'>{run['kept_from_log']:,}</div><div>preprocessor kept</div></div>
    <div class='card'><div class='num'>{run['filtered_from_log']:,}</div><div>preprocessor filtered</div></div>
    <div class='card'><div class='num'>{run['filtered_fraction']:.1f}%</div><div>filtered fraction</div></div>
    <div class='card'><div class='num'>{run['nonzero_samples']:,}</div><div>approx non-zero-adv samples</div></div>
    <div class='card'><div class='num'>{run['nonzero_fraction']:.1f}%</div><div>approx non-zero-adv fraction</div></div>
    <div class='card'><div class='num'>{run['mixed_group_keys']:,}</div><div>mixed reward group keys</div></div>
    <div class='card'><div class='num'>{len(run['chains']):,}</div><div>chains seen in actor stream</div></div>
  </div>
  {'<ul class="warnings">' + warnings + '</ul>' if warnings else ''}
  <div class='grid'>
    <div class='panel'>{build_svg_bar(model_rows, ['nonzero', 'zero'], ['non-zero approx advantage', 'zero approx advantage'], ['#207a5b', '#d8d1c4'], 'Approx actor advantage coverage by model version')}</div>
    <div class='panel'>{build_svg_line(run['filter_events'], 'model_version', 'filtered_fraction', 'Filtered-out fraction over preprocessor time')}</div>
  </div>
  <details open><summary>Chains with actor samples but no approximate non-zero advantage</summary>
    <p>These are chain rows for which every scanned actor sample falls in a same-reward LOO comparison group. If this persists with many samples, training never pushes on that row.</p>
    {table(['task', 'chain', 'company', 'pattern', 'len', 'actor samples', 'trace rollouts', 'max outcome', 'mean outcome', 'gold retrieved', 'gold read', 'problem'], never_rows)}
  </details>
  <details><summary>Possible impossible-to-retrieve chains</summary>
    <p>Heuristic: at least 16 actor samples, max outcome reward &lt; 0.34, and gold-parent retrieved rate missing or below 5%.</p>
    {table(['task', 'chain', 'company', 'pattern', 'actor samples', 'trace rollouts', 'max outcome', 'gold retrieved', 'gold read', 'problem'], impossible_rows)}
  </details>
  <details open><summary>Targeted trace probe for top never-nonzero chains</summary>
    <p>This uses <code>rg</code> over trace files for the top never-nonzero rows. If gold retrieved/read and accepted-hop counts are all zero across many traces, the row is likely not producing useful training signal because retrieval never reaches the evidence.</p>
    {table(['task', 'chain', 'company', 'pattern', 'actor samples', 'trace files', 'traces gold retrieved', 'traces gold read', 'traces any hop accepted', 'problem'], probe_rows)}
  </details>
</section>
"""
        )
    top_rows = [
        {
            "label": run["label"],
            "kept": run["kept_from_log"],
            "filtered": run["filtered_from_log"],
        }
        for run in data["runs"]
    ]
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Privacy HopQA training filter viewer</title>
<style>
:root {{ --ink:#1d2621; --muted:#67716a; --paper:#faf7ef; --panel:#fffdf7; --line:#d7ccba; --green:#207a5b; --amber:#ba7d1c; }}
body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:var(--ink); background:linear-gradient(145deg,#f7f0df,#edf4ee); }}
header {{ padding:32px 42px 18px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:32px; }}
p {{ line-height:1.45; }}
main {{ padding:24px 42px 60px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:18px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; box-shadow:0 10px 24px rgba(60,45,20,.06); }}
.num {{ font-weight:800; font-size:24px; }}
.run {{ margin:28px 0 44px; }}
.path {{ color:var(--muted); font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px; overflow-x:auto; }}
svg {{ width:100%; max-width:980px; }}
.axis {{ stroke:#8e826d; stroke-width:1; }}
.chart-title {{ font-weight:700; font-size:15px; fill:var(--ink); }}
.tick,.tiny,.legend {{ fill:var(--muted); font-size:10px; }}
details {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; margin:18px 0; }}
summary {{ cursor:pointer; font-weight:800; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:12px; }}
th,td {{ border-bottom:1px solid #e8decc; padding:7px 8px; text-align:left; vertical-align:top; }}
th {{ position:sticky; top:0; background:#f5eddd; }}
.muted {{ color:var(--muted); }}
.warnings {{ color:#8c3b20; }}
.empty {{ padding:30px; color:var(--muted); }}
</style>
</head>
<body>
<header>
<h1>Privacy HopQA training filter viewer</h1>
<p>This viewer answers three questions: what fraction of generated actor samples are filtered before training, whether that fraction changes by model version/time, and which chain rows appear to never receive non-zero LOO advantage. Actor advantage is approximate but uses the same grouping convention we currently rely on: prefix-step runs group by <code>group_id + step_index</code>; outcome-level runs group by <code>group_id</code>.</p>
</header>
<main>
<div class='panel'>{build_svg_bar(top_rows, ['kept', 'filtered'], ['kept by preprocessor', 'filtered by preprocessor'], ['#207a5b', '#ba7d1c'], 'Preprocessor kept vs filtered samples')}</div>
{''.join(run_sections)}
</main>
<script type='application/json' id='viewer-data'>{esc(json.dumps(data))}</script>
</body>
</html>
"""


def build(args: argparse.Namespace) -> dict:
    runs = []
    for label, path, scan_actor in DEFAULT_RUNS:
        summary = RunSummary(label=label, path=str(path), actor_scanned=scan_actor)
        if not path.exists():
            summary.warnings.append("Run directory does not exist")
            runs.append(summary)
            continue
        events, kept, filtered = parse_preprocess_log(path)
        summary.filter_events = events
        summary.kept_from_log = kept
        summary.filtered_from_log = filtered
        summary.stats_kept, summary.stats_filtered = parse_preprocessor_stats(path)
        should_scan = scan_actor or args.scan_all_actor
        if should_scan:
            records, warnings = scan_actor_samples(label, path, args.max_actor_lines)
            summary.warnings.extend(warnings)
            trace_stats, trace_warnings = scan_trace_stats(path, args.max_traces)
            summary.warnings.extend(trace_warnings)
            aggregate_records(summary, records, trace_stats)
            summary.targeted_trace_probe = probe_top_never_nonzero_traces(path, summary.never_nonzero_chains, args.trace_probe_limit)
        else:
            summary.warnings.append("Actor stream not scanned by default; use --scan-all-actor for this run")
        runs.append(summary)
    run_dicts = []
    for summary in runs:
        row = dict(summary.__dict__)
        row["filtered_fraction"] = summary.filtered_fraction
        row["nonzero_fraction"] = summary.nonzero_fraction
        run_dicts.append(row)
    data = {
        "generated_at_utc": args.generated_at,
        "max_actor_lines": args.max_actor_lines,
        "max_traces": args.max_traces,
        "trace_probe_limit": args.trace_probe_limit,
        "runs": run_dicts,
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-actor-lines", type=int, default=0, help="Cap actor JSONL lines per scanned run; 0 means unlimited")
    parser.add_argument("--max-traces", type=int, default=0, help="Cap trace files per scanned run; 0 skips trace enrichment")
    parser.add_argument("--trace-probe-limit", type=int, default=20, help="Run rg probes over trace files for this many never-nonzero chains")
    parser.add_argument("--scan-all-actor", action="store_true", help="Also scan older comparison runs' actor streams")
    parser.add_argument("--generated-at", default="2026-05-14")
    args = parser.parse_args()
    data = build(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "viewer_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (args.out_dir / "index.html").write_text(build_html(data), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'index.html'}")
    for run in data["runs"]:
        print(
            f"{run['label']}: kept={run['kept_from_log']} filtered={run['filtered_from_log']} "
            f"filtered_frac={run['filtered_fraction']:.1f}% actor={run['actor_samples']} "
            f"nonzero_adv={run['nonzero_samples']} ({run['nonzero_fraction']:.1f}%) "
            f"never_nonzero_chains={len(run['never_nonzero_chains'])}"
        )


if __name__ == "__main__":
    main()
