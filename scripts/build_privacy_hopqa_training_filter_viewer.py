#!/usr/bin/env python3
"""Build a Privacy HopQA training filtering / advantage viewer.

Actor stream records are huge because each sample contains token arrays. This script
therefore does not json.loads actor samples. It scans JSONL lines for top-level sample
starts, extracts only reward/group/privacy metadata, and aggregates by LOO comparison
group. That lets us scan all actor shard files while keeping memory proportional to
number of comparison groups rather than number of tokenized samples.
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

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "runtime_analysis" / "privacy_hopqa_training_filter_viewer_2026-05-15"
DEFAULT_RUNS = [
    (
        "4B hop-accuracy RRF",
        Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_hopacc_20260514202306"),
        True,
    ),
    (
        "4B prefix-step RRF",
        Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_rrf_4b_prefix_step_20260514201431"),
        True,
    ),
]

PROCESSED_RE = re.compile(
    r"Processed\s+(?P<kept>\d+)\s+samples\s+\(filtered out\s+(?P<filtered>\d+)\).*?total\s+(?P<total>\d+)\s+samples so far(?:.*?model version\s+(?P<version>\d+))?",
)
FILTERED_ONLY_RE = re.compile(r"Filtered out\s+(?P<filtered>\d+)\s+samples")
SAMPLE_START_RE = re.compile(r'(?:(?<=\[)|(?<=,))\{"text":')
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
}


def pct(numer: float, denom: float) -> float:
    return 0.0 if denom <= 0 else 100.0 * numer / denom


def esc(value: object) -> str:
    return html.escape(str(value))


def match_text(pattern_name: str, text: str, start: int, end: int) -> str | None:
    match = PATTERNS[pattern_name].search(text, start, end)
    return match.group(1) if match else None


def match_int(pattern_name: str, text: str, start: int, end: int, default: int = 0) -> int:
    value = match_text(pattern_name, text, start, end)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def match_float(pattern_name: str, text: str, start: int, end: int, default: float = 0.0) -> float:
    value = match_text(pattern_name, text, start, end)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def bytes_string(line: bytes, start: int, end: int, key: bytes, default: str = "") -> str:
    marker = b'"' + key + b'":"'
    pos = line.find(marker, start, end)
    if pos < 0:
        return default
    value_start = pos + len(marker)
    value_end = line.find(b'"', value_start, end)
    if value_end < 0:
        return default
    return line[value_start:value_end].decode("utf-8", errors="replace")


def bytes_number(line: bytes, start: int, end: int, key: bytes) -> bytes | None:
    marker = b'"' + key + b'":'
    pos = line.find(marker, start, end)
    if pos < 0:
        return None
    value_start = pos + len(marker)
    while value_start < end and line[value_start] in b" \t":
        value_start += 1
    value_end = value_start
    allowed = b"-+0123456789.eE"
    while value_end < end and line[value_end] in allowed:
        value_end += 1
    if value_end == value_start:
        return None
    return line[value_start:value_end]


def bytes_int(line: bytes, start: int, end: int, key: bytes, default: int = 0) -> int:
    value = bytes_number(line, start, end, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def bytes_float(line: bytes, start: int, end: int, key: bytes, default: float = 0.0) -> float:
    value = bytes_number(line, start, end, key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class GroupAccumulator:
    run_label: str
    group_key: str
    chain_key: str
    group_id: str
    first_line: int
    last_line: int
    first_sample_index: int
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
    count: int = 0
    reward_sum: float = 0.0
    reward_min: float = math.inf
    reward_max: float = -math.inf
    outcome_reward_max: float = -math.inf
    outcome_reward_min: float = math.inf
    outcome_reward_sum: float = 0.0
    prefix_correct_max: int = 0

    def add(self, reward: float, outcome_reward: float, prefix_correct_hops: int, line_no: int) -> None:
        self.count += 1
        self.reward_sum += reward
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)
        self.outcome_reward_sum += outcome_reward
        self.outcome_reward_max = max(self.outcome_reward_max, outcome_reward)
        self.outcome_reward_min = min(self.outcome_reward_min, outcome_reward)
        self.prefix_correct_max = max(self.prefix_correct_max, prefix_correct_hops)
        self.last_line = max(self.last_line, line_no)

    @property
    def has_nonzero_advantage(self) -> bool:
        return self.count > 1 and abs(self.reward_max - self.reward_min) > 1e-9

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.count if self.count else 0.0

    @property
    def mean_outcome_reward(self) -> float:
        return self.outcome_reward_sum / self.count if self.count else 0.0


@dataclass
class ChainAccumulator:
    task_id: str
    chain_id: str
    company: str
    pattern: str
    chain_length: int
    problem_id: str
    samples: int = 0
    nonzero_samples: int = 0
    groups: int = 0
    nonzero_groups: int = 0
    reward_sum: float = 0.0
    outcome_reward_sum: float = 0.0
    max_reward: float = -math.inf
    max_outcome_reward: float = -math.inf
    max_prefix_correct_hops: int = 0
    first_line: int = 0
    last_line: int = 0
    first_nonzero_line: int | None = None
    first_nonzero_sample_index: int | None = None
    zero_groups_before_first_nonzero: int = 0
    samples_before_first_nonzero: int = 0
    example_group_keys: list[str] = field(default_factory=list)
    group_history: list[dict] = field(default_factory=list)

    def add_group(self, group: GroupAccumulator) -> None:
        nonzero = group.has_nonzero_advantage
        if self.groups == 0:
            self.first_line = group.first_line
        self.last_line = max(self.last_line, group.last_line)
        if self.first_nonzero_line is None and nonzero:
            self.first_nonzero_line = group.first_line
            self.first_nonzero_sample_index = group.first_sample_index
        if self.first_nonzero_line is None and not nonzero:
            self.zero_groups_before_first_nonzero += 1
            self.samples_before_first_nonzero += group.count
        self.samples += group.count
        self.nonzero_samples += group.count if nonzero else 0
        self.groups += 1
        self.nonzero_groups += 1 if nonzero else 0
        self.reward_sum += group.reward_sum
        self.outcome_reward_sum += group.outcome_reward_sum
        self.max_reward = max(self.max_reward, group.reward_max)
        self.max_outcome_reward = max(self.max_outcome_reward, group.outcome_reward_max)
        self.max_prefix_correct_hops = max(self.max_prefix_correct_hops, group.prefix_correct_max)
        if len(self.example_group_keys) < 4:
            self.example_group_keys.append(group.group_key)
        self.group_history.append(
            {
                "line": group.first_line,
                "sample_index": group.first_sample_index,
                "nonzero": group.has_nonzero_advantage,
                "count": group.count,
                "reward_min": group.reward_min if group.reward_min != math.inf else 0.0,
                "reward_max": group.reward_max if group.reward_max != -math.inf else 0.0,
                "outcome_reward_min": group.outcome_reward_min if group.outcome_reward_min != math.inf else 0.0,
                "outcome_reward_max": group.outcome_reward_max if group.outcome_reward_max != -math.inf else 0.0,
            }
        )

    def to_row(self) -> dict:
        return {
            "task_id": self.task_id,
            "chain_id": self.chain_id,
            "company": self.company,
            "pattern": self.pattern,
            "chain_length": self.chain_length,
            "problem_id": self.problem_id,
            "samples": self.samples,
            "nonzero_samples": self.nonzero_samples,
            "nonzero_fraction": pct(self.nonzero_samples, self.samples),
            "groups": self.groups,
            "nonzero_groups": self.nonzero_groups,
            "nonzero_group_fraction": pct(self.nonzero_groups, self.groups),
            "mean_training_reward": self.reward_sum / self.samples if self.samples else 0.0,
            "max_training_reward": self.max_reward if self.max_reward != -math.inf else 0.0,
            "mean_outcome_reward": self.outcome_reward_sum / self.samples if self.samples else 0.0,
            "max_outcome_reward": self.max_outcome_reward if self.max_outcome_reward != -math.inf else 0.0,
            "max_prefix_correct_hops": self.max_prefix_correct_hops,
            "first_line": self.first_line,
            "last_line": self.last_line,
            "first_nonzero_line": self.first_nonzero_line,
            "first_nonzero_sample_index": self.first_nonzero_sample_index,
            "zero_groups_before_first_nonzero": self.zero_groups_before_first_nonzero,
            "samples_before_first_nonzero": self.samples_before_first_nonzero,
            "example_group_keys": self.example_group_keys,
            "group_history": sorted(self.group_history, key=lambda item: (item["line"], item["sample_index"])),
        }


@dataclass
class RunSummary:
    label: str
    path: str
    actor_scanned: bool
    warnings: list[str] = field(default_factory=list)
    filter_events: list[dict] = field(default_factory=list)
    stats_events: list[dict] = field(default_factory=list)
    latest_success_distribution: dict[str, float] = field(default_factory=dict)
    kept_from_log: int = 0
    filtered_from_log: int = 0
    stats_kept: int = 0
    stats_filtered: int = 0
    actor_files: int = 0
    actor_bytes: int = 0
    actor_lines: int = 0
    actor_samples: int = 0
    group_keys: int = 0
    nonzero_samples: int = 0
    nonzero_group_keys: int = 0
    mixed_group_keys: int = 0
    model_versions: dict[str, dict] = field(default_factory=dict)
    timeline_bins: list[dict] = field(default_factory=list)
    chains: list[dict] = field(default_factory=list)
    never_nonzero_chains: list[dict] = field(default_factory=list)
    became_nonzero_chains: list[dict] = field(default_factory=list)
    retrieval_impossible_candidates: list[dict] = field(default_factory=list)
    targeted_trace_probe: list[dict] = field(default_factory=list)
    flow_summary: dict = field(default_factory=dict)

    @property
    def filtered_fraction(self) -> float:
        kept = self.kept_from_log or self.stats_kept
        filtered = self.filtered_from_log or self.stats_filtered
        return pct(filtered, kept + filtered)

    @property
    def nonzero_fraction(self) -> float:
        return pct(self.nonzero_samples, self.actor_samples)


def actor_paths(run_dir: Path) -> list[Path]:
    base = run_dir / "streams" / "actor" / "0"
    if not base.exists():
        return []
    return sorted(base.glob("*/0.jsonl"))


def parse_preprocess_log(run_dir: Path) -> tuple[list[dict], int, int]:
    log_path = run_dir / "preprocess" / "info.log"
    events: list[dict] = []
    kept_total = 0
    filtered_total = 0
    if not log_path.exists():
        return events, 0, 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            match = PROCESSED_RE.search(line)
            if match:
                kept = int(match.group("kept"))
                filtered = int(match.group("filtered"))
                kept_total += kept
                filtered_total += filtered
                events.append(
                    {
                        "line": line_no,
                        "model_version": int(match.group("version") or 0),
                        "kept": kept,
                        "filtered": filtered,
                        "kept_cumulative": kept_total,
                        "filtered_cumulative": filtered_total,
                        "filtered_fraction": pct(filtered, kept + filtered),
                    }
                )
                continue
            if FILTERED_ONLY_RE.search(line):
                events.append(
                    {
                        "line": line_no,
                        "model_version": 0,
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


def parse_stats_stream(run_dir: Path) -> tuple[list[dict], dict[str, float]]:
    stats_path = run_dir / "streams" / "stats" / "0" / "0" / "0.jsonl"
    events: list[dict] = []
    latest: dict[str, float] = {}
    if not stats_path.exists():
        return events, latest
    with stats_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = {
                "line": line_no,
                "trainer_model_version": row.get("trainer_model_version"),
                "published_samples": row.get("published_samples"),
                "finished_groups": row.get("finished_groups"),
                "samples_per_second": row.get("samples_per_second"),
                "always_success": row.get("always_success"),
                "never_success": row.get("never_success"),
                "sometimes_success": row.get("sometimes_success"),
                "reward_mean": row.get("reward_mean"),
                "success_mean": row.get("success_mean"),
            }
            if event["never_success"] is not None:
                event["never_success_pct"] = 100.0 * float(event["never_success"])
            if event["sometimes_success"] is not None:
                event["sometimes_success_pct"] = 100.0 * float(event["sometimes_success"])
            if event["always_success"] is not None:
                event["always_success_pct"] = 100.0 * float(event["always_success"])
            events.append(event)
            latest = {key: value for key, value in event.items() if value is not None}
    return events, latest


def sample_ranges(line: bytes) -> list[tuple[int, int]]:
    starts: list[int] = []
    pos = 0
    marker = b'{"text":'
    while True:
        pos = line.find(marker, pos)
        if pos < 0:
            break
        if pos == 1 or (pos > 0 and line[pos - 1] == ord(",")):
            starts.append(pos)
        pos += len(marker)
    if not starts:
        return []
    starts.append(len(line))
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def scan_actor_groups(run_label: str, run_dir: Path, max_lines: int) -> tuple[dict[str, GroupAccumulator], dict, list[str]]:
    groups: dict[str, GroupAccumulator] = {}
    warnings: list[str] = []
    paths = actor_paths(run_dir)
    actor_bytes = sum(path.stat().st_size for path in paths)
    sample_index = 0
    line_count = 0
    for path in paths:
        with path.open("rb") as handle:
            for file_line_no, line in enumerate(handle, start=1):
                line_count += 1
                if max_lines and line_count > max_lines:
                    warnings.append(f"Actor scan capped at {max_lines} lines")
                    return groups, {"actor_files": len(paths), "actor_bytes": actor_bytes, "actor_lines": line_count - 1, "actor_samples": sample_index}, warnings
                for start, end in sample_ranges(line):
                    group_id = bytes_string(line, start, end, b"group_id")
                    if not group_id:
                        continue
                    reward = bytes_float(line, start, end, b"reward", math.nan)
                    if math.isnan(reward):
                        continue
                    step_index = bytes_int(line, start, end, b"step_index", 0)
                    mode = bytes_string(line, start, end, b"training_reward_mode", "outcome")
                    group_key = f"{group_id}::step={step_index}" if mode == "prefix_progress" else group_id
                    task_id = bytes_string(line, start, end, b"task_id", "unknown_task")
                    chain_id = bytes_string(line, start, end, b"chain_id", "unknown_chain")
                    chain_key = f"{task_id}::{chain_id}"
                    model_version = bytes_int(line, start, end, b"model_version", 0)
                    outcome_reward = bytes_float(line, start, end, b"outcome_reward", reward)
                    prefix_correct_hops = bytes_int(line, start, end, b"prefix_correct_hops", 0)
                    group = groups.get(group_key)
                    if group is None:
                        group = GroupAccumulator(
                            run_label=run_label,
                            group_key=group_key,
                            chain_key=chain_key,
                            group_id=group_id,
                            first_line=line_count,
                            last_line=line_count,
                            first_sample_index=sample_index,
                            model_version=model_version,
                            task_id=task_id,
                            chain_id=chain_id,
                            company=bytes_string(line, start, end, b"company", "unknown"),
                            pattern=bytes_string(line, start, end, b"pattern", "unknown"),
                            chain_length=bytes_int(line, start, end, b"chain_length", 0),
                            problem_id=bytes_string(line, start, end, b"problem_id"),
                            hop_number=bytes_int(line, start, end, b"hop_number", 0),
                            iteration=bytes_int(line, start, end, b"iteration", 0),
                            log_name=bytes_string(line, start, end, b"log_name"),
                        )
                        groups[group_key] = group
                    group.add(reward, outcome_reward, prefix_correct_hops, line_count)
                    sample_index += 1
    return groups, {"actor_files": len(paths), "actor_bytes": actor_bytes, "actor_lines": line_count, "actor_samples": sample_index}, warnings


def window_has_nonzero(history: list[dict], window_groups: int, from_end: bool) -> bool:
    if not history:
        return False
    ordered = sorted(history, key=lambda item: (item["line"], item["sample_index"]))
    window = ordered[-window_groups:] if from_end else ordered[:window_groups]
    return any(bool(item["nonzero"]) for item in window)


def build_flow_summary(chain_rows: list[dict], window_groups: int) -> dict:
    flows: dict[str, dict] = {}
    states = {
        "zero_adv": "Zero advantage",
        "mixed_adv": "Mixed advantage",
    }
    for row in chain_rows:
        history = row.get("group_history", [])
        if not history:
            continue
        start_key = "mixed_adv" if window_has_nonzero(history, window_groups, from_end=False) else "zero_adv"
        end_key = "mixed_adv" if window_has_nonzero(history, window_groups, from_end=True) else "zero_adv"
        flow_key = f"{start_key}->{end_key}"
        item = flows.setdefault(
            flow_key,
            {
                "from": start_key,
                "from_label": states[start_key],
                "to": end_key,
                "to_label": states[end_key],
                "chains": 0,
                "samples": 0,
                "examples": [],
            },
        )
        item["chains"] += 1
        item["samples"] += int(row.get("samples", 0) or 0)
        if len(item["examples"]) < 8:
            item["examples"].append(
                {
                    "task_id": row["task_id"],
                    "chain_id": row["chain_id"],
                    "company": row["company"],
                    "pattern": row["pattern"],
                    "samples": row["samples"],
                    "max_outcome_reward": row["max_outcome_reward"],
                }
            )
    total = sum(item["chains"] for item in flows.values())
    for item in flows.values():
        item["chain_fraction"] = pct(item["chains"], total)
    return {
        "window_groups": window_groups,
        "total_chains": total,
        "flows": flows,
    }


def strip_internal_history(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "group_history"}


def aggregate_groups(summary: RunSummary, groups: dict[str, GroupAccumulator], line_bin_size: int, flow_window_groups: int) -> None:
    chains: dict[str, ChainAccumulator] = {}
    version_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"samples": 0, "nonzero_samples": 0, "reward_sum": 0.0, "groups": 0, "nonzero_groups": 0})
    bins: dict[int, dict[str, float]] = defaultdict(lambda: {"line_start": 0, "line_end": 0, "samples": 0, "nonzero_samples": 0, "groups": 0, "nonzero_groups": 0, "reward_sum": 0.0})

    for group in groups.values():
        nonzero = group.has_nonzero_advantage
        summary.nonzero_samples += group.count if nonzero else 0
        summary.nonzero_group_keys += 1 if nonzero else 0
        summary.mixed_group_keys += 1 if nonzero else 0

        version = version_stats[group.model_version]
        version["samples"] += group.count
        version["nonzero_samples"] += group.count if nonzero else 0
        version["reward_sum"] += group.reward_sum
        version["groups"] += 1
        version["nonzero_groups"] += 1 if nonzero else 0

        bin_id = (group.first_line - 1) // max(1, line_bin_size)
        bucket = bins[bin_id]
        bucket["line_start"] = bin_id * line_bin_size + 1
        bucket["line_end"] = max(bucket["line_end"], group.last_line)
        bucket["samples"] += group.count
        bucket["nonzero_samples"] += group.count if nonzero else 0
        bucket["groups"] += 1
        bucket["nonzero_groups"] += 1 if nonzero else 0
        bucket["reward_sum"] += group.reward_sum

        chain = chains.get(group.chain_key)
        if chain is None:
            chain = ChainAccumulator(
                task_id=group.task_id,
                chain_id=group.chain_id,
                company=group.company,
                pattern=group.pattern,
                chain_length=group.chain_length,
                problem_id=group.problem_id,
            )
            chains[group.chain_key] = chain
        chain.add_group(group)

    summary.actor_samples = sum(group.count for group in groups.values())
    summary.group_keys = len(groups)
    summary.model_versions = {
        str(version): {
            "samples": int(stats["samples"]),
            "groups": int(stats["groups"]),
            "nonzero_samples": int(stats["nonzero_samples"]),
            "nonzero_groups": int(stats["nonzero_groups"]),
            "nonzero_fraction": pct(stats["nonzero_samples"], stats["samples"]),
            "nonzero_group_fraction": pct(stats["nonzero_groups"], stats["groups"]),
            "mean_reward": stats["reward_sum"] / stats["samples"] if stats["samples"] else 0.0,
        }
        for version, stats in sorted(version_stats.items())
    }
    summary.timeline_bins = [
        {
            "bin": int(bin_id),
            "line_start": int(row["line_start"]),
            "line_end": int(row["line_end"]),
            "samples": int(row["samples"]),
            "nonzero_samples": int(row["nonzero_samples"]),
            "nonzero_fraction": pct(row["nonzero_samples"], row["samples"]),
            "groups": int(row["groups"]),
            "nonzero_groups": int(row["nonzero_groups"]),
            "nonzero_group_fraction": pct(row["nonzero_groups"], row["groups"]),
            "mean_reward": row["reward_sum"] / row["samples"] if row["samples"] else 0.0,
        }
        for bin_id, row in sorted(bins.items())
    ]

    chain_rows = [chain.to_row() for chain in chains.values()]
    chain_rows.sort(key=lambda row: (row["nonzero_samples"], row["max_outcome_reward"], -row["samples"]))
    summary.chains = [strip_internal_history(row) for row in chain_rows]
    summary.never_nonzero_chains = [
        strip_internal_history(row) for row in chain_rows if row["samples"] >= 16 and row["nonzero_samples"] == 0
    ][:500]
    summary.became_nonzero_chains = sorted(
        [
            strip_internal_history(row)
            for row in chain_rows
            if row["first_nonzero_line"] is not None and row["zero_groups_before_first_nonzero"] > 0
        ],
        key=lambda row: (-row["samples_before_first_nonzero"], row["first_nonzero_line"] or 0),
    )[:500]
    summary.retrieval_impossible_candidates = [
        strip_internal_history(row)
        for row in chain_rows
        if row["samples"] >= 16 and row["max_outcome_reward"] < 0.34 and row["nonzero_samples"] == 0
    ][:500]
    summary.flow_summary = build_flow_summary(chain_rows, flow_window_groups)


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
    width = 980
    height = 260
    pad_l = 70
    pad_t = 34
    pad_b = 54
    inner_w = width - pad_l - 32
    inner_h = height - pad_t - pad_b
    if not rows:
        return f"<div class='empty'>No data for {esc(title)}</div>"
    max_total = max(sum(float(row.get(key, 0) or 0) for key in value_keys) for row in rows) or 1.0
    gap = inner_w / max(len(rows), 1)
    bar_w = max(12, gap * 0.72)
    parts = [f"<svg viewBox='0 0 {width} {height}'><title>{esc(title)}</title>"]
    parts.append(f"<text x='{pad_l}' y='23' class='chart-title'>{esc(title)}</text>")
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
        parts.append(f"<text x='{x + bar_w / 2:.1f}' y='{height - 28}' text-anchor='middle' class='tick'>{esc(label[-14:])}</text>")
        parts.append(f"<text x='{x + bar_w / 2:.1f}' y='{max(16, y_cursor - 4):.1f}' text-anchor='middle' class='tiny'>{total:.0f}</text>")
    legend_x = pad_l
    for label, color in zip(labels, colors, strict=False):
        parts.append(f"<rect x='{legend_x}' y='{height - 16}' width='10' height='10' fill='{color}'/>")
        parts.append(f"<text x='{legend_x + 14}' y='{height - 7}' class='legend'>{esc(label)}</text>")
        legend_x += 155
    parts.append("</svg>")
    return "".join(parts)


def build_svg_line(rows: list[dict], y_key: str, title: str, suffix: str = "%") -> str:
    width = 980
    height = 260
    pad_l = 70
    pad_t = 34
    pad_b = 42
    inner_w = width - pad_l - 32
    inner_h = height - pad_t - pad_b
    rows = [row for row in rows if row.get(y_key) is not None]
    if len(rows) < 2:
        return f"<div class='empty'>Not enough data for {esc(title)}</div>"
    ys = [float(row[y_key]) for row in rows]
    y_min = min(0.0, min(ys))
    y_max = max(1.0, max(ys))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    step = max(1, len(rows) // 180)
    def sx(i: int) -> float:
        return pad_l + inner_w * i / max(len(rows) - 1, 1)
    def sy(value: float) -> float:
        return pad_t + inner_h - inner_h * (value - y_min) / (y_max - y_min)
    points = " ".join(f"{sx(i):.1f},{sy(value):.1f}" for i, value in enumerate(ys))
    parts = [f"<svg viewBox='0 0 {width} {height}'><title>{esc(title)}</title>"]
    parts.append(f"<text x='{pad_l}' y='23' class='chart-title'>{esc(title)}</text>")
    parts.append(f"<line x1='{pad_l}' y1='{pad_t + inner_h}' x2='{pad_l + inner_w}' y2='{pad_t + inner_h}' class='axis'/>")
    parts.append(f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + inner_h}' class='axis'/>")
    parts.append(f"<polyline fill='none' stroke='#1f6f5b' stroke-width='2.4' points='{points}'/>")
    for i in range(0, len(rows), step):
        value = ys[i]
        label = rows[i].get("line_start", i)
        parts.append(f"<circle cx='{sx(i):.1f}' cy='{sy(value):.1f}' r='2.2' fill='#1f6f5b'><title>line/bin {esc(label)}: {value:.1f}{suffix}</title></circle>")
    parts.append(f"<text x='{pad_l}' y='{pad_t - 6}' class='tiny'>{y_max:.1f}{esc(suffix)}</text>")
    parts.append(f"<text x='{pad_l}' y='{pad_t + inner_h + 14}' class='tiny'>{y_min:.1f}{esc(suffix)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def build_flow_svg(flow_summary: dict) -> str:
    flows = flow_summary.get("flows", {})
    total = int(flow_summary.get("total_chains", 0) or 0)
    window = int(flow_summary.get("window_groups", 0) or 0)
    if not flows or total <= 0:
        return "<div class='empty'>No flow data.</div>"
    counts = {
        "zero_adv->zero_adv": int(flows.get("zero_adv->zero_adv", {}).get("chains", 0) or 0),
        "zero_adv->mixed_adv": int(flows.get("zero_adv->mixed_adv", {}).get("chains", 0) or 0),
        "mixed_adv->mixed_adv": int(flows.get("mixed_adv->mixed_adv", {}).get("chains", 0) or 0),
        "mixed_adv->zero_adv": int(flows.get("mixed_adv->zero_adv", {}).get("chains", 0) or 0),
    }
    start_zero = counts["zero_adv->zero_adv"] + counts["zero_adv->mixed_adv"]
    start_mixed = counts["mixed_adv->mixed_adv"] + counts["mixed_adv->zero_adv"]
    width = 980
    height = 300
    left_x = 80
    right_x = 720
    top_y = 72
    box_w = 190
    box_h = 62
    gap_y = 70

    def label(count: int) -> str:
        return f"{count:,} ({pct(count, total):.1f}%)"

    def flow_path(y1: float, y2: float, count: int, color: str) -> str:
        stroke = max(3.0, 34.0 * count / max(total, 1))
        return (
            f"<path d='M {left_x + box_w} {y1:.1f} C {left_x + box_w + 170} {y1:.1f}, "
            f"{right_x - 170} {y2:.1f}, {right_x} {y2:.1f}' "
            f"fill='none' stroke='{color}' stroke-width='{stroke:.1f}' stroke-linecap='round' opacity='.72'>"
            f"<title>{count:,} chains ({pct(count, total):.1f}%)</title></path>"
        )

    parts = [f"<svg viewBox='0 0 {width} {height}'><title>Zero-advantage flow</title>"]
    parts.append(f"<text x='70' y='28' class='chart-title'>First {window} comparison group(s) to last {window} comparison group(s)</text>")
    parts.append(flow_path(top_y + box_h / 2 - 10, top_y + box_h / 2 - 10, counts["zero_adv->zero_adv"], "#ba7d1c"))
    parts.append(flow_path(top_y + box_h / 2 + 14, top_y + gap_y + box_h / 2 + 14, counts["zero_adv->mixed_adv"], "#207a5b"))
    parts.append(flow_path(top_y + gap_y + box_h / 2 - 14, top_y + box_h / 2 + 14, counts["mixed_adv->zero_adv"], "#ba7d1c"))
    parts.append(flow_path(top_y + gap_y + box_h / 2 + 10, top_y + gap_y + box_h / 2 + 10, counts["mixed_adv->mixed_adv"], "#207a5b"))
    boxes = [
        (left_x, top_y, "Initially zero-advantage", label(start_zero), "#f3d9c6"),
        (left_x, top_y + gap_y, "Initially mixed-advantage", label(start_mixed), "#d7efe3"),
        (right_x, top_y, "Finally zero-advantage", label(counts["zero_adv->zero_adv"] + counts["mixed_adv->zero_adv"]), "#f3d9c6"),
        (right_x, top_y + gap_y, "Finally mixed-advantage", label(counts["zero_adv->mixed_adv"] + counts["mixed_adv->mixed_adv"]), "#d7efe3"),
    ]
    for x, y, title, value, fill in boxes:
        parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='14' fill='{fill}' stroke='#b8ab98'/>")
        parts.append(f"<text x='{x + 14}' y='{y + 25}' class='flow-label'>{esc(title)}</text>")
        parts.append(f"<text x='{x + 14}' y='{y + 48}' class='flow-num'>{esc(value)}</text>")
    flow_labels = [
        (365, top_y - 12, "zero -> zero", counts["zero_adv->zero_adv"]),
        (405, top_y + 58, "zero -> mixed", counts["zero_adv->mixed_adv"]),
        (397, top_y + gap_y + 4, "mixed -> zero", counts["mixed_adv->zero_adv"]),
        (365, top_y + gap_y + 90, "mixed -> mixed", counts["mixed_adv->mixed_adv"]),
    ]
    for x, y, text, count in flow_labels:
        parts.append(f"<text x='{x}' y='{y}' class='tiny'>{esc(text)}: {label(count)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def table(headers: list[str], rows: list[list[object]], limit: int = 250) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>")
    more = f"<p class='muted'>Showing {limit} of {len(rows)} rows.</p>" if len(rows) > limit else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def build_html(data: dict) -> str:
    run_sections = []
    for run in data["runs"]:
        model_rows = [
            {"label": version, "nonzero": stats["nonzero_samples"], "zero": stats["samples"] - stats["nonzero_samples"]}
            for version, stats in run["model_versions"].items()
        ]
        never_rows = [
            [
                row["task_id"], row["chain_id"], row["company"], row["pattern"], row["chain_length"], row["samples"], row["groups"],
                f"{row['max_outcome_reward']:.3f}", f"{row['mean_outcome_reward']:.3f}", row["max_prefix_correct_hops"], row["problem_id"],
            ]
            for row in run["never_nonzero_chains"]
        ]
        became_rows = [
            [
                row["task_id"], row["chain_id"], row["company"], row["pattern"], row["samples"], row["groups"],
                row["samples_before_first_nonzero"], row["zero_groups_before_first_nonzero"], row["first_nonzero_line"],
                f"{row['nonzero_fraction']:.1f}%", f"{row['max_outcome_reward']:.3f}", row["problem_id"],
            ]
            for row in run["became_nonzero_chains"]
        ]
        probe_rows = [
            [
                row["task_id"], row["chain_id"], row["company"], row["pattern"], row["actor_samples"], row["trace_files"],
                row["traces_with_gold_retrieved"], row["traces_with_gold_read"], row["traces_with_any_hop_accepted"], row["problem_id"],
            ]
            for row in run.get("targeted_trace_probe", [])
        ]
        flow_rows = []
        for flow_key, flow in sorted((run.get("flow_summary") or {}).get("flows", {}).items()):
            examples = ", ".join(f"{item['task_id']}:{item['chain_id']}" for item in flow.get("examples", [])[:5])
            flow_rows.append(
                [
                    flow_key,
                    flow.get("from_label", ""),
                    flow.get("to_label", ""),
                    flow.get("chains", 0),
                    f"{flow.get('chain_fraction', 0.0):.1f}%",
                    flow.get("samples", 0),
                    examples,
                ]
            )
        warnings = "".join(f"<li>{esc(warning)}</li>" for warning in run["warnings"])
        latest = run.get("latest_success_distribution", {})
        run_sections.append(
            f"""
<section class='run'>
  <h2>{esc(run['label'])}</h2>
  <p class='path'>{esc(run['path'])}</p>
  <div class='cards'>
    <div class='card'><div class='num'>{run['actor_files']:,}</div><div>actor files scanned</div></div>
    <div class='card'><div class='num'>{run['actor_bytes'] / (1024 ** 3):.1f}GB</div><div>actor bytes scanned</div></div>
    <div class='card'><div class='num'>{run['actor_lines']:,}</div><div>actor lines scanned</div></div>
    <div class='card'><div class='num'>{run['actor_samples']:,}</div><div>actor samples scanned</div></div>
    <div class='card'><div class='num'>{run['kept_from_log']:,}</div><div>preprocessor kept</div></div>
    <div class='card'><div class='num'>{run['filtered_from_log']:,}</div><div>preprocessor filtered</div></div>
    <div class='card'><div class='num'>{run['filtered_fraction']:.1f}%</div><div>filtered fraction</div></div>
    <div class='card'><div class='num'>{run['nonzero_fraction']:.1f}%</div><div>actor samples with non-zero LOO advantage</div></div>
    <div class='card'><div class='num'>{run['nonzero_group_keys']:,}/{run['group_keys']:,}</div><div>mixed reward groups</div></div>
    <div class='card'><div class='num'>{len(run['never_nonzero_chains']):,}</div><div>chains never non-zero in actor scan</div></div>
    <div class='card'><div class='num'>{100 * float(latest.get('never_success', 0.0)):.1f}%</div><div>latest groups never successful</div></div>
    <div class='card'><div class='num'>{100 * float(latest.get('sometimes_success', 0.0)):.1f}%</div><div>latest groups sometimes successful</div></div>
  </div>
  {'<ul class="warnings">' + warnings + '</ul>' if warnings else ''}
  <div class='panel flow-panel'>{build_flow_svg(run.get('flow_summary') or {})}</div>
  <details open><summary>Flow counts</summary>
    <p>The flow uses the first and last configured comparison-group windows for each chain. "Zero advantage" means all sibling rewards were identical in that window, so the preprocessor has no directional RL signal for that chain. "Mixed advantage" means at least one sibling reward differed.</p>
    {table(['flow', 'from', 'to', 'chains', 'chain fraction', 'actor samples', 'examples'], flow_rows)}
  </details>
  <div class='grid'>
    <div class='panel'>{build_svg_bar(model_rows, ['nonzero', 'zero'], ['non-zero LOO advantage', 'zero LOO advantage'], ['#207a5b', '#d8d1c4'], 'Actor advantage coverage by model version')}</div>
    <div class='panel'>{build_svg_line(run['timeline_bins'], 'nonzero_fraction', 'Non-zero actor advantage fraction over actor-stream time')}</div>
    <div class='panel'>{build_svg_line(run['filter_events'], 'filtered_fraction', 'Preprocessor filtered-out fraction over time')}</div>
    <div class='panel'>{build_svg_line(run['stats_events'], 'never_success_pct', 'Stats stream: groups with no successful rollout', '%')}</div>
    <div class='panel'>{build_svg_line(run['stats_events'], 'sometimes_success_pct', 'Stats stream: groups with mixed success', '%')}</div>
    <div class='panel'>{build_svg_line(run['stats_events'], 'samples_per_second', 'Stats stream: sample throughput', ' samples/s')}</div>
  </div>
  <details open><summary>Chains that start zero-advantage then become non-zero</summary>
    <p>Sorted by how many samples they spent at zero before the first mixed-reward group. These are rows where training eventually got a useful within-group comparison.</p>
    {table(['task', 'chain', 'company', 'pattern', 'samples', 'groups', 'samples before first non-zero', 'zero groups before first non-zero', 'first non-zero actor line', 'non-zero sample fraction', 'max outcome', 'problem'], became_rows)}
  </details>
  <details open><summary>Chains with actor samples but no non-zero advantage</summary>
    <p>Every scanned comparison group for these rows had identical rewards, so these rows produced no directional RL signal in the actor scan.</p>
    {table(['task', 'chain', 'company', 'pattern', 'len', 'samples', 'groups', 'max outcome', 'mean outcome', 'max prefix correct', 'problem'], never_rows)}
  </details>
  <details open><summary>Targeted trace probe for top never-nonzero chains</summary>
    <p>For the highest-volume never-nonzero rows, this uses <code>rg</code> over trace files to check whether any rollout ever retrieved/read a gold parent or accepted any hop.</p>
    {table(['task', 'chain', 'company', 'pattern', 'actor samples', 'trace files', 'traces gold retrieved', 'traces gold read', 'traces any hop accepted', 'problem'], probe_rows)}
  </details>
</section>
"""
        )
    top_rows = [{"label": run["label"], "kept": run["kept_from_log"], "filtered": run["filtered_from_log"]} for run in data["runs"]]
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Privacy HopQA training filter viewer</title>
<style>
:root {{ --ink:#1d2621; --muted:#66706a; --paper:#faf7ef; --panel:#fffdf7; --line:#d7ccba; --green:#207a5b; --amber:#ba7d1c; }}
body {{ margin:0; font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:var(--ink); background:linear-gradient(145deg,#f7f0df,#edf4ee); }}
header {{ padding:32px 42px 18px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:32px; }}
main {{ padding:24px 42px 60px; }}
p {{ line-height:1.45; }}
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
.flow-label {{ fill:var(--ink); font-weight:800; font-size:13px; }}
.flow-num {{ fill:var(--ink); font-weight:700; font-size:16px; }}
.flow-panel {{ margin:18px 0; }}
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
<p>This viewer scans actor streams without decoding token arrays. A comparison group has non-zero LOO advantage exactly when sibling rewards differ; if all rewards in the group are identical, the samples carry no directional RL signal.</p>
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
    summaries: list[RunSummary] = []
    for label, path, scan_actor_by_default in DEFAULT_RUNS:
        summary = RunSummary(label=label, path=str(path), actor_scanned=scan_actor_by_default or args.scan_all_actor)
        if not path.exists():
            summary.warnings.append("Run directory does not exist")
            summaries.append(summary)
            continue
        events, kept, filtered = parse_preprocess_log(path)
        summary.filter_events = events
        summary.kept_from_log = kept
        summary.filtered_from_log = filtered
        summary.stats_kept, summary.stats_filtered = parse_preprocessor_stats(path)
        summary.stats_events, summary.latest_success_distribution = parse_stats_stream(path)
        if summary.actor_scanned:
            groups, actor_stats, warnings = scan_actor_groups(label, path, args.max_actor_lines)
            summary.warnings.extend(warnings)
            summary.actor_files = int(actor_stats["actor_files"])
            summary.actor_bytes = int(actor_stats["actor_bytes"])
            summary.actor_lines = int(actor_stats["actor_lines"])
            aggregate_groups(summary, groups, args.line_bin_size, args.flow_window_groups)
            summary.targeted_trace_probe = probe_top_never_nonzero_traces(path, summary.never_nonzero_chains, args.trace_probe_limit)
        else:
            summary.warnings.append("Actor stream not scanned by default; pass --scan-all-actor to include this run")
        summaries.append(summary)

    run_dicts = []
    for summary in summaries:
        row = dict(summary.__dict__)
        row["filtered_fraction"] = summary.filtered_fraction
        row["nonzero_fraction"] = summary.nonzero_fraction
        run_dicts.append(row)
    return {
        "generated_at_utc": args.generated_at,
        "max_actor_lines": args.max_actor_lines,
        "line_bin_size": args.line_bin_size,
        "trace_probe_limit": args.trace_probe_limit,
        "runs": run_dicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-actor-lines", type=int, default=0, help="Cap actor JSONL lines per scanned run; 0 scans all lines")
    parser.add_argument("--line-bin-size", type=int, default=100, help="Actor JSONL lines per timeline bin")
    parser.add_argument("--flow-window-groups", type=int, default=3, help="Comparison groups per chain used for initial/final flow status")
    parser.add_argument("--trace-probe-limit", type=int, default=12, help="Run rg probes over trace files for this many never-nonzero chains")
    parser.add_argument("--scan-all-actor", action="store_true", help="Also scan older comparison runs' actor streams")
    parser.add_argument("--generated-at", default="2026-05-15")
    args = parser.parse_args()

    data = build(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "viewer_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (args.out_dir / "index.html").write_text(build_html(data), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'index.html'}")
    for run in data["runs"]:
        print(
            f"{run['label']}: kept={run['kept_from_log']} filtered={run['filtered_from_log']} "
            f"filtered_frac={run['filtered_fraction']:.1f}% actor_files={run['actor_files']} "
            f"actor_lines={run['actor_lines']} actor={run['actor_samples']} "
            f"nonzero_adv={run['nonzero_samples']} ({run['nonzero_fraction']:.1f}%) "
            f"never_nonzero_chains={len(run['never_nonzero_chains'])} "
            f"became_nonzero_chains={len(run['became_nonzero_chains'])}"
        )


if __name__ == "__main__":
    main()
