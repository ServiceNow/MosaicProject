#!/usr/bin/env python3
"""Build a retrieval debugging viewer for Privacy HopQA training traces.

This viewer is diagnostic only. It samples current-model hops across difficulty
buckets, summarizes the model's parallel retrieval behavior, and compares a few
query formulations against the current helper service. It does not change
retrieval behavior or dataset contents.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
import html
import json
import math
import random
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122"
)
DEFAULT_DATASET = (
    REPO
    / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO / "runtime_analysis/privacy_hopqa_retrieval_debug_2026-05-14"
DEFAULT_HELPER_URL = "http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30ps20260513173046:8012"
FULL_DOCS_DIR = Path("/home/toolkit/full_docs")
RANDOM_SEED = 20260514

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "over",
    "reported",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def compact(value: Any, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def pct(num: float, den: float, digits: int = 1) -> str:
    return f"{100.0 * num / den:.{digits}f}%" if den else "n/a"


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def query_terms(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens(text):
        if token in STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def parent_doc_id(doc_id: Any) -> str:
    text = "" if doc_id is None else str(doc_id)
    return text.split("#", 1)[0]


def hit_parent(hit: dict[str, Any]) -> str:
    return parent_doc_id(hit.get("parent_doc_id") or hit.get("doc_id") or hit.get("docid") or hit.get("id") or "")


def hit_title(hit: dict[str, Any]) -> str:
    return str(hit.get("title") or hit.get("source_title") or hit.get("url") or hit.get("docid") or "")


def hit_snippet(hit: dict[str, Any]) -> str:
    return str(hit.get("snippet") or hit.get("text") or hit.get("content") or "")


def hit_score(hit: dict[str, Any]) -> str:
    for key in ("score", "retrieval_score", "dense_score", "similarity", "similarity_score"):
        value = hit.get(key)
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
    return ""


def normalize_for_substring(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("\ufb01", "fi").replace("\ufb02", "fl")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@lru_cache(maxsize=4096)
def gold_doc_local_hints(path_text: str) -> list[str]:
    path = Path(path_text)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    hints: list[str] = []
    for line in text.splitlines()[:40]:
        clean = normalize_for_substring(line)
        if len(clean) >= 18:
            hints.append(clean)
            if len(hints) >= 3:
                break
    return hints


def gold_matches_hit(hop: dict[str, Any], hit: dict[str, Any]) -> bool:
    gold_id = str(hop.get("gold_doc_id") or hop.get("doc_id") or "")
    gold_ref = str(hop.get("gold_doc_ref") or hop.get("doc_ref") or "")
    parent = hit_parent(hit)
    doc_id = str(hit.get("doc_id") or hit.get("docid") or "")
    if gold_id.startswith("web/"):
        return parent == gold_id or doc_id == gold_id or doc_id.startswith(gold_id + "#")
    if gold_id and (parent == gold_id or doc_id == gold_id):
        return True
    locator = str(hit.get("locator") or hit.get("path") or "")
    title = hit_title(hit)
    gold_name = Path(gold_ref).name
    if gold_name and (locator.endswith(gold_name) or title == gold_name):
        return True
    content = normalize_for_substring(hit_snippet(hit))
    if content:
        for hint in gold_doc_local_hints(gold_ref):
            if hint and hint in content:
                return True
    return False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    hops: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            chain_id = str(row.get("chain_id") or row.get("problem_id") or "")
            for hop in row.get("hops") or []:
                hop_number = int(hop.get("hop_number") or len(hops) + 1)
                key = (task_id, chain_id, hop_number)
                hops[key] = {
                    "task_id": task_id,
                    "chain_id": chain_id,
                    "company": row.get("company", ""),
                    "pattern": row.get("pattern", ""),
                    "chain_length": row.get("n_hops") or len(row.get("hops") or []),
                    "hop_number": hop_number,
                    "hop_type": hop.get("hop_type", ""),
                    "question": hop.get("question") or hop.get("question_templated") or "",
                    "question_templated": hop.get("question_templated") or hop.get("question") or "",
                    "canonical_answer": hop.get("answer", ""),
                    "accepted_answer_variants": list(hop.get("accepted_answer_variants") or []),
                    "gold_doc_id": hop.get("doc_id", ""),
                    "gold_doc_ref": hop.get("doc_ref", ""),
                    "supporting_span": hop.get("supporting_span", ""),
                }
    return hops


def iter_trace_paths(run_root: Path) -> list[Path]:
    runs_dir = run_root / "privacy_hopqa_runs"
    if not runs_dir.exists():
        return []
    out: list[Path] = []
    for child in runs_dir.iterdir():
        trace = child / "trace.json"
        if trace.is_file():
            out.append(trace)
    out.sort(key=lambda path: path.parent.name)
    return out


def reservoir(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    rng = random.Random(seed)
    sample: list[Path] = []
    for idx, path in enumerate(paths):
        if idx < limit:
            sample.append(path)
            continue
        repl = rng.randint(0, idx)
        if repl < limit:
            sample[repl] = path
    sample.sort(key=lambda path: path.parent.name)
    return sample


@dataclass
class HopStats:
    task_id: str
    chain_id: str
    hop_number: int
    hop_type: str = ""
    company: str = ""
    pattern: str = ""
    chain_length: int = 0
    question: str = ""
    canonical_answer: str = ""
    gold_doc_id: str = ""
    gold_doc_ref: str = ""
    observed: int = 0
    attempted: int = 0
    correct: int = 0
    positive_reader: int = 0
    recorded_gold_retrieved: int = 0
    recorded_gold_read: int = 0
    query_counts: Counter[str] = field(default_factory=Counter)
    expected_query_counts: Counter[str] = field(default_factory=Counter)
    top_parent_counts: Counter[str] = field(default_factory=Counter)
    parallel_groups: int = 0
    parallel_pairs: int = 0
    parallel_jaccard_sum: float = 0.0
    same_top_parent_pairs: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        return self.correct / self.attempted if self.attempted else 0.0

    @property
    def positive_reader_rate(self) -> float:
        return self.positive_reader / self.attempted if self.attempted else 0.0

    @property
    def recorded_gold_retrieved_rate(self) -> float:
        return self.recorded_gold_retrieved / self.attempted if self.attempted else 0.0

    @property
    def recorded_gold_read_rate(self) -> float:
        return self.recorded_gold_read / self.attempted if self.attempted else 0.0

    @property
    def mean_parallel_jaccard(self) -> float | None:
        if not self.parallel_pairs:
            return None
        return self.parallel_jaccard_sum / self.parallel_pairs

    @property
    def same_top_parent_rate(self) -> float | None:
        if not self.parallel_pairs:
            return None
        return self.same_top_parent_pairs / self.parallel_pairs


def difficulty(stats: HopStats) -> str:
    if stats.attempted < 3:
        return "low-sample"
    if stats.correct_rate >= 0.70:
        return "easy"
    if stats.correct_rate >= 0.25:
        return "medium"
    if stats.positive_reader_rate >= 0.35:
        return "hard-answer"
    return "hard-retrieval"


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def summarize_trace_into_stats(
    trace_path: Path,
    dataset_hops: dict[tuple[str, str, int], dict[str, Any]],
    stats_by_key: dict[tuple[str, str, int], HopStats],
) -> None:
    data = read_json(trace_path)
    summary = data.get("summary") or {}
    task_id = str(summary.get("task_id") or "")
    chain_id = str(summary.get("chain_id") or "")
    actions_by_id = {
        str(action.get("id")): action for action in data.get("actions") or [] if action.get("id") is not None
    }
    for hop in data.get("hop_states") or []:
        hop_number = int(hop.get("hop_number") or 0)
        if hop_number <= 0:
            continue
        key = (task_id, chain_id, hop_number)
        meta = dataset_hops.get(key, {})
        item = stats_by_key.get(key)
        if item is None:
            item = HopStats(
                task_id=task_id,
                chain_id=chain_id,
                hop_number=hop_number,
                hop_type=str(hop.get("hop_type") or meta.get("hop_type") or ""),
                company=str(meta.get("company") or ""),
                pattern=str(meta.get("pattern") or ""),
                chain_length=int(meta.get("chain_length") or 0),
                question=str(hop.get("question") or meta.get("question") or ""),
                canonical_answer=str(hop.get("canonical_answer") or meta.get("canonical_answer") or ""),
                gold_doc_id=str(hop.get("gold_doc_id") or meta.get("gold_doc_id") or ""),
                gold_doc_ref=str(hop.get("gold_doc_ref") or meta.get("gold_doc_ref") or ""),
            )
            stats_by_key[key] = item
        item.observed += 1
        attempted = int(hop.get("attempts") or 0) > 0
        if not attempted:
            continue
        item.attempted += 1
        if bool(hop.get("matched_accepted_variant")):
            item.correct += 1
        if any(bool(result.get("can_answer")) for result in hop.get("reader_results") or []):
            item.positive_reader += 1
        if bool(hop.get("gold_parent_retrieved")):
            item.recorded_gold_retrieved += 1
        if bool(hop.get("gold_parent_read")):
            item.recorded_gold_read += 1

        histories = list(hop.get("search_history") or [])
        expected_type = "web_search" if item.hop_type == "W" else "local_document_search"
        by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
        compact_history: list[dict[str, Any]] = []
        for event in histories:
            query = str(event.get("query") or "").strip()
            action_type = str(event.get("type") or "")
            if query:
                item.query_counts[query] += 1
                if action_type == expected_type:
                    item.expected_query_counts[query] += 1
            parents = [parent_doc_id(parent) for parent in event.get("top_parent_doc_ids") or []]
            for parent in parents[:5]:
                if parent:
                    item.top_parent_counts[parent] += 1
            iteration = int(event.get("iteration") or 0)
            by_iteration[iteration].append(event)
            compact_history.append(
                {
                    "iteration": iteration,
                    "type": action_type,
                    "query": query,
                    "top_parent_doc_ids": parents[:5],
                    "result_count": event.get("result_count"),
                }
            )

        for events in by_iteration.values():
            if len(events) < 2:
                continue
            item.parallel_groups += 1
            for idx, left in enumerate(events):
                left_parents = {parent_doc_id(parent) for parent in left.get("top_parent_doc_ids") or [] if parent}
                left_top = next(iter(left_parents), "")
                for right in events[idx + 1 :]:
                    right_parents = {
                        parent_doc_id(parent) for parent in right.get("top_parent_doc_ids") or [] if parent
                    }
                    right_top = next(iter(right_parents), "")
                    item.parallel_pairs += 1
                    item.parallel_jaccard_sum += jaccard(left_parents, right_parents)
                    if left_top and left_top == right_top:
                        item.same_top_parent_pairs += 1

        should_keep_example = len(item.examples) < 4
        if should_keep_example:
            action_examples: list[dict[str, Any]] = []
            for event in histories[:8]:
                action = actions_by_id.get(str(event.get("action_id") or ""))
                results = []
                if action:
                    actual = action.get("actual_output") or {}
                    results = actual.get("results") or []
                action_examples.append(
                    {
                        "iteration": event.get("iteration"),
                        "type": event.get("type"),
                        "query": event.get("query"),
                        "top_results": [
                            {
                                "doc_id": result.get("doc_id"),
                                "parent_doc_id": result.get("parent_doc_id") or parent_doc_id(result.get("doc_id")),
                                "title": compact(hit_title(result), 120),
                                "snippet": compact(hit_snippet(result), 220),
                            }
                            for result in results[:5]
                        ],
                    }
                )
            item.examples.append(
                {
                    "rollout": trace_path.parent.name,
                    "status": hop.get("status"),
                    "answer": hop.get("answer"),
                    "matched": bool(hop.get("matched_accepted_variant")),
                    "attempts": hop.get("attempts"),
                    "resolution_reason": compact(hop.get("resolution_reason"), 240),
                    "history": compact_history[:10],
                    "actions": action_examples,
                }
            )


def select_hops(stats: list[HopStats], max_hops: int) -> list[HopStats]:
    rng = random.Random(RANDOM_SEED)
    eligible = [item for item in stats if item.attempted >= 3]
    buckets: dict[str, list[HopStats]] = defaultdict(list)
    for item in eligible:
        buckets[difficulty(item)].append(item)
    for values in buckets.values():
        values.sort(key=lambda item: (item.task_id, item.chain_id, item.hop_number))
        rng.shuffle(values)

    order = [
        ("hard-retrieval", "W"),
        ("hard-answer", "W"),
        ("medium", "W"),
        ("easy", "W"),
        ("hard-retrieval", "L"),
        ("hard-answer", "L"),
        ("medium", "L"),
        ("easy", "L"),
    ]
    selected: list[HopStats] = []
    seen: set[tuple[str, str, int]] = set()
    per_bucket = max(2, max_hops // max(1, len(order)))
    for bucket_name, hop_type in order:
        candidates = [item for item in buckets.get(bucket_name, []) if item.hop_type == hop_type]
        for item in candidates[:per_bucket]:
            key = (item.task_id, item.chain_id, item.hop_number)
            if key not in seen:
                seen.add(key)
                selected.append(item)
    if len(selected) < max_hops:
        remaining = [item for item in eligible if (item.task_id, item.chain_id, item.hop_number) not in seen]
        remaining.sort(key=lambda item: (difficulty(item), item.hop_type, item.task_id, item.chain_id, item.hop_number))
        rng.shuffle(remaining)
        selected.extend(remaining[: max_hops - len(selected)])
    selected = selected[:max_hops]
    selected.sort(key=lambda item: (difficulty(item), item.hop_type, item.task_id, item.chain_id, item.hop_number))
    return selected


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str, timeout_s: float) -> dict[str, Any]:
    with urlopen(base_url.rstrip("/") + path, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def helper_search(
    helper_url: str,
    hop: HopStats,
    query: str,
    k: int,
    timeout_s: float,
) -> dict[str, Any]:
    if hop.hop_type == "W":
        payload = {"query": query, "task_id": hop.task_id, "k": int(k)}
        path = "/browsecomp/search"
    else:
        payload = {"task_id": hop.task_id, "query": query, "k": int(k), "threshold": 0.0}
        path = "/local/search"
    try:
        response = post_json(helper_url, path, payload, timeout_s)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"error": str(exc), "hits": []}
    hits = response.get("hits") or []
    rank = None
    for idx, hit in enumerate(hits, start=1):
        if gold_matches_hit(
            {
                "gold_doc_id": hop.gold_doc_id,
                "gold_doc_ref": hop.gold_doc_ref,
            },
            hit,
        ):
            rank = idx
            break
    return {
        "rank": rank,
        "hits": [
            {
                "rank": idx,
                "doc_id": hit.get("doc_id") or hit.get("docid"),
                "parent_doc_id": hit.get("parent_doc_id") or parent_doc_id(hit.get("doc_id") or hit.get("docid")),
                "title": compact(hit_title(hit), 120),
                "score": hit_score(hit),
                "snippet": compact(hit_snippet(hit), 220),
                "is_gold": gold_matches_hit(
                    {
                        "gold_doc_id": hop.gold_doc_id,
                        "gold_doc_ref": hop.gold_doc_ref,
                    },
                    hit,
                ),
            }
            for idx, hit in enumerate(hits[:8], start=1)
        ],
        "error": None,
    }


def doc_title_hint(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        return path.name
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[:80]:
        text = line.strip()
        if not text:
            continue
        if text.lower().startswith(("http://", "https://")):
            continue
        if len(text) > 12:
            return compact(text, 120)
    return path.name


def build_query_variants(hop: HopStats) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(label: str, query: str) -> None:
        clean = re.sub(r"\s+", " ", query).strip()
        if not clean:
            return
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            variants.append({"label": label, "query": clean})

    for query, count in hop.expected_query_counts.most_common(3):
        add(f"model query ({count}x in sample)", query)
    add("literal materialized question", hop.question)
    terms = query_terms(hop.question)
    if terms:
        add("content-word query", " ".join(terms[:14]))
    numeric_terms = [term for term in terms if re.search(r"\d", term)]
    caps = re.findall(r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){0,4}", hop.question)
    entity_bits = [bit.strip(" ?.,") for bit in caps if bit.strip(" ?.,")]
    if numeric_terms or entity_bits:
        add("entity/numeric query", " ".join(entity_bits[:4] + numeric_terms[:6]))
    if hop.hop_type == "W":
        hint = doc_title_hint(hop.gold_doc_ref)
        add("gold-source title diagnostic", f"{hint} {' '.join(terms[:8])}")
    return variants[:7]


def build_web_lexical_index(queries: list[str]) -> dict[str, Any]:
    all_terms = sorted({term for query in queries for term in query_terms(query)})
    if not all_terms or not FULL_DOCS_DIR.exists():
        return {"available": False, "reason": "no query terms or full_docs missing"}
    wanted = set(all_terms)
    doc_counts: dict[str, Counter[str]] = {}
    doc_lengths: dict[str, int] = {}
    df: Counter[str] = Counter()
    for path in sorted(FULL_DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        toks = tokens(text)
        doc_lengths[f"web/{path.stem}"] = len(toks)
        counts = Counter(token for token in toks if token in wanted)
        if counts:
            doc_id = f"web/{path.stem}"
            doc_counts[doc_id] = counts
            for term in counts:
                df[term] += 1
    avg_len = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1.0
    return {
        "available": True,
        "doc_counts": doc_counts,
        "doc_lengths": doc_lengths,
        "df": df,
        "avg_len": avg_len,
        "n_docs": len(doc_lengths),
    }


def lexical_rank(index: dict[str, Any], query: str, gold_doc_id: str) -> dict[str, Any]:
    if not index.get("available"):
        return {"rank": None, "top_docs": [], "score": None}
    terms = query_terms(query)
    if not terms:
        return {"rank": None, "top_docs": [], "score": None}
    doc_counts: dict[str, Counter[str]] = index["doc_counts"]
    doc_lengths: dict[str, int] = index["doc_lengths"]
    df: Counter[str] = index["df"]
    avg_len = float(index["avg_len"])
    n_docs = int(index["n_docs"])
    k1 = 1.2
    b = 0.75
    scores: list[tuple[float, str]] = []
    for doc_id, counts in doc_counts.items():
        score = 0.0
        doc_len = max(1, doc_lengths.get(doc_id, 1))
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        if score > 0:
            scores.append((score, doc_id))
    scores.sort(reverse=True)
    gold_parent = parent_doc_id(gold_doc_id)
    rank = None
    gold_score = None
    for idx, (score, doc_id) in enumerate(scores, start=1):
        if doc_id == gold_parent:
            rank = idx
            gold_score = score
            break
    return {
        "rank": rank,
        "score": gold_score,
        "top_docs": [{"rank": idx, "doc_id": doc_id, "score": score} for idx, (score, doc_id) in enumerate(scores[:8], 1)],
    }


def summarize_stepfun_artifacts() -> dict[str, Any]:
    features = REPO / "runtime_analysis/privacy_hopqa_initial_state_temp07_r3_v2/privacy_leakage_features.jsonl"
    counts: Counter[str] = Counter()
    if features.is_file():
        with features.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts[str(row.get("model") or "unknown")] += 1
    model_runs_root = Path(
        "/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_initial_state_eval_20260507_temp07_r3_v2/model_runs"
    )
    model_runs = []
    if model_runs_root.is_dir():
        model_runs = [path.name for path in sorted(model_runs_root.iterdir()) if path.is_dir()]
    leakage_rows = []
    for summary_path in sorted(REPO.glob("runtime_analysis/privacy_hopqa_leakage_v2_stepfun_*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = data.get("summary") or data
        leakage_rows.append(
            {
                "name": summary_path.parent.name,
                "rows": summary.get("rows") or summary.get("detail_rows_total") or summary.get("prefix_rows"),
                "reward_positive": summary.get("reward_positive") or summary.get("reward_positive_rows"),
                "intent_leaked": summary.get("intent_leaked"),
                "answer_leaked": summary.get("answer_leaked"),
                "full_info_leaked": summary.get("full_info_leaked"),
                "mean_web_queries": summary.get("mean_web_queries"),
                "mean_hop_accuracy": summary.get("mean_hop_accuracy"),
            }
        )
    return {
        "viewer": str(REPO / "runtime_analysis/privacy_hopqa_initial_state_temp07_r3_v2/index.html"),
        "features": str(features),
        "feature_counts_by_model": dict(counts),
        "model_runs_root": str(model_runs_root),
        "model_runs": model_runs[:20],
        "stepfun_leakage_summaries": leakage_rows,
    }


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "<div class='muted'>No hits returned.</div>"
    rows = []
    for hit in hits:
        marker = " gold" if hit.get("is_gold") else ""
        rows.append(
            [
                esc(hit.get("rank")),
                f"<code class='{marker.strip()}'>{esc(hit.get('parent_doc_id'))}</code>",
                esc(hit.get("score")),
                esc(hit.get("title")),
                esc(hit.get("snippet")),
            ]
        )
    return render_table(["rank", "parent doc", "score", "title", "snippet"], rows)


def build_html(
    output_path: Path,
    *,
    run_root: Path,
    dataset: Path,
    helper_url: str,
    helper_health: dict[str, Any] | None,
    stats: list[HopStats],
    selected: list[dict[str, Any]],
    stepfun_summary: dict[str, Any],
    trace_count: int,
    sampled_trace_count: int,
) -> None:
    bucket_counts = Counter(difficulty(item) for item in stats if item.attempted >= 3)
    hop_type_counts = Counter(item.hop_type for item in stats if item.attempted >= 3)
    mean_jaccards = [item.mean_parallel_jaccard for item in stats if item.mean_parallel_jaccard is not None]
    same_top_rates = [item.same_top_parent_rate for item in stats if item.same_top_parent_rate is not None]
    summary_rows = [
        ["all trace dirs", f"{trace_count:,}"],
        ["sampled trace dirs parsed", f"{sampled_trace_count:,}"],
        ["distinct attempted hops in sample", f"{sum(1 for item in stats if item.attempted >= 3):,}"],
        ["difficulty buckets", esc(dict(bucket_counts))],
        ["hop types", esc(dict(hop_type_counts))],
        ["mean parallel-query top-doc Jaccard", fmt_float(sum(mean_jaccards) / len(mean_jaccards) if mean_jaccards else None)],
        ["mean same top-parent rate", fmt_float(sum(same_top_rates) / len(same_top_rates) if same_top_rates else None)],
    ]
    helper_rows = []
    if helper_health:
        helper_rows = [
            ["helper URL", f"<code>{esc(helper_url)}</code>"],
            ["browsecomp docs/windows", f"{esc(helper_health.get('browsecomp_size'))} docs, max windows {esc(helper_health.get('browsecomp_max_doc_windows'))}"],
            ["curated docs", esc(helper_health.get("curated_doc_count"))],
            ["browsecomp latency", esc((helper_health.get("metrics") or {}).get("/browsecomp/search"))],
            ["local latency", esc((helper_health.get("metrics") or {}).get("/local/search"))],
        ]
    else:
        helper_rows = [["helper URL", f"<code>{esc(helper_url)}</code>"], ["health", "unavailable"]]

    cards = []
    for item in selected:
        hop = item["stats"]
        variant_rows = []
        for variant in item["variants"]:
            dense_rank = variant.get("dense", {}).get("rank")
            lexical = variant.get("lexical") or {}
            dense_text = "not in top-k" if dense_rank is None else str(dense_rank)
            lexical_rank_text = "n/a"
            if hop.hop_type == "W":
                lexical_rank_text = "not in source-pool top" if lexical.get("rank") is None else str(lexical.get("rank"))
            variant_rows.append(
                [
                    esc(variant["label"]),
                    f"<div class='query'>{esc(variant['query'])}</div>",
                    esc(dense_text),
                    esc(lexical_rank_text),
                    render_hits(variant.get("dense", {}).get("hits") or []),
                ]
            )
        query_rows = []
        for query, count in hop.expected_query_counts.most_common(8):
            query_rows.append([esc(count), f"<div class='query'>{esc(query)}</div>"])
        top_parent_rows = []
        for parent, count in hop.top_parent_counts.most_common(8):
            top_parent_rows.append([esc(count), f"<code>{esc(parent)}</code>"])
        example_blocks = []
        for example in hop.examples[:3]:
            action_rows = []
            for action in example.get("actions") or []:
                action_rows.append(
                    [
                        esc(action.get("iteration")),
                        esc(action.get("type")),
                        f"<div class='query'>{esc(action.get('query'))}</div>",
                        render_hits(
                            [
                                {
                                    "rank": idx,
                                    "parent_doc_id": result.get("parent_doc_id"),
                                    "title": result.get("title"),
                                    "snippet": result.get("snippet"),
                                    "score": "",
                                    "is_gold": result.get("parent_doc_id") == hop.gold_doc_id,
                                }
                                for idx, result in enumerate(action.get("top_results") or [], start=1)
                            ]
                        ),
                    ]
                )
            example_blocks.append(
                f"""
                <details>
                  <summary>{esc(example.get('rollout'))} - matched={esc(example.get('matched'))}, answer={esc(example.get('answer'))}, attempts={esc(example.get('attempts'))}</summary>
                  <div class="muted">{esc(example.get('resolution_reason'))}</div>
                  {render_table(["iter", "type", "query", "top returned docs"], action_rows)}
                </details>
                """
            )
        cards.append(
            f"""
            <section class="card" data-difficulty="{esc(item['difficulty'])}" data-hoptype="{esc(hop.hop_type)}">
              <h3>{esc(item['difficulty'])} / {esc(hop.hop_type)} - {esc(hop.task_id)} {esc(hop.chain_id)} hop {esc(hop.hop_number)}</h3>
              <div class="meta">company={esc(hop.company)} | pattern={esc(hop.pattern)} | attempted={hop.attempted} | correct={hop.correct} ({pct(hop.correct, hop.attempted)}) | positive-reader={pct(hop.positive_reader, hop.attempted)} | recorded gold read={pct(hop.recorded_gold_read, hop.attempted)}</div>
              <p><b>Question:</b> {esc(hop.question)}</p>
              <p><b>Gold answer:</b> {esc(hop.canonical_answer)} | <b>Gold doc:</b> <code>{esc(hop.gold_doc_id)}</code> <span class="muted">{esc(hop.gold_doc_ref)}</span></p>
              <p><b>Parallel-query redundancy:</b> same-iteration groups={hop.parallel_groups}, pairs={hop.parallel_pairs}, mean top-doc Jaccard={esc(fmt_float(hop.mean_parallel_jaccard))}, same top parent={esc(fmt_float(hop.same_top_parent_rate))}</p>
              <div class="grid">
                <div>
                  <h4>Most common model queries for this hop type</h4>
                  {render_table(["count", "query"], query_rows) if query_rows else "<div class='muted'>No queries for expected source type.</div>"}
                </div>
                <div>
                  <h4>Most repeated returned parent docs</h4>
                  {render_table(["count", "parent doc"], top_parent_rows) if top_parent_rows else "<div class='muted'>No returned docs recorded.</div>"}
                </div>
              </div>
              <h4>Dense helper and lexical diagnostics</h4>
              <div class="explain">Dense rank is from the currently running helper, top-k only. Lexical rank is only for web hops and only within <code>/home/toolkit/full_docs</code>, so it is a source-pool diagnostic rather than the full browsecomp production corpus.</div>
              {render_table(["variant", "query", "dense gold rank", "lexical gold rank", "dense top hits"], variant_rows)}
              <h4>Rollout examples</h4>
              {''.join(example_blocks)}
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Privacy HopQA retrieval debug viewer</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #64748b;
      --line: #d7dee8;
      --paper: #fbfbf8;
      --panel: #ffffff;
      --accent: #0f766e;
      --gold: #9a6700;
    }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; color: var(--ink); background: linear-gradient(135deg, #f6f0df, #eef7f4 55%, #f9fbff); }}
    header {{ padding: 28px 34px 18px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,0.76); position: sticky; top: 0; z-index: 10; backdrop-filter: blur(10px); }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.03em; }}
    h2 {{ margin-top: 30px; }}
    h3 {{ margin: 0 0 10px; }}
    h4 {{ margin: 18px 0 8px; }}
    main {{ padding: 24px 34px 60px; max-width: 1500px; }}
    .card {{ background: rgba(255,255,255,0.92); border: 1px solid var(--line); border-radius: 16px; padding: 18px; margin: 18px 0; box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08); }}
    .meta, .muted, .explain {{ color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 13px; }}
    th, td {{ border: 1px solid var(--line); padding: 7px 8px; vertical-align: top; }}
    th {{ background: #edf6f4; text-align: left; }}
    code {{ background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 5px; padding: 1px 4px; }}
    code.gold {{ background: #fff7d6; border-color: #f5c542; color: var(--gold); font-weight: 700; }}
    .query {{ max-width: 520px; white-space: normal; }}
    details {{ margin: 10px 0; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 14px; }}
    button {{ border: 1px solid var(--line); background: white; border-radius: 999px; padding: 6px 12px; cursor: pointer; }}
    button.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
    input {{ padding: 7px 10px; min-width: 320px; border: 1px solid var(--line); border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>Privacy HopQA retrieval debug viewer</h1>
    <div class="meta">Generated {esc(datetime.now(UTC).isoformat())}. Run: <code>{esc(run_root)}</code>. Dataset: <code>{esc(dataset)}</code>.</div>
    <div class="controls">
      <button class="active" data-filter="all">all</button>
      <button data-filter="hard-retrieval">hard retrieval</button>
      <button data-filter="hard-answer">hard answer</button>
      <button data-filter="medium">medium</button>
      <button data-filter="easy">easy</button>
      <button data-hoptype="W">web only</button>
      <button data-hoptype="L">local only</button>
      <input id="search" placeholder="filter by task, chain, question, doc id">
    </div>
  </header>
  <main>
    <section class="card">
      <h2>What this viewer measures</h2>
      <p>It samples current training traces, buckets hops by observed model difficulty, and shows whether alternative query formulations would surface the gold document under the current helper. It also shows whether same-iteration parallel queries return distinct evidence or collapse onto the same top documents.</p>
      <p><b>Non-trained stage estimate:</b> from the prior stage-bucket sample, pure doc-read failures after the gold doc was read were about 0.5% of attempted hop states and about 1.6% of attempted-hop failures. So the claim is directionally right, but I would not phrase it as strictly below 1% unless the denominator includes blocked hops/all hop states.</p>
      {render_table(["metric", "value"], summary_rows)}
      <h3>Helper status</h3>
      {render_table(["field", "value"], helper_rows)}
      <h3>Broader StepFun / initial-state artifacts</h3>
      {render_table(["field", "value"], [
          ["viewer", f"<code>{esc(stepfun_summary.get('viewer'))}</code>"],
          ["features", f"<code>{esc(stepfun_summary.get('features'))}</code>"],
          ["feature rows by model", esc(stepfun_summary.get('feature_counts_by_model'))],
          ["model runs root", f"<code>{esc(stepfun_summary.get('model_runs_root'))}</code>"],
          ["model run dirs", esc(stepfun_summary.get('model_runs'))],
      ])}
      <h4>StepFun leakage/privacy evaluator runs found</h4>
      {render_table(
          ["run", "rows", "reward positive", "intent leaked", "answer leaked", "full info leaked", "mean web queries", "mean hop accuracy"],
          [
              [
                  esc(row.get("name")),
                  esc(row.get("rows")),
                  esc(row.get("reward_positive")),
                  esc(row.get("intent_leaked")),
                  esc(row.get("answer_leaked")),
                  esc(row.get("full_info_leaked")),
                  esc(row.get("mean_web_queries")),
                  esc(row.get("mean_hop_accuracy")),
              ]
              for row in stepfun_summary.get("stepfun_leakage_summaries", [])
          ],
      )}
    </section>
    {''.join(cards)}
  </main>
  <script>
    const buttons = Array.from(document.querySelectorAll('button'));
    const cards = Array.from(document.querySelectorAll('.card[data-difficulty]'));
    let difficulty = 'all';
    let hopType = 'all';
    function applyFilters() {{
      const needle = document.getElementById('search').value.toLowerCase();
      cards.forEach(card => {{
        const diffOk = difficulty === 'all' || card.dataset.difficulty === difficulty;
        const typeOk = hopType === 'all' || card.dataset.hoptype === hopType;
        const textOk = !needle || card.innerText.toLowerCase().includes(needle);
        card.style.display = diffOk && typeOk && textOk ? '' : 'none';
      }});
    }}
    buttons.forEach(button => {{
      button.addEventListener('click', () => {{
        if (button.dataset.filter) {{
          difficulty = button.dataset.filter;
          buttons.filter(b => b.dataset.filter).forEach(b => b.classList.toggle('active', b === button));
        }}
        if (button.dataset.hoptype) {{
          hopType = hopType === button.dataset.hoptype ? 'all' : button.dataset.hoptype;
          buttons.filter(b => b.dataset.hoptype).forEach(b => b.classList.toggle('active', hopType === b.dataset.hoptype));
        }}
        applyFilters();
      }});
    }});
    document.getElementById('search').addEventListener('input', applyFilters);
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--helper-url", default=DEFAULT_HELPER_URL)
    parser.add_argument("--max-traces", type=int, default=800)
    parser.add_argument("--sample-hops", type=int, default=24)
    parser.add_argument("--helper-k", type=int, default=20)
    parser.add_argument("--helper-timeout-s", type=float, default=20.0)
    parser.add_argument("--no-helper", action="store_true")
    args = parser.parse_args()

    dataset_hops = load_dataset(args.dataset)
    trace_paths = iter_trace_paths(args.run_root)
    sampled_paths = reservoir(trace_paths, args.max_traces, RANDOM_SEED)
    stats_by_key: dict[tuple[str, str, int], HopStats] = {}
    for path in sampled_paths:
        summarize_trace_into_stats(path, dataset_hops, stats_by_key)
    all_stats = list(stats_by_key.values())
    selected_stats = select_hops(all_stats, args.sample_hops)

    variants_by_key: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    web_queries: list[str] = []
    for hop in selected_stats:
        variants = build_query_variants(hop)
        variants_by_key[(hop.task_id, hop.chain_id, hop.hop_number)] = variants
        if hop.hop_type == "W":
            web_queries.extend(variant["query"] for variant in variants)

    lexical_index = build_web_lexical_index(web_queries)
    helper_health = None
    if not args.no_helper:
        try:
            helper_health = get_json(args.helper_url, "/health", args.helper_timeout_s)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            helper_health = None

    selected_payload: list[dict[str, Any]] = []
    for hop in selected_stats:
        enriched_variants = []
        for variant in variants_by_key[(hop.task_id, hop.chain_id, hop.hop_number)]:
            dense = {"rank": None, "hits": [], "error": "helper disabled"}
            if not args.no_helper and helper_health:
                dense = helper_search(args.helper_url, hop, variant["query"], args.helper_k, args.helper_timeout_s)
            lexical = None
            if hop.hop_type == "W":
                lexical = lexical_rank(lexical_index, variant["query"], hop.gold_doc_id)
            enriched_variants.append({**variant, "dense": dense, "lexical": lexical})
        selected_payload.append(
            {
                "stats": hop,
                "difficulty": difficulty(hop),
                "variants": enriched_variants,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "index.html"
    build_html(
        output_path,
        run_root=args.run_root,
        dataset=args.dataset,
        helper_url=args.helper_url,
        helper_health=helper_health,
        stats=all_stats,
        selected=selected_payload,
        stepfun_summary=summarize_stepfun_artifacts(),
        trace_count=len(trace_paths),
        sampled_trace_count=len(sampled_paths),
    )
    selected_json = []
    for item in selected_payload:
        hop = item["stats"]
        selected_json.append(
            {
                "difficulty": item["difficulty"],
                "task_id": hop.task_id,
                "chain_id": hop.chain_id,
                "hop_number": hop.hop_number,
                "hop_type": hop.hop_type,
                "company": hop.company,
                "pattern": hop.pattern,
                "question": hop.question,
                "canonical_answer": hop.canonical_answer,
                "gold_doc_id": hop.gold_doc_id,
                "gold_doc_ref": hop.gold_doc_ref,
                "attempted": hop.attempted,
                "correct": hop.correct,
                "correct_rate": hop.correct_rate,
                "positive_reader_rate": hop.positive_reader_rate,
                "recorded_gold_read_rate": hop.recorded_gold_read_rate,
                "mean_parallel_jaccard": hop.mean_parallel_jaccard,
                "same_top_parent_rate": hop.same_top_parent_rate,
                "query_variants": [
                    {
                        "label": variant["label"],
                        "query": variant["query"],
                        "dense_gold_rank": (variant.get("dense") or {}).get("rank"),
                        "lexical_gold_rank": (variant.get("lexical") or {}).get("rank")
                        if variant.get("lexical")
                        else None,
                    }
                    for variant in item["variants"]
                ],
            }
        )
    (args.out_dir / "selected_hops.json").write_text(json.dumps(selected_json, indent=2), encoding="utf-8")
    mean_jaccards = [item.mean_parallel_jaccard for item in all_stats if item.mean_parallel_jaccard is not None]
    same_top_rates = [item.same_top_parent_rate for item in all_stats if item.same_top_parent_rate is not None]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_root": str(args.run_root),
        "dataset": str(args.dataset),
        "output": str(output_path),
        "trace_count": len(trace_paths),
        "sampled_trace_count": len(sampled_paths),
        "selected_hops": len(selected_payload),
        "helper_url": args.helper_url,
        "helper_health_ok": bool(helper_health),
        "lexical_source_pool_docs": lexical_index.get("n_docs") if lexical_index.get("available") else None,
        "difficulty_counts": dict(Counter(difficulty(item) for item in all_stats if item.attempted >= 3)),
        "mean_parallel_query_top_doc_jaccard": sum(mean_jaccards) / len(mean_jaccards) if mean_jaccards else None,
        "mean_same_top_parent_rate": sum(same_top_rates) / len(same_top_rates) if same_top_rates else None,
        "selected_hops_json": str(args.out_dir / "selected_hops.json"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
