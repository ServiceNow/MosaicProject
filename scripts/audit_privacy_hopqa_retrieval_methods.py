#!/usr/bin/env python3
"""Audit dense, lexical, and fused retrieval ranks for Privacy HopQA hops.

This is an offline diagnostic. It samples hops from current training traces,
compares the live helper's dense retrieval with BM25-style lexical retrieval,
and evaluates reciprocal-rank fusion. It reports two query sets separately:

1. question_baseline: deterministic queries derived only from the materialized
   hop question. These are clean baselines.
2. actual_model_queries: retrieval queries the model actually issued in the
   sampled traces. These answer the main counterfactual: if the model made the
   same search calls, would lexical/fusion have surfaced the gold doc?

No production retrieval behavior is changed by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
import html
import json
import math
from pathlib import Path
import random
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122")
DEFAULT_DATASET = REPO / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl"
DEFAULT_HELPER_URL = "http://dns-24e3447c-506e-4b21-92df-156e18db5087-phq30ps20260513173046:8012"
DEFAULT_OUT_DIR = REPO / "runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14"
DEFAULT_LOCAL_INDEX_ROOT = Path("/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/local_indices_goldcoverage_20260519")
FULL_DOCS_DIR = Path("/home/toolkit/full_docs")
RANDOM_SEED = 20260514
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "over", "reported", "that", "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "with",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def compact(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text.replace("\ufb01", "fi").replace("\ufb02", "fl"))]


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


def normalize_pathish(value: Any) -> str:
    text = str(value or "")
    return text.replace("\\", "/")


def parent_doc_id(doc_id: Any) -> str:
    return str(doc_id or "").split("#", 1)[0]


def hit_doc_id(hit: dict[str, Any]) -> str:
    return parent_doc_id(hit.get("parent_doc_id") or hit.get("doc_id") or hit.get("docid") or hit.get("id") or "")


def hit_text(hit: dict[str, Any]) -> str:
    return str(hit.get("snippet") or hit.get("text") or hit.get("content") or "")


def hit_title(hit: dict[str, Any]) -> str:
    return str(hit.get("title") or hit.get("url") or hit.get("docid") or hit.get("source") or "")


def hit_score(hit: dict[str, Any]) -> float | None:
    for key in ("score", "similarity_score", "similarity", "dense_score", "retrieval_score"):
        value = hit.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return ">50/not found"
    if rank <= 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return ">50/not found"


def difficulty_from_rate(correct: int, attempted: int, positive_reader: int) -> str:
    if attempted < 3:
        return "low-sample"
    correct_rate = correct / attempted
    positive_rate = positive_reader / attempted
    if correct_rate >= 0.70:
        return "easy"
    if correct_rate >= 0.25:
        return "medium"
    if positive_rate >= 0.35:
        return "hard-answer"
    return "hard-retrieval"


@dataclass
class HopStats:
    task_id: str
    chain_id: str
    hop_number: int
    hop_type: str
    question: str
    answer: str
    gold_doc_id: str
    gold_doc_ref: str
    company: str
    pattern: str
    attempted: int = 0
    correct: int = 0
    positive_reader: int = 0
    model_queries: Counter[str] = field(default_factory=Counter)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.chain_id, self.hop_number)

    @property
    def difficulty(self) -> str:
        return difficulty_from_rate(self.correct, self.attempted, self.positive_reader)

    @property
    def expected_action_type(self) -> str:
        return "web_search" if self.hop_type == "W" else "local_document_search"


def load_hops(dataset_path: Path) -> dict[tuple[str, str, int], HopStats]:
    out: dict[tuple[str, str, int], HopStats] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            chain_id = str(row.get("chain_id") or row.get("problem_id") or "")
            for hop in row.get("hops") or []:
                hop_number = int(hop.get("hop_number") or 0)
                hop_type = str(hop.get("hop_type") or "")
                if hop_type not in {"L", "W"}:
                    continue
                item = HopStats(
                    task_id=task_id,
                    chain_id=chain_id,
                    hop_number=hop_number,
                    hop_type=hop_type,
                    question=str(hop.get("question") or hop.get("question_templated") or ""),
                    answer=str(hop.get("answer") or ""),
                    gold_doc_id=str(hop.get("doc_id") or ""),
                    gold_doc_ref=str(hop.get("doc_ref") or ""),
                    company=str(row.get("company") or ""),
                    pattern=str(row.get("pattern") or ""),
                )
                out[item.key] = item
    return out


def iter_trace_paths(run_root: Path) -> list[Path]:
    runs_dir = run_root / "privacy_hopqa_runs"
    if not runs_dir.exists():
        return []
    paths = [child / "trace.json" for child in runs_dir.iterdir() if (child / "trace.json").is_file()]
    paths.sort(key=lambda path: path.parent.name)
    return paths


def reservoir_sample(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    rng = random.Random(RANDOM_SEED)
    sample: list[Path] = []
    for idx, path in enumerate(paths):
        if idx < limit:
            sample.append(path)
            continue
        repl = rng.randint(0, idx)
        if repl < limit:
            sample[repl] = path
    return sample


def add_trace_stats(trace_path: Path, hops: dict[tuple[str, str, int], HopStats]) -> None:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    task_id = str(summary.get("task_id") or "")
    chain_id = str(summary.get("chain_id") or "")
    for hop_state in data.get("hop_states") or []:
        hop_number = int(hop_state.get("hop_number") or 0)
        item = hops.get((task_id, chain_id, hop_number))
        if item is None:
            continue
        if int(hop_state.get("attempts") or 0) <= 0:
            continue
        item.attempted += 1
        if bool(hop_state.get("matched_accepted_variant")):
            item.correct += 1
        if any(bool(result.get("can_answer")) for result in hop_state.get("reader_results") or []):
            item.positive_reader += 1
        for event in hop_state.get("search_history") or []:
            if event.get("type") == item.expected_action_type:
                query = str(event.get("query") or "").strip()
                if query:
                    item.model_queries[query] += 1


def select_hops(hops: list[HopStats], sample_size: int) -> list[HopStats]:
    rng = random.Random(RANDOM_SEED)
    eligible = [hop for hop in hops if hop.attempted >= 3 and hop.model_queries]
    buckets: dict[tuple[str, str], list[HopStats]] = defaultdict(list)
    for hop in eligible:
        buckets[(hop.hop_type, hop.difficulty)].append(hop)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.key)
        rng.shuffle(bucket)

    order = [
        ("W", "hard-retrieval"), ("W", "hard-answer"), ("W", "medium"), ("W", "easy"),
        ("L", "hard-retrieval"), ("L", "hard-answer"), ("L", "medium"), ("L", "easy"),
    ]
    target_each = max(1, sample_size // len(order))
    selected: list[HopStats] = []
    seen: set[tuple[str, str, int]] = set()
    for key in order:
        for hop in buckets.get(key, [])[:target_each]:
            selected.append(hop)
            seen.add(hop.key)
    if len(selected) < sample_size:
        remaining = [hop for hop in eligible if hop.key not in seen]
        remaining.sort(key=lambda item: (item.hop_type, item.difficulty, item.key))
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    selected = selected[:sample_size]
    selected.sort(key=lambda item: (item.hop_type, item.difficulty, item.task_id, item.chain_id, item.hop_number))
    return selected


def question_variants(hop: HopStats) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(label: str, query: str) -> None:
        clean = re.sub(r"\s+", " ", query).strip()
        if not clean:
            return
        key = clean.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append({"label": label, "query": clean})

    add("literal_question", hop.question)
    terms = query_terms(hop.question)
    add("content_words", " ".join(terms[:16]))
    numeric_terms = [term for term in terms if any(char.isdigit() for char in term)]
    caps = [
        match.strip(" ?.,")
        for match in re.findall(r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){0,4}", hop.question)
    ]
    add("entity_numeric", " ".join(caps[:4] + numeric_terms[:6]))
    return variants


def model_query_variants(hop: HopStats, max_queries: int) -> list[dict[str, str]]:
    return [
        {"label": f"model_observed_{count}x", "query": query, "observed_count": count}
        for query, count in hop.model_queries.most_common(max_queries)
    ]


def post_helper_search(helper_url: str, hop: HopStats, query: str, k: int, timeout_s: float) -> dict[str, Any]:
    if hop.hop_type == "W":
        path = "/browsecomp/search"
        payload = {"query": query, "task_id": hop.task_id, "k": int(k)}
    else:
        path = "/local/search"
        payload = {"query": query, "task_id": hop.task_id, "k": int(k), "threshold": 0.0}
    request = Request(
        helper_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def local_hit_matches_gold(hop: HopStats, hit: dict[str, Any]) -> bool:
    gold_ref = normalize_pathish(hop.gold_doc_ref)
    gold_name = Path(gold_ref).name
    locator = normalize_pathish(hit.get("locator") or hit.get("path") or "")
    source = normalize_pathish(hit.get("source") or "")
    title = normalize_pathish(hit_title(hit))
    if gold_name and (locator.endswith(gold_name) or source.endswith(gold_name) or title == gold_name):
        return True
    gold_parts = [part for part in normalize_pathish(gold_ref).split("/") if part]
    if len(gold_parts) >= 2:
        suffix = "/".join(gold_parts[-2:])
        if suffix and (suffix in locator or suffix in source or suffix in title):
            return True
    return False


def hit_matches_gold(hop: HopStats, hit: dict[str, Any]) -> bool:
    doc_id = hit_doc_id(hit)
    if hop.hop_type == "W":
        return doc_id == hop.gold_doc_id
    if hop.gold_doc_id and doc_id == hop.gold_doc_id:
        return True
    return local_hit_matches_gold(hop, hit)


def dense_result(helper_url: str, hop: HopStats, variant: dict[str, Any], k: int, timeout_s: float) -> dict[str, Any]:
    try:
        response = post_helper_search(helper_url, hop, str(variant["query"]), k, timeout_s)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"rank": None, "all_hits": [], "hits": [], "error": str(exc)}
    hits = response.get("hits") or []
    all_hits = []
    rank = None
    for idx, hit in enumerate(hits, start=1):
        is_gold = hit_matches_gold(hop, hit)
        if rank is None and is_gold:
            rank = idx
        all_hits.append(
            {
                "rank": idx,
                "doc_id": hit_doc_id(hit),
                "score": hit_score(hit),
                "title": compact(hit_title(hit), 140),
                "snippet": compact(hit_text(hit), 240),
                "is_gold": is_gold,
            }
        )
    return {"rank": rank, "all_hits": all_hits, "hits": all_hits[:10], "error": None}


@dataclass
class LexicalCorpus:
    doc_counts: dict[str, Counter[str]]
    doc_lengths: dict[str, int]
    df: Counter[str]
    avg_len: float
    n_docs: int
    doc_refs: dict[str, str] = field(default_factory=dict)


def build_web_lexical_corpus(queries: list[str]) -> LexicalCorpus:
    wanted = {term for query in queries for term in query_terms(query)}
    doc_counts: dict[str, Counter[str]] = {}
    doc_lengths: dict[str, int] = {}
    df: Counter[str] = Counter()
    doc_refs: dict[str, str] = {}
    for path in sorted(FULL_DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        toks = tokens(text)
        doc_id = f"web/{path.stem}"
        doc_lengths[doc_id] = len(toks)
        doc_refs[doc_id] = str(path)
        counts = Counter(token for token in toks if token in wanted)
        if counts:
            doc_counts[doc_id] = counts
            for term in counts:
                df[term] += 1
    avg_len = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1.0
    return LexicalCorpus(doc_counts=doc_counts, doc_lengths=doc_lengths, df=df, avg_len=avg_len, n_docs=len(doc_lengths), doc_refs=doc_refs)


def load_local_documents_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("documents", "docs", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        keyed_items = []
        for key, value in data.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("doc_id", key)
                keyed_items.append(item)
        return keyed_items
    return []


def local_doc_id(item: dict[str, Any], idx: int) -> str:
    return str(item.get("doc_id") or item.get("id") or item.get("document_id") or item.get("source") or f"doc_{idx}")


def local_doc_text(item: dict[str, Any]) -> str:
    pieces = []
    for key in ("content", "text", "page_content", "document", "snippet"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value)
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("title", "source", "path", "file_path", "locator"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value)
    return "\n".join(pieces)


def local_doc_ref(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for source in (item, metadata):
        for key in (
            "locator",
            "path",
            "file_path",
            "original_path",
            "source_identifier",
            "extracted_path",
            "relative_path",
            "filename",
            "source",
            "title",
        ):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def build_local_lexical_corpora(task_ids: set[str], queries: list[str], local_index_root: Path) -> dict[str, LexicalCorpus]:
    wanted = {term for query in queries for term in query_terms(query)}
    corpora: dict[str, LexicalCorpus] = {}
    for task_id in sorted(task_ids):
        docs_path = local_index_root / task_id / "documents.json"
        if not docs_path.is_file():
            continue
        doc_counts: dict[str, Counter[str]] = {}
        doc_lengths: dict[str, int] = {}
        df: Counter[str] = Counter()
        doc_refs: dict[str, str] = {}
        for idx, item in enumerate(load_local_documents_json(docs_path)):
            doc_id = local_doc_id(item, idx)
            text = local_doc_text(item)
            toks = tokens(text)
            doc_lengths[doc_id] = len(toks)
            doc_refs[doc_id] = local_doc_ref(item)
            counts = Counter(token for token in toks if token in wanted)
            if counts:
                doc_counts[doc_id] = counts
                for term in counts:
                    df[term] += 1
        avg_len = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1.0
        corpora[task_id] = LexicalCorpus(doc_counts=doc_counts, doc_lengths=doc_lengths, df=df, avg_len=avg_len, n_docs=len(doc_lengths), doc_refs=doc_refs)
    return corpora


def lexical_scores(corpus: LexicalCorpus, query: str) -> list[tuple[float, str]]:
    terms = query_terms(query)
    if not terms:
        return []
    k1 = 1.2
    b = 0.75
    scores: list[tuple[float, str]] = []
    for doc_id, counts in corpus.doc_counts.items():
        score = 0.0
        doc_len = max(1, corpus.doc_lengths.get(doc_id, 1))
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (corpus.n_docs - corpus.df[term] + 0.5) / (corpus.df[term] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / corpus.avg_len))
        if score > 0:
            scores.append((score, doc_id))
    scores.sort(reverse=True)
    return scores


def local_ref_matches_gold(gold_ref_raw: str, candidate_ref_raw: str) -> bool:
    gold_ref = normalize_pathish(gold_ref_raw)
    candidate_ref = normalize_pathish(candidate_ref_raw)
    if not gold_ref or not candidate_ref:
        return False
    gold_path = Path(gold_ref)
    candidate_path = Path(candidate_ref)
    if candidate_ref.endswith(gold_path.name):
        return True
    if gold_path.stem and candidate_path.stem == gold_path.stem:
        return True
    gold_parts = [part for part in gold_ref.split("/") if part]
    if len(gold_parts) >= 2:
        suffix_without_ext = "/".join(gold_parts[-2:])
        suffix_stem = str(Path(suffix_without_ext).with_suffix(""))
        candidate_no_ext = str(Path(candidate_ref).with_suffix(""))
        if suffix_stem and suffix_stem in candidate_no_ext:
            return True
    return False


def lexical_rank_for_hop(hop: HopStats, scores: list[tuple[float, str]], corpus: LexicalCorpus) -> int | None:
    if hop.hop_type == "W":
        for idx, (_, doc_id) in enumerate(scores, start=1):
            if doc_id == hop.gold_doc_id:
                return idx
        return None
    for idx, (_, doc_id) in enumerate(scores, start=1):
        ref = corpus.doc_refs.get(doc_id, "")
        if hop.gold_doc_id and doc_id == hop.gold_doc_id:
            return idx
        if local_ref_matches_gold(hop.gold_doc_ref, ref):
            return idx
    return None


def dense_rank_for_hop(hop: HopStats, dense_hits: list[dict[str, Any]], corpus: LexicalCorpus) -> int | None:
    ranked = [(1.0 / max(1, int(hit.get("rank") or idx)), str(hit.get("doc_id") or "")) for idx, hit in enumerate(dense_hits, start=1)]
    return lexical_rank_for_hop(hop, ranked, corpus)


def fusion_rank(
    dense_hits: list[dict[str, Any]],
    lexical: list[tuple[float, str]],
    hop: HopStats,
    corpus: LexicalCorpus,
    *,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    rrf_k: float = 60.0,
    lexical_limit: int = 200,
    dense_limit: int | None = None,
) -> int | None:
    scores: defaultdict[str, float] = defaultdict(float)
    dense_source = dense_hits[:dense_limit] if dense_limit is not None else dense_hits
    for idx, hit in enumerate(dense_source, start=1):
        doc_id = str(hit.get("doc_id") or "")
        if doc_id:
            scores[doc_id] += dense_weight / (rrf_k + idx)
    for idx, (_, doc_id) in enumerate(lexical[:lexical_limit], start=1):
        scores[doc_id] += lexical_weight / (rrf_k + idx)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return lexical_rank_for_hop(hop, [(score, doc_id) for doc_id, score in ranked], corpus)


def fusion_rank_from_lists(
    dense_docs: list[str],
    lexical_docs: list[str],
    gold_doc_ids: set[str],
    *,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    rrf_k: float = 60.0,
    lexical_limit: int = 200,
    dense_limit: int | None = None,
) -> int | None:
    scores: defaultdict[str, float] = defaultdict(float)
    dense_source = dense_docs[:dense_limit] if dense_limit is not None else dense_docs
    for idx, doc_id in enumerate(dense_source, start=1):
        if doc_id:
            scores[doc_id] += dense_weight / (rrf_k + idx)
    for idx, doc_id in enumerate(lexical_docs[:lexical_limit], start=1):
        if doc_id:
            scores[doc_id] += lexical_weight / (rrf_k + idx)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for idx, (doc_id, _score) in enumerate(ranked, start=1):
        if doc_id in gold_doc_ids:
            return idx
    return None


def best_rank(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def topk_rate(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def median_rank(ranks: list[int]) -> float | None:
    if not ranks:
        return None
    values = sorted(ranks)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2


def method_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    ranks = [row.get(key) for row in rows]
    return {
        "top1": topk_rate(ranks, 1),
        "top5": topk_rate(ranks, 5),
        "top10": topk_rate(ranks, 10),
        "top20": topk_rate(ranks, 20),
        "found": sum(1 for rank in ranks if rank is not None),
        "median_rank_found": median_rank([rank for rank in ranks if isinstance(rank, int)]),
        "rank_buckets": dict(Counter(rank_bucket(rank) for rank in ranks)),
    }


def summarize_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "dense": method_summary(rows, f"{prefix}_dense_rank"),
        "lexical": method_summary(rows, f"{prefix}_lexical_rank"),
        "fusion": method_summary(rows, f"{prefix}_fusion_rank"),
    }


def grouped_summaries(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"difficulty={row['difficulty']}"] .append(row)
        groups[f"hop_type={row['hop_type']}"] .append(row)
        groups[f"hop_type={row['hop_type']}|difficulty={row['difficulty']}"] .append(row)
        groups[f"company={row['company']}"] .append(row)
        groups[f"pattern={row['pattern']}"] .append(row)
    return {name: {"n": len(group), **summarize_rows(group, prefix)} for name, group in sorted(groups.items())}


def fusion_tuning_settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for dense_weight in [0.0, 0.1, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0]:
        for rrf_k in [20, 60, 100]:
            settings.append({
                "label": f"d{dense_weight:g}_l1_k{rrf_k}",
                "dense_weight": dense_weight,
                "lexical_weight": 1.0,
                "rrf_k": rrf_k,
                "dense_limit": 50,
                "lexical_limit": 200,
            })
    for lexical_weight in [0.25, 0.5, 0.75]:
        settings.append({
            "label": f"d1_l{lexical_weight:g}_k60",
            "dense_weight": 1.0,
            "lexical_weight": lexical_weight,
            "rrf_k": 60,
            "dense_limit": 50,
            "lexical_limit": 200,
        })
    return settings


def tuned_rank_for_variant(variant: dict[str, Any], setting: dict[str, Any]) -> int | None:
    gold_doc_ids = set(variant.get("gold_doc_ids_for_tuning") or [])
    if not gold_doc_ids:
        return None
    return fusion_rank_from_lists(
        [str(doc_id) for doc_id in variant.get("dense_doc_ids") or []],
        [str(doc_id) for doc_id in variant.get("lexical_doc_ids") or []],
        gold_doc_ids,
        dense_weight=float(setting["dense_weight"]),
        lexical_weight=float(setting["lexical_weight"]),
        rrf_k=float(setting["rrf_k"]),
        dense_limit=int(setting["dense_limit"]),
        lexical_limit=int(setting["lexical_limit"]),
    )


def tuned_rank_for_row(row: dict[str, Any], variant_key: str, setting: dict[str, Any]) -> int | None:
    ranks = [
        rank
        for rank in (tuned_rank_for_variant(variant, setting) for variant in row.get(variant_key) or [])
        if rank is not None
    ]
    return min(ranks) if ranks else None


def summarize_rank_list(ranks: list[int | None]) -> dict[str, Any]:
    present = [rank for rank in ranks if isinstance(rank, int)]
    return {
        "n": len(ranks),
        "top1": topk_rate(ranks, 1),
        "top5": topk_rate(ranks, 5),
        "top10": topk_rate(ranks, 10),
        "top20": topk_rate(ranks, 20),
        "found": len(present),
        "median_rank_found": median_rank(present),
    }


def tuned_group_summaries(rows: list[dict[str, Any]], variant_key: str, setting: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        rank = tuned_rank_for_row(row, variant_key, setting)
        groups[f"hop_type={row['hop_type']}"].append(rank)
        groups[f"difficulty={row['difficulty']}"].append(rank)
    return {name: summarize_rank_list(ranks) for name, ranks in sorted(groups.items())}


def build_fusion_tuning_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_sets = {
        "actual_model_queries": "actual_variants",
        "question_baseline": "baseline_variants",
    }
    output: dict[str, Any] = {}
    for query_set, variant_key in query_sets.items():
        results = []
        for setting in fusion_tuning_settings():
            ranks = [tuned_rank_for_row(row, variant_key, setting) for row in rows]
            results.append({
                "label": setting["label"],
                "setting": setting,
                "overall": summarize_rank_list(ranks),
                "groups": tuned_group_summaries(rows, variant_key, setting),
            })
        results.sort(
            key=lambda item: (
                item["overall"]["top10"],
                item["overall"]["top5"],
                item["overall"]["top20"],
                item["overall"]["top1"],
            ),
            reverse=True,
        )
        output[query_set] = results
    return output


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{esc(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table>"
    )


def summary_method_rows(block: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for method, stats in block.items():
        rows.append([
            esc(method), pct(stats["top1"]), pct(stats["top5"]), pct(stats["top10"]), pct(stats["top20"]),
            esc(stats["found"]), esc(stats["median_rank_found"]), esc(stats["rank_buckets"]),
        ])
    return rows


def group_rows(grouped: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for group_name, block in grouped.items():
        rows.append([
            esc(group_name), esc(block["n"]),
            pct(block["dense"]["top10"]), pct(block["lexical"]["top10"]), pct(block["fusion"]["top10"]),
            pct(block["dense"]["top20"]), pct(block["lexical"]["top20"]), pct(block["fusion"]["top20"]),
        ])
    return rows


def difficulty_rows(grouped: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for difficulty in ["easy", "medium", "hard-answer", "hard-retrieval"]:
        block = grouped.get(f"difficulty={difficulty}")
        if not block:
            continue
        rows.append([
            esc(difficulty),
            esc(block["n"]),
            pct(block["dense"]["top1"]),
            pct(block["lexical"]["top1"]),
            pct(block["fusion"]["top1"]),
            pct(block["dense"]["top5"]),
            pct(block["lexical"]["top5"]),
            pct(block["fusion"]["top5"]),
            pct(block["dense"]["top10"]),
            pct(block["lexical"]["top10"]),
            pct(block["fusion"]["top10"]),
            pct(block["dense"]["top20"]),
            pct(block["lexical"]["top20"]),
            pct(block["fusion"]["top20"]),
        ])
    return rows


def tuning_rows(results: list[dict[str, Any]], limit: int = 10) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in results[:limit]:
        setting = item["setting"]
        stats = item["overall"]
        rows.append([
            esc(item["label"]),
            esc(setting["dense_weight"]),
            esc(setting["lexical_weight"]),
            esc(setting["rrf_k"]),
            pct(stats["top1"]),
            pct(stats["top5"]),
            pct(stats["top10"]),
            pct(stats["top20"]),
            esc(f"{stats['found']}/{stats['n']}"),
            esc(stats["median_rank_found"]),
        ])
    return rows


def tuning_group_rows(item: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for group, stats in item["groups"].items():
        rows.append([
            esc(group),
            esc(stats["n"]),
            pct(stats["top1"]),
            pct(stats["top5"]),
            pct(stats["top10"]),
            pct(stats["top20"]),
            esc(f"{stats['found']}/{stats['n']}"),
        ])
    return rows


def build_viewer(out_path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    detail_rows = []
    for row in rows:
        detail_rows.append([
            esc(row["hop_type"]), esc(row["difficulty"]), esc(f"{row['task_id']} {row['chain_id']} h{row['hop_number']}"),
            esc(f"{row['correct']}/{row['attempted']}"), esc(row["question"]),
            f"<code>{esc(row['gold_doc_id'])}</code><br>{esc(row['answer'])}",
            esc(row["actual_model_queries_used"]), esc(row["actual_model_queries_observed"]),
            esc(row["actual_dense_rank"]), esc(row["actual_lexical_rank"]), esc(row["actual_fusion_rank"]),
            esc(row["baseline_dense_rank"]), esc(row["baseline_lexical_rank"]), esc(row["baseline_fusion_rank"]),
            "<br>".join(
                f"<b>{esc(v['label'])}</b>: d={esc(v['dense_rank'])}, l={esc(v['lexical_rank'])}, f={esc(v['fusion_rank'])}<br><span class='query'>{esc(v['query'])}</span>"
                for v in row["actual_variants"][:8]
            ),
        ])
    fusion_tuning = summary.get("fusion_tuning") or {}
    actual_tuning = fusion_tuning.get("actual_model_queries") or []
    baseline_tuning = fusion_tuning.get("question_baseline") or []
    recommended_tuning = actual_tuning[0] if actual_tuning else None
    recommended_group_table = (
        render_table(["group", "n", "top1", "top5", "top10", "top20", "found"], tuning_group_rows(recommended_tuning))
        if recommended_tuning
        else "<p class='muted'>No tuning results available.</p>"
    )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Privacy HopQA retrieval method audit</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #f5f1e8; color: #17202a; }}
    header {{ padding: 28px 34px; background: #fffaf0; border-bottom: 1px solid #ded4c2; position: sticky; top: 0; z-index: 2; }}
    main {{ padding: 24px 34px 60px; max-width: 1700px; }}
    section {{ background: white; border: 1px solid #ded4c2; border-radius: 14px; padding: 18px; margin: 16px 0; box-shadow: 0 12px 30px rgba(40,31,18,0.08); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #ded4c2; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #edf6f4; text-align: left; }}
    code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
    .query {{ color: #576574; font-size: 12px; }}
    .muted {{ color: #65758b; }}
  </style>
</head>
<body>
<header>
  <h1>Privacy HopQA retrieval method audit</h1>
  <div class="muted">Generated {esc(summary['generated_at'])}. Sampled {summary['sample_size']} hops from current 4B prefix-step traces. Actual model queries are capped at {summary['max_model_queries_per_hop']} per hop by frequency.</div>
</header>
<main>
  <section>
    <h2>Actual model queries</h2>
    <p class="muted">For each sampled hop, this takes the retrieval queries the model actually issued for the expected source type. The row-level rank is the best rank over those actual observed queries.</p>
    {render_table(["method", "top1", "top5", "top10", "top20", "found", "median rank", "rank buckets"], summary_method_rows(summary['actual_model_queries']['overall']))}
  </section>
  <section>
    <h2>Question-derived baselines</h2>
    <p class="muted">These use deterministic variants derived only from the materialized hop question: literal question, content words, and entity/numeric terms. The row-level rank is the best over those variants.</p>
    {render_table(["method", "top1", "top5", "top10", "top20", "found", "median rank", "rank buckets"], summary_method_rows(summary['question_baseline']['overall']))}
  </section>
  <section>
    <h2>Fusion tuning sweep</h2>
    <p class="muted">These rows re-rank the stored dense top-50 and lexical top-200 document lists with weighted reciprocal-rank fusion. The live training retriever has not changed; this table estimates which fusion setting would surface the gold parent document earliest.</p>
    <h3>Actual model queries</h3>
    {render_table(["setting", "dense weight", "lexical weight", "RRF k", "top1", "top5", "top10", "top20", "found", "median rank"], tuning_rows(actual_tuning))}
    <h3>Question-derived baselines</h3>
    {render_table(["setting", "dense weight", "lexical weight", "RRF k", "top1", "top5", "top10", "top20", "found", "median rank"], tuning_rows(baseline_tuning))}
    <h3>Recommended actual-query setting by source/difficulty</h3>
    <p class="muted">The recommended row is selected by top-10, then top-5, then top-20, then top-1. In this run it is <code>{esc(recommended_tuning['label'] if recommended_tuning else 'n/a')}</code>.</p>
    {recommended_group_table}
  </section>
  <section>
    <h2>Breakdown by difficulty</h2>
    <p class="muted">Difficulty is assigned from the current rollout traces: easy hops are frequently correct, hard-answer hops often retrieve/read useful evidence but answer incorrectly, and hard-retrieval hops have low success and weak positive-reader evidence. Each rank is the best gold-parent rank over the query variants for that row.</p>
    <h3>Actual model queries</h3>
    {render_table(["difficulty", "n", "dense top1", "lexical top1", "fusion top1", "dense top5", "lexical top5", "fusion top5", "dense top10", "lexical top10", "fusion top10", "dense top20", "lexical top20", "fusion top20"], difficulty_rows(summary['actual_model_queries']['groups']))}
    <h3>Question-derived baselines</h3>
    {render_table(["difficulty", "n", "dense top1", "lexical top1", "fusion top1", "dense top5", "lexical top5", "fusion top5", "dense top10", "lexical top10", "fusion top10", "dense top20", "lexical top20", "fusion top20"], difficulty_rows(summary['question_baseline']['groups']))}
  </section>
  <section>
    <h2>Actual-query top-10/top-20 by group</h2>
    {render_table(["group", "n", "dense top10", "lexical top10", "fusion top10", "dense top20", "lexical top20", "fusion top20"], group_rows(summary['actual_model_queries']['groups']))}
  </section>
  <section>
    <h2>Question-baseline top-10/top-20 by group</h2>
    {render_table(["group", "n", "dense top10", "lexical top10", "fusion top10", "dense top20", "lexical top20", "fusion top20"], group_rows(summary['question_baseline']['groups']))}
  </section>
  <section>
    <h2>Per-hop details</h2>
    {render_table(["type", "difficulty", "hop", "correct", "question", "gold", "actual used", "actual observed", "actual dense", "actual lexical", "actual fusion", "baseline dense", "baseline lexical", "baseline fusion", "actual queries"], detail_rows)}
  </section>
</main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")




def gold_doc_ids_for_hop(hop: HopStats, corpus: LexicalCorpus) -> set[str]:
    if hop.hop_type == "W":
        return {hop.gold_doc_id} if hop.gold_doc_id else set()
    out: set[str] = set()
    for doc_id, ref in corpus.doc_refs.items():
        if hop.gold_doc_id and doc_id == hop.gold_doc_id:
            out.add(doc_id)
        elif local_ref_matches_gold(hop.gold_doc_ref, ref):
            out.add(doc_id)
    return out

def evaluate_variant(hop: HopStats, variant: dict[str, Any], dense: dict[str, Any], corpus: LexicalCorpus) -> dict[str, Any]:
    scores = lexical_scores(corpus, str(variant["query"]))
    lexical_rank = lexical_rank_for_hop(hop, scores, corpus)
    fused = fusion_rank(dense["all_hits"], scores, hop, corpus)
    gold_doc_ids = gold_doc_ids_for_hop(hop, corpus)
    return {
        "label": variant["label"],
        "query": variant["query"],
        "observed_count": variant.get("observed_count"),
        "dense_rank": dense_rank_for_hop(hop, dense["all_hits"], corpus),
        "lexical_rank": lexical_rank,
        "fusion_rank": fused,
        "gold_doc_ids_for_tuning": sorted(gold_doc_ids),
        "dense_doc_ids": [str(hit.get("doc_id") or "") for hit in dense["all_hits"][:50] if hit.get("doc_id")],
        "lexical_doc_ids": [doc_id for _score, doc_id in scores[:200]],
        "dense_hits": dense["hits"],
        "lexical_top_docs": [
            {"rank": idx, "doc_id": doc_id, "score": score, "ref": corpus.doc_refs.get(doc_id, "")}
            for idx, (score, doc_id) in enumerate(scores[:10], start=1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--helper-url", default=DEFAULT_HELPER_URL)
    parser.add_argument("--local-index-root", type=Path, default=DEFAULT_LOCAL_INDEX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-traces", type=int, default=2500)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--dense-k", type=int, default=50)
    parser.add_argument("--max-model-queries-per-hop", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    hops_by_key = load_hops(args.dataset)
    trace_paths = reservoir_sample(iter_trace_paths(args.run_root), args.max_traces)
    for path in trace_paths:
        add_trace_stats(path, hops_by_key)
    selected = select_hops(list(hops_by_key.values()), args.sample_size)

    baseline_jobs: list[tuple[HopStats, dict[str, Any], str]] = []
    actual_jobs: list[tuple[HopStats, dict[str, Any], str]] = []
    for hop in selected:
        for variant in question_variants(hop):
            baseline_jobs.append((hop, variant, "baseline"))
        for variant in model_query_variants(hop, args.max_model_queries_per_hop):
            actual_jobs.append((hop, variant, "actual"))

    all_queries = [variant["query"] for _, variant, _ in baseline_jobs + actual_jobs]
    web_corpus = build_web_lexical_corpus(all_queries)
    local_corpora = build_local_lexical_corpora({hop.task_id for hop in selected if hop.hop_type == "L"}, all_queries, args.local_index_root)

    dense_results: dict[tuple[tuple[str, str, int], str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(dense_result, args.helper_url, hop, variant, args.dense_k, args.timeout_s): (hop, variant, query_set)
            for hop, variant, query_set in baseline_jobs + actual_jobs
        }
        for future in as_completed(futures):
            hop, variant, query_set = futures[future]
            dense_results[(hop.key, query_set, variant["label"])] = future.result()

    rows: list[dict[str, Any]] = []
    for hop in selected:
        corpus = web_corpus if hop.hop_type == "W" else local_corpora.get(hop.task_id)
        if corpus is None:
            continue
        baseline_variants = []
        for variant in question_variants(hop):
            dense = dense_results[(hop.key, "baseline", variant["label"])]
            baseline_variants.append(evaluate_variant(hop, variant, dense, corpus))
        actual_variants = []
        for variant in model_query_variants(hop, args.max_model_queries_per_hop):
            dense = dense_results[(hop.key, "actual", variant["label"])]
            actual_variants.append(evaluate_variant(hop, variant, dense, corpus))
        row = {
            "task_id": hop.task_id,
            "chain_id": hop.chain_id,
            "hop_number": hop.hop_number,
            "hop_type": hop.hop_type,
            "company": hop.company,
            "pattern": hop.pattern,
            "question": hop.question,
            "answer": hop.answer,
            "gold_doc_id": hop.gold_doc_id,
            "gold_doc_ref": hop.gold_doc_ref,
            "attempted": hop.attempted,
            "correct": hop.correct,
            "positive_reader": hop.positive_reader,
            "difficulty": hop.difficulty,
            "actual_model_queries_observed": len(hop.model_queries),
            "actual_model_queries_used": len(actual_variants),
            "actual_query_count_capped": len(hop.model_queries) > len(actual_variants),
            "baseline_dense_rank": best_rank([variant["dense_rank"] for variant in baseline_variants]),
            "baseline_lexical_rank": best_rank([variant["lexical_rank"] for variant in baseline_variants]),
            "baseline_fusion_rank": best_rank([variant["fusion_rank"] for variant in baseline_variants]),
            "actual_dense_rank": best_rank([variant["dense_rank"] for variant in actual_variants]),
            "actual_lexical_rank": best_rank([variant["lexical_rank"] for variant in actual_variants]),
            "actual_fusion_rank": best_rank([variant["fusion_rank"] for variant in actual_variants]),
            "baseline_variants": baseline_variants,
            "actual_variants": actual_variants,
        }
        rows.append(row)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_root": str(args.run_root),
        "dataset": str(args.dataset),
        "helper_url": args.helper_url,
        "local_index_root": str(args.local_index_root),
        "trace_count_sampled_for_difficulty": len(trace_paths),
        "sample_size": len(rows),
        "dense_k": args.dense_k,
        "max_model_queries_per_hop": args.max_model_queries_per_hop,
        "web_lexical_pool": str(FULL_DOCS_DIR),
        "web_lexical_pool_docs": web_corpus.n_docs,
        "local_task_corpora": len(local_corpora),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
        "hop_type_counts": dict(Counter(row["hop_type"] for row in rows)),
        "actual_query_caps": sum(1 for row in rows if row["actual_query_count_capped"]),
        "actual_model_queries": {
            "overall": summarize_rows(rows, "actual"),
            "groups": grouped_summaries(rows, "actual"),
        },
        "question_baseline": {
            "overall": summarize_rows(rows, "baseline"),
            "groups": grouped_summaries(rows, "baseline"),
        },
        "fusion_tuning": build_fusion_tuning_summary(rows),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "fusion_tuning_summary.json").write_text(
        json.dumps(summary["fusion_tuning"], indent=2),
        encoding="utf-8",
    )
    with (args.out_dir / "retrieval_method_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    build_viewer(args.out_dir / "index.html", summary, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
