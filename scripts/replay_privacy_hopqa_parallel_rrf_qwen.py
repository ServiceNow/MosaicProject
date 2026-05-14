#!/usr/bin/env python3
"""Realistic Privacy HopQA current-vs-RRF replay with chooser, reader, resolver.

This diagnostic uses the retrieval-method audit rows as the sample frame. For
one hop at a time, it takes several actual model search queries, builds the
merged candidate parent-document pool under the current helper ranking and under
RRF ranking, then runs the production document chooser, document reader, and hop
resolver prompts with a local OpenAI-compatible Qwen endpoint.

It is still an isolated-hop replay: previous-hop answers are gold answers from
the dataset so the measurement focuses on retrieval ranking plus chooser/reader
behavior for the current hop.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import html
import importlib.util
import json
import math
import random
import re
import sys
import time
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
PIPELINERL_ROOT = Path("/home/toolkit/my_code/PipelineRL")
PROMPTS_PATH = PIPELINERL_ROOT / "pipelinerl/domains/privacy_hopqa/prompts.py"
PROMPTS_SPEC = importlib.util.spec_from_file_location("privacy_hopqa_prompts", PROMPTS_PATH)
if PROMPTS_SPEC is None or PROMPTS_SPEC.loader is None:
    raise RuntimeError(f"Could not load prompt builders from {PROMPTS_PATH}")
PROMPTS_MODULE = importlib.util.module_from_spec(PROMPTS_SPEC)
sys.modules[PROMPTS_SPEC.name] = PROMPTS_MODULE
PROMPTS_SPEC.loader.exec_module(PROMPTS_MODULE)

build_doc_choose_prompt = PROMPTS_MODULE.build_doc_choose_prompt
build_doc_reader_prompt = PROMPTS_MODULE.build_doc_reader_prompt
build_hop_resolve_prompt = PROMPTS_MODULE.build_hop_resolve_prompt

DEFAULT_AUDIT_ROWS = REPO / "runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14_large240/retrieval_method_rows.jsonl"
DEFAULT_DATASET = REPO / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl"
DEFAULT_RUN_ROOT = Path("/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122")
DEFAULT_LOCAL_INDEX_ROOT = Path("/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/local_indices_fixeddeps_20260415")
DEFAULT_OUT_DIR = REPO / "runtime_analysis/privacy_hopqa_parallel_rrf_replay_qwen4b_2026-05-14"
FULL_DOCS_DIR = Path("/home/toolkit/full_docs")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
RANDOM_SEED = 20260514


@dataclass(frozen=True)
class ReplayConfig:
    base_url: str
    retrieval_top_k: int
    queries_per_hop: int
    max_candidate_cards: int
    choose_top_k: int
    max_parallel_doc_reads: int
    rrf_k: int
    reader_text_chars: int
    max_tokens: int
    temperature: float
    include_trace_distractors: bool
    run_root: Path
    read_all_windows_max_count: int
    large_parent_top_windows: int
    large_parent_neighbor_windows: int


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def compact(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        task_id = str(row.get("task_id") or "")
        chain_id = str(row.get("chain_id") or "")
        answers_so_far: list[dict[str, Any]] = []
        for hop in row.get("hops") or []:
            hop_number = int(hop.get("hop_number") or 0)
            out[(task_id, chain_id, hop_number)] = {
                "row": row,
                "hop": hop,
                "answers_so_far": list(answers_so_far),
            }
            answers_so_far.append({"hop_number": hop_number, "answer": str(hop.get("answer") or "")})
    return out


def answers_so_far_text(answers: list[dict[str, Any]]) -> str:
    if not answers:
        return "No previous answers yet."
    return "\n".join(f"({item['hop_number']}): {item['answer']}" for item in answers)


def normalize_answer(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9%.$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = TOKEN_RE.findall(normalize_answer(prediction))
    gold_tokens = TOKEN_RE.findall(normalize_answer(gold))
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def match_answer(answer: str, variants: list[str], f1_threshold: float = 0.75) -> dict[str, Any]:
    normalized = normalize_answer(answer)
    exact = any(normalized == normalize_answer(variant) for variant in variants if str(variant).strip())
    best_f1 = 0.0
    best_variant = ""
    for variant in variants:
        score = token_f1(answer, str(variant))
        if score > best_f1:
            best_f1 = score
            best_variant = str(variant)
    return {
        "exact_variant_match": exact,
        "reward_match": exact or best_f1 >= f1_threshold,
        "best_f1": best_f1,
        "best_variant": best_variant,
    }


def parent_doc_id(doc_id: Any) -> str:
    return str(doc_id or "").split("#", 1)[0]


def rank_in(doc_ids: list[str], gold_ids: set[str]) -> int | None:
    for idx, doc_id in enumerate(doc_ids, start=1):
        if parent_doc_id(doc_id) in gold_ids or str(doc_id) in gold_ids:
            return idx
    return None


def rrf_ranked_docs(current_doc_ids: list[str], lexical_doc_ids: list[str], *, rrf_k: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for idx, doc_id in enumerate(current_doc_ids[:50], start=1):
        scores[parent_doc_id(doc_id)] += 1.0 / (rrf_k + idx)
    for idx, doc_id in enumerate(lexical_doc_ids[:200], start=1):
        scores[parent_doc_id(doc_id)] += 1.0 / (rrf_k + idx)
    return [doc_id for doc_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def selected_variants(row: dict[str, Any], queries_per_hop: int) -> list[dict[str, Any]]:
    variants = list(row.get("actual_variants") or [])
    variants.sort(key=lambda item: (-int(item.get("observed_count") or 1), str(item.get("query") or "")))
    return variants[:queries_per_hop]


def hit_doc_id(hit: dict[str, Any]) -> str:
    return parent_doc_id(hit.get("parent_doc_id") or hit.get("doc_id") or hit.get("docid") or hit.get("id") or "")


def hit_title(hit: dict[str, Any], doc_id: str) -> str:
    return str(hit.get("title") or hit.get("locator") or hit.get("ref") or doc_id)


def hit_snippet(hit: dict[str, Any]) -> str:
    return str(hit.get("snippet") or hit.get("text") or hit.get("content") or "")


def load_local_documents(task_id: str, cache: dict[str, dict[str, Any]], local_index_root: Path) -> dict[str, Any]:
    if task_id not in cache:
        cache[task_id] = json.loads((local_index_root / task_id / "documents.json").read_text(encoding="utf-8"))
    return cache[task_id]


def load_doc_text(doc_id: str, task_id: str, local_cache: dict[str, dict[str, Any]], local_index_root: Path, fallback: str, limit: int) -> tuple[str, str, str]:
    doc_id = parent_doc_id(doc_id)
    if doc_id.startswith("web/"):
        raw = doc_id.split("/", 1)[1]
        if raw.isdigit():
            path = FULL_DOCS_DIR / f"{raw}.txt"
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")[:limit]
                return path.name, str(path), text
        return doc_id, doc_id, fallback[:limit]

    docs = load_local_documents(task_id, local_cache, local_index_root)
    item = docs.get(doc_id)
    if isinstance(item, dict):
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        locator = str(metadata.get("file_path") or metadata.get("original_path") or metadata.get("source") or doc_id)
        title = Path(locator).name or doc_id
        text = str(item.get("content") or fallback)[:limit]
        return title, locator, text
    return doc_id, doc_id, fallback[:limit]


def base_card_from_doc(
    *,
    doc_id: str,
    query: str,
    rank: int,
    source: str,
    hit: dict[str, Any],
    task_id: str,
    local_cache: dict[str, dict[str, Any]],
    local_index_root: Path,
    reader_text_chars: int,
) -> dict[str, Any]:
    snippet = hit_snippet(hit)
    title, locator, text = load_doc_text(doc_id, task_id, local_cache, local_index_root, snippet, reader_text_chars)
    if not snippet:
        snippet = compact(text, 500)
    return {
        "doc_id": parent_doc_id(doc_id),
        "source": source,
        "title": title or hit_title(hit, doc_id),
        "locator": locator,
        "best_rank": int(rank),
        "hit_count": 1,
        "top_queries": [query],
        "snippet": compact(snippet, 500),
        "matched_window_count": int(hit.get("matched_window_count") or 1),
        "total_window_count": int(hit.get("window_count") or hit.get("total_window_count") or 1),
        "reader_text": text,
    }


def merge_card(existing: dict[str, Any], new_card: dict[str, Any]) -> None:
    existing["best_rank"] = min(int(existing.get("best_rank") or 999), int(new_card.get("best_rank") or 999))
    existing["hit_count"] = int(existing.get("hit_count") or 0) + 1
    queries = list(existing.get("top_queries") or [])
    for query in new_card.get("top_queries") or []:
        if query not in queries:
            queries.append(query)
    existing["top_queries"] = queries[:5]
    if len(str(new_card.get("snippet") or "")) > len(str(existing.get("snippet") or "")):
        existing["snippet"] = new_card.get("snippet")
    existing["matched_window_count"] = max(int(existing.get("matched_window_count") or 1), int(new_card.get("matched_window_count") or 1))
    existing["total_window_count"] = max(int(existing.get("total_window_count") or 1), int(new_card.get("total_window_count") or 1))


def trace_distractor_cards(
    row: dict[str, Any],
    cfg: ReplayConfig,
    local_cache: dict[str, dict[str, Any]],
    local_index_root: Path,
) -> list[dict[str, Any]]:
    if not cfg.include_trace_distractors:
        return []
    expected_source = "web" if row.get("hop_type") == "W" else "local"
    pattern = f"{row['task_id']}_{row['chain_id']}_*/trace.json"
    for path in sorted((cfg.run_root / "privacy_hopqa_runs").glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        for hop_state in data.get("hop_states") or []:
            if int(hop_state.get("hop_number") or 0) != int(row["hop_number"]):
                continue
            cards: list[dict[str, Any]] = []
            for idx, raw_card in enumerate(hop_state.get("candidate_cards") or [], start=1):
                source = str(raw_card.get("source") or "")
                if source == expected_source:
                    continue
                doc_id = parent_doc_id(raw_card.get("parent_doc_id") or raw_card.get("doc_id") or "")
                if not doc_id:
                    continue
                cards.append(
                    base_card_from_doc(
                        doc_id=doc_id,
                        query=str((raw_card.get("top_queries") or ["trace distractor"])[0]),
                        rank=int(raw_card.get("best_rank") or idx),
                        source=source,
                        hit=raw_card,
                        task_id=str(row.get("task_id") or ""),
                        local_cache=local_cache,
                        local_index_root=local_index_root,
                        reader_text_chars=cfg.reader_text_chars,
                    )
                )
            return cards
    return []


def cards_for_mode(
    row: dict[str, Any],
    variants: list[dict[str, Any]],
    mode: str,
    cfg: ReplayConfig,
    local_cache: dict[str, dict[str, Any]],
    local_index_root: Path,
    shared_distractor_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    source = "web" if row.get("hop_type") == "W" else "local"
    for variant in variants:
        query = str(variant.get("query") or "")
        dense_hits_by_doc = {hit_doc_id(hit): hit for hit in variant.get("dense_hits") or []}
        lexical_hits_by_doc = {parent_doc_id(hit.get("doc_id")): hit for hit in variant.get("lexical_top_docs") or []}
        if mode == "current":
            ranked_docs = [hit_doc_id(hit) for hit in (variant.get("dense_hits") or [])]
        elif mode == "bm25":
            ranked_docs = [parent_doc_id(x) for x in variant.get("lexical_doc_ids") or []]
        elif mode == "rrf":
            ranked_docs = rrf_ranked_docs(
                [str(x) for x in variant.get("dense_doc_ids") or []],
                [str(x) for x in variant.get("lexical_doc_ids") or []],
                rrf_k=cfg.rrf_k,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        for rank, doc_id in enumerate(ranked_docs[: cfg.retrieval_top_k], start=1):
            doc_id = parent_doc_id(doc_id)
            hit = dense_hits_by_doc.get(doc_id) or lexical_hits_by_doc.get(doc_id) or {"doc_id": doc_id}
            card = base_card_from_doc(
                doc_id=doc_id,
                query=query,
                rank=rank,
                source=source,
                hit=hit,
                task_id=str(row.get("task_id") or ""),
                local_cache=local_cache,
                local_index_root=local_index_root,
                reader_text_chars=cfg.reader_text_chars,
            )
            if doc_id in merged:
                merge_card(merged[doc_id], card)
            else:
                merged[doc_id] = card
    for card in shared_distractor_cards:
        doc_id = parent_doc_id(card.get("doc_id"))
        if doc_id in merged:
            merge_card(merged[doc_id], card)
        else:
            merged[doc_id] = dict(card)
    cards = list(merged.values())
    cards.sort(key=lambda item: (int(item.get("best_rank") or 999), -int(item.get("hit_count") or 0), str(item.get("doc_id") or "")))
    return cards[: cfg.max_candidate_cards]


def public_card(card: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in card.items() if key != "reader_text"}


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.S)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    obj_match = re.search(r"\{.*\}", stripped, re.S)
    if obj_match:
        candidates.append(obj_match.group(0))
    last_error = "no JSON object found"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception as exc:
            last_error = str(exc)
    return None, last_error


def call_chat(cfg: ReplayConfig, prompt: str) -> tuple[str, float]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    started = time.time()
    request = Request(
        cfg.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"].get("content") or ""), time.time() - started


def run_chooser(cfg: ReplayConfig, context: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = build_doc_choose_prompt(
        numbered_questions=context["numbered_questions"],
        current_hop_number=context["hop_number"],
        current_hop_question=context["question"],
        current_answers_so_far=context["answers_so_far"],
        candidate_cards=[public_card(card) for card in cards],
        choose_top_k=cfg.choose_top_k,
        read_all_windows_max_count=cfg.read_all_windows_max_count,
        large_parent_top_windows=cfg.large_parent_top_windows,
        large_parent_neighbor_windows=cfg.large_parent_neighbor_windows,
        previously_useful_documents=[],
    )
    raw, seconds = call_chat(cfg, prompt)
    parsed, error = parse_json_object(raw)
    selected: list[str] = []
    if parsed is not None and isinstance(parsed.get("selected_doc_ids"), list):
        allowed = {str(card["doc_id"]): str(card["doc_id"]) for card in cards}
        for value in parsed["selected_doc_ids"][: cfg.choose_top_k]:
            doc_id = parent_doc_id(value)
            if doc_id in allowed and doc_id not in selected:
                selected.append(doc_id)
    return {"raw_output": raw, "latency_s": seconds, "parse_error": error, "selected_doc_ids": selected}


def read_one(cfg: ReplayConfig, context: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    document = {
        "doc_id": card["doc_id"],
        "source": card.get("source"),
        "title": card.get("title"),
        "locator": card.get("locator"),
        "text": card.get("reader_text") or card.get("snippet") or "",
    }
    prompt = build_doc_reader_prompt(
        numbered_questions=context["numbered_questions"],
        current_hop_number=context["hop_number"],
        current_hop_question=context["question"],
        current_answers_so_far=context["answers_so_far"],
        document=document,
    )
    raw, seconds = call_chat(cfg, prompt)
    parsed, error = parse_json_object(raw)
    out = {
        "doc_id": card["doc_id"],
        "source": card.get("source"),
        "title": card.get("title"),
        "locator": card.get("locator"),
        "raw_output": raw,
        "latency_s": seconds,
        "parse_error": error,
        "can_answer": False,
        "proposed_answer": "",
        "justification": "",
        "confidence": 0.0,
        "missing_information": "",
    }
    if parsed is not None:
        out.update(
            {
                "can_answer": bool(parsed.get("can_answer")),
                "proposed_answer": str(parsed.get("proposed_answer") or ""),
                "justification": str(parsed.get("justification") or ""),
                "confidence": float(parsed.get("confidence") or 0.0),
                "missing_information": str(parsed.get("missing_information") or ""),
            }
        )
    answer_match = match_answer(out["proposed_answer"], context["accepted_variants"])
    out.update({f"reader_{key}": value for key, value in answer_match.items()})
    return out


def run_readers(cfg: ReplayConfig, context: dict[str, Any], selected_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not selected_cards:
        return []
    readers: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(cfg.max_parallel_doc_reads, len(selected_cards)))) as executor:
        futures = [executor.submit(read_one, cfg, context, card) for card in selected_cards[: cfg.max_parallel_doc_reads]]
        for future in as_completed(futures):
            readers.append(future.result())
    order = {str(card["doc_id"]): idx for idx, card in enumerate(selected_cards)}
    readers.sort(key=lambda item: order.get(str(item.get("doc_id")), 999))
    return readers


def run_resolver(cfg: ReplayConfig, context: dict[str, Any], reader_results: list[dict[str, Any]], unread_cards: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = build_hop_resolve_prompt(
        numbered_questions=context["numbered_questions"],
        current_hop_number=context["hop_number"],
        current_hop_question=context["question"],
        current_answers_so_far=context["answers_so_far"],
        search_history=context["search_history"],
        reader_results=reader_results,
        unread_candidate_cards=[public_card(card) for card in unread_cards],
        choose_top_k=cfg.choose_top_k,
    )
    raw, seconds = call_chat(cfg, prompt)
    parsed, error = parse_json_object(raw)
    out = {"raw_output": raw, "latency_s": seconds, "parse_error": error, "answered": False, "answer": "", "justification": "", "next_step": ""}
    if parsed is not None:
        out.update(
            {
                "answered": bool(parsed.get("answered")),
                "answer": str(parsed.get("answer") or ""),
                "justification": str(parsed.get("justification") or ""),
                "next_step": str(parsed.get("next_step") or ""),
            }
        )
    out.update(match_answer(out["answer"], context["accepted_variants"]))
    return out


def run_mode(
    cfg: ReplayConfig,
    mode: str,
    row: dict[str, Any],
    variants: list[dict[str, Any]],
    context: dict[str, Any],
    local_cache: dict[str, dict[str, Any]],
    local_index_root: Path,
    shared_distractor_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    cards = cards_for_mode(row, variants, mode, cfg, local_cache, local_index_root, shared_distractor_cards)
    chooser = run_chooser(cfg, context, cards)
    selected_ids = set(chooser["selected_doc_ids"])
    selected_cards = [card for card in cards if card["doc_id"] in selected_ids]
    unread_cards = [card for card in cards if card["doc_id"] not in selected_ids]
    readers = run_readers(cfg, context, selected_cards)
    resolver = run_resolver(cfg, context, readers, unread_cards)
    gold_ids = set(context["gold_doc_ids"])
    card_ids = [str(card["doc_id"]) for card in cards]
    selected_card_ids = [str(card["doc_id"]) for card in selected_cards]
    return {
        "mode": mode,
        "candidate_cards": [public_card(card) for card in cards],
        "chooser": chooser,
        "reader_results": readers,
        "resolver": resolver,
        "gold_in_candidates": rank_in(card_ids, gold_ids) is not None,
        "gold_candidate_rank": rank_in(card_ids, gold_ids),
        "gold_selected": rank_in(selected_card_ids, gold_ids) is not None,
        "positive_readers": sum(1 for result in readers if result.get("can_answer")),
        "reader_reward_matches": sum(1 for result in readers if result.get("reader_reward_match")),
    }


def row_context(row: dict[str, Any], variants: list[dict[str, Any]], dataset_map: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    key = (str(row["task_id"]), str(row["chain_id"]), int(row["hop_number"]))
    ds = dataset_map[key]
    hop = ds["hop"]
    chain = ds["row"]
    accepted = [str(hop.get("answer") or "")]
    accepted.extend(str(value) for value in hop.get("accepted_answer_variants") or [])
    accepted.extend(str(value) for value in hop.get("alternate_valid_answers") or [])
    accepted = list(dict.fromkeys(value for value in accepted if value.strip()))
    gold_doc_ids: set[str] = set()
    for variant in variants:
        gold_doc_ids.update(parent_doc_id(value) for value in variant.get("gold_doc_ids_for_tuning") or [])
    if not gold_doc_ids:
        gold_doc_ids.add(parent_doc_id(row.get("gold_doc_id")))
    return {
        "task_id": str(row["task_id"]),
        "chain_id": str(row["chain_id"]),
        "hop_number": int(row["hop_number"]),
        "hop_type": str(row["hop_type"]),
        "difficulty": str(row["difficulty"]),
        "company": str(row.get("company") or ""),
        "pattern": str(row.get("pattern") or ""),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "accepted_variants": accepted,
        "gold_doc_id": str(row.get("gold_doc_id") or ""),
        "gold_doc_ids": sorted(gold_doc_ids),
        "numbered_questions": str(chain.get("numbered_questions") or ""),
        "answers_so_far": answers_so_far_text(ds["answers_so_far"]),
        "search_history": [
            {
                "type": "web_search" if row["hop_type"] == "W" else "local_document_search",
                "query": str(variant.get("query") or ""),
            }
            for variant in variants
        ],
        "queries": [str(variant.get("query") or "") for variant in variants],
    }


def run_case(cfg: ReplayConfig, row: dict[str, Any], dataset_map: dict[tuple[str, str, int], dict[str, Any]], local_index_root: Path) -> dict[str, Any]:
    variants = selected_variants(row, cfg.queries_per_hop)
    context = row_context(row, variants, dataset_map)
    local_cache: dict[str, dict[str, Any]] = {}
    shared_distractors = trace_distractor_cards(row, cfg, local_cache, local_index_root)
    context["shared_trace_distractor_count"] = len(shared_distractors)
    modes: dict[str, Any] = {}
    for mode in ["current", "bm25", "rrf"]:
        modes[mode] = run_mode(cfg, mode, row, variants, context, local_cache, local_index_root, shared_distractors)
    return {"context": context, "modes": modes}


def select_rows(rows: list[dict[str, Any]], per_bucket: int) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("actual_variants"):
            continue
        if row.get("difficulty") not in {"easy", "medium", "hard-answer", "hard-retrieval"}:
            continue
        buckets[(str(row.get("hop_type")), str(row.get("difficulty")))].append(row)
    selected: list[dict[str, Any]] = []
    for source in ["L", "W"]:
        for difficulty in ["easy", "medium", "hard-answer", "hard-retrieval"]:
            items = buckets.get((source, difficulty), [])
            items.sort(key=lambda item: (str(item.get("task_id")), str(item.get("chain_id")), int(item.get("hop_number") or 0)))
            rng.shuffle(items)
            selected.extend(items[:per_bucket])
    return selected


def summarize(cases: list[dict[str, Any]], cfg: ReplayConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_cases": len(cases),
        "model": MODEL,
        "queries_per_hop": cfg.queries_per_hop,
        "retrieval_top_k_per_query": cfg.retrieval_top_k,
        "max_candidate_cards": cfg.max_candidate_cards,
        "choose_top_k": cfg.choose_top_k,
        "max_parallel_doc_reads": cfg.max_parallel_doc_reads,
        "rrf_k": cfg.rrf_k,
        "include_trace_distractors": cfg.include_trace_distractors,
        "buckets": {},
        "overall": {},
    }
    bucket_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        ctx = case["context"]
        bucket_cases["overall"].append(case)
        bucket_cases[f"{ctx['hop_type']}|{ctx['difficulty']}"].append(case)
        bucket_cases[f"{ctx['hop_type']}"].append(case)
        bucket_cases[f"{ctx['difficulty']}"].append(case)

    def metrics(items: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "gold_in_candidates": sum(bool(item["modes"][mode].get("gold_in_candidates")) for item in items),
            "gold_selected": sum(bool(item["modes"][mode].get("gold_selected")) for item in items),
            "positive_reader": sum(int(item["modes"][mode].get("positive_readers") or 0) > 0 for item in items),
            "reader_reward_match": sum(int(item["modes"][mode].get("reader_reward_matches") or 0) > 0 for item in items),
            "resolver_exact_match": sum(bool(item["modes"][mode]["resolver"].get("exact_variant_match")) for item in items),
            "resolver_reward_match": sum(bool(item["modes"][mode]["resolver"].get("reward_match")) for item in items),
            "chooser_parse_errors": sum(bool(item["modes"][mode]["chooser"].get("parse_error")) for item in items),
            "reader_parse_errors": sum(sum(bool(result.get("parse_error")) for result in item["modes"][mode].get("reader_results") or []) for item in items),
            "resolver_parse_errors": sum(bool(item["modes"][mode]["resolver"].get("parse_error")) for item in items),
        }

    for bucket, items in sorted(bucket_cases.items()):
        entry = {"current": metrics(items, "current"), "bm25": metrics(items, "bm25"), "rrf": metrics(items, "rrf")}
        if bucket == "overall":
            summary["overall"] = entry
        else:
            summary["buckets"][bucket] = entry
    return summary


def pct(count: int, total: int) -> str:
    if not total:
        return "n/a"
    return f"{100 * count / total:.1f}% ({count}/{total})"


def summary_table(summary: dict[str, Any]) -> str:
    rows = []
    ordered_keys = ["overall", "L", "W", "easy", "medium", "hard-answer", "hard-retrieval"]
    ordered_keys.extend(f"{s}|{d}" for s in ["L", "W"] for d in ["easy", "medium", "hard-answer", "hard-retrieval"])
    for key in ordered_keys:
        entry = summary["overall"] if key == "overall" else summary["buckets"].get(key)
        if not entry:
            continue
        cur = entry["current"]
        bm25 = entry["bm25"]
        rrf = entry["rrf"]
        n = int(cur.get("n") or 0)
        rows.append(
            "<tr>"
            f"<td>{esc(key)}</td><td>{n}</td>"
            f"<td>{pct(cur['gold_in_candidates'], n)}</td><td>{pct(bm25['gold_in_candidates'], n)}</td><td>{pct(rrf['gold_in_candidates'], n)}</td>"
            f"<td>{pct(cur['gold_selected'], n)}</td><td>{pct(bm25['gold_selected'], n)}</td><td>{pct(rrf['gold_selected'], n)}</td>"
            f"<td>{pct(cur['resolver_reward_match'], n)}</td><td>{pct(bm25['resolver_reward_match'], n)}</td><td>{pct(rrf['resolver_reward_match'], n)}</td>"
            f"<td>{pct(cur['resolver_exact_match'], n)}</td><td>{pct(bm25['resolver_exact_match'], n)}</td><td>{pct(rrf['resolver_exact_match'], n)}</td>"
            "</tr>"
        )
    return "".join(rows)


def case_row(case: dict[str, Any]) -> str:
    ctx = case["context"]
    mode_cells = []
    for mode in ["current", "bm25", "rrf"]:
        data = case["modes"][mode]
        resolver = data["resolver"]
        chooser = data["chooser"]
        readers = data.get("reader_results") or []
        candidates = data.get("candidate_cards") or []
        mode_cells.append(
            f"<b>{mode}</b><br>"
            f"gold candidate rank: {esc(data.get('gold_candidate_rank'))}<br>"
            f"gold selected: {esc(data.get('gold_selected'))}<br>"
            f"selected: <code>{esc(chooser.get('selected_doc_ids'))}</code><br>"
            f"answer: <b>{esc(resolver.get('answer'))}</b><br>"
            f"reward match: {esc(resolver.get('reward_match'))}; exact: {esc(resolver.get('exact_variant_match'))}; f1={float(resolver.get('best_f1') or 0):.2f}<br>"
            f"<details><summary>candidate cards</summary><pre>{esc(json.dumps(candidates[:10], indent=2))}</pre></details>"
            f"<details><summary>reader outputs</summary><pre>{esc(json.dumps(readers, indent=2))}</pre></details>"
            f"<details><summary>chooser raw</summary><pre>{esc(chooser.get('raw_output'))}</pre></details>"
            f"<details><summary>resolver raw</summary><pre>{esc(resolver.get('raw_output'))}</pre></details>"
        )
    return (
        "<tr>"
        f"<td>{esc(ctx['hop_type'])}</td><td>{esc(ctx['difficulty'])}</td>"
        f"<td>{esc(ctx['task_id'])} {esc(ctx['chain_id'])} h{esc(ctx['hop_number'])}<br>{esc(ctx['pattern'])}</td>"
        f"<td>{esc(ctx['question'])}<br><b>gold:</b> {esc(ctx['answer'])}<br><b>accepted:</b> {esc(ctx['accepted_variants'][:12])}<br><b>queries:</b><pre>{esc(json.dumps(ctx['queries'], indent=2))}</pre></td>"
        f"<td>{mode_cells[0]}</td><td>{mode_cells[1]}</td><td>{mode_cells[2]}</td>"
        "</tr>"
    )


def build_viewer(out_dir: Path, cases: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    bm25_improving = [case for case in cases if not case["modes"]["current"]["resolver"].get("reward_match") and case["modes"]["bm25"]["resolver"].get("reward_match")]
    bm25_regressing = [case for case in cases if case["modes"]["current"]["resolver"].get("reward_match") and not case["modes"]["bm25"]["resolver"].get("reward_match")]
    improving = [case for case in cases if not case["modes"]["current"]["resolver"].get("reward_match") and case["modes"]["rrf"]["resolver"].get("reward_match")]
    regressing = [case for case in cases if case["modes"]["current"]["resolver"].get("reward_match") and not case["modes"]["rrf"]["resolver"].get("reward_match")]
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Privacy HopQA parallel RRF replay</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#fbf7ef;color:#1f2933}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:16px 0}}
td,th{{border:1px solid #d8cdbb;padding:8px;vertical-align:top}}
th{{background:#dbeee6;position:sticky;top:0}}
pre{{white-space:pre-wrap;max-height:360px;overflow:auto;background:#f6f1e8;padding:8px}}
code{{font-size:12px}}
.badge{{display:inline-block;background:#1f6f5b;color:white;padding:3px 8px;border-radius:999px;margin-right:6px}}
</style></head><body>
<h1>Privacy HopQA parallel-query RRF replay</h1>
<p>This replay uses actual observed model queries, merges multiple query result lists, adds opposite-source distractor cards from the trace when available, runs the production document chooser, reads selected documents, and resolves the hop. Previous-hop answers are gold answers to isolate current-hop retrieval and reading behavior.</p>
<p><span class='badge'>model: {esc(MODEL)}</span><span class='badge'>queries/hop: {esc(summary['queries_per_hop'])}</span><span class='badge'>doc reads: {esc(summary['max_parallel_doc_reads'])}</span></p>
<h2>Summary</h2>
<table><thead><tr><th>bucket</th><th>n</th><th>current gold in candidates</th><th>BM25 gold in candidates</th><th>RRF gold in candidates</th><th>current gold selected</th><th>BM25 gold selected</th><th>RRF gold selected</th><th>current reward match</th><th>BM25 reward match</th><th>RRF reward match</th><th>current exact</th><th>BM25 exact</th><th>RRF exact</th></tr></thead><tbody>{summary_table(summary)}</tbody></table>
<h2>Interpretation aid</h2>
<p><b>gold in candidates</b> means the gold parent document reached the merged chooser pool. <b>gold selected</b> means Qwen's chooser selected it for reading. <b>reward match</b> is exact accepted variant or token-F1 >= 0.75, matching the current PipelineRL answer-checking mode. <b>exact</b> is exact accepted-variant matching only.</p>
<h2>BM25 improves reward match ({len(bm25_improving)})</h2>
<table><thead><tr><th>source</th><th>difficulty</th><th>hop</th><th>question/query</th><th>current</th><th>BM25</th><th>RRF</th></tr></thead><tbody>{''.join(case_row(case) for case in bm25_improving)}</tbody></table>
<h2>BM25 regresses reward match ({len(bm25_regressing)})</h2>
<table><thead><tr><th>source</th><th>difficulty</th><th>hop</th><th>question/query</th><th>current</th><th>BM25</th><th>RRF</th></tr></thead><tbody>{''.join(case_row(case) for case in bm25_regressing)}</tbody></table>
<h2>RRF improves reward match ({len(improving)})</h2>
<table><thead><tr><th>source</th><th>difficulty</th><th>hop</th><th>question/query</th><th>current</th><th>BM25</th><th>RRF</th></tr></thead><tbody>{''.join(case_row(case) for case in improving)}</tbody></table>
<h2>RRF regresses reward match ({len(regressing)})</h2>
<table><thead><tr><th>source</th><th>difficulty</th><th>hop</th><th>question/query</th><th>current</th><th>BM25</th><th>RRF</th></tr></thead><tbody>{''.join(case_row(case) for case in regressing)}</tbody></table>
<h2>All cases</h2>
<table><thead><tr><th>source</th><th>difficulty</th><th>hop</th><th>question/query</th><th>current</th><th>BM25</th><th>RRF</th></tr></thead><tbody>{''.join(case_row(case) for case in cases)}</tbody></table>
</body></html>"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-rows", type=Path, default=DEFAULT_AUDIT_ROWS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--local-index-root", type=Path, default=DEFAULT_LOCAL_INDEX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--no-trace-distractors", action="store_true")
    parser.add_argument("--per-bucket", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--queries-per-hop", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--max-candidate-cards", type=int, default=20)
    parser.add_argument("--choose-top-k", type=int, default=3)
    parser.add_argument("--max-parallel-doc-reads", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--reader-text-chars", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cfg = ReplayConfig(
        base_url=args.base_url,
        retrieval_top_k=args.retrieval_top_k,
        queries_per_hop=args.queries_per_hop,
        max_candidate_cards=args.max_candidate_cards,
        choose_top_k=args.choose_top_k,
        max_parallel_doc_reads=args.max_parallel_doc_reads,
        rrf_k=args.rrf_k,
        reader_text_chars=args.reader_text_chars,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        include_trace_distractors=not args.no_trace_distractors,
        run_root=args.run_root,
        read_all_windows_max_count=10,
        large_parent_top_windows=4,
        large_parent_neighbor_windows=1,
    )
    rows = select_rows(load_jsonl(args.audit_rows), args.per_bucket)
    dataset_map = load_dataset(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_case, cfg, row, dataset_map, args.local_index_root): row for row in rows}
        for idx, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            print(f"[{idx}/{len(rows)}] {row['hop_type']} {row['difficulty']} {row['task_id']} {row['chain_id']} h{row['hop_number']}", flush=True)
            case = future.result()
            cases.append(case)
            with (args.out_dir / "replay_cases.jsonl").open("w", encoding="utf-8") as handle:
                for item in cases:
                    handle.write(json.dumps(item) + "\n")

    cases.sort(key=lambda item: (item["context"]["hop_type"], item["context"]["difficulty"], item["context"]["task_id"], item["context"]["chain_id"], item["context"]["hop_number"]))
    summary = summarize(cases, cfg)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_viewer(args.out_dir, cases, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
