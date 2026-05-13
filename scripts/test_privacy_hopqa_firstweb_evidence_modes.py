#!/usr/bin/env python3
"""Test Privacy HopQA answerability when the model is given gold evidence.

This script is intentionally diagnostic and does not mutate datasets.

For each selected hop it runs two checks:

1. correct_window: show the model the reader-size document window containing
   the dataset supporting span.
2. all_windows: read every reader-size window from the gold document in
   parallel, then ask a resolver to answer from those reader results.

The input may be the first-web-hop extraction or the full multi-hop dataset.
For the full dataset, every hop is audited independently against its gold
document. The goal is to identify unfair reward candidates where the gold
document contains the answer but the actor still receives a negative reward
because of answer variants, underspecified questions, chunking, or resolver
behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import types

import aiohttp

from audit_privacy_hopqa_hop_answerability_stepfun import extract_document_text


DEFAULT_DATASET = Path(
    "/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset/"
    "privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/toolkit/my_code/MosaicProject/runtime_analysis/"
    "privacy_hopqa_gold_evidence_modes_2026-05-11_variantfix2_questionfix2"
)
DEFAULT_LOCAL_INDEX_ROOT = Path(
    os.getenv(
        "PRIVACY_AGENT_LOCAL_INDEX_ROOT",
        "/home/toolkit/my_code/PipelineRL/.privacy_agent/local_indices",
    )
)
DEFAULT_PIPELINERL_ROOT = Path(os.getenv("PIPELINERL_ROOT", "/home/toolkit/my_code/PipelineRL"))
DEFAULT_BASE_URLS = [
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8080",
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8081",
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8082",
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8083",
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8084",
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8085",
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8086",
    "http://dns-d888b689-cf5f-40d5-8eba-12fcf101469c-phrl87204e9c-1:8087",
]


@dataclass(frozen=True)
class Case:
    row_id: str
    chain_id: str
    hop_number: int
    task_id: str
    company: str
    numbered_questions: str
    current_answers_so_far: str
    question: str
    gold_answer: str
    accepted_variants: list[str]
    doc_id: str
    doc_ref: str
    supporting_span: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for row in read_jsonl(path):
        hops = row.get("hops") or []
        numbered_questions = str(row.get("numbered_questions") or row.get("global_question") or "")
        for hop_index, hop in enumerate(hops, start=1):
            hop_number = int(hop.get("hop_number") or hop_index)
            gold = str(hop.get("answer") or row.get("global_answer") or "")
            variants = [gold]
            for key in ("accepted_answer_variants", "alternate_valid_answers"):
                for value in hop.get(key) or []:
                    text = str(value)
                    if text and text not in variants:
                        variants.append(text)
            if hop_index == len(hops):
                for value in row.get("global_answer_variants") or []:
                    text = str(value)
                    if text and text not in variants:
                        variants.append(text)
            row_id = str(row.get("id") or row.get("chain_id") or "")
            prior_answers = []
            for prior_index, prior_hop in enumerate(hops[: hop_index - 1], start=1):
                prior_number = int(prior_hop.get("hop_number") or prior_index)
                prior_answer = str(prior_hop.get("answer") or "").strip()
                if prior_answer:
                    prior_answers.append(f"({prior_number}) {prior_answer}")
            cases.append(
                Case(
                    row_id=f"{row_id}::hop{hop_number}",
                    chain_id=str(row.get("chain_id") or ""),
                    hop_number=hop_number,
                    task_id=str(row.get("task_id") or ""),
                    company=str(row.get("company") or ""),
                    numbered_questions=numbered_questions,
                    current_answers_so_far="\n".join(prior_answers) if prior_answers else "None yet.",
                    question=str(hop.get("question") or row.get("global_question") or ""),
                    gold_answer=gold,
                    accepted_variants=variants,
                    doc_id=str(hop.get("doc_id") or ""),
                    doc_ref=str(hop.get("doc_ref") or ""),
                    supporting_span=str(hop.get("supporting_span") or ""),
                )
            )
    return cases


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


def token_f1(predicted: str | None, truth: str | None) -> float:
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


def match_answer(answer: str, variants: list[str], f1_threshold: float) -> tuple[bool, float, str]:
    normalized = normalize_answer(answer)
    normalized_variants = [normalize_answer(value) for value in variants if normalize_answer(value)]
    if normalized and normalized in normalized_variants:
        return True, 1.0, "exact"
    best = max((token_f1(answer, value) for value in variants), default=0.0)
    if best >= f1_threshold:
        return True, best, "f1"
    return False, best, "miss"


_CONTENT_PROCESSOR: Any | None = None
_PROMPT_BUILDERS: tuple[Any, Any] | None = None
_PRODUCTION_IMPORT_ERROR: str | None = None


def install_pipelinerl_privacy_namespace(pipelinerl_root: Path) -> None:
    """Expose the privacy slices without importing PipelineRL's top-level dispatcher.

    Importing ``pipelinerl.domains...`` normally executes ``pipelinerl/domains/__init__.py``,
    which pulls in the training dispatcher and heavy ML dependencies that are irrelevant for
    this diagnostic. The production content processor and privacy_hopqa prompt builders are
    pure Python modules under those package paths, so this installs lightweight namespace
    packages pointing at the same source files.
    """

    root = pipelinerl_root.expanduser()
    package_paths = {
        "pipelinerl": root / "pipelinerl",
        "pipelinerl.domains": root / "pipelinerl" / "domains",
        "pipelinerl.domains.privacy_agent": root / "pipelinerl" / "domains" / "privacy_agent",
        "pipelinerl.domains.privacy_agent.drbench": root
        / "pipelinerl"
        / "domains"
        / "privacy_agent"
        / "drbench",
        "pipelinerl.domains.privacy_agent.drbench.agent_tools": root
        / "pipelinerl"
        / "domains"
        / "privacy_agent"
        / "drbench"
        / "agent_tools",
        "pipelinerl.domains.privacy_hopqa": root / "pipelinerl" / "domains" / "privacy_hopqa",
    }
    for name, path in package_paths.items():
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
        elif not hasattr(module, "__path__"):
            raise RuntimeError(f"Cannot install package namespace for non-package module {name}")


def production_content_processor(pipelinerl_root: Path, workspace_dir: Path) -> Any | None:
    global _CONTENT_PROCESSOR
    global _PRODUCTION_IMPORT_ERROR
    if _CONTENT_PROCESSOR is not None:
        return _CONTENT_PROCESSOR
    root = pipelinerl_root.expanduser()
    if not root.exists():
        _PRODUCTION_IMPORT_ERROR = f"PipelineRL root does not exist: {root}"
        return None
    try:
        install_pipelinerl_privacy_namespace(root)
        content_processor_module = importlib.import_module(
            "pipelinerl.domains.privacy_agent.drbench.agent_tools.content_processor"
        )
        config_module = importlib.import_module("pipelinerl.domains.privacy_agent.drbench.config")
        ContentProcessor = content_processor_module.ContentProcessor
        RunConfig = config_module.RunConfig
    except Exception as exc:
        _PRODUCTION_IMPORT_ERROR = repr(exc)
        return None
    _CONTENT_PROCESSOR = ContentProcessor(
        workspace_dir=str(workspace_dir),
        model="diagnostic",
        llm_adapter=None,
        run_config=RunConfig(no_web=True),
        vector_store=None,
        session_cache=None,
    )
    return _CONTENT_PROCESSOR


def production_prompt_builders(pipelinerl_root: Path) -> tuple[Any, Any] | None:
    global _PROMPT_BUILDERS
    global _PRODUCTION_IMPORT_ERROR
    if _PROMPT_BUILDERS is not None:
        return _PROMPT_BUILDERS
    root = pipelinerl_root.expanduser()
    if not root.exists():
        _PRODUCTION_IMPORT_ERROR = f"PipelineRL root does not exist: {root}"
        return None
    try:
        install_pipelinerl_privacy_namespace(root)
        prompts_module = importlib.import_module(
            "pipelinerl.domains.privacy_hopqa.prompts"
        )
        build_doc_reader_prompt = prompts_module.build_doc_reader_prompt
        build_hop_resolve_prompt = prompts_module.build_hop_resolve_prompt
    except Exception as exc:
        _PRODUCTION_IMPORT_ERROR = repr(exc)
        return None
    _PROMPT_BUILDERS = (build_doc_reader_prompt, build_hop_resolve_prompt)
    return _PROMPT_BUILDERS


def read_text(path: str, args: argparse.Namespace | None = None) -> tuple[str, str]:
    ref = Path(path).expanduser()
    if not ref.exists():
        return "", "missing_file"
    if args is not None:
        processor = production_content_processor(args.pipelinerl_root, args.output_dir / "_doc_extract_workspace")
        if processor is not None:
            text = processor.extract_text_from_file(ref)
            extraction_failed = (
                text.startswith("Error extracting text")
                or "library not available" in text
                or "Install " in text and " to process " in text
            )
            if text.strip() and not extraction_failed:
                return text, "production_content_processor"
        if ref.suffix.lower() == ".txt":
            text = ref.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text, "plain_text_fallback"
        if args.require_production_stack:
            raise RuntimeError(f"Production content processor unavailable for {ref}: {_PRODUCTION_IMPORT_ERROR}")
    return extract_document_text(ref), "stepfun_extract_document_text_fallback"


_LOCAL_INDEX_CACHE: dict[Path, dict[str, Any] | None] = {}


def load_local_index_documents(task_id: str, local_index_root: Path) -> dict[str, Any] | None:
    path = local_index_root.expanduser() / task_id / "documents.json"
    if path not in _LOCAL_INDEX_CACHE:
        if not path.exists():
            _LOCAL_INDEX_CACHE[path] = None
        else:
            _LOCAL_INDEX_CACHE[path] = json.loads(path.read_text(encoding="utf-8"))
    return _LOCAL_INDEX_CACHE[path]


def local_index_content(case: Case, local_index_root: Path) -> str | None:
    if not case.doc_id.startswith("local/") or not case.task_id:
        return None
    documents = load_local_index_documents(case.task_id, local_index_root)
    if not documents:
        return None
    doc_path = str(Path(case.doc_ref).expanduser())
    doc_name = Path(case.doc_ref).name
    doc_stem = Path(case.doc_ref).stem
    doc_rel = "/".join(Path(case.doc_ref).parts[-2:])
    matches: list[str] = []
    for doc_info in documents.values():
        metadata = doc_info.get("metadata") or {}
        candidates = [
            metadata.get("file_path"),
            metadata.get("original_path"),
            metadata.get("source_identifier"),
            metadata.get("extracted_path"),
            metadata.get("relative_path"),
            metadata.get("filename"),
        ]
        candidate_text = " ".join(str(value) for value in candidates if value)
        if (
            doc_path in candidate_text
            or doc_rel in candidate_text
            or doc_name in candidate_text
            or doc_stem in candidate_text
        ):
            content = str(doc_info.get("content") or "")
            if content.strip():
                title = str(metadata.get("title") or metadata.get("subject") or metadata.get("filename") or "").strip()
                if title:
                    matches.append(f"### {title}\n{content}")
                else:
                    matches.append(content)
    if matches:
        return "\n\n---\n\n".join(matches)
    return None


def read_case_text(case: Case, args: argparse.Namespace) -> tuple[str, str]:
    content = local_index_content(case, args.local_index_root)
    if content is not None:
        return content, "local_index_documents_json"
    return read_text(case.doc_ref, args)


def find_span(text: str, span: str) -> tuple[int, str]:
    if not text or not span.strip():
        return -1, "missing_span"
    pos = text.find(span)
    if pos >= 0:
        return pos, "exact"
    compact = " ".join(span.split())
    for chars in (400, 240, 160, 100, 60):
        needle = compact[:chars]
        if needle:
            pos = text.find(needle)
            if pos >= 0:
                return pos, f"prefix_{chars}"
    return -1, "not_found"


def split_windows(text: str, window_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    if not text:
        return []
    step = max(1, window_chars - overlap_chars)
    count = max(1, math.ceil(max(1, len(text) - overlap_chars) / step))
    windows: list[dict[str, Any]] = []
    for idx in range(count):
        start = idx * step
        end = min(len(text), start + window_chars)
        if start >= len(text):
            break
        windows.append(
            {
                "window_id": f"w{idx + 1:02d}of{count:02d}",
                "window_index": idx + 1,
                "window_count": count,
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )
    return windows


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def document_for_window(case: Case, window: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": f"{case.doc_id}#{window['window_id']}",
        "source": case.doc_id.split("/", 1)[0] if "/" in case.doc_id else "",
        "title": Path(case.doc_ref).name or case.doc_id,
        "locator": case.doc_ref,
        "queries": [],
        "excerpt": window["text"],
        "parent_doc_id": case.doc_id,
        "evidence_window_id": window["window_id"],
        "window_index": window["window_index"],
        "window_count": window["window_count"],
        "window_char_start": window["start"],
        "window_char_end": window["end"],
    }


def reader_prompt(case: Case, window: dict[str, Any], args: argparse.Namespace) -> str:
    builders = production_prompt_builders(args.pipelinerl_root)
    if builders is not None:
        build_doc_reader_prompt, _ = builders
        return build_doc_reader_prompt(
            numbered_questions=case.numbered_questions,
            current_hop_number=case.hop_number,
            current_hop_question=case.question,
            current_answers_so_far=case.current_answers_so_far,
            document=document_for_window(case, window),
        )
    if args.require_production_stack:
        raise RuntimeError(f"Production prompt builders unavailable: {_PRODUCTION_IMPORT_ERROR}")
    return f"""You are testing whether a document evidence window can answer a Privacy HopQA question.

Use only this evidence window. Do not use outside knowledge.
Answer only the question being asked.
Return the shortest answer possible, usually under 5 words.
The answer will be string matched, so do not write a full sentence.

Question:
{case.question}

Document id: {case.doc_id}#{window['window_id']}
Window position: {window['window_index']} of {window['window_count']}

Evidence:
<<<EVIDENCE
{window['text']}
EVIDENCE

Return valid JSON only:
{{
  "can_answer": true,
  "proposed_answer": "short answer",
  "justification": "brief support using only this evidence",
  "confidence": 0.9,
  "missing_information": ""
}}

If the evidence is insufficient, set "can_answer" to false.
"""


def resolver_prompt(case: Case, reader_results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    builders = production_prompt_builders(args.pipelinerl_root)
    if builders is not None:
        _, build_hop_resolve_prompt = builders
        compact_results = []
        for row in reader_results:
            compact_results.append(
                {
                    "doc_id": row.get("doc_id"),
                    "source": case.doc_id.split("/", 1)[0] if "/" in case.doc_id else "",
                    "title": Path(case.doc_ref).name or case.doc_id,
                    "locator": case.doc_ref,
                    "can_answer": row.get("can_answer"),
                    "proposed_answer": row.get("proposed_answer"),
                    "justification": row.get("justification"),
                    "confidence": row.get("confidence"),
                    "missing_information": row.get("missing_information"),
                    "parent_doc_id": case.doc_id,
                    "evidence_window_id": row.get("window_id"),
                    "window_index": row.get("window_index"),
                    "window_count": row.get("window_count"),
                }
            )
        return build_hop_resolve_prompt(
            numbered_questions=case.numbered_questions,
            current_hop_number=case.hop_number,
            current_hop_question=case.question,
            current_answers_so_far=case.current_answers_so_far,
            search_history=[],
            reader_results=compact_results,
            unread_candidate_cards=[],
            choose_top_k=0,
        )
    if args.require_production_stack:
        raise RuntimeError(f"Production prompt builders unavailable: {_PRODUCTION_IMPORT_ERROR}")
    compact_results = [
        {
            "window_id": row.get("window_id"),
            "can_answer": row.get("can_answer"),
            "proposed_answer": row.get("proposed_answer"),
            "justification": row.get("justification"),
            "confidence": row.get("confidence"),
            "missing_information": row.get("missing_information"),
        }
        for row in reader_results
    ]
    return f"""Resolve the answer to a Privacy HopQA question using only document-reading results.

Question:
{case.question}

Document-reading results:
{json.dumps(compact_results, indent=2, ensure_ascii=False)}

Return the shortest answer possible, usually under 5 words.
The answer will be string matched, so do not write a full sentence.

Return valid JSON only:
{{
  "answered": true,
  "answer": "short answer",
  "reason": "brief reason based only on the reader results",
  "selected_window_ids": ["w01of03"]
}}

If the reader results are insufficient, set "answered" to false and answer to "".
"""


class EndpointPool:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.next_index = 0
        self.lock = asyncio.Lock()

    async def next_url(self) -> str:
        async with self.lock:
            url = self.urls[self.next_index % len(self.urls)]
            self.next_index += 1
            return url


async def call_model(
    session: aiohttp.ClientSession,
    pool: EndpointPool,
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, dict[str, Any], str]:
    url = await pool.next_url()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started = time.time()
    try:
        async with session.post(
            f"{url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as response:
            body = await response.text()
            response.raise_for_status()
            data = json.loads(body)
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            usage["latency_s"] = round(time.time() - started, 3)
            return content, usage, url
    except Exception as exc:
        return "", {"error": repr(exc), "latency_s": round(time.time() - started, 3)}, url


async def read_window(
    case: Case,
    window: dict[str, Any],
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    pool: EndpointPool,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt = reader_prompt(case, window, args)
    async with sem:
        raw, usage, url = await call_model(
            session,
            pool,
            args.model,
            prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
    payload = parse_json_object(raw)
    answer = str(payload.get("proposed_answer") or "").strip()
    matched, score, method = match_answer(answer, case.accepted_variants, args.f1_threshold)
    return {
        "window_id": window["window_id"],
        "window_index": window["window_index"],
        "window_count": window["window_count"],
        "start": window["start"],
        "end": window["end"],
        "can_answer": bool(payload.get("can_answer")) and bool(answer),
        "proposed_answer": answer,
        "match": matched,
        "match_score": score,
        "match_method": method,
        "justification": str(payload.get("justification") or "").strip(),
        "confidence": payload.get("confidence"),
        "missing_information": str(payload.get("missing_information") or "").strip(),
        "raw_output": raw,
        "usage": usage,
        "base_url": url,
    }


async def run_case(
    index: int,
    case: Case,
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    pool: EndpointPool,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    text, doc_read_method = read_case_text(case, args)
    span_pos, span_method = find_span(text, case.supporting_span)
    windows = split_windows(text, args.window_chars, args.overlap_chars)
    correct_window = None
    if windows and span_pos >= 0:
        correct_window = next((w for w in windows if w["start"] <= span_pos < w["end"]), None)
    if correct_window is None and windows:
        correct_window = windows[0]
    if correct_window is None:
        correct_window = {
            "window_id": "supporting_span_only",
            "window_index": 1,
            "window_count": 1,
            "start": 0,
            "end": len(case.supporting_span),
            "text": case.supporting_span,
        }
        windows = [correct_window]

    correct_result = await read_window(case, correct_window, args, session, pool, sem)
    limited_windows = windows[: args.max_windows_per_doc] if args.max_windows_per_doc else windows
    all_results = await asyncio.gather(
        *(read_window(case, window, args, session, pool, sem) for window in limited_windows)
    )
    resolver_raw, resolver_usage, resolver_url = await call_model(
        session,
        pool,
        args.model,
        resolver_prompt(case, all_results, args),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )
    resolver_payload = parse_json_object(resolver_raw)
    resolved_answer = str(resolver_payload.get("answer") or "").strip()
    resolver_match, resolver_score, resolver_method = match_answer(
        resolved_answer,
        case.accepted_variants,
        args.f1_threshold,
    )
    return {
        "row_index": index,
        "row_id": case.row_id,
        "chain_id": case.chain_id,
        "hop_number": case.hop_number,
        "task_id": case.task_id,
        "company": case.company,
        "question": case.question,
        "gold_answer": case.gold_answer,
        "accepted_variants": case.accepted_variants,
        "doc_id": case.doc_id,
        "doc_ref": case.doc_ref,
        "doc_read_method": doc_read_method,
        "doc_chars": len(text),
        "span_pos": span_pos,
        "span_method": span_method,
        "window_chars": args.window_chars,
        "overlap_chars": args.overlap_chars,
        "total_windows": len(windows),
        "tested_windows": len(limited_windows),
        "correct_window": correct_result,
        "any_window_match": any(row.get("match") for row in all_results),
        "any_window_can_answer": any(row.get("can_answer") for row in all_results),
        "all_window_results": all_results,
        "resolver": {
            "answered": bool(resolver_payload.get("answered")) and bool(resolved_answer),
            "answer": resolved_answer,
            "match": resolver_match,
            "match_score": resolver_score,
            "match_method": resolver_method,
            "reason": str(resolver_payload.get("reason") or "").strip(),
            "selected_window_ids": resolver_payload.get("selected_window_ids")
            if isinstance(resolver_payload.get("selected_window_ids"), list)
            else [],
            "raw_output": resolver_raw,
            "usage": resolver_usage,
            "base_url": resolver_url,
        },
    }


async def main_async(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "evidence_mode_results.jsonl"
    summary_path = args.output_dir / "summary.json"
    config_path = args.output_dir / "config.json"

    cases = load_cases(args.dataset)
    if args.row_id_file:
        for line in args.row_id_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                args.chain_id.append(line)
    if args.chain_id:
        wanted = set(args.chain_id)
        cases = [
            case
            for case in cases
            if case.chain_id in wanted
            or case.row_id in wanted
            or f"{case.chain_id}::hop{case.hop_number}" in wanted
        ]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    existing_records: list[dict[str, Any]] = []
    completed_indexes: set[int] = set()
    if args.resume_existing and all_path.exists():
        for line in all_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            existing_records.append(record)
            completed_indexes.add(int(record["row_index"]))
    else:
        all_path.write_text("", encoding="utf-8")

    pending_cases = [
        (index, case)
        for index, case in enumerate(cases)
        if index not in completed_indexes
    ]

    config_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "output_dir": str(args.output_dir),
                "model": args.model,
                "base_urls": args.base_urls,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "window_chars": args.window_chars,
                "overlap_chars": args.overlap_chars,
                "max_windows_per_doc": args.max_windows_per_doc,
                "local_index_root": str(args.local_index_root),
                "pipelinerl_root": str(args.pipelinerl_root),
                "case_count": len(cases),
                "existing_case_count": len(existing_records),
                "pending_case_count": len(pending_cases),
                "case_concurrency": args.case_concurrency,
                "f1_threshold": args.f1_threshold,
                "resume_existing": args.resume_existing,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pool = EndpointPool(args.base_urls)
    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()
    records: list[dict[str, Any]] = list(existing_records)
    write_lock = asyncio.Lock()
    case_sem = asyncio.Semaphore(args.case_concurrency)
    async with aiohttp.ClientSession() as session:
        async def worker(index: int, case: Case) -> None:
            async with case_sem:
                record = await run_case(index, case, args, session, pool, sem)
            async with write_lock:
                records.append(record)
                with all_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed = len(records)
                if completed % args.log_every == 0:
                    correct_matches = sum(1 for row in records if row["correct_window"]["match"])
                    resolver_matches = sum(1 for row in records if row["resolver"]["match"])
                    print(
                        f"completed={completed}/{len(cases)} "
                        f"correct_window_match={correct_matches} "
                        f"resolver_match={resolver_matches}",
                        flush=True,
                    )

        if existing_records:
            print(
                f"resuming with existing={len(existing_records)} pending={len(pending_cases)} total={len(cases)}",
                flush=True,
            )
        await asyncio.gather(*(worker(index, case) for index, case in pending_cases))

    summary = {
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "total": len(records),
        "correct_window_match": sum(1 for row in records if row["correct_window"]["match"]),
        "correct_window_can_answer": sum(1 for row in records if row["correct_window"]["can_answer"]),
        "any_window_match": sum(1 for row in records if row["any_window_match"]),
        "any_window_can_answer": sum(1 for row in records if row["any_window_can_answer"]),
        "resolver_match": sum(1 for row in records if row["resolver"]["match"]),
        "resolver_answered": sum(1 for row in records if row["resolver"]["answered"]),
        "span_located": sum(1 for row in records if row["span_pos"] >= 0),
        "doc_read_methods": dict(Counter(row.get("doc_read_method", "unknown") for row in records)),
        "elapsed_s": round(time.time() - started, 3),
        "f1_threshold": args.f1_threshold,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", dest="base_urls", action="append", default=[])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--case-concurrency", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--chain-id", action="append", default=[])
    parser.add_argument("--row-id-file", type=Path)
    parser.add_argument("--window-chars", type=int, default=16000)
    parser.add_argument("--overlap-chars", type=int, default=1000)
    parser.add_argument("--max-windows-per-doc", type=int, default=0)
    parser.add_argument("--local-index-root", type=Path, default=DEFAULT_LOCAL_INDEX_ROOT)
    parser.add_argument("--pipelinerl-root", type=Path, default=DEFAULT_PIPELINERL_ROOT)
    parser.add_argument("--require-production-stack", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--f1-threshold", type=float, default=0.75)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if not args.base_urls:
        args.base_urls = DEFAULT_BASE_URLS
    args.dataset = args.dataset.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.local_index_root = args.local_index_root.expanduser()
    args.pipelinerl_root = args.pipelinerl_root.expanduser()
    return args


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
