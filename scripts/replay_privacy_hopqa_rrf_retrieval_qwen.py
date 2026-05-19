#!/usr/bin/env python3
"""Replay Privacy HopQA reader/resolve prompts under current vs RRF retrieval ordering.

This is a diagnostic, not a training/eval harness. It uses observed model search
queries from a retrieval-method audit, selects representative hops by
source+difficulty, and asks a local OpenAI-compatible Qwen endpoint to read the
current top-5 documents versus RRF top-5 documents.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
import argparse
import html
import importlib.util
import json
import re
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

PROMPTS_PATH = Path("/home/toolkit/my_code/PipelineRL/pipelinerl/domains/privacy_hopqa/prompts.py")
PROMPTS_SPEC = importlib.util.spec_from_file_location("privacy_hopqa_prompts", PROMPTS_PATH)
if PROMPTS_SPEC is None or PROMPTS_SPEC.loader is None:
    raise RuntimeError(f"Could not load prompt builders from {PROMPTS_PATH}")
PROMPTS_MODULE = importlib.util.module_from_spec(PROMPTS_SPEC)
sys.modules[PROMPTS_SPEC.name] = PROMPTS_MODULE
PROMPTS_SPEC.loader.exec_module(PROMPTS_MODULE)
build_doc_reader_prompt = PROMPTS_MODULE.build_doc_reader_prompt
build_hop_resolve_prompt = PROMPTS_MODULE.build_hop_resolve_prompt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_ROWS = REPO / "runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14_large240/retrieval_method_rows.jsonl"
DEFAULT_DATASET = REPO / "data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl"
DEFAULT_OUT_DIR = REPO / "runtime_analysis/privacy_hopqa_rrf_replay_qwen4b_2026-05-14"
DEFAULT_LOCAL_INDEX_ROOT = Path("/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/local_indices_goldcoverage_20260519")
FULL_DOCS_DIR = Path("/home/toolkit/full_docs")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        task_id = str(row.get("task_id") or "")
        chain_id = str(row.get("chain_id") or "")
        answers = []
        for hop in row.get("hops") or []:
            hop_number = int(hop.get("hop_number") or 0)
            out[(task_id, chain_id, hop_number)] = {
                "row": row,
                "hop": hop,
                "answers_so_far": list(answers),
            }
            answers.append({"hop_number": hop_number, "answer": str(hop.get("answer") or "")})
    return out


def current_answers_so_far(answers: list[dict[str, Any]]) -> str:
    if not answers:
        return "No previous answers yet."
    return "\n".join(f"({item['hop_number']}): {item['answer']}" for item in answers)


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def answer_matches(answer: str, variants: list[str]) -> bool:
    got = normalize_answer(answer)
    return any(got == normalize_answer(v) for v in variants if str(v).strip())


def rank_in(doc_ids: list[str], gold_ids: set[str]) -> int | None:
    for idx, doc_id in enumerate(doc_ids, start=1):
        if doc_id in gold_ids:
            return idx
    return None


def rrf_ranked_docs(dense_doc_ids: list[str], lexical_doc_ids: list[str], *, rrf_k: int = 20) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for idx, doc_id in enumerate(dense_doc_ids[:50], start=1):
        scores[str(doc_id)] += 1.0 / (rrf_k + idx)
    for idx, doc_id in enumerate(lexical_doc_ids[:200], start=1):
        scores[str(doc_id)] += 1.0 / (rrf_k + idx)
    return [doc_id for doc_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def choose_variant(row: dict[str, Any]) -> dict[str, Any] | None:
    variants = row.get("actual_variants") or []
    if not variants:
        return None

    def key(variant: dict[str, Any]) -> tuple[int, int, int]:
        gold = set(variant.get("gold_doc_ids_for_tuning") or [])
        current_rank = rank_in([str(x) for x in variant.get("dense_doc_ids") or []], gold) or 999
        rrf_rank = rank_in(rrf_ranked_docs([str(x) for x in variant.get("dense_doc_ids") or []], [str(x) for x in variant.get("lexical_doc_ids") or []]), gold) or 999
        improved_to_top5 = int(current_rank > 5 and rrf_rank <= 5)
        return (-improved_to_top5, rrf_rank, current_rank)

    return sorted(variants, key=key)[0]


def select_rows(rows: list[dict[str, Any]], per_bucket: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any], int, int]]] = defaultdict(list)
    for row in rows:
        variant = choose_variant(row)
        if variant is None:
            continue
        gold = set(variant.get("gold_doc_ids_for_tuning") or [])
        dense_docs = [str(x) for x in variant.get("dense_doc_ids") or []]
        rrf_docs = rrf_ranked_docs(dense_docs, [str(x) for x in variant.get("lexical_doc_ids") or []])
        current_rank = rank_in(dense_docs, gold) or 999
        rrf_rank = rank_in(rrf_docs, gold) or 999
        buckets[(str(row.get("hop_type")), str(row.get("difficulty")))].append((row, variant, current_rank, rrf_rank))
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for source in ["L", "W"]:
        for difficulty in ["easy", "medium", "hard-answer", "hard-retrieval"]:
            items = buckets.get((source, difficulty), [])
            items.sort(key=lambda item: (item[3] > 5, item[3], item[2], str(item[0].get("task_id")), str(item[0].get("chain_id"))))
            selected.extend((row, variant) for row, variant, _current, _rrf in items[:per_bucket])
    return selected


def load_local_documents(task_id: str, cache: dict[str, dict[str, Any]], local_index_root: Path) -> dict[str, Any]:
    if task_id in cache:
        return cache[task_id]
    docs_path = local_index_root / task_id / "documents.json"
    data = json.loads(docs_path.read_text(encoding="utf-8"))
    cache[task_id] = data
    return data


def doc_payload(doc_id: str, row: dict[str, Any], local_cache: dict[str, dict[str, Any]], local_index_root: Path) -> dict[str, Any] | None:
    source = str(row.get("hop_type"))
    if source == "W":
        raw = doc_id.split("/", 1)[1] if doc_id.startswith("web/") else doc_id
        path = FULL_DOCS_DIR / f"{raw}.txt"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        return {"doc_id": doc_id, "title": path.name, "locator": str(path), "text": text[:20000]}
    docs = load_local_documents(str(row.get("task_id")), local_cache, local_index_root)
    item = docs.get(doc_id)
    if not isinstance(item, dict):
        return None
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    locator = metadata.get("file_path") or metadata.get("original_path") or metadata.get("source") or doc_id
    return {"doc_id": doc_id, "title": Path(str(locator)).name, "locator": locator, "text": str(item.get("content") or "")[:20000]}


def parse_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.S)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    obj_match = re.search(r"\{.*\}", stripped, re.S)
    if obj_match:
        candidates.append(obj_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value, None
        except Exception as exc:
            last_error = str(exc)
    return None, last_error if 'last_error' in locals() else 'no JSON object found'


def call_chat(base_url: str, prompt: str, *, max_tokens: int = 4096, temperature: float = 0.7) -> tuple[str, float]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started = time.time()
    req = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"].get("content") or ""), time.time() - started


def read_one(args: tuple[str, str, str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    base_url, mode, doc_id, document, context, prompt = args
    raw, seconds = call_chat(base_url, prompt, max_tokens=4096, temperature=0.7)
    parsed, error = parse_json(raw)
    out = {
        "mode": mode,
        "doc_id": doc_id,
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
        out.update({
            "can_answer": bool(parsed.get("can_answer")),
            "proposed_answer": str(parsed.get("proposed_answer") or ""),
            "justification": str(parsed.get("justification") or ""),
            "confidence": float(parsed.get("confidence") or 0.0),
            "missing_information": str(parsed.get("missing_information") or ""),
        })
    return out


def resolve_one(base_url: str, mode: str, context: dict[str, Any], reader_results: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = build_hop_resolve_prompt(
        numbered_questions=context["numbered_questions"],
        current_hop_number=context["hop_number"],
        current_hop_question=context["question"],
        current_answers_so_far=context["answers_so_far"],
        search_history=[{"type": context["action_type"], "query": context["query"]}],
        reader_results=reader_results,
        unread_candidate_cards=[],
        choose_top_k=3,
    )
    raw, seconds = call_chat(base_url, prompt, max_tokens=512, temperature=0.7)
    parsed, error = parse_json(raw)
    out = {"mode": mode, "raw_output": raw, "latency_s": seconds, "parse_error": error, "answered": False, "answer": "", "justification": "", "next_step": ""}
    if parsed is not None:
        out.update({
            "answered": bool(parsed.get("answered")),
            "answer": str(parsed.get("answer") or ""),
            "justification": str(parsed.get("justification") or ""),
            "next_step": str(parsed.get("next_step") or ""),
        })
    out["matches_gold"] = answer_matches(out["answer"], context["accepted_variants"])
    return out


def run_case(base_url: str, row: dict[str, Any], variant: dict[str, Any], dataset_map: dict[tuple[str, str, int], dict[str, Any]], local_cache: dict[str, dict[str, Any]], local_index_root: Path) -> dict[str, Any]:
    key = (str(row["task_id"]), str(row["chain_id"]), int(row["hop_number"]))
    ds = dataset_map[key]
    chain = ds["row"]
    hop = ds["hop"]
    dense_docs = [str(x) for x in variant.get("dense_doc_ids") or []]
    lexical_docs = [str(x) for x in variant.get("lexical_doc_ids") or []]
    rrf_docs = rrf_ranked_docs(dense_docs, lexical_docs)
    gold_ids = set(variant.get("gold_doc_ids_for_tuning") or [])
    context = {
        "task_id": row["task_id"],
        "chain_id": row["chain_id"],
        "hop_number": int(row["hop_number"]),
        "hop_type": row["hop_type"],
        "difficulty": row["difficulty"],
        "question": row["question"],
        "answer": row["answer"],
        "accepted_variants": list(dict.fromkeys([str(hop.get("answer") or "")] + [str(v) for v in hop.get("accepted_answer_variants") or []])),
        "numbered_questions": str(chain.get("numbered_questions") or ""),
        "answers_so_far": current_answers_so_far(ds["answers_so_far"]),
        "query": str(variant.get("query") or ""),
        "action_type": "web_search" if row["hop_type"] == "W" else "local_document_search",
        "gold_doc_ids": sorted(gold_ids),
        "current_rank": rank_in(dense_docs, gold_ids),
        "rrf_rank": rank_in(rrf_docs, gold_ids),
        "current_top5": dense_docs[:5],
        "rrf_top5": rrf_docs[:5],
    }
    case = {"context": context, "modes": {}}
    read_jobs = []
    for mode, doc_ids in [("current", context["current_top5"]), ("rrf", context["rrf_top5"] )]:
        for doc_id in doc_ids:
            doc = doc_payload(doc_id, row, local_cache, local_index_root)
            if doc is None:
                continue
            prompt = build_doc_reader_prompt(
                numbered_questions=context["numbered_questions"],
                current_hop_number=context["hop_number"],
                current_hop_question=context["question"],
                current_answers_so_far=context["answers_so_far"],
                document=doc,
            )
            read_jobs.append((base_url, mode, doc_id, doc, context, prompt))
    reads_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(read_one, job) for job in read_jobs]
        for future in as_completed(futures):
            result = future.result()
            reads_by_mode[result["mode"]].append(result)
    for mode in ["current", "rrf"]:
        reads = sorted(reads_by_mode.get(mode, []), key=lambda item: context[f"{mode}_top5"].index(item["doc_id"]) if item["doc_id"] in context[f"{mode}_top5"] else 999)
        resolver = resolve_one(base_url, mode, context, reads)
        case["modes"][mode] = {
            "reader_results": reads,
            "resolver": resolver,
            "any_reader_gold_match": any(answer_matches(str(r.get("proposed_answer") or ""), context["accepted_variants"]) for r in reads if r.get("can_answer")),
            "positive_readers": sum(1 for r in reads if r.get("can_answer")),
        }
    return case


def build_viewer(out_dir: Path, cases: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rows = []
    for case in cases:
        ctx = case["context"]
        mode_cells = []
        for mode in ["current", "rrf"]:
            data = case["modes"].get(mode, {})
            resolver = data.get("resolver") or {}
            readers = data.get("reader_results") or []
            mode_cells.append(
                f"<b>{mode}</b><br>rank={esc(ctx.get(mode + '_rank'))}<br>answer={esc(resolver.get('answer'))}<br>match={esc(resolver.get('matches_gold'))}<br>positive_readers={esc(data.get('positive_readers'))}<br>docs=<code>{esc(ctx.get(mode + '_top5'))}</code><details><summary>reader outputs</summary>"
                + "<br>".join(f"<code>{esc(r.get('doc_id'))}</code> can={esc(r.get('can_answer'))} ans={esc(r.get('proposed_answer'))} conf={esc(r.get('confidence'))}<br>{esc(r.get('justification'))}" for r in readers)
                + f"</details><details><summary>resolver raw</summary><pre>{esc(resolver.get('raw_output'))}</pre></details>"
            )
        rows.append(
            "<tr>"
            f"<td>{esc(ctx['hop_type'])}</td><td>{esc(ctx['difficulty'])}</td><td>{esc(ctx['task_id'])} {esc(ctx['chain_id'])} h{esc(ctx['hop_number'])}</td>"
            f"<td>{esc(ctx['question'])}<br><b>gold:</b> {esc(ctx['answer'])}<br><b>query:</b> {esc(ctx['query'])}</td>"
            f"<td>{mode_cells[0]}</td><td>{mode_cells[1]}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Privacy HopQA RRF Qwen replay</title>
<style>body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#f8f4ec;color:#1f2933}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}td,th{{border:1px solid #d8cdbb;padding:8px;vertical-align:top}}th{{background:#e8f3ef}}pre{{white-space:pre-wrap;max-height:360px;overflow:auto}}code{{font-size:12px}}</style></head>
<body><h1>Privacy HopQA RRF Qwen replay</h1><p>Local model: {esc(MODEL)}. This replay uses observed model queries and compares current helper ordering against RRF top-5 documents. It uses gold previous-hop answers to isolate retrieval/reader behavior.</p>
<pre>{esc(json.dumps(summary, indent=2))}</pre>
<table><thead><tr><th>source</th><th>difficulty</th><th>hop</th><th>question/query</th><th>current top5 replay</th><th>RRF top5 replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-rows", type=Path, default=DEFAULT_AUDIT_ROWS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--local-index-root", type=Path, default=DEFAULT_LOCAL_INDEX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-bucket", type=int, default=2)
    args = parser.parse_args()

    rows = load_jsonl(args.audit_rows)
    selected = select_rows(rows, args.per_bucket)
    dataset_map = load_dataset(args.dataset)
    local_cache: dict[str, dict[str, Any]] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for idx, (row, variant) in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {row['hop_type']} {row['difficulty']} {row['task_id']} {row['chain_id']} h{row['hop_number']}", flush=True)
        cases.append(run_case(args.base_url, row, variant, dataset_map, local_cache, args.local_index_root))
        with (args.out_dir / "replay_cases.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case) + "\n")

    summary: dict[str, Any] = {"n_cases": len(cases), "per_bucket": args.per_bucket, "model": MODEL, "buckets": {}}
    for case in cases:
        ctx = case["context"]
        key = f"{ctx['hop_type']}|{ctx['difficulty']}"
        bucket = summary["buckets"].setdefault(key, {"n": 0, "current_match": 0, "rrf_match": 0, "current_gold_top5": 0, "rrf_gold_top5": 0})
        bucket["n"] += 1
        bucket["current_match"] += int(bool(case["modes"]["current"]["resolver"].get("matches_gold")))
        bucket["rrf_match"] += int(bool(case["modes"]["rrf"]["resolver"].get("matches_gold")))
        bucket["current_gold_top5"] += int(ctx.get("current_rank") is not None and int(ctx["current_rank"]) <= 5)
        bucket["rrf_gold_top5"] += int(ctx.get("rrf_rank") is not None and int(ctx["rrf_rank"]) <= 5)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_viewer(args.out_dir, cases, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
