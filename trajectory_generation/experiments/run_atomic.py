#!/usr/bin/env python3
"""Run chain-building steps atomically and save raw outputs for inspection.

Steps: 0 load → 1 seed → 2 find_bridge → 3 intra+pick → 4 checks → 5 rank.
Each step writes prompts, raw LLM output, and parsed results. Use to debug
where instructions are unclear or validation fails.

Usage:
  # Full LW chain (seed → find_bridge → pick → checks → rank)
  python -m trajectory_generation.experiments.run_atomic --output-dir /tmp/atomic_exp --task DR0001 --pattern LW --seed 42

  # Run only specific steps
  python -m trajectory_generation.experiments.run_atomic --output-dir /tmp/atomic_exp --task DR0001 --steps 1,2,3

  # W seed (needs LLM for seed selection)
  python -m trajectory_generation.experiments.run_atomic --output-dir /tmp/atomic_exp --task DR0001 --pattern WL --seed 42
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

from trajectory_generation.src.data_loading import (
    build_doc_lookup,
    filter_seed_secrets,
    load_chunks_local,
    load_secrets,
)
from trajectory_generation.src.llm import LLMClient
from trajectory_generation.src.entity_index import EntityIndex
from trajectory_generation.src.find_bridge import find_bridge, sort_entities_by_proximity
from trajectory_generation.src.retrieval.hybrid import HybridSearcher
from trajectory_generation.src.seed import select_seed, select_web_seed
from trajectory_generation.src.types import ChainState, HopRecord

from trajectory_generation import config
from trajectory_generation.chain_builder import build_one_chain
from trajectory_generation.checks import (
    check_answer_needs_backref,
    check_answerable_without_doc,
    check_question,
    check_search_leaks_bridge,
)
from trajectory_generation.questions import (
    generate_question_constrained,
    generate_question_pick,
    rank_questions,
)

logger = logging.getLogger(__name__)


def _save(out_dir: Path, name: str, data: dict | str, ext: str = "json") -> None:
    path = out_dir / f"{name}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", path)


def _save_raw(out_dir: Path, name: str, text: str) -> None:
    _save(out_dir, name, text, ext="txt")


def _load(out_dir: Path, name: str, ext: str = "json") -> dict | None:
    path = out_dir / f"{name}.{ext}"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_step_0_load(
    out_dir: Path,
    task_id: str | None,
    use_bm25: bool = True,
    web_chunks_override: list[Path] | None = None,
) -> dict:
    """Load data. Returns manifest and loaded objects. Builds BM25 searchers when use_bm25=True."""
    chunks_local = config.CHUNKS_LOCAL
    chunks_web_paths = (
        web_chunks_override
        if web_chunks_override is not None
        else [Path(p) for p in config.CHUNKS_WEB if Path(p).exists()]
    )
    secret_inventory = config.SECRET_INVENTORY
    entity_index_dir = config.ENTITY_INDEX_DIR

    chunks = load_chunks_local(chunks_local)
    doc_lookup = build_doc_lookup(chunks)
    local_docs = {did: doc.text for did, doc in doc_lookup.items()}
    local_doc_ids = set(local_docs.keys())

    secrets = load_secrets(secret_inventory)
    eligible = filter_seed_secrets(secrets, doc_lookup, task_id=task_id)

    task_local_doc_ids: dict[str, set[str]] = {}
    for did in local_doc_ids:
        doc = doc_lookup.get(did)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_local_doc_ids.setdefault(tid, set()).add(did)

    task_secrets: dict[str, list] = {}
    for s in eligible:
        doc = doc_lookup.get(s.doc_id)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_secrets.setdefault(tid, []).append(s)

    web_docs: dict[str, list[tuple[int, str]]] = {}
    for p in chunks_web_paths:
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                doc_id = row.get("doc_id", row.get("chunk_id", ""))
                text = row.get("text", "")
                offset = (row.get("offsets") or {}).get("start", 0)
                if doc_id and text:
                    web_docs.setdefault(doc_id, []).append((offset, text))
    web_docs_merged: dict[str, str] = {}
    for doc_id, chunks in web_docs.items():
        chunks.sort(key=lambda x: x[0])
        web_docs_merged[doc_id] = "\n\n".join(t for _, t in chunks)
    web_docs = web_docs_merged

    web_doc_ids = set(web_docs.keys())
    all_docs = {**local_docs, **web_docs}

    import spacy
    nlp = spacy.load(config.SPACY_MODEL)
    cache_path = entity_index_dir / f"entity_index_{config.SPACY_MODEL}.json"
    if cache_path.exists():
        entity_index = EntityIndex.load(cache_path, all_docs, nlp=nlp)
    else:
        entity_index = EntityIndex(nlp, all_docs)
        entity_index.save(cache_path)

    local_searcher = None
    web_searcher = None
    if use_bm25:
        logger.info("Building BM25 indexes...")
        if chunks_local.exists():
            local_searcher = HybridSearcher(chunks_path=chunks_local)
            logger.info("Local searcher: %d chunks", local_searcher.size)
        existing_web = [p for p in chunks_web_paths if p.exists()]
        if existing_web:
            web_searcher = HybridSearcher.from_paths(existing_web)
            logger.info("Web searcher: %d chunks", web_searcher.size)

    manifest = {
        "chunks_local": str(chunks_local),
        "n_local_docs": len(local_doc_ids),
        "n_web_docs": len(web_doc_ids),
        "n_secrets_eligible": len(eligible),
        "task_ids": sorted(task_local_doc_ids.keys()),
        "bm25_enabled": use_bm25,
    }
    _save(out_dir, "00_manifest", manifest)

    return {
        "manifest": manifest,
        "doc_lookup": doc_lookup,
        "local_doc_ids": local_doc_ids,
        "web_doc_ids": web_doc_ids,
        "task_local_doc_ids": task_local_doc_ids,
        "task_secrets": task_secrets,
        "entity_index": entity_index,
        "local_searcher": local_searcher,
        "web_searcher": web_searcher,
    }


def run_step_1_seed(
    out_dir: Path,
    load_data: dict,
    pattern: str,
    task_id: str,
    seed: int,
) -> ChainState:
    """Select seed (L or W). Saves seed state and for W: prompt + raw LLM response."""
    task_local_ids = load_data["task_local_doc_ids"].get(task_id)
    task_secrets = load_data["task_secrets"].get(task_id, [])
    doc_lookup = load_data["doc_lookup"]
    entity_index = load_data["entity_index"]
    web_doc_ids = load_data["web_doc_ids"]

    doc0 = next(iter(task_local_ids))
    company = doc_lookup[doc0].meta.get("company_name", "?") if doc0 in doc_lookup else "?"
    rng = random.Random(seed)

    if pattern[0] == "L":
        state = select_seed(
            task_secrets, doc_lookup,
            pattern=pattern, task_id=task_id, company=company, rng=rng,
        )
        _save(out_dir, "01_seed_l", {
            "pattern": pattern,
            "task_id": task_id,
            "company": company,
            "seed": seed,
            "question": state.global_question,
            "answer": state.global_answer,
            "doc_id": state.hop_history[0].doc_id,
        })
    else:
        next_pool = task_local_ids if pattern[1] == "L" else web_doc_ids
        llm = LLMClient(
            model=config.DEFAULT_MODEL,
            base_url=config.DEFAULT_BASE_URL,
            min_max_tokens=config.DEFAULT_MAX_TOKENS,
        )
        from trajectory_generation.src.seed import PROMPT_WEB_SEED
        web_list = list(web_doc_ids)
        rng.shuffle(web_list)
        candidates = []
        for web_id in web_list[:100]:
            for ent in entity_index.entities_in_doc(web_id):
                if entity_index.docs_containing_text(ent, pool=next_pool):
                    candidates.append((web_id, ent))
        rng.shuffle(candidates)
        web_doc_id, entity = candidates[0]
        doc_text = entity_index.doc_text(web_doc_id)[:8000]
        prompt = PROMPT_WEB_SEED.format(entity=entity, doc_text=doc_text)
        _save_raw(out_dir, "01_seed_w_prompt", prompt)
        raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1024)
        _save_raw(out_dir, "01_seed_w_raw", raw)
        state = select_web_seed(
            entity_index, web_doc_ids, next_pool, llm,
            pattern=pattern, task_id=task_id, company=company, rng=rng,
        )
        _save(out_dir, "01_seed_w", {
            "pattern": pattern,
            "question": state.global_question,
            "answer": state.global_answer,
            "doc_id": state.hop_history[0].doc_id,
        })

    return state


def run_step_2_find_bridge(
    out_dir: Path,
    load_data: dict,
    state: ChainState,
    pattern: str,
    task_id: str,
) -> list:
    """Find bridge candidates. Saves candidates list."""
    src_type, dst_type = pattern[0], pattern[1]
    if dst_type == "L":
        next_pool = load_data["task_local_doc_ids"].get(task_id, set())
    else:
        next_pool = load_data["web_doc_ids"]
    current_doc_id = state.hop_history[-1].doc_id
    current_answer = state.global_answer
    used_doc_ids = state.used_doc_ids
    used_answers = {a.lower() for a in [state.global_answer]}

    entity_index = load_data["entity_index"]
    dst_searcher = load_data.get("local_searcher") if dst_type == "L" else load_data.get("web_searcher")
    candidates = find_bridge(
        entity_index, current_doc_id, current_answer,
        next_pool, used_doc_ids, used_answers,
        searcher=dst_searcher, retrieval_k=config.RETRIEVAL_K,
    )

    _save(out_dir, "02_find_bridge", {
        "pattern": pattern,
        "current_doc_id": current_doc_id,
        "current_answer": current_answer,
        "bm25_used": dst_searcher is not None,
        "n_candidates": len(candidates),
        "candidates": [
            {
                "target_doc_id": c.target_doc_id,
                "bridge_entity": c.bridge_entity,
                "needs_intra": c.needs_intra,
                "score": c.score,
            }
            for c in candidates[:10]
        ],
    })
    return candidates


def run_step_3_intra(
    out_dir: Path,
    load_data: dict,
    cand,
    state: ChainState,
    jump_idx: int,
) -> tuple[str, str, str] | None:
    """Generate intra question. Saves prompt, raw, parsed. Returns (q, a, quote) or None."""
    from trajectory_generation.prompts import PROMPT_CONSTRAINED

    entity_index = load_data["entity_index"]
    current_doc_id = state.hop_history[-1].doc_id
    current_answer = state.global_answer
    bridge_entity = cand.bridge_entity
    target_doc_text = entity_index.doc_text(cand.target_doc_id)
    current_doc_text = entity_index.doc_text(current_doc_id)

    def _focused_context(text: str, term: str, window: int = 4000) -> str:
        idx = text.lower().find(term.lower())
        if idx == -1:
            return text[:window]
        half = window // 2
        start = max(0, idx - half)
        end = min(len(text), idx + len(term) + half)
        return text[start:end]

    intra_context = _focused_context(current_doc_text, current_answer)
    prompt = PROMPT_CONSTRAINED.format(
        prev_answer=current_answer,
        target_answer=bridge_entity,
        doc_text=intra_context[:8000],
    )
    _save_raw(out_dir, "03a_intra_prompt", prompt)

    llm = LLMClient(
        model=config.DEFAULT_MODEL,
        base_url=config.DEFAULT_BASE_URL,
        min_max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    _save_raw(out_dir, "03a_intra_raw", raw)

    from trajectory_generation.src.parsing import parse_qa_with_quote
    result = parse_qa_with_quote(raw)
    if result is None:
        _save(out_dir, "03a_intra_parsed", {"error": "parse failed", "raw_preview": raw[:500]})
        return None
    q, a, quote = result
    _save(out_dir, "03a_intra_parsed", {"question": q, "answer": a, "quote": quote})
    return (q, a, quote)


def run_step_3_pick(
    out_dir: Path,
    load_data: dict,
    cand,
    state: ChainState,
    prev_answer: str,
) -> tuple[str, str, str] | None:
    """Generate pick question. Saves prompt, raw, parsed."""
    from trajectory_generation.prompts import PROMPT_PICK
    from trajectory_generation.src.entity_index import is_entity

    entity_index = load_data["entity_index"]
    target_doc_id = cand.target_doc_id
    bridge_entity = cand.bridge_entity
    target_doc_text = entity_index.doc_text(target_doc_id)

    def _focused_context(text: str, term: str, window: int = 4000) -> str:
        idx = text.lower().find(term.lower())
        if idx == -1:
            return text[:window]
        half = window // 2
        start = max(0, idx - half)
        end = min(len(text), idx + len(term) + half)
        return text[start:end]

    used_answers = {h.answer.lower() for h in state.hop_history}
    target_entities = entity_index.entities_in_doc(target_doc_id)
    filtered = [
        e for e in target_entities
        if e.lower() != bridge_entity.lower()
        and e.lower() not in used_answers
        and len(e) >= 3
        and is_entity(e)
    ]
    entity_list = sort_entities_by_proximity(target_doc_text, bridge_entity, filtered)[:20]

    inter_context = _focused_context(target_doc_text, bridge_entity)
    entity_str = "\n".join(f"- {e}" for e in entity_list)
    prompt = PROMPT_PICK.format(
        prev_answer=prev_answer,
        entity_list=entity_str,
        doc_text=inter_context[:8000],
    )
    _save_raw(out_dir, "03b_pick_prompt", prompt)

    llm = LLMClient(
        model=config.DEFAULT_MODEL,
        base_url=config.DEFAULT_BASE_URL,
        min_max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    _save_raw(out_dir, "03b_pick_raw", raw)

    from trajectory_generation.src.parsing import parse_qa_with_quote
    result = parse_qa_with_quote(raw)
    if result is None:
        _save(out_dir, "03b_pick_parsed", {"error": "parse failed", "raw_preview": raw[:500]})
        return None
    q, a, quote = result
    _save(out_dir, "03b_pick_parsed", {
        "question": q, "answer": a, "quote": quote,
        "entity_list": entity_list,
    })
    return (q, a, quote)


def run_step_4_checks(
    out_dir: Path,
    question: str,
    answer: str,
    required_phrase: str,
    prev_answer: str,
    doc_text: str,
    quote: str,
    *,
    allowed_answers: set[str] | None = None,
) -> dict:
    """Run 4a–4d checks: question validity, triviality, backref dependency, search leak.
    Saves prompt/raw for 4c (backref) so you can inspect the blanked question and response.
    """
    llm = LLMClient(
        model=config.DEFAULT_MODEL,
        base_url=config.DEFAULT_BASE_URL,
        min_max_tokens=config.DEFAULT_MAX_TOKENS,
    )

    err = check_question(
        question, answer,
        required_phrase=required_phrase,
        allowed_answers=allowed_answers,
        prev_answers=[prev_answer],
        quote=quote,
        doc_text=doc_text,
    )
    _save(out_dir, "04a_check_question", {
        "question": question,
        "answer": answer,
        "required_phrase": required_phrase,
        "passed": err is None,
        "error": err,
    })

    trivial = check_answerable_without_doc(question, llm, correct_answer=answer)
    _save(out_dir, "04b_check_trivial", asdict(trivial))

    from trajectory_generation.checks import BACKREF_CHECK_PROMPT, _parse_answer_justification
    from trajectory_generation.checks import BackrefCheckResult
    placeholder = "an unknown entity"
    blanked = question.replace(prev_answer, placeholder)
    if blanked == question:
        backref = BackrefCheckResult(independent=False, answer=None, justification="prev_answer not in question")
        _save_raw(out_dir, "04c_backref_prompt", "(skipped: prev_answer not in question)")
        _save_raw(out_dir, "04c_backref_raw", "")
    else:
        backref_prompt = BACKREF_CHECK_PROMPT.format(question=blanked, placeholder=placeholder)
        _save_raw(out_dir, "04c_backref_prompt", backref_prompt)
        backref_raw = llm.chat(
            [{"role": "user", "content": backref_prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        _save_raw(out_dir, "04c_backref_raw", backref_raw)
        answer_line, justification = _parse_answer_justification(backref_raw)
        backref = BackrefCheckResult(
            independent=answer_line.upper() != "NOT_ANSWERABLE",
            answer=answer_line if answer_line.upper() != "NOT_ANSWERABLE" else None,
            justification=justification,
        )
    _save(out_dir, "04c_check_backref", {
        "original_question": question,
        "blanked_question": blanked,
        "prev_answer_replaced": required_phrase,
        **asdict(backref),
    })

    leak = check_search_leaks_bridge(question, required_phrase, llm)
    _save(out_dir, "04d_check_search_leak", {"would_leak": leak.would_leak, "queries": leak.queries, "justification": leak.justification})

    return {
        "check_question": err is None,
        "trivial": trivial.trivial,
        "backref_independent": backref.independent,
        "would_leak": leak.would_leak,
    }


def run_step_5_rank(
    out_dir: Path,
    candidates: list[dict],
    prev_answer: str,
) -> list[int]:
    """Rank candidates. Saves prompt, raw, ranking."""
    from trajectory_generation.prompts import PROMPT_RANK

    cand_str = "\n".join(
        f"{i + 1}. Q: {c['question']}\n   A: {c['answer']}"
        for i, c in enumerate(candidates)
    )
    prompt = PROMPT_RANK.format(prev_answer=prev_answer, candidates=cand_str)
    _save_raw(out_dir, "05_rank_prompt", prompt)

    llm = LLMClient(
        model=config.DEFAULT_MODEL,
        base_url=config.DEFAULT_BASE_URL,
        min_max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=config.DEFAULT_TEMPERATURE,
        max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    _save_raw(out_dir, "05_rank_raw", raw)

    import re
    numbers = [int(x) for x in re.findall(r'\d+', raw)]
    indices = [n - 1 for n in numbers if 1 <= n <= len(candidates)]
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    for i in range(len(candidates)):
        if i not in seen:
            unique.append(i)

    _save(out_dir, "05_rank_parsed", {"ranking": unique, "raw_preview": raw[:300]})
    return unique


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Run chain-building steps atomically")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/atomic_exp"))
    p.add_argument("--task", default="DR0001")
    p.add_argument("--pattern", default="LW")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", default=None, help="Comma-separated step numbers to run (0-5), default all")
    p.add_argument("--candidate-idx", type=int, default=0, help="Bridge candidate index to use for step 3")
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--base-url", default=config.DEFAULT_BASE_URL)
    p.add_argument("--no-bm25", action="store_true", help="Disable BM25 (exhaustive search, faster load, less realistic)")
    p.add_argument(
        "--web-chunks",
        type=Path,
        nargs="+",
        default=None,
        help="Use only these web chunk files (faster load; default: full config.CHUNKS_WEB)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.DEFAULT_MODEL = args.model
    config.DEFAULT_BASE_URL = args.base_url

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = set()
    if args.steps:
        steps = {int(x) for x in args.steps.split(",")}
    else:
        steps = {0, 1, 2, 3, 4, 5}

    load_data = None
    state = None
    candidates = None

    if 0 in steps:
        logger.info("Step 0: Load")
        load_data = run_step_0_load(
            out_dir, args.task,
            use_bm25=not args.no_bm25,
            web_chunks_override=args.web_chunks,
        )

    if 1 in steps:
        logger.info("Step 1: Seed")
        if load_data is None:
            load_data = run_step_0_load(
                out_dir, args.task,
                use_bm25=not args.no_bm25,
                web_chunks_override=args.web_chunks,
            )
        state = run_step_1_seed(
            out_dir, load_data, args.pattern, args.task, args.seed,
        )

    if 2 in steps:
        logger.info("Step 2: Find bridge")
        if load_data is None:
            load_data = run_step_0_load(
                out_dir, args.task,
                use_bm25=not args.no_bm25,
                web_chunks_override=args.web_chunks,
            )
        if state is None:
            state = run_step_1_seed(out_dir, load_data, args.pattern, args.task, args.seed)
        candidates = run_step_2_find_bridge(out_dir, load_data, state, args.pattern, args.task)

    if 3 in steps:
        logger.info("Step 3: Question generation (intra + pick)")
        if load_data is None:
            load_data = run_step_0_load(
                out_dir, args.task,
                use_bm25=not args.no_bm25,
                web_chunks_override=args.web_chunks,
            )
        if state is None:
            state = run_step_1_seed(out_dir, load_data, args.pattern, args.task, args.seed)
        if candidates is None:
            candidates = run_step_2_find_bridge(out_dir, load_data, state, args.pattern, args.task)

        cand = candidates[args.candidate_idx]
        prev_answer = state.global_answer
        jump_idx = 0

        if cand.needs_intra:
            result = run_step_3_intra(out_dir, load_data, cand, state, jump_idx)
            if result:
                prev_answer = result[1]

        result = run_step_3_pick(out_dir, load_data, cand, state, prev_answer)
        if result:
            q, a, quote = result

            if 4 in steps:
                logger.info("Step 4: Checks")
                doc_text = load_data["entity_index"].doc_text(cand.target_doc_id)
                pick_parsed = _load(out_dir, "03b_pick_parsed")
                allowed = None
                if pick_parsed and "entity_list" in pick_parsed:
                    allowed = {e.strip().lower() for e in pick_parsed["entity_list"]}
                run_step_4_checks(
                    out_dir, q, a,
                    required_phrase=cand.bridge_entity if cand.needs_intra else state.global_answer,
                    prev_answer=prev_answer,
                    doc_text=doc_text,
                    quote=quote,
                    allowed_answers=allowed,
                )

            if 5 in steps and result:
                logger.info("Step 5: Rank (single candidate, skipped)")
                _save(out_dir, "05_rank_skipped", {"reason": "single candidate"})

    summary = {
        "task": args.task,
        "pattern": args.pattern,
        "seed": args.seed,
        "steps_run": sorted(steps),
        "output_dir": str(out_dir),
    }
    _save(out_dir, "SUMMARY", summary)
    logger.info("Outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
