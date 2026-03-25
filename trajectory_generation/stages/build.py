"""Build chains. Uses improved prompts and checks (via builder patch).

Accepts patterns, max_chains/per_task, and optional seeds_from file.
Output: chains.jsonl, append-only with resume by chain index.
"""

from __future__ import annotations

import gc
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import spacy

from trajectory_generation.src.data_loading import (
    build_doc_lookup,
    filter_seed_secrets,
    load_chunks_local,
    load_secrets,
)
from trajectory_generation.src.retrieval.hybrid import HybridSearcher
from trajectory_generation.src.format_questions import format_numbered_questions
from trajectory_generation.src.llm import LLMClient
from trajectory_generation.src.entity_index import EntityIndex
from trajectory_generation.src.seed import select_seed, select_web_seed
from trajectory_generation.src.types import ChainState, HopRecord

from trajectory_generation import config
from trajectory_generation.chain_builder import build_one_chain

logger = logging.getLogger(__name__)

# Max chars of doc text per hop in output (avoid huge JSONL)
DOC_TEXT_LIMIT_OUTPUT = 6000


def _chain_to_dict(chain) -> dict:
    """Convert Chain to JSON-serializable dict for output."""
    hops_for_format = [
        {"question": h.question, "answer": h.answer, "hop_number": h.hop_number}
        for h in chain.hop_history
    ]
    try:
        numbered = format_numbered_questions(hops_for_format)
    except ValueError as e:
        logger.warning("format_numbered_questions failed: %s", e)
        numbered = chain.hop_history[0].question if chain.hop_history else ""

    return {
        "chain_id": chain.chain_id,
        "pattern": chain.pattern,
        "numbered_questions": numbered,
        "global_question": chain.global_question,
        "global_answer": chain.global_answer,
        "hops": [
            {
                "hop_number": h.hop_number,
                "hop_type": h.hop_type,
                "question": h.question,
                "answer": h.answer,
                "doc_id": h.doc_id,
                "doc_text": h.doc_text[:DOC_TEXT_LIMIT_OUTPUT],
                "quote": h.quote,
            }
            for h in chain.hop_history
        ],
        "valid": chain.verification.is_valid if chain.verification else False,
        "metadata": {**chain.metadata, "complete": chain.metadata.get("complete", False)},
    } | (
        {"verification": {
            "is_valid": chain.verification.is_valid,
            "no_docs_pass": chain.verification.no_docs_pass,
            "first_only_pass": chain.verification.first_only_pass,
            "last_only_pass": chain.verification.last_only_pass,
            "all_docs_pass": chain.verification.all_docs_pass,
            "all_docs_answer": chain.verification.all_docs_answer,
        }}
        if chain.verification else {}
    )


def _build_chain_plan(
    load_state: dict,
    patterns: list[str],
    per_task: int | None,
    max_chains: int,
    seed: int | None,
    seeds_from: Path | None,
) -> list[dict]:
    """Build the list of (pattern, task_id, seed, ...) to process. Deterministic given seed."""
    if seeds_from and seeds_from.exists():
        return _load_seeds_from_file(seeds_from, max_chains)

    manifest = load_state.get("manifest", load_state)
    task_ids = manifest.get("task_ids", [])
    if not task_ids:
        raise ValueError("No tasks in load state")

    rng = random.Random(seed)
    records = []
    idx = 0
    if per_task:
        for pattern in patterns:
            for task_id in task_ids:
                for _ in range(per_task):
                    records.append({
                        "idx": idx, "pattern": pattern, "task_id": task_id,
                        "seed": rng.randint(0, config.MAX_RANDOM_SEED - 1),
                    })
                    idx += 1
    else:
        for _ in range(max_chains):
            records.append({
                "idx": idx,
                "pattern": rng.choice(patterns),
                "task_id": rng.choice(task_ids),
                "seed": rng.randint(0, config.MAX_RANDOM_SEED - 1),
            })
            idx += 1
    return records[:max_chains] if max_chains else records


def _load_seeds_from_file(path: Path, max_chains: int | None) -> list[dict]:
    """Load chain specs from JSONL (e.g. from extract_bad_seeds for smoke test)."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in %s line %d: %s", path, i + 1, e)
                continue
            rec["idx"] = rec.get("idx", i)
            records.append(rec)
            if max_chains and len(records) >= max_chains:
                break
    return records


def run_build(
    run_dir: Path,
    load_state: dict,
    patterns: list[str],
    per_task: int | None,
    max_chains: int,
    seed: int | None,
    seeds_from: Path | None,
    model: str,
    base_url: str,
    api_key: str | None,
    workers: int = 1,
    resume: bool = True,
) -> Path:
    """Build chains. Returns path to chains.jsonl."""
    manifest = load_state.get("manifest", load_state)
    seeds = _build_chain_plan(load_state, patterns, per_task, max_chains, seed, seeds_from)
    chunks_local = Path(manifest["chunks_local"])
    # Load local docs and secrets, group by task
    chunks_web_paths = [Path(p) for p in manifest["chunks_web"]]
    secret_inventory = Path(manifest["secret_inventory"])
    entity_index_dir = Path(manifest["entity_index_dir"])
    spacy_model = manifest["spacy_model"]

    chunks = load_chunks_local(chunks_local)
    doc_lookup = build_doc_lookup(chunks)
    local_docs = {did: doc.text for did, doc in doc_lookup.items()}
    local_doc_ids = set(local_docs.keys())

    secrets = load_secrets(secret_inventory)
    eligible = filter_seed_secrets(secrets, doc_lookup)
    task_local_doc_ids: dict[str, set[str]] = {}
    for did in local_doc_ids:
        doc = doc_lookup.get(did)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_local_doc_ids.setdefault(tid, set()).add(did)

    task_secrets: dict[str, list] = defaultdict(list)
    for s in eligible:
        doc = doc_lookup.get(s.doc_id)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_secrets[tid].append(s)

    web_docs = _load_web_docs(chunks_web_paths)
    web_doc_ids = set(web_docs.keys())
    all_docs = {**local_docs, **web_docs}

    # BM25 searchers for bridge finding (matches chain_builder)
    local_searcher = None
    web_searcher = None
    if chunks_local.exists():
        local_searcher = HybridSearcher(chunks_path=chunks_local)
        logger.info("Local BM25 searcher: %d chunks", local_searcher.size)
    existing_web = [p for p in chunks_web_paths if p.exists()]
    if existing_web:
        web_searcher = HybridSearcher.from_paths(existing_web)
        logger.info("Web BM25 searcher: %d chunks", web_searcher.size)

    # Build or load entity index for NER-based bridge finding
    nlp = spacy.load(spacy_model)
    cache_path = entity_index_dir / f"entity_index_{spacy_model}.json"
    if cache_path.exists():
        entity_index = EntityIndex.load(cache_path, all_docs, nlp=nlp)
    else:
        entity_index = EntityIndex(nlp, all_docs)
        entity_index.save(cache_path)

    secret_by_entity = defaultdict(list)
    for secret in secrets:
        doc_entities = entity_index.entities_in_doc(secret.doc_id)
        for ent in doc_entities:
            if len(ent) >= 3 and ent.lower() in secret.question.lower():
                secret_by_entity[ent.lower()].append((secret.doc_id, secret))

    del local_docs, web_docs, all_docs, chunks
    gc.collect()

    llm = LLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        min_max_tokens=config.DEFAULT_MAX_TOKENS,
    )

    output_path = run_dir / "chains.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_indices: set[int] = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    idx = d.get("metadata", {}).get("seed_idx")
                    if idx is not None:
                        done_indices.add(idx)
                except json.JSONDecodeError:
                    pass
        logger.info("Resume: %d chains already built", len(done_indices))

    for seed_rec in seeds:
        idx = seed_rec.get("idx", -1)
        if idx in done_indices:
            continue

        pattern = seed_rec["pattern"]
        task_id = seed_rec["task_id"]
        task_seed = seed_rec.get("seed", 0)
        forced_hop = seed_rec.get("forced_hop")
        task_rng = random.Random(task_seed)
        chain_local_ids = task_local_doc_ids.get(task_id)
        chain_secrets = task_secrets.get(task_id, [])
        if not chain_local_ids:
            logger.warning("No docs for task %s, skipping seed %d", task_id, idx)
            continue

        doc0 = next(iter(chain_local_ids))
        chain_company = doc_lookup[doc0].meta.get("company_name", "?") if doc0 in doc_lookup else "?"

        try:
            if forced_hop:
                doc_id = forced_hop["doc_id"]
                doc = doc_lookup.get(doc_id)
                if not doc:
                    logger.warning("forced_hop doc_id %s not found, skipping", doc_id)
                    continue
                hop = HopRecord(
                    hop_number=1,
                    hop_type=pattern[0],
                    question=forced_hop["question"],
                    answer=forced_hop["answer"],
                    doc_id=doc_id,
                    doc_text=doc.text,
                )
                state = ChainState(
                    pattern=pattern,
                    hop_history=[hop],
                    global_question=forced_hop["question"],
                    global_answer=forced_hop["answer"],
                    used_doc_ids={doc_id},
                    task_id=task_id,
                    company=chain_company,
                )
            elif pattern[0] == "L":
                state = select_seed(
                    chain_secrets, doc_lookup,
                    pattern=pattern,
                    task_id=task_id,
                    company=chain_company,
                    rng=task_rng,
                )
            else:
                next_pool = chain_local_ids if pattern[1] == "L" else web_doc_ids
                state = select_web_seed(
                    entity_index, web_doc_ids, next_pool, llm,
                    pattern=pattern,
                    task_id=task_id,
                    company=chain_company,
                    rng=task_rng,
                )
        except ValueError as e:
            logger.warning("Seed failed for %s #%d: %s", pattern, idx, e)
            continue
        except Exception as e:
            logger.exception("Unexpected error building seed %s #%d: %s", pattern, idx, e)
            continue

        chain = build_one_chain(
            state=state,
            llm=llm,
            entity_index=entity_index,
            local_doc_ids=chain_local_ids,
            web_doc_ids=web_doc_ids,
            record_trace=True,
            local_searcher=local_searcher,
            web_searcher=web_searcher,
            retrieval_k=config.RETRIEVAL_K,
            secret_by_entity=secret_by_entity,
        )

        d = _chain_to_dict(chain)
        d["metadata"]["seed_idx"] = idx
        d["metadata"]["seed_pattern"] = pattern
        d["metadata"]["seed_task_id"] = task_id

        with open(output_path, "a") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

        status = "VALID" if chain.verification and chain.verification.is_valid else "INVALID"
        logger.info("%s %s #%d %s [%s %s]", status, pattern, idx, chain.chain_id, task_id, chain_company)

    return output_path


def _load_web_docs(paths: list[Path]) -> dict[str, str]:
    docs: dict[str, list[tuple[int, str]]] = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                doc_id = row.get("doc_id", row.get("chunk_id", ""))
                text = row.get("text", "")
                offset = (row.get("offsets") or {}).get("start", 0)
                if doc_id and text:
                    docs.setdefault(doc_id, []).append((offset, text))

    result = {}
    for doc_id, chunks in docs.items():
        chunks.sort(key=lambda x: x[0])
        result[doc_id] = "\n\n".join(t for _, t in chunks)
    return result
