"""Build chains for the standalone trajectory generator.

Accepts patterns, max_chains/per_task, and optional seeds_from file.
Output: chains.jsonl, append-only with resume by chain index.
"""

import gc
import json
import logging
import random
import re
from difflib import SequenceMatcher
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import spacy

from trajectory_generation.src.data_loading import (
    build_doc_lookup,
    filter_seed_secrets,
    load_chunks_local,
    load_secrets,
)
from trajectory_generation.src.retrieval.browsecomp import (
    BrowseCompBM25Searcher,
    BrowseCompHybridSearcher,
)
from trajectory_generation.src.retrieval.composite import FilteredSearcher, WeightedRRFSearcher
from trajectory_generation.src.retrieval.dense import OpenAICompatibleBackend
from trajectory_generation.src.retrieval.search_client import SearchClient
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


def _task_web_searcher_for_task(
    *,
    task_id: str,
    curated_doc_ids_by_task: dict[str, set[str]],
    curated_web_doc_ids: set[str],
    browsecomp_searcher: Any,
    curated_searcher: Any,
) -> Any:
    chain_curated_ids = curated_doc_ids_by_task.get(task_id, set()) & curated_web_doc_ids
    searchers: list[tuple[Any, float]] = []
    if browsecomp_searcher is not None:
        searchers.append((browsecomp_searcher, config.BROWSECOMP_WEB_WEIGHT))
    if curated_searcher is not None and chain_curated_ids:
        searchers.append((
            FilteredSearcher(curated_searcher, allowed_doc_ids=chain_curated_ids),
            config.CURATED_WEB_WEIGHT,
        ))
    if len(searchers) == 1:
        return searchers[0][0]
    if searchers:
        return WeightedRRFSearcher(searchers, rrf_k=config.RRF_K)
    return None


def _build_seed_result(
    *,
    idx: int,
    pattern: str,
    task_id: str,
    task_seed: int,
    forced_hop: dict | None,
    chain_local_ids: set[str],
    chain_web_doc_ids: set[str],
    chain_secrets: list,
    chain_company: str,
    doc_lookup: dict[str, Any],
    llm: LLMClient,
    entity_index: EntityIndex,
    local_searcher: Any,
    task_web_searcher: Any,
    secret_by_entity: dict[str, list],
    local_doc_family_map: dict[str, str],
) -> dict | None:
    task_rng = random.Random(task_seed)

    try:
        if forced_hop:
            doc_id = forced_hop["doc_id"]
            doc = doc_lookup.get(doc_id)
            if not doc:
                logger.warning("forced_hop doc_id %s not found, skipping", doc_id)
                return None
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
                chain_secrets,
                doc_lookup,
                pattern=pattern,
                task_id=task_id,
                company=chain_company,
                rng=task_rng,
            )
        else:
            next_pool = chain_local_ids if pattern[1] == "L" else chain_web_doc_ids
            state = select_web_seed(
                entity_index,
                chain_web_doc_ids,
                next_pool,
                llm,
                pattern=pattern,
                task_id=task_id,
                company=chain_company,
                rng=task_rng,
            )
    except ValueError as e:
        logger.warning("Seed failed for %s #%d: %s", pattern, idx, e)
        return None
    except Exception as e:
        logger.exception("Unexpected error building seed %s #%d: %s", pattern, idx, e)
        return None

    chain = build_one_chain(
        state=state,
        llm=llm,
        entity_index=entity_index,
        local_doc_ids=chain_local_ids,
        web_doc_ids=chain_web_doc_ids,
        record_trace=True,
        local_searcher=local_searcher,
        web_searcher=task_web_searcher,
        retrieval_k=config.RETRIEVAL_K,
        secret_by_entity=secret_by_entity,
        local_doc_family_map=local_doc_family_map,
    )

    d = _chain_to_dict(chain)
    d["metadata"]["seed_idx"] = idx
    d["metadata"]["seed_pattern"] = pattern
    d["metadata"]["seed_task_id"] = task_id
    d["metadata"]["generator_variant"] = "trajectory_generation"
    d["metadata"]["pick_prompt_variant"] = "current_best_of_10"

    status = "VALID" if chain.verification and chain.verification.is_valid else "INVALID"
    return {
        "record": d,
        "status": status,
        "chain_id": chain.chain_id,
        "task_id": task_id,
        "company": chain_company,
        "pattern": pattern,
        "seed_idx": idx,
    }


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
    schedule: str,
    num_shards: int,
    shard_index: int,
) -> list[dict]:
    """Build the list of (pattern, task_id, seed, ...) to process. Deterministic given seed."""
    if seeds_from and seeds_from.exists():
        records = _load_seeds_from_file(seeds_from, max_chains)
        records = _apply_seed_schedule(records, schedule)
        return _select_seed_shard(records, num_shards, shard_index)

    manifest = load_state.get("manifest", load_state)
    task_ids = manifest.get("task_ids", [])
    if not task_ids:
        raise ValueError("No tasks in load state")

    rng = random.Random(seed)
    records = []
    idx = 0
    if per_task:
        per_task_patterns: dict[str, list[str]] = {
            task_id: [rng.choice(patterns) for _ in range(per_task)]
            for task_id in task_ids
        }
        if schedule == "grouped":
            for task_id in task_ids:
                for attempt_idx in range(per_task):
                    records.append({
                        "idx": idx,
                        "pattern": per_task_patterns[task_id][attempt_idx],
                        "task_id": task_id,
                        "seed": rng.randint(0, config.MAX_RANDOM_SEED - 1),
                    })
                    idx += 1
        else:
            for attempt_idx in range(per_task):
                for task_id in task_ids:
                    records.append({
                        "idx": idx,
                        "pattern": per_task_patterns[task_id][attempt_idx],
                        "task_id": task_id,
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
    records = records[:max_chains] if max_chains else records
    return _select_seed_shard(records, num_shards, shard_index)


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


def _apply_seed_schedule(records: list[dict], schedule: str) -> list[dict]:
    """Reorder existing seed records while preserving each task's internal order."""
    if schedule == "grouped":
        return records

    by_task: dict[str, list[dict]] = defaultdict(list)
    task_order: list[str] = []
    for rec in records:
        task_id = rec.get("task_id", "")
        if task_id not in by_task:
            task_order.append(task_id)
        by_task[task_id].append(rec)

    interleaved: list[dict] = []
    max_len = max((len(v) for v in by_task.values()), default=0)
    for round_idx in range(max_len):
        for task_id in task_order:
            task_recs = by_task[task_id]
            if round_idx < len(task_recs):
                interleaved.append(task_recs[round_idx])
    return interleaved


def _select_seed_shard(records: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    """Pick the seeds for one shard using stable idx-based partitioning."""
    if num_shards <= 1:
        return records
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard_index={shard_index} for num_shards={num_shards}")
    return [rec for rec in records if int(rec.get("idx", -1)) % num_shards == shard_index]


def _is_direct_curated_doc_record(row: dict[str, Any]) -> bool:
    if row.get("extractor") == "insights_fallback":
        return False
    text = (row.get("text") or "").strip()
    err = row.get("error")
    status = row.get("http_status")
    return bool(text) and not err and isinstance(status, int) and 200 <= status < 300


def _load_curated_doc_metadata(docs_path: Path) -> tuple[set[str], dict[str, set[str]]]:
    allowed_doc_ids: set[str] = set()
    by_task: dict[str, set[str]] = defaultdict(set)
    if not docs_path.exists():
        return allowed_doc_ids, by_task

    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                continue
            if not _is_direct_curated_doc_record(row):
                continue
            allowed_doc_ids.add(doc_id)
            for task_id in row.get("task_ids") or []:
                if isinstance(task_id, str) and task_id:
                    by_task[task_id].add(doc_id)
    return allowed_doc_ids, by_task


_LOCAL_FAMILY_NAME_JACCARD = 0.6
_LOCAL_FAMILY_CONTENT_SIM = 0.75
_LOCAL_FAMILY_TEXT_CHARS = 5_000


def _local_name_stem(doc_id: str) -> str:
    name = doc_id.rsplit('/', 1)[-1]
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    return name.lower()


def _local_name_tokens(doc_id: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _local_name_stem(doc_id)))


def _name_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _normalized_text_prefix(text: str, limit: int = _LOCAL_FAMILY_TEXT_CHARS) -> str:
    return " ".join(text[:limit].lower().split())


def _build_local_doc_family_map(doc_lookup: dict[str, Any]) -> dict[str, str]:
    """Collapse obvious export variants and conservative near-duplicates within a task."""
    docs_by_task: dict[str, list[tuple[str, str, set[str], str]]] = defaultdict(list)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for doc_id, doc in doc_lookup.items():
        if not isinstance(doc_id, str) or not doc_id.startswith('local/'):
            continue
        parent[doc_id] = doc_id
        task_id = doc.meta.get('task_id', '') if doc else ''
        if not task_id:
            parts = doc_id.split('/')
            task_id = parts[1] if len(parts) > 1 else ''
        docs_by_task[task_id].append((
            doc_id,
            _local_name_stem(doc_id),
            _local_name_tokens(doc_id),
            _normalized_text_prefix(doc.text if doc else ''),
        ))

    for task_docs in docs_by_task.values():
        by_stem: dict[str, list[str]] = defaultdict(list)
        for doc_id, stem, _tokens, _text in task_docs:
            by_stem[stem].append(doc_id)
        for members in by_stem.values():
            if len(members) < 2:
                continue
            first = members[0]
            for other in members[1:]:
                union(first, other)

        for idx, (doc_a, _stem_a, toks_a, text_a) in enumerate(task_docs):
            for doc_b, _stem_b, toks_b, text_b in task_docs[idx + 1:]:
                if find(doc_a) == find(doc_b):
                    continue
                if _name_jaccard(toks_a, toks_b) < _LOCAL_FAMILY_NAME_JACCARD:
                    continue
                if SequenceMatcher(None, text_a, text_b).ratio() >= _LOCAL_FAMILY_CONTENT_SIM:
                    union(doc_a, doc_b)

    groups: dict[str, list[str]] = defaultdict(list)
    for doc_id in parent:
        groups[find(doc_id)].append(doc_id)

    family_map: dict[str, str] = {}
    for members in groups.values():
        canonical = min(members)
        for doc_id in members:
            family_map[doc_id] = canonical

    merged_groups = sum(1 for members in groups.values() if len(members) > 1)
    logger.info(
        'Local doc family map: %d docs -> %d families (%d merged groups)',
        len(parent),
        len(groups),
        merged_groups,
    )
    return family_map


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
    seed_schedule: str = "round_robin",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Path:
    """Build chains. Returns path to chains.jsonl."""
    manifest = load_state.get("manifest", load_state)
    seeds = _build_chain_plan(
        load_state,
        patterns,
        per_task,
        max_chains,
        seed,
        seeds_from,
        seed_schedule,
        num_shards,
        shard_index,
    )
    chunks_local = Path(manifest["chunks_local"])
    # Load local docs and secrets, group by task
    chunks_web_paths = [Path(p) for p in manifest.get("chunks_web", [])]
    curated_web_chunk_paths = [Path(p) for p in manifest.get("curated_web_chunks", [])]
    secret_inventory = Path(manifest["secret_inventory"])
    entity_index_dir = Path(manifest["entity_index_dir"])
    spacy_model = manifest["spacy_model"]
    curated_web_docs = Path(manifest.get("curated_web_docs", config.CURATED_WEB_DOCS))

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

    local_doc_family_map = _build_local_doc_family_map(doc_lookup)

    allowed_curated_doc_ids, curated_doc_ids_by_task = _load_curated_doc_metadata(curated_web_docs)

    browsecomp_web_docs = _load_web_docs(chunks_web_paths)
    curated_web_docs_by_id = _load_web_docs(
        curated_web_chunk_paths,
        allowed_doc_ids=allowed_curated_doc_ids,
    )
    browsecomp_web_doc_ids = set(browsecomp_web_docs)
    curated_web_doc_ids = set(curated_web_docs_by_id)
    web_docs = {**browsecomp_web_docs, **curated_web_docs_by_id}
    all_docs = {**local_docs, **web_docs}

    logger.info(
        "Web corpus loaded: %d BrowseComp docs, %d direct curated docs",
        len(browsecomp_web_doc_ids),
        len(curated_web_doc_ids),
    )

    # Retrieval searchers for bridge finding.
    local_searcher = None
    browsecomp_searcher = None
    curated_searcher = None
    if chunks_local.exists():
        local_searcher = HybridSearcher(chunks_path=chunks_local)
        logger.info("Local BM25 searcher: %d chunks", local_searcher.size)

    existing_browsecomp = [p for p in chunks_web_paths if p.exists()]
    if existing_browsecomp:
        if config.BROWSECOMP_SEARCH_URL:
            try:
                browsecomp_searcher = SearchClient(
                    config.BROWSECOMP_SEARCH_URL,
                    pool="web",
                    timeout=config.BROWSECOMP_SEARCH_TIMEOUT,
                )
                logger.info(
                    "BrowseComp shared hybrid searcher: %d docs via %s",
                    browsecomp_searcher.size,
                    config.BROWSECOMP_SEARCH_URL,
                )
            except Exception as e:
                logger.warning(
                    "BrowseComp shared retrieval unavailable (%s); falling back to local hybrid",
                    e,
                )
        if browsecomp_searcher is None:
            try:
                browsecomp_embedder = OpenAICompatibleBackend(
                    model=config.BROWSECOMP_EMBEDDING_MODEL,
                    base_url=config.BROWSECOMP_EMBEDDING_BASE_URL,
                )
                browsecomp_searcher = BrowseCompHybridSearcher(
                    index_glob=config.BROWSECOMP_EMBEDDING_INDEX_GLOB,
                    dataset_name=config.BROWSECOMP_DATASET_NAME,
                    embedder=browsecomp_embedder,
                    bm25_recall_k=config.BROWSECOMP_BM25_RECALL_K,
                    dense_recall_k=config.BROWSECOMP_DENSE_RECALL_K,
                )
                logger.info(
                    "BrowseComp local hybrid searcher: %d docs via %s",
                    browsecomp_searcher.size,
                    config.BROWSECOMP_EMBEDDING_BASE_URL,
                )
            except Exception as e:
                logger.warning(
                    "BrowseComp local hybrid retrieval unavailable (%s); falling back to BM25",
                    e,
                )
        if browsecomp_searcher is None:
            try:
                browsecomp_searcher = BrowseCompBM25Searcher(
                    index_glob=config.BROWSECOMP_EMBEDDING_INDEX_GLOB,
                    dataset_name=config.BROWSECOMP_DATASET_NAME,
                )
                logger.info(
                    "BrowseComp BM25 fallback searcher: %d docs via internal corpus index",
                    browsecomp_searcher.size,
                )
            except Exception as e:
                logger.warning(
                    "BrowseComp internal BM25 unavailable (%s); falling back to chunk BM25",
                    e,
                )
        if browsecomp_searcher is None:
            browsecomp_searcher = HybridSearcher.from_paths(existing_browsecomp)
            logger.info("BrowseComp chunk-BM25 fallback searcher: %d chunks", browsecomp_searcher.size)

    existing_curated = [p for p in curated_web_chunk_paths if p.exists()]
    if existing_curated and allowed_curated_doc_ids:
        try:
            curated_searcher = HybridSearcher.from_paths(
                existing_curated,
                doc_ids=allowed_curated_doc_ids,
            )
            logger.info(
                "Curated web searcher: %d chunks across %d docs",
                curated_searcher.size,
                len(curated_web_doc_ids),
            )
        except ValueError:
            curated_searcher = None
            logger.info("Curated web searcher: no direct-fetch chunks available")

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

    logger.info(
        "Build plan: %d seeds (schedule=%s, shard=%d/%d, workers=%d)",
        len(seeds),
        seed_schedule,
        shard_index,
        num_shards,
        workers,
    )

    task_web_searchers: dict[str, Any] = {}
    task_companies: dict[str, str] = {}
    task_payloads: list[dict[str, Any]] = []

    for seed_rec in seeds:
        idx = seed_rec.get("idx", -1)
        if idx in done_indices:
            continue

        pattern = seed_rec["pattern"]
        task_id = seed_rec["task_id"]
        task_seed = seed_rec.get("seed", 0)
        forced_hop = seed_rec.get("forced_hop")
        chain_local_ids = task_local_doc_ids.get(task_id)
        if not chain_local_ids:
            logger.warning("No docs for task %s, skipping seed %d", task_id, idx)
            continue

        task_web_searcher = task_web_searchers.get(task_id)
        if task_web_searcher is None:
            task_web_searcher = _task_web_searcher_for_task(
                task_id=task_id,
                curated_doc_ids_by_task=curated_doc_ids_by_task,
                curated_web_doc_ids=curated_web_doc_ids,
                browsecomp_searcher=browsecomp_searcher,
                curated_searcher=curated_searcher,
            )
            task_web_searchers[task_id] = task_web_searcher

        if task_id not in task_companies:
            doc0 = next(iter(chain_local_ids))
            task_companies[task_id] = doc_lookup[doc0].meta.get("company_name", "?") if doc0 in doc_lookup else "?"

        chain_curated_ids = curated_doc_ids_by_task.get(task_id, set()) & curated_web_doc_ids
        chain_web_doc_ids = browsecomp_web_doc_ids | chain_curated_ids
        task_payloads.append({
            "seed_idx": idx,
            "pattern": pattern,
            "task_id": task_id,
            "task_seed": task_seed,
            "forced_hop": forced_hop,
            "chain_local_ids": chain_local_ids,
            "chain_web_doc_ids": chain_web_doc_ids,
            "chain_secrets": task_secrets.get(task_id, []),
            "chain_company": task_companies[task_id],
            "task_web_searcher": task_web_searcher,
        })

    logger.info("Build queue: %d pending seeds after resume filtering", len(task_payloads))

    if workers <= 1:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for payload in task_payloads:
                result = _build_seed_result(
                    idx=payload["seed_idx"],
                    pattern=payload["pattern"],
                    task_id=payload["task_id"],
                    task_seed=payload["task_seed"],
                    forced_hop=payload["forced_hop"],
                    chain_local_ids=payload["chain_local_ids"],
                    chain_web_doc_ids=payload["chain_web_doc_ids"],
                    chain_secrets=payload["chain_secrets"],
                    chain_company=payload["chain_company"],
                    doc_lookup=doc_lookup,
                    llm=llm,
                    entity_index=entity_index,
                    local_searcher=local_searcher,
                    task_web_searcher=payload["task_web_searcher"],
                    secret_by_entity=secret_by_entity,
                    local_doc_family_map=local_doc_family_map,
                )
                if result is None:
                    continue
                out_f.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                out_f.flush()
                logger.info(
                    "%s %s #%d %s [%s %s]",
                    result["status"],
                    result["pattern"],
                    result["seed_idx"],
                    result["chain_id"],
                    result["task_id"],
                    result["company"],
                )
        return output_path

    logger.info("Build execution: using %d in-process worker threads", workers)
    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="build") as pool:
            future_to_payload = {
                pool.submit(
                    _build_seed_result,
                    idx=payload["seed_idx"],
                    pattern=payload["pattern"],
                    task_id=payload["task_id"],
                    task_seed=payload["task_seed"],
                    forced_hop=payload["forced_hop"],
                    chain_local_ids=payload["chain_local_ids"],
                    chain_web_doc_ids=payload["chain_web_doc_ids"],
                    chain_secrets=payload["chain_secrets"],
                    chain_company=payload["chain_company"],
                    doc_lookup=doc_lookup,
                    llm=llm,
                    entity_index=entity_index,
                    local_searcher=local_searcher,
                    task_web_searcher=payload["task_web_searcher"],
                    secret_by_entity=secret_by_entity,
                    local_doc_family_map=local_doc_family_map,
                ): payload
                for payload in task_payloads
            }
            for future in as_completed(future_to_payload):
                payload = future_to_payload[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception(
                        "Unhandled worker error for %s #%d",
                        payload["pattern"],
                        payload["seed_idx"],
                    )
                    continue
                if result is None:
                    continue
                out_f.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                out_f.flush()
                logger.info(
                    "%s %s #%d %s [%s %s]",
                    result["status"],
                    result["pattern"],
                    result["seed_idx"],
                    result["chain_id"],
                    result["task_id"],
                    result["company"],
                )

    return output_path


def _load_web_docs(
    paths: list[Path],
    *,
    allowed_doc_ids: set[str] | None = None,
) -> dict[str, str]:
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
                if (
                    allowed_doc_ids is not None
                    and doc_id not in allowed_doc_ids
                ):
                    continue
                if doc_id and text:
                    docs.setdefault(doc_id, []).append((offset, text))

    result = {}
    for doc_id, chunks in docs.items():
        chunks.sort(key=lambda x: x[0])
        result[doc_id] = "\n\n".join(t for _, t in chunks)
    return result
