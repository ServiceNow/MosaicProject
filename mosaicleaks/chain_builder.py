"""Chain builder for the MosaicLeaks dataset generation pipeline.

The builder uses explicit retrieval and validation steps:

1. retrieve bridge candidates
2. generate intra/inter questions with model-side best-of-many search
3. run a small set of hard checks
4. let an LLM rerank surviving options
5. verify only after the chain is complete
"""

import logging
import time
import uuid
from collections import defaultdict

from mosaicleaks import config
from mosaicleaks.src.entity_index import EntityIndex, is_entity
from mosaicleaks.src.find_bridge import find_bridge, sort_entities_by_proximity
from mosaicleaks.src.format_questions import format_numbered_questions
from mosaicleaks.src.llm import LLMClient
from mosaicleaks.src.types import Chain, ChainState, HopRecord
from mosaicleaks.src.verify_chain import verify_chain
from mosaicleaks.checks import (
    check_answer_needs_backref,
    check_answerable_without_doc,
    check_question,
    check_search_leaks_bridge,
)
from mosaicleaks.normalization import (
    canonicalize_surface,
)
from mosaicleaks.questions import (
    generate_question_constrained,
    generate_question_constrained_batch,
    generate_question_pick,
    generate_question_pick_batch,
    rank_questions,
)

logger = logging.getLogger(__name__)

MAX_CANDIDATES = max(10, config.BUILD_MAX_CANDIDATES)
MAX_DOCS_PER_CANONICAL_BRIDGE = 2
MAX_ALLOWED_ANSWERS = 12
CONTEXT_WINDOW = 4_000
EVIDENCE_WINDOW = 140
GENERATION_BATCH_SIZE = config.GENERATION_BATCH_SIZE


def _focused_context(doc_text: str, term: str, window: int = CONTEXT_WINDOW) -> str:
    idx = doc_text.lower().find(term.lower())
    if idx == -1:
        return doc_text[:window]
    half = window // 2
    start = max(0, idx - half)
    end = min(len(doc_text), idx + len(term) + half)
    return doc_text[start:end]


def _extract_evidence(doc_text: str, term: str, max_snippets: int = 2) -> list[str]:
    lower_text = doc_text.lower()
    lower_term = term.strip().lower()
    snippets: list[str] = []
    start = 0
    while len(snippets) < max_snippets:
        idx = lower_text.find(lower_term, start)
        if idx == -1:
            break
        sent_start = max(0, doc_text.rfind(".", max(0, idx - EVIDENCE_WINDOW), idx) + 1)
        if sent_start == 0 and idx > EVIDENCE_WINDOW:
            sent_start = idx - EVIDENCE_WINDOW
        sent_end = doc_text.find(".", idx + len(lower_term))
        if sent_end == -1 or sent_end > idx + len(lower_term) + EVIDENCE_WINDOW:
            sent_end = min(len(doc_text), idx + len(lower_term) + EVIDENCE_WINDOW)
        else:
            sent_end += 1
        snippet = " ".join(doc_text[sent_start:sent_end].split())
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        start = idx + len(lower_term)
    return snippets


def _source_excerpt(doc_text: str, answer: str) -> str:
    snippets = _extract_evidence(doc_text, answer, max_snippets=2)
    if snippets:
        return " | ".join(snippets)
    return _focused_context(doc_text, answer, window=1_000)


def _limit_bridge_candidates(candidates: list) -> list:
    counts: dict[str, int] = defaultdict(int)
    limited: list = []
    for cand in candidates:
        canonical = canonicalize_surface(cand.bridge_entity) or cand.bridge_entity.strip().lower()
        if counts[canonical] >= MAX_DOCS_PER_CANONICAL_BRIDGE:
            continue
        counts[canonical] += 1
        limited.append(cand)
    return limited


def _filter_allowed_answers(
    *,
    target_entities: list[str],
    target_excerpt: str,
    bridge_value: str,
    used_answers: set[str],
) -> list[str]:
    excerpt_lower = target_excerpt.lower()
    bridge_lower = bridge_value.lower().strip()
    viable: list[str] = []
    seen: set[str] = set()

    for ent in target_entities:
        ent_clean = ent.strip()
        ent_lower = ent_clean.lower()
        ent_canonical = canonicalize_surface(ent_clean)
        if not ent_clean or ent_lower == bridge_lower:
            continue
        if ent_lower in used_answers or ent_canonical in used_answers:
            continue
        if len(ent_clean) < 2 or not is_entity(ent_clean):
            continue
        if ent_lower not in excerpt_lower:
            continue
        if ent_canonical in seen:
            continue
        seen.add(ent_canonical)
        viable.append(ent_clean)

    ordered = sort_entities_by_proximity(target_excerpt, bridge_value, viable)
    return ordered[:MAX_ALLOWED_ANSWERS]


def _match_allowed_answer(answer: str, allowed_answers: list[str]) -> str | None:
    answer_lower = answer.strip().lower()
    for candidate in allowed_answers:
        if candidate.strip().lower() == answer_lower:
            return candidate
    return None


def build_one_chain(
    *,
    state: ChainState,
    llm: LLMClient,
    entity_index: EntityIndex,
    local_doc_ids: set[str],
    web_doc_ids: set[str],
    record_trace: bool = True,
    local_searcher=None,
    web_searcher=None,
    retrieval_k: int = 50,
    secret_by_entity: dict[str, list] | None = None,
    local_doc_family_map: dict[str, str] | None = None,
) -> Chain:
    """Build one chain from a seeded state."""
    t0 = time.time()
    llm_calls = 0
    prev_answers = [hop.answer for hop in state.hop_history]
    used_answers = {hop.answer.lower() for hop in state.hop_history}
    for hop in state.hop_history:
        used_answers.add(canonicalize_surface(hop.answer))

    pattern = state.pattern
    current_doc_id = state.hop_history[-1].doc_id
    current_answer = state.global_answer

    for jump_idx in range(len(pattern) - 1):
        src_type = pattern[jump_idx]
        dst_type = pattern[jump_idx + 1]
        next_pool = local_doc_ids if dst_type == "L" else web_doc_ids
        dst_searcher = local_searcher if dst_type == "L" else web_searcher
        current_doc_text = entity_index.doc_text(current_doc_id)

        logger.info(
            "=== Transition %d/%d (%s->%s) answer=%r ===",
            jump_idx + 1,
            len(pattern) - 1,
            src_type,
            dst_type,
            current_answer,
        )

        candidates = find_bridge(
            entity_index,
            current_doc_id,
            current_answer,
            next_pool,
            state.used_doc_ids,
            used_answers,
            searcher=dst_searcher,
            retrieval_k=retrieval_k,
            allow_fast_path=config.ENABLE_FAST_PATH,
            local_doc_family_map=local_doc_family_map,
        )
        candidates = _limit_bridge_candidates(candidates)

        if record_trace:
            state.trace.append({
                "step": "find_bridge",
                "transition": jump_idx + 1,
                "current_answer": current_answer,
                "current_doc": current_doc_id[:80],
                "n_candidates": len(candidates),
                "top_candidates": [
                    {
                        "doc": cand.target_doc_id[:80],
                        "entity": cand.bridge_entity,
                        "needs_intra": cand.needs_intra,
                        "score": cand.score,
                    }
                    for cand in candidates[:8]
                ],
            })

        if not candidates:
            logger.error("No bridge candidates for transition %d", jump_idx + 1)
            break

        valid_options: list[dict] = []
        intra_cache: dict[str, dict] = {}
        previous_hop = state.hop_history[-1]
        candidate_batch = candidates[:MAX_CANDIDATES]
        intra_context = _focused_context(current_doc_text, current_answer)
        current_source_excerpt = _source_excerpt(current_doc_text, current_answer)

        constrained_jobs: list[dict] = []
        constrained_meta: list[tuple] = []
        for cand in candidate_batch:
            bridge_value = cand.bridge_entity
            bridge_key = canonicalize_surface(bridge_value) or bridge_value.lower()
            logger.info(
                "  Trying: %s via %r (intra=%s)",
                cand.target_doc_id[:60],
                bridge_value,
                cand.needs_intra,
            )
            if not cand.needs_intra or bridge_key in intra_cache:
                continue
            constrained_jobs.append({
                "doc_text": intra_context,
                "prev_question": previous_hop.question,
                "prev_answer": current_answer,
                "target_answer": bridge_value,
                "source_excerpt": current_source_excerpt,
            })
            constrained_meta.append((bridge_key, bridge_value))

        if constrained_jobs:
            constrained_results = generate_question_constrained_batch(
                jobs=constrained_jobs,
                llm=llm,
                batch_size=GENERATION_BATCH_SIZE,
            )
            llm_calls += len(constrained_jobs)
            for (bridge_key, bridge_value), result in zip(constrained_meta, constrained_results):
                if isinstance(result, Exception):
                    logger.info("  Intra question failed for %r: %s", bridge_value, result)
                    continue
                q_intra, a_intra, trace_intra = result
                if record_trace:
                    trace_intra["transition"] = jump_idx + 1
                    state.trace.append(trace_intra)
                quote_intra = trace_intra.get("quote", "")
                err = check_question(
                    q_intra,
                    a_intra,
                    prev_answers=prev_answers,
                )
                if record_trace:
                    state.trace.append({
                        "step": "check_intra",
                        "transition": jump_idx + 1,
                        "passed": err is None,
                        "error": err,
                        "question": q_intra,
                        "answer": a_intra,
                        "quote": quote_intra,
                    })
                if err:
                    logger.info("  Intra FAIL: %s | Q=%r A=%r", err, q_intra[:100], a_intra)
                    continue
                intra_cache[bridge_key] = {
                    "intra_q": q_intra,
                    "intra_a": a_intra,
                    "intra_quote": quote_intra,
                }
                logger.info("  Intra OK: %r -> %r", q_intra[:80], a_intra)

        inter_jobs: list[dict] = []
        inter_job_meta: list[dict] = []

        for cand in candidate_batch:
            bridge_value = cand.bridge_entity
            bridge_key = canonicalize_surface(bridge_value) or bridge_value.lower()
            target_doc_id = cand.target_doc_id
            target_doc_text = entity_index.doc_text(target_doc_id)

            option = {
                "cand": cand,
                "target_doc_text": target_doc_text,
                "bridge_value": bridge_value,
            }

            if cand.needs_intra:
                cached = intra_cache.get(bridge_key)
                if cached is None:
                    continue
                option.update(cached)
                logger.info("  Intra cached: %r -> %r", cached["intra_q"][:80], cached["intra_a"])

            inter_prev_question = option.get("intra_q", previous_hop.question)
            inter_prev_answer = option.get("intra_a", current_answer)
            inter_source_excerpt = option.get("intra_quote") or _source_excerpt(current_doc_text, inter_prev_answer)
            option["inter_prev_question"] = inter_prev_question
            option["inter_prev_answer"] = inter_prev_answer

            if dst_type == "L" and secret_by_entity is not None:
                lookup_key = bridge_value.lower() if cand.needs_intra else current_answer.lower()
                for secret_doc_id, secret in secret_by_entity.get(lookup_key, []):
                    if secret.doc_id != target_doc_id:
                        continue
                    if canonicalize_surface(secret.answer) in {canonicalize_surface(a) for a in prev_answers}:
                        continue
                    if len(secret.answer.split()) > 5:
                        continue
                    option_secret = dict(option)
                    option_secret["inter_q"] = secret.question
                    option_secret["inter_a"] = secret.answer
                    option_secret["inter_quote"] = secret.quote or ""
                    option_secret["secret_hop"] = True
                    valid_options.append(option_secret)
                    logger.info("  Secret-as-hop: %r -> %r", secret.question[:80], secret.answer)

            inter_context = _focused_context(target_doc_text, inter_prev_answer)
            target_entities = entity_index.entities_in_doc(target_doc_id)
            allowed_answers = _filter_allowed_answers(
                target_entities=target_entities,
                target_excerpt=inter_context,
                bridge_value=inter_prev_answer,
                used_answers=used_answers,
            )
            if not allowed_answers:
                logger.info("  No allowed target answers near bridge %r", inter_prev_answer)
                continue

            inter_jobs.append({
                "target_excerpt": inter_context,
                "prev_question": inter_prev_question,
                "prev_answer": inter_prev_answer,
                "bridge_value": inter_prev_answer,
                "source_excerpt": inter_source_excerpt,
                "allowed_answers": allowed_answers,
            })
            inter_job_meta.append({
                "option": option,
                "allowed_answers": allowed_answers,
                "inter_context": inter_context,
                "inter_prev_answer": inter_prev_answer,
            })

        if inter_jobs:
            inter_results = generate_question_pick_batch(
                jobs=inter_jobs,
                llm=llm,
                batch_size=GENERATION_BATCH_SIZE,
            )
            llm_calls += len(inter_jobs)
            for meta, result in zip(inter_job_meta, inter_results):
                option = meta["option"]
                allowed_answers = meta["allowed_answers"]
                inter_context = meta["inter_context"]
                inter_prev_answer = meta["inter_prev_answer"]

                if isinstance(result, Exception):
                    logger.info("  Inter question failed: %s", result)
                    continue
                q_inter, a_inter, trace_inter = result
                if record_trace:
                    trace_inter["transition"] = jump_idx + 1
                    state.trace.append(trace_inter)

                matched_answer = _match_allowed_answer(a_inter, allowed_answers)
                if matched_answer is None:
                    logger.info("  Inter FAIL: answer %r not copied from allowed answers", a_inter)
                    continue
                a_inter = matched_answer

                quote_inter = trace_inter.get("quote", "")
                err = check_question(
                    q_inter,
                    a_inter,
                    prev_answers=prev_answers,
                )
                if record_trace:
                    state.trace.append({
                        "step": "check_inter",
                        "transition": jump_idx + 1,
                        "passed": err is None,
                        "error": err,
                        "question": q_inter,
                        "answer": a_inter,
                        "quote": quote_inter,
                        "bridge_value": inter_prev_answer,
                        "allowed_answers": allowed_answers,
                        "target_excerpt": inter_context,
                    })
                if err:
                    logger.info("  Inter FAIL: %s | Q=%r A=%r", err, q_inter[:100], a_inter)
                    continue

                option["inter_q"] = q_inter
                option["inter_a"] = a_inter
                option["inter_quote"] = quote_inter
                option["inter_bridge_value"] = inter_prev_answer
                option["allowed_answers"] = allowed_answers
                option["target_excerpt"] = inter_context
                valid_options.append(option)
                logger.info("  Inter OK: %r -> %r", q_inter[:80], a_inter)

        if not valid_options:
            logger.error("Failed transition %d (%s->%s): no valid options", jump_idx + 1, src_type, dst_type)
            break

        if len(valid_options) > 1:
            ranking = rank_questions(
                candidates=[
                    {
                        "bridge_value": opt.get("inter_bridge_value", opt.get("bridge_value", current_answer)),
                        "question": opt["inter_q"],
                        "answer": opt["inter_a"],
                        "quote": opt.get("inter_quote", ""),
                    }
                    for opt in valid_options
                ],
                prev_question=previous_hop.question,
                prev_answer=current_answer,
                llm=llm,
            )
            llm_calls += 1
            logger.info("  LLM ranker order: %s", [idx + 1 for idx in ranking])
        else:
            ranking = [0]

        best = None
        for idx in ranking[:3]:
            candidate = valid_options[idx]
            q_inter = candidate["inter_q"]
            a_inter = candidate["inter_a"]
            inter_prev_answer = candidate["inter_prev_answer"]

            trivial = check_answerable_without_doc(q_inter, llm, correct_answer=a_inter)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_trivial_inter",
                    "transition": jump_idx + 1,
                    "question": q_inter,
                    "trivial": trivial.trivial,
                    "model_answer": trivial.answer,
                    "justification": trivial.justification,
                    "rank_index": idx,
                })
            if trivial.trivial:
                logger.info("  Ranked option %d TRIVIAL: %r -> %r", idx + 1, q_inter[:80], trivial.answer)
                continue

            backref = check_answer_needs_backref(q_inter, inter_prev_answer, llm)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_backref_inter",
                    "transition": jump_idx + 1,
                    "question": q_inter,
                    "prev_answer": inter_prev_answer,
                    "independent": backref.independent,
                    "model_answer": backref.answer,
                    "justification": backref.justification,
                    "rank_index": idx,
                })
            if backref.independent:
                logger.info("  Ranked option %d BACKREF-INDEPENDENT: %r -> %r", idx + 1, q_inter[:80], backref.answer)
                continue

            if dst_type == "W":
                search_leak = check_search_leaks_bridge(q_inter, inter_prev_answer, llm)
                llm_calls += 1
                if record_trace:
                    state.trace.append({
                        "step": "check_search_leak",
                        "transition": jump_idx + 1,
                        "question": q_inter,
                        "bridge_value": inter_prev_answer,
                        "would_leak": search_leak.would_leak,
                        "queries": search_leak.queries,
                        "justification": search_leak.justification,
                        "rank_index": idx,
                    })
                if not search_leak.would_leak:
                    logger.info("  Ranked option %d NO-LEAK: %s", idx + 1, search_leak.justification)
                    continue

            best = candidate
            logger.info("  LLM ranker accepted option %d/%d", idx + 1, len(valid_options))
            break

        if best is None:
            logger.error("Failed transition %d (%s->%s): ranked options did not pass final checks", jump_idx + 1, src_type, dst_type)
            break

        cand = best["cand"]

        if cand.needs_intra:
            hop_intra = HopRecord(
                hop_number=state.current_hop + 1,
                hop_type=src_type,
                question=best["intra_q"],
                answer=best["intra_a"],
                doc_id=current_doc_id,
                doc_text=current_doc_text,
                quote=best.get("intra_quote", ""),
            )
            state.hop_history.append(hop_intra)
            prev_answers.append(best["intra_a"])
            used_answers.add(best["intra_a"].lower())
            used_answers.add(canonicalize_surface(best["intra_a"]))

        hop_inter = HopRecord(
            hop_number=state.current_hop + 1,
            hop_type=dst_type,
            question=best["inter_q"],
            answer=best["inter_a"],
            doc_id=cand.target_doc_id,
            doc_text=best["target_doc_text"],
            quote=best.get("inter_quote", ""),
        )
        state.hop_history.append(hop_inter)
        state.used_doc_ids.add(cand.target_doc_id)
        state.jumps_completed += 1
        prev_answers.append(best["inter_a"])
        used_answers.add(best["inter_a"].lower())
        used_answers.add(canonicalize_surface(best["inter_a"]))
        current_doc_id = cand.target_doc_id
        current_answer = best["inter_a"]
        state.global_answer = best["inter_a"]

        if dst_type == "L" and not state.task_id:
            parts = cand.target_doc_id.split("/")
            if len(parts) >= 2 and parts[0] == "local":
                state.task_id = parts[1]

        if record_trace:
            state.trace.append({
                "step": "commit_transition",
                "transition": jump_idx + 1,
                "question": best["inter_q"],
                "answer": best["inter_a"],
                "target_doc_id": cand.target_doc_id,
                "bridge_value": best.get("inter_bridge_value", best.get("bridge_value", current_answer)),
            })

        logger.info(
            "  Committed: %r -> %r (doc=%s)",
            best["inter_q"][:80],
            best["inter_a"],
            cand.target_doc_id[:60],
        )

    hops_for_format = [
        {"question": hop.question, "answer": hop.answer, "hop_number": hop.hop_number}
        for hop in state.hop_history
    ]
    numbered_q = format_numbered_questions(hops_for_format)
    state.global_question = numbered_q

    verification = None
    if state.is_complete:
        logger.info("Chain complete (%d hops), verifying...", state.current_hop)
        verification, verify_traces = verify_chain(state, llm)
        if record_trace:
            state.trace.extend(verify_traces)
        llm_calls += 4
        logger.info("Verification: valid=%s", verification.is_valid)

    elapsed = time.time() - t0
    return Chain(
        chain_id=str(uuid.uuid4())[:8],
        pattern=state.pattern,
        hop_history=state.hop_history,
        global_question=numbered_q,
        global_answer=state.global_answer,
        verification=verification,
        metadata={
            "task_id": state.task_id,
            "company": state.company,
            "llm_calls": llm_calls,
            "elapsed_seconds": round(elapsed, 1),
            "complete": state.is_complete,
            "n_hops": state.current_hop,
            "n_jumps": state.jumps_completed,
            "trace": state.trace if record_trace else [],
        },
    )
