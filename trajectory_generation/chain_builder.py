"""Atomic entity-bridge chain builder.

Builds multi-hop QA chains by iteratively: find_bridge → generate intra (if needed)
→ generate pick → validate with checks → rank → commit. Uses trajectory_generation
src, questions, and checks.
"""

from __future__ import annotations

import logging
import time
import uuid

from trajectory_generation.checks import (
    check_answer_needs_backref,
    check_answerable_without_doc,
    check_question,
    check_search_leaks_bridge,
)
from trajectory_generation.src.format_questions import format_numbered_questions
from trajectory_generation.src.llm import LLMClient
from trajectory_generation.src.entity_index import EntityIndex, is_entity
from trajectory_generation.src.find_bridge import find_bridge, sort_entities_by_proximity
from trajectory_generation.src.verify_chain import verify_chain
from trajectory_generation.src.types import Chain, ChainState, HopRecord
from trajectory_generation.questions import (
    generate_question_constrained,
    generate_question_pick,
    rank_questions,
)

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 10  # Bridge candidates to try per transition
MAX_ENTITY_LIST = 20  # Entities shown in pick prompt
CONTEXT_WINDOW = 4000  # Chars around term for question-generation context


def _focused_context(doc_text: str, term: str, window: int = CONTEXT_WINDOW) -> str:
    """Extract window chars centered on first occurrence of term."""
    idx = doc_text.lower().find(term.lower())
    if idx == -1:
        return doc_text[:window]
    half = window // 2
    start = max(0, idx - half)
    end = min(len(doc_text), idx + len(term) + half)
    return doc_text[start:end]


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
) -> Chain:
    """Build one chain from seeded state. For each transition: find bridge,
    generate intra (if needs_intra), generate pick, run checks, rank, commit.
    """
    t0 = time.time()
    llm_calls = 0
    prev_answers: list[str] = [state.global_answer]
    used_answers: set[str] = {state.global_answer.lower()}

    pattern = state.pattern
    current_doc_id = state.hop_history[-1].doc_id
    current_answer = state.global_answer

    for jump_idx in range(len(pattern) - 1):
        src_type = pattern[jump_idx]
        dst_type = pattern[jump_idx + 1]
        next_pool = local_doc_ids if dst_type == "L" else web_doc_ids

        logger.info(
            "=== Transition %d/%d (%s→%s) answer=%r ===",
            jump_idx + 1, len(pattern) - 1, src_type, dst_type, current_answer,
        )

        dst_searcher = local_searcher if dst_type == "L" else web_searcher
        candidates = find_bridge(
            entity_index, current_doc_id, current_answer,
            next_pool, state.used_doc_ids, used_answers,
            searcher=dst_searcher, retrieval_k=retrieval_k,
        )

        if record_trace:
            state.trace.append({
                "step": "find_bridge",
                "transition": jump_idx + 1,
                "current_answer": current_answer,
                "current_doc": current_doc_id[:60],
                "n_candidates": len(candidates),
                "top_candidates": [
                    {"doc": c.target_doc_id[:60], "entity": c.bridge_entity,
                     "needs_intra": c.needs_intra, "score": c.score}
                    for c in candidates[:5]
                ],
            })

        if not candidates:
            logger.error("No bridge candidates for transition %d", jump_idx + 1)
            break

        valid_options: list[dict] = []
        intra_cache: dict[str, dict | None] = {}

        for cand in candidates[:MAX_CANDIDATES]:
            bridge_entity = cand.bridge_entity
            target_doc_id = cand.target_doc_id
            target_doc_text = entity_index.doc_text(target_doc_id)

            logger.info(
                "  Trying: %s via %r (intra=%s)",
                target_doc_id[:50], bridge_entity, cand.needs_intra,
            )

            option: dict = {"cand": cand, "target_doc_text": target_doc_text}

            if cand.needs_intra:
                prev_question = state.hop_history[-1].question
                if bridge_entity.lower() in prev_question.lower():
                    logger.info("  Skipping bridge %r (already in prev question)", bridge_entity)
                    continue

                be_key = bridge_entity.lower()
                if be_key in intra_cache:
                    cached = intra_cache[be_key]
                    if cached is None:
                        logger.info("  Skipping (intra already failed for %r)", bridge_entity)
                        continue
                    option.update(cached)
                    logger.info("  Intra cached: %r → %r", cached["intra_q"][:60], cached["intra_a"])
                else:
                    current_doc_text = entity_index.doc_text(current_doc_id)
                    intra_context = _focused_context(current_doc_text, current_answer)
                    try:
                        q_intra, a_intra, trace_intra = generate_question_constrained(
                            intra_context,
                            current_answer, bridge_entity, llm,
                        )
                        llm_calls += 1
                        if record_trace:
                            trace_intra["transition"] = jump_idx + 1
                            state.trace.append(trace_intra)
                    except ValueError as e:
                        logger.info("  Intra question failed: %s", e)
                        intra_cache[be_key] = None
                        continue

                    quote_intra = trace_intra.get("quote", "")
                    err = check_question(
                        q_intra, a_intra,
                        required_phrase=current_answer,
                        expected_answer=bridge_entity,
                        prev_answers=prev_answers,
                        quote=quote_intra,
                        doc_text=entity_index.doc_text(current_doc_id),
                    )
                    if record_trace:
                        state.trace.append({
                            "step": "check_intra", "transition": jump_idx + 1,
                            "passed": err is None, "error": err,
                            "question": q_intra, "answer": a_intra,
                            "quote": quote_intra,
                        })
                    if err:
                        logger.info("  Intra FAIL: %s | Q=%r A=%r", err, q_intra[:80], a_intra)
                        intra_cache[be_key] = None
                        continue

                    trivial = check_answerable_without_doc(q_intra, llm, correct_answer=a_intra)
                    llm_calls += 1
                    if record_trace:
                        state.trace.append({
                            "step": "check_trivial_intra", "transition": jump_idx + 1,
                            "question": q_intra, "trivial": trivial.trivial,
                            "model_answer": trivial.answer, "justification": trivial.justification,
                        })
                    if trivial.trivial:
                        logger.info("  Intra TRIVIAL: %r → %r (%s)", q_intra[:60], trivial.answer, trivial.justification)
                        intra_cache[be_key] = None
                        continue

                    backref = check_answer_needs_backref(q_intra, current_answer, llm)
                    llm_calls += 1
                    if record_trace:
                        state.trace.append({
                            "step": "check_backref_intra", "transition": jump_idx + 1,
                            "question": q_intra, "prev_answer": current_answer,
                            "independent": backref.independent,
                            "model_answer": backref.answer, "justification": backref.justification,
                        })
                    if backref.independent:
                        logger.info("  Intra BACKREF-INDEPENDENT: %r answerable without %r → %r (%s)",
                                    q_intra[:60], current_answer, backref.answer, backref.justification)
                        intra_cache[be_key] = None
                        continue

                    intra_result = {"intra_q": q_intra, "intra_a": a_intra, "intra_quote": quote_intra}
                    intra_cache[be_key] = intra_result
                    option.update(intra_result)
                    logger.info("  Intra OK: %r → %r", q_intra[:60], a_intra)

            if dst_type == "L" and secret_by_entity is not None:
                bridge_lower = bridge_entity.lower() if cand.needs_intra else current_answer.lower()
                for s_doc_id, secret in secret_by_entity.get(bridge_lower, []):
                    if secret.doc_id != target_doc_id:
                        continue
                    if secret.answer.lower() in used_answers:
                        continue
                    if len(secret.answer.split()) > 3:
                        continue
                    option_secret = dict(option)
                    option_secret["inter_q"] = secret.question
                    option_secret["inter_a"] = secret.answer
                    option_secret["inter_quote"] = secret.quote or ""
                    option_secret["secret_hop"] = True
                    valid_options.append(option_secret)
                    logger.info("  Secret-as-hop: %r → %r", secret.question[:60], secret.answer)
                    if record_trace:
                        state.trace.append({
                            "step": "secret_as_hop", "transition": jump_idx + 1,
                            "bridge_entity": bridge_entity,
                            "secret_question": secret.question,
                            "secret_answer": secret.answer,
                            "secret_doc_id": secret.doc_id,
                        })

            target_entities = entity_index.entities_in_doc(target_doc_id)
            filtered = [
                e for e in target_entities
                if e.lower() != bridge_entity.lower()
                and e.lower() not in used_answers
                and len(e) >= 3
                and is_entity(e)
            ]
            entity_list = sort_entities_by_proximity(target_doc_text, bridge_entity, filtered)[:MAX_ENTITY_LIST]

            if not entity_list:
                logger.info("  No viable entities in target doc, skipping")
                continue

            inter_context = _focused_context(target_doc_text, bridge_entity)
            try:
                q_inter, a_inter, trace_inter = generate_question_pick(
                    inter_context, bridge_entity, entity_list, llm,
                )
                llm_calls += 1
                if record_trace:
                    trace_inter["transition"] = jump_idx + 1
                    state.trace.append(trace_inter)
            except ValueError as e:
                logger.info("  Inter question failed: %s", e)
                continue

            quote_inter = trace_inter.get("quote", "")
            err = check_question(
                q_inter, a_inter,
                required_phrase=bridge_entity,
                allowed_answers={e.strip().lower() for e in entity_list},
                prev_answers=prev_answers,
                quote=quote_inter,
                doc_text=target_doc_text,
            )
            if record_trace:
                state.trace.append({
                    "step": "check_inter", "transition": jump_idx + 1,
                    "passed": err is None, "error": err,
                    "question": q_inter, "answer": a_inter,
                    "quote": quote_inter,
                })
            if err:
                logger.info("  Inter FAIL: %s | Q=%r A=%r", err, q_inter[:80], a_inter)
                continue

            trivial = check_answerable_without_doc(q_inter, llm, correct_answer=a_inter)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_trivial_inter", "transition": jump_idx + 1,
                    "question": q_inter, "trivial": trivial.trivial,
                    "model_answer": trivial.answer, "justification": trivial.justification,
                })
            if trivial.trivial:
                logger.info("  Inter TRIVIAL: %r → %r (%s)", q_inter[:60], trivial.answer, trivial.justification)
                continue

            backref_inter = check_answer_needs_backref(q_inter, bridge_entity, llm)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_backref_inter", "transition": jump_idx + 1,
                    "question": q_inter, "prev_answer": bridge_entity,
                    "independent": backref_inter.independent,
                    "model_answer": backref_inter.answer, "justification": backref_inter.justification,
                })
            if backref_inter.independent:
                logger.info("  Inter BACKREF-INDEPENDENT: %r answerable without %r → %r (%s)",
                            q_inter[:60], bridge_entity, backref_inter.answer, backref_inter.justification)
                continue

            search_leak = check_search_leaks_bridge(q_inter, bridge_entity, llm)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_search_leak", "transition": jump_idx + 1,
                    "question": q_inter, "bridge_value": bridge_entity,
                    "would_leak": search_leak.would_leak,
                    "queries": search_leak.queries, "justification": search_leak.justification,
                })
            if not search_leak.would_leak:
                logger.info("  Inter NO-LEAK: bridge %r not in search queries %s",
                            bridge_entity, search_leak.queries)
                continue

            option["inter_q"] = q_inter
            option["inter_a"] = a_inter
            option["inter_quote"] = quote_inter
            logger.info("  Inter OK (leaks bridge): %r → %r  queries=%s", q_inter[:60], a_inter, search_leak.queries)
            valid_options.append(option)

        if not valid_options:
            logger.error("Failed transition %d (%s→%s): no valid options", jump_idx + 1, src_type, dst_type)
            break

        if len(valid_options) > 1:
            judge_candidates = [
                {"question": opt["inter_q"], "answer": opt["inter_a"]}
                for opt in valid_options
            ]
            ranking = rank_questions(judge_candidates, current_answer, llm)
            llm_calls += 1
            best_idx = ranking[0]
            logger.info("  LLM judge picked option %d/%d", best_idx + 1, len(valid_options))
        else:
            best_idx = 0

        best = valid_options[best_idx]
        cand = best["cand"]

        if cand.needs_intra:
            hop_intra = HopRecord(
                hop_number=state.current_hop + 1,
                hop_type=src_type,
                question=best["intra_q"],
                answer=best["intra_a"],
                doc_id=current_doc_id,
                doc_text=entity_index.doc_text(current_doc_id),
                quote=best.get("intra_quote", ""),
            )
            state.hop_history.append(hop_intra)
            prev_answers.append(best["intra_a"])
            used_answers.add(best["intra_a"].lower())

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
        current_doc_id = cand.target_doc_id
        current_answer = best["inter_a"]
        state.global_answer = best["inter_a"]

        if dst_type == "L" and not state.task_id:
            parts = cand.target_doc_id.split("/")
            if len(parts) >= 2 and parts[0] == "local":
                state.task_id = parts[1]

        logger.info(
            "  Committed: %r → %r (doc=%s)",
            best["inter_q"][:60], best["inter_a"], cand.target_doc_id[:40],
        )

    hops_for_format = [
        {"question": h.question, "answer": h.answer, "hop_number": h.hop_number}
        for h in state.hop_history
    ]
    try:
        numbered_q = format_numbered_questions(hops_for_format)
    except ValueError:
        numbered_q = state.hop_history[0].question if state.hop_history else ""
    state.global_question = numbered_q

    verification = None
    if state.is_complete:
        logger.info("Chain complete (%d hops), verifying...", state.current_hop)
        verification, traces_s7 = verify_chain(state, llm)
        if record_trace:
            state.trace.extend(traces_s7)
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
