"""Bridge finding for document transitions.

Given current doc, current answer, and a pool of candidate next docs,
finds (target_doc, bridge_entity, needs_intra) tuples. With BM25 searcher,
narrows to topically relevant docs first. Skips bridge entities where one
is a substring of the other (e.g. "2025" vs "Q2 2025").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from trajectory_generation.src.entity_index import EntityIndex, is_entity

logger = logging.getLogger(__name__)

# spaCy NER artifacts that are useless as bridges
_BRIDGE_BLOCKLIST = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth",
    "us", "no", "may", "la", "ok", "an", "al",
})


def _is_good_bridge(ent: str) -> bool:
    """Filter out blocklisted and non-entity bridge candidates."""
    if ent.strip().lower() in _BRIDGE_BLOCKLIST:
        return False
    if not is_entity(ent):
        return False
    return True


@dataclass
class BridgeCandidate:
    """A candidate (target_doc, bridge_entity) for the next hop."""

    target_doc_id: str
    bridge_entity: str
    needs_intra: bool  # True if bridge_entity != current_answer (requires intra hop)
    score: float = 0.0


def _min_distance(text_lower: str, a: str, b: str) -> float:
    a_lower, b_lower = a.lower(), b.lower()
    a_positions: list[int] = []
    b_positions: list[int] = []
    start = 0
    while True:
        idx = text_lower.find(a_lower, start)
        if idx == -1:
            break
        a_positions.append(idx)
        start = idx + 1
    start = 0
    while True:
        idx = text_lower.find(b_lower, start)
        if idx == -1:
            break
        b_positions.append(idx)
        start = idx + 1

    if not a_positions or not b_positions:
        return float('inf')

    best = float('inf')
    for ap in a_positions:
        a_end = ap + len(a_lower)
        for bp in b_positions:
            b_end = bp + len(b_lower)
            if a_end <= bp:
                d = bp - a_end
            elif b_end <= ap:
                d = ap - b_end
            else:
                d = 0
            best = min(best, d)
            if best == 0:
                return 0
    return best


def sort_entities_by_proximity(doc_text: str, phrase: str, entities: list[str]) -> list[str]:
    """Sort entities by min char distance to phrase in doc. Closest first. Unknown → end."""
    text_lower = doc_text.lower()
    phrase_lower = phrase.lower().strip()
    scored = [(_min_distance(text_lower, phrase_lower, e.strip().lower()), e) for e in entities]
    scored.sort(key=lambda x: x[0])
    return [e for _, e in scored]


def _retrieval_query(doc_text: str, answer: str, window: int = 400) -> str:
    """Extract context around answer for BM25 query."""
    idx = doc_text.lower().find(answer.lower())
    if idx == -1:
        return answer
    half = window // 2
    start = max(0, idx - half)
    end = min(len(doc_text), idx + len(answer) + half)
    return doc_text[start:end]


def _find_in_pool(
    entity_index: EntityIndex,
    current_doc_id: str,
    current_answer: str,
    pool: set[str],
    used: set[str],
) -> tuple[list[BridgeCandidate], list[BridgeCandidate]]:
    answer_lower = current_answer.strip().lower()
    current_doc_text_lower = entity_index.doc_text(current_doc_id).lower()
    current_entities = entity_index.entities_in_doc(current_doc_id)
    seen_keys: set[tuple[str, str]] = set()

    fast_candidates: list[BridgeCandidate] = []
    entity_candidates: list[BridgeCandidate] = []

    for doc_id in entity_index.docs_containing_text(current_answer, pool=pool):
        key = (doc_id, answer_lower)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        fast_candidates.append(BridgeCandidate(
            target_doc_id=doc_id,
            bridge_entity=current_answer.strip(),
            needs_intra=False,
        ))

    for ent in current_entities:
        ent_lower = ent.lower()
        if ent_lower == answer_lower or len(ent_lower) < 3 or ent_lower in used or not _is_good_bridge(ent):
            continue
        # Skip when one is substring of the other (e.g. "2025" vs "Q2 2025")
        if ent_lower in answer_lower or answer_lower in ent_lower:
            continue
        for doc_id in entity_index.docs_containing_text(ent, pool=pool):
            key = (doc_id, ent_lower)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            proximity = _min_distance(current_doc_text_lower, current_answer, ent)
            entity_candidates.append(BridgeCandidate(
                target_doc_id=doc_id,
                bridge_entity=ent,
                needs_intra=True,
                score=-proximity,
            ))

    entity_candidates.sort(key=lambda c: c.score, reverse=True)
    return fast_candidates, entity_candidates


def find_bridge(
    entity_index: EntityIndex,
    current_doc_id: str,
    current_answer: str,
    next_pool: set[str],
    seen_docs: set[str] | None = None,
    used_answers: set[str] | None = None,
    searcher: Any = None,
    retrieval_k: int = 50,
) -> list[BridgeCandidate]:
    """Find bridges from current_doc to docs in next_pool.

    With searcher: BM25 narrows to topically relevant docs, then entity matching.
    Without searcher: exhaustive over pool (capped for entity path).
    """
    seen = seen_docs or set()
    used = {a.lower() for a in (used_answers or set())}
    search_pool = next_pool - seen

    if not search_pool:
        return []

    if searcher is not None:
        query = _retrieval_query(entity_index.doc_text(current_doc_id), current_answer)
        hits = searcher.search(query, k=retrieval_k, mode="bm25")
        bm25_pool = {h.doc_id for h in hits} & search_pool

        fast_candidates, entity_candidates = _find_in_pool(
            entity_index, current_doc_id, current_answer, bm25_pool, used,
        )
        candidates = fast_candidates + entity_candidates

        logger.info(
            "find_bridge: %d candidates (fast=%d, entity=%d, bm25_pool=%d) from %s with answer=%r",
            len(candidates), len(fast_candidates), len(entity_candidates),
            len(bm25_pool), current_doc_id[:40], current_answer,
        )
    else:
        _EXHAUSTIVE_ENTITY_LIMIT = 1000
        answer_lower = current_answer.strip().lower()
        current_doc_text_lower = entity_index.doc_text(current_doc_id).lower()
        current_entities = entity_index.entities_in_doc(current_doc_id)
        seen_keys: set[tuple[str, str]] = set()

        fast_candidates: list[BridgeCandidate] = []
        entity_candidates: list[BridgeCandidate] = []

        for doc_id in entity_index.docs_containing_text(current_answer, pool=search_pool):
            key = (doc_id, answer_lower)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fast_candidates.append(BridgeCandidate(
                target_doc_id=doc_id,
                bridge_entity=current_answer.strip(),
                needs_intra=False,
            ))

        if len(search_pool) <= _EXHAUSTIVE_ENTITY_LIMIT:
            for ent in current_entities:
                ent_lower = ent.lower()
                if ent_lower == answer_lower or len(ent_lower) < 3 or ent_lower in used or not _is_good_bridge(ent):
                    continue
                if ent_lower in answer_lower or answer_lower in ent_lower:
                    continue
                for doc_id in entity_index.docs_containing_text(ent, pool=search_pool):
                    key = (doc_id, ent_lower)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    proximity = _min_distance(current_doc_text_lower, current_answer, ent)
                    entity_candidates.append(BridgeCandidate(
                        target_doc_id=doc_id,
                        bridge_entity=ent,
                        needs_intra=True,
                        score=-proximity,
                    ))

        entity_candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = fast_candidates + entity_candidates

        logger.info(
            "find_bridge (no searcher): %d candidates (fast=%d, entity=%d) from %s with answer=%r",
            len(candidates), len(fast_candidates), len(entity_candidates),
            current_doc_id[:40], current_answer,
        )

    return candidates
