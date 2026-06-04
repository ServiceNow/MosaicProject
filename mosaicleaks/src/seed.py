"""Step 1: Select seed for chain. L-seeds from inventory, W-seeds via LLM."""

import logging
import random
from typing import Sequence

from mosaicleaks import config
from mosaicleaks.src.data_loading import LocalDoc, Secret
from mosaicleaks.src.llm import LLMClient
from mosaicleaks.src.parsing import parse_qa_with_quote
from mosaicleaks.src.entity_index import EntityIndex, is_entity
from mosaicleaks.src.types import ChainState, HopRecord

logger = logging.getLogger(__name__)


def select_seed(
    secrets: Sequence[Secret],
    doc_lookup: dict[str, LocalDoc],
    *,
    pattern: str = "LW",
    task_id: str | None = None,
    company: str | None = None,
    rng: random.Random | None = None,
    require_entity: bool = True,
) -> ChainState:
    candidates = list(secrets)
    if require_entity:
        candidates = [s for s in candidates if is_entity(s.answer)]
    if not candidates:
        raise ValueError("No seed secrets available after filtering.")

    rng = rng or random.Random()
    secret = rng.choice(candidates)

    doc = doc_lookup[secret.doc_id]

    hop = HopRecord(
        hop_number=1,
        hop_type="L",
        question=secret.question,
        answer=secret.answer,
        doc_id=secret.doc_id,
        doc_text=doc.text,
    )

    state = ChainState(
        pattern=pattern,
        hop_history=[hop],
        global_question=secret.question,
        global_answer=secret.answer,
        used_doc_ids={secret.doc_id},
        task_id=task_id or doc.meta.get("task_id"),
        company=company or doc.meta.get("company_name"),
    )
    return state


PROMPT_WEB_SEED = '''\
Given this document, write a factual question whose answer is exactly "{entity}".
The question should be specific and answerable from the document text alone.
The answer "{entity}" must NOT appear in the question text.

<document>
{doc_text}
</document>

<answer>
QUOTE: <exact sentence from document containing "{entity}">
QUESTION: ... (must NOT contain "{entity}")
ANSWER: {entity}
</answer>'''


def select_web_seed(
    entity_index: EntityIndex,
    web_doc_ids: set[str],
    next_pool: set[str],
    llm: LLMClient,
    *,
    pattern: str = "WL",
    task_id: str | None = None,
    company: str | None = None,
    rng: random.Random | None = None,
) -> ChainState:
    rng = rng or random.Random()

    _MAX_WEB_SAMPLE = 200
    web_list = list(web_doc_ids)
    rng.shuffle(web_list)

    candidates: list[tuple[str, str]] = []
    for web_id in web_list[:_MAX_WEB_SAMPLE]:
        entities = entity_index.entities_in_doc(web_id)
        for ent in entities:
            if not is_entity(ent):
                continue
            if entity_index.docs_containing_text(ent, pool=next_pool):
                candidates.append((web_id, ent))

    if not candidates:
        pool_label = "local" if next_pool != web_doc_ids else "web"
        raise ValueError(f"No bridgeable web docs (to {pool_label}) in sample of {min(_MAX_WEB_SAMPLE, len(web_list))}.")

    rng.shuffle(candidates)
    logger.info("Web seed: %d bridgeable (doc, entity) candidates", len(candidates))

    for web_doc_id, entity in candidates[:20]:
        doc_text = entity_index.doc_text(web_doc_id)
        context = doc_text[:8000]

        prompt = PROMPT_WEB_SEED.format(entity=entity, doc_text=context)
        try:
            raw = llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=config.SHORT_CALL_MAX_TOKENS,
                use_min_max_tokens=False,
            )
        except Exception as e:
            logger.debug("Web seed LLM call failed: %s", e)
            continue

        result = parse_qa_with_quote(raw)
        if result is None:
            logger.debug("Web seed parse failed for entity=%r", entity)
            continue

        question, answer, quote = result

        if answer.lower().strip() != entity.lower().strip():
            logger.debug("Web seed answer mismatch: got %r, expected %r", answer, entity)
            continue

        if entity.lower() in question.lower():
            logger.debug("Web seed: entity %r leaked into question", entity)
            continue

        logger.info("Web seed OK: Q=%r A=%r doc=%s", question[:60], answer, web_doc_id[:40])

        hop = HopRecord(
            hop_number=1,
            hop_type="W",
            question=question,
            answer=answer,
            doc_id=web_doc_id,
            doc_text=doc_text,
            quote=quote,
        )

        return ChainState(
            pattern=pattern,
            hop_history=[hop],
            global_question=question,
            global_answer=answer,
            used_doc_ids={web_doc_id},
            task_id=task_id,
            company=company,
        )

    raise ValueError("Failed to generate web seed after 20 attempts.")
