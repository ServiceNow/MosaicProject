#!/usr/bin/env python3
"""Investigate entity-list failures: why did the model pick X when X wasn't in entity_list?

Usage:
  python -m trajectory_generation.scripts.investigate_entity_fail --doc web/54770 --model-answer Kenya --bridge "72 hours"

Loads entity index, prints: target_doc entities, entity_list (after filters), whether model_answer
appears in doc text, and why it might be missing from entity_list.
"""

import argparse
import sys
from pathlib import Path

# Add repo root
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from trajectory_generation import config
from trajectory_generation.experiments.run_atomic import run_step_0_load
from trajectory_generation.src.entity_index import is_entity
from trajectory_generation.src.find_bridge import sort_entities_by_proximity

MAX_ENTITY_LIST = 20


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--doc", required=True, help="Target doc_id (e.g. web/54770)")
    p.add_argument("--model-answer", required=True, help="Answer the model gave (e.g. Kenya)")
    p.add_argument("--bridge", required=True, help="Bridge entity for that candidate")
    p.add_argument("--used", nargs="*", default=["72 hours"], help="used_answers")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/investigate_entity"))
    p.add_argument("--spacy-model", default=None, help="Override config (e.g. en_core_web_sm if trf unavailable)")
    args = p.parse_args()

    if args.spacy_model:
        config.SPACY_MODEL = args.spacy_model
    loaded = run_step_0_load(args.output_dir, task_id=None, use_bm25=False)
    entity_index = loaded["entity_index"]

    if args.doc not in entity_index.doc_ids:
        print(f"Doc {args.doc} not found in entity index (have {len(entity_index.doc_ids)} docs)")
        return 1

    target_entities = entity_index.entities_in_doc(args.doc)
    used_answers = {a.lower() for a in args.used}
    bridge_lower = args.bridge.lower()
    doc_text = entity_index.doc_text(args.doc)
    filtered = [
        e for e in target_entities
        if e.lower() != bridge_lower
        and e.lower() not in used_answers
        and len(e) >= 3
        and is_entity(e)
    ]
    entity_list = sort_entities_by_proximity(doc_text, args.bridge, filtered)[:MAX_ENTITY_LIST]
    model_answer = args.model_answer
    in_doc = model_answer.lower() in doc_text.lower()
    in_target_entities = model_answer in target_entities or any(
        e.lower() == model_answer.lower() for e in target_entities
    )
    in_entity_list = model_answer in entity_list or any(
        e.lower() == model_answer.lower() for e in entity_list
    )

    print("=== Entity list investigation ===\n")
    print(f"Target doc: {args.doc}")
    print(f"Bridge entity: {args.bridge}")
    print(f"Model answer: {model_answer}\n")

    print(f"1. Does '{model_answer}' appear in doc text? {in_doc}")
    if in_doc:
        idx = doc_text.lower().find(model_answer.lower())
        snippet = doc_text[max(0, idx - 50) : idx + len(model_answer) + 50]
        print(f"   Snippet: ...{snippet}...\n")

    print(f"2. Is '{model_answer}' in target_entities (spaCy NER)? {in_target_entities}")
    print(f"   Total target_entities: {len(target_entities)}")
    print(f"   First 30: {target_entities[:30]}\n")

    print(f"3. Is '{model_answer}' in entity_list (after filters)? {in_entity_list}")
    print(f"   entity_list (max 20): {entity_list}\n")

    print("4. Why might it be excluded?")
    if model_answer.lower() == bridge_lower:
        print("   - Excluded: model_answer == bridge_entity")
    elif model_answer.lower() in used_answers:
        print("   - Excluded: model_answer in used_answers")
    elif len(model_answer) < 3:
        print("   - Excluded: len < 3")
    elif not is_entity(model_answer):
        print("   - Excluded: is_entity() returned False")
    elif not in_target_entities:
        print("   - Excluded: spaCy NER did not extract it from doc")
        print("     (NER can miss entities; limited to first 100K chars)")
    else:
        print("   - Would be in list if not filtered by bridge/used/len/is_entity")
        print("   - Check if it's beyond position 20 (we cap at MAX_ENTITY_LIST=20)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
