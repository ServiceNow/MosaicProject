"""Precompute spaCy NER entities for all local documents and update the entity index cache.

Usage:
    python -m trajectory_generation.scripts.precompute_entities

Loads chunks_local_all.jsonl, runs spaCy NER on every local doc, merges with
any existing cache, and writes the updated entity_index_{model}.json.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import spacy

from trajectory_generation.config import CHUNKS_LOCAL, ENTITY_INDEX_DIR, SPACY_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPACY_CHAR_LIMIT = 100_000
MIN_ENTITY_LEN = 2


def extract_entities(nlp, text: str) -> list[str]:
    doc = nlp(text[:SPACY_CHAR_LIMIT])
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        clean = ent.text.strip()
        if "\n" in clean or "\r" in clean:
            continue
        key = clean.lower()
        if len(key) < MIN_ENTITY_LEN or key in seen:
            continue
        seen.add(key)
        entities.append(clean)
    return entities


def load_local_docs(chunks_path: Path) -> dict[str, str]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with open(chunks_path) as f:
        for line in f:
            row = json.loads(line)
            did = row.get("doc_id")
            if did:
                by_doc[did].append(row)

    docs: dict[str, str] = {}
    for doc_id, chunks in by_doc.items():
        chunks.sort(key=lambda c: ((c.get("offsets") or {}).get("start", 0), c.get("chunk_id", "")))
        docs[doc_id] = "\n\n".join(c.get("text", "") for c in chunks)
    return docs


def main():
    cache_path = ENTITY_INDEX_DIR / f"entity_index_{SPACY_MODEL}.json"

    # Load existing cache
    existing: dict[str, list[str]] = {}
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        existing = data.get("doc_entities", {})
        logger.info("Loaded existing cache: %d docs", len(existing))

    # Load all local docs
    logger.info("Loading local docs from %s", CHUNKS_LOCAL)
    local_docs = load_local_docs(CHUNKS_LOCAL)
    logger.info("Loaded %d local documents", len(local_docs))

    # Find docs that need entity extraction
    need_extraction = [did for did in local_docs if did not in existing]
    logger.info("Need extraction: %d (already cached: %d)", len(need_extraction), len(local_docs) - len(need_extraction))

    if not need_extraction:
        logger.info("All local docs already cached, nothing to do")
        return

    # Load spaCy
    logger.info("Loading spaCy model %s", SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL)

    # Extract entities for uncached docs
    t0 = time.time()
    new_count = 0
    total_entities = 0
    for i, doc_id in enumerate(need_extraction):
        text = local_docs[doc_id]
        entities = extract_entities(nlp, text)
        existing[doc_id] = entities
        new_count += 1
        total_entities += len(entities)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(need_extraction) - i - 1) / rate
            logger.info(
                "  %d/%d docs (%.1f docs/s, ~%.0fs remaining)",
                i + 1, len(need_extraction), rate, remaining,
            )

    elapsed = time.time() - t0
    logger.info(
        "Extracted entities for %d docs in %.1fs (%.1f docs/s, %d total entities)",
        new_count, elapsed, new_count / elapsed if elapsed > 0 else 0, total_entities,
    )

    # Save updated cache
    cache_path.write_text(json.dumps({"doc_entities": existing}, ensure_ascii=False))
    logger.info("Saved updated cache to %s (%d total docs)", cache_path, len(existing))

    # Summary by task
    tasks: dict[str, int] = defaultdict(int)
    for did in existing:
        if did.startswith("local/"):
            parts = did.split("/")
            if len(parts) >= 2:
                tasks[parts[1]] += 1
    logger.info("Tasks with cached entities: %d", len(tasks))


if __name__ == "__main__":
    main()
