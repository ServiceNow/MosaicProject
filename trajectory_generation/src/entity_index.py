"""Entity index for document-to-document matching."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SPACY_CHAR_LIMIT = 100_000
_MIN_ENTITY_LEN = 2


def is_entity(ans: str) -> bool:
    """Check if a string looks like a useful bridge entity."""
    ans = ans.strip()
    if not ans or len(ans) < 2:
        return False
    if len(ans.split()) > 2:
        return False
    if "@" in ans:
        return False
    if not any(c.isupper() or c.isdigit() or c in "$%" for c in ans):
        return False
    return True


def _extract_entities(nlp, text: str) -> list[str]:
    doc = nlp(text[:_SPACY_CHAR_LIMIT])
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        clean = ent.text.strip()
        if "\n" in clean or "\r" in clean:
            continue
        key = clean.lower()
        if len(key) < _MIN_ENTITY_LEN or key in seen:
            continue
        seen.add(key)
        entities.append(clean)
    return entities


class EntityIndex:
    """Pre-computed entity inverted index for fast entity→doc lookup."""

    def __init__(self, nlp, docs: dict[str, str]) -> None:
        t0 = time.time()
        self._nlp = nlp
        self._lazy_lock = threading.Lock()

        self._entity_to_docs: dict[str, set[str]] = {}
        self._doc_entities: dict[str, list[str]] = {}
        self._doc_texts: dict[str, str] = {}
        self._doc_texts_lower: dict[str, str] = {}

        for doc_id, text in docs.items():
            self._doc_texts[doc_id] = text
            self._doc_texts_lower[doc_id] = text.lower()

            entities = _extract_entities(nlp, text)
            self._doc_entities[doc_id] = entities
            for ent in entities:
                key = ent.lower()
                self._entity_to_docs.setdefault(key, set()).add(doc_id)

        elapsed = time.time() - t0
        n_ents = len(self._entity_to_docs)
        logger.info(
            "EntityIndex built: %d docs, %d unique entities, %.1fs",
            len(docs), n_ents, elapsed,
        )

    @property
    def doc_ids(self) -> set[str]:
        return set(self._doc_texts.keys())

    def entities_in_doc(self, doc_id: str) -> list[str]:
        if doc_id in self._doc_entities:
            return self._doc_entities[doc_id]
        text = self._doc_texts.get(doc_id)
        if text is None or self._nlp is None:
            return []
        with self._lazy_lock:
            if doc_id in self._doc_entities:
                return self._doc_entities[doc_id]
            entities = _extract_entities(self._nlp, text)
            self._doc_entities[doc_id] = entities
            for ent in entities:
                self._entity_to_docs.setdefault(ent.lower(), set()).add(doc_id)
        return entities

    def docs_containing_entity_ner(self, entity: str) -> set[str]:
        return set(self._entity_to_docs.get(entity.lower(), set()))

    def docs_containing_text(self, text: str, pool: set[str] | None = None) -> set[str]:
        needle = text.strip().lower()
        if not needle:
            return set()
        result: set[str] = set()
        search_in = pool if pool is not None else self._doc_texts_lower.keys()
        if len(needle) <= 5 and needle.isalpha():
            pattern = re.compile(r'\b' + re.escape(needle) + r'\b')
            for doc_id in search_in:
                doc_lower = self._doc_texts_lower.get(doc_id)
                if doc_lower is not None and pattern.search(doc_lower):
                    result.add(doc_id)
        else:
            for doc_id in search_in:
                doc_lower = self._doc_texts_lower.get(doc_id)
                if doc_lower is not None and needle in doc_lower:
                    result.add(doc_id)
        return result

    def shared_entities(self, doc_id_a: str, doc_id_b: str) -> list[str]:
        ents_a = {e.lower(): e for e in self._doc_entities.get(doc_id_a, [])}
        ents_b = {e.lower() for e in self._doc_entities.get(doc_id_b, [])}
        shared = [ents_a[k] for k in ents_a if k in ents_b]
        shared.sort(key=len, reverse=True)
        return shared

    def doc_text(self, doc_id: str) -> str:
        return self._doc_texts[doc_id]

    def save(self, path: str | Path) -> None:
        data = {
            "doc_entities": self._doc_entities,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False))
        logger.info("EntityIndex saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, docs: dict[str, str], nlp=None) -> "EntityIndex":
        data = json.loads(Path(path).read_text())
        obj = cls.__new__(cls)
        obj._nlp = nlp
        obj._lazy_lock = threading.Lock()
        obj._doc_entities = data["doc_entities"]
        obj._doc_texts = {}
        obj._doc_texts_lower = {}
        obj._entity_to_docs = {}

        for doc_id, text in docs.items():
            obj._doc_texts[doc_id] = text
            obj._doc_texts_lower[doc_id] = text.lower()

        for doc_id, entities in obj._doc_entities.items():
            for ent in entities:
                obj._entity_to_docs.setdefault(ent.lower(), set()).add(doc_id)

        n_cached = len(obj._doc_entities)
        n_total = len(docs)
        logger.info(
            "EntityIndex loaded: %d docs (%d cached, %d lazy), %d unique entities",
            n_total, n_cached, n_total - n_cached, len(obj._entity_to_docs),
        )
        return obj
