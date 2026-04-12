"""Entity index for document-to-document matching."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_ENTITY_INDEX_VERSION = 2
_SPACY_WINDOW_CHARS = 20_000
_SPACY_WINDOW_OVERLAP = 1_000
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


def _iter_text_windows(text: str) -> list[str]:
    if len(text) <= _SPACY_WINDOW_CHARS:
        return [text]

    windows: list[str] = []
    step = max(1, _SPACY_WINDOW_CHARS - _SPACY_WINDOW_OVERLAP)
    start = 0
    while start < len(text):
        end = min(len(text), start + _SPACY_WINDOW_CHARS)
        windows.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return windows


def _extract_entities(nlp, text: str) -> list[str]:
    windows = _iter_text_windows(text)
    seen: set[str] = set()
    entities: list[str] = []

    docs = nlp.pipe(windows, batch_size=4) if len(windows) > 1 else [nlp(windows[0])]
    for doc in docs:
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
        self._cache_version = _ENTITY_INDEX_VERSION
        self._stale_docs: set[str] = set()

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

    def extract_entities(self, text: str) -> list[str]:
        if self._nlp is None:
            return []
        return _extract_entities(self._nlp, text)

    def entities_in_doc(self, doc_id: str) -> list[str]:
        if doc_id in self._doc_entities and doc_id not in self._stale_docs:
            return self._doc_entities[doc_id]
        text = self._doc_texts.get(doc_id)
        if text is None or self._nlp is None:
            return self._doc_entities.get(doc_id, [])
        with self._lazy_lock:
            if doc_id in self._doc_entities and doc_id not in self._stale_docs:
                return self._doc_entities[doc_id]
            old_entities = self._doc_entities.get(doc_id, [])
            for ent in old_entities:
                docs = self._entity_to_docs.get(ent.lower())
                if docs is not None:
                    docs.discard(doc_id)
                    if not docs:
                        self._entity_to_docs.pop(ent.lower(), None)
            entities = _extract_entities(self._nlp, text)
            self._doc_entities[doc_id] = entities
            for ent in entities:
                self._entity_to_docs.setdefault(ent.lower(), set()).add(doc_id)
            self._stale_docs.discard(doc_id)
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
            "version": _ENTITY_INDEX_VERSION,
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
        obj._cache_version = data.get("version", 1)
        obj._stale_docs = set(docs.keys()) if (obj._cache_version < _ENTITY_INDEX_VERSION and nlp is not None) else set()
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
