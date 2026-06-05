"""Load and index chunked document JSONL files."""

from pathlib import Path
from typing import Any, Iterator

from mosaicleaks.src.jsonl import iter_jsonl_objects


def iter_chunks(path: Path) -> Iterator[dict[str, Any]]:
    yield from iter_jsonl_objects(path)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return list(iter_chunks(path))


def build_chunk_store(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("chunk_id")
        if not cid:
            continue
        store[str(cid)] = rec
    return store
