"""Load and index chunked document JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_chunks(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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
