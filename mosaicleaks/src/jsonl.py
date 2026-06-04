"""JSONL helpers with line-numbered parse errors."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield payload


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_objects(path))


def write_jsonl_objects(
    path: Path,
    rows: Iterable[dict[str, Any]],
    sort_keys: bool = False,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n")
