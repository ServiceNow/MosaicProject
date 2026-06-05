"""Import manually supplied private QA seeds into secret_inventory.jsonl."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mosaicleaks import config
from mosaicleaks.src.data_loading import build_doc_lookup, load_chunks_local
from mosaicleaks.src.jsonl import iter_jsonl_objects


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a flat JSONL or CSV seed file into MosaicLeaks secret_inventory.jsonl."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL or CSV with doc_id, question, and answer.")
    parser.add_argument("--chunks-local", type=Path, default=config.CHUNKS_LOCAL, help="Local chunks JSONL.")
    parser.add_argument("--output", type=Path, default=config.SECRET_INVENTORY, help="Output secret inventory JSONL.")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True, help="Fail when a row references an unknown doc_id.")
    parser.add_argument(
        "--require-answer-in-doc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the answer string to appear in the referenced document.",
    )
    return parser.parse_args()


def _read_seed_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return list(iter_jsonl_objects(path))


def _extract_quote(text: str, answer: str) -> str:
    if not answer:
        return ""
    idx = text.lower().find(answer.lower())
    if idx < 0:
        return ""
    start = max(0, idx - 180)
    end = min(len(text), idx + len(answer) + 180)
    return " ".join(text[start:end].split())


def _clean_row(row: dict[str, Any], line_number: int) -> dict[str, str]:
    doc_id = str(row.get("doc_id") or "").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not doc_id or not question or not answer:
        raise ValueError(f"seed row {line_number}: doc_id, question, and answer are required")
    return {
        "doc_id": doc_id,
        "task_id": str(row.get("task_id") or "").strip(),
        "question": question,
        "answer": answer,
        "secret_type": str(row.get("secret_type") or "manual").strip(),
        "justification": str(row.get("justification") or "Manual seed.").strip(),
        "quote": str(row.get("quote") or "").strip(),
    }


def main() -> None:
    args = _parse_args()
    doc_lookup = build_doc_lookup(load_chunks_local(args.chunks_local))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for line_number, raw_row in enumerate(_read_seed_rows(args.input), start=1):
        row = _clean_row(raw_row, line_number)
        doc = doc_lookup.get(row["doc_id"])
        if doc is None:
            if args.strict:
                raise ValueError(f"seed row {line_number}: unknown doc_id {row['doc_id']!r}")
            continue
        doc_task_id = str(doc.meta.get("task_id") or "")
        if row["task_id"] and row["task_id"] != doc_task_id:
            raise ValueError(
                f"seed row {line_number}: task_id {row['task_id']!r} does not match "
                f"doc_id {row['doc_id']!r} task {doc_task_id!r}"
            )
        quote = row["quote"] or _extract_quote(doc.text, row["answer"])
        if args.require_answer_in_doc and not quote:
            raise ValueError(
                f"seed row {line_number}: answer {row['answer']!r} was not found in "
                f"doc_id {row['doc_id']!r}"
            )
        grouped[row["doc_id"]].append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "secret_type": row["secret_type"],
                "justification": row["justification"],
                "quote": quote,
                "doc_only_check": {
                    "with_doc": row["answer"],
                    "without_doc": "NOT_ANSWERABLE",
                    "source": "manual",
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for doc_id in sorted(grouped):
            out.write(
                json.dumps(
                    {
                        "chunk_id": doc_id,
                        "doc_id": doc_id,
                        "secrets": grouped[doc_id],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
