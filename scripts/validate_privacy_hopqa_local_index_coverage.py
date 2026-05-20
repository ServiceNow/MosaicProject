#!/usr/bin/env python3
"""Check that every local gold document in Privacy HopQA is present in local helper indices."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_DATASET_DIR = Path("/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset")
DEFAULT_DATASETS = [
    DEFAULT_DATASET_DIR / "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl",
    DEFAULT_DATASET_DIR / "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_val.jsonl",
    DEFAULT_DATASET_DIR / "privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_test.jsonl",
]
DEFAULT_LOCAL_INDEX_ROOT = Path(
    "/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/"
    "q30_final_residual_review_20260513/local_indices_goldcoverage_20260519"
)
TASK_FILES_RE = re.compile(r"/tasks/(DR\d+)/files/(.+)$")
LOCAL_DOC_RE = re.compile(r"^local/(DR\d+)/(.+)$")
WINDOW_SUFFIX_RE = re.compile(r"#w\d+of\d+$")
LOCAL_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".ppt",
    ".pptx",
    ".txt",
    ".xls",
    ".xlsx",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def strip_extension(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in LOCAL_EXTENSIONS:
        return path[: -len(suffix)]
    return path


def normalized_local_aliases(value: str | None, task_id: str | None = None) -> set[str]:
    if not value:
        return set()

    text = WINDOW_SUFFIX_RE.sub("", str(value).replace("\\", "/"))
    aliases = {text, strip_extension(text)}

    task_match = TASK_FILES_RE.search(text)
    if task_match:
        matched_task, rel_path = task_match.groups()
        aliases.add(f"local/{matched_task}/{rel_path}")
        aliases.add(f"local/{matched_task}/{strip_extension(rel_path)}")
        aliases.add(rel_path)
        aliases.add(strip_extension(rel_path))

    local_match = LOCAL_DOC_RE.match(text)
    if local_match:
        matched_task, rel_path = local_match.groups()
        aliases.add(rel_path)
        aliases.add(strip_extension(rel_path))
        aliases.add(f"files/{rel_path}")
        aliases.add(strip_extension(f"files/{rel_path}"))
        task_id = task_id or matched_task

    if task_id:
        marker = f"/{task_id}/files/"
        if marker in text:
            rel_path = text.split(marker, 1)[1]
            aliases.add(f"local/{task_id}/{rel_path}")
            aliases.add(f"local/{task_id}/{strip_extension(rel_path)}")
            aliases.add(rel_path)
            aliases.add(strip_extension(rel_path))

    return {alias.strip("/") for alias in aliases if alias}


def document_aliases(document: dict, task_id: str) -> set[str]:
    aliases: set[str] = set()
    def add_value(value: object) -> None:
        if isinstance(value, str):
            aliases.update(normalized_local_aliases(value, task_id=task_id))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    aliases.update(normalized_local_aliases(item, task_id=task_id))

    for key in (
        "doc_id",
        "document_id",
        "id",
        "parent_doc_id",
        "source_path",
        "path",
        "file_path",
        "relative_path",
        "extracted_path",
        "locator",
        "title",
        "filename",
        "file_name",
        "source_identifier",
        "original_path",
    ):
        add_value(document.get(key))

    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        for value in metadata.values():
            add_value(value)
    return aliases


def load_index_aliases(local_index_root: Path, task_id: str) -> set[str]:
    documents_path = local_index_root / task_id / "documents.json"
    if not documents_path.exists():
        return set()

    payload = json.loads(documents_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        documents = payload.get("documents") or payload.get("items") or payload.values()
    else:
        documents = payload

    aliases: set[str] = set()
    for document in documents:
        if isinstance(document, dict):
            aliases.update(document_aliases(document, task_id))
    return aliases


def iter_local_hops(row: dict) -> list[tuple[int, dict]]:
    return [
        (idx, hop)
        for idx, hop in enumerate(row.get("hops", []), start=1)
        if hop.get("hop_type") == "L"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, default=[])
    parser.add_argument("--local-index-root", type=Path, default=DEFAULT_LOCAL_INDEX_ROOT)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    dataset_paths = args.dataset or DEFAULT_DATASETS
    rows_by_split = {path.stem: read_jsonl(path) for path in dataset_paths}
    index_alias_cache: dict[str, set[str]] = {}
    missing: list[dict] = []
    total_local_hops = 0

    for split, rows in rows_by_split.items():
        for row in rows:
            task_id = row["task_id"]
            if task_id not in index_alias_cache:
                index_alias_cache[task_id] = load_index_aliases(args.local_index_root, task_id)
            index_aliases = index_alias_cache[task_id]

            for hop_idx, hop in iter_local_hops(row):
                total_local_hops += 1
                gold_aliases = set()
                gold_aliases.update(normalized_local_aliases(hop.get("doc_id"), task_id=task_id))
                gold_aliases.update(normalized_local_aliases(hop.get("doc_ref"), task_id=task_id))
                if not gold_aliases.intersection(index_aliases):
                    missing.append(
                        {
                            "split": split,
                            "task_id": task_id,
                            "chain_id": row.get("chain_id"),
                            "pattern": row.get("pattern"),
                            "hop": hop_idx,
                            "question": hop.get("question"),
                            "doc_id": hop.get("doc_id"),
                            "doc_ref": hop.get("doc_ref"),
                        }
                    )

    summary = {
        "datasets": [str(path) for path in dataset_paths],
        "local_index_root": str(args.local_index_root),
        "rows": sum(len(rows) for rows in rows_by_split.values()),
        "local_hops": total_local_hops,
        "missing_local_hops": len(missing),
        "missing_unique_docs": len({item["doc_id"] or item["doc_ref"] for item in missing}),
        "missing_examples": missing[: args.max_examples],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
