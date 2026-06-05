"""Build local chunks from a generic folder or manifest of private documents."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from mosaicleaks.data_prep.chunk_local_tasks import (
    ChunkConfig,
    emit_local_doc_chunks,
    read_local_file,
    slugify_company,
)
from mosaicleaks.src.jsonl import iter_jsonl_objects

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk arbitrary local documents into MosaicLeaks chunks_local.jsonl."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-root",
        type=Path,
        help="Folder of documents to treat as one local task.",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="JSONL manifest with path, task_id, and optional company_name fields.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output chunks JSONL path.")
    parser.add_argument("--task-id", help="Task ID used with --input-root.")
    parser.add_argument("--company-name", default="", help="Company/group label used with --input-root.")
    parser.add_argument("--target-words", type=int, default=450, help="Approximate target words per chunk.")
    parser.add_argument("--overlap-words", type=int, default=50, help="Approximate overlap words between chunks.")
    parser.add_argument("--min-words", type=int, default=300, help="Try to avoid chunks shorter than this.")
    parser.add_argument("--min-chars", type=int, default=50, help="Skip files with less extracted text.")
    parser.add_argument(
        "--prefer-md",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a PDF when a same-stem Markdown file exists next to it.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print skipped-file warnings.")
    return parser.parse_args()


def _company_info(company_name: str) -> dict[str, str]:
    company_name = company_name.strip()
    if not company_name:
        return {}
    return {
        "company_name": company_name,
        "company": slugify_company(company_name),
    }


def _should_read(path: Path, prefer_md: bool) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    if prefer_md and path.suffix.lower() == ".pdf" and path.with_suffix(".md").exists():
        return False
    return True


def _folder_records(input_root: Path, task_id: str, company_name: str, prefer_md: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("*")):
        if not _should_read(path, prefer_md):
            continue
        records.append(
            {
                "path": path,
                "relative_path": path.relative_to(input_root),
                "task_id": task_id,
                "company_name": company_name,
            }
        )
    return records


def _manifest_records(manifest_path: Path, prefer_md: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifest_root = manifest_path.parent
    for line_number, row in enumerate(iter_jsonl_objects(manifest_path), start=1):
        raw_path = row.get("path")
        task_id = str(row.get("task_id") or "").strip()
        if not raw_path or not task_id:
            raise ValueError(f"{manifest_path}:{line_number}: path and task_id are required")
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = manifest_root / path
        path = path.resolve()
        if not _should_read(path, prefer_md):
            logger.warning("Skipping unsupported or missing file from manifest: %s", path)
            continue
        relative_path = row.get("relative_path")
        if relative_path:
            rel = Path(str(relative_path))
        else:
            rel = Path(path.name)
        records.append(
            {
                "path": path,
                "relative_path": rel,
                "task_id": task_id,
                "company_name": str(row.get("company_name") or ""),
            }
        )
    return records


def _build_chunks(record: dict[str, Any], cfg: ChunkConfig, min_chars: int) -> list[dict]:
    path = record["path"]
    text = read_local_file(path)
    if text is None or len(text.strip()) < min_chars:
        return []
    return emit_local_doc_chunks(
        task_id=record["task_id"],
        relative_path=record["relative_path"],
        text=text,
        company_info=_company_info(record.get("company_name", "")),
        cfg=cfg,
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    cfg = ChunkConfig(
        target_words=args.target_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
    )
    if args.input_root is not None:
        if not args.task_id:
            raise ValueError("--task-id is required with --input-root")
        records = _folder_records(
            input_root=args.input_root.expanduser().resolve(),
            task_id=args.task_id,
            company_name=args.company_name,
            prefer_md=args.prefer_md,
        )
    else:
        records = _manifest_records(args.manifest.expanduser().resolve(), args.prefer_md)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    total_docs = 0
    with args.output.open("w", encoding="utf-8") as out:
        for record in records:
            chunks = _build_chunks(record, cfg=cfg, min_chars=args.min_chars)
            if not chunks:
                logger.warning("No chunks emitted for %s", record["path"])
                continue
            total_docs += 1
            total_chunks += len(chunks)
            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks from %d documents to %s", total_chunks, total_docs, args.output)


if __name__ == "__main__":
    main()
