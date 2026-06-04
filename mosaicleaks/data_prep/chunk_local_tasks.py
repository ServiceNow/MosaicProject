"""Build local document chunks from a DrBench-style task directory."""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import docx
import openpyxl
import pypdf

logger = logging.getLogger(__name__)

# Source task folders can contain corrupt office/PDF files. The individual
# readers return None for unreadable files so one bad attachment does not stop
# chunking the rest of the task.


@dataclass(frozen=True)
class ChunkConfig:
    """Chunk sizing knobs for local private documents."""

    target_words: int = 450
    overlap_words: int = 50
    min_words: int = 300


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk local task files into chunks_local.jsonl.")
    parser.add_argument("--tasks-root", type=Path, required=True, help="Root directory containing DR task folders.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--target-words", type=int, default=450, help="Approximate target words per chunk.")
    parser.add_argument("--overlap-words", type=int, default=50, help="Approximate overlap words between chunks.")
    parser.add_argument("--min-words", type=int, default=300, help="Try to avoid chunks shorter than this.")
    parser.add_argument("--include-tasks", nargs="*", default=None, help="Optional task IDs to include.")
    parser.add_argument(
        "--prefer-md",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a PDF when a same-stem Markdown file exists next to it.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _list_task_dirs(tasks_root: Path, include_tasks: list[str] | None) -> list[Path]:
    task_dirs: list[Path] = []
    include = set(include_tasks or [])
    for entry in sorted(tasks_root.iterdir()):
        if not entry.is_dir():
            continue
        if not re.fullmatch(r"DR\d{4}", entry.name):
            continue
        if include and entry.name not in include:
            continue
        task_dirs.append(entry)
    return task_dirs


def _load_company_info(task_dir: Path) -> dict[str, str]:
    context_path = task_dir / "context.json"
    if not context_path.exists():
        return {}
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context_path}: invalid JSON: {exc}") from exc
    company_name = ((context.get("company_info") or {}).get("name") or "").strip()
    if not company_name:
        return {}
    return {
        "company_name": company_name,
        "company": _slugify(company_name),
    }


def _should_skip_file(path: Path) -> bool:
    return path.name in {"qa_dict.json", "file_dict.json"}


def _prefer_markdown(path: Path) -> bool:
    return path.suffix.lower() == ".pdf" and path.with_suffix(".md").exists()


def _read_pdf(path: Path) -> str | None:
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        # pypdf uses several exception types for corrupt source files.
        logger.warning("Failed to open PDF %s: %s", path, exc)
        return None
    parts: list[str] = []
    page_failures: list[int] = []
    first_error: Exception | None = None
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # Keep the rest of the PDF when pypdf fails on an individual page.
            if first_error is None:
                first_error = exc
            page_failures.append(page_number)
            text = ""
        if text.strip():
            parts.append(text)
    if page_failures:
        logger.warning(
            "Failed to extract text from %d page(s) in %s; first failure on page %d: %s",
            len(page_failures),
            path,
            page_failures[0],
            first_error,
        )
    return "\n\n".join(parts).strip()


def _read_docx(path: Path) -> str | None:
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        # python-docx wraps many invalid archive/XML errors; treat them as
        # unreadable source files and keep processing the task.
        logger.warning("Failed to open DOCX %s: %s", path, exc)
        return None
    return "\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()


def _read_xlsx(path: Path) -> str | None:
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        # openpyxl can raise different errors for corrupt workbooks.
        logger.warning("Failed to open spreadsheet %s: %s", path, exc)
        return None
    lines: list[str] = []
    try:
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows(values_only=True):
                line = " | ".join("" if cell is None else str(cell) for cell in row).strip()
                if line and line.replace("|", "").strip():
                    lines.append(line)
    finally:
        workbook.close()
    return "\n".join(lines).strip()


def _read_jsonl(path: Path) -> str | None:
    lines: list[str] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        logger.warning("Failed to decode JSONL %s as UTF-8: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning("Failed to read JSONL %s: %s", path, exc)
        return None
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            # Some task-local JSONL exports contain plain-text fallback lines.
            lines.append(raw)
            continue
        if isinstance(row, dict) and "user" in row and "message" in row:
            lines.append(f"{row['user']}: {row['message']}")
        elif isinstance(row, dict) and "from" in row and "body" in row:
            lines.append(f"From: {row.get('from', '')}")
            if row.get("to"):
                lines.append(f"To: {row['to']}")
            if row.get("subject"):
                lines.append(f"Subject: {row['subject']}")
            lines.append(str(row.get("body", "")))
        else:
            lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines).strip()


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1").strip()
        except OSError as exc:
            logger.warning("Failed to read text file %s: %s", path, exc)
            return None
        except UnicodeDecodeError as exc:
            logger.warning("Failed to decode text file %s as UTF-8 or latin-1: %s", path, exc)
            return None
    except OSError as exc:
        logger.warning("Failed to read text file %s: %s", path, exc)
        return None


def _read_file(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in {".md", ".txt", ".csv"}:
        return _read_text_file(path)
    if ext == ".jsonl":
        return _read_jsonl(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in {".doc", ".docx"}:
        return _read_docx(path)
    if ext in {".xls", ".xlsx"}:
        return _read_xlsx(path)
    return _read_text_file(path)


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    title: str | None = None
    h2: str | None = None
    h3: str | None = None
    current_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    for line in text.splitlines():
        if line.startswith(("# ", "## ", "### ")):
            content = "\n".join(current_lines).strip()
            if content:
                prefix_parts = [part for part in (title, h2, h3) if part]
                sections.append((" > ".join(prefix_parts), content))
            current_lines = []

        if line.startswith("# "):
            title = line[2:].strip()
            h2 = None
            h3 = None
            continue
        if line.startswith("## "):
            h2 = line[3:].strip()
            h3 = None
            continue
        if line.startswith("### "):
            h3 = line[4:].strip()
            continue
        current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            prefix_parts = [part for part in (title, h2, h3) if part]
            sections.append((" > ".join(prefix_parts), content))
    return sections


def _chunk_paragraphs(paragraphs: list[str], cfg: ChunkConfig) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def word_count(text: str) -> int:
        return len(text.split())

    for paragraph in paragraphs:
        paragraph_words = word_count(paragraph)
        if current and current_words >= cfg.min_words and current_words + paragraph_words > cfg.target_words:
            chunks.append("\n\n".join(current).strip())
            overlap = " ".join(" ".join(current).split()[-cfg.overlap_words :]).strip()
            current = [overlap] if overlap else []
            current_words = word_count(overlap)
        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunks.append("\n\n".join(current).strip())

    if len(chunks) >= 2 and word_count(chunks[-1]) < cfg.min_words:
        merged = chunks[-2].rstrip() + "\n\n" + chunks[-1].lstrip()
        if word_count(merged) <= cfg.target_words + cfg.min_words:
            chunks[-2] = merged
            chunks.pop()
    return [chunk for chunk in chunks if chunk]


def _chunk_text(text: str, extension: str, cfg: ChunkConfig) -> list[str]:
    if extension == ".md":
        sections = _split_markdown_sections(text)
        if sections:
            stitched: list[str] = []
            for prefix, section_text in sections:
                if prefix:
                    stitched.append(prefix)
                stitched.append(section_text)
            text = "\n\n".join(stitched)
    paragraphs = _paragraphs(text)
    if not paragraphs:
        return []
    return _chunk_paragraphs(paragraphs, cfg)


def _file_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
        if stripped:
            return stripped[:160]
    return fallback


def _doc_id(task_id: str, relative_path: Path) -> str:
    return f"local/{task_id}/{relative_path.as_posix()}"


def _emit_doc_chunks(
    task_id: str,
    relative_path: Path,
    text: str,
    company_info: dict[str, str],
    cfg: ChunkConfig,
) -> list[dict]:
    extension = relative_path.suffix.lower()
    doc_id = _doc_id(task_id, relative_path)
    chunks = _chunk_text(text, extension, cfg)
    paragraphs = _paragraphs(text)
    subdir = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
    meta = {
        "task_id": task_id,
        "subdir": subdir,
        "filename": relative_path.name,
        "relative_path": f"{task_id}/files/{relative_path.as_posix()}",
        "extension": extension,
        "company_name": company_info.get("company_name", ""),
        "company": company_info.get("company", ""),
        "file_title": _file_title(text, Path(relative_path.name).stem),
        "file_name": Path(relative_path.name).stem,
        "introduction": paragraphs[0][:800] if paragraphs else "",
    }

    rows: list[dict] = []
    start = 0
    for idx, chunk in enumerate(chunks, start=1):
        end = start + len(chunk)
        rows.append({
            "chunk_id": f"{doc_id}#{idx:04d}",
            "doc_id": doc_id,
            "source_type": "local",
            "text": chunk,
            "offsets": {"start": start, "end": end},
            "meta": meta,
        })
        start = end + 2
    return rows


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = ChunkConfig(
        target_words=args.target_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
    )
    task_dirs = _list_task_dirs(args.tasks_root, args.include_tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for task_dir in task_dirs:
            task_id = task_dir.name
            company_info = _load_company_info(task_dir)
            files_dir = task_dir / "files"
            if not files_dir.exists():
                continue
            for file_path in sorted(path for path in files_dir.rglob("*") if path.is_file()):
                if _should_skip_file(file_path):
                    continue
                if args.prefer_md and _prefer_markdown(file_path):
                    continue
                text = _read_file(file_path)
                if text is None:
                    continue
                if len(text.strip()) < 50:
                    continue
                relative_path = file_path.relative_to(files_dir)
                rows = _emit_doc_chunks(
                    task_id=task_id,
                    relative_path=relative_path,
                    text=text,
                    company_info=company_info,
                    cfg=cfg,
                )
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    main()
