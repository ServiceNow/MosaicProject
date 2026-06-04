"""Chunk DrBench seed-URL docs into passage-sized retrieval units.

Input: docs JSONL (from fetch step) with doc_id, url, text, industry, domain, task_ids, etc.
Output: chunks JSONL with chunk_id, doc_id, text, offsets, meta.

Chunking is paragraph/heading aware:
- target_words ~260, overlap ~50, min ~140 (configurable)
- Splits at paragraph boundaries, adds heading prefix (Title > H2 > H3)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mosaicleaks.data_prep.drbench_urls import WEB_POOL
from mosaicleaks.src.jsonl import iter_jsonl_objects


@dataclass(frozen=True)
class ParaBlock:
    text: str
    start: int
    end: int
    title: str = ""
    h2: str = ""
    h3: str = ""


def _finish_paragraph(
    *,
    lines: list[str],
    start: int | None,
    end: int,
    title: str,
    h2: str,
    h3: str,
) -> ParaBlock | None:
    if not lines or start is None:
        return None
    para_text = "\n".join(lines).strip()
    if not para_text:
        return None
    return ParaBlock(text=para_text, start=start, end=end, title=title, h2=h2, h3=h3)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def parse_paragraph_blocks(text: str) -> list[ParaBlock]:
    """Parse markdown-ish doc into paragraph blocks with heading context and offsets."""
    blocks: list[ParaBlock] = []
    title = ""
    h2 = ""
    h3 = ""
    pos = 0
    current_lines: list[str] = []
    para_start: int | None = None

    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        s = line.strip()
        if not s:
            block = _finish_paragraph(
                lines=current_lines,
                start=para_start,
                end=pos,
                title=title,
                h2=h2,
                h3=h3,
            )
            if block is not None:
                blocks.append(block)
            current_lines = []
            para_start = None
            continue
        if s.startswith("#"):
            block = _finish_paragraph(
                lines=current_lines,
                start=para_start,
                end=line_start,
                title=title,
                h2=h2,
                h3=h3,
            )
            if block is not None:
                blocks.append(block)
            current_lines = []
            para_start = None
            level = len(s) - len(s.lstrip("#"))
            if level in {1, 2, 3} and len(s) > level and s[level] == " ":
                heading_text = s[level + 1 :].strip()
                if level == 1:
                    title, h2, h3 = heading_text, "", ""
                elif level == 2:
                    h2, h3 = heading_text, ""
                else:
                    h3 = heading_text
            continue
        if para_start is None:
            para_start = line_start
        current_lines.append(s)

    block = _finish_paragraph(
        lines=current_lines,
        start=para_start,
        end=pos,
        title=title,
        h2=h2,
        h3=h3,
    )
    if block is not None:
        blocks.append(block)
    return blocks


def _count_words(text: str) -> int:
    return len(text.split())


def _chunk_blocks(
    blocks: list[ParaBlock],
    *,
    target_words: int,
    overlap_words: int,
    min_words: int,
) -> list[list[ParaBlock]]:
    chunks: list[list[ParaBlock]] = []
    current: list[ParaBlock] = []
    current_words = 0

    def carry_overlap(prev: list[ParaBlock]) -> list[ParaBlock]:
        if overlap_words <= 0:
            return []
        tail: list[ParaBlock] = []
        tail_words = 0
        for b in reversed(prev):
            tail.insert(0, b)
            tail_words += _count_words(b.text)
            if tail_words >= overlap_words:
                break
        return tail

    for b in blocks:
        w = _count_words(b.text)
        if current and current_words >= min_words and current_words + w > target_words:
            chunks.append(current)
            current = carry_overlap(current)
            current_words = sum(_count_words(x.text) for x in current)
        current.append(b)
        current_words += w

    if current:
        chunks.append(current)

    if len(chunks) >= 2:
        last = chunks[-1]
        last_words = sum(_count_words(x.text) for x in last)
        if last_words < min_words:
            merged = chunks[-2] + last
            merged_words = sum(_count_words(x.text) for x in merged)
            if merged_words <= target_words + min_words:
                chunks[-2] = merged
                chunks.pop()

    return chunks


def _heading_prefix(block: ParaBlock) -> str:
    parts = [p for p in [block.title, block.h2, block.h3] if p]
    return " > ".join(parts)


def emit_chunk_records(
    doc: dict[str, Any],
    blocks: list[ParaBlock],
    *,
    target_words: int = 260,
    overlap_words: int = 50,
    min_words: int = 140,
) -> list[dict[str, Any]]:
    """Emit chunk records for a doc."""
    doc_id = doc.get("doc_id", "")
    meta_base = {
        "web_pool": WEB_POOL,
        "url": doc.get("url"),
        "industry": doc.get("industry"),
        "domain": doc.get("domain"),
        "seed_date": doc.get("seed_date"),
        "task_ids": doc.get("task_ids") or [],
        "title": doc.get("title") or "",
    }
    chunk_groups = _chunk_blocks(
        blocks, target_words=target_words, overlap_words=overlap_words, min_words=min_words
    )
    records: list[dict[str, Any]] = []
    for idx, group in enumerate(chunk_groups, start=1):
        if not group:
            continue
        prefix = _heading_prefix(group[0])
        body = "\n\n".join(b.text.strip() for b in group if b.text.strip())
        chunk_text = (prefix + "\n\n" + body) if prefix else body
        chunk_text = _normalize_text(chunk_text)
        if not chunk_text:
            continue
        records.append({
            "chunk_id": f"{doc_id}#{idx:04d}",
            "doc_id": doc_id,
            "source_type": "web",
            "text": chunk_text,
            "offsets": {"start": int(group[0].start), "end": int(group[-1].end)},
            "meta": dict(meta_base),
        })
    return records


def _is_chunkable_doc(obj: dict[str, Any]) -> bool:
    """Keep only successful direct-fetch docs in the curated retrieval sidecar."""
    if obj.get("extractor") == "insights_fallback":
        return False
    text = (obj.get("text") or "").strip()
    err = obj.get("error")
    status = obj.get("http_status")
    return bool(text) and not err and isinstance(status, int) and 200 <= status < 300


def load_docs_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load docs JSONL, keeping latest success per doc_id."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for obj in iter_jsonl_objects(path):
        doc_id = obj.get("doc_id")
        if not doc_id:
            continue
        if doc_id not in by_id:
            order.append(doc_id)
        if _is_chunkable_doc(obj):
            by_id[doc_id] = obj
    return [by_id[d] for d in order if d in by_id]


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk DrBench seed URL docs.")
    parser.add_argument("--docs", required=True, help="Docs JSONL from fetch step")
    parser.add_argument("--output", required=True, help="Output chunks JSONL")
    parser.add_argument("--target-words", type=int, default=260)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=140)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    docs_path = Path(args.docs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs = load_docs_jsonl(docs_path)
    if args.limit:
        docs = docs[: args.limit]

    total = 0
    with output_path.open("w", encoding="utf-8") as out:
        for doc in docs:
            text = _normalize_text(doc.get("text") or "")
            if not text:
                continue
            blocks = parse_paragraph_blocks(text)
            if not blocks:
                continue
            recs = emit_chunk_records(
                doc, blocks,
                target_words=args.target_words,
                overlap_words=args.overlap_words,
                min_words=args.min_words,
            )
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} chunks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
