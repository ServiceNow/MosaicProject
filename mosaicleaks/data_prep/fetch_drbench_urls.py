"""Fetch and refresh the curated DrBench context-URL corpus.

This script treats task ``context.json`` as the source of truth for curated web
documents. It refreshes URL metadata on every run, reuses previously fetched
successful pages when possible, and only hits the network for missing or failed
entries unless ``--force-refetch`` is set.
"""

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mosaicleaks import config
from mosaicleaks.data_prep.drbench_urls import (
    doc_id_for_url,
    docid_for_url,
    iter_urls_from_tasks,
)
from mosaicleaks.src.jsonl import iter_jsonl_objects

logger = logging.getLogger(__name__)

DEFAULT_TASKS_ROOT = config.TASKS_DIR
DEFAULT_OUTPUT = config.CURATED_WEB_DOCS
DEFAULT_CORPUS_DIR = config.DATA_DIR / "curated_web_corpus"
DEFAULT_DOWNLOADS_DIR = DEFAULT_CORPUS_DIR / "downloads"
DEFAULT_EXTRACTED_DIR = DEFAULT_CORPUS_DIR / "extracted"
DEFAULT_CHUNK_OUTPUT = config.CURATED_WEB_CHUNKS

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class FetchResult:
    """Network fetch output before it is converted into a JSONL row."""

    http_status: int | None
    content_type: str | None
    raw_suffix: str
    raw_bytes: bytes | None
    text: str
    title: str
    error: str | None


def _is_reusable_success(rec: dict[str, Any] | None) -> bool:
    """True when a stored record is a real fetched page we should keep reusing."""
    if not rec:
        return False
    if rec.get("extractor") == "insights_fallback":
        return False
    text = (rec.get("text") or "").strip()
    err = rec.get("error")
    status = rec.get("http_status")
    return bool(text) and not err and isinstance(status, int) and 200 <= status < 300


def _json_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_line(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    return text


def _extract_title(soup: BeautifulSoup) -> str:
    for attr_name, attr_value in (
        ("property", "og:title"),
        ("name", "twitter:title"),
    ):
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            title = _clean_line(tag["content"])
            if title:
                return title
    if soup.title and soup.title.get_text(strip=True):
        return _clean_line(soup.title.get_text(" ", strip=True))
    h1 = soup.find("h1")
    if h1:
        title = _clean_line(h1.get_text(" ", strip=True))
        if title:
            return title
    return ""


def _score_root(node: Any) -> int:
    score = 0
    for tag in node.find_all(["p", "li", "h1", "h2", "h3"]):
        text = _clean_line(tag.get_text(" ", strip=True))
        if len(text) >= 25:
            score += len(text)
    return score


def _select_content_root(soup: BeautifulSoup) -> Any:
    candidates: list[Any] = []
    seen: set[int] = set()
    selectors = [
        "article",
        "main",
        "[role='main']",
        "#main",
        "#content",
        "#main-content",
        ".main-content",
        ".article",
        ".article-body",
        ".entry-content",
        ".post-content",
        ".story-body",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            ident = id(node)
            if ident not in seen:
                seen.add(ident)
                candidates.append(node)
    body = soup.body or soup
    if id(body) not in seen:
        candidates.append(body)
    if not candidates:
        return soup
    return max(candidates, key=_score_root)


def _extract_structured_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "form",
        "button",
        "input",
        "textarea",
        "footer",
        "nav",
        "aside",
    ]):
        tag.decompose()

    title = _extract_title(soup)
    root = _select_content_root(soup)

    lines: list[str] = []
    seen_text: set[str] = set()

    if title:
        lines.append(f"# {title}")

    for tag in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = _clean_line(tag.get_text(" ", strip=True))
        if not text:
            continue
        if tag.name in {"p", "li"} and len(text) < 25:
            continue
        dedupe_key = text.casefold()
        if dedupe_key in seen_text:
            continue
        seen_text.add(dedupe_key)
        if tag.name == "h1":
            line = f"# {text}"
        elif tag.name == "h2":
            line = f"## {text}"
        elif tag.name == "h3":
            line = f"### {text}"
        elif tag.name == "li":
            line = f"- {text}"
        else:
            line = text
        lines.append(line)

    text = "\n\n".join(lines).strip()
    if text:
        return title, text

    fallback = _clean_line(root.get_text(" ", strip=True))
    if title and fallback and not fallback.startswith(title):
        fallback = f"# {title}\n\n{fallback}"
    return title, fallback


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": _UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    })
    return session


def _guess_raw_suffix(content_type: str | None) -> str:
    if not content_type:
        return ".html"
    ctype = content_type.lower()
    if "html" in ctype or "xml" in ctype:
        return ".html"
    if "json" in ctype:
        return ".json"
    if ctype.startswith("text/"):
        return ".txt"
    return ".bin"


def fetch_url(session: requests.Session, url: str, timeout: float) -> FetchResult:
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        return FetchResult(
            http_status=None,
            content_type=None,
            raw_suffix=".html",
            raw_bytes=None,
            text="",
            title="",
            error=f"{type(exc).__name__}: {exc}",
        )

    content_type = resp.headers.get("content-type", "")
    raw_suffix = _guess_raw_suffix(content_type)
    raw_bytes = resp.content
    status = resp.status_code

    if not resp.ok:
        return FetchResult(
            http_status=status,
            content_type=content_type or None,
            raw_suffix=raw_suffix,
            raw_bytes=raw_bytes,
            text="",
            title="",
            error=f"HTTP {status}",
        )

    text = ""
    title = ""
    try:
        if "html" in content_type.lower() or resp.text.lstrip().startswith("<"):
            title, text = _extract_structured_text(resp.text)
        else:
            text = resp.text
    except Exception as exc:  # pragma: no cover - defensive
        # HTML extraction is best effort; keep the raw response text so the
        # URL refresh still records a usable artifact for manual review.
        logger.warning("Extraction failed for %s: %s", url, exc)
        text = resp.text

    return FetchResult(
        http_status=status,
        content_type=content_type or None,
        raw_suffix=raw_suffix,
        raw_bytes=raw_bytes,
        text=text.strip(),
        title=title.strip(),
        error=None,
    )


def _load_best_existing_records(docs_path: Path) -> dict[str, dict[str, Any]]:
    if not docs_path.exists():
        return {}

    best: dict[str, dict[str, Any]] = {}
    for rec in iter_jsonl_objects(docs_path):
        url = rec.get("url")
        if not isinstance(url, str) or not url:
            continue
        prev = best.get(url)
        is_success = _is_reusable_success(rec)
        prev_success = _is_reusable_success(prev)
        if prev is None or is_success or not prev_success:
            best[url] = rec
    return best


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _hydrate_from_existing(
    live: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    rec = dict(existing)
    rec.update({
        "doc_id": doc_id_for_url(live["url"]),
        "docid": docid_for_url(live["url"]),
        "url": live["url"],
        "industry": live.get("industry", ""),
        "domain": live.get("domain", ""),
        "seed_date": live.get("seed_date", ""),
        "task_ids": live.get("task_ids", []),
    })
    if live.get("title"):
        rec["title"] = live["title"]
    return rec


def refresh_corpus(
    tasks_root: Path,
    docs_path: Path,
    downloads_dir: Path,
    extracted_dir: Path,
    force_refetch: bool = False,
    limit: int | None = None,
    timeout: float = 30.0,
    sleep_sec: float = 0.0,
) -> list[dict[str, Any]]:
    live_records = iter_urls_from_tasks(tasks_root)
    if limit is not None:
        live_records = live_records[:limit]
    existing_by_url = _load_best_existing_records(docs_path)

    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    rows: list[dict[str, Any]] = []
    fetched = reused = failed = 0

    for live in live_records:
        url = live["url"]
        docid = docid_for_url(url)
        doc_id = doc_id_for_url(url)
        raw_path = downloads_dir / f"{docid}.html"
        extracted_path = extracted_dir / f"{docid}.txt"
        existing = existing_by_url.get(url)
        existing_success = _is_reusable_success(existing)

        if existing_success and not force_refetch:
            rec = _hydrate_from_existing(live, existing)
            rows.append(rec)
            reused += 1
            continue

        result = fetch_url(session, url, timeout=timeout)
        used_existing_after_fail = False
        if result.raw_bytes is not None:
            raw_path = raw_path.with_suffix(result.raw_suffix)
            raw_path.write_bytes(result.raw_bytes)

        text = result.text
        title = live.get("title") or result.title

        if text:
            extracted_path.write_text(text, encoding="utf-8")
            fetched += 1
        elif existing_success:
            rec = _hydrate_from_existing(live, existing)
            rows.append(rec)
            reused += 1
            used_existing_after_fail = True
            logger.warning("Fetch failed for %s; kept previous successful content", url)
        else:
            failed += 1

        if not used_existing_after_fail:
            raw_rel = _relative_to_repo(raw_path) if raw_path.exists() else None
            extracted_rel = _relative_to_repo(extracted_path) if extracted_path.exists() else None
            rows.append({
                "doc_id": doc_id,
                "docid": docid,
                "url": url,
                "industry": live.get("industry", ""),
                "domain": live.get("domain", ""),
                "seed_date": live.get("seed_date", ""),
                "task_ids": live.get("task_ids", []),
                "title": title or "",
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "http_status": result.http_status,
                "content_type": result.content_type,
                "raw_path": raw_rel,
                "extracted_path": extracted_rel,
                "error": result.error,
                "text": text,
            })

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    logger.info(
        "DRBench URL refresh complete: %d total, %d reused, %d fetched, %d failed",
        len(rows),
        reused,
        fetched,
        failed,
    )
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and refresh current DrBench context URLs.")
    p.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    p.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    p.add_argument("--chunk-output", type=Path, default=None, help="Optional chunks JSONL path to regenerate after fetch.")
    p.add_argument("--force-refetch", action="store_true", help="Refetch even previously successful URLs.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--sleep-sec", type=float, default=0.0, help="Optional pause between direct fetches.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rows = refresh_corpus(
        tasks_root=args.tasks_root,
        docs_path=args.output,
        downloads_dir=args.downloads_dir,
        extracted_dir=args.extracted_dir,
        force_refetch=args.force_refetch,
        limit=args.limit,
        timeout=args.timeout,
        sleep_sec=args.sleep_sec,
    )
    _json_dump(args.output, rows)
    logger.info("Wrote %d URL docs to %s", len(rows), args.output)

    if args.chunk_output is not None:
        from mosaicleaks.data_prep.chunk_web_drbench import (
            emit_chunk_records,
            load_docs_jsonl,
            parse_paragraph_blocks,
        )

        docs = load_docs_jsonl(args.output)
        total = 0
        args.chunk_output.parent.mkdir(parents=True, exist_ok=True)
        with args.chunk_output.open("w", encoding="utf-8") as out:
            for doc in docs:
                blocks = parse_paragraph_blocks(doc.get("text") or "")
                if not blocks:
                    continue
                for rec in emit_chunk_records(doc, blocks):
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
        logger.info("Wrote %d chunks to %s", total, args.chunk_output)

    return 0


if __name__ == "__main__":
    main()
