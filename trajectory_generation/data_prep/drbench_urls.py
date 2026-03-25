"""Extract DrBench seed URLs from task context.json files."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

WEB_POOL = "drbench_urls"
DOCID_PREFIX = f"{WEB_POOL}_"
DOC_ID_PREFIX = f"web/{WEB_POOL}/"


def docid_for_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{DOCID_PREFIX}{digest}"


def doc_id_for_docid(docid: str) -> str:
    return f"{DOC_ID_PREFIX}{docid}"


def doc_id_for_url(url: str) -> str:
    return doc_id_for_docid(docid_for_url(url))


def build_url_to_task_ids(tasks_root: Path) -> dict[str, list[str]]:
    """Map seed URL -> list of task IDs that reference it.

    Reads each task's context.json; expects url.url (or url as string).
    Skips SANITY0. Returns mapping for all DRxxxx tasks.
    """
    mapping: dict[str, list[str]] = defaultdict(list)
    for entry in sorted(tasks_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "SANITY0":
            continue
        context_path = entry / "context.json"
        if not context_path.exists():
            continue
        try:
            ctx = json.loads(context_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        url_block = ctx.get("url")
        if isinstance(url_block, dict):
            url = (url_block.get("url") or "").strip()
        elif isinstance(url_block, str):
            url = url_block.strip()
        else:
            url = ""
        if not url:
            continue
        mapping[url].append(entry.name)

    return {u: sorted(set(tids)) for u, tids in mapping.items()}


def iter_urls_from_tasks(tasks_root: Path) -> list[dict[str, Any]]:
    """Yield URL records with task_ids, industry, domain, etc. from task context.json."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for entry in sorted(tasks_root.iterdir()):
        if not entry.is_dir() or entry.name == "SANITY0":
            continue
        context_path = entry / "context.json"
        if not context_path.exists():
            continue
        try:
            ctx = json.loads(context_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        url_block = ctx.get("url")
        if isinstance(url_block, dict):
            url = (url_block.get("url") or "").strip()
            industry = url_block.get("industry", "")
            domain = url_block.get("domain", "")
            seed_date = url_block.get("date", "")
            title = url_block.get("title", "")
        else:
            url = (url_block or "").strip() if isinstance(url_block, str) else ""
            industry = domain = seed_date = title = ""
        if not url or url in seen:
            continue
        seen.add(url)
        mapping = build_url_to_task_ids(tasks_root)
        task_ids = mapping.get(url, [entry.name])
        out.append({
            "url": url,
            "industry": industry,
            "domain": domain,
            "seed_date": seed_date,
            "title": title,
            "task_ids": task_ids,
        })
    return out
