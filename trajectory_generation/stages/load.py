"""Stage 0: Load chunks, secrets, entity index.

Outputs a JSON manifest with paths and metadata for downstream stages.
"""

import json
import logging
from pathlib import Path

from trajectory_generation.src.data_loading import (
    build_doc_lookup,
    filter_seed_secrets,
    load_chunks_local,
    load_secrets,
)

logger = logging.getLogger(__name__)


def run_load(
    chunks_local: Path,
    chunks_web: list[Path],
    curated_web_chunks: list[Path],
    secret_inventory: Path,
    entity_index_dir: Path,
    spacy_model: str,
    curated_web_docs: Path,
    task_id: str | None = None,
    company: str | None = None,
) -> dict:
    """Load all data. Returns manifest dict for serialization."""
    chunks = load_chunks_local(chunks_local)
    doc_lookup = build_doc_lookup(chunks)
    local_doc_ids = set(doc_lookup.keys())

    secrets = load_secrets(secret_inventory)
    eligible = filter_seed_secrets(
        secrets, doc_lookup, task_id=task_id, company=company
    )

    task_local_doc_ids: dict[str, set[str]] = {}
    for doc_id in local_doc_ids:
        doc = doc_lookup.get(doc_id)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_local_doc_ids.setdefault(tid, set()).add(doc_id)

    task_secrets: dict[str, list] = {}
    for s in eligible:
        doc = doc_lookup.get(s.doc_id)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_secrets.setdefault(tid, []).append(s)

    if task_id:
        task_local_doc_ids = {k: v for k, v in task_local_doc_ids.items() if k == task_id}
        task_secrets = {k: v for k, v in task_secrets.items() if k == task_id}
    elif company:
        keep = {
            doc.meta.get("task_id", "")
            for doc in doc_lookup.values()
            if company.lower() in (doc.meta.get("company_name") or "").lower()
        }
        task_local_doc_ids = {k: v for k, v in task_local_doc_ids.items() if k in keep}
        task_secrets = {k: v for k, v in task_secrets.items() if k in keep}

    manifest = {
        "chunks_local": str(chunks_local),
        "chunks_web": [str(p) for p in chunks_web],
        "curated_web_chunks": [str(p) for p in curated_web_chunks],
        "secret_inventory": str(secret_inventory),
        "entity_index_dir": str(entity_index_dir),
        "spacy_model": spacy_model,
        "curated_web_docs": str(curated_web_docs),
        "n_local_docs": len(local_doc_ids),
        "n_secrets_eligible": len(eligible),
        "n_tasks": len(task_local_doc_ids),
        "task_ids": sorted(task_local_doc_ids.keys()),
    }
    return {"manifest": manifest}


def load_state(path: Path) -> dict:
    """Load saved state from JSON."""
    with open(path) as f:
        return json.load(f)


def save_state(state: dict, path: Path) -> None:
    """Save manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
