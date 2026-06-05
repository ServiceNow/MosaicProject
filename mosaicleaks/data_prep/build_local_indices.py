"""Build per-task local retrieval indices for the retrieval helper."""

import argparse
import json
from pathlib import Path

import numpy as np

from mosaicleaks import config
from mosaicleaks.src.data_loading import build_doc_lookup, load_chunks_local
from mosaicleaks.src.retrieval.dense import OpenAICompatibleBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval-helper local indices from local chunks."
    )
    parser.add_argument(
        "--chunks-local",
        type=Path,
        default=config.CHUNKS_LOCAL,
        help="Input local chunks JSONL.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory that will contain one index subdirectory per task.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=config.DEFAULT_BASE_URL,
        help="OpenAI-compatible embedding endpoint.",
    )
    parser.add_argument(
        "--embedding-model",
        default=config.BROWSECOMP_EMBEDDING_MODEL,
        help="Embedding model name.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for the embedding endpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding request batch size.")
    parser.add_argument("--task", default=None, help="Optional task ID filter.")
    return parser.parse_args()


def _maybe_v1_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _group_docs_by_task(chunks_path: Path, task_id: str | None) -> dict[str, list[dict]]:
    docs = build_doc_lookup(load_chunks_local(chunks_path))
    grouped: dict[str, list[dict]] = {}
    for doc_id, doc in sorted(docs.items()):
        doc_task_id = str(doc.meta.get("task_id") or "")
        if not doc_task_id:
            continue
        if task_id is not None and doc_task_id != task_id:
            continue
        grouped.setdefault(doc_task_id, []).append(
            {
                "doc_id": doc_id,
                "content": doc.text,
                "content_preview": doc.text[:300],
                "metadata": {
                    **doc.meta,
                    "chunk_ids": doc.chunk_ids,
                },
            }
        )
    return grouped


def _write_task_index(task_dir: Path, docs: list[dict], embeddings: np.ndarray) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = [doc["doc_id"] for doc in docs]
    documents = {
        doc["doc_id"]: {
            "content": doc["content"],
            "content_preview": doc["content_preview"],
            "metadata": doc["metadata"],
        }
        for doc in docs
    }
    (task_dir / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (task_dir / "index.json").write_text(
        json.dumps({"doc_ids": doc_ids}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(task_dir / "embeddings.npy", embeddings.astype(np.float32, copy=False))


def main() -> None:
    args = _parse_args()
    grouped_docs = _group_docs_by_task(args.chunks_local, args.task)
    if not grouped_docs:
        raise ValueError(f"No task-local documents found in {args.chunks_local}")

    embedder = OpenAICompatibleBackend(
        model=args.embedding_model,
        api_key=args.api_key,
        base_url=_maybe_v1_url(args.embedding_base_url),
        batch_size=args.batch_size,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "chunks_local": str(args.chunks_local),
        "embedding_model": args.embedding_model,
        "tasks": {},
    }
    for task_id, docs in sorted(grouped_docs.items()):
        texts = [doc["content"] for doc in docs]
        embeddings = embedder.embed_texts(texts)
        task_dir = args.output_root / task_id
        _write_task_index(task_dir, docs, embeddings)
        embedding_dim = 0
        if embeddings.ndim == 2 and embeddings.shape[0] > 0:
            embedding_dim = int(embeddings.shape[1])
        manifest["tasks"][task_id] = {
            "documents": len(docs),
            "embedding_dim": embedding_dim,
        }

    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
