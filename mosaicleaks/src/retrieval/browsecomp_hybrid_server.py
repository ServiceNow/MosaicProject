#!/usr/bin/env python3
"""Small HTTP service for the web retriever.

This lets you load the heavier web retrieval stack once and reuse it across
multiple generation runs.
"""

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mosaicleaks import config
from mosaicleaks.src.retrieval.browsecomp import BrowseCompHybridSearcher
from mosaicleaks.src.retrieval.dense import OpenAICompatibleBackend

logger = logging.getLogger(__name__)


class _Handler(BaseHTTPRequestHandler):
    backend: BrowseCompHybridSearcher | None = None

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        backend = self.backend
        self._write_json(HTTPStatus.OK, {
            "ok": True,
            "size": backend.size if backend is not None else 0,
            "service": "browsecomp_hybrid",
        })

    def do_POST(self) -> None:
        if self.path != "/search":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        backend = self.backend
        if backend is None:
            self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "backend unavailable"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            query = str(data.get("query", "")).strip()
            k = int(data.get("k", 10))
            mode = str(data.get("mode", "bm25"))
            allowed_doc_ids = data.get("doc_ids")
            hits = backend.search(query, k=max(k, 200 if allowed_doc_ids else k), mode=mode)
            if allowed_doc_ids:
                allowed = set(str(x) for x in allowed_doc_ids)
                hits = [h for h in hits if h.doc_id in allowed]
            hits = hits[:k]
            self._write_json(HTTPStatus.OK, {
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "doc_id": h.doc_id,
                        "score": h.score,
                        "text": h.text,
                        "meta": h.meta,
                    }
                    for h in hits
                ]
            })
        except Exception as exc:
            # HTTP boundary: log backend failures and return a 500 response
            # rather than crashing the long-lived retrieval process.
            logger.exception("search failed")
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> int:
    p = argparse.ArgumentParser(description="Run the reusable web retrieval service.")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    p.add_argument("--port", type=int, default=8011, help="Port to bind to.")
    p.add_argument("--index-glob", default=config.BROWSECOMP_EMBEDDING_INDEX_GLOB, help="Path glob for embedding index shards.")
    p.add_argument("--embedding-base-url", default=config.BROWSECOMP_EMBEDDING_BASE_URL, help="OpenAI-compatible embedding endpoint.")
    p.add_argument("--model-name", default=config.BROWSECOMP_EMBEDDING_MODEL, help="Embedding model name.")
    p.add_argument("--dataset-name", default=config.BROWSECOMP_DATASET_NAME, help="BrowseComp dataset name or local path.")
    p.add_argument("--bm25-recall-k", type=int, default=config.BROWSECOMP_BM25_RECALL_K, help="How many BM25 candidates to pull before reranking.")
    p.add_argument("--dense-recall-k", type=int, default=config.BROWSECOMP_DENSE_RECALL_K, help="How many dense candidates to pull before reranking.")
    p.add_argument("--max-chars", type=int, default=8000, help="Maximum text length returned per document.")
    p.add_argument("--verbose", action="store_true", help="Print request and startup logs.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading web retriever...")
    embedder = OpenAICompatibleBackend(
        model=args.model_name,
        base_url=args.embedding_base_url,
    )
    backend = BrowseCompHybridSearcher(
        index_glob=args.index_glob,
        dataset_name=args.dataset_name,
        embedder=embedder,
        bm25_recall_k=args.bm25_recall_k,
        dense_recall_k=args.dense_recall_k,
        max_chars=args.max_chars,
    )
    _Handler.backend = backend
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info("Web retriever listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
