#!/usr/bin/env python3
"""HTTP service for MosaicLeaks retrieval helper."""

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from typing import Any

import anyio.to_thread
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from mosaicleaks import config
from mosaicleaks.src.retrieval.retrieval_helper_backend import (
    RetrievalHelperBackend,
    RetrievalHelperConfig,
)

logger = logging.getLogger(__name__)
_CONFIG_ENV_VAR = "MOSAICLEAKS_RETRIEVAL_HELPER_CONFIG_JSON"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    model: str | None = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class LocalSearchRequest(BaseModel):
    task_id: str
    query: str
    k: int = 10
    threshold: float = 0.0


class LocalSearchResponse(BaseModel):
    hits: list[dict[str, Any]]


class BrowseCompSearchRequest(BaseModel):
    query: str
    task_id: str | None = None
    k: int = 5
    max_chars: int | None = None


class BrowseCompSearchResponse(BaseModel):
    hits: list[dict[str, Any]]


class HelperAppLifecycle:
    def __init__(self, app: FastAPI, config_obj: RetrievalHelperConfig) -> None:
        self.app = app
        self.config_obj = config_obj

    async def __aenter__(self) -> None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = max(
            1,
            int(self.config_obj.threadpool_tokens),
        )
        logger.info("Loading retrieval helper backend...")
        self.app.state.backend = RetrievalHelperBackend(self.config_obj)
        logger.info("Retrieval helper backend ready")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class HelperLifespan:
    def __init__(self, config_obj: RetrievalHelperConfig) -> None:
        self.config_obj = config_obj

    def __call__(self, app: FastAPI) -> HelperAppLifecycle:
        return HelperAppLifecycle(app, self.config_obj)


def add_helper_arguments(
    argument_target: Any,
    *,
    default_host: str = "127.0.0.1",
    default_port: int = 8012,
    default_embedding_base_url: str | None = "http://127.0.0.1:8000",
) -> None:
    argument_target.add_argument("--host", default=default_host, help="Host to bind to.")
    argument_target.add_argument("--port", type=int, default=default_port, help="Port to bind to.")
    argument_target.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of helper worker processes to run.",
    )
    argument_target.add_argument(
        "--threadpool-tokens",
        type=int,
        default=16,
        help="Maximum concurrent blocking request handlers per helper worker.",
    )
    argument_target.add_argument(
        "--local-index-root",
        default=str(Path.home() / "mosaicleaks" / "local_indices"),
        help="Root directory containing prebuilt per-task local indices.",
    )
    argument_target.add_argument(
        "--local-index-mmap-mode",
        default="r",
        help="np.load mmap_mode for local embedding matrices.",
    )
    argument_target.add_argument(
        "--embedding-base-url",
        default=default_embedding_base_url,
        help="Base URL of the colocated OpenAI-compatible embedding endpoint.",
    )
    argument_target.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model served by the colocated vLLM endpoint.",
    )
    argument_target.add_argument(
        "--embedding-timeout-s",
        type=float,
        default=60.0,
        help="Timeout for helper->embedding-server requests.",
    )
    argument_target.add_argument(
        "--embedding-batch-size",
        type=int,
        default=256,
        help="Batch size used for /embed requests against the embedding server.",
    )
    argument_target.add_argument(
        "--browsecomp-index-glob",
        default=config.BROWSECOMP_EMBEDDING_INDEX_GLOB,
        help="Path glob for BrowseComp index shards.",
    )
    argument_target.add_argument(
        "--browsecomp-corpus",
        default=config.BROWSECOMP_DATASET_NAME,
        help="BrowseComp dataset name or local corpus path.",
    )
    argument_target.add_argument(
        "--browsecomp-index-device",
        default="cpu",
        help="FAISS device for BrowseComp dense search.",
    )
    argument_target.add_argument(
        "--browsecomp-bm25-recall-k",
        type=int,
        default=config.BROWSECOMP_BM25_RECALL_K,
        help="BM25 recall fanout before reranking.",
    )
    argument_target.add_argument(
        "--browsecomp-dense-recall-k",
        type=int,
        default=config.BROWSECOMP_DENSE_RECALL_K,
        help="Dense recall fanout before reranking.",
    )
    argument_target.add_argument(
        "--browsecomp-rank-mode",
        choices=["hybrid", "rrf", "bm25"],
        default="rrf",
        help="How to rank BrowseComp BM25/dense candidates: dense rerank hybrid, rank-fusion RRF, or BM25 only.",
    )
    argument_target.add_argument(
        "--browsecomp-rrf-k",
        type=int,
        default=20,
        help="RRF denominator constant for --browsecomp-rank-mode=rrf.",
    )
    argument_target.add_argument(
        "--local-rank-mode",
        choices=["dense", "rrf", "bm25"],
        default="rrf",
        help="How to rank task-local documents: dense cosine, dense+BM25 RRF, or BM25 only.",
    )
    argument_target.add_argument(
        "--local-bm25-recall-k",
        type=int,
        default=100,
        help="Local BM25 recall fanout for --local-rank-mode=rrf.",
    )
    argument_target.add_argument(
        "--local-dense-recall-k",
        type=int,
        default=100,
        help="Local dense recall fanout for --local-rank-mode=rrf.",
    )
    argument_target.add_argument(
        "--local-rrf-k",
        type=int,
        default=20,
        help="RRF denominator constant for --local-rank-mode=rrf.",
    )
    argument_target.add_argument(
        "--browsecomp-max-chars",
        type=int,
        default=8000,
        help="Default text length returned per BrowseComp hit when the request does not specify max_chars.",
    )
    argument_target.add_argument(
        "--browsecomp-exclude-doc-ids-file",
        default="",
        help="Optional newline-delimited BrowseComp parent doc IDs to exclude from web retrieval.",
    )
    argument_target.add_argument(
        "--browsecomp-max-doc-windows",
        type=int,
        default=10,
        help="Live safety filter: drop BrowseComp parents longer than this many 16k-character reader windows. Use 0 to disable.",
    )
    argument_target.add_argument(
        "--browsecomp-reader-window-chars",
        type=int,
        default=16000,
        help="Reader window size used to convert --browsecomp-max-doc-windows into an exact character cap.",
    )
    argument_target.add_argument(
        "--browsecomp-reader-window-overlap-chars",
        type=int,
        default=1000,
        help="Reader window overlap used to convert --browsecomp-max-doc-windows into an exact character cap.",
    )
    argument_target.add_argument(
        "--curated-web-docs",
        default=str(config.CURATED_WEB_DOCS),
        help="Direct-fetch curated web metadata JSONL.",
    )
    argument_target.add_argument(
        "--curated-web-chunks",
        action="append",
        default=None,
        help="Curated web chunk JSONL. Pass multiple times to merge files.",
    )
    argument_target.add_argument("--verbose", action="store_true", help="Enable debug logging.")


def build_arg_parser(
    *,
    default_host: str = "127.0.0.1",
    default_port: int = 8012,
    default_embedding_base_url: str | None = "http://127.0.0.1:8000",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reusable MosaicLeaks retrieval helper service.")
    add_helper_arguments(
        parser,
        default_host=default_host,
        default_port=default_port,
        default_embedding_base_url=default_embedding_base_url,
    )
    return parser


def _parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RetrievalHelperConfig:
    curated_chunks = args.curated_web_chunks or [str(config.CURATED_WEB_CHUNKS)]
    return RetrievalHelperConfig(
        local_index_root=args.local_index_root,
        local_index_mmap_mode=args.local_index_mmap_mode,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        embedding_timeout_s=float(args.embedding_timeout_s),
        embedding_batch_size=int(args.embedding_batch_size),
        browsecomp_index_glob=args.browsecomp_index_glob,
        browsecomp_corpus=args.browsecomp_corpus,
        browsecomp_index_device=args.browsecomp_index_device,
        browsecomp_bm25_recall_k=int(args.browsecomp_bm25_recall_k),
        browsecomp_dense_recall_k=int(args.browsecomp_dense_recall_k),
        browsecomp_rank_mode=args.browsecomp_rank_mode,
        browsecomp_rrf_k=int(args.browsecomp_rrf_k),
        local_rank_mode=args.local_rank_mode,
        local_bm25_recall_k=int(args.local_bm25_recall_k),
        local_dense_recall_k=int(args.local_dense_recall_k),
        local_rrf_k=int(args.local_rrf_k),
        browsecomp_max_chars=int(args.browsecomp_max_chars),
        curated_web_docs=args.curated_web_docs,
        curated_web_chunks=curated_chunks,
        browsecomp_exclude_doc_ids_file=args.browsecomp_exclude_doc_ids_file,
        browsecomp_max_doc_windows=int(args.browsecomp_max_doc_windows),
        browsecomp_reader_window_chars=int(args.browsecomp_reader_window_chars),
        browsecomp_reader_window_overlap_chars=int(args.browsecomp_reader_window_overlap_chars),
        threadpool_tokens=int(args.threadpool_tokens),
    )


def create_app_from_env() -> FastAPI:
    payload = os.environ.get(_CONFIG_ENV_VAR)
    if not payload:
        raise RuntimeError(f"{_CONFIG_ENV_VAR} is not set")
    return create_app(RetrievalHelperConfig(**json.loads(payload)))


def create_app(config_obj: RetrievalHelperConfig) -> FastAPI:
    app = FastAPI(
        title="mosaicleaks-retrieval-helper",
        version="0.1.0",
        lifespan=HelperLifespan(config_obj),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def backend() -> RetrievalHelperBackend:
        return app.state.backend

    @app.get("/health")
    def health() -> dict[str, Any]:
        helper = backend()
        started = helper.metrics.start("/health")
        ok = True
        try:
            return helper.health()
        except Exception as exc:
            ok = False
            logger.exception("health failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            helper.metrics.finish("/health", started, ok=ok)

    @app.post("/embed", response_model=EmbedResponse)
    def embed(request: EmbedRequest) -> EmbedResponse:
        helper = backend()
        started = helper.metrics.start("/embed")
        ok = True
        try:
            return EmbedResponse(
                embeddings=helper.embed(request.texts, model_name=request.model)
            )
        except ValueError as exc:
            ok = False
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            ok = False
            logger.exception("embed failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            helper.metrics.finish("/embed", started, ok=ok)

    @app.post("/local/search", response_model=LocalSearchResponse)
    def search_local(request: LocalSearchRequest) -> LocalSearchResponse:
        helper = backend()
        started = helper.metrics.start("/local/search")
        ok = True
        try:
            return LocalSearchResponse(
                hits=helper.search_local(
                    task_id=request.task_id,
                    query=request.query,
                    k=request.k,
                    threshold=request.threshold,
                )
            )
        except FileNotFoundError as exc:
            ok = False
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            ok = False
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            ok = False
            logger.exception("local search failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            helper.metrics.finish("/local/search", started, ok=ok)

    @app.post("/browsecomp/search", response_model=BrowseCompSearchResponse)
    def search_browsecomp(request: BrowseCompSearchRequest) -> BrowseCompSearchResponse:
        helper = backend()
        started = helper.metrics.start("/browsecomp/search")
        ok = True
        try:
            return BrowseCompSearchResponse(
                hits=helper.search_browsecomp(
                    query=request.query,
                    task_id=request.task_id,
                    k=request.k,
                    max_chars=request.max_chars,
                )
            )
        except ValueError as exc:
            ok = False
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            ok = False
            logger.exception("browsecomp search failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            helper.metrics.finish("/browsecomp/search", started, ok=ok)

    return app


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_obj = config_from_args(args)
    if args.workers > 1:
        os.environ[_CONFIG_ENV_VAR] = json.dumps(asdict(config_obj))
        uvicorn.run(
            "mosaicleaks.src.retrieval.retrieval_helper_server:create_app_from_env",
            host=args.host,
            port=args.port,
            log_level="debug" if args.verbose else "info",
            access_log=False,
            workers=args.workers,
            factory=True,
        )
    else:
        app = create_app(config_obj)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="debug" if args.verbose else "info",
            access_log=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
