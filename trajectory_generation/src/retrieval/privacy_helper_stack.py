#!/usr/bin/env python3
"""Run the privacy helper and a colocated vLLM embedding server."""

import argparse
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from trajectory_generation.src.retrieval.privacy_helper_server import add_helper_arguments

logger = logging.getLogger(__name__)


def repo_id_to_cache_dir(model_name: str) -> str:
    return f"models--{model_name.replace('/', '--')}"


def resolve_cached_snapshot(model_name: str, cache_root: str | None) -> str | None:
    if cache_root is None or Path(model_name).exists():
        return None

    repo_root = Path(cache_root) / repo_id_to_cache_dir(model_name)
    snapshots_dir = repo_root / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    refs_main = repo_root / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot = snapshots_dir / revision
        if snapshot.is_dir():
            return str(snapshot)

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if snapshots:
        return str(snapshots[0])
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a colocated vLLM embedding server and the privacy helper API.",
    )
    stack_args = parser.add_argument_group("stack options")
    helper_args = parser.add_argument_group("helper options")

    stack_args.add_argument(
        "--embed-host",
        default="127.0.0.1",
        help="Host for the colocated vLLM embedding server.",
    )
    stack_args.add_argument(
        "--embed-port",
        type=int,
        default=int(os.getenv("PRIVACY_HELPER_EMBED_PORT", "8000")),
        help="Port for the colocated vLLM embedding server.",
    )
    stack_args.add_argument(
        "--served-model-name",
        default=os.getenv("PRIVACY_HELPER_SERVED_MODEL_NAME", "qwen3-embedding-4b"),
        help="Optional alias exposed by the colocated vLLM embedding server.",
    )
    stack_args.add_argument(
        "--embedding-model-cache-root",
        default=os.getenv("PRIVACY_HELPER_MODEL_CACHE_ROOT", "/transformers_cache/hub"),
        help="Read-only Hugging Face hub cache used to resolve a local embedding model snapshot.",
    )
    stack_args.add_argument(
        "--vllm-dtype",
        default="bfloat16",
        help="dtype passed to the vLLM embedding server.",
    )
    stack_args.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=32768,
        help="Maximum model length for the colocated vLLM embedding server.",
    )
    stack_args.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=160,
        help="Maximum concurrent sequences for the colocated vLLM embedding server.",
    )
    stack_args.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization target for the colocated vLLM embedding server.",
    )
    add_helper_arguments(
        helper_args,
        default_host="0.0.0.0",
        default_port=8012,
        default_embedding_base_url=None,
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.embedding_base_url is None:
        args.embedding_base_url = f"http://{args.embed_host}:{args.embed_port}"
    if args.port == args.embed_port and args.host == args.embed_host:
        parser.error("helper port and embed port must be different")
    return args


def wait_for_http_ok(url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}")


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    model_source = resolve_cached_snapshot(args.embedding_model, args.embedding_model_cache_root)
    if model_source is None:
        model_source = args.embedding_model
    else:
        logger.info("Using cached embedding model snapshot at %s", model_source)

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.embed_host,
        "--port",
        str(args.embed_port),
        "--model",
        model_source,
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--trust-remote-code",
        "--dtype",
        args.vllm_dtype,
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--max-model-len",
        str(args.vllm_max_model_len),
        "--max-num-seqs",
        str(args.vllm_max_num_seqs),
        "--served-model-name",
        args.embedding_model,
    ]
    if args.served_model_name and args.served_model_name != args.embedding_model:
        command.append(args.served_model_name)
    command.extend(
        [
            "--disable-frontend-multiprocessing",
            "--no-enable-log-requests",
        ]
    )
    return command


def build_helper_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "trajectory_generation.src.retrieval.privacy_helper_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--local-index-root",
        args.local_index_root,
        "--workers",
        str(args.workers),
        "--threadpool-tokens",
        str(args.threadpool_tokens),
        "--local-index-mmap-mode",
        args.local_index_mmap_mode,
        "--embedding-base-url",
        args.embedding_base_url,
        "--embedding-model",
        args.embedding_model,
        "--embedding-timeout-s",
        str(args.embedding_timeout_s),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        "--browsecomp-index-glob",
        args.browsecomp_index_glob,
        "--browsecomp-corpus",
        args.browsecomp_corpus,
        "--browsecomp-index-device",
        args.browsecomp_index_device,
        "--browsecomp-bm25-recall-k",
        str(args.browsecomp_bm25_recall_k),
        "--browsecomp-dense-recall-k",
        str(args.browsecomp_dense_recall_k),
        "--browsecomp-max-chars",
        str(args.browsecomp_max_chars),
        "--browsecomp-exclude-doc-ids-file",
        args.browsecomp_exclude_doc_ids_file,
        "--browsecomp-max-doc-windows",
        str(args.browsecomp_max_doc_windows),
        "--browsecomp-reader-window-chars",
        str(args.browsecomp_reader_window_chars),
        "--browsecomp-reader-window-overlap-chars",
        str(args.browsecomp_reader_window_overlap_chars),
        "--curated-web-docs",
        args.curated_web_docs,
    ]
    if args.curated_web_chunks:
        for chunk_path in args.curated_web_chunks:
            command.extend(["--curated-web-chunks", chunk_path])
    if args.verbose:
        command.append("--verbose")
    return command


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()


def wait_for_exit(process: subprocess.Popen[bytes] | None, timeout_seconds: float = 10.0) -> None:
    if process is None:
        return
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    vllm_process: subprocess.Popen[bytes] | None = None
    helper_process: subprocess.Popen[bytes] | None = None

    def handle_signal(signum, _frame) -> None:
        logger.info("Received signal %s, shutting down helper stack", signum)
        stop_process(helper_process)
        stop_process(vllm_process)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        vllm_command = build_vllm_command(args)
        logger.info("Starting colocated vLLM embedding server")
        vllm_process = subprocess.Popen(vllm_command)
        wait_for_http_ok(f"http://{args.embed_host}:{args.embed_port}/health", 300)

        helper_command = build_helper_command(args)
        logger.info("Starting privacy helper API")
        helper_process = subprocess.Popen(helper_command)

        while True:
            if vllm_process.poll() is not None:
                return int(vllm_process.returncode or 0)
            if helper_process.poll() is not None:
                return int(helper_process.returncode or 0)
            time.sleep(1)
    finally:
        stop_process(helper_process)
        stop_process(vllm_process)
        wait_for_exit(helper_process)
        wait_for_exit(vllm_process)


if __name__ == "__main__":
    raise SystemExit(main())
