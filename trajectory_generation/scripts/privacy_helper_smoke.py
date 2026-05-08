#!/usr/bin/env python3
"""End-to-end smoke test for the privacy helper HTTP service."""

import json
import pickle
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


class EmbeddingHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _write_json(self, status: int, payload: dict) -> None:
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
        self._write_json(HTTPStatus.OK, {"ok": True})

    def do_POST(self) -> None:
        if self.path != "/v1/embeddings":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        texts = payload.get("input") or []
        if isinstance(texts, str):
            texts = [texts]

        data = []
        for idx, text in enumerate(texts):
            vector = embed_text(str(text))
            data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": vector,
                }
            )

        self._write_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": data,
                "model": payload.get("model", "dummy-embedding-model"),
            },
        )


def embed_text(text: str) -> list[float]:
    lowered = text.lower()
    alpha = 1.0 if "alpha" in lowered else 0.0
    beta = 1.0 if "beta" in lowered else 0.0
    gamma = 1.0 if "gamma" in lowered else 0.0
    vector = np.asarray([alpha, beta, gamma], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return vector.tolist()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_http_ok(url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {url}")


def http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def build_local_index(index_root: Path, task_id: str) -> None:
    task_dir = index_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    documents = {
        "local-1": {
            "content": "alpha private document",
            "content_preview": "alpha private document",
            "metadata": {"file_name": "alpha.txt"},
        },
        "local-2": {
            "content": "beta private document",
            "content_preview": "beta private document",
            "metadata": {"file_name": "beta.txt"},
        },
    }
    embeddings = np.asarray(
        [
            embed_text(documents["local-1"]["content"]),
            embed_text(documents["local-2"]["content"]),
        ],
        dtype=np.float32,
    )

    (task_dir / "documents.json").write_text(json.dumps(documents), encoding="utf-8")
    (task_dir / "index.json").write_text(json.dumps({"doc_ids": ["local-1", "local-2"]}), encoding="utf-8")
    np.save(task_dir / "embeddings.npy", embeddings)


def build_browsecomp_corpus(root: Path) -> tuple[Path, Path]:
    shard_path = root / "corpus.shard00000_of_00001.pkl"
    corpus_path = root / "browsecomp_corpus.jsonl"

    lookup = ["1", "2"]
    embeddings = np.asarray(
        [
            embed_text("alpha web document"),
            embed_text("beta web document"),
        ],
        dtype=np.float32,
    )
    with shard_path.open("wb") as handle:
        pickle.dump((embeddings, lookup), handle)

    corpus_path.write_text(
        json.dumps({"docid": "1", "url": "https://example.com/alpha", "text": "alpha web document"}) + "\n"
        + json.dumps({"docid": "2", "url": "https://example.com/beta", "text": "beta web document"}) + "\n",
        encoding="utf-8",
    )
    return shard_path, corpus_path


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        local_index_root = root / "local_indices"
        build_local_index(local_index_root, "TASK001")
        shard_path, corpus_path = build_browsecomp_corpus(root)

        embed_port = free_port()
        helper_port = free_port()

        embedding_server = ThreadingHTTPServer(("127.0.0.1", embed_port), EmbeddingHandler)
        embedding_thread = threading.Thread(target=embedding_server.serve_forever, daemon=True)
        embedding_thread.start()
        wait_for_http_ok(f"http://127.0.0.1:{embed_port}/health", 5)

        helper_command = [
            sys.executable,
            "-m",
            "trajectory_generation.src.retrieval.privacy_helper_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(helper_port),
            "--local-index-root",
            str(local_index_root),
            "--embedding-base-url",
            f"http://127.0.0.1:{embed_port}",
            "--embedding-model",
            "Qwen/Qwen3-Embedding-4B",
            "--browsecomp-index-glob",
            str(shard_path),
            "--browsecomp-corpus",
            str(corpus_path),
            "--browsecomp-index-device",
            "cpu",
            "--browsecomp-bm25-recall-k",
            "2",
            "--browsecomp-dense-recall-k",
            "2",
            "--browsecomp-max-chars",
            "200",
        ]
        helper_process = subprocess.Popen(helper_command)

        try:
            wait_for_http_ok(f"http://127.0.0.1:{helper_port}/health", 30)

            health = http_json("GET", f"http://127.0.0.1:{helper_port}/health")
            embed = http_json(
                "POST",
                f"http://127.0.0.1:{helper_port}/embed",
                {"texts": ["alpha private document"]},
            )
            local_search = http_json(
                "POST",
                f"http://127.0.0.1:{helper_port}/local/search",
                {"task_id": "TASK001", "query": "alpha question", "k": 2, "threshold": 0.0},
            )
            browsecomp_search = http_json(
                "POST",
                f"http://127.0.0.1:{helper_port}/browsecomp/search",
                {"query": "alpha question", "task_id": "TASK001", "k": 2},
            )

            assert health["ok"] is True
            assert health["embedding_server_ok"] is True
            assert len(embed["embeddings"]) == 1
            assert local_search["hits"][0]["doc_id"] == "local-1"
            assert browsecomp_search["hits"][0]["docid"] == "web/1"

            print("privacy_helper smoke passed")
            print(json.dumps(
                {
                    "health": {
                        "local_task_count": health["local_task_count"],
                        "browsecomp_size": health["browsecomp_size"],
                    },
                    "local_top_hit": local_search["hits"][0]["doc_id"],
                    "browsecomp_top_hit": browsecomp_search["hits"][0]["docid"],
                },
                indent=2,
            ))
            return 0
        finally:
            helper_process.terminate()
            try:
                helper_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                helper_process.kill()
                helper_process.wait()
            embedding_server.shutdown()
            embedding_server.server_close()
            embedding_thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
