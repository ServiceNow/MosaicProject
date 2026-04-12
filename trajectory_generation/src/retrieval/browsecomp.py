"""BrowseComp retrieval adapters for trajectory generation.

This module contains the small amount of BrowseComp code needed by the
generator so the package can run without importing the larger neighboring repos.
"""

from __future__ import annotations

import glob
import logging
import pickle
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_generation.src.retrieval.bm25 import BM25Index
from trajectory_generation.src.retrieval.types import RetrievalHit

logger = logging.getLogger(__name__)


def _load_dataset(dataset_name: str):
    from datasets import load_dataset

    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")

    docid_to_idx: dict[str, int] = {}
    docid_to_url: dict[str, str] = {}
    for idx, row in enumerate(dataset):
        docid = row.get("docid")
        if docid is None:
            continue
        docid_str = str(docid)
        docid_to_idx[docid_str] = idx
        docid_to_url[docid_str] = row.get("url") or ""
    return dataset, docid_to_idx, docid_to_url


def _resolve_cache_dir(index_glob: str | None) -> Path | None:
    if not index_glob:
        return None
    index_files = sorted(glob.glob(index_glob))
    if not index_files:
        return None
    return Path(index_files[0]).parent


def _load_cached_bm25(cache_path: Path):
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as f:
        data = pickle.load(f)

    bm25 = BM25Index.__new__(BM25Index)
    bm25._docs = []
    bm25._k1 = data["k1"]
    bm25._b = data["b"]
    bm25._N = data["N"]
    bm25._doc_len = data["doc_len"]
    bm25._avgdl = data["avgdl"]
    bm25._df = data["df"]
    bm25._inv = data["inv"]
    return bm25, data["mapping"]


def _save_cached_bm25(cache_path: Path, bm25: BM25Index, mapping: Any) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {
                "k1": bm25._k1,
                "b": bm25._b,
                "N": bm25._N,
                "doc_len": bm25._doc_len,
                "avgdl": bm25._avgdl,
                "df": dict(bm25._df),
                "inv": dict(bm25._inv),
                "mapping": mapping,
            },
            f,
        )


class _BrowseCompBase:
    dataset: Any
    docid_to_idx: dict[str, int]
    docid_to_url: dict[str, str]
    max_chars: int

    @staticmethod
    def _normalize_docid(docid: str) -> str:
        docid = str(docid)
        return docid if docid.startswith("web/") else f"web/{docid}"

    def _get_doc(self, docid: str) -> dict[str, str]:
        idx = self.docid_to_idx.get(str(docid))
        if idx is None:
            return {"docid": str(docid), "url": "", "text": ""}
        row = self.dataset[int(idx)]
        return {
            "docid": str(docid),
            "url": row.get("url") or "",
            "text": row.get("text") or "",
        }

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + "\n\n[truncated]"


class BrowseCompHybridSearcher(_BrowseCompBase):
    """BrowseComp searcher that mixes BM25 recall with local embedding reranking."""

    def __init__(
        self,
        *,
        index_glob: str,
        model_name: str,
        dataset_name: str,
        device: str = "cuda",
        bm25_recall_k: int = 200,
        dense_recall_k: int = 200,
        max_chars: int = 8_000,
    ) -> None:
        self.index_glob = index_glob
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device
        self.bm25_recall_k = int(bm25_recall_k)
        self.dense_recall_k = int(dense_recall_k)
        self.max_chars = int(max_chars)

        self.searcher = None
        self.lookup: list[str] = []
        self.dataset = None
        self.docid_to_idx = {}
        self.docid_to_url = {}
        self.model = None
        self.tokenizer = None
        self.bm25: BM25Index | None = None
        self._bm25_idx_to_lookup_idx: dict[int, int] | None = None
        self._encode_lock = threading.Lock()
        self.task_prefix = (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:"
        )

        self._load_faiss_index()
        self._load_model_and_tokenizer()
        self.dataset, self.docid_to_idx, self.docid_to_url = _load_dataset(dataset_name)
        self._build_bm25()

    @property
    def size(self) -> int:
        return len(self.lookup)

    def _load_faiss_index(self) -> None:
        from tevatron.retriever.searcher import FaissFlatSearcher

        def pickle_load(path: str):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), list(lookup)

        index_files = sorted(glob.glob(self.index_glob))
        if not index_files:
            raise RuntimeError(f"No index shards found for pattern: {self.index_glob}")

        reps0, lookup0 = pickle_load(index_files[0])
        self.searcher = FaissFlatSearcher(reps0)
        self.searcher.add(reps0)
        self.lookup = list(lookup0)

        for path in index_files[1:]:
            reps, shard_lookup = pickle_load(path)
            self.searcher.add(reps)
            self.lookup.extend(shard_lookup)

    def _load_model_and_tokenizer(self) -> None:
        import torch
        from transformers import AutoTokenizer
        from tevatron.retriever.modeling.dense import DenseModel

        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = DenseModel.load(
            self.model_name,
            pooling="eos",
            normalize=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_bm25(self) -> None:
        if not self.lookup:
            self.bm25 = None
            self._bm25_idx_to_lookup_idx = None
            return

        cache_dir = _resolve_cache_dir(self.index_glob)
        cache_path = cache_dir / "bm25_cache.pkl" if cache_dir is not None else None
        if cache_path is not None:
            cached = _load_cached_bm25(cache_path)
            if cached is not None:
                self.bm25, self._bm25_idx_to_lookup_idx = cached
                return

        lookup_idx_by_docid = {str(docid): i for i, docid in enumerate(self.lookup)}
        texts: list[str] = []
        idx_to_lookup: dict[int, int] = {}

        for docid, ds_idx in self.docid_to_idx.items():
            row = self.dataset[int(ds_idx)]
            text = (row.get("text") or "").strip()
            lookup_i = lookup_idx_by_docid.get(docid)
            if not text or lookup_i is None:
                continue
            bm25_i = len(texts)
            texts.append(text)
            idx_to_lookup[bm25_i] = lookup_i

        self.bm25 = BM25Index(texts)
        self._bm25_idx_to_lookup_idx = idx_to_lookup
        if cache_path is not None:
            _save_cached_bm25(cache_path, self.bm25, idx_to_lookup)

    def _encode_query(self, query: str) -> np.ndarray:
        import torch

        batch = self.tokenizer(
            self.task_prefix + query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.device.startswith("cuda"):
            ctx = torch.amp.autocast(device_type="cuda")
        else:
            ctx = nullcontext()

        with self._encode_lock:
            with ctx:
                with torch.no_grad():
                    reps = self.model.encode_query(batch)
        return reps.cpu().numpy()

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str = "bm25",
        **_kwargs,
    ) -> list[RetrievalHit]:
        if not query or k <= 0:
            return []
        if mode not in {"bm25", "hybrid"}:
            raise ValueError("BrowseCompHybridSearcher supports mode='bm25' or mode='hybrid'")
        if self.searcher is None or not self.lookup:
            return []

        q_reps = self._encode_query(query)
        candidate_lookup_idxs: set[int] = set()

        if self.bm25 is not None and self._bm25_idx_to_lookup_idx is not None:
            for bm25_idx, _score in self.bm25.search(query, k=self.bm25_recall_k):
                lookup_i = self._bm25_idx_to_lookup_idx.get(int(bm25_idx))
                if lookup_i is not None:
                    candidate_lookup_idxs.add(int(lookup_i))

        dense_k = min(self.dense_recall_k, len(self.lookup))
        if dense_k > 0:
            _scores, dense_indices = self.searcher.search(q_reps, dense_k)
            for idx in dense_indices[0]:
                candidate_lookup_idxs.add(int(idx))

        if not candidate_lookup_idxs:
            return []

        candidate_list = sorted(candidate_lookup_idxs)
        candidate_embs = np.array([self.searcher.index.reconstruct(i) for i in candidate_list])
        rerank_scores = candidate_embs @ q_reps[0]
        ranked = sorted(
            zip(candidate_list, rerank_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        hits: list[RetrievalHit] = []
        for lookup_idx, score in ranked[:k]:
            raw_docid = str(self.lookup[lookup_idx])
            doc = self._get_doc(raw_docid)
            hits.append(
                RetrievalHit(
                    chunk_id=self._normalize_docid(raw_docid),
                    doc_id=self._normalize_docid(raw_docid),
                    score=float(score),
                    text=self._truncate(doc.get("text") or ""),
                    meta={
                        "url": doc.get("url") or "",
                        "raw_docid": raw_docid,
                        "source": "browsecomp_hybrid",
                    },
                )
            )
        return hits


class BrowseCompBM25Searcher(_BrowseCompBase):
    """Dependency-light BrowseComp BM25 searcher built from the fixed corpus."""

    def __init__(
        self,
        *,
        dataset_name: str,
        index_glob: str | None = None,
        max_chars: int = 8_000,
    ) -> None:
        self.dataset_name = dataset_name
        self.index_glob = index_glob
        self.max_chars = int(max_chars)

        self.dataset, self.docid_to_idx, self.docid_to_url = _load_dataset(dataset_name)
        self.bm25, self._docids = self._build_bm25()

    def _build_bm25(self) -> tuple[BM25Index, list[str]]:
        cache_dir = _resolve_cache_dir(self.index_glob)
        cache_path = cache_dir / "bm25_dataset_cache.pkl" if cache_dir is not None else None
        if cache_path is not None:
            cached = _load_cached_bm25(cache_path)
            if cached is not None:
                bm25, docids = cached
                return bm25, list(docids)

        texts: list[str] = []
        docids: list[str] = []
        for docid, ds_idx in self.docid_to_idx.items():
            row = self.dataset[int(ds_idx)]
            text = (row.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            docids.append(docid)

        bm25 = BM25Index(texts)
        if cache_path is not None:
            _save_cached_bm25(cache_path, bm25, docids)
        return bm25, docids

    @property
    def size(self) -> int:
        return len(self._docids)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str = "bm25",
        **_kwargs,
    ) -> list[RetrievalHit]:
        if mode != "bm25":
            raise ValueError("BrowseCompBM25Searcher only supports mode='bm25'")

        hits: list[RetrievalHit] = []
        for bm25_hit in self.bm25.search(query, k=k):
            raw_docid = self._docids[int(bm25_hit.doc_idx)]
            doc = self._get_doc(raw_docid)
            hits.append(
                RetrievalHit(
                    chunk_id=self._normalize_docid(raw_docid),
                    doc_id=self._normalize_docid(raw_docid),
                    score=float(bm25_hit.score),
                    text=self._truncate(doc.get("text") or ""),
                    meta={
                        "url": doc.get("url") or "",
                        "raw_docid": raw_docid,
                        "source": "browsecomp_bm25",
                    },
                )
            )
        return hits
