"""BrowseComp retrieval adapters for MosaicLeaks dataset generation.

The helper service uses the same hybrid retrieval logic as the generator:
BM25 recall plus dense recall, followed by dense reranking. The dense query
encoder can be supplied externally so the helper can use a colocated vLLM
embedding server instead of loading another model in-process.
"""

import glob
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from mosaicleaks.src.retrieval.bm25 import BM25Index, BM25State
from mosaicleaks.src.retrieval.types import RetrievalHit

logger = logging.getLogger(__name__)

TASK_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


def _load_dataset(dataset_name: str) -> tuple[Any, dict[str, int], dict[str, str]]:
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")

    first_row = dataset[0] if len(dataset) > 0 else {}
    if isinstance(first_row, dict) and "docid" not in first_row:
        flattened_docs: list[dict[str, str]] = []
        seen_docids: set[str] = set()
        for row in dataset:
            for field_name in ("gold_docs", "negative_docs", "evidence_docs"):
                docs = row.get(field_name) or []
                if not isinstance(docs, list):
                    continue
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    docid = doc.get("docid")
                    if docid is None:
                        continue
                    docid_str = str(docid)
                    if docid_str in seen_docids:
                        continue
                    seen_docids.add(docid_str)
                    flattened_docs.append(
                        {
                            "docid": docid_str,
                            "url": doc.get("url") or "",
                            "text": doc.get("text") or "",
                        }
                    )
        dataset = flattened_docs

    docid_to_idx: dict[str, int] = {}
    docid_to_url: dict[str, str] = {}
    for idx, row in enumerate(dataset):
        docid = row.get("docid")
        if docid is None:
            continue
        docid_str = str(docid)
        docid_to_idx[docid_str] = idx
        docid_to_idx[f"web/{docid_str}"] = idx
        docid_to_url[docid_str] = row.get("url") or ""
        docid_to_url[f"web/{docid_str}"] = row.get("url") or ""
    return dataset, docid_to_idx, docid_to_url


def _iter_dataset_docs(dataset: Any):
    for idx, row in enumerate(dataset):
        if not isinstance(row, dict):
            continue
        docid = row.get("docid")
        if docid is None:
            continue
        yield str(docid), idx, row


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

    mapping = data.get("mapping")
    if mapping is None:
        mapping = data.get("idx_to_lookup")
    if mapping is None:
        logger.warning("Ignoring incompatible BM25 cache at %s", cache_path)
        return None

    state = BM25State(
        k1=float(data["k1"]),
        b=float(data["b"]),
        N=int(data["N"]),
        doc_len=list(data["doc_len"]),
        avgdl=float(data["avgdl"]),
        df=dict(data["df"]),
        inv={term: list(postings) for term, postings in dict(data["inv"]).items()},
    )
    bm25 = BM25Index.from_state(state)
    return bm25, mapping


def _save_cached_bm25(cache_path: Path, bm25: BM25Index, mapping: Any) -> None:
    state = bm25.to_state()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {
                "k1": state.k1,
                "b": state.b,
                "N": state.N,
                "doc_len": state.doc_len,
                "avgdl": state.avgdl,
                "df": state.df,
                "inv": state.inv,
                "mapping": mapping,
            },
            f,
        )


class _BrowseCompBase:
    """Shared document lookup and truncation helpers for BrowseComp searchers."""

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

    def _truncate(self, text: str, max_chars: int | None = None) -> str:
        limit = self.max_chars if max_chars is None else int(max_chars)
        if limit <= 0:
            return text
        if len(text) <= limit:
            return text
        return text[:limit] + "\n\n[truncated]"


class BrowseCompHybridSearcher(_BrowseCompBase):
    """BrowseComp searcher that mixes BM25 recall with dense reranking."""

    def __init__(
        self,
        index_glob: str,
        dataset_name: str,
        embedder: Any,
        bm25_recall_k: int = 200,
        dense_recall_k: int = 200,
        max_chars: int = 8_000,
        index_device: str = "cpu",
    ) -> None:
        self.index_glob = index_glob
        self.dataset_name = dataset_name
        # The generator and helper pass different embedding clients, but both
        # expose the same small embed_query method.
        self.embedder = embedder
        self.bm25_recall_k = int(bm25_recall_k)
        self.dense_recall_k = int(dense_recall_k)
        self.max_chars = int(max_chars)
        self.index_device = index_device

        self.lookup: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.dataset = None
        self.docid_to_idx = {}
        self.docid_to_url = {}
        self.bm25: BM25Index | None = None
        self._bm25_idx_to_lookup_idx: dict[int, int] | None = None
        self._faiss_index = None
        self._dense_backend = "faiss_cpu"
        self._torch_device = None
        self._torch_embeddings = None

        self._load_faiss_index()
        self.dataset, self.docid_to_idx, self.docid_to_url = _load_dataset(dataset_name)
        self._build_bm25()

    @property
    def size(self) -> int:
        return len(self.lookup)

    @property
    def dense_backend(self) -> str:
        return self._dense_backend

    def _has_dense_backend(self) -> bool:
        if self._dense_backend.startswith("torch:"):
            return self._torch_embeddings is not None
        return self._faiss_index is not None

    def _load_torch_dense_backend(self, embeddings: np.ndarray) -> None:
        import torch

        device = torch.device(self.index_device)
        self._torch_device = device
        with torch.no_grad():
            self._torch_embeddings = torch.as_tensor(embeddings, device=device, dtype=torch.float32)
        self._dense_backend = f"torch:{device}"

    def _torch_query(self, query_vector: np.ndarray):
        import torch

        if self._torch_device is None:
            raise RuntimeError("Torch dense backend requested before initialization")
        return torch.as_tensor(query_vector, device=self._torch_device, dtype=torch.float32)

    def _search_dense(self, query_vector: np.ndarray, dense_k: int) -> list[int]:
        if self._dense_backend.startswith("torch:"):
            import torch

            if self._torch_embeddings is None:
                raise RuntimeError("Torch dense backend is not initialized")
            with torch.no_grad():
                scores = torch.mv(self._torch_embeddings, self._torch_query(query_vector))
                topk = torch.topk(scores, k=min(dense_k, scores.shape[0]), dim=0)
            return [int(idx) for idx in topk.indices.detach().cpu().tolist()]

        query_matrix = np.expand_dims(query_vector, axis=0)
        _scores, dense_indices = self._faiss_index.search(query_matrix, dense_k)
        return [int(idx) for idx in dense_indices[0] if int(idx) >= 0]

    def _rerank_scores(self, candidate_list: list[int], query_vector: np.ndarray) -> np.ndarray:
        if self._dense_backend.startswith("torch:"):
            import torch

            if self._torch_embeddings is None:
                raise RuntimeError("Torch dense backend is not initialized")
            with torch.no_grad():
                candidate_embs = self._torch_embeddings[candidate_list]
                scores = torch.mv(candidate_embs, self._torch_query(query_vector))
            return scores.detach().cpu().numpy()

        if self.embeddings is None:
            raise RuntimeError("CPU embeddings are required for FAISS hybrid reranking")
        candidate_embs = self.embeddings[candidate_list]
        return candidate_embs @ query_vector

    def _load_faiss_index(self) -> None:
        import faiss

        shard_embeddings: list[np.ndarray] = []
        shard_docids: list[str] = []

        index_files = sorted(glob.glob(self.index_glob))
        if not index_files:
            raise RuntimeError(f"No index shards found for pattern: {self.index_glob}")

        for path in index_files:
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            shard_embeddings.append(np.asarray(reps, dtype=np.float32))
            shard_docids.extend(str(docid) for docid in lookup)

        embeddings = np.concatenate(shard_embeddings, axis=0)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(shard_docids):
            raise RuntimeError("BrowseComp index shards are inconsistent")

        self.lookup = shard_docids

        if self.index_device.startswith("cuda"):
            gpu_id = 0
            if ":" in self.index_device:
                gpu_id = int(self.index_device.split(":", 1)[1])
            if hasattr(faiss, "StandardGpuResources"):
                index = faiss.IndexFlatIP(int(embeddings.shape[1]))
                index.add(embeddings)
                resources = faiss.StandardGpuResources()
                options = faiss.GpuClonerOptions()
                options.useFloat16 = True
                index = faiss.index_cpu_to_gpu(resources, gpu_id, index, options)
                self._dense_backend = f"faiss_gpu:{gpu_id}"
                self.embeddings = embeddings
                self._faiss_index = index
                return
            else:
                logger.warning(
                    "faiss-gpu bindings are unavailable; using torch dense retrieval on %s",
                    self.index_device,
                )
                self._load_torch_dense_backend(embeddings)
                self.embeddings = None
                self._faiss_index = None
                return

        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        self.embeddings = embeddings
        self._faiss_index = index

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

        for docid, _ds_idx, row in _iter_dataset_docs(self.dataset):
            text = (row.get("text") or "").strip()
            lookup_i = lookup_idx_by_docid.get(docid)
            if lookup_i is None:
                lookup_i = lookup_idx_by_docid.get(self._normalize_docid(docid))
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
        text = f"{TASK_PREFIX}{query}"
        return np.asarray(self.embedder.embed_query(text), dtype=np.float32)

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "bm25",
        max_chars: int | None = None,
        rrf_k: int = 20,
    ) -> list[RetrievalHit]:
        if not query or k <= 0:
            return []
        if mode not in {"bm25", "hybrid", "rrf"}:
            raise ValueError("BrowseCompHybridSearcher supports mode='bm25', mode='hybrid', or mode='rrf'")
        if not self.lookup:
            return []

        bm25_lookup_ranks: dict[int, int] = {}
        bm25_lookup_scores: dict[int, float] = {}

        if self.bm25 is not None and self._bm25_idx_to_lookup_idx is not None:
            bm25_k = k if mode == "bm25" else self.bm25_recall_k
            for rank, (bm25_idx, score) in enumerate(self.bm25.search(query, k=bm25_k), start=1):
                lookup_i = self._bm25_idx_to_lookup_idx.get(int(bm25_idx))
                if lookup_i is not None:
                    bm25_lookup_ranks.setdefault(int(lookup_i), rank)
                    bm25_lookup_scores.setdefault(int(lookup_i), float(score))

        if mode == "bm25":
            hits: list[RetrievalHit] = []
            for lookup_idx, rank in sorted(bm25_lookup_ranks.items(), key=lambda item: item[1])[:k]:
                raw_docid = str(self.lookup[lookup_idx])
                doc = self._get_doc(raw_docid)
                hits.append(
                    RetrievalHit(
                        chunk_id=self._normalize_docid(raw_docid),
                        doc_id=self._normalize_docid(raw_docid),
                        score=bm25_lookup_scores.get(lookup_idx, 0.0),
                        text=self._truncate(doc.get("text") or "", max_chars=max_chars),
                        meta={
                            "url": doc.get("url") or "",
                            "raw_docid": raw_docid,
                            "source": "browsecomp_bm25",
                        },
                    )
                )
            return hits

        if not self._has_dense_backend():
            return []

        query_vector = self._encode_query(query).astype(np.float32, copy=False)
        candidate_lookup_idxs: set[int] = set(bm25_lookup_ranks)
        dense_lookup_ranks: dict[int, int] = {}

        dense_k = min(self.dense_recall_k, len(self.lookup))
        if dense_k > 0:
            for rank, idx in enumerate(self._search_dense(query_vector, dense_k), start=1):
                if idx >= 0:
                    dense_lookup_ranks.setdefault(int(idx), rank)
                    candidate_lookup_idxs.add(int(idx))

        if not candidate_lookup_idxs:
            return []

        if mode == "rrf":
            rrf_base = max(1, int(rrf_k))
            ranked = []
            for lookup_idx in sorted(candidate_lookup_idxs):
                score = 0.0
                if lookup_idx in bm25_lookup_ranks:
                    score += 1.0 / (rrf_base + bm25_lookup_ranks[lookup_idx])
                if lookup_idx in dense_lookup_ranks:
                    score += 1.0 / (rrf_base + dense_lookup_ranks[lookup_idx])
                ranked.append((lookup_idx, score))
            ranked.sort(key=lambda item: float(item[1]), reverse=True)
            source = "browsecomp_rrf"
        else:
            candidate_list = sorted(candidate_lookup_idxs)
            rerank_scores = self._rerank_scores(candidate_list, query_vector)
            ranked = sorted(
                zip(candidate_list, rerank_scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            source = "browsecomp_hybrid"

        hits: list[RetrievalHit] = []
        for lookup_idx, score in ranked[:k]:
            raw_docid = str(self.lookup[lookup_idx])
            doc = self._get_doc(raw_docid)
            hits.append(
                RetrievalHit(
                    chunk_id=self._normalize_docid(raw_docid),
                    doc_id=self._normalize_docid(raw_docid),
                    score=float(score),
                    text=self._truncate(doc.get("text") or "", max_chars=max_chars),
                    meta={
                        "url": doc.get("url") or "",
                        "raw_docid": raw_docid,
                        "source": source,
                    },
                )
            )
        return hits


class BrowseCompBM25Searcher(_BrowseCompBase):
    """Dependency-light BrowseComp BM25 searcher built from the fixed corpus."""

    def __init__(
        self,
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
        for docid, _ds_idx, row in _iter_dataset_docs(self.dataset):
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
        k: int = 10,
        mode: str = "bm25",
        max_chars: int | None = None,
    ) -> list[RetrievalHit]:
        if mode != "bm25":
            raise ValueError("BrowseCompBM25Searcher only supports mode='bm25'")

        hits: list[RetrievalHit] = []
        for bm25_idx, score in self.bm25.search(query, k=k):
            raw_docid = self._docids[int(bm25_idx)]
            doc = self._get_doc(raw_docid)
            hits.append(
                RetrievalHit(
                    chunk_id=self._normalize_docid(raw_docid),
                    doc_id=self._normalize_docid(raw_docid),
                    score=float(score),
                    text=self._truncate(doc.get("text") or "", max_chars=max_chars),
                    meta={
                        "url": doc.get("url") or "",
                        "raw_docid": raw_docid,
                        "source": "browsecomp_bm25",
                    },
                )
            )
        return hits
