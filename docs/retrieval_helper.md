# Retrieval Helper

The retrieval helper is a reusable HTTP service used by dataset generation and
agent-style evaluation runs. It is provider-agnostic: embeddings come from an
OpenAI-compatible embedding endpoint.

## API

`GET /health`

Returns service configuration, index sizes, embedding health, and rolling
latency metrics.

`POST /embed`

```json
{"texts": ["text to embed"], "model": "optional-model-name"}
```

`POST /local/search`

```json
{"task_id": "DR0001", "query": "question text", "k": 10, "threshold": 0.0}
```

`POST /browsecomp/search`

```json
{"task_id": "DR0001", "query": "web query", "k": 5, "max_chars": 8000}
```

## Local Index Format

Build local indices from `chunks_local.jsonl` before starting the helper:

```bash
python3 -m mosaicleaks.data_prep.build_local_indices \
  --chunks-local mosaicleaks/data/chunks_local.jsonl \
  --output-root mosaicleaks/data/local_indices \
  --embedding-base-url http://127.0.0.1:8000 \
  --embedding-model Qwen/Qwen3-Embedding-4B
```

`--local-index-root` points to the output directory, which has one subdirectory
per task:

```text
local_indices/
  DR0001/
    documents.json
    index.json
    embeddings.npy
  manifest.json
```

`documents.json` maps document IDs to content and metadata. `index.json`
contains a `doc_ids` list aligned with rows in `embeddings.npy`.

## BrowseComp Inputs

The helper needs:

- `--browsecomp-corpus`: Hugging Face dataset name or local corpus JSONL
- `--browsecomp-index-glob`: dense embedding shard glob; this is required
- `--embedding-base-url`: OpenAI-compatible embedding endpoint
- `--embedding-model`: embedding model name served by that endpoint

The helper supports dense, BM25, and reciprocal-rank-fusion retrieval modes.
RRF is the recommended default for replication because it is less brittle than
pure dense ranking when queries contain rare identifiers.

For training runs, the local index should cover every local gold document in the
published dataset. If the dataset is regenerated, rebuild the local indices from
the same `chunks_local.jsonl` that was used during generation rather than
reusing an older index root.

Long BrowseComp parents can make reader prompts exceed the model context. The
helper can drop or cap those parents with `--browsecomp-max-doc-windows`,
`--browsecomp-reader-window-chars`, `--browsecomp-reader-window-overlap-chars`,
and `--browsecomp-exclude-doc-ids-file`.

## Launch

```bash
./scripts/start_retrieval_helper.sh \
  --local-index-root /path/to/local_indices \
  --embedding-base-url http://127.0.0.1:8000 \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --browsecomp-index-glob "/path/to/corpus.shard*_of_*.pkl" \
  --browsecomp-corpus Tevatron/browsecomp-plus-corpus \
  --browsecomp-max-doc-windows 10 \
  --browsecomp-rank-mode rrf \
  --local-rank-mode rrf
```
