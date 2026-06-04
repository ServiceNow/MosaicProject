# External Dependencies

MosaicLeaks does not include private task files, BrowseComp-Plus corpora, dense
embedding shards, model weights, or API credentials. The pipeline expects those
artifacts to be supplied by path, dataset name, or environment variable.

## Local Private Documents

The generation pipeline consumes local private documents as `chunks_local.jsonl`.
Each chunk row must include:

- `chunk_id`
- `doc_id`
- `source_type`
- `text`
- `offsets`
- `meta.task_id`

Recommended metadata:

- `meta.company_name`
- `meta.file_name`
- `meta.relative_path`
- `meta.extension`

For paper replication, build this file from a DRBench-style task root with
folders named like `DR0001`, `DR0002`, and so on. Each task may contain:

- task-local files to chunk: Markdown, text, CSV, JSONL, PDF, DOCX, XLSX
- `context.json` with `company_info.name`
- `config/eval.json` with optional enterprise-fact seed QA pairs

For other local corpora, use `mosaicleaks.data_prep.chunk_local_folder`. The
manifest form is recommended when you have more than one task:

```json
{"path": "/path/to/file.pdf", "task_id": "task_001", "company_name": "Example Company"}
```

Manifest fields:

- `path`: required file path, absolute or relative to the manifest file
- `task_id`: required group ID used for chain construction and local retrieval
- `company_name`: optional label used for metadata and split grouping
- `relative_path`: optional stable path for generated `doc_id` values

Manual private QA seeds can also be supplied as JSONL and imported into
`secret_inventory.jsonl`:

```json
{"doc_id": "local/task_001/file.pdf", "question": "What was the internal launch date?", "answer": "Q3 2026", "secret_type": "dates"}
```

## BrowseComp-Plus

BrowseComp-Plus is the primary public-web corpus. Dataset generation uses
prepared web chunks for bridge discovery:

```text
chunks_web.jsonl
```

Each chunk row should include `chunk_id`, `doc_id`, `text`, `offsets`, and
optional metadata. The retrieval helper can additionally use the full corpus in
two ways:

- `--browsecomp-corpus Tevatron/browsecomp-plus-corpus`, loaded through the
  `datasets` package.
- `--browsecomp-corpus /path/to/corpus.jsonl`, where each row has `docid`,
  `url`, and `text`.

Dense BrowseComp retrieval additionally needs embedding shards compatible with
the MosaicLeaks `BrowseCompHybridSearcher`:

```text
corpus.shard00000_of_000NN.pkl
corpus.shard00001_of_000NN.pkl
...
```

Each shard stores the dense vectors and lookup IDs used to map search results
back to BrowseComp document IDs. The pickle payload must be a two-item tuple:
`(embeddings, lookup)`, where `embeddings` is a 2D float array and `lookup`
contains the corresponding BrowseComp document IDs in row order. The query
embedding endpoint should serve the same embedding model used to create those
shards.

## Curated Task URLs

The generation pipeline can supplement BrowseComp-Plus with small curated web
files derived from task URLs:

- `docs_curated_web.jsonl`: fetched URL metadata and full text
- `chunks_curated_web.jsonl`: paragraph/heading-aware chunks

These files are optional. When present, the helper and generator restrict their
use by task ID so they do not become a general web source.

## OpenAI-Compatible Endpoints

All LLM and embedding calls use OpenAI-compatible APIs:

- chat completions for generation, validation, filtering, and rewording
- embeddings for dense local and BrowseComp retrieval

StepFun can be used as the default provider by setting:

```bash
export MOSAICLEAKS_LLM_BASE_URL=http://your-endpoint:8000/v1
export MOSAICLEAKS_LLM_MODEL=step-3.5-flash
```

Equivalent local vLLM, OpenAI, OpenRouter, or other compatible endpoints should
work when they implement the same API shape.
