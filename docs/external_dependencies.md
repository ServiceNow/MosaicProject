# External Dependencies

MosaicLeaks does not vendor DRBench or BrowseComp-Plus. It consumes their task
and corpus artifacts through explicit paths.

## DRBench-Style Tasks

The local-document side of the dataset is built from a task root with folders
named like `DR0001`, `DR0002`, and so on. Each task may contain:

- task-local files to chunk: Markdown, text, CSV, JSONL, PDF, DOCX, XLSX
- `context.json` with `company_info.name`
- `config/eval.json` with optional enterprise-fact seed QA pairs

Core generation code does not import DRBench Python modules. The task directory
is treated as an input data format.

The local chunker emits records with:

- `chunk_id`
- `doc_id`
- `source_type`
- `text`
- `offsets`
- `meta.task_id`
- `meta.company_name`
- `meta.file_name`

## BrowseComp-Plus

BrowseComp-Plus is the primary public-web corpus. MosaicLeaks can use it in two
ways:

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
back to BrowseComp document IDs. The clean repo documents this artifact format
instead of assuming a local BrowseComp-Plus checkout.

## Curated Task URLs

The generation pipeline can supplement BrowseComp-Plus with a small curated web
sidecar derived from task URLs. The sidecar has two files:

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
