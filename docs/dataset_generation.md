# Dataset Generation

Use this guide to run the MosaicLeaks dataset build. The model endpoint can
vary, but every run should record the endpoint, model name, input artifact
hashes, generation settings, and output manifest.

## 1. Chunk Local Documents

For paper replication, chunk a DRBench-style task root:

```bash
python3 -m mosaicleaks.data_prep.chunk_local_tasks \
  --tasks-root /path/to/drbench/tasks \
  --output mosaicleaks/data/chunks_local.jsonl
```

For other local document collections, chunk a folder as one task:

```bash
python3 -m mosaicleaks.data_prep.chunk_local_folder \
  --input-root /path/to/private_docs \
  --task-id example_task \
  --company-name "Example Company" \
  --output mosaicleaks/data/chunks_local.jsonl
```

Or pass a JSONL manifest with one file per row. This is the best format when a
collection has multiple tasks or groups:

```json
{"path": "/path/to/file.pdf", "task_id": "task_001", "company_name": "Example Company"}
```

```bash
python3 -m mosaicleaks.data_prep.chunk_local_folder \
  --manifest local_docs.jsonl \
  --output mosaicleaks/data/chunks_local.jsonl
```

Manifest fields:

- `path`: required file path, absolute or relative to the manifest file
- `task_id`: required group ID used for chain construction and local retrieval
- `company_name`: optional label used for split grouping and metadata
- `relative_path`: optional stable path to use inside generated `doc_id` values

The chunker has dedicated readers for PDF, DOC, DOCX, XLS, XLSX, and JSONL,
and falls back to text decoding for other files. Hidden files are skipped.

Validation:

- every included task has at least one chunk
- chunk offsets are monotonic within each document
- document metadata includes task ID and company name when provided
- unsupported files are skipped; parse errors in included files fail the run

## 2. Extract Private QA Candidates

Input: `chunks_local.jsonl`.

```bash
python3 -m mosaicleaks.data_prep.extract_local_secrets \
  --chunks-local mosaicleaks/data/chunks_local.jsonl \
  --output mosaicleaks/data/secret_inventory_raw.jsonl \
  --base-url "$MOSAICLEAKS_LLM_BASE_URL" \
  --model "$MOSAICLEAKS_LLM_MODEL"
```

Validation:

- generated questions are answerable from the local document chunk
- each candidate records source chunk, answer, quote, and secret type
- malformed model outputs are preserved in logs for review

If you already have private QA seeds, import them instead of running extraction.
JSONL is the recommended format:

```json
{"doc_id": "local/task_001/file.pdf", "question": "What was the internal launch date?", "answer": "Q3 2026", "secret_type": "dates"}
```

```bash
python3 -m mosaicleaks.data_prep.import_seeds \
  --input private_seeds.jsonl \
  --chunks-local mosaicleaks/data/chunks_local.jsonl \
  --output mosaicleaks/data/secret_inventory.jsonl
```

The importer validates that each `doc_id` exists, optional `task_id` values
match the referenced document, and the answer appears in the document text.

## 3. Filter Private QA Candidates

Input: raw secret inventory.

```bash
python3 -m mosaicleaks.data_prep.filter_local_secrets \
  --input mosaicleaks/data/secret_inventory_raw.jsonl \
  --output-dir mosaicleaks/data \
  --base-url "$MOSAICLEAKS_LLM_BASE_URL" \
  --model "$MOSAICLEAKS_LLM_MODEL"
```

Validation:

- document-only checks pass for retained QA pairs
- names, emails, placeholders, and not-answerable rows are removed
- accepted answers are short enough to be usable as chain seeds

## 4. Prepare Web Inputs

Primary web corpus:

- use BrowseComp-Plus through `Tevatron/browsecomp-plus-corpus`, or
- pass a local BrowseComp-style JSONL corpus with `docid`, `url`, and `text`

BrowseComp/web chunks are required for any generation pattern containing `W`.

Optional curated task URL files:

```bash
python3 -m mosaicleaks.data_prep.fetch_drbench_urls \
  --tasks-root /path/to/drbench/tasks \
  --output mosaicleaks/data/docs_curated_web.jsonl \
  --chunk-output mosaicleaks/data/chunks_curated_web.jsonl
```

Validation:

- fetched URL rows record HTTP status, extraction method, and task IDs
- only successful direct-fetch rows enter curated web retrieval
- curated chunks preserve source URL and task allowlist metadata

## 5. Build Chains

```bash
python3 -m mosaicleaks.run \
  --chunks-local mosaicleaks/data/chunks_local.jsonl \
  --chunks-web mosaicleaks/data/chunks_web.jsonl \
  --curated-web-chunks mosaicleaks/data/chunks_curated_web.jsonl \
  --curated-web-docs mosaicleaks/data/docs_curated_web.jsonl \
  --secrets mosaicleaks/data/secret_inventory.jsonl \
  --base-url "$MOSAICLEAKS_LLM_BASE_URL" \
  --model "$MOSAICLEAKS_LLM_MODEL" \
  --output-dir runs/example_generation
```

The pipeline stages are:

- `load`: load chunks, private QA seeds, task metadata, and entity indexes
- `build`: construct multi-hop chains over local and web evidence
- `verify`: check answerability under context ablations
- `filter`: score chain quality with an LLM judge
- `reword`: make minimal clarity edits without changing semantics

Validation:

- every hop has a source document, question, answer, and supporting text
- bridge substitutions do not leak private answers into web-only hops
- no-doc, first-doc-only, last-doc-only, and all-doc checks are recorded
- invalid chains are retained in intermediate outputs but excluded from final
  dataset publication

## 6. Publish A Versioned Dataset

Publishing should create a self-contained bundle:

```text
dataset.jsonl
train.jsonl
val.jsonl
test.jsonl
manifest.json
dataset_stats.json
schema.json
```

```bash
python3 -m mosaicleaks.publish_dataset \
  --input runs/example_generation/chains_final.jsonl \
  --output-dir datasets/mosaicleaks_example \
  --dataset-name mosaicleaks_example \
  --split-key company
```

Validation:

- splits are deterministic and non-overlapping
- split union equals the full dataset
- held-out test grouping is documented
- every final answer appears in its accepted-answer list
- no empty questions, answers, hop lists, or document IDs
- all artifact paths in the manifest are relative or release-stable
- source code commit and input artifact hashes are recorded

## 7. Paper Replication

Paper tables and plots should read from the published dataset bundle. Runtime
rollout or evaluation scripts should write their own manifest with the dataset
bundle, code commit, model, retrieval helper, and output paths.
