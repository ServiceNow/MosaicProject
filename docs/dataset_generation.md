# Dataset Generation

This page is the intended replication guide for the MosaicLeaks dataset build.
The exact model endpoint can vary, but every run should record the endpoint,
model name, input artifact hashes, generation settings, and output manifest.

## 1. Chunk Local Documents

Input: a DRBench-style task root.

```bash
python3 -m mosaicleaks.data_prep.chunk_local_tasks \
  --tasks-root /path/to/drbench/tasks \
  --output mosaicleaks/data/chunks_local.jsonl
```

Validation:

- every included task has at least one chunk
- chunk offsets are monotonic within each document
- document metadata includes task ID and company name when available
- unsupported or unreadable files are logged, not silently ignored

## 2. Extract Private QA Candidates

Input: `chunks_local.jsonl`.

```bash
python3 -m mosaicleaks.data_prep.extract_local_secrets \
  --chunks mosaicleaks/data/chunks_local.jsonl \
  --output mosaicleaks/data/secret_inventory_raw.jsonl \
  --base-url "$MOSAICLEAKS_LLM_BASE_URL" \
  --model "$MOSAICLEAKS_LLM_MODEL"
```

Validation:

- generated questions are answerable from the local document chunk
- each candidate records source chunk, answer, quote, and secret type
- malformed model outputs are preserved in logs for review

## 3. Filter Private QA Candidates

Input: raw secret inventory.

```bash
python3 -m mosaicleaks.data_prep.filter_local_secrets \
  --input mosaicleaks/data/secret_inventory_raw.jsonl \
  --output mosaicleaks/data/secret_inventory.jsonl \
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
- pass a local JSONL corpus with `docid`, `url`, and `text`

Optional curated task URL sidecar:

```bash
python3 -m mosaicleaks.data_prep.fetch_drbench_urls \
  --tasks-root /path/to/drbench/tasks \
  --output-docs mosaicleaks/data/docs_curated_web.jsonl \
  --output-chunks mosaicleaks/data/chunks_curated_web.jsonl
```

Validation:

- fetched URL rows record HTTP status, extraction method, and task IDs
- only successful direct-fetch rows enter the curated retrieval sidecar
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

Paper tables and plots should read only the published dataset bundle and should
not depend on intermediate generation files. If a figure requires a runtime rollout,
that script belongs in a separate experiment/evaluation area with its own
manifest.
