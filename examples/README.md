# Minimal Example Workflow

This example shows the file shapes used by the MosaicLeaks pipeline without
requiring DRBench, BrowseComp, an LLM endpoint, or PipelineRL. It uses synthetic
documents and manually supplied private QA seeds.

Run commands from the repository root.

## 1. Chunk Local Documents

Convert a folder of private documents into `chunks_local.jsonl`:

```bash
python3 -m mosaicleaks.data_prep.chunk_local_folder \
  --input-root examples/local_docs \
  --task-id example_task \
  --company-name "Example Health" \
  --output examples/output/chunks_local.jsonl \
  --min-words 1 \
  --min-chars 1
```

For multiple task folders or mixed document roots, use a JSONL manifest:

```bash
python3 -m mosaicleaks.data_prep.chunk_local_folder \
  --manifest examples/local_docs_manifest.jsonl \
  --output examples/output/chunks_from_manifest.jsonl \
  --min-words 1 \
  --min-chars 1
```

## 2. Import Manual Private QA Seeds

Manual seeds are useful when a developer already knows which private facts they
want to protect. Each row needs `doc_id`, `question`, and `answer`.

```bash
python3 -m mosaicleaks.data_prep.import_seeds \
  --input examples/seeds.jsonl \
  --chunks-local examples/output/chunks_local.jsonl \
  --output examples/output/secret_inventory.jsonl
```

In a full run, the generation pipeline uses this private QA inventory together
with BrowseComp and optional curated web documents to build multi-hop chains.

## 3. Publish A Tiny Bundle

`examples/chains_final.jsonl` is a tiny synthetic stand-in for the final output
of `python3 -m mosaicleaks.run`.

```bash
python3 -m mosaicleaks.publish_dataset \
  --input examples/chains_final.jsonl \
  --output-dir examples/output/dataset_bundle \
  --dataset-name mosaicleaks_example \
  --split-key company \
  --test-group "Example Retail"
```

The publisher writes:

```text
dataset.jsonl
train.jsonl
val.jsonl
test.jsonl
manifest.json
dataset_stats.json
schema.json
```

## 4. Inspect The Training Command

This dry run does not require PipelineRL. It only prints the command that would
be launched.

```bash
MOSAICLEAKS_DATASET_BUNDLE=examples/output/dataset_bundle \
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
REQUIRE_RETRIEVAL_HELPER=0 \
DRY_RUN=1 \
./scripts/launch_training.sh
```

For a real training run, start the retrieval helper and set
`MOSAICLEAKS_RETRIEVAL_HELPER_URL` before launching.
