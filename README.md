# MosaicLeaks

MosaicLeaks contains the dataset-generation and retrieval-helper code for the
MosaicLeaks paper. The repository is intended to be reproducible without hidden
machine paths: all large corpora, DRBench-style task files, BrowseComp-Plus
indexes, and model endpoints are passed through CLI flags or `MOSAICLEAKS_*`
environment variables.

The code uses OpenAI-compatible chat and embedding endpoints. The default model
name is `step-3.5-flash`, but StepFun is not a code dependency; any compatible
endpoint can be used by changing `--model`, `--base-url`, and the relevant API
key environment variable.

## Install

```bash
python3 -m pip install -e ".[browsecomp,server,dev]"
```

Install a spaCy English NER model before running generation:

```bash
python3 -m spacy download en_core_web_trf
```

## External Inputs

MosaicLeaks expects two external data sources:

- A DRBench-style task directory containing `DRxxxx` task folders, task-local
  files, and optional `context.json` / `config/eval.json` metadata.
- BrowseComp-Plus corpus/index artifacts, either through the Hugging Face
  dataset name `Tevatron/browsecomp-plus-corpus` or a local JSONL corpus plus
  embedding index shards.

See [External Dependencies](docs/external_dependencies.md) for the exact
contract.

## Dataset Generation

The high-level pipeline is:

1. Chunk task-local private documents.
2. Extract candidate private QA pairs.
3. Filter private QA pairs with document-only checks.
4. Prepare the web corpus and optional curated task URL sidecar.
5. Build multi-hop chains.
6. Verify, filter, and lightly reword chains.
7. Publish a versioned dataset bundle with deterministic splits and manifests.

See [Dataset Generation](docs/dataset_generation.md) for the full validation
checklist.

Publish final chains as a versioned bundle before training:

```bash
python3 -m mosaicleaks.publish_dataset \
  --input runs/example_generation/chains_final.jsonl \
  --output-dir datasets/mosaicleaks_example \
  --split-key company
```

## Retrieval Helper

The reusable helper service exposes:

- `GET /health`
- `POST /embed`
- `POST /local/search`
- `POST /browsecomp/search`

It can run locally against an OpenAI-compatible embedding endpoint:

```bash
./scripts/start_retrieval_helper.sh \
  --local-index-root /path/to/local_indices \
  --embedding-base-url http://127.0.0.1:8000 \
  --browsecomp-index-glob "/path/to/corpus.shard*_of_*.pkl" \
  --browsecomp-corpus Tevatron/browsecomp-plus-corpus
```

See [Retrieval Helper](docs/retrieval_helper.md) for the index format and API
contract.

## Training Runs

MosaicLeaks does not vendor a trainer. The clean interface is a published
dataset bundle, a retrieval-helper URL, and explicit trainer overrides:

```bash
cp configs/training.example.env configs/training.local.env
set -a
source configs/training.local.env
set +a
DRY_RUN=1 ./scripts/launch_training.sh
```

See [Training Runs](docs/training_runs.md) for the launch contract.
