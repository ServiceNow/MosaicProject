# MosaicLeaks: Privacy Risks in Querying-in-the-Open for Deep Research Agents

This repository contains the dataset-generation pipeline, retrieval helper, and
training-launch interface used in
[MosaicLeaks: Privacy Risks in Querying-in-the-Open for Deep Research
Agents](https://arxiv.org/abs/2605.30727). It is organized so a developer can
reproduce the MosaicLeaks build or adapt the same local-document plus public-web
generation pattern to another task collection.

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

- Local private documents, provided as `chunks_local.jsonl`. For paper
  replication, build this from a DRBench-style task directory. For other
  datasets, use the generic folder or manifest chunker to convert your document
  collection to the same chunk format.
- BrowseComp-Plus corpus/index artifacts, either through the Hugging Face
  dataset name `Tevatron/browsecomp-plus-corpus` or a local JSONL corpus plus
  embedding index shards. BrowseComp is required for the current generation and
  retrieval-helper flow.

See [External Dependencies](docs/external_dependencies.md) for the required
input files.

## Dataset Generation

The high-level pipeline is:

1. Chunk task-local private documents.
2. Extract candidate private QA pairs.
3. Filter private QA pairs with document-only checks.
4. Prepare the web corpus and optional curated task URL files.
5. Build multi-hop chains.
6. Verify, filter, and lightly reword chains.
7. Publish a versioned dataset bundle with deterministic splits and manifests.

See [Dataset Generation](docs/dataset_generation.md) for more details and the
full validation checklist.

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
python3 -m mosaicleaks.data_prep.build_local_indices \
  --chunks-local mosaicleaks/data/chunks_local.jsonl \
  --output-root mosaicleaks/data/local_indices \
  --embedding-base-url http://127.0.0.1:8000

./scripts/start_retrieval_helper.sh \
  --local-index-root mosaicleaks/data/local_indices \
  --embedding-base-url http://127.0.0.1:8000 \
  --browsecomp-index-glob "/path/to/corpus.shard*_of_*.pkl" \
  --browsecomp-corpus Tevatron/browsecomp-plus-corpus
```

See [Retrieval Helper](docs/retrieval_helper.md) for the index format and API
details.

## Training Runs

We train with PipelineRL using the MosaicLeaks privacy-agent configuration. The
launcher passes the published dataset bundle, retrieval-helper URL, model path,
and MosaicLeaks-specific overrides into `pipelinerl.launch`:

```bash
cp configs/training.example.env configs/training.local.env
set -a
source configs/training.local.env
set +a
DRY_RUN=1 ./scripts/launch_training.sh
```

See [Training Runs](docs/training_runs.md) for the PipelineRL launch interface
and the MosaicLeaks-specific training choices.

## Citation

```bibtex
@misc{gurung2026mosaicleaks,
  title  = {MosaicLeaks: Privacy Risks in Querying-in-the-Open for Deep Research Agents},
  author = {Alexander Gurung and Spandana Gella and Alexandre Drouin and Issam H. Laradji and Perouz Taslakian and Rafael Pardinas},
  year   = {2026},
  eprint = {2605.30727},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2605.30727}
}
```
