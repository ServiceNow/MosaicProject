# MosaicLeaks: Privacy Risks in Querying-in-the-Open for Deep Research Agents

This repository contains the dataset-generation pipeline, retrieval helper,
privacy evaluation, reward model, and training-launch interface used in
[MosaicLeaks: Privacy Risks in Querying-in-the-Open for Deep Research
Agents](https://arxiv.org/abs/2605.30727) (EACL 2027). You can either reproduce
MosaicLeaks or create your own dataset interleaving your own local and web
document sources.

Four pieces, in the order you would use them:

1. **Dataset generation** builds multi-hop questions that interleave private local
   documents with public web documents.
2. **Retrieval helper** serves those documents to an agent at rollout time.
3. **Privacy evaluation** scores a trajectory: could an outside observer
   reconstruct a private fact from the agent's web queries alone?
4. **Reward model** distils those scores into a one-token classifier, so RL can
   price a rollout's queries without running the full judge pipeline in the loop.

The RL trainer itself is not vendored; see [Training Runs](docs/training_runs.md).

The code uses OpenAI-compatible chat and embedding endpoints. The default model
name is `step-3.5-flash`, but StepFun is not a code dependency; any compatible
endpoint can be used by changing `--model`, `--base-url`, and the relevant API
key environment variable.

## Released Data

The chains and the Private QA Set used to score leakage are on Hugging Face as
[`ServiceNow/mosaicleaks`](https://huggingface.co/datasets/ServiceNow/mosaicleaks)
(`chains` and `private_qa` subsets). Document text is not included: local
documents come from DRBench and web documents from BrowseComp-Plus.
[Privacy Evaluation](docs/privacy_evaluation.md#private-qa-from-the-released-dataset)
shows how to turn the `private_qa` subset into the Private QA inventory that the
evaluation and the privacy reward read.

## Install

```bash
python3 -m pip install -e ".[browsecomp,server]"
```

Install a spaCy English NER model before running generation:

```bash
python3 -m spacy download en_core_web_trf
```

## External Inputs

The pipeline uses two external data sources:

- Local private documents, provided as `chunks_local.jsonl`. For paper
  replication, build this from a DRBench-style task directory. For other
  datasets, use the generic folder or manifest chunker to convert your document
  collection to the same chunk format.
- BrowseComp-Plus corpus and index files, either through the Hugging Face
  dataset name `Tevatron/browsecomp-plus-corpus` or a local JSONL corpus plus
  embedding index shards. We use BrowseComp-Plus for the current generation and
  retrieval-helper flow.

See [External Dependencies](docs/external_dependencies.md) for input file
details.

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

For a tiny synthetic walkthrough of chunking local documents, importing manual
QA seeds, publishing a bundle, and inspecting the PipelineRL launch command, see
[examples/README.md](examples/README.md).

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

## Privacy Evaluation

Scores agent trajectories for leakage: an adversary sees only the visible web
queries, a judge compares what it recovered against the gold private facts, and
the result collapses to a 1-5 score.

```bash
mosaicleaks-privacy-eval --traces-root RUN_DIR --dataset BUNDLE_DIR \
  --private-qa-inventory secrets.jsonl --output-dir scores/ \
  --base-url http://127.0.0.1:8000/v1 --model MODEL
mosaicleaks-privacy-aggregate scores/
```

Any OpenAI-compatible endpoint works, local vLLM or OpenRouter. See
[Privacy Evaluation](docs/privacy_evaluation.md).

## Reward Model

Trains a one-token Yes/No classifier over those scores, so RL can price a
rollout's queries without running the full judge pipeline in the loop.

```bash
pip install -e ".[reward]"
mosaicleaks-reward-dataset scores/ --output-dir rm_data/
mosaicleaks-reward-train --dataset-dir rm_data/private_fact_context \
  --base-model BASE --output-dir runs/rm
mosaicleaks-reward-serve --model privacy_rm --base-model BASE
```

See [Reward Model](docs/reward_model.md).

## Training Runs

We train with PipelineRL using the `privacy_hopqa` domain config, which is the
MosaicLeaks privacy-agent rollout used for PA-DR experiments. The launcher
passes the published dataset bundle, retrieval-helper URL, model path, capture
mode, and reward settings into `pipelinerl.launch`:

```bash
cp configs/training.example.env configs/training.local.env
set -a
source configs/training.local.env
set +a
DRY_RUN=1 ./scripts/launch_training.sh
```

The PA-DR models in the paper were trained with the `privacy_agent` branch of
[ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL/tree/privacy_agent);
[Training Runs](docs/training_runs.md#6-reward-families) lists the exact
privacy-reward overrides.

See [Training Runs](docs/training_runs.md) for the PipelineRL launch interface,
capture modes, outcome/situational/privacy rewards, and leakage metrics.

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
