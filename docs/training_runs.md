# PipelineRL Training Runs

MosaicLeaks training uses PipelineRL with a MosaicLeaks privacy-agent rollout
configuration. PipelineRL launches through a Hydra entrypoint
(`python -m pipelinerl.launch`) and each task provides a problem loader plus a
rollout function. In our setup, the external PipelineRL config owns those
training components, while this repository owns the dataset bundle, retrieval
helper, and MosaicLeaks-specific override values.

The public PipelineRL docs are here:
[ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL).

## 1. Inputs

A training run needs:

- a published MosaicLeaks dataset bundle with `train.jsonl`, `val.jsonl`,
  `test.jsonl`, and `manifest.json`
- a running retrieval helper with the same local-document and BrowseComp indexes
  used during generation
- a PipelineRL checkout or installed package
- a PipelineRL Hydra config named `mosaicleaks`
- a base model path or checkpoint path

The held-out test split is for final reporting. Use `train.jsonl` for training
rollouts and `val.jsonl` for online selection, early stopping, or routine
evaluation during development.

## 2. Start Retrieval

Training should use the same retrieval helper as generation so the privacy
agent sees the same document universe at rollout time. Start the helper before
launching PipelineRL:

```bash
./scripts/start_retrieval_helper.sh \
  --local-index-root mosaicleaks/data/local_indices \
  --embedding-base-url http://127.0.0.1:8000 \
  --browsecomp-index-glob "/path/to/corpus.shard*_of_*.pkl" \
  --browsecomp-corpus Tevatron/browsecomp-plus-corpus
```

Then export the helper URL:

```bash
export MOSAICLEAKS_RETRIEVAL_HELPER_URL=http://127.0.0.1:8012
```

## 3. Configure PipelineRL

Copy the example environment file and fill in local paths:

```bash
cp configs/training.example.env configs/training.local.env
```

The important settings are:

- `MOSAICLEAKS_DATASET_BUNDLE`: published dataset bundle directory
- `MOSAICLEAKS_RETRIEVAL_HELPER_URL`: retrieval helper URL
- `TRAIN_ROOT`: PipelineRL checkout, if PipelineRL is not installed in the
  active environment
- `TRAIN_ENTRYPOINT`: PipelineRL module, usually `pipelinerl.launch`
- `TRAIN_CONFIG_NAME`: Hydra config, usually `mosaicleaks`
- `TRAIN_CONFIG_DIR`: directory containing the MosaicLeaks PipelineRL config
- `MODEL_PATH`: base model or checkpoint to train

The `mosaicleaks` PipelineRL config should point to the privacy-agent rollout
function and the MosaicLeaks problem loader. The launcher passes the dataset
paths as overrides, so the config does not need hard-coded local bundle paths.

## 4. Launch

Inspect the exact command first:

```bash
set -a
source configs/training.local.env
set +a

DRY_RUN=1 ./scripts/launch_training.sh
```

Then launch:

```bash
./scripts/launch_training.sh
```

The launcher passes these PipelineRL/Hydra overrides:

- `train_dataset_names=<bundle>/train.jsonl`
- `test_dataset_names=<bundle>/val.jsonl`
- `mosaicleaks.dataset_bundle=<bundle>`
- `mosaicleaks.final_dataset_path=<bundle>/dataset.jsonl`
- `mosaicleaks.final_train_dataset_path=<bundle>/train.jsonl`
- `mosaicleaks.final_val_dataset_path=<bundle>/val.jsonl`
- `mosaicleaks.final_test_dataset_path=<bundle>/test.jsonl`
- `mosaicleaks.retrieval_helper_url=<helper-url>`

The default MosaicLeaks training choices are:

- `CAPTURE_MODE=planning_only`
- `TRAIN_POLICY_LOSS=gspo`
- `TRAIN_LEARNING_RATE=1e-6`
- `TRAIN_SEQ_LENGTH=16384`
- `TRAIN_MAX_TOKENS=4096`
- `TRAIN_MAX_ROLLOUTS=32`

Put one-off PipelineRL overrides in a file and set
`TRAIN_EXTRA_OVERRIDES_FILE=/path/to/overrides.txt`. Use one Hydra override per
line so the run command stays readable and repeatable.

## 5. Run Record

Every training run should record:

- dataset bundle path and `manifest.json` hash
- MosaicLeaks code commit
- PipelineRL code commit
- model path
- retrieval helper URL and helper manifest
- Hydra config name, config directory, and extra overrides
- output directory
- random seed
- final checkpoint paths
