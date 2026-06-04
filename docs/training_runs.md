# Training Runs

MosaicLeaks does not include a training framework. A training run needs:

1. publish a dataset bundle,
2. start the retrieval helper when the trainer needs live document lookup,
3. launch an external trainer with explicit dataset, model, helper, and output
   paths.

The launcher in this repository prepares those inputs for a PipelineRL-style
trainer or another trainer with a similar command-line override interface.

## 1. Publish A Dataset Bundle

Publish final chains with `mosaicleaks.publish_dataset` before training. See
[Dataset Generation](dataset_generation.md#6-publish-a-versioned-dataset) for
the command, bundle files, and split validation checklist.

## 2. Start Retrieval

Training runs that read documents during rollout should use the same retrieval
helper and index format as generation. See
[Retrieval Helper](retrieval_helper.md) for the API, index files, and launch
command.

Record the helper URL in the training run manifest or experiment tracker:

```bash
export MOSAICLEAKS_RETRIEVAL_HELPER_URL=http://127.0.0.1:8012
```

## 3. Launch Training

Copy the example config and fill in local paths:

```bash
cp configs/training.example.env configs/training.local.env
```

Then launch or inspect the command:

```bash
set -a
source configs/training.local.env
set +a

DRY_RUN=1 ./scripts/launch_training.sh
./scripts/launch_training.sh
```

The launcher targets a PipelineRL-style `python -m <entrypoint>` command by
default, but the same pattern works for another trainer if it accepts explicit
dataset paths and key-value overrides. The important paths passed by the
launcher are:

- `train_dataset_names=<bundle>/train.jsonl`
- `test_dataset_names=<bundle>/val.jsonl`
- `mosaicleaks.final_dataset_path=<bundle>/dataset.jsonl`
- `mosaicleaks.final_train_dataset_path=<bundle>/train.jsonl`
- `mosaicleaks.final_val_dataset_path=<bundle>/val.jsonl`
- `mosaicleaks.final_test_dataset_path=<bundle>/test.jsonl`
- `mosaicleaks.retrieval_helper_url=<helper-url>`

A complete run record should include:

- dataset bundle path and `manifest.json` hash
- code commit
- model path
- helper URL and helper manifest
- trainer config and extra overrides
- output directory
- random seed
- final checkpoint paths

The held-out test split should be reserved for final reporting. Use train for
optimization and val for online selection or early stopping.
