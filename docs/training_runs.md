# PipelineRL Training Runs

MosaicLeaks training uses PipelineRL with the `privacy_hopqa` domain config.
That name comes from the original internal task name; it is the PipelineRL
adapter for the MosaicLeaks privacy-agent rollouts described in the paper.

The public PipelineRL docs are here:
[ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL).

## 1. What PipelineRL Runs

The PipelineRL side owns the research-agent rollout and reward code. This repo
provides the published dataset bundle, retrieval helper, and launch overrides.

Each rollout works through the MosaicLeaks chain one hop at a time:

- `hop_plan`: decide whether to search local documents, search BrowseComp, or
  stop searching for the current hop
- `doc_choose`: choose which retrieved document or document window to inspect
- `doc_read`: read the selected document content
- `hop_resolve`: answer the current hop before moving to the next hop

The launcher defaults to `CAPTURE_MODE=planning_only`, which trains the
planning calls. `CAPTURE_MODE=all_calls` also trains chooser, reader, and
resolver calls, but those traces are much larger and were mainly useful for
debugging and ablations.

## 2. Inputs

A training run needs:

- a published MosaicLeaks dataset bundle with `dataset.jsonl`, `train.jsonl`,
  `val.jsonl`, `test.jsonl`, and `manifest.json`
- a running retrieval helper with the same local-document and BrowseComp indexes
  used during generation
- a PipelineRL checkout or installed package
- a PipelineRL Hydra config named `privacy_hopqa`
- a base model path or checkpoint path

Use `train.jsonl` for RL training and `val.jsonl` for routine PipelineRL
evaluation. Keep `test.jsonl` held out for final reporting. For paper-style
replication, the split key should hold out a full company or similarly coherent
group, not random individual chains.

## 3. Start Retrieval

Training should use the same retrieval helper as generation so the agent sees
the same document universe during rollout. RRF is the recommended default for
both local and BrowseComp retrieval. Before a serious run, check that the local
index covers every local gold document in the published dataset.

```bash
./scripts/start_retrieval_helper.sh \
  --local-index-root mosaicleaks/data/local_indices \
  --embedding-base-url http://127.0.0.1:8000 \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --browsecomp-index-glob "/path/to/corpus.shard*_of_*.pkl" \
  --browsecomp-corpus Tevatron/browsecomp-plus-corpus \
  --browsecomp-rank-mode rrf \
  --local-rank-mode rrf
```

Then export the helper URL:

```bash
export MOSAICLEAKS_RETRIEVAL_HELPER_URL=http://127.0.0.1:8012
```

For long BrowseComp parents, use the helper's document-window controls instead
of letting rollouts build prompts that exceed the model context:
`--browsecomp-max-doc-windows`, `--browsecomp-reader-window-chars`,
`--browsecomp-reader-window-overlap-chars`, and
`--browsecomp-exclude-doc-ids-file`.

## 4. Configure PipelineRL

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
- `TRAIN_CONFIG_NAME`: Hydra config, usually `privacy_hopqa`
- `TRAIN_CONFIG_DIR`: directory containing the PipelineRL configs
- `PIPELINERL_DOMAIN`: override prefix, usually `privacy_hopqa`
- `MODEL_PATH`: base model or checkpoint to train

The `privacy_hopqa` PipelineRL config should point to the MosaicLeaks
privacy-agent rollout and problem loader. The launcher passes the dataset paths
as overrides, so the config should not hard-code local bundle paths.

## 5. Launch

Inspect the launch command first:

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
- `privacy_hopqa.final_dataset_path=<bundle>/dataset.jsonl`
- `privacy_hopqa.final_train_dataset_path=<bundle>/train.jsonl`
- `privacy_hopqa.final_val_dataset_path=<bundle>/val.jsonl`
- `privacy_hopqa.final_test_dataset_path=<bundle>/test.jsonl`
- `privacy_hopqa.helper_service_url=<helper-url>`
- `privacy_hopqa.capture_mode=<capture-mode>`

The default MosaicLeaks training choices are:

- `CAPTURE_MODE=planning_only`
- `TRAIN_REWARD_MODE=all_hops_correct`
- `TRAINING_REWARD_MODE=outcome`
- `ANSWER_MATCH_MODE=exact_or_f1`
- `FILTER_ZERO_ADVANTAGE_GROUPS=true`
- `HELPER_TIMEOUT_S=180`
- `MAX_PARALLEL_DOC_READS=5`
- `TRAIN_POLICY_LOSS=gspo`
- `TRAIN_LEARNING_RATE=1e-6`
- `TRAIN_SEQ_LENGTH=16384`
- `TRAIN_MAX_TOKENS=4096`
- `TRAIN_MAX_ROLLOUTS=32`

Use `TRAIN_EXTRA_OVERRIDES_FILE=/path/to/overrides.txt` for one-off PipelineRL
settings. Put one Hydra override per line.

## 6. Reward Families

The paper reports three kinds of training signal. Keep them separate in run
names and experiment notes because they answer different questions.

Outcome or output task reward:

- use `TRAIN_REWARD_MODE=all_hops_correct` for a binary final-chain reward
- use `TRAIN_REWARD_MODE=hop_accuracy` when partial hop correctness should
  matter
- keep `TRAINING_REWARD_MODE=outcome` when the reward should be assigned at the
  rollout outcome level

Situational task rewards:

- use PipelineRL overrides such as
  `privacy_hopqa.training_reward_mode=prefix_progress` or
  `privacy_hopqa.training_reward_mode=hop_step`
- these rewards give denser credit to agent stages such as planning the right
  source type, choosing the gold document when it is visible, and resolving the
  current hop
- common supporting overrides are
  `privacy_hopqa.stop_after_incorrect_hop=true`,
  `finetune.rl.step_reward_advantages=true`, and
  `finetune.rl.pad_step_rewards_for_advantage=true`

Privacy rewards:

- use a learned privacy reward model served through an OpenAI-compatible helper
- set `PRIVACY_REWARD_SERVICE_URL=http://<host>:<port>` and
  `PRIVACY_REWARD_MODEL=privacy_reward`
- the privacy reward applies to `hop_plan` traces with web queries; it should
  judge visible web-query text, not private documents or final answers
- combine answer and privacy objectives with PipelineRL overrides such as
  `finetune.rl.multi_objective_advantages=true`,
  `finetune.rl.reward_objective_weights.answer=0.5`, and
  `finetune.rl.reward_objective_weights.privacy=0.5`

Example situational override file:

```text
privacy_hopqa.reward_mode=hop_accuracy
privacy_hopqa.training_reward_mode=hop_step
privacy_hopqa.hop_step_efficiency_metric=llm_calls
privacy_hopqa.stop_after_incorrect_hop=true
privacy_hopqa.zero_error_step_reward=true
finetune.rl.step_reward_advantages=true
finetune.rl.pad_step_rewards_for_advantage=true
```

## 7. Metrics

Track task performance and privacy separately.

Task metrics:

- strict chain success
- hop accuracy
- `reward_by_pattern/<pattern>` and `success_by_pattern/<pattern>`, such as
  `reward_by_pattern/LLW_mean`
- `reward_by_chain_length/<n>` and `success_by_chain_length/<n>`, such as
  `reward_by_chain_length/3_mean`

Privacy metrics:

- intent leakage: whether web queries reveal the private research intent
- answer leakage: whether web queries reveal answers to specific private
  questions
- full-information leakage: whether web queries allow verifiable private claims
  about the enterprise documents

The paper reports privacy leakage from an adversary that observes only external
web queries. Final-answer correctness alone is not enough to evaluate a
MosaicLeaks-trained agent.

## 8. Run Record

Every training run should record:

- dataset bundle path and `manifest.json` hash
- MosaicLeaks code commit
- PipelineRL code commit
- model path
- retrieval helper URL and helper manifest
- capture mode and reward family
- privacy reward model path or helper URL, when used
- Hydra config name, config directory, and extra overrides
- output directory
- random seed
- final checkpoint paths
