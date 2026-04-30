# Privacy HopQA Training

This note documents the current PipelineRL launch path for Privacy HopQA runs.
The intent is to keep the exact operational choices visible, not to add another
launcher layer.

## Files

- `run_privacy_hopqa.sh` builds the `pipelinerl.launch` command from
  environment variables and submits through the Makefile.
- `Makefile` owns the `multi-replica-job` EAI submission target.
- `multinode_rl.yaml` is the EAI multi-node resource spec used by default.
- PipelineRL code and Hydra configs are mounted from
  `/home/toolkit/my_code/PipelineRL` with `SNAPSHOT=0`.

## Helper

Pass the active helper URL through `PRIVACY_AGENT_HELPER_URL`. Do not commit
run-specific EAI job IDs, access URLs, or internal DNS hostnames into this doc.

The intended helper shape is 16 API workers, 2 GPUs, BrowseComp recall
`100/100`, and dense retrieval on `cuda:1` via the torch fallback because
faiss-gpu bindings are unavailable in the image.

## Current Launch Template

Use a fresh timestamp for the job, output directory, and W&B run name:

```bash
cd /home/toolkit/my_code/MosaicProject

TS=$(date -u +%Y%m%d_%H%M%S)

PRIVACY_AGENT_HELPER_URL="${PRIVACY_AGENT_HELPER_URL}" \
JOB_NAME=privacy_hopqa_qwen3_8b_async_roll32_retry3_${TS} \
WANDB_PROJECT=privacy_hopqa_qwen3_8b_roll32_debug \
WANDB_GROUP=privacy_hopqa_variants \
WANDB_NAME=privacy_hopqa_qwen3_8b_async_roll32_retry3_${TS} \
EXTRA_OVERRIDES="actor.llm_max_rollouts=32 finetune.seq_length=20480 vllm_config.vllm_kwargs.max_model_len=20480" \
./run_privacy_hopqa.sh
```

Important details:

- `actor.llm_max_rollouts=32` reduces vLLM/helper pressure without changing the
  agent retrieval/action budgets.
- `finetune.seq_length=20480` and
  `vllm_config.vllm_kwargs.max_model_len=20480` keep the rollout context limit
  aligned.
- The script already defaults to the by-docs split:
  `/home/toolkit/nice_code/drbench/chain_review/datasets/privacy_hopqa_userreviewed_bydocs_v1/{train,val}.jsonl`.
- The script intentionally uses `privacy_hopqa.capture_mode=all_calls` so
  retrieved documents remain available for debugging and reward review.

## Run Record Template

Keep run-specific identifiers in local notes or W&B, not in committed docs:

```text
job: <eai-job-id>
name: <eai-job-name>
wandb project: <wandb-project>
output_dir: <run-output-dir>
```

The local EAI CLI is currently `1.13.1`, which supports `--replicas` but not
`--replica-failures-allowed`. Do not add that flag back unless the interactive
environment is upgraded and the launch path is re-tested.

## Monitoring

Basic status:

```bash
eai job get "$JOB_ID" \
  --fields id,state,name,runs.state,runs.stateInfo,runs.startedOn,accessUrl \
  --format table
```

Startup logs:

```bash
eai job log "$JOB_ID" --tail 200
```

Watch for:

- healthy scheduler/startup messages from PipelineRL
- vLLM actor server startup on the expected ports
- W&B initialization under the expected project and run name
- rollout or train-step progress
- early failures such as `AsyncEngineDead`, `empty training texts`,
  `CUDA out of memory`, helper connection failures, or repeated HTTP 5xx

The background monitor refreshes the full analysis bundle and a compact
single-file viewer every 30 minutes using `scripts/monitor_privacy_hopqa_runtime.sh`.
The compact viewer is the default artifact to copy locally for quick inspection,
and the path is stable across refreshes:

```bash
scp toolkit@localhost:/home/toolkit/my_code/MosaicProject/runtime_analysis/latest_privacy_hopqa_slim_viewer.html ./privacy_hopqa_viewer.html
open ./privacy_hopqa_viewer.html
```

The full bundle remains at
`runtime_analysis/privacy_hopqa_qwen3_8b_async_roll32_retry3_20260429_205615_live/`
when complete CSVs, plots, or per-rollout artifacts are needed. The slim viewer
is built by `scripts/build_privacy_hopqa_slim_viewer.py` from that bundle and
embeds the key plots, token/action summaries, early/middle/late behavior
distribution plots, source-usage comparisons, moving-average plots, timeline
summaries, and a small representative sample of rollouts. The monitor defaults
to 128 rollout examples per early/middle/late window and a 256 rollout moving
average window; override with `TIME_SLICE_SIZE` or `MOVING_AVERAGE_WINDOW` if a
different view is needed. Temporal windows are labeled by completed-rollout
index, for example `Early (0-127)`, and include side-by-side source usage,
searches-vs-docs, and hop/source alignment plots.
The question-type section groups by local/web hop composition and hop pattern,
including retrieval-alignment bubbles, local-hop sensitivity, source-mix
matching, and missing-source retrieval checks.
