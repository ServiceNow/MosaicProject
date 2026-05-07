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

## Dataset and split

The canonical finalized dataset lives in:

- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-05.jsonl`
- `data_files/privacy_hopqa_final_dataset/train.jsonl`
- `data_files/privacy_hopqa_final_dataset/val.jsonl`
- `data_files/privacy_hopqa_final_dataset/test.jsonl`

Split counts:

- train: `559`
- val: `98`
- test: `344`
- total: `1001`

Company split:

- train: `Lee's Market` (`248`), `Elexion Automotive` (`311`)
- val: `Lee's Market` (`42`), `Elexion Automotive` (`56`)
- test: `MediConn Solutions` (`344`)

The test split is a full held-out-company split. Use train for RL training, val
for online evaluation and model selection, and reserve test for final
generalization checks.

PipelineRL entry points:

- Config: `/home/toolkit/my_code/PipelineRL/conf/privacy_hopqa.yaml`
- Domain params: `/home/toolkit/my_code/PipelineRL/conf/domain_params/privacy_hopqa.yaml`
- Dataset loader: `/home/toolkit/my_code/PipelineRL/pipelinerl/domains/privacy_hopqa/dataset.py`
- Reward / answer matching: `/home/toolkit/my_code/PipelineRL/pipelinerl/domains/privacy_hopqa/reward.py`

## EAI CLI

Use the user-local EAI 2 CLI:

```bash
export PATH=/home/toolkit/.local/bin:$PATH
eai --version
```

The system EAI binary at `/usr/bin/eai` was `1.13.1` and failed against the
current EAI API for fresh replica submissions. A user-local copy was upgraded to
`2.0.14` at `/home/toolkit/.local/bin/eai`.

The old Makefile command shape:

```bash
eai job submit --replicas <N> --file multinode_rl.yaml ...
```

is currently not sufficient by itself. With current EAI, the multi-node YAML
must have a non-empty `options.internal-dns.name`, and the replica count should
be materialized into the temporary YAML as `resources.replicas` instead of being
passed with `--replicas`.

## Helper

Pass the active helper URL through `PRIVACY_AGENT_HELPER_URL`. Do not commit
run-specific EAI job IDs, access URLs, or internal DNS hostnames into this doc.

Use the existing helper wrapper:

```bash
cd /home/toolkit/my_code/MosaicProject
export PATH=/home/toolkit/.local/bin:$PATH
ACCOUNT=snow.research.adea JOB_SPEC=privacy_helper_1gpu.yaml ./run_privacy_helper.sh
```

The current intended helper shape is one GPU for the embedding vLLM server, four
API workers, BrowseComp recall `100/100`, and CPU FAISS for BrowseComp dense
retrieval. GPU FAISS wheels tested in this environment saw the H100 but aborted
with CUDA kernel-image errors, so CPU FAISS is the known-good path until a
compatible FAISS build is installed.

Check health before training:

```bash
curl -s "${PRIVACY_AGENT_HELPER_URL}/health"
```

Expected fields:

- `ok: true`
- `embedding_server_ok: true`
- `local_task_count: 101`
- `browsecomp_size: 100195`
- `browsecomp_index_device: cpu`
- `browsecomp_dense_backend: faiss_cpu`

## Current Launch Template

The intended high-level launch path is still:

```bash
cd /home/toolkit/my_code/MosaicProject
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL=http://<helper-internal-dns>:8012

TS=$(date -u +%Y%m%d_%H%M%S)
JOB_NAME=privacy_hopqa_final_qwen3_4b_instruct_2507_r3_${TS} \
WANDB_PROJECT=privacy_hopqa_final_qwen3_4b \
WANDB_GROUP=privacy_hopqa_final_company_holdout \
WANDB_NAME=privacy_hopqa_final_qwen3_4b_instruct_2507_r3_${TS} \
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
REPLICAS=2 \
./run_privacy_hopqa.sh
```

Important details:

- The defaults for dataset paths, helper URL propagation, capture mode,
  filtering, GSPO settings, sequence length, max tokens, and world fractions
  live in `run_privacy_hopqa.sh`.
- To launch the 8B run, use the same template but set
  `MODEL_PATH=/mnt/llmd/base_models/Qwen3-8B`, use an `8b` run name, and use a
  separate W&B project.
- `actor.llm_max_rollouts=64` is the current full-run setting.
- `finetune.seq_length=16384` and
  `vllm_config.vllm_kwargs.max_model_len=16384` keep the rollout context limit
  aligned.
- The script defaults to the finalized Mosaic dataset split:
  `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/{train,val,test}.jsonl`.
- `test.jsonl` is a full held-out-company split for generalization checks. The
  training launch uses `train.jsonl` for training and `val.jsonl` for online
  evaluation; do not use the held-out test split for training-time selection.
- The script intentionally uses `privacy_hopqa.capture_mode=all_calls` so
  retrieved documents remain available for debugging and reward review.

Current EAI submit caveat:

- EAI 2 requires a non-empty `options.internal-dns.name`.
- `run_privacy_hopqa.sh` now handles this by default. It writes a temporary copy
  of `multinode_rl.yaml`, fills in a short DNS name, materializes
  `resources.replicas`, and asks `Makefile` not to pass `--replicas`.
- Set `EAI_REPLICAS_IN_FILE=0` only if the EAI CLI/API path is reverted and the
  old `--replicas --file` behavior is known to work again.

## Why These Fixes Exist

Privacy HopQA rollouts are much larger than ordinary single-answer rollouts because `capture_mode=all_calls` keeps every planner, chooser, reader, and resolver call, plus token ids, rewards, logprobs, reference logprobs, and metadata. Preprocessing expands those traces into RL rows and sends them through a shared-memory queue; one retry run produced a roughly 123 MB preprocessed payload for a 100 MB queue slot. The fixes were to raise the preprocess shared-memory entry size, split oversized preprocessed outputs into multiple queue messages, and filter before queue output when possible.

A separate class of failures came from rollout infrastructure rather than agent behavior. We saw vLLM/backend symptoms including HTTP 500s, DNS/connect failures, helper disconnects, and an `AsyncEngineDeadError` caused by a CUDA unknown error inside a vLLM actor shard. Those should not become low-reward training trajectories, so the privacy HopQA adapter now treats transport errors, timeouts, HTTP 5xx, and `AsyncEngineDead` as infrastructure failures and re-raises them for PipelineRL to handle at the actor/job level.

The empty-output path was also hardened. If a rollout group produces no training texts, the actor skips publishing that group instead of writing an empty list into the stream. Preprocessing also handles empty inputs and worker error messages before applying RL filtering, which avoids misleading dataframe or type errors.

## Run Record Template

Keep run-specific identifiers in local notes or W&B, not in committed docs:

```text
job: <eai-job-id>
name: <eai-job-name>
wandb project: <wandb-project>
output_dir: <run-output-dir>
```

Use the upgraded user-local EAI 2 binary at `/home/toolkit/.local/bin/eai` for
current launches. Do not use `/usr/bin/eai` unless it has also been upgraded.

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
