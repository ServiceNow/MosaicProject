# Privacy HopQA advisor runbook, q30, 2026-05-13

This is the current reproducible launch path for Privacy HopQA q30 training runs.
It covers the three pieces that must stay synchronized: code, dataset files, and helper retrieval artifacts.

## Current source of truth

| Item | Value |
| --- | --- |
| MosaicProject branch | `main` |
| PipelineRL branch | `privacy_agent` |
| Dataset version | `2026-05-13_q30_final_residual_review` |
| Dataset root | `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset` |
| Shared helper artifact root | `/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513` |
| Local index root for helper and fallback | `/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/local_indices_fixeddeps_20260415` |
| BrowseComp index glob | `/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513/browsecomp_indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl` |
| Default helper job spec | `privacy_helper_2gpu.yaml` |
| Default training wrapper | `run_privacy_hopqa.sh` |

The full/train/val/test split is `1001/559/98/344` chains. MediConn Solutions is held out entirely as test.

## Step 1: clean code and dataset state

Use these commands before launching new experiments:

```bash
cd /home/toolkit/my_code/MosaicProject
git checkout main
git pull --ff-only origin main

cd /home/toolkit/my_code/PipelineRL
git checkout privacy_agent
git pull --ff-only origin privacy_agent

cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
python3 scripts/validate_privacy_hopqa_advisor_setup.py
```

Expected validation facts:

| Check | Expected |
| --- | ---: |
| Full dataset rows | 1001 |
| Train rows | 559 |
| Val rows | 98 |
| Test rows | 344 |
| Total hops | 3403 |
| Local index task directories | 101 |
| BrowseComp embedding shards | 4 |

If you make a new dataset version, update these files together:

| File | Why it matters |
| --- | --- |
| `configs/privacy_hopqa_q30_current.env` | Single shell source of truth for paths and default overrides. |
| `run_privacy_hopqa.sh` | Wrapper defaults used by most EAI training launches. |
| `data_files/privacy_hopqa_final_dataset/README.md` | Human-readable dataset provenance. |
| `docs/privacy_hopqa_q30_training_defaults_2026-05-13.md` | Current dataset/split/training metrics note. |
| `docs/privacy_hopqa_advisor_runbook_2026-05-13.md` | Advisor-facing copy/paste commands. |
| `PipelineRL/conf/domain_params/privacy_hopqa.yaml` | PipelineRL defaults when the wrapper is not used. |

If new helper artifacts are required, rebuild or copy them into a new shared directory and update `privacy_helper_2gpu.yaml` plus `configs/privacy_hopqa_q30_current.env`. Do not silently point helpers at untracked local files under `data_files/` or `/home/toolkit/BrowseComp-Plus`.

## Step 2: shared retrieval artifacts

The current helper artifact bundle is already staged at:

```bash
/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513
```

It contains:

| Subpath | Contents |
| --- | --- |
| `artifact_manifest.json` | Checksummed manifest for the helper-facing bundle. |
| `local_indices_fixeddeps_20260415/` | Task-local DRBench document indices and embeddings. |
| `browsecomp_indexes/qwen3-embedding-4b/` | BrowseComp dense shards and BM25 cache. |
| `helper_inputs/` | Long-doc blacklist and curated short web documents/chunks. |

The helper YAML mounts `snow.research.rlar.results:/mnt/llmd/results:ro`, so EAI helper jobs can read this bundle without needing repo-local ignored files.

If the bundle needs to be recreated, use the same structure and then run the validation command above. The important rule is that `privacy_helper_2gpu.yaml`, `run_privacy_hopqa.sh`, `configs/privacy_hopqa_q30_current.env`, and `PipelineRL/conf/domain_params/privacy_hopqa.yaml` must all agree on the shared root.

## Step 3: launch helper jobs

Start one helper per simultaneous training run when comparing reward modes or model sizes. Sharing one helper across multiple trainers can work, but it couples throughput and failure modes.

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH

TS=$(date +%Y%m%d%H%M%S)
HELPER_JOB="privacy_helper_q30_${TS}"
HELPER_DNS="phq30${TS}"

ACCOUNT="$PRIVACY_HOPQA_ACCOUNT" \
JOB_SPEC="$PRIVACY_HOPQA_HELPER_JOB_SPEC" \
JOB_NAME="$HELPER_JOB" \
DNS_NAME="$HELPER_DNS" \
./run_privacy_helper.sh
```

The wrapper prints `internal_url=http://<resolved-name>:8012`. Use that exact URL for `PRIVACY_AGENT_HELPER_URL` in the training command:

```bash
export PRIVACY_AGENT_HELPER_URL="http://<resolved-helper-internal-dns>:8012"
```

Before launching a trainer, optionally check helper health from a node that can route to the internal DNS:

```bash
curl "$PRIVACY_AGENT_HELPER_URL/health"
```

## Step 4: launch training jobs

Common defaults:

| Setting | Default |
| --- | --- |
| Capture mode | `planning_only` |
| Rollouts per group cap | `actor.llm_max_rollouts=64` |
| Max parallel doc reads | `privacy_hopqa.max_parallel_doc_reads=10` |
| Stage max tokens | `4096` for planner, chooser, reader, resolver, train LLM, and test LLM |
| Answer matching | `exact_or_f1` |
| Helper timeout | `180` seconds |
| Zero-advantage filtering | enabled |
| Learning rate | `1e-6` in `run_privacy_hopqa.sh` |
| Policy loss | `gspo` in `run_privacy_hopqa.sh` |
| GSPO clipping | `epsilon_low=3e-3`, `epsilon_high=4e-3` |

### 4B, binary all-hops reward

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_binary_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_BINARY_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

### 4B, fractional hop-accuracy reward

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_hopacc_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_4B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_HOPACC_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

### 8B, binary all-hops reward

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
export PATH=/home/toolkit/.local/bin:$PATH
export PRIVACY_AGENT_HELPER_URL="http://<helper-for-this-run>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_8b_binary_${TS}"

PIPELINERL_ROOT="$PIPELINERL_ROOT" \
OUTPUT_DIR_BASE="$PRIVACY_HOPQA_OUTPUT_DIR_BASE" \
JOB_NAME="$JOB" \
WANDB_PROJECT="$PRIVACY_HOPQA_DEFAULT_WANDB_PROJECT" \
WANDB_GROUP="$PRIVACY_HOPQA_DEFAULT_WANDB_GROUP" \
WANDB_NAME="$JOB" \
REPLICAS="$PRIVACY_HOPQA_REPLICAS" \
MODEL_PATH="$PRIVACY_HOPQA_MODEL_8B" \
CAPTURE_MODE="$PRIVACY_HOPQA_CAPTURE_MODE" \
FILTER_ZERO_ADVANTAGE_GROUPS="$PRIVACY_HOPQA_FILTER_ZERO_ADVANTAGE_GROUPS" \
EXTRA_OVERRIDES="$PRIVACY_HOPQA_BINARY_REWARD_OVERRIDE $PRIVACY_HOPQA_COMMON_EXTRA_OVERRIDES" \
./run_privacy_hopqa.sh
```

## Monitoring

Check job state:

```bash
eai job get <helper-job-id> <trainer-job-id> \
  --fields id,state,name,runs.state,runs.runTime,runs.exitCode \
  --format table
```

Check trainer progress after the job starts writing output:

```bash
OUT=/mnt/llmd/results/exps/alexg/reason/<JOB_NAME>
rg -n "Completed steps|rl/reward|Published weight update request" "$OUT/finetune/log/info_0.log"
rg -n "PrivacyHopQAHelperInfrastructureError|retry [2-9]|failed after|Traceback|Exception" "$OUT/actor/info.log" "$OUT/actor/error.log"
```

A run is not healthy until `finetune/log/info_0.log` shows `Completed steps 1` and a published weight update after that step. Actor-side `PrivacyHopQAHelperInfrastructureError, retry 1` warnings indicate helper pressure. Occasional retry-1 events are tolerable, but retry-2 or final failures need investigation.

## Killing jobs

```bash
eai job kill <trainer-job-id>
eai job kill <helper-job-id>
```

Kill helpers after their paired trainers stop; helpers consume GPUs even when no trainer is using them.
