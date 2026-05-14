# Privacy HopQA retrieval RRF audit, 2026-05-14

This note documents the retrieval-only audit and local Qwen replay used to decide whether to test reciprocal-rank fusion in the Privacy HopQA helper.

## What changed

The helper now supports explicit rank modes:

```bash
--browsecomp-rank-mode {hybrid,rrf,bm25}
--browsecomp-rrf-k 20
--local-rank-mode {dense,rrf,bm25}
--local-bm25-recall-k 100
--local-dense-recall-k 100
--local-rrf-k 20
```

The default helper behavior is unchanged unless these flags are passed. Use `privacy_helper_2gpu_rrf.yaml` to launch the RRF variant.

RRF score:

```text
score(doc) = dense_weight / (rrf_k + dense_rank(doc)) + lexical_weight / (rrf_k + lexical_rank(doc))
```

The implemented live mode uses equal weights and `rrf_k=20`.

## Current training-side document read default

PipelineRL and the runbook now use:

```text
privacy_hopqa.max_parallel_doc_reads=5
```

This limits concurrent document-reader calls. It does not change `choose_top_k=3`, `retrieval_top_k=5`, or `max_parallel_retrieval_actions=3`.

## Larger retrieval-only audit

Command:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/audit_privacy_hopqa_retrieval_methods.py \
  --sample-size 240 \
  --max-traces 5000 \
  --dense-k 50 \
  --max-model-queries-per-hop 20 \
  --workers 24 \
  --out-dir runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14_large240
```

Inputs:

```text
run_root=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_4b_prefix_step_20260513173122
dataset=data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl
sample_size=240
local_hops=122
web_hops=118
```

The audit uses actual model search queries from traces. `current` is the existing helper ranking for that source, `lexical` is BM25-style lexical retrieval against the same corpus, and `fusion` is equal-weight RRF.

### Overall actual-query gold parent rank

| method | top1 | top5 | top10 | top20 |
| --- | ---: | ---: | ---: | ---: |
| current | 18.8% | 48.3% | 68.8% | 79.2% |
| lexical | 75.4% | 84.6% | 88.3% | 90.0% |
| fusion | 64.2% | 85.8% | 90.8% | 94.2% |

### By source

| source | method | top5 | top10 | top20 |
| --- | --- | ---: | ---: | ---: |
| local | current | 46.7% | 65.6% | 82.0% |
| local | lexical | 89.3% | 91.8% | 91.8% |
| local | fusion | 83.6% | 89.3% | 91.8% |
| web | current | 50.0% | 72.0% | 76.3% |
| web | lexical | 79.7% | 84.7% | 88.1% |
| web | fusion | 88.1% | 92.4% | 96.6% |

### By source and difficulty

Values are top5/top10/top20.

| source | difficulty | current | lexical | fusion |
| --- | --- | ---: | ---: | ---: |
| local | easy | 81.2 / 96.9 / 100.0 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 |
| local | medium | 53.3 / 76.7 / 86.7 | 100.0 / 100.0 / 100.0 | 93.3 / 93.3 / 100.0 |
| local | hard-answer | 33.3 / 56.7 / 76.7 | 83.3 / 90.0 / 90.0 | 76.7 / 86.7 / 90.0 |
| local | hard-retrieval | 16.7 / 30.0 / 63.3 | 73.3 / 76.7 / 76.7 | 63.3 / 76.7 / 76.7 |
| web | easy | 86.7 / 93.3 / 93.3 | 73.3 / 73.3 / 76.7 | 93.3 / 96.7 / 100.0 |
| web | medium | 73.3 / 90.0 / 93.3 | 76.7 / 83.3 / 83.3 | 93.3 / 96.7 / 100.0 |
| web | hard-answer | 35.7 / 53.6 / 60.7 | 85.7 / 96.4 / 100.0 | 85.7 / 92.9 / 96.4 |
| web | hard-retrieval | 3.3 / 50.0 / 56.7 | 83.3 / 86.7 / 93.3 | 80.0 / 83.3 / 90.0 |

Interpretation:

- RRF is clearly better than the current web ranking on this sample.
- Lexical-only is very strong for local retrieval, but RRF is safer as a single helper mode because it preserves dense signal and is much better than current on web.
- The biggest current bottleneck is hard-retrieval, especially web hard-retrieval.

Viewer:

```text
runtime_analysis/privacy_hopqa_retrieval_method_audit_2026-05-14_large240/index.html
```

## Local Qwen replay

This replay asks local `Qwen/Qwen3-4B-Instruct-2507` to answer with the production doc-reader and resolver prompt builders using the top-5 documents from current ranking versus RRF ranking. It uses observed model queries and gold previous-hop answers to isolate retrieval/reader effects.

Start the local model:

```bash
source ~/initmamba.sh
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 32768
```

Run the replay:

```bash
cd /home/toolkit/my_code/MosaicProject
python3 scripts/replay_privacy_hopqa_rrf_retrieval_qwen.py \
  --per-bucket 5 \
  --out-dir runtime_analysis/privacy_hopqa_rrf_replay_qwen4b_2026-05-14_large40
```

Results:

| bucket | n | current match | RRF match | current gold top5 | RRF gold top5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local easy | 5 | 5 | 5 | 5 | 5 |
| local medium | 5 | 5 | 3 | 5 | 5 |
| local hard-answer | 5 | 3 | 4 | 4 | 5 |
| local hard-retrieval | 5 | 0 | 3 | 2 | 5 |
| web easy | 5 | 5 | 5 | 5 | 5 |
| web medium | 5 | 3 | 3 | 5 | 5 |
| web hard-answer | 5 | 2 | 2 | 5 | 5 |
| web hard-retrieval | 5 | 0 | 4 | 1 | 5 |

Interpretation:

- RRF substantially helps when the current ranking does not surface the gold parent in the top 5.
- RRF is not a strict row-by-row dominance result; local medium had two cases where current answered and RRF did not despite both surfacing the gold parent. That points to reader/resolver variance and candidate-set composition, not just retrieval rank.
- The replay is a diagnostic, not a replacement for a live helper experiment.

Viewer:

```text
runtime_analysis/privacy_hopqa_rrf_replay_qwen4b_2026-05-14_large40/index.html
```

## Recommended live experiment

Launch one RRF helper:

```bash
cd /home/toolkit/my_code/MosaicProject
ACCOUNT=snow.research.adea JOB_SPEC=privacy_helper_2gpu_rrf.yaml ./run_privacy_helper.sh
```

Run a comparable 4B planning-only trainer against that helper:

```bash
cd /home/toolkit/my_code/MosaicProject
export PRIVACY_AGENT_HELPER_URL="http://<rrf-helper-internal-dns>:8012"

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_q30_4b_rrf_prefix_step_${TS}"

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_q30_retrieval_rrf \
WANDB_GROUP=q30_full_rrf \
WANDB_NAME="$JOB" \
REPLICAS=2 \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
CAPTURE_MODE=planning_only \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=5 actor.llm_max_rollouts=64 privacy_hopqa.training_reward_mode=prefix_progress privacy_hopqa.stop_after_incorrect_hop=true finetune.rl.step_reward_advantages=true finetune.rl.pad_step_rewards_for_advantage=true" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

Monitor:

- gold parent retrieved/read rates
- reward by pattern and chain length
- parse errors
- helper timeout/error rate
- whether hard-retrieval chains start receiving non-zero advantage
