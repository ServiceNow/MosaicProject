# Privacy HopQA first-web q12 runbook, 2026-05-12

This is the current runbook for the first-web debugging run after the
`variantfix12/questionfix12` dataset cleanup.

## Dataset

Full finalized dataset:

```text
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-12_variantfix12_questionfix12.jsonl
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-12_variantfix12_questionfix12_train.jsonl
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-12_variantfix12_questionfix12_val.jsonl
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-12_variantfix12_questionfix12_test.jsonl
```

First-web-anywhere training dataset:

```text
data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12.jsonl
data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12_train.jsonl
data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12_val.jsonl
data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12_test.jsonl
```

The split still holds out MediConn as the test company. The q12 cleanup did not
change gold answers relative to the accepted q7 baseline; it made
answer-preserving question clarifications and answer-variant expansions.

## Evidence audit summary

Latest incremental audit:

```text
runtime_analysis/privacy_hopqa_questionfix12_2026-05-12/final_incremental_audit_summary.json
```

Headline result when the model is shown the gold-document setup:

```text
strict pass: 3276 / 3403 hops = 96.3%
answerable/dataset-okay if resolver misses are counted separately: 3355 / 3403 = 98.6%
```

Known residuals are retained in:

```text
runtime_analysis/privacy_hopqa_questionfix12_2026-05-12/current_residual_queue_merged_2026-05-12.jsonl
```

Most residuals are reader/resolver misses rather than accepted dataset defects.

## Helper changes relevant to this run

`privacy_helper_2gpu.yaml` launches one 2-GPU helper with:

```text
workers: 16
threadpool tokens: 16
BrowseComp dense retrieval on cuda:1
--browsecomp-max-chars 0
```

`--browsecomp-max-chars 0` means the helper does not truncate BrowseComp parent
text before returning it to the windowing logic. The PipelineRL privacy HopQA
config then uses windowed evidence selection rather than prefix-only snippets.

## Launch helper

Use a fresh helper for this run so helper overload or worker churn is isolated
from older experiments.

```bash
cd /home/toolkit/my_code/MosaicProject

ACCOUNT=snow.research.adea \
JOB_SPEC=/home/toolkit/my_code/MosaicProject/privacy_helper_2gpu.yaml \
JOB_NAME=privacy_helper_firstweb_q12_0512_4b \
DNS_NAME=phfwq1205124b \
./run_privacy_helper.sh
```

The wrapper prints the internal helper URL. Use that URL below as
`PRIVACY_AGENT_HELPER_URL`.

## Launch 4B first-web planning-only trainer

This is the current debugging run requested on 2026-05-12: first-web-anywhere,
Qwen3 4B Instruct 2507, planning-only training texts, `replicas=4`, and
`actor.llm_max_rollouts=64`.

```bash
cd /home/toolkit/my_code/MosaicProject

TS=$(date +%Y%m%d%H%M%S)
JOB="privacy_hopqa_firstweb_q12_qwen3_4b_planning_${TS}"
FW_ROOT=/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset

PIPELINERL_ROOT=/home/toolkit/my_code/PipelineRL \
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/alexg/reason \
JOB_NAME="$JOB" \
WANDB_PROJECT=privacy_hopqa_firstweb_q12_qwen3_4b \
WANDB_GROUP=firstweb_q12_planning_rep4 \
WANDB_NAME="$JOB" \
REPLICAS=4 \
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
CAPTURE_MODE=planning_only \
DATASET_SPLIT_ROOT="$FW_ROOT" \
FINAL_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12.jsonl" \
TRAIN_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12_train.jsonl" \
VAL_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12_val.jsonl" \
TEST_DATASET_PATH="$FW_ROOT/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12_test.jsonl" \
PRIVACY_AGENT_HELPER_URL="http://<helper-internal-dns>:8012" \
FILTER_ZERO_ADVANTAGE_GROUPS=true \
EXTRA_OVERRIDES="privacy_hopqa.reward_mode=all_hops_correct privacy_hopqa.answer_match_mode=exact_or_f1 privacy_hopqa.enable_local_search_bootstrap=false privacy_hopqa.iteration_budget_per_hop=3 privacy_hopqa.helper_timeout_s=180 privacy_hopqa.retrieval_source_chars=0 privacy_hopqa.reader_window_chars=20000 privacy_hopqa.reader_window_overlap_chars=2000 privacy_hopqa.max_windows_per_parent=0 privacy_hopqa.large_parent_window_policy=top_with_neighbors privacy_hopqa.read_all_windows_max_count=5 privacy_hopqa.large_parent_top_windows=4 privacy_hopqa.large_parent_neighbor_windows=1 privacy_hopqa.large_parent_max_read_windows=0 privacy_hopqa.max_parallel_doc_reads=6 actor.llm_max_rollouts=64" \
bash /home/toolkit/my_code/MosaicProject/run_privacy_hopqa.sh
```

## What to monitor

Primary reward-signal checks:

```text
actor/privacy_hopqa/reward_mean
actor/privacy_hopqa/hop_accuracy
actor/privacy_hopqa/final_answer_correct
actor/privacy_hopqa/gold_parent_retrieved_rate
actor/privacy_hopqa/gold_parent_read_rate
actor/privacy_hopqa/gold_parent_read_answer_accuracy
actor/privacy_hopqa/gold_reader_correct_but_final_incorrect_hops
```

Helper-pressure checks:

```text
actor/privacy_hopqa/total_errors
actor/privacy_hopqa/helper_infrastructure_errors
actor/privacy_hopqa/selected_windows_read
actor/privacy_hopqa/large_parent_docs_limited
```

If helper infrastructure errors become frequent, treat the run as infrastructure
limited before interpreting reward trends.
