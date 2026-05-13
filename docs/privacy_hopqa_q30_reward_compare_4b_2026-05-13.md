# Privacy HopQA q30 reward comparison, 4B, 2026-05-13

Purpose: compare binary all-hops reward against fractional hop-accuracy reward on the same finalized q30 dataset, with separate helper jobs so helper load/failures do not couple the two runs.

## Code and dataset

- MosaicProject dataset commit: `8d8993d` (`Finalize Privacy HopQA q30 dataset cleanup`)
- PipelineRL prompt commit: `aabcccb` (`Clarify Privacy HopQA planning and evidence prompts`)
- Dataset root: `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset`
- Full dataset: `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.jsonl`
- Train split: `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl`
- Val split: `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_val.jsonl`
- Test split: `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_test.jsonl`

## Shared trainer settings

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Replicas: `2`
- Capture mode: `planning_only`
- WandB project: `privacy_hopqa_q30_reward_compare_4b_20260513`
- WandB group: `privacy_hopqa_q30_reward_compare_4b_20260513140609`
- Answer matching: `privacy_hopqa.answer_match_mode=exact_or_f1`
- Max rollout calls: `actor.llm_max_rollouts=64`
- Max parallel document reads: `privacy_hopqa.max_parallel_doc_reads=10`
- Stage generation caps: `privacy_hopqa.hop_plan_max_tokens=4096`, `privacy_hopqa.doc_choose_max_tokens=4096`, `privacy_hopqa.doc_read_max_tokens=4096`, `privacy_hopqa.hop_resolve_max_tokens=4096`, plus `llm.parameters.max_tokens=4096` and `test_llm.parameters.max_tokens=4096` from `run_privacy_hopqa.sh`.
- Helper timeout: `privacy_hopqa.helper_timeout_s=180`
- Zero-advantage filtering: enabled via `+finetune.rl.filter_zero_advantage_groups=true`

## Jobs

| Role | Job ID | Job name | Initial state | URL / internal URL |
| --- | --- | --- | --- | --- |
| Binary helper | `95c2c979-5a94-473c-82d0-f2afcfaeabf3` | `privacy_helper_q30_reward_binary_4b_20260513140411` | RUNNING at launch check | `http://dns-24e3447c-506e-4b21-92df-156e18db5087-qrb140411:8012` |
| Hop-accuracy helper | `1d129b2a-05b3-419e-96a5-69f8dfec0ab3` | `privacy_helper_q30_reward_hopacc_4b_20260513140411` | RUNNING at launch check | `http://dns-24e3447c-506e-4b21-92df-156e18db5087-qrh140411:8012` |
| Binary trainer | `92a79b95-dee3-4b6b-bbe7-b314af79c03f` | `privacy_hopqa_q30_reward_binary_4b_20260513140609_1778681169_0` | QUEUING at submit | `https://92a79b95-dee3-4b6b-bbe7-b314af79c03f.job.toolkit-sp.yul201.service-now.com` |
| Hop-accuracy trainer | `da2c80da-1747-4e17-b016-338cb39f4a81` | `privacy_hopqa_q30_reward_hopacc_4b_20260513140609_1778681169_0` | QUEUING at submit | `https://da2c80da-1747-4e17-b016-338cb39f4a81.job.toolkit-sp.yul201.service-now.com` |

## Helper launch commands

```bash
cd /home/toolkit/my_code/MosaicProject
export PATH=/home/toolkit/.local/bin:$PATH

ACCOUNT=snow.research.adea \
JOB_SPEC=privacy_helper_2gpu.yaml \
JOB_NAME=privacy_helper_q30_reward_binary_4b_20260513140411 \
DNS_NAME=qrb140411 \
./run_privacy_helper.sh

ACCOUNT=snow.research.adea \
JOB_SPEC=privacy_helper_2gpu.yaml \
JOB_NAME=privacy_helper_q30_reward_hopacc_4b_20260513140411 \
DNS_NAME=qrh140411 \
./run_privacy_helper.sh
```

## Binary reward training command

```bash
cd /home/toolkit/my_code/MosaicProject
export PATH=/home/toolkit/.local/bin:$PATH

PRIVACY_AGENT_HELPER_URL=http://dns-24e3447c-506e-4b21-92df-156e18db5087-qrb140411:8012 \
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
REPLICAS=2 \
CAPTURE_MODE=planning_only \
WANDB_PROJECT=privacy_hopqa_q30_reward_compare_4b_20260513 \
WANDB_GROUP=privacy_hopqa_q30_reward_compare_4b_20260513140609 \
WANDB_NAME=privacy_hopqa_q30_reward_binary_4b_20260513140609 \
JOB_NAME=privacy_hopqa_q30_reward_binary_4b_20260513140609 \
EXTRA_OVERRIDES="privacy_hopqa.reward_mode=all_hops_correct privacy_hopqa.answer_match_mode=exact_or_f1 privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=10 privacy_hopqa.hop_plan_max_tokens=4096 privacy_hopqa.doc_choose_max_tokens=4096 privacy_hopqa.doc_read_max_tokens=4096 privacy_hopqa.hop_resolve_max_tokens=4096 actor.llm_max_rollouts=64" \
./run_privacy_hopqa.sh
```

## Hop-accuracy reward training command

```bash
cd /home/toolkit/my_code/MosaicProject
export PATH=/home/toolkit/.local/bin:$PATH

PRIVACY_AGENT_HELPER_URL=http://dns-24e3447c-506e-4b21-92df-156e18db5087-qrh140411:8012 \
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
REPLICAS=2 \
CAPTURE_MODE=planning_only \
WANDB_PROJECT=privacy_hopqa_q30_reward_compare_4b_20260513 \
WANDB_GROUP=privacy_hopqa_q30_reward_compare_4b_20260513140609 \
WANDB_NAME=privacy_hopqa_q30_reward_hopacc_4b_20260513140609 \
JOB_NAME=privacy_hopqa_q30_reward_hopacc_4b_20260513140609 \
EXTRA_OVERRIDES="privacy_hopqa.reward_mode=hop_accuracy privacy_hopqa.answer_match_mode=exact_or_f1 privacy_hopqa.helper_timeout_s=180 privacy_hopqa.max_parallel_doc_reads=10 privacy_hopqa.hop_plan_max_tokens=4096 privacy_hopqa.doc_choose_max_tokens=4096 privacy_hopqa.doc_read_max_tokens=4096 privacy_hopqa.hop_resolve_max_tokens=4096 actor.llm_max_rollouts=64" \
./run_privacy_hopqa.sh
```
