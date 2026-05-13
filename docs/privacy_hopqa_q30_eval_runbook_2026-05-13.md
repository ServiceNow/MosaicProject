# Privacy HopQA q30 rollout and leakage evaluation runbook

This is the live q30 evaluation setup for the model/performance/privacy plots.

## Source of truth

Recent q30 training artifacts and the older May 7 training artifacts both use:

```text
privacy_hopqa.choose_top_k: 3
privacy_hopqa.max_iterations: 21   # q30 runs
privacy_hopqa.read_all_windows_max_count: 10
```

The rollout harness follows `choose_top_k=3`. The q30 rollout workers use
`max_tokens=16000` and per-stage generation caps of `16000`; the context window
is capped separately. vLLM jobs are launched without `max-num-seqs`.

Per-model serving exceptions:

```text
gpt_oss_20b:   locally render the chat template and call /v1/completions, with
               temperature=0.7, to avoid known GPT-OSS/Harmony chat-parser
               failures under vLLM.
gemma4_31b_it: uses an isolated /tmp/gemma4_vllm_env overlay with vLLM 0.19.1
               and current Transformers, because the shared PipelineRL env has
               vLLM 0.18.1 / Transformers 4.57.6 and cannot load model_type=gemma4.
```

## Durable artifact root

```text
/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_model_rollouts_20260513
```

Layout:

```text
model_runs/<model_label>/
  config/worker_metadata.json
  logs/
  rollouts/<model_label>/<split>/privacy_hopqa_runs/*/trace.json
  tables/rollout_index.jsonl
  tables/privacy_leakage_suite_v2_stepfun_full.jsonl
  tables/privacy_leakage_suite_v2_qwen3_4b_full.jsonl
  tables/privacy_binary_leakage_predictor_qwen3_4b.jsonl
```

`rollout_index.jsonl` now records per-rollout `prompt_tokens`,
`output_tokens`, and `total_tokens` from PipelineRL `TrainingText` records.

## Jobs

Submit or resubmit the full set:

```bash
python scripts/submit_privacy_hopqa_q30_eval_jobs.py --push-viewer
```

Resubmit only selected rollout shards:

```bash
python scripts/submit_privacy_hopqa_q30_eval_jobs.py \
  --skip-privacy-eval \
  --job-name-suffix _rerun1 \
  --only-model-label gpt_oss_20b \
  --only-model-label gemma4_31b_it
```

Rollout workers:

```text
qwen3_4b_instruct_2507: train val test
qwen3_8b:                train val test
step35_flash:            train val test
gpt_oss_20b:             test
gemma4_31b_it:           test
```

The privacy-eval monitor starts Qwen3-4B as a local judge, then repeatedly runs:

```text
StepFun leakage suite v2
Qwen3-4B leakage suite v2
Qwen3-4B web-query-only binary predictor against StepFun score >= 4
```

All evaluators resume from their output JSONL files.

## Viewer

Build once:

```bash
python scripts/build_privacy_hopqa_q30_live_analysis_viewer.py \
  --artifact-root /mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_model_rollouts_20260513 \
  --output-dir runtime_analysis/privacy_hopqa_q30_live_analysis \
  --latest-output runtime_analysis/latest_privacy_hopqa_q30_live_analysis.html
```

Open:

```text
runtime_analysis/privacy_hopqa_q30_live_analysis/index.html
runtime_analysis/latest_privacy_hopqa_q30_live_analysis.html
```

The monitor can commit and push viewer updates with `--push-viewer`, so pulling
MosaicProject is enough to see the latest committed dashboard snapshot.

## Resume notes

EAI jobs are preemptable and restartable. If a rollout job restarts, the worker
reruns the same fixed model-run artifact root and the harness skips completed
`(model_label, split, chain_id, repeat)` rows. If a job is cancelled rather than
preempted, retry it with EAI or rerun the submitter; existing output files are
the resume state.

The privacy monitor is also restartable. It can be rerun at any point and will
only append missing judge rows.
