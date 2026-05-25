# StepFun Judge Validation Sample

This directory contains a reproducible random sample of 200 successful held-out test rollouts from `cross_model`.

Sampling seed: `20260525`

## Files

- `sample_manifest.jsonl`: one slim metadata row per sampled rollout.
- `sample_prompts.jsonl`: replay/debug dataset with prompts, model-visible inputs, hidden judge references, and StepFun outputs without `_stepfun` metadata.
- `trace_manifest.txt`: trace paths for rerunning the leakage-suite v2 evaluator.
- `trace_manifest.jsonl`: trace paths plus sample ids and StepFun labels.
- `stepfun_reference_sample.jsonl`: selected original StepFun rows, usable as the reference file for comparison.
- `summary.json`: sample counts and StepFun-like token estimate.
- `openrouter_smoke_opus47.env`: smoke-test config. This does not contain the API key.

## Token Estimate

If the OpenRouter model uses roughly StepFun-like prompt and completion lengths, this 200-row validation should use approximately:

- Input tokens: 1,224,517
- Output tokens: 1,346,998
- Total tokens: 2,571,515
- Mean per privacy judgment: 12858 tokens
- Estimated Opus 4.7 cost at $5/1M input and $25/1M output: $39.80

These are for adversary + judging only. They do not include agent rollouts.

## Smoke Run

Run a tiny three-row smoke first. This calls OpenRouter, so inspect the config before running:

```bash
cd /home/toolkit/my_code/MosaicProject
source runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_smoke_opus47.env
bash scripts/run_privacy_hopqa_openrouter_judge_smoke.sh
```

The smoke script:

- Uses `/home/toolkit/my_code/openkey.env` for `OPENROUTER_API_KEY`; it does not print the key.
- Defaults to `LIMIT=3`, `SAMPLE_MODE=spread`, `RANDOM_SEED=20260525`, `CONCURRENCY=1`, `MAX_TOKENS=16384`, and refuses `LIMIT>5` unless `ALLOW_LARGE_SMOKE=1`.
- Checks OpenRouter account/key status before and after the model calls.
- Writes candidate rows, errors, a JSON summary, disagreement JSONL, and an HTML report into this sample directory.
- Prints a Rich table with score agreement, binary agreement, score confusion, row-level labels, errors, and raw candidate output snippets for disagreements.

Override the model if needed:

```bash
OPENROUTER_MODEL='<openrouter-model-id>' bash scripts/run_privacy_hopqa_openrouter_judge_smoke.sh
```

Manual equivalent:

```bash
.venv/bin/python scripts/evaluate_privacy_hopqa_leakage_suite_v2.py \
  --artifact-root /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525 \
  --trace-manifest /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/trace_manifest.txt \
  --output /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_smoke_privacy_leakage_suite_v2.jsonl \
  --errors /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_smoke_privacy_leakage_suite_v2_errors.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env-file /home/toolkit/my_code/openkey.env \
  --model '<openrouter-model-id>' \
  --api-mode chat \
  --limit 3 \
  --sample-mode spread \
  --random-seed 20260525 \
  --concurrency 1 \
  --max-tokens 16384 \
  --disable-response-format \
  --continue-on-error
```

## Full 200-Row Run

```bash
.venv/bin/python scripts/evaluate_privacy_hopqa_leakage_suite_v2.py \
  --artifact-root /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525 \
  --trace-manifest /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/trace_manifest.txt \
  --output /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_candidate_privacy_leakage_suite_v2.jsonl \
  --errors /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_candidate_privacy_leakage_suite_v2_errors.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env-file /home/toolkit/my_code/openkey.env \
  --model '<openrouter-model-id>' \
  --api-mode chat \
  --concurrency 2 \
  --max-tokens 16384 \
  --continue-on-error
```

If OpenRouter rejects `response_format`, add `--disable-response-format`.

## Compare

```bash
.venv/bin/python scripts/compare_privacy_hopqa_leakage_v2_judges.py \
  --reference /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/stepfun_reference_sample.jsonl \
  --candidate /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_candidate_privacy_leakage_suite_v2.jsonl \
  --output-disagreements /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_candidate_disagreements.jsonl \
  --output-summary /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525/openrouter_candidate_comparison_summary.json
```
