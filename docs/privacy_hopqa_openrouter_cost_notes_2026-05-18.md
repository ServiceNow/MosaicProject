# Privacy HopQA OpenRouter Rollout Cost Notes

These notes capture the May 18, 2026 working assumptions for running Privacy
HopQA q30 agent rollouts through OpenRouter-hosted closed-source models.

## Harness support

The q30 rollout harness is OpenAI-compatible and can call OpenRouter with:

```bash
export OPENROUTER_API_KEY=...

/home/toolkit/.conda/envs/pipeline-rl/bin/python scripts/run_privacy_hopqa_q30_rollout_worker.py \
  --artifact-root /mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_openrouter_YYYYMMDD \
  --model-label MODEL_LABEL \
  --served-model-name OPENROUTER_MODEL_ID \
  --tokenizer-name TOKENIZER_FOR_LOCAL_COUNTING \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --request-api-mode chat \
  --splits train val test \
  --repeats 1 \
  --helper-url http://ACTIVE_Q30_HELPER:8012
```

The harness normalizes base URLs with or without `/v1`. `--api-key-env` keeps
the API key out of command logs; the runner copies the env value into
`PIPELINERL_LLM_TOKEN` for PipelineRL's request layer.

Use `--request-api-mode chat` for OpenRouter unless a specific model requires
the local chat-template `/v1/completions` path. For hosted closed-source models,
the model ID should be the OpenRouter model name, while `--tokenizer-name` is
only for local prompt length estimates and can be the closest compatible local
tokenizer.

## Dataset scale

The q30 full dataset is 1,001 chains:

| split | chains |
| --- | ---: |
| train | 559 |
| val | 98 |
| test | 344 |
| total | 1,001 |

One "full pass" below means train+val+test once for one agent prompt condition.
Baseline plus privacy-prompted rollouts are two full passes. Three stochastic
attempts are three full passes.

## Token estimate basis

Existing q30 all-split rollout artifacts give the following rough one-pass
successful-rollout token totals. These are model-dependent and were measured
before the latest doc-read window-budget change was smoke-tested.

| Model-like behavior | Input tokens | Output tokens | Notes |
| --- | ---: | ---: | --- |
| Qwen3 4B-like | 143M | 5.8M | compact outputs |
| GPT-OSS-like | 150M | 16M | moderate verbosity |
| Qwen3 8B-like | 149M | 24M | verbose reasoning |
| StepFun-like partial coverage | 90M-125M | 35M-50M | very verbose; coverage was incomplete |

Budget with a conservative closed-source estimate of:

```text
150M input tokens + 20M output tokens per full pass
```

For very verbose reasoning models, use:

```text
150M input tokens + 50M output tokens per full pass
```

The recent PipelineRL change `817bd36` reduces document-reader prompt pressure:
reader evidence is truncated to `reader_excerpt_chars=2500`, default reader
windows are `reader_window_chars=10000`, and selected parent documents now read
all windows only up to `read_all_windows_max_count=5`. This may reduce input
tokens and improve model behavior, but should be re-estimated from a small
current-harness smoke before buying full OpenRouter runs.

## Price assumptions

Prices below are user-provided working prices as of May 18, 2026.

| Model | Input price | Output price |
| --- | ---: | ---: |
| GPT-5.4 | $2.50 / 1M tokens | $15.00 / 1M tokens |
| GPT-5.1 | $1.25 / 1M tokens | $10.00 / 1M tokens |

Formula:

```text
cost = input_tokens_M * input_price_per_M + output_tokens_M * output_price_per_M
```

## Cost estimates

| Scenario | Tokens | GPT-5.4 | GPT-5.1 |
| --- | --- | ---: | ---: |
| One full pass, conservative | 150M in + 20M out | $675 | $388 |
| One full pass, verbose | 150M in + 50M out | $1,125 | $688 |
| Baseline + privacy prompt, conservative | 300M in + 40M out | $1,350 | $775 |
| Baseline + privacy prompt, verbose | 300M in + 100M out | $2,250 | $1,375 |
| Three repeats, one condition, conservative | 450M in + 60M out | $2,025 | $1,163 |
| Three repeats, one condition, verbose | 450M in + 150M out | $3,375 | $2,063 |
| Three repeats x two prompt conditions, conservative | 900M in + 120M out | $4,050 | $2,325 |
| Three repeats x two prompt conditions, verbose | 900M in + 300M out | $6,750 | $4,125 |

Sensitivity:

| Token reduction | GPT-5.4 savings | GPT-5.1 savings |
| --- | ---: | ---: |
| 10M fewer input tokens | $25 | $12.50 |
| 1M fewer output tokens | $15 | $10 |

## Before full runs

1. Run a small Qwen3 4B smoke with the current PipelineRL harness.
2. Compare prompt/output tokens and chain behavior against the previous q30
   all-split artifacts.
3. Decide whether the OpenRouter run should be one full pass or three repeats,
   and whether to run only baseline first before adding the privacy prompt.

## Current-harness Qwen3 4B smoke

Smoke run:

```text
runtime_analysis/privacy_hopqa_q30_current_harness_qwen4b_smoke_20260518
```

Setup:

- model: `Qwen/Qwen3-4B-Instruct-2507`
- local vLLM, 16,384 context
- active q30 RRF helper
- `train`, `val`, and `test`
- first 3 chains per split, 9 chains total
- current PipelineRL at `e176f10`, including doc-read window-budget fix

Results:

| Metric | Value |
| --- | ---: |
| Rollouts | 9 |
| Successful rollouts | 9 |
| Agent/system failures | 0 / 0 |
| Mean latency | 25.9 s |
| Mean prompt tokens / rollout | 141,272 |
| Mean output tokens / rollout | 6,771 |
| Mean total tokens / rollout | 148,043 |
| Extrapolated full-dataset input tokens | 141.4M |
| Extrapolated full-dataset output tokens | 6.8M |
| Mean hop accuracy | 0.688 |
| Strict chain success | 3 / 9 |

Same-chain comparison against the older q30 all1 Qwen3 4B run:

| Comparison | Prompt tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: |
| Current smoke, 9 chains | 1.271M | 0.061M | 1.332M |
| Older all1 run, same 9 chains | 1.452M | 0.061M | 1.513M |
| Delta | -12.4% | -0.1% | -11.9% |

This is a small non-random smoke, but it suggests the current window-budget
changes may lower prompt-side cost by roughly 10-15% without changing output
length much. For OpenRouter budgeting, keep the conservative 150M input + 20M
output estimate unless we run a larger random sample; a current-harness Qwen3
4B-like estimate from this smoke is about 141M input + 7M output for one full
dataset pass.
