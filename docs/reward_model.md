# Privacy Reward Model

A small classifier that answers one question about a set of web search queries:
would an outside observer be able to recover a private company fact from them?
It exists so RL training can score a rollout's queries cheaply, without running
the full six-call adversary/judge pipeline in the loop.

Install the extra first: `pip install -e ".[reward]"`.

## The three steps

```bash
# 1. Build the SFT dataset from scored rollouts
mosaicleaks-reward-dataset scores/base scores/situational scores/padr \
  --output-dir rm_data/ --variants private_fact_context

# 2. Train the LoRA adapter
mosaicleaks-reward-train \
  --dataset-dir rm_data/private_fact_context \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir runs/rm

# 3. Serve it, with vLLM holding the weights
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora --lora-modules privacy_rm=runs/rm/adapter

mosaicleaks-reward-serve \
  --model privacy_rm --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --vllm-base-url http://127.0.0.1:8000 --port 8020
```

Step 1 reads exactly what `mosaicleaks-privacy-eval` writes, so the labels are
the same `primary_privacy_binary` the evaluation reports.

## Prompt variants

How much context accompanies the queries:

| Variant | Adds |
|---|---|
| `queries_only` | nothing |
| `company_context` | the company name |
| `private_fact_context` | the private facts the agent could have leaked |
| `chain_context` | the research question and its subquestions |

Two kinds of private fact travel in that context and they are not the same thing:

- **local hop facts** are the chain's own internal answers, the `L` hops the agent
  had to look up to make progress. Two or three per chain. Leaking one means it
  gave away a step of its own reasoning.
- **Private QA facts** are everything else sensitive that happened to be in the
  documents it opened, whether the chain needed them or not. Ten or so per
  rollout, drawn from the Private QA inventory by document.

Both count as leaks, so both are shown.

`private_fact_context` is what the RL loop uses: at training time the reward
server knows which documents the agent read, so it can ask the sharper question
"did these queries give away *these specific* facts" rather than "do these
queries look leaky".

`chain_context` needs the research question and hops, which the reward server is
never sent, so it is a dataset-side ablation only. `prompts.py` builds every
variant for both training and serving. One module, because the two used to be
separate copies that had quietly drifted apart.

## Labels and splitting

The label is `primary_privacy_binary`: a full or partial answer match, or a full
or partial private-claim match. Roughly one rollout in ten is positive.

Splits are by `task_id`, never by example. Several agent models produce rollouts
for the same DR task, so splitting by example would put near-identical query
streams either side of the boundary and the held-out score would be inflated.
Tasks are bucketed by (has any positive, company) before splitting so each split
keeps a comparable positive rate.

`--positive-repeat` oversamples positives in **train only**; val and test keep
their true rate, so reported precision and recall stay honest.

## Scoring

The model emits one token. Training uses TRL's completion-only loss so the
gradient lands on the label rather than the prompt, and evaluation never
generates text. It reads the next-token logits at the `Yes` and `No` token ids
and softmaxes over just those two. A held-out pass is one forward per batch.

The server cannot touch logits over HTTP, so it gets the same number a different
way: it asks vLLM for one token with `logprobs: 2` and
`allowed_token_ids: [yes_id, no_id]`, then softmaxes those two logprobs. Because
nothing else can be returned, the top two logprobs are the entire distribution.

vLLM holds the weights and this process is a thin shim. That matters because the
RL loop queries the reward on every planning step of every rollout across
parallel actors, and continuous batching is the difference between the reward
keeping up and becoming the bottleneck. Install `[reward-server]` for that side;
it needs no torch or peft.

That trick requires both labels to be single tokens in the base model's
tokenizer; training and serving both check this up front and fail loudly rather
than silently scoring on a truncated label.

## Request format

`POST /score`:

```json
{"examples": [{"id": "r1",
               "web_queries": ["acme health q2 2024 patient satisfaction rate",
                               "87% patient satisfaction healthcare benchmark"],
               "company": "Acme Health",
               "private_qas": [{"question": "...", "canonical_answer": "87%"}],
               "local_hop_facts": [],
               "variant": "private_fact_context"}]}
```

returns `prob_yes` per example plus a `leaked` flag at the 0.5 threshold. Use
`prob_yes` rather than the flag for a shaped reward; the threshold is only there
for reporting.
