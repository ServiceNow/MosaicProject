# Privacy HopQA multi-objective rewards

This change keeps the existing scalar `TrainingText.reward` path intact and adds optional objective-specific rewards for RL advantage calculation.

## Objective semantics

- `answer`: the existing Privacy HopQA answer reward. It is `outcome` or `prefix_progress` depending on `privacy_hopqa.training_reward_mode`.
- `privacy`: a safety reward from the binary privacy reward model. `No` leakage maps to `1.0`; `Yes` leakage maps to `0.0`.

The privacy objective is only applicable to `hop_plan` traces that parse into one or more `web_search` actions. Non-planning traces and planning traces with no web query have privacy mask `0.0`, so privacy contributes no advantage there.

When `finetune.rl.multi_objective_advantages=true`, PipelineRL computes leave-one-out advantages separately per active objective and combines them with `finetune.rl.reward_objective_weights`. If weights are not set, active objectives are weighted equally. For the first privacy-reward experiment, use:

```bash
finetune.rl.multi_objective_advantages=true \
finetune.rl.reward_objective_weights.answer=0.5 \
finetune.rl.reward_objective_weights.privacy=0.5
```

## Stepwise answer reward

For per-step answer rewards, use the existing prefix-progress mode:

```bash
privacy_hopqa.training_reward_mode=prefix_progress \
privacy_hopqa.stop_after_incorrect_hop=true \
finetune.rl.step_reward_advantages=true \
finetune.rl.pad_step_rewards_for_advantage=true
```

Outcome-only answer reward also supports multi-objective advantages:

```bash
privacy_hopqa.training_reward_mode=outcome \
finetune.rl.step_reward_advantages=false \
finetune.rl.multi_objective_advantages=true
```

In outcome mode, the answer objective is still compared once per rollout, matching the original outcome-reward baseline. The privacy objective remains per applicable planning trace.

## Privacy reward-model helper

Recommended deployable reward model from MosaicProject:

```text
/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515/models/company_context_lora_full_ep2_rep4
```

It is a LoRA on `Qwen/Qwen3-4B-Instruct-2507`. A vLLM OpenAI-compatible helper can be started with:

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8014 \
  --enable-lora \
  --lora-modules privacy_reward=/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_q30_binary_reward_sft_20260515/models/company_context_lora_full_ep2_rep4 \
  --served-model-name privacy_reward \
  --max-model-len 4096
```

Pass the helper URL into training with:

```bash
privacy_hopqa.privacy_reward_service_url=http://<dns-name>:8014 \
privacy_hopqa.privacy_reward_model=privacy_reward
```

The reward model sees only the company name and the visible web query strings for the current planning step. It does not receive the task question, private/local documents, local search results, retrieved documents, or final answers.

## Proposed segment-aligned step advantages

This section is a design target for the next Privacy HopQA reward iteration. It replaces raw global `step_index` alignment with hop/stage-local segment alignment.

### Notation and terms

This section uses `value` for the scalar reward/return assigned before baseline subtraction, and `advantage` only for the leave-one-out normalized training weight.

Indices:

```text
g = rollout group / chain id
r = rollout index within group g
h = hop number in the chain
s = trainable control stage, one of {hop_plan, doc_choose, hop_resolve}
t = real call index within (r,h,s), starting at 0
m = aligned position after right-aligning variable-length call sequences
```

Per-hop quantities:

```text
C_{r,h} = 1 if rollout r answered hop h correctly, else 0
I_{r,h} = number of agent iterations used while working on hop h
E_h = expected source type for hop h from the L/W pattern
Q_{r,h,t} = 1 if hop_plan call t searched E_h, else 0
```

`I_{r,h}` is a hop-local iteration count, not a global rollout length. It is used only to define the raw answer value below; it is not the advantage.

For rollout `r`, hop `h`, stage `s`, and stage-call index `t`:

```text
s in {hop_plan, doc_choose, hop_resolve}
segment(r,h,s) = all real training texts for rollout r, hop h, stage s
n_{r,h,s} = |segment(r,h,s)|
M_{g,h,s} = max_r n_{r,h,s} for group g
```

Right-align each segment to length `M_{g,h,s}`. If a segment is shorter, prefix-pad with its first real value. Padding is used only for baseline/statistics, never for gradient tokens.

For a real call at local index `t`, its aligned position is:

```text
pos(r,h,s,t) = M_{g,h,s} - n_{r,h,s} + t
```

### Answer objective

For `doc_choose` and `hop_resolve`:

```text
V_answer(r,h,s,t) = C_{r,h} / max(I_{r,h}, 1)
```

For `hop_plan`, add source-type partial credit when the hop was not solved:

```text
V_answer(r,h,hop_plan,t) =
    C_{r,h} / max(I_{r,h}, 1)                 if C_{r,h}=1
    source_credit * Q_{r,h,t}                 if C_{r,h}=0
    0                                         otherwise
```

Default:

```text
source_credit = 0.5
```

This is the raw answer value. It is intentionally not called an advantage. The advantage is computed only after comparing this value against other rollout values in the same aligned bucket.

Prefix padding for answer uses the same value definition. If no real value exists because the hop was never reached, mask that rollout from this hop/stage bucket.

The answer baseline bucket is:

```text
B_answer = (group_id, hop_number, stage, aligned_position)
```

The leave-one-out answer advantage for a real text `i` is:

```text
Adv_answer_i =
    V_answer_i
    - mean({V_answer_j : j in B_answer, rollout(j) != rollout(i)})
```

Optionally divide by bucket standard deviation, matching existing RL config behavior.

### Privacy objective

Privacy applies only to `hop_plan` calls with at least one parsed `web_search` query.

Collect web-query planning calls across the entire rollout, not reset per hop:

```text
W_r = [w_{r,0}, w_{r,1}, ..., w_{r,K_r-1}]
```

For each web-plan call `k`, run the privacy classifier on the cumulative visible web-query transcript:

```text
T_{r,k} = all web queries from w_{r,0} through w_{r,k}
L_{r,k} = 1 if T_{r,k} is leaky, else 0
```

Let:

```text
F_r = first k where L_{r,k}=1, or null if no leak
```

The privacy penalty uses discounted distance to the first cumulative leak. This is a raw value definition, not the advantage:

```text
P_{r,k} =
    0                                  if F_r is null
    0                                  if F_r - k > max_back
    gamma^(F_r-k)                      if k <= F_r and F_r-k <= max_back
    post_leak_penalty                  if k > F_r
```

Default:

```text
gamma = 0.75
max_back = 10
post_leak_penalty = 0.25
```

Equivalently, the raw privacy value is:

```text
V_privacy_{r,k} = 1 - P_{r,k}
```

The privacy baseline bucket right-aligns global web-plan sequences:

```text
B_privacy = (group_id, hop_plan, global_web_plan_aligned_position)
```

Prefix padding for privacy uses the same value convention: pad with the first real privacy value for that rollout's right-aligned web-plan sequence. Padding affects baseline/statistics only.

The privacy advantage for a real web-query planning text is:

```text
Adv_privacy_i =
    V_privacy_i
    - mean({V_privacy_j : j in B_privacy, rollout(j) != rollout(i)})
```

### Combining objectives

For each real training text, combine active objective advantages and renormalize over active objectives:

```text
Adv_i = sum_o weight_o * Adv_{o,i} / sum_o weight_o
```

Active objectives:

```text
hop_plan with web_search: answer + privacy
doc_choose: answer only
hop_resolve: answer only
hop_plan without web_search: answer only
```

Default weights when both are active:

```text
answer_weight = 0.5
privacy_weight = 0.5
```

Thus non-web/control texts keep full answer-learning scale, while web-query planning texts trade off task progress against privacy risk.

## Basic examples

### Example 1: full answer-objective pipeline for one local hop

Suppose hop 1 is local (`L`), and three rollout samples were generated for the same chain.

First collect the real trainable control calls. These are the only calls that receive gradients:

```text
R1:
  iter0: hop_plan, doc_choose, hop_resolve
  hop correct: yes

R2:
  iter0: hop_plan, doc_choose, hop_resolve
  iter1: hop_plan, doc_choose, hop_resolve
  hop correct: yes

R3:
  iter0: hop_plan, doc_choose, hop_resolve
  iter1: hop_plan, doc_choose, hop_resolve
  iter2: hop_plan, doc_choose, hop_resolve
  hop correct: no
```

Compute hop-local correctness and iteration values:

```text
R1: C=1, I=1 -> correctness/efficiency value = 1/1 = 1.0
R2: C=1, I=2 -> correctness/efficiency value = 1/2 = 0.5
R3: C=0, I=3 -> correctness/efficiency value = 0
```

Because this is a local hop, the expected source type is `local_document_search`. For `hop_plan`, failed attempts can get source-type partial credit. Assume:

```text
source_credit = 0.5
```

For this example, suppose `R3` searched the wrong source type on iter0, then searched local docs on iter1 and iter2:

```text
R3 hop_plan iter0: Q=0 -> V_answer=0
R3 hop_plan iter1: Q=1 -> V_answer=0.5
R3 hop_plan iter2: Q=1 -> V_answer=0.5
```

For solved rollouts, source partial credit is ignored because the solved-hop value is already used:

```text
R1 hop_plan: [1.0]
R2 hop_plan: [0.5, 0.5]
R3 hop_plan: [0.0, 0.5, 0.5]
```

For `doc_choose` and `hop_resolve`, no source-type partial credit is used:

```text
R1 doc_choose:  [1.0]
R2 doc_choose:  [0.5, 0.5]
R3 doc_choose:  [0.0, 0.0, 0.0]

R1 hop_resolve: [1.0]
R2 hop_resolve: [0.5, 0.5]
R3 hop_resolve: [0.0, 0.0, 0.0]
```

Now align and pad separately for each stage. For `hop_plan`, the longest sequence is length 3:

```text
R1 hop_plan padded: [1.0, 1.0, 1.0]
R2 hop_plan padded: [0.5, 0.5, 0.5]
R3 hop_plan padded: [0.0, 0.5, 0.5]
```

Compute leave-one-out advantages at each aligned position:

```text
position 0 values: R1=1.0, R2=0.5, R3=0.0
  Adv R1 = 1.0 - mean(0.5,0.0) = +0.75
  Adv R2 = 0.5 - mean(1.0,0.0) = 0.0
  Adv R3 = 0.0 - mean(1.0,0.5) = -0.75

position 1 values: R1=1.0, R2=0.5, R3=0.5
  Adv R1 = 1.0 - mean(0.5,0.5) = +0.5
  Adv R2 = 0.5 - mean(1.0,0.5) = -0.25
  Adv R3 = 0.5 - mean(1.0,0.5) = -0.25

position 2 values: R1=1.0, R2=0.5, R3=0.5
  same as position 1
```

Apply advantages only to real texts, never padded texts:

```text
R1 real hop_plan iter0 uses position 2 advantage: +0.5
R2 real hop_plan iter0 uses position 1 advantage: -0.25
R2 real hop_plan iter1 uses position 2 advantage: -0.25
R3 real hop_plan iter0 uses position 0 advantage: -0.75
R3 real hop_plan iter1 uses position 1 advantage: -0.25
R3 real hop_plan iter2 uses position 2 advantage: -0.25
```

`doc_choose` and `hop_resolve` are calculated the same way, but without source partial credit.

### Example 2: full pipeline for a web hop with answer and privacy

Suppose hop 2 is web (`W`). Two rollout samples both reach hop 2.

Real control calls:

```text
R1:
  iter0: hop_plan(web query A), doc_choose, hop_resolve
  hop correct: yes

R2:
  iter0: hop_plan(web query B), doc_choose, hop_resolve
  iter1: hop_plan(web query C), doc_choose, hop_resolve
  hop correct: yes
```

Answer values:

```text
R1: C=1, I=1 -> V_answer=1.0 for hop_plan/doc_choose/hop_resolve
R2: C=1, I=2 -> V_answer=0.5 for hop_plan/doc_choose/hop_resolve
```

For `hop_plan`, answer alignment is:

```text
R1 hop_plan padded: [1.0, 1.0]
R2 hop_plan padded: [0.5, 0.5]
```

Answer advantages:

```text
position 0: R1=1.0, R2=0.5
  Adv_answer R1 = +0.5
  Adv_answer R2 = -0.5

position 1: R1=1.0, R2=0.5
  Adv_answer R1 = +0.5
  Adv_answer R2 = -0.5
```

Now compute privacy only for web-query `hop_plan` calls. The privacy classifier sees cumulative web queries across the whole chain. Suppose `R1` never leaks, and `R2` first leaks at its second web query:

```text
R1 web transcript after query A: safe
R2 web transcript after query B: safe
R2 web transcript after query C: leaky
```

With `gamma=0.75`:

```text
R1 privacy values: [1.0]
R2 privacy values: [1 - 0.75^1, 1 - 0.75^0] = [0.25, 0.0]
```

Right-align privacy sequences:

```text
R1 privacy padded: [1.0, 1.0]
R2 privacy padded: [0.25, 0.0]
```

Privacy advantages:

```text
position 0: R1=1.0, R2=0.25
  Adv_privacy R1 = +0.75
  Adv_privacy R2 = -0.75

position 1: R1=1.0, R2=0.0
  Adv_privacy R1 = +1.0
  Adv_privacy R2 = -1.0
```

Combine active objectives on real web-query `hop_plan` texts:

```text
R1 hop_plan iter0:
  uses aligned position 1
  Adv_answer=+0.5
  Adv_privacy=+1.0
  combined = 0.5*(+0.5) + 0.5*(+1.0) = +0.75

R2 hop_plan iter0:
  uses aligned position 0
  Adv_answer=-0.5
  Adv_privacy=-0.75
  combined = 0.5*(-0.5) + 0.5*(-0.75) = -0.625

R2 hop_plan iter1:
  uses aligned position 1
  Adv_answer=-0.5
  Adv_privacy=-1.0
  combined = 0.5*(-0.5) + 0.5*(-1.0) = -0.75
```

For `doc_choose` and `hop_resolve`, privacy is inactive:

```text
combined advantage = Adv_answer
```

So `R2`'s second web plan is penalized more than its first web plan because it is the call where cumulative mosaic leakage appears.

### Example 3: privacy penalty after a leak

Suppose a rollout has six web-query planning calls across the chain. The cumulative transcript first becomes leaky at call 4.

```text
gamma = 0.75
post_leak_penalty = 0.25
```

The values are:

```text
call:          0      1      2      3      4      5
distance:      4      3      2      1      0      post-leak
penalty:       .316   .422   .562   .750   1.000  .250
V_privacy:     .684   .578   .438   .250   0      .750
```

This treats the first leaking query as worst. Queries shortly before the leak are penalized according to distance. Later queries receive a smaller state penalty because the transcript is already leaky, but they are not treated as the original cause.

### Example 4: failed web hop with source partial credit

Suppose hop 3 is web (`W`) and the rollout fails to answer it. It makes two `hop_plan` calls:

```text
iter0 plan: local_document_search only
iter1 plan: web_search
```

Answer values for `hop_plan`:

```text
iter0: expected source searched? no  -> V_answer=0
iter1: expected source searched? yes -> V_answer=0.5
```

Answer values for `doc_choose` and `hop_resolve`:

```text
hop incorrect -> V_answer=0
```

So the plan that at least tried web search can receive partial credit relative to other failed plans, but downstream `doc_choose` and `hop_resolve` are not credited unless the hop is actually answered correctly.
