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

### Example 1: answer value and padding for one hop/stage

Suppose three rollouts are answering the same hop, and we are looking only at `hop_plan` calls for that hop.

```text
R1: solved the hop in 1 iteration, made 1 hop_plan call
R2: solved the hop in 2 iterations, made 2 hop_plan calls
R3: did not solve the hop, made 3 hop_plan calls
```

The raw answer value is:

```text
correct in 1 iteration -> 1 / 1 = 1.0
correct in 2 iterations -> 1 / 2 = 0.5
incorrect -> 0
```

The real sequences are:

```text
R1: [1.0]
R2: [0.5, 0.5]
R3: [0.0, 0.0, 0.0]
```

The longest sequence has length 3, so we right-align and prefix-pad shorter sequences with their first real value:

```text
R1: [1.0, 1.0, 1.0]
R2: [0.5, 0.5, 0.5]
R3: [0.0, 0.0, 0.0]
```

These padded values are only used to compute baseline statistics. We do not train on the padded entries. The real `R1` call uses the advantage from the rightmost aligned position.

### Example 2: source-type partial credit for failed hop planning

Suppose hop 2 is a web hop (`W`), so the expected source type is `web_search`.

```text
source_credit = 0.5
```

Three failed `hop_plan` calls:

```text
Plan A: local_document_search only -> Q=0 -> V_answer=0
Plan B: web_search only -> Q=1 -> V_answer=0.5
Plan C: web_search + local_document_search -> Q=1 -> V_answer=0.5
```

If the hop is eventually correct, source partial credit is not used; the value is the correctness/efficiency value:

```text
correct in 2 iterations -> V_answer = 0.5 for the hop's control calls
```

This means source credit only distinguishes failures: searching the right type of source is better than searching the wrong type, but still worse than solving the hop.

### Example 3: leave-one-out advantage

Using the rightmost aligned position from Example 1:

```text
R1 value = 1.0
R2 value = 0.5
R3 value = 0.0
```

Leave-one-out baselines:

```text
baseline for R1 = mean(0.5, 0.0) = 0.25
baseline for R2 = mean(1.0, 0.0) = 0.5
baseline for R3 = mean(1.0, 0.5) = 0.75
```

Advantages:

```text
R1 advantage = 1.0 - 0.25 = +0.75
R2 advantage = 0.5 - 0.5 = 0.0
R3 advantage = 0.0 - 0.75 = -0.75
```

So the fast successful rollout is reinforced, the slower successful rollout is neutral here, and the failed rollout is downweighted.

### Example 4: privacy penalty around a mosaic leak

Suppose a rollout has six web-query planning calls across the chain. The cumulative transcript first becomes leaky at call 4.

```text
gamma = 0.75
post_leak_penalty = 0.25
```

Distances to first leak:

```text
call:      0      1      2      3      4      5
distance:  4      3      2      1      0      post-leak
```

Penalties:

```text
call 0: 0.75^4 = 0.316
call 1: 0.75^3 = 0.422
call 2: 0.75^2 = 0.562
call 3: 0.75^1 = 0.750
call 4: 0.75^0 = 1.000
call 5: post_leak_penalty = 0.250
```

Privacy values are `1 - penalty`:

```text
call:            0      1      2      3      4      5
V_privacy:       .684   .578   .438   .250   0      .750
```

This gives the leak-causing call the lowest privacy value, penalizes nearby earlier calls, and keeps a smaller state penalty after the transcript is already leaky.

### Example 5: combining answer and privacy

For a web-query `hop_plan` call, both objectives are active:

```text
answer_weight = 0.5
privacy_weight = 0.5
```

If:

```text
Adv_answer = +0.6
Adv_privacy = -0.2
```

then:

```text
combined_advantage = 0.5*(+0.6) + 0.5*(-0.2) = +0.2
```

For `doc_choose` or `hop_resolve`, privacy is inactive, so the answer advantage is used at full scale:

```text
combined_advantage = Adv_answer
```
