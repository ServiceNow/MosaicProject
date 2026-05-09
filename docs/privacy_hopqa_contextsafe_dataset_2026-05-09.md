# Privacy HopQA 2026-05-09 Context-Safe Answer Variant Dataset

This document records the 2026-05-09 dataset and harness update for Privacy HopQA.

## Dataset version

Version name:

```text
2026-05-09_contextsafe_answer_variants
```

Primary files:

```text
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-09_contextsafe_answer_variants.jsonl
data_files/privacy_hopqa_final_dataset/train_2026-05-09_contextsafe_answer_variants.jsonl
data_files/privacy_hopqa_final_dataset/val_2026-05-09_contextsafe_answer_variants.jsonl
data_files/privacy_hopqa_final_dataset/test_2026-05-09_contextsafe_answer_variants.jsonl
data_files/privacy_hopqa_final_dataset/manifest_2026-05-09_contextsafe_answer_variants.json
```

Split shape:

```text
full: 1001 chains, 3403 hops
train: 559 chains, 1888 hops
val: 98 chains, 340 hops
test: 344 chains, 1175 hops
```

The test split continues to hold out MediConn Solutions as the full-company generalization set.

## What changed

The 2026-05-09 dataset is a dated copy of the current curated Privacy HopQA dataset after the latest answer-variant cleanup pass.

The important policy distinction is:

```text
accepted answer variants are for reward/evaluation
canonical hop answers are for chain context
```

For example, if a hop asks for a day of the month, the canonical answer may be:

```text
30
```

but validation may accept a more specific answer when it is clearly the same fact:

```text
May 30, 2025
```

This does not mean the next hop should receive `May 30, 2025` as the substituted answer. The harness now keeps the model's raw answer for audit, but uses the canonical hop answer as the context form after the raw answer matches an accepted variant.

## PipelineRL harness behavior

The `privacy_hopqa` rollout harness now separates:

```text
raw answer: what the model actually produced
context answer: the canonical hop answer used for later references after an accepted-variant match
```

Prompts show this in a simple notes format:

```text
Current Answers So Far:
(1) May 30, 2025
    For references to (1), use: 30
    Justification: The evidence states that ownership of Red was sold back on May 30, 2025. [DOC:...]
```

The full numbered chain remains visible, so the model can still reason from the original question text and the raw answer surface.

This is intended to prevent accepted-but-awkward aliases from poisoning downstream retrieval queries or `(N)` substitutions.

The current-answer notes also include the stored previous-hop justification, truncated for prompt size. This gives later hops lightweight provenance without requiring the model to reconstruct why earlier answers were accepted.

## PipelineRL code changes in this batch

The PipelineRL changes that go with this dataset version are intentionally broader than a path bump:

```text
pipelinerl/domains/privacy_hopqa/dataset.py
pipelinerl/domains/privacy_hopqa/agent.py
pipelinerl/domains/privacy_hopqa/prompts.py
pipelinerl/domains/privacy_hopqa/settings.py
pipelinerl/domains/privacy_hopqa/llm_adapter.py
pipelinerl/domains/privacy_hopqa/rollouts.py
pipelinerl/async_llm.py
conf/domain_params/privacy_hopqa.yaml
```

The changes are:

```text
1. Default dataset path now points at the 2026-05-09 context-safe dataset.
2. Hop state stores the canonical answer and accepted variants, so a raw accepted alias can be mapped to a context-safe answer for later hops.
3. Prompts use "Current Answers So Far" notes instead of a JSON resolved-answer block.
4. Web retrieval defaults to windowed evidence: retrieve larger parent text, expose selectable evidence windows, and include parent/window metadata in traces.
5. Local and web reader context sizes are configurable through `retrieval_context_mode` and window settings.
6. LLM calls can pass per-stage `max_tokens` overrides and cap output tokens against a configured context limit.
7. Helper transport/server errors are treated as infrastructure errors so rollouts fail fast/retry instead of recording a normal wrong answer.
8. The resolver is not allowed to mark a hop answered unless at least one document read produced `can_answer=true`.
```

Non-obvious behavior:

```text
retrieval_context_mode: windowed_evidence
```

is now the default. The old prefix-only behavior can be restored with:

```text
privacy_hopqa.retrieval_context_mode=legacy_prefix
```

Decimal percent aliases, such as `0.3` for `30%`, remain accepted in the dataset for now. They may be awkward as chain context, but the context-safe answer mapping should prevent them from being substituted into later-hop questions after an accepted match.

Follow-up answer-variant update:

```text
035c3d89 H2
```

asks for Lee's Market's disaster recovery infrastructure investment. The gold support states `$250,000`, while the dataset only accepted bare numeric forms. The following evaluation aliases were added:

```text
$250,000
$250000
250,000 dollars
250000 dollars
250 thousand
250 thousand dollars
```

## Final report justifications

The model does not need to carry justifications forward in its own notes. The harness stores them.

For each hop:

```text
1. `doc_read` proposes an answer and a document-grounded justification.
2. `hop_resolve` decides whether the hop is answered and returns the final hop justification.
3. The agent stores the raw answer, context answer, confidence, and justification in `HopState`.
4. The final report is deterministic: it writes the stored hop answers and stored hop justifications.
```

Later-hop prompts now show the current answers so far plus the stored previous-hop justifications. This gives the model provenance for earlier answers while keeping the final report deterministic: the report still writes the stored hop answers and stored hop justifications rather than asking the model to regenerate them.

## Review artifacts

First-web targeted answer-variant review:

```text
runtime_analysis/firstweb_answer_variant_review_2026-05-09.html
runtime_analysis/firstweb_answer_variant_review_2026-05-09.csv
```

Full main-dataset review:

```text
runtime_analysis/privacy_hopqa_full_dataset_answer_review_2026-05-09.html
runtime_analysis/privacy_hopqa_full_dataset_answer_review_2026-05-09.csv
runtime_analysis/privacy_hopqa_full_dataset_answer_review_2026-05-09.json
```

Focused high-priority review notes:

```text
runtime_analysis/privacy_hopqa_high_priority_review_2026-05-09.md
runtime_analysis/privacy_hopqa_high_priority_review_2026-05-09.json
```

The full review currently covers all 3403 hops and joins the latest full-dataset StepFun answer-unit proposal file:

```text
data_files/privacy_hopqa_final_dataset/stepfun_answer_unit_variant_proposals_2026-05-07_v11.jsonl
```

The review queue is a prioritization surface, not an automatic patch. In particular, old StepFun proposal rows often reject punctuation aliases or decimal percent aliases; those should be sampled and curated rather than bulk-applied.

## Current cleanup status

The full-dataset review currently reports:

```text
high priority: 12 hops
medium priority: 1657 hops
low priority: 12 hops
no flag: 1722 hops
missing StepFun proposal coverage: 123 hops
```

The high-priority queue is the next recommended direct review target. The medium-priority queue is broad and intentionally conservative; most of it is not necessarily wrong.

The initial high-priority review suggests that most high flags are stale metadata or noisy StepFun proposal artifacts rather than true dataset errors. Non-obvious suggested rewrites, especially generated rewrites containing references like `FY (1)` or `0. (1)`, should not be applied automatically.

## Recommended next work

1. Directly review the 12 high-priority hops in the full-dataset viewer against their gold support spans and source documents.
2. Decide keep/rewrite/drop for those rows, preserving the original hop answer granularity whenever possible.
3. Run or reuse StepFun only for rows where the support span is insufficient or where answer aliases remain ambiguous.
4. Sample the medium-priority queue by flag type, especially `stepfun_rejected_existing_variant`, before applying any broad label cleanup.
5. Run a small PipelineRL smoke rollout with the new context-safe answer behavior before starting another full training job.
