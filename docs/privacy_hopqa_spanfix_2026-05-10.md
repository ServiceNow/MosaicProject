# Privacy HopQA Spanfix and First-Web Rerun Notes, 2026-05-10

## Purpose

This pass fixes the last dataset issue found before rerunning the first-web
training experiment: several audited hops had correct questions, answers, and
document assignments, but stale `supporting_span` values. Those stale spans
made long-document StepFun answerability checks unreliable because the audit
prompt could truncate away the real evidence.

## New dataset version

Full dataset:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed.jsonl`

Split files:

- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed_train.jsonl`
- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed_val.jsonl`
- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed_test.jsonl`

First-web dataset:

- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed_train.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed_val.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed_test.jsonl`

Counts:

- full dataset: `1001` chains and `3403` hops
- full train: `559` chains
- full val: `98` chains
- full test: `344` chains, all MediConn
- first-web dataset: `104` chains
- first-web train: `58` chains
- first-web val: `12` chains
- first-web test: `34` chains, all MediConn

## Span fixes

The spanfix script updates nine hops:

- `fd3a3f26::hop2`
- `0ee7622b::hop2`
- `13d66edb::hop2`
- `cfae3321::hop1`
- `73bea4c4::hop2`
- `7c388d5e::hop3`
- `bff7f93c::hop4`
- `bc18cce3::hop1`
- `8687c9b5::hop7`

The change log is:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed.spanfix_log.jsonl`

## Build commands

```bash
cd /home/toolkit/my_code/MosaicProject

python3 scripts/finalize_privacy_hopqa_spanfix_2026_05_10.py

python3 scripts/build_privacy_hopqa_first_web_hop_dataset.py \
  --source-dir /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset \
  --source-basename privacy_hopqa_finalized_2026-05-10_spanfixed \
  --output-dir /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset \
  --output-basename privacy_hopqa_first_web_hop_2026-05-10_spanfixed
```

## Validation

Local structural validation passed:

- full split IDs cover the full dataset exactly once
- first-web IDs are a subset of the full spanfixed dataset
- no missing document files
- no empty questions
- no empty accepted-answer lists
- no duplicate answer variants
- every gold answer is present in `accepted_answers`
- every changed `supporting_span` appears in the assigned document
- every first-web row is single-hop and starts with a web hop

Targeted StepFun answerability validation was rerun on the 31 changed hops with
the patched audit script. Result:

- `30/31` exact matches
- `1/31` mismatch, `bff7f93c::hop4`

The remaining mismatch is expected. The document supports the gold answer
`2025-01-30`, but the question requires selecting the audit row with audit ID
one less than Jason Wong's audit ID. StepFun returned Jason Wong's own row date
instead, so the row remains a legitimate `requires_calculation` case rather
than a dataset defect.

## StepFun truncation fix

`scripts/audit_privacy_hopqa_hop_answerability_stepfun.py` now centers the
document window on `supporting_span` and accepted answers before falling back to
the document head. This makes long-document answerability checks test the
actual evidence instead of whatever happens to fit into the first truncation
window.

Three changed-hop audit rows still report `doc_truncated` because their source
documents are large, but the relevant evidence span was included and StepFun
answered each correctly.

## Training rerun setting

The rerun should use:

- first-web spanfixed train/val/test files
- `MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507`
- `REPLICAS=2`
- `CAPTURE_MODE=all_calls`
- `privacy_hopqa.reward_mode=all_hops_correct`
- `privacy_hopqa.iteration_budget_per_hop=3`
- `privacy_hopqa.helper_timeout_s=180`
- `actor.llm_max_rollouts=16`

`CAPTURE_MODE=all_calls` is intentional for this rerun. The previous
planning-only setting may have been a confounder when diagnosing reward
stagnation because it excluded chooser, reader, and resolver texts from
training.
