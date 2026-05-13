# Privacy HopQA Final Dataset Bundle

## Current trainable version: 2026-05-13 q30 final residual review

The historical trainable dataset is:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.jsonl`

Split files:

- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl`
- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_val.jsonl`
- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_test.jsonl`

First-web debugging dataset:

- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-13_q30_final_residual_review.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-13_q30_final_residual_review_train.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-13_q30_final_residual_review_val.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-13_q30_final_residual_review_test.jsonl`

Build script:

`scripts/apply_privacy_hopqa_q30_final_residual_review_fixes_2026_05_13.py`

Patch log and manifest:

- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.applied_log.jsonl`
- `data_files/privacy_hopqa_final_dataset/manifest_privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.json`

Counts:

| Split | Chains | Hops | Company coverage |
|---|---:|---:|---|
| Full | 1,001 | 3,403 | 367 Elexion, 290 Lee's, 344 MediConn |
| Train | 559 | 1,888 | 311 Elexion, 248 Lee's |
| Val | 98 | 340 | 56 Elexion, 42 Lee's |
| Test | 344 | 1,175 | 344 MediConn |
| First-web full | 794 | 794 | 296 Elexion, 223 Lee's, 275 MediConn |
| First-web train | 444 | 444 | 251 Elexion, 193 Lee's |
| First-web val | 75 | 75 | 45 Elexion, 30 Lee's |
| First-web test | 275 | 275 | 275 MediConn |

### q30 targeted residual answerability audit

This targeted audit reran the 24 residual rows from the q29 full gold-evidence
audit after the q30 fixes. It uses Qwen3-4B-Instruct-2507 with the production
document extraction path, `window_chars=16000`, `overlap_chars=1000`, all
windows enabled, and `f1_threshold=0.8`.

Run output:

`runtime_analysis/privacy_hopqa_q30_targeted_residual_reaudit_4b_2026-05-13/`

Final result:

| Metric | Value | Meaning |
|---|---:|---|
| `total` | 24 | The q29 residual hops selected for targeted q30 reaudit. |
| `resolver_match` | 24 / 24 | Final resolver answer matched the gold answer or an accepted variant. This is the primary pass/fail metric for the targeted residual audit. |
| `resolver_answered` | 24 / 24 | Resolver produced a non-empty final answer. |
| `any_window_match` | 23 / 24 | At least one reader pass over any gold-document window produced a matching answer. |
| `any_window_can_answer` | 23 / 24 | At least one reader pass over any gold-document window claimed the question was answerable. |
| `correct_window_match` | 20 / 24 | The reader pass on the supporting-span-centered window produced a matching answer. |
| `correct_window_can_answer` | 21 / 24 | The reader pass on the supporting-span-centered window claimed the question was answerable. |
| `span_located` | 16 / 24 | The diagnostic script found the exact `supporting_span` string in the extracted document text. Lower values here can reflect legitimate span wording drift; it is not the reward metric. |
| `doc_read_methods` | 24 production content processor | All targeted rows used the same production document extraction path. |
| `elapsed_s` | 13.336 | Wall-clock time for this targeted audit on the local 4B vLLM server. |

Metric interpretation:

- `correct_window_*` tests whether the model can answer when shown the window
  centered on the dataset supporting span.
- `any_window_*` tests whether any window from the gold document contains enough
  evidence for the reader to produce the gold answer.
- `resolver_*` tests the full diagnostic reader/resolver path: read all gold
  document windows, then choose the final answer from the reader results.
- For training readiness, `resolver_match` is the most relevant diagnostic,
  because it checks the final answer-selection behavior after the document has
  been retrieved and read.

Audit command:

```bash
cd /home/toolkit/my_code/MosaicProject

python3 - <<'PY'
import csv
from pathlib import Path
src = Path('runtime_analysis/privacy_hopqa_q29_final_full_gold_evidence_docref_4b_2026-05-13/resolver_failures_investigation.csv')
out = Path('runtime_analysis/privacy_hopqa_q30_targeted_residual_rows_2026-05-13.txt')
rows = list(csv.DictReader(src.open()))
out.write_text('\\n'.join(f"{r['chain_id']}::hop{r['hop_number']}" for r in rows) + '\\n')
PY

/home/toolkit/bin/micromamba run -p /home/toolkit/.conda/envs/pipeline-rl \
  python scripts/test_privacy_hopqa_firstweb_evidence_modes.py \
    --dataset data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.jsonl \
    --output-dir runtime_analysis/privacy_hopqa_q30_targeted_residual_reaudit_4b_2026-05-13 \
    --row-id-file runtime_analysis/privacy_hopqa_q30_targeted_residual_rows_2026-05-13.txt \
    --base-url http://127.0.0.1:8000 \
    --case-concurrency 8 \
    --concurrency 48 \
    --f1-threshold 0.8 \
    --log-every 4
```

q30 changes from q29:

- no rows dropped
- no bridge answers changed
- 14 residual question rewrites
- 19 residual answer-variant expansions
- 14 supporting-span rewrites to exact or more direct evidence text

## Historical trainable version: 2026-05-12 variantfix12/questionfix12

The current trainable dataset is:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-12_variantfix12_questionfix12.jsonl`

The historical first-web debugging dataset is:

`data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-12_variantfix12_questionfix12.jsonl`

The exact helper and trainer launch commands are in:

`docs/privacy_hopqa_firstweb_q12_2026-05-12.md`

The historical 2026-05-10 spanfixed bundle is documented below.

## Historical trainable version: 2026-05-10 spanfixed

The 2026-05-10 spanfixed dataset is:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed.jsonl`

Split files:

- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed_train.jsonl`
- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed_val.jsonl`
- `data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed_test.jsonl`

First-web-only debug split:

- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed_train.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed_val.jsonl`
- `data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_2026-05-10_spanfixed_test.jsonl`

Counts:

- full dataset: `1001` chains, `3403` hops
- full train: `559` chains (`311` Elexion, `248` Lee's)
- full val: `98` chains (`56` Elexion, `42` Lee's)
- full test: `344` chains (`344` MediConn)
- first-web debug: `104` chains
- first-web train: `58` chains (`33` Elexion, `25` Lee's)
- first-web val: `12` chains (`8` Elexion, `4` Lee's)
- first-web test: `34` chains (`34` MediConn)

The 2026-05-10 spanfixed version starts from
`privacy_hopqa_finalized_2026-05-10.jsonl` and fixes stale
`supporting_span` values for nine hops whose questions, answers, or document
assignments changed during the final audit pass. The patched spans are recorded
in:

`data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-10_spanfixed.spanfix_log.jsonl`

The manifest is:

`data_files/privacy_hopqa_final_dataset/manifest_privacy_hopqa_finalized_2026-05-10_spanfixed.json`

Validation performed before using this version for training:

- split IDs cover the full dataset exactly once
- first-web source IDs are a subset of the full spanfixed dataset
- no empty questions or accepted-answer lists
- no missing document files
- no duplicate answer variants
- every gold answer appears in its accepted-answer list
- all nine spanfixed `supporting_span` values are present in the assigned document
- first-web rows are single-hop chains whose first hop is web

Targeted StepFun answerability re-audit was rerun on the 31 changed hops with
evidence-centered truncation. Result: `30/31` exact matches. The only mismatch
is `bff7f93c::hop4`, a `requires_calculation` audit-date row where the document
supports the gold answer but StepFun selected Jason Wong's row instead of the
row with audit ID one less than Jason Wong's audit ID. Three long-document rows
still report `doc_truncated`, but their relevant evidence span is now included
in the prompt and StepFun answered them correctly.

Build commands:

```bash
python3 scripts/finalize_privacy_hopqa_spanfix_2026_05_10.py

python3 scripts/build_privacy_hopqa_first_web_hop_dataset.py \
  --source-dir /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset \
  --source-basename privacy_hopqa_finalized_2026-05-10_spanfixed \
  --output-dir /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_first_web_hop_dataset \
  --output-basename privacy_hopqa_first_web_hop_2026-05-10_spanfixed
```

The StepFun audit script now centers truncation windows on `supporting_span` and
accepted answers before falling back to the document head. This avoids the
previous false negatives where the model never saw the relevant evidence in very
large documents.

The finalized Privacy HopQA dataset bundle is published into:

`runtime_analysis/privacy_hopqa_final_dataset/`

Publish command:

```bash
python /home/toolkit/my_code/MosaicProject/scripts/publish_privacy_hopqa_final_dataset.py
```

The publisher refuses to run while the source quality-pass export contains `NOT_READY.txt`.

Expected bundle contents:

- `privacy_hopqa_finalized_2026-05-05.jsonl`
- `privacy_hopqa_finalized_2026-05-05.meta.json`
- `dataset_stats.json`
- `question_uniqueness_stats.json`
- `dataset_viewer.html`
- `manifest.json`

Viewer:

```bash
cd /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_final_dataset
python3 -m http.server 8765
```

Then open `http://localhost:8765/dataset_viewer.html`.

The viewer includes filters for task, pattern, review status, manual-review status, and question-quality warnings. The stats page summarizes tasks, companies, hop-pattern frequency, doc inference source, question issues, and estimated near-duplicate question clusters.

## 2026-05-11 variantfix2/questionfix2

A new versioned dataset was produced after the gold-document answerability
review:

```text
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.jsonl
data_files/privacy_hopqa_first_web_hop_dataset/privacy_hopqa_first_web_hop_anywhere_2026-05-11_variantfix2_questionfix2.jsonl
```

The patch applies 14 question rewrites and 5 answer-variant expansions. No rows
were dropped. Rewrites keep the same gold answers and preserve bridge
placeholders in the full chains; first-web rows use the realized question text
with prior-hop answers filled in.

Review artifacts:

```text
data_files/privacy_hopqa_final_dataset/proposed_question_changes_2026-05-11_questionfix2.jsonl
data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.applied_log.jsonl
data_files/privacy_hopqa_final_dataset/manifest_privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.json
```

Gold-document answerability audit command:

```bash
cd /home/toolkit/my_code/MosaicProject

/home/toolkit/bin/micromamba run -p /home/toolkit/.conda/envs/pipeline-rl \
  python scripts/test_privacy_hopqa_firstweb_evidence_modes.py \
    --dataset data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-11_variantfix2_questionfix2.jsonl \
    --output-dir runtime_analysis/privacy_hopqa_full_gold_evidence_modes_2026-05-11_variantfix2_questionfix2 \
    --case-concurrency 8 \
    --concurrency 48
```

### Question-template synchronization

A follow-up integrity pass synchronized inherited `question_templated` metadata
with realized hop questions for rows where rounding qualifiers were present in
`hop.question` but missing from the full-chain prompt. This keeps
`numbered_questions` consistent with the question the hop is actually evaluated
against.

```text
full-chain realization mismatches before sync: 135
full-chain realization mismatches after sync: 0
first-web placeholder questions after sync: 0
```

One non-mechanical placeholder repair was made: `16213b4e::hop4` now uses `(3)`
for the year `2020` and `(2)` for `Basel III`, matching the realized question
and preserving chain bridges.

### Full-hop gold-evidence audit run3

The full-hop audit was rerun after refactoring the diagnostic script to use
shared document handling instead of duplicated Office parsing.

Document source policy:

```text
1. For local hops, prefer the PipelineRL task-local index documents.json content when available.
2. Otherwise, use scripts/audit_privacy_hopqa_hop_answerability_stepfun.py::extract_document_text(doc_ref).
3. Web docs use the same shared extract_document_text(doc_ref) path.
```

Run output:

```text
runtime_analysis/privacy_hopqa_full_gold_evidence_modes_2026-05-11_variantfix2_questionfix2_run3_shared_doc_reader/
```

Summary:

```text
total hops: 3403
resolver_match: 3136
any_window_match: 3171
correct_window_match: 3045
any_window_can_answer: 3365
correct_window_can_answer: 3249
resolver_answered: 3341
span_located: 2294
elapsed_s: 471.479
```

Failure triage:

```text
ok: 3110
reader_answered_but_no_variant_match: 194
resolver_lost_reader_correct_answer: 43
no_gold_window_answer: 38
correct_window_answered_but_no_match: 18
```

Document read methods observed on this host:

```text
doc_ref_extract_document_text: 3375
local_index_documents_json: 28
```

The low local-index count means this host only has a small local-index subset
under the default path. The audit is still valid for document-answerability
triage because it uses the shared dataset StepFun extractor fallback, but a
helper-identical local-doc audit should be run with `--local-index-root` pointed
at the full helper index root if that is mounted locally.

Primary triage file:

```text
runtime_analysis/privacy_hopqa_full_gold_evidence_modes_2026-05-11_variantfix2_questionfix2_run3_shared_doc_reader/triage_failures.jsonl
```

### Full-hop gold-evidence audit run4

`run3_shared_doc_reader` is superseded. The audit now uses both production-style
document text extraction and the actual PipelineRL PrivacyHopQA reader/resolver
prompt builders. This matters because the simplified diagnostic prompt created
false negatives for local documents whose text did not restate the company name
or whose question required prior-hop context.

Run output:

```text
runtime_analysis/privacy_hopqa_full_gold_evidence_modes_2026-05-11_variantfix2_questionfix2_run4_production_reader_prompt/
```

Summary:

```text
total hops: 3403
resolver_match: 3167
any_window_match: 3200
correct_window_match: 3085
any_window_can_answer: 3358
correct_window_can_answer: 3265
resolver_answered: 3324
span_located: 2303
elapsed_s: 878.774
```

Failure triage:

```text
ok: 3103
reader_answered_but_no_variant_match: 158
resolver_lost_reader_correct_answer: 69
no_gold_window_answer: 45
correct_window_answered_but_no_match: 28
```

Document read methods:

```text
production_content_processor: 3375
local_index_documents_json: 28
```

Primary triage file:

```text
runtime_analysis/privacy_hopqa_full_gold_evidence_modes_2026-05-11_variantfix2_questionfix2_run4_production_reader_prompt/triage_failures.jsonl
```
