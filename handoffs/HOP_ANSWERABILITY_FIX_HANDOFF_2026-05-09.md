# PrivacyHopQA hop answerability fix handoff

Date: 2026-05-09

## Goal

Review hop-level questions that StepFun could not confidently answer from the full gold document, then propose dataset fixes.

The main objective is not to make every string match StepFun. The objective is to identify real dataset defects:

- The hop question is underspecified or points to the wrong fact.
- The gold answer is wrong or too narrow.
- Accepted answer variants are incomplete.
- The gold document does not actually support the hop question.
- The document is parse/truncation-sensitive and needs manual review.

Every proposed fix must be grounded in the gold document. Do not use external sources to justify a fix.

## Audit artifacts

Primary files:

- Script: `/home/toolkit/my_code/MosaicProject/scripts/audit_privacy_hopqa_hop_answerability_stepfun.py`
- Full audit JSONL: `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/stepfun_full_doc_hop_answerability_2026-05-09.jsonl`
- Review CSV: `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/stepfun_full_doc_hop_answerability_review_2026-05-09.csv`
- HTML viewer: `/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_stepfun_answerability_review_2026-05-09.html`
- Source dataset: `/home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset/privacy_hopqa_finalized_2026-05-09_contextsafe_answer_variants.jsonl`

The viewer is standalone and can be opened locally. It supports filtering by flag, label, company, match state, and free text.

## How the audit was run

For each hop question in the final dataset:

- StepFun was shown the full gold document and the hop question.
- StepFun was not shown the gold answer or accepted answer variants.
- StepFun returned a short answer, an answerability label, evidence, and a brief reason.
- The script compared StepFun's answer against the accepted variants.

Important implementation notes:

- Local `.docx`, `.pptx`, and `.xlsx` files are extracted before prompting StepFun.
- Long documents are truncated at the configured context limit and flagged `doc_truncated`.
- One malformed long StepFun output was salvaged from answer fields and flagged `json_salvaged`.

Final summary:

- Total hop questions: 3403
- Exact matches: 3042
- Exact mismatches: 361
- Numeric-only matches: 129
- StepFun label `no`: 48
- StepFun label `requires_calculation`: 23
- Truncated documents: 119
- Unresolved StepFun errors: 0

## Recommended review priority

1. `label_no`

These are likely true issues. StepFun had the full document and still could not find an answer.

2. `exact_mismatch` without `numeric_match_only`

These include wrong gold answers, underspecified questions, and answer-variant gaps. Review manually.

3. `numeric_match_only`

These are often harmless answer-format/variant gaps, for example `20%` vs `20% reduction`, but they should still be checked before adding variants.

4. `label_requires_calculation`

Not automatically bad. These are useful to inspect because some hop questions may be more complex than intended.

5. `doc_truncated`

Review cautiously. If a mismatch occurs with `doc_truncated`, the audit may not have seen the full relevant evidence.

## Output expected from a fix-proposal agent

Produce a JSONL or CSV with one row per proposed change. Include:

- `chain_id`
- `hop_number`
- `hop_type`
- `company`
- `doc_id`
- `doc_ref`
- `issue_type`
- `current_question`
- `current_gold_answer`
- `stepfun_answer`
- `proposed_question`
- `proposed_gold_answer`
- `proposed_accepted_variants`
- `evidence_quote`
- `rationale`
- `confidence`
- `needs_human_review`

Use these `issue_type` values:

- `rewrite_question_keep_answer`
- `change_gold_answer`
- `add_answer_variants`
- `drop_or_manual_review`
- `audit_false_positive`

Bias toward `rewrite_question_keep_answer` when possible, because each hop answer may feed later hops in the chain. Changing a hop answer can invalidate downstream questions.

## Rules for proposing fixes

- Preserve the current gold answer if a light rewrite can make the question clearly ask for that answer.
- Keep answers short. The downstream harness expects compact answers, usually under 5 words.
- Do not make a question ask for a full date if only the day/month/year component is needed downstream.
- Do not add overly verbose answer variants just because StepFun produced a long phrase.
- Do not rewrite with vague wording such as "according to the document".
- Make the question specific enough to point to the intended document/fact, but not so specific that it leaks the answer.
- For local docs that omit the company name but are clearly in that company's task directory, do not automatically mark as wrong. Still prefer questions that remain natural when the document itself does not include the company name.
- For web scorecards/tables, verify that the question asks about the right entity. Fielding dismissals like `c Mahmudullah b Mohammad Saifuddin` do not mean Mahmudullah was batting.

## Known examples to review first

### `19ddd451::hop1`

Current question:

`What percentage of county-wide job growth do tax-incentive deals averaging 1,500 jobs represent?`

Current gold answer:

`14`

StepFun answer:

`NOT_FOUND`

Why it is suspicious:

The document supports that firm-specific subsidies generally increased from 2002 to 2017, with a minimum of 14 in 2003 and a maximum of 53 in 2012. It does not support the county-wide job-growth percentage question.

Candidate fix:

`What was the minimum number of firm-specific subsidies offered in any year from 2002 to 2017?`

Candidate answer:

`14`

Evidence:

`minimum of 14 firm-specific subsidies offered in 2003`

This preserves the answer `14`, which matters because a downstream local hop asks about `14 employee retention bonuses`.

### `7a967dc7::hop1`

Current question:

`How many balls did Mahmudullah face while batting for Bangladesh against West Indies?`

Current gold answer:

`15`

StepFun answer:

`15`

Why it is still suspicious:

The evidence row is a West Indies batting row whose dismissal says `c Mahmudullah b Mohammad Saifuddin`. Mahmudullah is the fielder, not the batter. The Bangladesh batting table does not show Mahmudullah batting.

The question should be rewritten to target the scorecard row more precisely while preserving answer `15`, because the next hop uses `15%`.

Candidate direction:

Ask for the balls faced by the West Indies batter whose dismissal was caught by Mahmudullah and bowled by Mohammad Saifuddin in the WI vs BAN scorecard.

Avoid awkward phrasing if possible, but the corrected question must not imply Mahmudullah was batting.

### `57473242::hop2`

Current question:

`Following its Q2 2025 full CRM implementation, which company expects a 15% increase in customer retention rates if trends hold?`

Current gold answer:

`Elexion Automotive`

StepFun answer:

`NOT_FOUND`

Review reason:

StepFun says the document mentions related retention/customer-loyalty facts but not this exact Q2 2025 CRM implementation claim. Inspect the gold document and either rewrite to a supported fact that still answers `Elexion Automotive`, or mark for manual review.

### `85ea1c50::hop2`

Current question:

`What year is associated with the 92% sensitivity of lymphoscintigraphy for lymphedema diagnosis?`

Current gold answer:

`2016`

StepFun answer:

`NOT_FOUND`

Review reason:

StepFun says the gold document does not contain this sensitivity claim. Check whether the doc match is wrong or the question should target a different supported 2016 fact.

### `aa1409f9::hop3`

Current question:

`Which 2019 film, four words long, had domestic earnings equal to 40% of its worldwide total?`

Current gold answer:

`Jumanji: The Next Level`

StepFun answer:

`Toy Story 4`

Review reason:

This looks like a real ambiguity or wrong target. StepFun found another four-word 2019 film matching the 40% condition. Inspect the document and either add a disambiguating constraint or change the gold answer if the current gold is wrong.

## Fast review workflow

1. Open the HTML viewer.
2. Filter `label_no`.
3. For each row, open `doc_ref` and verify the StepFun reason against the gold document.
4. Record proposed fixes in the structured format above.
5. Repeat for `exact_mismatch`, excluding obvious `numeric_match_only` rows.
6. For `numeric_match_only`, add variants only when the predicted answer is semantically identical and still short.
7. Keep a separate list of rows that require human judgment.

## Commands

Rebuild viewer from the current audit JSONL:

```bash
python3 /home/toolkit/my_code/MosaicProject/scripts/audit_privacy_hopqa_hop_answerability_stepfun.py --rebuild-only
```

Rerun a specific chain:

```bash
python3 /home/toolkit/my_code/MosaicProject/scripts/audit_privacy_hopqa_hop_answerability_stepfun.py --chain-id 19ddd451 --concurrency 1
```

Run the full audit:

```bash
python3 /home/toolkit/my_code/MosaicProject/scripts/audit_privacy_hopqa_hop_answerability_stepfun.py --concurrency 64
```
