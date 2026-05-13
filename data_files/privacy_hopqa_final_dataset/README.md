# Privacy HopQA q30 final dataset

Current canonical dataset version: `2026-05-13_q30_final_residual_review`.

Files:

| File | Purpose |
| --- | --- |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.jsonl` | Full 1,001-chain dataset. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_train.jsonl` | Train split, 559 chains. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_val.jsonl` | Validation split, 98 chains. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review_test.jsonl` | Test split, 344 MediConn Solutions chains. |
| `manifest_privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.json` | Dataset provenance, counts, rewrite notes, first-web counts. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.applied_log.jsonl` | Applied change log. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.quarter_cleanup_audit.jsonl` | Quarter/year answer audit. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.quarter_cleanup_log.jsonl` | Quarter/year cleanup changes. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.contradictory_qyear_variant_cleanup_2026-05-13.jsonl` | Removed contradictory quarter/year aliases. |
| `privacy_hopqa_finalized_2026-05-13_q30_final_residual_review.targeted_question_variant_fixes_2026-05-13.jsonl` | Final targeted question and variant fixes. |

Split policy:

| split | chains | company mix |
| --- | ---: | --- |
| train | 559 | Elexion Automotive 311, Lee's Market 248 |
| val | 98 | Elexion Automotive 56, Lee's Market 42 |
| test | 344 | MediConn Solutions 344 |

Helper-facing retrieval artifacts are not expected to live in this ignored dataset directory. The current shared helper bundle is:

```bash
/mnt/llmd/results/exps/alexg/privacy_hopqa_shared/q30_final_residual_review_20260513
```

Run this consistency check before launching new experiments:

```bash
cd /home/toolkit/my_code/MosaicProject
source configs/privacy_hopqa_q30_current.env
python3 scripts/validate_privacy_hopqa_setup.py
```
