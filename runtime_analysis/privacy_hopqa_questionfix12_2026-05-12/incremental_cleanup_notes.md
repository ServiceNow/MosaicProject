# Privacy HopQA incremental cleanup notes, 2026-05-12

Latest dataset version: `2026-05-12_variantfix12_questionfix12`.

This cleanup intentionally did not rerun the full 3,403-hop gold-document audit. It used the existing production-stack audit plus small changed-row reaudits for q8-q12.

Key safeguards:
- No gold answers changed from q7 to q12.
- Every patch is versioned and has an applied log plus decision JSONL with per-row rationale.
- Office-document false positives from the earlier audit were overridden by the office-dependency reaudit.
- Remaining failures are tracked in `current_residual_queue_merged_2026-05-12.jsonl` and should be treated as a queue, not as a reason to restart a full audit.

q8-q12 changes:
- q8: seven q7 residual clarifications; reverted the only proposed answer change and preserved `266::hop2 = 1`.
- q9: concrete mismatch bucket cleanup with answer-preserving question rewrites and answer variants.
- q10: tighter wording for rows still failing due adjacent values or grammar.
- q11: one malformed table wording pass for `727::hop3`; further rewriting would leak the answer, so the row remains flagged as model/table-parse weakness.
- q12: fixed `744::hop7` and removed unsupported `September 2025` wording from `999::hop3` while preserving answer `PDE`.

Known residual interpretation:
- Some remaining concrete residuals are clear model/evaluator errors despite source support, e.g. `141::hop1`, `144::hop3`, `20361464::hop7`.
- Some are audit-window/span-location misses, e.g. `266::hop2`, `814::hop4`, `999::hop3`.
- `727::hop3` is a malformed markdown table parse issue; the source supports `1000`, but the 4B evaluator repeatedly misreads the table columns.
