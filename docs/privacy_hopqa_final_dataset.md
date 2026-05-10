# Privacy HopQA Final Dataset Bundle

## Current trainable version: 2026-05-10 spanfixed

The current trainable dataset is:

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
