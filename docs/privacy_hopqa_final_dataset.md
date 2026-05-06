# Privacy HopQA Final Dataset Bundle

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
