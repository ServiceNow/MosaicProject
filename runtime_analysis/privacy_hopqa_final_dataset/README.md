# Privacy HopQA Final Dataset

Open `dataset_viewer.html` from a local HTTP server in this directory.

```bash
cd /home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_final_dataset
python3 -m http.server 8765
```

Then visit `http://localhost:8765/dataset_viewer.html`.

Files:

- `privacy_hopqa_finalized_2026-05-05.jsonl`: finalized dataset rows.
- `dataset_stats.json`: task/company/pattern/hop/doc statistics.
- `question_uniqueness_stats.json`: approximate near-duplicate question clusters.
- `manifest.json`: publish provenance.
