# Privacy HopQA Final Dataset

Open `dataset_viewer.html` from a local HTTP server in this directory.

```bash
cd /home/toolkit/my_code/MosaicProject/data_files/privacy_hopqa_final_dataset
python3 -m http.server 8765
```

Then visit `http://localhost:8765/dataset_viewer.html`.

Files:

- `privacy_hopqa_finalized_2026-05-07_answer_variants.jsonl`: finalized dataset rows with StepFun-vetted accepted-answer variants.
- `train.jsonl`, `val.jsonl`, `test.jsonl`: canonical split; `test.jsonl` holds out one full company.
- `split_manifest.json`: split strategy, counts, and held-out company.
- `dataset_stats.json`: task/company/pattern/hop/doc statistics.
- `dataset_analysis.json`: derived transition, effective-hop, and answer-variant statistics.
- `question_uniqueness_stats.json`: approximate near-duplicate question clusters.
- `plots/`: distribution plots used by the viewer.
- `manifest.json`: publish provenance.
