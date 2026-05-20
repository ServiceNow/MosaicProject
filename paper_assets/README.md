# Paper Assets

Paper-facing generated artifacts live here.

- `privacy_hopqa_prompts/index.html`: screenshot-friendly prompt appendix with private QA generation prompts, test examples, adversary prompts, research-agent step prompts, and sample outputs.
- `privacy_hopqa_prompts/prompt_appendix_data.json`: source data used to render the appendix.
- `privacy_hopqa_dataset_tables/`: LaTeX dataset summary tables for the paper.

Regenerate the prompt appendix from the repository root with:

```bash
python3 scripts/build_privacy_hopqa_prompt_appendix.py
```

Regenerate the dataset tables with:

```bash
python3 scripts/build_privacy_hopqa_dataset_tables.py
```
