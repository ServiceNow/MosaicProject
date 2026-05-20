# Privacy HopQA Prompt Appendix

Generated paper-facing prompt appendix for Privacy HopQA.

- `index.html`: screenshot-friendly single-page view.
- `prompt_appendix_data.json`: extracted prompts, examples, and sample outputs.

Regenerate from the MosaicProject root with:

```bash
python3 scripts/build_privacy_hopqa_prompt_appendix.py
```

The generator reads source prompt files in this repository plus the PipelineRL Privacy HopQA prompt builders and saved runtime/leakage artifacts listed at the bottom of `index.html`.
