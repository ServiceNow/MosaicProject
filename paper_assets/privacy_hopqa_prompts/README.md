# Privacy HopQA Prompt Appendix

Generated paper-facing prompt appendix for Privacy HopQA.

- `index.html`: screenshot-friendly single-page view.
- `prompt_appendix_data.json`: extracted prompts, examples, and sample outputs.
- `latex_prompt_snippets/`: one copyable `tcolorbox` prompt per file.
- `latex_prompt_short_snippets/`: short variants with example-heavy sections omitted where available.
- `latex_prompt_output_snippets/`: prompt boxes paired with sample outputs where available.
- `latex_table_snippets/`: paper-ready `table*` snippets for examples and leakage cases.
- `overleaf_prompt_pngs/`: PNG prompt boxes ready for Overleaf upload.
- `overleaf_prompt_short_pngs/`: PNG short prompt boxes ready for Overleaf upload.
- `overleaf_prompt_output_pngs/`: PNG prompt-plus-output boxes ready for Overleaf upload.

LaTeX snippets expect these paper preamble pieces:

```latex
\usepackage{xcolor}
\usepackage{tcolorbox}
\tcbuselibrary{breakable}
\usepackage{booktabs}
\usepackage{enumitem}
\definecolor{my-green}{HTML}{166B54}
```

Equivalent shortcut: `\usepackage[most]{tcolorbox}` loads the breakable library too. The generated boxes include the `breakable` option so long prompts can continue across page breaks.

Regenerate from the MosaicProject root with:

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3.12 scripts/build_privacy_hopqa_prompt_appendix.py
```

The PNG export uses Pillow when available. The generator reads source prompt files in this repository plus the PipelineRL Privacy HopQA prompt builders and saved runtime/leakage artifacts listed at the bottom of `index.html`.
