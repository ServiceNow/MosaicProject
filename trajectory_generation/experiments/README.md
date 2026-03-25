# Atomic Chain-Building Experiments

Run each part of the chain-building pipeline in isolation and save raw outputs to files. Use this to debug where instructions are unclear or things go wrong.

## Setup

Requires:
- vLLM (or OpenAI-compatible API) serving a model at `--base-url`
- Data: `chunks_local_all.jsonl`, `chunks_web*.jsonl`, `secret_inventory_all_filtered.jsonl`
- spacy `en_core_web_trf`

## Usage

```bash
# Full LW chain with BM25 (default, matches real pipeline)
PYTHON=/path/to/vllm013/bin/python python -m trajectory_generation.experiments.run_atomic \
  --output-dir /tmp/atomic_exp \
  --task DR0001 \
  --pattern LW \
  --seed 42

# Run only specific steps (0=load, 1=seed, 2=find_bridge, 3=gen, 4=checks, 5=rank)
python -m trajectory_generation.experiments.run_atomic \
  --output-dir /tmp/atomic_exp \
  --task DR0001 \
  --steps 0,1,2,3

# Use different candidate (default 0 = first bridge candidate)
python -m trajectory_generation.experiments.run_atomic \
  --output-dir /tmp/atomic_exp \
  --candidate-idx 2

# Override model/API
python -m trajectory_generation.experiments.run_atomic \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507

# Quick BM25 test with smaller web corpus (~1 min load vs ~5 min for full 100K)
python -m trajectory_generation.experiments.run_atomic \
  --web-chunks making_dataset_2/outputs/chunks_web_drbench_urls.jsonl \
  --steps 0,1,2

# No BM25 (fastest, exhaustive match, less realistic)
python -m trajectory_generation.experiments.run_atomic --no-bm25 --steps 0,1,2
```

## Output Files

| File | Step | Contents |
|------|------|----------|
| `00_manifest.json` | 0 | Load stats: n_local_docs, n_web_docs, n_secrets, task_ids |
| `01_seed_l.json` | 1 | L seed: question, answer, doc_id |
| `01_seed_w_prompt.txt` | 1 | W seed: LLM prompt |
| `01_seed_w_raw.txt` | 1 | W seed: raw LLM response |
| `02_find_bridge.json` | 2 | Bridge candidates: target_doc_id, bridge_entity, needs_intra |
| `03a_intra_prompt.txt` | 3 | Intra question prompt (if needs_intra) |
| `03a_intra_raw.txt` | 3 | Intra raw LLM response |
| `03a_intra_parsed.json` | 3 | Intra: question, answer, quote |
| `03b_pick_prompt.txt` | 3 | Pick question prompt |
| `03b_pick_raw.txt` | 3 | Pick raw LLM response |
| `03b_pick_parsed.json` | 3 | Pick: question, answer, quote, entity_list |
| `04a_check_question.json` | 4 | check_question: passed, error |
| `04b_check_trivial.json` | 4 | check_answerable_without_doc |
| `04c_backref_prompt.txt` | 4 | Backref check prompt (question with placeholder) |
| `04c_backref_raw.txt` | 4 | Backref check raw LLM response |
| `04c_check_backref.json` | 4 | check_answer_needs_backref: blanked_question, result |
| `04d_check_search_leak.json` | 4 | check_search_leaks_bridge |
| `05_rank_prompt.txt` | 5 | Rank prompt (if multiple candidates) |
| `05_rank_raw.txt` | 5 | Rank raw LLM response |

## Inspecting Outputs

- **Prompts** (`.txt`): What we send to the model. Check for unclear instructions.
- **Raw** (`.txt`): Model response before parsing. Check format, hallucinations.
- **Parsed** (`.json`): Extracted fields. Compare to raw to debug parse failures.
- **Checks** (`.json`): Pass/fail and reasons. See which validation fails and why.
