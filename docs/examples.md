# Examples

## Historical Large-Run Defaults

These were the main settings for the late-March 2026 reruns:

- Step path: `step-3.5-flash` with `--max-tokens 16384`
- Local Qwen path: `Qwen/Qwen3-30B-A3B-Instruct-2507` with `--max-tokens 8192`
- In the older split code, some short calls were written as `150`, `300`, or `1024`, but the shared client often inflated them to the stage floor. In this standalone repo, those short calls now use an explicit `8192` cap so the visible settings match reality better.

## Build Local Inputs

```bash
./scripts/build_local_chunks.sh
./scripts/extract_local_secrets_step.sh
./scripts/filter_local_secrets_step.sh
```

## Full Pipeline With Local Qwen

```bash
./scripts/run_generation_local_qwen.sh \
  --chunks-local trajectory_generation/data/chunks_local.jsonl \
  --chunks-web trajectory_generation/data/chunks_web.jsonl \
  --curated-web-chunks trajectory_generation/data/chunks_curated_web.jsonl \
  --curated-web-docs trajectory_generation/data/docs_curated_web.jsonl \
  --secrets trajectory_generation/data/secret_inventory.jsonl \
  --output-dir ~/trajectory_generation_runs/example_run
```

## Full Pipeline With Step

```bash
./scripts/run_generation_step.sh \
  --base-url http://your-step-endpoint:8000/v1 \
  --chunks-local trajectory_generation/data/chunks_local.jsonl \
  --chunks-web trajectory_generation/data/chunks_web.jsonl \
  --curated-web-chunks trajectory_generation/data/chunks_curated_web.jsonl \
  --curated-web-docs trajectory_generation/data/docs_curated_web.jsonl \
  --secrets trajectory_generation/data/secret_inventory.jsonl \
  --seeds-from /path/to/seed_plan.jsonl \
  --output-dir ~/trajectory_generation_runs/example_run
```

## Reuse the Web Retriever

If web retrieval startup time is getting in the way, you can keep that service running in a separate process:

```bash
./scripts/start_web_retriever.sh --port 8011
python -m trajectory_generation.run \
  --output-dir ~/trajectory_generation_runs/example_run \
  --base-url http://127.0.0.1:8000/v1 \
  --browsecomp-search-url http://127.0.0.1:8011
```

## Render an Output File

```bash
python -m trajectory_generation.scripts.view_generation \
  ~/trajectory_generation_runs/example_run/chains_filtered.jsonl \
  --open
```
