# Examples

## Full Pipeline

```bash
python -m trajectory_generation.run \
  --chunks-local trajectory_generation/data/chunks_local.jsonl \
  --chunks-web trajectory_generation/data/chunks_web.jsonl \
  --curated-web-chunks trajectory_generation/data/chunks_curated_web.jsonl \
  --curated-web-docs trajectory_generation/data/docs_curated_web.jsonl \
  --output-dir ~/trajectory_generation_runs/example_run
```

## Reuse the Web Retriever

If web retrieval startup time is getting in the way, you can keep that service running in a separate process:

```bash
python -m trajectory_generation.src.retrieval.browsecomp_hybrid_server --port 8011
export MOSAIC_TRAJ_BROWSECOMP_SEARCH_URL=http://127.0.0.1:8011
python -m trajectory_generation.run --output-dir ~/trajectory_generation_runs/example_run
```

## Render an Output File

```bash
python -m trajectory_generation.scripts.view_generation \
  ~/trajectory_generation_runs/example_run/chains_filtered.jsonl \
  --open
```
