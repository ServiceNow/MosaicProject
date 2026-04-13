# Curated Web Prep

These utilities preserve the workflow used to refresh the curated task-URL web corpus that supplements the main fixed web corpus.

## Defaults

The scripts now take their defaults from `trajectory_generation.config`:

- tasks root: `MOSAIC_TRAJ_TASKS_DIR` or `trajectory_generation/data/tasks`
- docs output: `trajectory_generation/data/docs_curated_web.jsonl`
- chunk output: `trajectory_generation/data/chunks_curated_web.jsonl`

## Typical Flow

```bash
python -m trajectory_generation.data_prep.fetch_drbench_urls
```

That command:

1. reads task `context.json` files
2. refreshes the curated URL corpus JSONL
3. optionally emits chunk JSONL for retrieval

The underlying module names still reference `drbench_urls` because the current curated corpus originated from that task set, but the package no longer assumes the original neighboring repos are present.
