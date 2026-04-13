# Trajectory Generation

This package builds multi-hop research chains from local document chunks and a fixed set of web documents. It is the main generator inside MosaicProject.

## Pipeline

- `load`: collect local chunks, secrets, and manifest metadata
- `build`: generate candidate chains with retrieval, validation, and reranking
- `verify`: check answerability under multiple context ablations
- `filter`: score chains with an LLM judge
- `reword`: make small clarity fixes without changing semantics

## Data Layout

The package defaults to `trajectory_generation/data/`:

- `chunks_local.jsonl`
- `chunks_web.jsonl`
- `secret_inventory.jsonl`
- optional `chunks_curated_web.jsonl`
- optional `docs_curated_web.jsonl`
- optional `tasks/` for curated URL refresh utilities

For experiments, prefer passing explicit CLI arguments or using the repo-level shell wrappers instead of relying on shell-wide defaults.

## Entry Points

```bash
python -m trajectory_generation.run
python -m trajectory_generation.src.retrieval.browsecomp_hybrid_server
python -m trajectory_generation.scripts.view_generation path/to/chains.jsonl
```

See the repo-level [examples](../docs/examples.md), [architecture notes](../docs/architecture.md), and [run scripts](../scripts) for the intended workflow.
