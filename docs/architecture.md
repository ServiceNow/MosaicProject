# Architecture

Multi-hop chain generation lives inside `trajectory_generation`.

## Runtime Shape

- `run.py` is the main pipeline entrypoint.
- `stages/` holds the resumable pipeline stages: load, build, verify, filter, reword.
- `chain_builder.py`, `questions.py`, `checks.py`, and `normalization.py` implement the active chain-construction logic.
- `src/retrieval/` contains local search, web search, and the code that combines search results.
- `data/` is the default place for local inputs and caches.

## Retrieval Model

The build stage combines:

- local document retrieval over chunk JSONL
- web retrieval over the fixed BrowseComp collection
- an optional second web source limited to documents linked to the current task
- when both web sources are available, it mixes the ranked lists and gives extra credit to documents that show up near the top in both

## Package Boundary

The package includes its own runtime configuration, retrieval adapters, data-loading utilities, and pipeline entrypoints. The expected input files live under `trajectory_generation/data/` by default, with environment-variable overrides if you want the data somewhere else.
