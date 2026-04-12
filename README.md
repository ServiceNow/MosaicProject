# MosaicProject

MosaicProject is a standalone package for generating multi-hop research chains from local documents and a fixed set of web documents. The repo exposes one main Python package, [`trajectory_generation`](./trajectory_generation).

The current package shape is meant to be readable and presentation-ready:
- one generator package
- no runtime dependency on the neighboring `drbench` or `making_dataset` repos
- default data/config paths under `trajectory_generation/data/`
- concise documentation for architecture, system behavior, and representative commands

## Quick Start

```bash
pip install -e .
```

Place or symlink the required inputs into `trajectory_generation/data/`, or pass explicit paths on the command line:

- `chunks_local.jsonl`
- `chunks_web.jsonl`
- `secret_inventory.jsonl`
- optional `chunks_curated_web.jsonl`
- optional `docs_curated_web.jsonl`

Historical rerun defaults are captured in the repo-level scripts:

```bash
./scripts/run_generation_local_qwen.sh \
  --output-dir ~/trajectory_generation_runs/example_run
```

```bash
./scripts/run_generation_step.sh \
  --base-url http://your-step-endpoint:8000/v1 \
  --output-dir ~/trajectory_generation_runs/example_run
```

Render a generated JSONL file to HTML:

```bash
python -m trajectory_generation.scripts.view_generation \
  ~/trajectory_generation_runs/example_run/chains_filtered.jsonl \
  --open
```

## Docs

- [Architecture](./docs/architecture.md)
- [Current System](./docs/system.md)
- [Examples](./docs/examples.md)

## Repository Layout

- [`trajectory_generation/`](./trajectory_generation) main package
- [`trajectory_generation/data/`](./trajectory_generation/data) default location for local inputs
- [`trajectory_generation/src/retrieval/`](./trajectory_generation/src/retrieval) local and web retrieval code
- [`scripts/`](./scripts) explicit run wrappers for the local Qwen path, the Step path, and the reusable web retriever
- [`docs/`](./docs) advisor-facing overview notes
