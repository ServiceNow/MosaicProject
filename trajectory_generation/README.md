# Trajectory Generation Pipeline

The pipeline for generating multi-hop question chains for privacy evaluation.

## Pipeline Stages

1: load chunks
2: build chains
3: verify chains meet answerability conditions
4: filter chains with llm judge
5: fix minor rewording

## Usage

```bash
# Default: run all stages, output to ~/detailed_trajectory_logs/run_<timestamp>/
python -m trajectory_generation.run

# Resume from stage 2 (e.g. after build completed)
python -m trajectory_generation.run --from-stage 2

# Smoke test: chain specs from bad chains
python -m trajectory_generation.run --seeds-from chains_bad_seeds.jsonl --max-chains 10

# Custom model and paths
python -m trajectory_generation.run \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --base-url http://127.0.0.1:8000/v1 \
    --output-dir /path/to/output
```


