#!/usr/bin/env bash
# Smoke test: run trajectory generation on same seeds as bad chains.
# Requires vLLM serving Qwen3-30B locally at http://127.0.0.1:8000
#
# Usage:
#   bash trajectory_generation/scripts/smoke_test.sh

set -euo pipefail

PYTHON="${PYTHON:-/home/toolkit/.mamba/envs/vllm013/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# Extract seeds from bad chains (Jem/Hasbro example + similar)
SEEDS_FILE="${SEEDS_FILE:-$HOME/detailed_trajectory_logs/bad_seeds.jsonl}"
if [ ! -f "$SEEDS_FILE" ]; then
    echo "Extracting seeds from bad chains..."
    "$PYTHON" -m trajectory_generation.scripts.extract_bad_seeds \
        --input making_dataset_2/outputs/chains_all.jsonl \
        --chain-ids 17453ea8 b0d08ed6 \
        --output "$SEEDS_FILE"
fi

echo "Running pipeline with seeds from $SEEDS_FILE"
"$PYTHON" -m trajectory_generation.run \
    --seeds-from "$SEEDS_FILE" \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --base-url "http://127.0.0.1:8000/v1" \
    --verbose
