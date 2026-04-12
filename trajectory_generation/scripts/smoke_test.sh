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

# Extract seeds from an existing run when a seeds file is not already present.
SEEDS_FILE="${SEEDS_FILE:-$HOME/trajectory_generation_runs/bad_seeds.jsonl}"
INPUT_CHAINS="${INPUT_CHAINS:-}"
CHAIN_IDS="${CHAIN_IDS:-}"
SEED_MAX="${SEED_MAX:-5}"
if [ ! -f "$SEEDS_FILE" ]; then
    if [ -z "$INPUT_CHAINS" ]; then
        echo "Set INPUT_CHAINS to an existing chains.jsonl file before running this script."
        exit 1
    fi
    echo "Extracting seeds from bad chains..."
    EXTRA_ARGS=(--max "$SEED_MAX")
    if [ -n "$CHAIN_IDS" ]; then
        EXTRA_ARGS=(--chain-ids $CHAIN_IDS)
    fi
    "$PYTHON" -m trajectory_generation.scripts.extract_bad_seeds \
        --input "$INPUT_CHAINS" \
        "${EXTRA_ARGS[@]}" \
        --output "$SEEDS_FILE"
fi

echo "Running pipeline with seeds from $SEEDS_FILE"
"$PYTHON" -m trajectory_generation.run \
    --seeds-from "$SEEDS_FILE" \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --base-url "http://127.0.0.1:8000/v1" \
    --verbose
