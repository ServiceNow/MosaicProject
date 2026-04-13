#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

cmd=(
  python -m trajectory_generation.data_prep.chunk_local_tasks
  --tasks-root /home/toolkit/nice_code/drbench/drbench/data/tasks
  --output trajectory_generation/data/chunks_local.jsonl
)

cmd+=("$@")

printf 'Running:'
for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
done
printf '\n'

exec "${cmd[@]}"
