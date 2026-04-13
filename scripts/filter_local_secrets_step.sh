#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

has_flag() {
    local wanted="$1"
    shift
    for arg in "$@"; do
        case "$arg" in
            "$wanted"|"$wanted"=*)
                return 0
                ;;
        esac
    done
    return 1
}

cmd=(
  python -m trajectory_generation.data_prep.filter_local_secrets
  --input trajectory_generation/data/secret_inventory.jsonl
  --output-dir trajectory_generation/data
  --chunks-local trajectory_generation/data/chunks_local.jsonl
)

if ! has_flag --model "$@"; then
    cmd+=(--model "step-3.5-flash")
fi
if ! has_flag --base-url "$@"; then
    cmd+=(--base-url "http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000/v1")
fi

cmd+=("$@")

printf 'Running:'
for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
done
printf '\n'

exec "${cmd[@]}"
