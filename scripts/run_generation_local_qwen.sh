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

cmd=(python -m trajectory_generation.run)

if ! has_flag --model "$@"; then
    cmd+=(--model "Qwen/Qwen3-30B-A3B-Instruct-2507")
fi
if ! has_flag --base-url "$@"; then
    cmd+=(--base-url "http://127.0.0.1:8000/v1")
fi
if ! has_flag --max-tokens "$@"; then
    cmd+=(--max-tokens 8192)
fi
if ! has_flag --short-max-tokens "$@"; then
    cmd+=(--short-max-tokens 8192)
fi
if ! has_flag --max-chains "$@"; then
    cmd+=(--max-chains 5000)
fi
if ! has_flag --patterns "$@"; then
    cmd+=(--patterns LL LW LWL LWW LLW LWWL LWLW LLWW)
fi
if ! has_flag --workers "$@"; then
    cmd+=(--workers 1)
fi

cmd+=("$@")

printf 'Running:'
for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
done
printf '\n'

exec "${cmd[@]}"
