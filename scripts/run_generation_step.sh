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

if ! has_flag --base-url "$@"; then
    cat >&2 <<'EOF'
run_generation_step.sh expects an explicit OpenAI-compatible endpoint.

Example:
  ./scripts/run_generation_step.sh \
    --base-url http://your-endpoint:8000/v1 \
    --output-dir ~/mosaicleaks_runs/step_run \
    --seeds-from /path/to/seed_plan.jsonl \
    --chunks-local /path/to/chunks_local.jsonl \
    --chunks-web /path/to/chunks_web.jsonl \
    --secrets /path/to/secret_inventory.jsonl
EOF
    exit 1
fi

cmd=("${PYTHON:-python3}" -m mosaicleaks.run)

if ! has_flag --model "$@"; then
    cmd+=(--model "step-3.5-flash")
fi
if ! has_flag --max-tokens "$@"; then
    cmd+=(--max-tokens 16384)
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
