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

cmd=("${PYTHON:-python3}" -m mosaicleaks.src.retrieval.retrieval_helper_server)

if ! has_flag --port "$@"; then
    cmd+=(--port 8012)
fi

cmd+=("$@")

printf 'Running:'
for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
done
printf '\n'

exec "${cmd[@]}"
