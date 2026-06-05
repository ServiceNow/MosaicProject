#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

user_passed_flag() {
    local flag_name="$1"
    shift

    # Accept both "--flag value" and "--flag=value" forms. We only need this
    # check for defaults supplied by the wrapper; all real parsing happens in
    # the Python server.
    for arg in "$@"; do
        case "$arg" in
            "$flag_name"|"$flag_name"=*)
                return 0
                ;;
        esac
    done
    return 1
}

# Keep the wrapper intentionally small. The Python module defines the full CLI;
# this script only gives users a stable command from the repo root.
cmd=("${PYTHON:-python3}" -m mosaicleaks.src.retrieval.retrieval_helper_server)

# Use the paper/training default helper port unless the caller chooses another.
if ! user_passed_flag --port "$@"; then
    cmd+=(--port 8012)
fi

# Forward every caller-provided flag to the Python server unchanged.
cmd+=("$@")

printf 'Running:'
for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
done
printf '\n'

# Replace the shell process with the server so signals such as Ctrl-C and
# scheduler termination go directly to uvicorn.
exec "${cmd[@]}"
