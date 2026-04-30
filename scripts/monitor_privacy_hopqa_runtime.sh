#!/usr/bin/env bash
set -u

if [[ -z "${JOB_ID:-}" || -z "${RUN_DIR:-}" ]]; then
  echo "JOB_ID and RUN_DIR are required." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="$(basename "$RUN_DIR")"
VIEWER_DIR="${VIEWER_DIR:-$ROOT_DIR/runtime_analysis/${RUN_NAME}_live}"
SLIM_VIEWER="${SLIM_VIEWER:-$ROOT_DIR/runtime_analysis/${RUN_NAME}_live.slim.html}"
LATEST_SLIM="${LATEST_SLIM:-$ROOT_DIR/runtime_analysis/latest_privacy_hopqa_slim_viewer.html}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
TAIL_LINES="${TAIL_LINES:-3000}"
TIME_SLICE_SIZE="${TIME_SLICE_SIZE:-128}"
MOVING_AVERAGE_WINDOW="${MOVING_AVERAGE_WINDOW:-256}"

cd "$ROOT_DIR"

refresh_once() {
  local ts status logs errors scheduler train_marker summary phase plot_status slim_status status_line
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  plot_status=ok
  slim_status=ok

  status="$(eai job get "$JOB_ID" --fields id,state,name,runs.state,runs.stateInfo,runs.startedOn,runs.runTime --format table 2>&1 || true)"
  logs="$(eai job log "$JOB_ID" --tail "$TAIL_LINES" 2>&1 || true)"

  python scripts/plot_privacy_hopqa_runtime.py \
    --run-dir "$RUN_DIR" \
    --output-dir "$VIEWER_DIR" >/tmp/privacy_hopqa_viewer_refresh.log 2>&1 || plot_status=failed

  python scripts/build_privacy_hopqa_slim_viewer.py \
    --analysis-dir "$VIEWER_DIR" \
    --output "$SLIM_VIEWER" \
    --time-slice-size "$TIME_SLICE_SIZE" \
    --moving-average-window "$MOVING_AVERAGE_WINDOW" >/tmp/privacy_hopqa_slim_viewer_refresh.log 2>&1 || slim_status=failed

  cp -f "$SLIM_VIEWER" "$LATEST_SLIM" 2>/dev/null || true

  errors="$(printf "%s\n" "$logs" | grep -Ei "Traceback|AsyncEngineDead|empty training texts|CUDA out of memory|CUBLAS_STATUS|RuntimeError|\\bERROR\\b|Exception" | tail -8 || true)"
  scheduler="$(printf "%s\n" "$logs" | grep -E "train scheduler for llms" | tail -1 || true)"
  train_marker="$(printf "%s\n" "$logs" | grep -Ei "global_step|optimizer step|train step|train_step|checkpoint|Saving checkpoint|Completed steps|reward_mean|advantage" | tail -8 || true)"
  status_line="$(printf "%s\n" "$status" | awk '$1 ~ /^[0-9a-f-]{36}$/ {print; exit}')"
  summary="$(python - "$VIEWER_DIR/summary.json" "$LATEST_SLIM" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
slim_path = Path(sys.argv[2])
if summary_path.exists():
    data = json.loads(summary_path.read_text())
    size_mb = slim_path.stat().st_size / (1024 * 1024) if slim_path.exists() else 0.0
    print(
        f"viewer_rollouts={data.get('rollout_count')} "
        f"mean_reward={float(data.get('mean_reward') or 0):.3f} "
        f"completed_rollouts_per_s={float(data.get('completed_rollouts_per_s') or 0):.3f} "
        f"slim_mb={size_mb:.2f}"
    )
PY
)"

  if [[ -n "$errors" ]]; then
    phase=ERROR_SIGNAL
  elif [[ -n "$train_marker" ]]; then
    phase=TRAIN_MARKER
  elif [[ -n "$scheduler" ]]; then
    phase=ROLLOUT_COLLECTION
  elif printf "%s\n" "$logs" | grep -q "privacy_hopqa llm call summary"; then
    phase=LLM_CALLS_ACTIVE
  elif printf "%s\n" "$logs" | grep -q "Tracking run with wandb"; then
    phase=WANDB_AND_MODEL_STARTUP
  else
    phase=NO_CLEAR_PROGRESS_MARKER
  fi

  echo "[$ts] phase=$phase plot_refresh=$plot_status slim_refresh=$slim_status slim_viewer=$LATEST_SLIM"
  [[ -n "$status_line" ]] && echo "$status_line"
  [[ -n "$summary" ]] && echo "$summary"
  [[ -n "$scheduler" ]] && echo "$scheduler"
  if [[ -n "$train_marker" ]]; then
    echo "train markers:"
    printf "%s\n" "$train_marker"
  fi
  if [[ -n "$errors" ]]; then
    echo "error markers:"
    printf "%s\n" "$errors"
  fi
  echo "---"
}

while true; do
  refresh_once
  [[ "${RUN_ONCE:-0}" == "1" ]] && exit 0
  sleep "$INTERVAL_SECONDS"
done
