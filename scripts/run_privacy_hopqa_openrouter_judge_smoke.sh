#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/toolkit/my_code/MosaicProject}"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT/runtime_analysis/privacy_hopqa_q30_stepfun_judge_validation_cross_model_200_20260525}"
OPENROUTER_ENV_FILE="${OPENROUTER_ENV_FILE:-/home/toolkit/my_code/openkey.env}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-anthropic/claude-opus-4.7}"
LIMIT="${LIMIT:-3}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0}"
DISABLE_RESPONSE_FORMAT="${DISABLE_RESPONSE_FORMAT:-1}"
CHECK_CREDITS="${CHECK_CREDITS:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing python: $PYTHON" >&2
  exit 1
fi
if [[ ! -s "$OPENROUTER_ENV_FILE" ]]; then
  echo "Missing OpenRouter env file: $OPENROUTER_ENV_FILE" >&2
  exit 1
fi
if [[ ! -s "$SAMPLE_DIR/trace_manifest.txt" ]]; then
  echo "Missing trace manifest under sample dir: $SAMPLE_DIR" >&2
  exit 1
fi
if (( LIMIT > 5 )) && [[ "${ALLOW_LARGE_SMOKE:-0}" != "1" ]]; then
  echo "Refusing LIMIT=$LIMIT for smoke. Set ALLOW_LARGE_SMOKE=1 if intentional." >&2
  exit 1
fi
if (( CONCURRENCY > LIMIT )); then
  echo "Refusing CONCURRENCY=$CONCURRENCY > LIMIT=$LIMIT." >&2
  exit 1
fi

SLUG="$(printf '%s' "$OPENROUTER_MODEL" | tr '/:@' '___' | tr -cd 'A-Za-z0-9_.-')"
OUTPUT="${OUTPUT:-$SAMPLE_DIR/openrouter_smoke_${SLUG}_limit${LIMIT}_${RUN_ID}.jsonl}"
ERRORS="${ERRORS:-$SAMPLE_DIR/openrouter_smoke_${SLUG}_limit${LIMIT}_${RUN_ID}_errors.jsonl}"
SUMMARY="${SUMMARY:-$SAMPLE_DIR/openrouter_smoke_${SLUG}_limit${LIMIT}_${RUN_ID}_summary.json}"
DISAGREEMENTS="${DISAGREEMENTS:-$SAMPLE_DIR/openrouter_smoke_${SLUG}_limit${LIMIT}_${RUN_ID}_disagreements.jsonl}"

"$PYTHON" - <<'PY'
import importlib
for name in ["openai", "rich"]:
    importlib.import_module(name)
print("python environment ok: openai and rich are importable")
PY

if [[ "$CHECK_CREDITS" == "1" ]]; then
  "$PYTHON" scripts/check_openrouter_key_usage.py --env-file "$OPENROUTER_ENV_FILE"
fi

extra_args=()
if [[ "$DISABLE_RESPONSE_FORMAT" == "1" ]]; then
  extra_args+=(--disable-response-format)
fi

echo "Running OpenRouter judge smoke:"
echo "  model=$OPENROUTER_MODEL"
echo "  limit=$LIMIT concurrency=$CONCURRENCY max_tokens=$MAX_TOKENS"
echo "  output=$OUTPUT"

"$PYTHON" scripts/evaluate_privacy_hopqa_leakage_suite_v2.py \
  --artifact-root "$SAMPLE_DIR" \
  --trace-manifest "$SAMPLE_DIR/trace_manifest.txt" \
  --output "$OUTPUT" \
  --errors "$ERRORS" \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env-file "$OPENROUTER_ENV_FILE" \
  --model "$OPENROUTER_MODEL" \
  --api-mode chat \
  --limit "$LIMIT" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --continue-on-error \
  "${extra_args[@]}"

"$PYTHON" scripts/report_privacy_hopqa_judge_smoke.py \
  --sample-dir "$SAMPLE_DIR" \
  --reference "$SAMPLE_DIR/stepfun_reference_sample.jsonl" \
  --candidate "$OUTPUT" \
  --errors "$ERRORS" \
  --output-summary "$SUMMARY" \
  --output-disagreements "$DISAGREEMENTS"

if [[ "$CHECK_CREDITS" == "1" ]]; then
  "$PYTHON" scripts/check_openrouter_key_usage.py --env-file "$OPENROUTER_ENV_FILE"
fi

echo
echo "Smoke complete."
echo "Output:        $OUTPUT"
echo "Errors:        $ERRORS"
echo "Summary:       $SUMMARY"
echo "Disagreements: $DISAGREEMENTS"
