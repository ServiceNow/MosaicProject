#!/bin/bash
# Quick test: build a few chains. Use vllm013 env for spaCy en_core_web_trf.
# Requires vLLM (or OpenAI-compatible API) at --base-url.

set -euo pipefail

PYTHON="${PYTHON:-/home/toolkit/.mamba/envs/vllm013/bin/python}"
OUT_DIR="${1:-/tmp/traj_test_$(date +%Y%m%d_%H%M%S)}"
MAX_CHAINS="${MAX_CHAINS:-5}"
PATTERNS="${PATTERNS:-LW}"

echo "Output: $OUT_DIR"
echo "Max chains: $MAX_CHAINS  Patterns: $PATTERNS"
echo ""

$PYTHON -m trajectory_generation.run \
  --output-dir "$OUT_DIR" \
  --max-chains "$MAX_CHAINS" \
  --patterns $PATTERNS \
  --verbose

echo ""
echo "Chains: $OUT_DIR/chains.jsonl"
if [[ -f "$OUT_DIR/chains.jsonl" ]]; then
  echo "---"
  head -1 "$OUT_DIR/chains.jsonl" | $PYTHON -c "
import json, sys
d = json.load(sys.stdin)
print('Chain ID:', d.get('chain_id'))
print('Pattern:', d.get('pattern'))
print('Numbered Q:', d.get('numbered_questions', '')[:200] + '...' if len(d.get('numbered_questions','')) > 200 else d.get('numbered_questions'))
print('Final answer:', d.get('global_answer'))
"
fi
