#!/usr/bin/env bash
# Phase A serving-profile capture for FP6 vs FP8 on Behemoth-R1-123B.
#
# Prerequisites:
#   - FP6 server already up on BASE_FP6 (default :8000), launched with
#       PROFILE=1 PROFILE_DIR=/tmp/vllm_prof_fp6 ./behemoth123b-r1-v2-fp6.sh KEY
#     (PROFILE=1 is what registers /start_profile and /stop_profile; without it
#      the capture aborts with a 404)
#   - FP8 server already up on BASE_FP8 (default :8001), same PROFILE=1 rule
#     with its own PROFILE_DIR
#   - Export PROFILE_DIR here too so the traces get collected; they are written
#     by the server into PROFILE_DIR, never into OUT_ROOT
#   - API key in VLLM_API_KEY or as $1
#
# Captures:
#   1) cold-ish prefill at ctx≈8192, max_tokens=1
#   2) decode window: short prompt, max_tokens=128
# for each server, then summarizes traces into kernel-class budgets.
#
# Usage:
#   ./scripts/phase_a_profile_behemoth.sh [API_KEY]
#   BASE_FP6=http://localhost:8000 BASE_FP8=http://localhost:8001 \
#     MODEL_FP6=Behemoth-R1-123B-v2-FP6-W6A6 \
#     MODEL_FP8=Behemoth-R1-123B-v2-W8A8-FP8-BLOCK \
#     ./scripts/phase_a_profile_behemoth.sh "$VLLM_API_KEY"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_KEY="${1:-${VLLM_API_KEY:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API_KEY" >&2
  exit 2
fi

BASE_FP6="${BASE_FP6:-http://127.0.0.1:8000}"
BASE_FP8="${BASE_FP8:-http://127.0.0.1:8001}"
MODEL_FP6="${MODEL_FP6:-Behemoth-R1-123B-v2-FP6-W6A6}"
MODEL_FP8="${MODEL_FP8:-Behemoth-R1-123B-v2-W8A8-FP8-BLOCK}"
OUT_ROOT="${OUT_ROOT:-/tmp/fp6_phase_a_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT_ROOT"
echo "Phase A profile output: $OUT_ROOT"

# ~8k-token-ish prompt via repetition (tokenizer-dependent; long enough to
# exercise the large-M GEMM + host GS path). Override with PREFILL_PROMPT_FILE.
make_prefill_prompt() {
  local dest="$1"
  if [[ -n "${PREFILL_PROMPT_FILE:-}" ]]; then
    cp "$PREFILL_PROMPT_FILE" "$dest"
    return
  fi
  "$PYTHON" - <<'PY' >"$dest"
word = "alpha "
# ~32k chars ≈ several thousand tokens depending on tokenizer.
print(word * 8000)
PY
}

capture_one() {
  local tag="$1" base="$2" model="$3" mode="$4" max_tokens="$5" prompt_file="$6"
  local out="$OUT_ROOT/${tag}_${mode}"
  mkdir -p "$out"
  echo "=== capture ${tag} ${mode} -> ${out} ==="
  "$PYTHON" "$ROOT/scripts/capture_vllm_native_profile.py" \
    --base-url "$base" \
    --model "$model" \
    --api-key "$API_KEY" \
    --prompt-file "$prompt_file" \
    --max-tokens "$max_tokens" \
    --capture-seconds "${CAPTURE_SECONDS:-8}" \
    --out-dir "$out" \
    --mode decode
  # Copy any traces the server wrote into PROFILE_DIR if the user exported one.
  if [[ -n "${PROFILE_DIR:-}" && -d "$PROFILE_DIR" ]]; then
    mkdir -p "$out/traces"
    find "$PROFILE_DIR" -type f \( -name '*.pt.trace.json.gz' -o -name '*.pt.trace.json' \) \
      -newer "$out" -exec cp -t "$out/traces" {} + 2>/dev/null || true
  fi
}

PREFILL_PROMPT="$OUT_ROOT/prefill_prompt.txt"
DECODE_PROMPT="$OUT_ROOT/decode_prompt.txt"
make_prefill_prompt "$PREFILL_PROMPT"
echo "Count from one to twenty in words, then keep counting." >"$DECODE_PROMPT"

# Prefill: long prompt, 1 output token.
capture_one fp6 "$BASE_FP6" "$MODEL_FP6" prefill 1 "$PREFILL_PROMPT"
capture_one fp8 "$BASE_FP8" "$MODEL_FP8" prefill 1 "$PREFILL_PROMPT"

# Decode: short prompt, longer generation.
capture_one fp6 "$BASE_FP6" "$MODEL_FP6" decode 128 "$DECODE_PROMPT"
capture_one fp8 "$BASE_FP8" "$MODEL_FP8" decode 128 "$DECODE_PROMPT"

echo
echo "=== summarizing traces under $OUT_ROOT ==="
"$PYTHON" "$ROOT/scripts/summarize_vllm_trace.py" "$OUT_ROOT" \
  --json-out "$OUT_ROOT/attribution.json" || {
  echo "WARN: no traces found under $OUT_ROOT yet." >&2
  echo "If the server used PROFILE_DIR, re-run summarize with that path:" >&2
  echo "  $PYTHON $ROOT/scripts/summarize_vllm_trace.py \"\$PROFILE_DIR\" --json-out $OUT_ROOT/attribution.json" >&2
}

# Evidence stub for the evidence notebook / PR comment.
{
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "out_root: $OUT_ROOT"
  echo "BASE_FP6=$BASE_FP6 MODEL_FP6=$MODEL_FP6"
  echo "BASE_FP8=$BASE_FP8 MODEL_FP8=$MODEL_FP8"
  date -u +"captured_utc: %Y-%m-%dT%H:%M:%SZ"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active,power.draw --format=csv
  fi
} >"$OUT_ROOT/evidence_header.txt"

echo "Wrote $OUT_ROOT/evidence_header.txt"
echo "Done."
