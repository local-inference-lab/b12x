#!/usr/bin/env bash
# A/B the MX scale-factor copy strategy on the Behemoth TP=2 shard shapes.
#
# Decides G2b, which the SASS census alone cannot. Vectorizing the scale-factor
# copy trades 80 LDS.U8 for 96 PRMT and lands 35 instructions HEAVIER in the
# k-loop (488 -> 523). Instruction count says stop; pipe placement says measure.
# The loads it removes sit on the LSU/MIO pipe, which ncu flags as throttled
# (+0.44 vs FP8) in exactly this loop, while the permutes it adds sit on the ALU
# pipe, which is not the limiter here. Shared-load latency and LSU throughput
# are both far worse than ALU throughput, so a heavier loop can still be faster.
#
# Numerics: the scale BYTES are unchanged, only the route from smem to register
# differs, so both arms must be bit-identical. --no-check is NOT used here; run
# the FP6 bit-equality suite too before believing any win.
#
# Usage:
#   ./scripts/ab_sf_copy_mode.sh                     # prefill, all 4 shards
#   MODE=decode ./scripts/ab_sf_copy_mode.sh         # decode regression check
#   ARMS="off recast32" ./scripts/ab_sf_copy_mode.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${MODE:-prefill}"
ARMS="${ARMS:-off recast32}"
OUT_DIR="${OUT_DIR:-/tmp/fp6_sfcopy_ab_$(date +%Y%m%d_%H%M%S)}"

# shellcheck source=scripts/_behemoth_tp2_shapes.sh
source "$ROOT/scripts/_behemoth_tp2_shapes.sh"
SHAPES="${SHAPES:-gate_up qkv down o}"

if [[ "$MODE" == "decode" ]]; then
  M="${M:-1}"
  ARM_FLAGS=(--no-expanded --no-fp8)
else
  M="${M:-8192}"
  ARM_FLAGS=(--no-packed --no-fp8)
fi

mkdir -p "$OUT_DIR"
echo "mode=$MODE  M=$M  arms='$ARMS'  out=$OUT_DIR"

# Each arm is a separate PROCESS: the compiled-kernel cache is per-process and
# sf_copy_mode is read once at import.
for shape in $SHAPES; do
  # :- so an unknown shape reaches the guard below; set -u would otherwise
  # abort the whole sweep on the missing key.
  n="${SHAPE_N[$shape]:-}"
  k="${SHAPE_K[$shape]:-}"
  [[ -z "$n" || -z "$k" ]] && { echo "unknown shape '$shape'" >&2; continue; }
  echo ""
  echo "=== $shape  M=$M N=$n K=$k ==="
  for arm in $ARMS; do
    log="$OUT_DIR/${MODE}_${shape}_${arm}.log"
    if ! SPARKINFER_DENSE_SF_COPY_MODE="$arm" \
      "$PYTHON" "$ROOT/benchmarks/benchmark_dense_gemm_fp6.py" \
        --m "$M" --n "$n" --k "$k" "${ARM_FLAGS[@]}" \
        --warmup 20 --iters 200 \
      >"$log" 2>&1
    then
      echo "  $arm: FAILED (see $log)"
      tail -n 20 "$log" >&2
      continue
    fi
    # The bench prints one row per arm; med_us and tflops are what we compare,
    # and cos is the correctness column the run computes for us.
    printf '  %-10s %s\n' "$arm" \
      "$(grep -E 'fp6_(expanded|packed)' "$log" | tail -n 1)"
  done
done

echo ""
{
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "mode: $MODE"
  echo "m: $M"
  echo "arms: $ARMS"
  date -u +"captured_utc: %Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active --format=csv 2>/dev/null
} >"$OUT_DIR/evidence_header.txt"
echo "Logs and evidence_header.txt in $OUT_DIR"
