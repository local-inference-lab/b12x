#!/usr/bin/env bash
# SASS instruction-mix census of the FP6 dense GEMM mainloop (plan item G1).
#
# ncu told us the FP6 prefill GEMM issues MMA at 72-75% of peak where FP8 hits
# 93-95%, and that FP6's stall cycles concentrate in memory-pipe and barrier
# reasons FP8 barely touches. That localizes the problem to the mainloop but
# does not say what is in it. This script recovers the actual instruction
# budget: how many shared-memory loads, address computations, and barriers the
# kernel executes per MMA.
#
# No GPU counters are involved, so this needs no ncu setup and no reboot - it
# only needs the kernel to have been COMPILED, which the benchmark does anyway.
#
# The prefill and decode arms compile from the same DSL class with different
# tiles and stage counts. Capturing both is the cheap control: decode is the arm
# we already tuned to lead FP8, so a category that is heavy in prefill and light
# in decode is a much stronger suspect than one that is heavy in both.
#
# Usage:
#   ./scripts/sass_census_fp6_gemm.sh                    # prefill, all 4 shards
#   MODE=decode ./scripts/sass_census_fp6_gemm.sh        # the M=1 control
#   SHAPES=gate_up ./scripts/sass_census_fp6_gemm.sh     # single shard
#   OUT_DIR=/tmp/g1 ./scripts/sass_census_fp6_gemm.sh
#
# Scale-factor copy strategy A/B (G2b). The default tracks the shipping default
# so a census describes the kernel that actually runs; "off" emits one LDS.U8
# per UE8M0 byte and is the pre-G2b arm. CuTe rejects an impossible strategy at
# compile time, so a failed arm is a legitimate answer and the bench log carries
# the layout error:
#   SF_COPY=off      OUT_DIR=/tmp/g2b_off ./scripts/sass_census_fp6_gemm.sh
#   SF_COPY=recast32 OUT_DIR=/tmp/g2b_r32 ./scripts/sass_census_fp6_gemm.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${MODE:-prefill}"
OUT_DIR="${OUT_DIR:-/tmp/fp6_sass_$(date +%Y%m%d_%H%M%S)}"

# shellcheck source=scripts/_behemoth_tp2_shapes.sh
source "$ROOT/scripts/_behemoth_tp2_shapes.sh"
SHAPES="${SHAPES:-gate_up qkv down o}"

if [[ "$MODE" == "decode" ]]; then
  M="${M:-1}"
  ARM_FLAGS="--no-expanded --no-fp8"
else
  M="${M:-8192}"
  ARM_FLAGS="--no-packed --no-fp8"
fi

if ! command -v nvdisasm >/dev/null 2>&1; then
  echo "nvdisasm not found; install the CUDA toolkit or add it to PATH." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "mode=$MODE  M=$M  out=$OUT_DIR"

for shape in $SHAPES; do
  # :- so an unknown shape reaches the guard below; set -u would otherwise
  # abort the whole sweep on the missing key.
  n="${SHAPE_N[$shape]:-}"
  k="${SHAPE_K[$shape]:-}"
  [[ -z "$n" || -z "$k" ]] && { echo "unknown shape '$shape'" >&2; continue; }

  tag="${MODE}_${shape}_m${M}"
  # One shape per cache directory. The census walks every object it is given,
  # and a shared cache would mix four shards' kernels into one report with no
  # way to tell which is which - the mangled names are identical.
  cache="$OUT_DIR/cache_$tag"
  rm -rf "$cache"
  mkdir -p "$cache"

  echo ""
  echo "=== $shape  M=$M N=$n K=$k ==="

  SPARKINFER_COMPILE_CACHE_DIR="$cache" \
  SPARKINFER_COMPILE_DISK_CACHE=1 \
  SPARKINFER_DENSE_SF_COPY_MODE="${SF_COPY:-autovec}" \
    "$PYTHON" "$ROOT/benchmarks/benchmark_dense_gemm_fp6.py" \
      --m "$M" --n "$n" --k "$k" $ARM_FLAGS \
      --warmup 2 --iters 2 --no-check \
    >"$OUT_DIR/$tag.bench.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "benchmark exited $rc; tail of $OUT_DIR/$tag.bench.log:" >&2
    tail -n 30 "$OUT_DIR/$tag.bench.log" >&2
    continue
  fi

  objects=$(find "$cache" -name '*.o' | wc -l)
  if [[ "$objects" -eq 0 ]]; then
    echo "no objects cached - is SPARKINFER_COMPILE_DISK_CACHE honored?" >&2
    continue
  fi
  echo "cached $objects object(s)"

  "$PYTHON" "$ROOT/scripts/sass_instruction_mix.py" \
      "$cache" --kernel DenseGemmKernel --whole-kernel \
    >"$OUT_DIR/$tag.census.txt" 2>&1
  if [[ ! -s "$OUT_DIR/$tag.census.txt" ]]; then
    echo "census produced no output for $shape" >&2
    continue
  fi
  sed -n '1,60p' "$OUT_DIR/$tag.census.txt"
done

echo ""
echo "Censuses in $OUT_DIR/*.census.txt"
{
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "mode: $MODE"
  echo "m: $M"
  echo "shapes: $SHAPES"
  echo "sf_copy_mode: ${SF_COPY:-autovec}"
  echo "nvdisasm: $(nvdisasm --version 2>/dev/null | tr '\n' ' ')"
  date -u +"captured_utc: %Y-%m-%dT%H:%M:%SZ"
} >"$OUT_DIR/evidence_header.txt"
echo "Wrote $OUT_DIR/evidence_header.txt"
