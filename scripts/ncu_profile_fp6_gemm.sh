#!/usr/bin/env bash
# Nsight Compute profile of the FP6 dense GEMM on Behemoth TP=2 shard shapes
# (plan item E1). Run scripts/setup_ncu.sh first; this aborts if counters are
# still admin-restricted.
#
# Why the microbenchmark and not the server: benchmarks/benchmark_dense_gemm_fp6.py
# already carries the exact per-GPU shard shapes, builds the same kernel through
# the same policy selection, and runs one shape at a time, so the report holds
# the kernel under test instead of 1977 kernels per decode step.
#
# What we are chasing. At decode M=1, so these GEMMs are pure weight streaming
# and DRAM throughput is the only metric that decides them. Measured Jul 28 on
# the current build: 1.07-1.38 TB/s on a 1.792 TB/s card (60-77%). Serving
# corroborates it - Behemoth needs 1.32 TB/s to hold its measured 26.8 tok/s, so
# decode is pinned to this roofline and latency work does not move it.
#
# The open question is where the missing bandwidth goes. FP6 moves 33% fewer
# weight bytes per token than FP8 yet decodes only 5-7% faster, which implies
# FP8 sustains ~1.55 TB/s where FP6 gets ~1.30. BYTES=1 measures whether the
# 6-bit packed layout reads more DRAM than its values require.
#
# An earlier header here claimed 1.51 TB/s / 84% for FP6 and 1.72 TB/s for the
# FP8 arm. Those did not reproduce; treat the numbers above as current.
#
# Usage:
#   ./scripts/ncu_profile_fp6_gemm.sh                 # decode sweep, all 4 shards
#   MODE=prefill ./scripts/ncu_profile_fp6_gemm.sh    # M=8192, expanded-B arm
#   SHAPES=gate_up ./scripts/ncu_profile_fp6_gemm.sh  # single shard
#   M=4 ./scripts/ncu_profile_fp6_gemm.sh             # override the M value
#   STALLS=1 ./scripts/ncu_profile_fp6_gemm.sh        # add warp stall counters
#   ARM=fp8 MODE=prefill ./scripts/ncu_profile_fp6_gemm.sh   # the comparison arm
#
# ARM=fp8 profiles vLLM's blockwise CUTLASS FP8 kernel on the SAME (M,N,K) so the
# two are directly comparable: identical shape, identical card, identical
# capture settings, one metric set. That comparison is the point of the prefill
# capture - FP6 trails FP8 by 16-30% there and we have no direct evidence why,
# whereas at decode FP6 already leads.
#
# Occupancy/tile A/B (both are numerics-neutral, and both must be separate
# PROCESSES because the compiled-kernel cache is per-process):
#   SPARKINFER_DENSE_TARGET_OCCUPANCY=2 OUT_DIR=/tmp/occ2 ./scripts/ncu_profile_fp6_gemm.sh
#   SPARKINFER_FP6_DECODE_TILE=16x64 OUT_DIR=/tmp/t1664 ./scripts/ncu_profile_fp6_gemm.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
MODE="${MODE:-decode}"
OUT_DIR="${OUT_DIR:-/tmp/fp6_ncu_$(date +%Y%m%d_%H%M%S)}"

# (N, K) per shard, mirroring BEHEMOTH_TP2_SHAPES in the benchmark.
declare -A SHAPE_N=([qkv]=7168  [o]=12288 [gate_up]=28672 [down]=12288)
declare -A SHAPE_K=([qkv]=12288 [o]=6144  [gate_up]=12288 [down]=14336)
# gate_up first: at 28672x12288 it is 275 MB of the 541 MB each layer streams,
# so it dominates the 20.33 ms/step that the K=12288 pair costs.
SHAPES="${SHAPES:-gate_up qkv down o}"

if [[ "$MODE" == "decode" ]]; then
  M="${M:-1}"
  # Decode streams packed-B; the expanded arm is the prefill path and would
  # profile a kernel that never runs at M=1.
  ARM_FLAGS="--no-expanded --no-fp8"
else
  M="${M:-8192}"
  ARM_FLAGS="--no-packed --no-fp8"
fi

ARM="${ARM:-fp6}"
# sparkinfer compiles one CuTe DSL class, so its name is stable. vLLM's FP8 path
# resolves to a CUTLASS 3.x instantiation whose symbol varies with the selected
# tile; match the SM120 blockwise family broadly rather than pinning one mangled
# name, and let --launch-count 1 keep the report to a single kernel.
if [[ "$ARM" == "fp8" ]]; then
  ARM_FLAGS="--no-packed --no-expanded"
  # Broad on purpose: the arm resolves to vLLM's CUTLASS instantiation when
  # cutlass_scaled_mm accepts the call and to a cuBLASLt kernel (nvjet_/xmma_
  # families on Blackwell) when it falls back, and neither symbol is stable
  # across builds. --launch-count 1 keeps the report to one kernel regardless.
  KERNEL_RE="${KERNEL_RE:-regex:sm120|cutlass|gemm_universal|scaled_mm|nvjet|xmma|gemm}"
else
  KERNEL_RE="${KERNEL_RE:-regex:DenseGemmKernel}"
fi

if [[ ! -x "$NCU" ]]; then
  echo "ncu not found at '$NCU'. Run scripts/setup_ncu.sh." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "mode=$MODE  M=$M  out=$OUT_DIR"

# Memory-bound kernel: SpeedOfLight for the DRAM/compute roofline position,
# MemoryWorkloadAnalysis for where the bytes actually go (L2 hit rate, sector
# efficiency), Occupancy + LaunchStats for the CTA-starvation hypothesis, and
# WarpState/Scheduler for what the warps are waiting on.
SECTIONS=(
  --section SpeedOfLight
  --section SpeedOfLight_RooflineChart
  --section MemoryWorkloadAnalysis
  --section MemoryWorkloadAnalysis_Tables
  --section Occupancy
  --section LaunchStats
  --section WarpStateStats
  --section SchedulerStats
)

# ncu takes one --metrics list; a second occurrence replaces the first rather
# than adding to it, so STALLS=1 BYTES=1 together would silently drop the stall
# counters. Accumulate here and append once.
_EXTRA_METRICS=""

# STALLS=1 adds the per-reason warp stall counters. The WarpStateStats section
# alone does not put them in the details CSV, and they are what distinguishes
# "waiting on DRAM" (long_scoreboard) from "waiting on the pipeline"
# (barrier/membar/mio_throttle) once occupancy is no longer the limiter. Costs
# extra replay passes, so it is opt-in.
if [[ "${STALLS:-0}" == "1" ]]; then
  _stall_reasons=(
    long_scoreboard short_scoreboard barrier membar mio_throttle
    lg_throttle tex_throttle imc_miss no_instruction wait drain
    dispatch_stall not_selected selected sleeping misc
  )
  for r in "${_stall_reasons[@]}"; do
    _EXTRA_METRICS+="smsp__average_warps_issue_stalled_${r}_per_issue_active.ratio,"
  done
fi

# BYTES=1 adds the DRAM/L2/sector counters needed to compute read amplification:
# actual bytes fetched divided by the bytes the shard's packed weights occupy.
# The percentage throughput metrics cannot answer this - a kernel reading 30%
# waste at 75% of peak and one reading nothing spare at 75% of peak are
# indistinguishable there, and the two call for opposite fixes. Sector counts
# come along because partial-sector reads are the expected failure mode for a
# 6-bit layout, whose natural granularity divides neither the 32-byte sector nor
# the 16-byte vector load.
if [[ "${BYTES:-0}" == "1" ]]; then
  _byte_metrics=(
    dram__bytes_read.sum
    dram__bytes_write.sum
    dram__sectors_read.sum
    lts__t_sectors_srcunit_tex_op_read.sum
    lts__t_sector_hit_rate.pct
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  )
  for m in "${_byte_metrics[@]}"; do
    _EXTRA_METRICS+="${m},"
  done
fi

if [[ -n "$_EXTRA_METRICS" ]]; then
  SECTIONS+=(--metrics "${_EXTRA_METRICS%,}")
fi

for shape in $SHAPES; do
  n="${SHAPE_N[$shape]}"
  k="${SHAPE_K[$shape]}"
  [[ -z "$n" ]] && { echo "unknown shape '$shape'" >&2; continue; }
  rep="$OUT_DIR/${MODE}_${ARM}_${shape}_m${M}"
  echo ""
  echo "=== $shape  M=$M N=$n K=$k -> $rep.ncu-rep ==="

  # --graph-profiling node: the benchmark times through CUDA graph replay, and
  # without this ncu profiles the graph launch as one opaque node.
  # --launch-skip drops the warmup launches so the captured launch is steady
  # state (cache-warm weights, resolved policy).
  "$NCU" \
    --target-processes all \
    --graph-profiling node \
    --kernel-name "$KERNEL_RE" \
    --launch-skip 12 --launch-count 1 \
    "${SECTIONS[@]}" \
    --export "$rep" --force-overwrite \
    "$PYTHON" "$ROOT/benchmarks/benchmark_dense_gemm_fp6.py" \
      --m "$M" --n "$n" --k "$k" $ARM_FLAGS \
      --warmup 10 --iters 4 --no-check \
    >"$rep.log" 2>&1
  rc=$?

  if grep -q 'ERR_NVGPUCTRPERM' "$rep.log"; then
    echo "ERR_NVGPUCTRPERM - counters still admin-restricted; run scripts/setup_ncu.sh." >&2
    exit 1
  fi
  if [[ $rc -ne 0 ]]; then
    echo "ncu exited $rc; tail of $rep.log:" >&2
    tail -n 30 "$rep.log" >&2
    continue
  fi

  # The FP8 arm self-skips (returns a `skipped=` result) when vLLM's
  # cutlass_scaled_mm is missing or rejects the shape, and ncu reports that as a
  # clean run with zero profiled kernels. Say so instead of writing an empty CSV
  # that reads as a measurement.
  if grep -qi 'skipped' "$rep.log" && ! grep -q 'DenseGemmKernel' "$rep.log"; then
    echo "arm '$ARM' skipped this shape; see $rep.log" >&2
  fi

  # Flat CSV next to the report so the key numbers are greppable without the UI.
  "$NCU" -i "$rep.ncu-rep" --page details --csv >"$rep.details.csv" 2>/dev/null
  if [[ ! -s "$rep.details.csv" ]]; then
    echo "no kernel matched '$KERNEL_RE' - nothing profiled for $shape" >&2
    continue
  fi
  echo "--- headline metrics ---"
  grep -E 'Duration|DRAM Throughput|Memory Throughput|Achieved Occupancy|Compute \(SM\) Throughput|Waves Per SM|Block Limit|L2 Hit Rate' \
    "$rep.details.csv" 2>/dev/null | head -n 20
done

echo ""
echo "Reports in $OUT_DIR"
echo "Open one with:  ncu-ui $OUT_DIR/${MODE}_${ARM}_gate_up_m${M}.ncu-rep"
{
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "mode: $MODE"
  echo "arm: $ARM"
  echo "kernel_filter: $KERNEL_RE"
  echo "m: $M"
  echo "ncu: $("$NCU" --version 2>/dev/null | grep -i version | head -n1)"
  date -u +"captured_utc: %Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active --format=csv 2>/dev/null
} >"$OUT_DIR/evidence_header.txt"
echo "Wrote $OUT_DIR/evidence_header.txt"
