# Behemoth-R1-123B-v2 per-GPU shard shapes at TP=2 (hidden=12288,
# intermediate=28672, 96 heads x 128). Sourced by the FP6 profiling scripts so
# the four shard geometries have one definition: every timing, SASS census and
# ncu capture in the FP6 evidence is keyed to these numbers, and a copy that
# drifts silently measures a shape nothing else measured.
#
# Mirrors BEHEMOTH_TP2_SHAPES in benchmarks/benchmark_dense_gemm_fp6.py and
# _SHARD_NK in scripts/summarize_ncu_details.py; those two are Python and
# cannot source this file, so they must be updated in step.

declare -A SHAPE_N=([qkv]=7168  [o]=12288 [gate_up]=28672 [down]=12288)
declare -A SHAPE_K=([qkv]=12288 [o]=6144  [gate_up]=12288 [down]=14336)
