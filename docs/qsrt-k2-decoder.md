# QSRT-K2 shared-memory decoding

Fused W4A16 MoE can decode coupled-Hadamard, uniform 2-bit QSRT weights
through a 64 KiB table in each thread block's shared memory. The table stores
the complete codeword-to-E4M3 mapping. It replaces repeated integer rank
calculation; it does not change checkpoint weights, activation precision,
matrix multiplication, or reduction order.

`B12X_QSRT_K2_LUT_MODE` controls canonical fused-MoE preparation:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use the direct table for planned capacities of at most 16 tokens when the shared-memory and cooperative-residency contracts permit it. Other capacities and weight formats retain compact decoding. |
| `compact` | Use the procedural rank calculation and 4 KiB staircase table. |
| `shared` | Require the direct table; fail preparation for an incompatible kernel geometry. This is intended for component qualification. |

For an explicit reference run, set the variable before declaring plans:

```bash
export B12X_QSRT_K2_LUT_MODE=compact
```

Canonical preparation saves the setting in its immutable query and propagates
it to compiler workers. Changing the environment after declaration cannot
change that plan. Binding and replay retain compiled programs and use
caller-owned scratch. The table is process-shared immutable execution data,
not a decoded copy of model weights.

The direct table is admitted only for uniform K2 SQG-E4M3 weights with coupled
Hadamard transforms, one resident cooperative CTA per SM, and enough shared
memory for both the original GEMM staging and the 64 KiB table. It never
reduces the planned resident-CTA count. Large-capacity prefill retains its
existing staging footprint.

`tests/moe/test_trellis_direct_lut_decode.py` checks all 65,536 codewords and
random overlapping windows against the compact decoder, including shared
tables for each supported bitrate. Only uniform K2 is enabled in fused MoE;
testing the lookup primitive at K3/K4 does not enable those execution paths.
