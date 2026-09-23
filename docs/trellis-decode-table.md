# Trellis decode table

Fused W4A16 MoE can decode uniform 2-bit `lut_e4m3` trellis weights with an
intermediate Hadamard through a 64 KiB table in each thread block's shared
memory. The table stores the complete codeword-to-E4M3 mapping. It replaces
the per-window permutation arithmetic; it does not change checkpoint weights,
activation precision, matrix multiplication, or reduction order.

`B12X_TRELLIS_DECODE_TABLE` controls canonical fused-MoE preparation:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use the full table for planned capacities of at most 16 tokens when the shared-memory and cooperative-residency contracts permit it. Other capacities and weight formats keep compact decoding. |
| `compact` | Compute the permutation per window and read the 4 KiB value table. |
| `full` | Require the full table; fail preparation for an incompatible kernel geometry. This is intended for component qualification. |

For an explicit reference run, set the variable before declaring plans:

```bash
export B12X_TRELLIS_DECODE_TABLE=compact
```

Canonical preparation saves the setting in its immutable query and propagates
it to compiler workers. Changing the environment after declaration cannot
change that plan. Binding and replay retain compiled programs and use
caller-owned scratch. The table is process-shared immutable execution data,
not a decoded copy of model weights.

The full table is admitted only for uniform 2-bit `lut_e4m3` weights with an
intermediate Hadamard, one resident cooperative CTA per SM, and enough shared
memory for both the original GEMM staging and the 64 KiB table. It never
reduces the planned resident-CTA count. Large-capacity prefill keeps its
existing staging footprint.

`tests/moe/test_trellis_direct_lut_decode.py` checks all 65,536 codewords and
random overlapping windows against the compact decoder, including shared
tables for each supported bit width. Only uniform 2-bit weights use the full
table in fused MoE; testing the lookup primitive at 3 and 4 bits does not
enable those execution paths.
