# Prepared attention-output projection qualification

The prepared B12X attention-output projection preserves loader-owned tiled
weights and replicated block scales, selects the existing small-row grouped
GEMM, and binds caller-owned workspace without scale-padding writes.
Quantization, checkpoint values and the mathematical operation are unchanged.

Status: **implemented and qualified** for the checks below. This is not a
claim that every DS4 serving regression is resolved.

## Conditions

DeepSeek-V4-Flash-0731, TP2, five DSpark draft tokens, FP8 KV cache, 4096-token
batch budget, eight request slots, temperature 1/top-p 1. Two RTX PRO 6000
Blackwell Max-Q Workstation GPUs, VRAM +6000, automatic graphics clocks and
325 W limits. Five warmed, unprofiled 30-second windows per decode cell;
five uncached 32K prefill windows. C8 is aggregate throughput.

[Raw evidence](results.json) includes immutable image/source identities,
checkpoint revision, complete launch/benchmark commands, individual samples,
validation results and GPU telemetry. Model responses are omitted.

## Result

| Five-window median | WO-B/storage repair and MoE grid tuning | Plus packed WO-A repair |
|---|---:|---:|
| Uncached 32K prefill, tokens/s | 11,395 | 11,427 |
| C1 output, tokens/s | 191.045 | 196.171 |
| C1 verifier, steps/s | 71.684 | 71.899 |
| C1 accepted length | 2.678 | 2.732 |
| C8 output, tokens/s | 669.416 | 660.109 |
| C8 verifier, steps/s | 246.848 | 247.371 |

All functional checks and measured cells pass. Verifier changes are +0.30%
C1 and +0.21% C8. Acceptance varies in these unseeded requests; output changes
do not establish a distribution change or guaranteed token-rate improvement.
The separate [WO-B/storage comparison](../deepseek_wo_layout/results.json)
shows a 1.61% C1 verifier gain.

A four-step rank-zero trace identifies 184 first-projection calls by their
position after inverse-RoPE quantization and before the second projection.
The repaired first projection uses 96 threads, 48,128 bytes of shared memory,
and averages 11.02 microseconds. The saved community reference is 11.23
microseconds; the frozen KK implementation is 13.41 microseconds with 288
threads and 78,848 bytes. The binding's two padding fills are also absent.
Kernel times can overlap other work and are not an unprofiled speed estimate.

The repaired image still measures 71.899 versus the saved reference's 73.217
C1 verifier steps/s, a 1.80% deficit. Its cause remains outside this qualified
projection repair.

## Correctness

`uv run python -m pytest tests/gemm/test_wo_projection.py -v` passes all
31 CPU/GPU cases in the source-locked serving image. Coverage includes:

- Actual DeepSeek V4 TP2 and V4.1 dimensions, nonuniform checkpoint scales,
  exact packed/generic intermediate parity and an independent dequantized
  FP32 matrix reference for the first projection.
- The second projection's four atomic BF16 split-K partials, checked against
  their valid accumulation orders at the original pointwise tolerance.
- Poisoned scratch, finite/nonzero output, fixed pointers, allocation-free
  CUDA graph replay and variable prefill tails under frozen compilation.

Five complete inverse-RoPE/projection cases at 1, 6, 8, 48 and 4096 rows pass
numerical and graph checks. Their microtimings are **research-only** because
clock event mask `0x400` violates the declared timing envelope; they are not
used for speed claims.

The broader CPU preparation suite reports 533 passes, 59 CUDA skips and one
failure reproduced on unmodified master: the device-reclaim test expects a
cached selection while autotuning is disabled and returns the default.
