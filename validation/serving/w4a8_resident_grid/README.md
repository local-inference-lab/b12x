# Repacked W4A8 decode launch-grid qualification

B12X preparation can select the physical block count of the repacked W4A8
decode kernel. The runtime consumes that selection without recompilation.
The prepared capacity also fixes the intermediate representation: reducing
the live row count must not request a different, unprepared kernel.

Status: **implemented**; the numerical, capture and serving checks below are
**qualified**. The isolated grid extension does not establish an end-to-end
execution speedup. Measuring candidates with their concurrent shared expert
is a separate, research-only integration change.

## Conditions and results

DeepSeek V4 Flash, TP2, five DSpark drafts, batch budget 4096, eight request
slots, FP8 KV, temperature 1/top-p 1. Hardware is two RTX PRO 6000 Blackwell
Max-Q Workstation GPUs, VRAM +6000, automatic graphics clocks, 325 W limits.
Each cell contains five warmed 30-second decode windows or five uncached
32K prefill windows. C8 output is aggregate throughput.

| Median | Prepared attention-output storage repair | Plus repacked grid/capacity changes |
| --- | ---: | ---: |
| Uncached 32K prefill, tokens/s | 11,362 | 11,395 |
| C1 output, tokens/s | 204.789 | 191.045 |
| C1 verifier, steps/s | 71.694 | 71.684 |
| C1 accepted length | 2.863 | 2.678 |
| C8 output, tokens/s | 655.399 | 669.416 |
| C8 verifier, steps/s | 247.623 | 246.848 |

All functional checks and timed cells pass. Verifier changes are -0.014%
C1 and -0.313% C8. Output varies with acceptance in these unseeded requests;
these samples do not establish a distribution change. The automatic choice
changes the target grid from 376 to 188 blocks, without resolving the serving
gap by itself. No GPU-specific grid is pinned by this change.

[Raw evidence](results.json) includes every window, complete launch and
benchmark commands, immutable image/source identities, checkpoint revision,
native-library hashes and coarse GPU clock/power telemetry. The benchmark
source is identified there by SHA-256 and public commit. A version-string-only
difference from that commit is recorded explicitly. The client's KV budget
is an admission limit; both recorded capacities exceed the 65,536 tokens
needed for eight maximum-length context-zero responses. No cell is
capacity-limited or underfilled.

## Numerical and capture contract

```bash
uv run python -m pytest tests/moe/test_w4a8_mx_tp_moe.py \
  -k 'repacked_decode_grid_reaches_launch or compact_n64_capacity_plan' -v
uv run python -m pytest tests/preparation/test_tuning_predicates.py -q
```

The GPU command passes 24 cases in the candidate image identified in
`results.json`. Repacked N1024/K4096 coverage includes capacities 1/8, live
counts 1/2/6/8, M16/M32 tiles, and physical grids uncapped/1/64/128. Preparation
freezes before the first production launch; every live count reuses the same
compiled callable. Three graph replays per count keep tensor addresses
stable, allocate no CUDA storage, leave inactive output rows poisoned, and
produce finite nonzero output with cosine similarity above 0.998 to an
independent quantized reference. Eight compact-N64 cases retain their
existing capacity behavior. This test changes launch geometry, not the
mathematical tolerance.

CPU predicate tests cover 48-SM and 188-SM devices, domain exclusions,
resident bounds, explicit overrides and one compilation identity across
runtime grid choices. The broader CPU preparation suite reports 535 passes,
59 CUDA skips and one failure reproduced on unmodified master: the
device-reclaim test expects a cached selection with autotuning disabled,
whereas the implementation returns the default. This is not an all-green
full-suite claim.
