# Qwen packed-MXFP8 capacity policy evidence

This report measures BF16-input/MXFP8-weight projection planning in a
Qwen3.8-Flash-Next-NVFP4 server. Large prefills get a capacity-hinted native
program; short inputs retain the unhinted program. Both are prepared before
execution and share the caller's workspace. Model weights and activation
quantization are unchanged.

Status: implemented and qualified for the numerical/replay tests and bounded
serving conditions below. Native microbenchmark speedups are not qualified.

## Conditions and results

The model is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, revision
`b797d2e1160b9596b2570e56c1d3590faa09d4ed`. One RTX PRO 6000 Blackwell Max-Q
Workstation GPU runs tensor parallelism one, multi-token prediction (MTP) with
three draft tokens, CPU-backed per-layer embeddings (PLE),
6,019-token batches, sixteen request slots, 8 GiB physical KV allocation and
262,144-token maximum context. Memory offset is +6000 (16,365 MHz when busy),
graphics clocks are automatic and the power limit is 325 W. Sampling uses
temperature 1, top-p .95, top-k 20. Decode is unseeded.

Five warmed 30-second windows are retained for every cell. C1 means one
concurrent client; C8 means eight concurrent clients and reports aggregate
throughput. Prefill is uncached 32K input tokens divided by client time to first
token, not a pure GPU-kernel timer. EOS and repetition/error checks remain on.

The isolated baseline already contains the immutable-expert-scale declaration
from [vLLM #810](https://github.com/local-inference-lab/vllm/pull/810). The
capacity-policy arm changes only the B12X packed preparation/tuning files.

| Metric | Scale contract only | Scale contract plus capacity policy | Change |
|---|---:|---:|---:|
| 32K prefill, tokens/s | 11,748 | 12,188 | +3.75% |
| C1 output, tokens/s | 174.416 | 173.194 | -0.70% |
| C1 verifier, steps/s | 82.078 | 84.597 | +3.07% |
| C8 output, tokens/s | 704.526 | 698.033 | -0.92% |
| C8 verifier, steps/s | 334.708 | 336.848 | +0.64% |

All functional checks and timed cells pass. Acceptance changes between runs;
the exact-row A16 plans also undergo independent startup tuning. Therefore the
decode changes cannot all be attributed to the large-prefill program.

A separate compatibility run merges canonical vLLM
`af9e4dca109e0348323c0182e98a3aaf7282bfc3` into the beta integration at
`67bb922f6f401b304abb8a6d9da450445c962662`, and canonical B12X
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` into its beta integration at
`eea3ced11fc14625b683d3c575cf2d702ad3706a`. It adds the immutable-expert-scale
declaration from vLLM #810 and the MXFP8 capacity policy described here.
The resulting source identities are in the `complete_source_composition`
entry of [results.json](results.json). Five windows per cell
pass: prefill 12,118 tokens/s; C1 177.731 tokens/s / 84.059 steps/s; C8
705.608 tokens/s / 340.997 steps/s. This is not an isolated-PR speedup claim.

The historical serving reference called R35 is the Jovian Judgement community
image `localinferencelab/vllm:jovian-judgement-community-20260911-r35`:
12,104 prefill / 157.58 C1 / 674.22 C8 tokens/s under the same model settings.
The frozen Karmic Kraken comparison is the image identified by registry digest
`495b340eede3bbb348fd6c9662d1535e0d2f27e47c08cc71802fea6bc68caf40`:
10,822 prefill / 172.645 C1 / 696.501 C8 tokens/s. These labels name explicit
comparison artifacts, not moving release aliases.

## Public evidence and reproduction

[results.json](results.json) records every prefill and decode window, exact
commands, model/image/source identities, native-library hashes, serving
arguments and runtime environment. It includes physical GPU UUID, timing
intervals, busy-clock summaries and hashes of the originating result files.
Output text is not copied. The `commands` arrays retain exact invocations;
`serving_argv` is the actual process command after launcher expansion.
Its `terms` object defines abbreviations used by the unchanged benchmark
metadata and runtime arguments. In particular, decode context parallelism
(DCP) is one, so attention context is not split across additional ranks.

The candidate-contract version invalidates cached tuning candidates after the
lowering changes; previously selected activation-precision and tile choices
must be measured again.

The recorded model source worktrees were committed; the diagnostic builders
reject tracked modifications, archive Git source and verify installed runtime
files against that source. Measurement ran from `/root/vllm/qwen38next`.
The two isolated source worktrees were
`/root/vllm/worktrees/vllm-b12x-static-expert-scales` and
`/root/vllm/worktrees/b12x-mxfp8-capacity-prefill-regimes`.
The complete composition used
`/root/vllm/worktrees/vllm-karmic-qmax-scale-sync` and
`/root/vllm/worktrees/b12x-karmic-qmax-dense-sync`.

The host benchmark checkout was dirty, so its checkout HEAD is not treated as
the benchmark identity. The frozen script is byte-for-byte the public
[llm-inference-bench source at 80d1f1b0](https://github.com/local-inference-lab/llm-inference-bench/blob/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py)
with only the displayed `VERSION` changed from `0.6.1` to `0.6.2`.
There is no executable-behavior difference. SHA256 of the frozen file:
`053989edff8c9c93e2b96e61342b2ffbd9851e03deba17e6d3fc96fcd6694c1e`.
[benchmark-version.patch](benchmark-version.patch) reproduces that exact file
from the linked public source. No host benchmark checkout was modified.

For example, the recorded isolated-arm commands are equivalent to:

```bash
uv run --with httpx --with rich python llm_decode_bench.py \
  --host http://192.168.0.115:5068 --model Qwen3.8-Flash-Next \
  --display-mode plain --no-hw-monitor --no-resume --respect-eos \
  --temperature 1 --token-targeting exact --max-tokens 8192 \
  --decode-warmup-seconds 15 --kv-budget 448705 \
  --prefill-only --prefill-contexts 32k --prefill-duration 30 \
  --prefill-metric client --output prefill.json

uv run --with httpx --with rich python llm_decode_bench.py \
  --host http://192.168.0.115:5068 --model Qwen3.8-Flash-Next \
  --display-mode plain --no-hw-monitor --no-resume --respect-eos \
  --temperature 1 --token-targeting exact --max-tokens 8192 \
  --decode-warmup-seconds 15 --kv-budget 448705 \
  --skip-prefill --concurrency 1,8 --contexts 0 --duration 30 \
  --output decode.json
```

Use the endpoint and reported logical KV budget of the server being tested.
Top-p/top-k are the explicitly configured server defaults. Execute five
windows per command with separate output filenames, retaining failures;
the published JSON records all filenames and timestamps. Recorded local
image IDs are forensic identities, not registry pull references. A rerun
must build the source composition or use its subsequently published image,
not silently substitute a moving beta tag.

## Numerical and timing limits

Fourteen CPU metadata cases and eighteen GPU cases pass, including both
workspace forms, graph replay, poisoned scratch, exact-row preservation and
multiple live counts under frozen compilation. Twenty-one native comparisons
pass an FP8 numerical oracle and bitwise cross-arm equality across three
projection geometries and the 2,047/2,048-row dispatch boundary.

The native timing protocol requires P1, stable memory clocks, at most 60 MHz
paired graphics-clock delta, and only clock-event masks `0x0` or `0x4`.
It observed `0x400`, so its timing qualification fails. Serving measurements
use automatic graphics clocks and retain actual clock/power telemetry, but
do not sample clock-event masks. Do not turn this report into a claim of
clock-controlled native kernel speedup or universal performance improvement.
