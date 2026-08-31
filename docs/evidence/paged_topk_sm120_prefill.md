# Paged top-k-512 candidate-capacity qualification

## Status and operation contract

Status: **implemented** as an architecture-neutral top-k policy and
**qualified** on NVIDIA SM120.

The top-k-512 selector uses a 1,024-entry shared candidate buffer. Selectors
with top-k 1,024 or 2,048 use an 8,192-entry buffer. Candidate capacity is an
immutable compile-time property derived only from top-k width; selector launch
does not query CUDA capability or request state.

The exact overflow rescan preserves selection semantics when a threshold bucket
contains more candidates than the shared buffer. For a row with fewer live
candidates than top-k, each live candidate appears at most once and unused
entries are `-1`. An indices-only terminal fold leaves the score output buffer
untouched. Intermediate folds continue writing scores because later folds read
them.

Performance qualification is specific to the hardware and production geometry
below. The operation contract and compile-time policy are not architecture
gated.

## Source, hardware, and command

- B12X source revision: `a232cfe96c340bfcaebe61644b069e3a7b789b3a`,
  based on `master` revision `a45d3f7690e5b2f2e9bdcc0f76d76a48a0c490aa`.
- Worktree: `/root/vllm/worktrees/b12x-glm53-merged-260-20260831`.
- Physical GPU 2: `GPU-167fbc3f-fd06-7f08-9e06-ee02946d041c`, NVIDIA RTX PRO
  6000 Blackwell Workstation Edition, compute capability 12.0, 188 SMs, PCIe
  Gen5 x16, stock clocks, and a 600 W power limit.
- Toolchain: CUDA 13.3, PyTorch 2.13.0, CUTLASS DSL 4.6.2, and
  `CUTE_DSL_ARCH=sm_120a`.
- Runtime image: `local/jovian-judgement-community-20260831-r8-reviewed`.

Both arms used the same source, random seed, graph-replay path, inputs,
indices-only output contract, and cache state. The benchmark-only
`--topk-candidate-capacity` argument selected the compile-time capacity:

```bash
CUDA_VISIBLE_DEVICES=2 CUTE_DSL_ARCH=sm_120a \
  /opt/venv/bin/python -B benchmarks/benchmark_paged_indexer.py \
  --rows 4080 \
  --global-heads 64 \
  --tp-size 4 \
  --page-table-width 128 \
  --seq-len 8192 \
  --mode supertile-topk \
  --route paged-tiled \
  --topk 512 \
  --topk-candidate-capacity 1024 \
  --indices-only \
  --warmup 10 \
  --iters 30
```

The 8,192-entry arm changed only
`--topk-candidate-capacity 1024` to
`--topk-candidate-capacity 8192`. One preconditioning process per arm preceded
five measured processes per arm. Measured processes ran in balanced order, and
each process reported the median of 30 graph replays.

## Correctness

The focused selector suite used the production source and 1,024-entry policy:

```bash
CUDA_VISIBLE_DEVICES=2 CUTE_DSL_ARCH=sm_120a \
  /opt/venv/bin/python -B -m pytest -q \
  tests/attention/test_attention_dsa_indexer_api.py \
  tests/attention/test_paged_prefill_topk_long_context.py \
  -k "topk_candidate_capacity or row_topk or paged_prefill_topk"
```

Result: `16 passed, 29 deselected`. The selection covers top-k policy,
indices-only output, short-row padding, exact overflow selection, and CUDA graph
replay with changing live inputs.

## Selector performance

| Candidate capacity | Process medians, microseconds | Median |
| ---: | --- | ---: |
| 1,024 | 1029.38, 1029.73, 1030.16, 1030.14, 1031.10 | 1030.14 us |
| 8,192 | 1044.19, 1045.50, 1044.48, 1045.74, 1044.48 | 1044.48 us |

The 1,024-entry policy reduces selector latency by 1.37%. The ratio direction is
8,192-entry latency divided by 1,024-entry latency:
`1044.48 / 1030.14 = 1.01392`, or 1.39% higher selector throughput.

This comparison isolates candidate capacity. It does not attribute the separate
gain from omitting unused terminal score writes, because both arms use the same
indices-only contract.

## Compiled-resource evidence

The B12X compile manifests and the embedded `sm_120a` fatbins identify the exact
selector objects:

| Candidate capacity | Object SHA-256 | Object bytes | Dynamic shared memory | Registers/thread | Stack | Local memory | Static shared memory |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | `771c9ddf8f831bff42c45846744167faf9e51134cc15c297fc2df48632bf55ce` | 61,288 | 20,096 B | 27 | 0 B | 0 B | 1,024 B |
| 8,192 | `3302781c9b37e18662de61381d25bdf4db96c1b4366da7ac30ab90772760b5e3` | 63,576 | 77,440 B | 27 | 0 B | 0 B | 1,024 B |

`cuobjdump --dump-resource-usage` reports no stack, spill, or local-memory use
for either object. The GPU permits 1,536 threads, 65,536 registers, and 102,400
shared-memory bytes per SM. Each selector CTA has 1,024 threads, so both objects
are limited to one resident CTA per SM by the thread limit. Both therefore have
32 active warps out of 48, or 66.7% theoretical warp occupancy. The smaller
candidate buffer improves latency without changing CTA occupancy, register use,
or spill behavior.

## Scope

The selector result does not establish end-to-end serving throughput. GLM-5.3
qualification must also include C4 scoring, page-table preparation,
communication, attention, MoE, and the remaining model work.
