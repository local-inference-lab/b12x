# GLM-5.3 M8 Routed-Expert Evidence

## Status and contract

Status: **implemented; not qualified by this record** for the GLM-5.3 Flash
target-model routed-expert shape.

The specialization applies only to ModelOpt NVFP4 SiLU experts with 288
experts, hidden size 4096, intermediate size 512, eight input rows, top-k 8,
grouped external routing, and a materialized or persistent work source. It
separates fixed-geometry route preparation from expert computation. Other
shapes retain the existing dynamic fused path.

The intended execution contract makes host policy immutable after
`plan_tp_moe_execution` returns, precompiles and retains the route-preparation
and compute artifacts for both int32 and int64 route indices, and keeps live
input-row counts and cluster limits out of compiled-kernel cache identities.
This record does not independently qualify that contract during CUDA graph
capture.

## Source identity

The isolated performance comparison used the same vLLM package and changed
only the B12X source range shown below:

| Role | B12X commit | Git tree | Behavior |
| --- | --- | --- | --- |
| Reference | `b85d9e88fcdc1ae8c0dfef2ab907e357f7b53331` | `42f0cc48b605af343e6427829e9c208459e25037` | Packed MXFP8 fill elision, without M8 route/compute specialization |
| Specialized | `f5274e4c369b8252612c5c66118686a3a8e5f234` | `134dfa06eb2b6d3994aeddab4666bbb7bf3e2a92` | Reference behavior plus the M8 route/compute specialization and its safety checks |

The pull-request representation of the specialized source ends at
`83a10936753e2f9aeec2bdb416dc026f7e2caba5`. Commit
`13bbc002f0d4cc1e4ce8b929ff61e2341bdcd880` adds immutable plan metadata and
plan-time artifact compilation without changing the kernel implementation.
The implementation commit retains `MadeBy561 <madeby561@gmail.com>` as author.

## Performance evidence limitation

The historical comparison notes identify the B12X commits and Git trees above,
the GLM-5.3 checkpoint revision
`378ca54585c46542bad1f3cb3ed0d73ae51cdb62`, and this benchmark command:

```bash
python /root/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5051 \
  --model GLM-5.3-Flash \
  --concurrency 1,8 \
  --contexts 0 \
  --max-tokens 8192 \
  --duration 30 \
  --decode-warmup-seconds 5 \
  --temperature 0 \
  --skip-prefill \
  --display-mode plain \
  --no-hw-monitor \
  --no-resume \
  --output RESULT.json
```

They do not identify the per-arm worktrees, image or compiled-artifact hashes,
physical GPU UUIDs and operating modes, or the CUTLASS/PTXAS artifact map. The
retained notes also do not bind a correctness result to each measured arm.
Therefore this document retains no throughput values, ratio, or performance
qualification from that comparison.

## Plan-time correctness evidence limitation

The following test commands were recorded:

```bash
pytest -q tests/moe/test_fused_moe_planning.py
pytest -q \
  tests/moe/test_cute_migration_moe_standard_corpus.py::test_standard_moe_glm53_m8_split_route_compute_live_graph_oracle
pytest -q \
  tests/moe/test_tp_moe_scratch_bindings.py \
  tests/moe/test_w4a8_dynamic_kernel.py
```

The record does not identify the worktree, exact source tree, runtime artifact,
compiled kernels, physical GPU UUID and mode, or raw test output for these
commands. No correctness or CUDA graph-replay qualification is retained from
them.

## End-to-end evidence limitation

The historical integration notes identify reference B12X integration commit
`035a74c2` and planned-artifact integration commit `d393b6a2` with package tree
`d3f504f98ca7f645c322304a4cb3674ffeab6569`. They do not retain the benchmark
command, both worktree and artifact identities, physical GPU UUIDs and modes,
correctness state, or independent raw records needed to bind the reported
serving measurements to those revisions.

The supplemental sustained-decode and cold-prefill measurements likewise lack
independent commands, source and worktree identities, artifact identities, GPU
identities and modes, correctness state, and raw samples. This document
therefore retains neither those numbers nor a decode or prefill regression
verdict.
