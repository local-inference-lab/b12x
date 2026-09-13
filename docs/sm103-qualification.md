# SM103 / B300 qualification

Status: **implemented prototype, unqualified on B300**. The normal b12x API
selects a native NVFP4 MoE backend for SM103. Physical SM103 execution, complete
GLM serving, V4.1 serving and Station RDMA remain unqualified. No B300 performance
numbers or measured B300 policy profile are included.

## Support and architecture boundaries

| Area | Implementation and evidence | Required Station work |
| --- | --- | --- |
| Architecture, dispatch, policy, scratch | Implemented; host tests pass | Check actual device identity and launch limits |
| NVFP4 MoE | Native CuTe TMA/tcgen05/TMEM projections, route quantization, SiLU requantization, weighted reduction; cross-compiled | Numeric oracle, TMA bounds, graph replay, profiling |
| Trellis | Existing t256 SQG E4M3 K2/K3/K4 reconstruction; exact SM120 tests and SM103 compilation | Decoder execution; scale/rotation staging and UMMA expert implementation |
| Engram | Existing hashing/lookup plus owning device/mapped/Grace placement; SM120 lookup and graph checks | Grace allocation, visibility, large-table and serving measurements |
| RoCEnante | Explicit experimental Grace TP2 selection; shared peer protocol; cross-compiled GPU kernels | Registration, ordering, epochs, failure behavior, NCCL comparison |
| NSA/MLA, DSA, KDA/GDN, quantized linears | SM103 unsupported; capability and plan gates reject them | Implement and qualify individual backends |
| DFlash2, full GLM/V4.1, HBM GDR | Unsupported as complete execution paths | Integration after operator qualification |

SM100 and SM103 use tcgen05 and TMEM; SM120/SM121 use warp MMA. The architecture
descriptor records 512 TMEM columns and a 227 KiB block SMEM limit for SM103,
versus no TMEM and 99 KiB for SM12x. Physical opt-in limits are checked before
launch compilation. SM100 recognition does not enable an implementation.
The `sm_103a` target is architecture-specific; its cubins are not SM12x binaries.
See NVIDIA's [GPU capability table](https://developer.nvidia.com/cuda/gpus),
[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html),
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), and
[CUTLASS NVFP4 tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py).

The first MoE schedule assigns one selected expert row to a 128x128x256 tile,
uses two TMA operand stages and one TMEM accumulator stage, and materializes
intermediate activations. It uses ordinary native NVFP4 block-scaled MMA; it
does not claim to exploit every enhanced SM103 FP4 mode. The padding and
128-row activation-scale layout are deliberate bring-up costs. TMEM allocation,
register pressure and repeated expert loads need measurement before tiling or
occupancy decisions. Cluster multicast, persistent expert scheduling, and
producer/consumer specialization are future tuning work.

The architecture-independent source/checkpoint schema, loader, geometry,
scratch layout and graph lifecycle remain shared. Warp-MMA linears, MoE,
sparse-MLA dots, and DSA score kernels require architecture-specific execution.
SIMT metadata, reductions, recurrent kernels, Engram gather and communication
are candidates for reuse after compilation and correctness checks. CuTe/CUTLASS
supplies matrix, layout, TMA and synchronization primitives; b12x owns routing,
fusion, capacity, lifecycle and policy. Generic NVIDIA attention/linear backends
remain appropriate integration fallbacks where b12x has no SM103 implementation.

## Bring-up and compilation

Use a clean checkout and an isolated environment. The dependency pins in
`pyproject.toml` select CUTLASS DSL 4.6.2. Install a CUDA-enabled Torch build that
supports B300, plus `pytest`, `triton`, and profiling tools.

```bash
python -m pip install -e '.[dev]'
nvidia-smi -L
nvidia-smi -q
python - <<'PY'
import dataclasses, importlib.metadata, torch, b12x
from b12x.comm.roce import probe_platform
assert torch.cuda.get_device_capability() == (10, 3)
print(torch.cuda.get_device_properties(0))
print(dataclasses.asdict(b12x.architecture_for((10, 3))))
print(dataclasses.asdict(probe_platform('cuda:0')))
for name in ('torch', 'cuda-python', 'nvidia-cutlass-dsl', 'triton'):
    print(name, importlib.metadata.version(name))
print('MoE', b12x.moe.fused_moe.is_supported())
PY
export CUTE_DSL_ARCH=sm_103a
python scripts/compile_sm103.py --component all --output-dir /tmp/sm103-compile
```

The output directory must be empty. Optional `--nvdisasm /path/to/nvdisasm`
and `--cuobjdump /path/to/cuobjdump` retain SASS and resource reports. The manifest
records source/toolchain identity and per-file hashes. Twenty callables comprise
nine MoE launchers, eight TP2 communication launchers and three reconstruction
launchers. The GLM MoE compile defaults are K=4096, N=2048, E=288, top-k=8,
capacity=8. `--capacity 128` exercises a separate prefill capacity. No CUDA
context is needed for this offline command. Successful compilation does not
establish valid runtime descriptors, numerics, ordering or performance.

## MoE correctness, graphs and timing

Implemented numeric contract: source-native ModelOpt NVFP4, BF16 input/output,
SiLU, K and N divisible by 256, local expert IDs, router weights after FC2.
Both int32 and int64 route IDs are supported; invalid IDs contribute zero.
Global activation scales may be scalar or per expert. A16, FP8/MX formats,
expert maps, input router weighting and calibration are rejected for SM103.
The compile key contains geometry and planned capacity, never live token counts.
Caller-owned scratch, bound launch arguments and output addresses survive replay.
SM103 planning resolves policy once at the declared capacity, including AUTO
precision. Warmup counts share that lowering and its original provenance.
Prewarm compiles the retained plan; bind and replay do not resolve policy.

The materialized numeric contract rounds FC1 to BF16 before SiLU, rounds the
SiLU result before NVFP4 requantization, and rounds FC2 before FP32 router
weighting and accumulation. The final output is BF16. E4M3 block scales saturate
at 448 and E2M1 payloads use nearest-even rounding. These boundaries differ
from the FP32 intermediate accumulation in the existing SM12x oracle.
The tests, benchmark and SM103 profile generator all use
`b12x/moe/_shared/kernels/materialized_nvfp4_reference.py`; production execution
does not import this qualification oracle. Real-checkpoint and complete-layer
comparisons remain required to assess the resulting model accuracy.

```bash
python -m pytest tests/architecture -q
python -m pytest tests/moe/test_sm103_pointwise.py tests/moe/test_sm103_nvfp4.py -q
compute-sanitizer --tool memcheck python -m pytest tests/moe/test_sm103_nvfp4.py -q
compute-sanitizer --tool synccheck python -m pytest tests/moe/test_sm103_nvfp4.py -q
python benchmarks/benchmark_sm103_moe.py --tokens 1 4 8 128 \
  --output /tmp/sm103-glm-synthetic.json
nsys profile -o /tmp/sm103-moe --trace=cuda,nvtx \
  python benchmarks/benchmark_sm103_moe.py --tokens 1 4 8 \
  --output /tmp/sm103-glm-profile.json
ncu --set full --target-processes all -o /tmp/sm103-moe-ncu \
  python benchmarks/benchmark_sm103_moe.py --tokens 1 \
  --output /tmp/sm103-glm-ncu.json
```

Tests cover A4 in both gate orders and AUTO in the supported up/gate order.
They use `atol=0.02`, `rtol=0.03` and cosine >0.999
against an independent unpack/quantize/Torch oracle with the specified BF16
stage boundaries. They mutate inputs/routes, poison output, check stable storage
and allocated bytes, and reuse capacity at M1/M4/M8/M2 under frozen compilation.
Collect sanitizer diagnostics before interpreting timings. A TMA bounds fault,
TMEM guardrail trap, wrong scale stripe, swapped gate/up or stale graph output
is a correctness failure. The benchmark records raw graph samples; profiler
runs are diagnostics and must not supply unprofiled latency claims.

Host checks cover FP4 ties, saturated and underflowed E4M3 scales, zero output,
and an analytical FC2/router-rounding case. Portable quantizer tests execute
random, saturated and zero inputs on SM103 or existing SM12x hardware.

For real weights, use a GLM-5.3-Flash checkpoint converted and calibrated to
**ModelOpt NVFP4**. The published
[GLM configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)
describes an FP8 model; it is not itself an NVFP4 checkpoint. Preserve the
loader's packed bytes, declared gate/up order, swizzled E4M3 block-scale planes,
weight global scales and activation calibration. Export the existing API inputs
at the loader boundary:

```python
# packed is the loader's b12x.moe.fused_moe.PackedWeights object.
# activation_rows contains actual BF16 layer inputs, at least 128 rows.
from dataclasses import fields
import torch
torch.save({
    'weights': {f.name: getattr(packed, f.name).detach().cpu()
                for f in fields(packed) if getattr(packed, f.name) is not None},
    'w13_layout': source.w13_layout.value,
    'activations': activation_rows.detach().cpu(),
}, '/tmp/glm-layer-nvfp4.pt')
```

```bash
python benchmarks/benchmark_sm103_moe.py --bundle /tmp/glm-layer-nvfp4.pt \
  --top-k 8 --tokens 1 4 8 128 --output /tmp/sm103-glm-real.json
```

The receipt hashes the bundle and labels synthetic versus checkpoint input.
Routing in this harness is generated; serving qualification must also use
captured real routing and compare the complete layer against its existing
backend. Do not promote a profile on eager-versus-graph equality alone.

## Policy and model qualification

SM103 AUTO policy chooses the implemented materialized NVFP4 strategy. No
measured B300 profile is embedded. Schema 4 permits architecture-specific MoE
backends; SM12x planner values remain unchanged. M1, M2-4, M5-8, M9-32 and prefill
are explicit internal regimes, with one implemented strategy across them.
Monolithic and TMEM-pipelined arms are not selectable until implemented.

```bash
python scripts/generate_gpu_profile.py --device cuda:0 --components moe.decode \
  --dry-run --work-dir /tmp/b300-policy
# After correctness qualification; this is a broad sweep, not a smoke test:
python scripts/generate_gpu_profile.py --device cuda:0 --components moe.decode \
  --work-dir /tmp/b300-policy --output /tmp/b300-moe-profile.json
```

The SM103 generator filters out unsupported recipes, dimensions and route
capacities. Its one candidate still requires real GPU correctness and timing.
Candidate contract version 19 invalidates checkpoints qualified with the
different FP32-stage oracle.
It does not emit an `nvfp4_auto` precision comparison or a full device profile.
Use the generator's partition/resume facilities for expensive corpus runs.

Complete GLM has 45 transformer layers, with 288 routed experts, top-k 8,
hidden 4096 and expert intermediate 2048 in the published text configuration;
the first three layers are dense. MoE geometry support does not qualify its
NSA, indexer, KDA, dense linears or draft model.

The serving sequence is deferred until those backends pass their operation
tests. Use the existing LIL vLLM model interfaces; do not turn on a global
SM103 b12x model gate. No companion serving patch is included in this subset.
The established `plan_weights` / `prepare_weights` / `plan_execution` /
`prewarm` / `bind` / `run` interface is unchanged. Once per-operation integration
is qualified, record exact vLLM/b12x revisions and launch the real checkpoint:

```bash
vllm serve "$GLM_NVFP4_CHECKPOINT" --tensor-parallel-size 1 --enable-prefix-caching
python scripts/bench_serving_tps.py --help
```

Use that benchmark's installed CLI to run C1/C2/C4/C8, short/8K/32K/long context,
target-only first. Check short generation and prefix-cache reuse, NSA cache
records, DSA top-k identity, KDA state mutation and request reordering. Add the
actual DFlash2 checkpoint and its configuration only after target correctness;
verify M8 verifier batches, acceptance/rejection, committed state, MTP feedback
and graph replay. Compare full verifier-step latency and acceptance rates with
the same draft length and sampler. These serving commands are not a claim that
the branch can execute the complete model today.

## Trellis and V4.1

```bash
python -m pytest tests/moe/test_sm103_trellis.py -q
python scripts/compile_sm103.py --component trellis --output-dir /tmp/sm103-trellis
python benchmarks/benchmark_sm103_trellis.py --bits 3 --output /tmp/trellis-k3.json
python benchmarks/benchmark_sm103_trellis.py --bits 4 --output /tmp/trellis-k4.json
```

These commands qualify unscaled t256 reconstruction only, with exact equality.
The offline capacity represents E=384, K=2304, N=5120 FC2 tile storage with
64-bit offsets. It does not compile a complete Trellis expert GEMM. Use existing
`TrellisConfig`, rate tables, codebooks, scale vectors and transform draws for
subsequent mixed K3/K4 expert tests. Compare reconstructed scaled weights,
per-expert outputs, routed sums and native-FP4 model quality; reconstruction
equality alone cannot qualify activation rotations or final serving output.

The published [V4.1 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)
gives 40 text transformer layers, E=384, hidden=5120, intermediate=2304, top-k=6,
three prediction layers and a separate DSpark expert configuration. The routed
transformer count `40*384*5120*2304*3` is 543,581,798,400 parameters. At exactly
3.0/3.2 effective bpw that is **203.84/217.43 decimal GB**. This estimate excludes
shared experts, attention, embeddings, prediction/draft/vision modules and
runtime allocations. A 225-235 GB total residency is not established.

The two Engram tables contain 384,006,168 and 384,016,682 rows. Their 256-byte
FP8 values plus eight scale bytes per row total **202.76 decimal GB**, before
allocator overhead. Inspect actual checkpoint tensor byte counts, mixed-rate
metadata, padding, repacks and allocator peaks before claiming a single-Station
fit. Preserve HBM for KV, scratch and graph pools. Native Trellis projection,
V4.1 attention/indexing, mHC and complete MTP execution remain model blockers.

## Engram placement

NVIDIA documents GB300 HBM, Grace LPDDR memory and coherent CPU/GPU access for
[DGX Station](https://docs.nvidia.com/dgx/dgx-station-development-guide/overview.html).
The [coherency guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html)
also distinguishes the optional discrete GPU. Probe the selected device.
ARM CPU identity plus host atomics, pageable access and host page-table support
are required for `memory='grace'`; an SM103 name alone does not satisfy them.

```bash
python benchmarks/benchmark_engram_memory.py --memory device --tokens 8 \
  --output /tmp/engram-hbm.json
python benchmarks/benchmark_engram_memory.py --memory grace --tokens 8 \
  --output /tmp/engram-grace.json
# One checkpoint-sized table; verify available host memory before this allocation.
python benchmarks/benchmark_engram_memory.py --memory grace --tokens 1 \
  --base-table-size 16000000 --output /tmp/engram-grace-full.json
```

Use the same `--trace` tensor `[samples,tokens,24]` for placement comparisons.
Receipts include table bytes, lookups/token, logical bytes/token, graph latency,
logical requested bandwidth and trace repetition. GPU caching can make logical
bandwidth exceed link traffic; measure C2C traffic separately with platform
counters. HBM row-cache hit rate is null because no row cache exists. Measure
C1 decode impact in the model before choosing placement. The allocation owner
must outlive every binding and graph; close it only after retiring those users.

## RoCEnante TP2

Station ConnectX and coherent memory do not prove HBM registration or NIC/GPU
visibility. Inspect topology, firmware, driver, GID, MTU, routing, memlock and
NUMA placement on both machines. NVIDIA's
[Grace Blackwell GPUDirect guide](https://docs.nvidia.com/multi-node-nvlink-systems/grace-blackwell-cx8-gpudirect-rdma-guide/index.html)
describes prerequisites; it does not qualify this transport implementation.

```bash
nvidia-smi topo -m
ibv_devinfo
rdma link
ulimit -l
python - <<'PY'
from b12x.comm.roce._proxy import load, Layout
print('ABI', load().roce_abi_version())
print('TP2 allocation bytes', Layout(2, 1 << 20).total_bytes)
PY
```

Run on both Stations with `RANK`, `MASTER` and validated HCA/GID environment
values supplied by the operator. Install libibverbs development files and a C
compiler for the proxy. Use an external timeout to bound peer failures.

```bash
B12X_TEST_ROCE_TRANSPORT=grace_mapped B12X_TEST_ROCE_EXPERIMENTAL=1 \
timeout 600 torchrun --nnodes=2 --nproc-per-node=1 --node-rank="$RANK" \
  --master-addr="$MASTER" --master-port=29650 \
  -m pytest -x tests/comm/test_roce_oneshot_gpu.py
timeout 600 torchrun --nnodes=2 --nproc-per-node=1 --node-rank="$RANK" \
  --master-addr="$MASTER" --master-port=29651 \
  benchmarks/benchmark_roce_oneshot.py --transport grace_mapped \
  --experimental-transport --sizes 8192,32768,65536,262144,1048576,4194304 \
  --max-size 4194304 --gather-rows 1,4,8,128 --gather-cols 4096 \
  --output /tmp/station-tp2.json
```

All-reduce compares against NCCL with dtype-specific tolerances; all-gather
requires exact bytes. The suite covers replay, epochs, stream ordering, tail
padding and failures. Benchmark arms alternate NCCL and RoCE eager/graph calls
and preserve raw samples and ratio direction. BF16 is the first transport target;
FP8 compression and direct HBM are unsupported. `transport='hbm_gdr'` raises
before allocating a proxy. Grace TP2 requires `experimental=True` and is never
automatically selected instead of NCCL. Repeat at actual GLM hidden/logit shard
sizes and compare complete C1/C4 serving before changing integration policy.

## Source-level completion order

| Work | Source seam and acceptance condition |
| --- | --- |
| B300 MoE correctness | `sm103/nvfp4_gemm.py`, `pointwise.py`, `launch.py`: oracle, sanitizer, live-capacity and real-weight graph tests |
| Tiny-M/prefill scheduling | `fused_moe/_sm103.py`, `_policy.py`: implement separate strategies, then race M1/M4/M8/prefill under native plans |
| GLM NSA/MLA | `attention/sparse_mla`, `attention/dense_mla`: preserve GLM cache/head traits; add TMA/UMMA execution and split reduction |
| DSA | `attention/dsa_indexer/{kernel,scratch,_policy}.py`: separate scoring from top-k; qualify sparse IDs, ties and pages beyond 32-bit byte offsets |
| KDA/GDN | `sequence/{gdn_decode,kda_prefill,gdn_prefill}`: recurrence, state layouts, null slots, checkpoints and graphs; retain independent-load scheduling |
| Dense/draft linears | `gemm/blockscaled`, `_lib/dense_gemm.py`: use native CUTLASS operations under existing layouts and plans |
| Trellis experts | `sm103/trellis.py`, `fused_moe/trellis.py`: scale/rotation staging, mixed rates, compressed-to-SMEM pipeline and BF16/FP8 UMMA; no full-model BF16 repack |
| Grace/NIC ordering | `comm/roce/_transport.py`, `_roce_proxy.c`, `_cute_intrinsics.py`: hardware stress, registration and visibility; retain fatal timeout semantics |
| Full serving | existing LIL vLLM per-operation capability routing: complete target hot path before DFlash2 or V4.1 enables b12x globally |

Highest-risk assumptions are TMA descriptor bounds on a single live M row,
TMEM synchronization and lifetime, real-checkpoint calibration/gate order,
BF16 stage tolerances, Grace allocation visibility and NIC-to-GPU ordering.
Compilation cannot resolve those risks.
