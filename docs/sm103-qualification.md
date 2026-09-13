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
| KDA/GDN | Implemented CuTe decode and sequential prefill; SM120 correctness, state-pool, and graph tests; SM103 compilation | Physical SM103 execution; GDN chunk-parallel algorithm remains unsupported |
| Dense MLA | Implemented BF16/E4M3 compressed-cache attention for (QK,V) widths (576,512) and (1088,1024); SM120 tests and SM103 compilation | SM103 correctness, high-pid, split, query-quantization, and graph qualification |
| Unquantized projections | Implemented BF16/FP32 SIMT and BF16 warp-MMA/TMA paths; SM120 tests and SM103 compilation | SM103 numeric and graph qualification |
| GLM sparse NSA/MLA | Implemented planned FP8/BF16 warp-MMA path for packed GLM NSA and GLM Next FP8/NVFP4 caches; SM120 correctness and sanitizer checks; SM103 compilation | Physical SM103 numeric, high-pid, graph, and resource qualification |
| DSA indexer | Implemented FP8 scoring and exact radix selection; inline BF16 MXFP4 decode/prefill with the V4.1 rounding contract; SM120 regressions and SM103 compilation | SM103 score/top-k, high-pid, graph, and cooperative-merge qualification |
| Quantized linears | Implemented NVFP4/MXFP4/MXFP8 tcgen05/TMEM GEMM, inline W4A16/W8A16, tensor-scaled FP8, compact K128 block-FP8 warp MMA, and planned BF16/FP16 block-FP8 linear | Physical SM103 numerics, grouped strides, boundaries, frozen resolution, and graphs; MXFP6 remains unsupported |
| DFlash2, full GLM/V4.1, HBM GDR | Unsupported as complete execution paths | Integration after operator qualification |

Native block-scaled MoE on SM103 uses tcgen05 and TMEM; SM120/SM121 use warp MMA. The architecture
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
scratch layout and graph lifecycle remain shared. Ordinary BF16 and E4M3 warp
MMA also compile for SM103; the dense MLA and unquantized projection paths reuse
those instructions. Architecture-specific SM12x block-scaled MMA requires a
different SM103 implementation. Recurrent kernels reuse portable SIMT and
BF16 warp-MMA instructions. Engram gather and communication require their
platform-specific execution checks. CuTe/CUTLASS
supplies matrix, layout, TMA and synchronization primitives; b12x owns routing,
fusion, capacity, lifecycle and policy. Generic NVIDIA attention/linear backends
remain appropriate integration fallbacks where b12x has no SM103 implementation.

## Bring-up and compilation

Use a clean checkout and an isolated environment. The dependency pins in
`pyproject.toml` select CUTLASS DSL 4.6.2. Install a CUDA-enabled Torch build that
supports B300, plus `pytest`, `triton`, and profiling tools.

Full source archives also support compilation and qualification. Git exports
retain their base revision in `.git_archival.txt`; archives without that file
record an unavailable revision. Both tools hash the actual package sources and
resolve paths from their source root, including when launched from another
directory. Archive receipts have no Git working-tree status. Use a full source
archive or checkout for qualification because wheel installations omit the test
and script directories.

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
records source/toolchain identity and per-file hashes. The representative corpus
contains nine MoE launchers, eight TP2 communication launchers, three reconstruction
launchers, 36 recurrent launchers, 17 dense MLA launchers, 45 GLM sparse MLA
and cache-writer launchers, 58 indexer launchers, ten unquantized projection launchers,
36 quantized-linear launchers, 19 tensor/compact FP8 launchers, and 28 MXFP8
activation-quantizer launchers: 269
callables in total. This is not an exhaustive specialization census. The GLM MoE compile defaults are K=4096, N=2048, E=288, top-k=8,
capacity=8. `--capacity 128` exercises a separate prefill capacity. No CUDA
context is needed for this offline command. Successful compilation does not
establish valid runtime descriptors, numerics, ordering or performance.

`scripts/qualify_sm103.py` prepares an execution manifest without inspecting
CUDA. With `--execute`, it requires an explicit physical GPU UUID and rejects
any target other than SM103. It records package/test source identities,
device identity, GPU snapshots, test logs, and JUnit counts. A failed or skipped
required test prevents an operator qualification result. Optional compile
manifests must match the source and every retained artifact hash.

```bash
python scripts/qualify_sm103.py --output-dir /tmp/sm103-prepared \
  --compile-manifest /tmp/sm103-compile/manifest.json
# Execute only on the selected physical B300 after transferring the checkout.
python scripts/qualify_sm103.py --execute --device-uuid GPU-actual-B300-UUID \
  --output-dir /tmp/sm103-runtime
```

Use `--component kda_decode --sanitizer /path/to/compute-sanitizer` to run the
same suite under memcheck; `--sanitizer-tool synccheck` checks synchronization.
Full-model serving, compressed DeepSeek sparse MLA, MXFP6 linears, Trellis experts,
Grace/Station behavior, and plugin installation are explicitly outside this
operator suite. Passing it does not enable a model-wide serving route.

## Quantized projections

`gemm.blockscaled.mm` selects the dense SM103 tcgen05/TMEM implementation for
NVFP4, MXFP4, and MXFP8. W4A16 and W8A16 preserve BF16 activations and inline
weight dequantization in the shared warp-MMA engine. W4A16 applies only the
weight global scale; quantized NVFP4 additionally applies the activation
quantizer multiplier. Packed weights, precision policy, prewarm, caller-owned
output, and quantization workspace use the existing public interfaces.

The dense native pipeline shares its TMA, scale staging, TMEM allocation, and
accumulator lifecycle with routed NVFP4 MoE. Dense CTAs compute 128x128 tiles;
they do not create route IDs or pad each source row into a separate MoE route.
Native values are K-major, K is divisible by 128, and N by eight. Scale storage
is F8_128x4. Grouped output is physical `[groups,M,N]` viewed as `[M,N,groups]`.
Live M and group strides are runtime arguments; compile keys contain immutable
weight geometry, recipe, dtype, and device/toolchain identity.

The precision generator races actual public A16 and native quantized calls.
Its NVFP4 oracle independently rounds activations to E2M1/E4M3 and evaluates
FP32 GEMM. It excludes the SM12x fused activation-quantization candidate on
SM103. Candidate contract version 4 invalidates incompatible checkpoints;
serialized precision configs and existing measured SM12x profiles retain their
schema. SM103 has no measured embedded profile. Its initial tiny-M A16 routes
are unmeasured defaults.

`tests/gemm/test_sm103_blockscaled.py` covers inline A16, independent native
oracles, M1/M3/M8/M129/M257 reuse under frozen resolution, two-group strides,
N/K tails, FP32/FP16/BF16 output, graph mutation, stable addresses, and replay
allocation. A large-output case allocates about 4 GiB and verifies the CTA row
whose element offset begins at 2^31; output row and group strides use Int64.
The native cases require physical SM103. They are included in:

```bash
python scripts/qualify_sm103.py --component blockscaled --execute \
  --device-uuid GPU-actual-B300-UUID --output-dir /tmp/sm103-blockscaled \
  --sanitizer /path/to/compute-sanitizer
```

Run synccheck after memcheck, then use the existing precision benchmark and
profile generator. SM103 timing qualification requires P0, a zero throttle mask,
stable memory clocks, and an SM-clock delta no greater than 30 MHz. Compile
success and SM120 A16 regressions do not establish B300 correctness or latency.
MXFP6, fused activation quantization, and
SM12x-specific launch overrides remain rejected. Complete draft/target serving
remains unsupported.

The [quantized-linear validation receipt](sm103-blockscaled-validation.json)
binds the compile corpus, SM120 regression logs, and wheel to package source.
The 36 linear callables report no stack or local memory. Shared-pipeline MoE
projections use 142 allocated GPRs, up from 138 for Int32 routes and 140 for
Int64 routes in the preceding corpus. Those positive deltas remain flagged
for target profiling. Static resource reports omit dynamic launch SMEM and
do not establish occupancy or latency.

## Recurrent, dense MLA, and unquantized projection contracts

SM103 GDN planning selects CuTe for both Qwen's 3:1 value/key-head recurrence
and GLM/Kimi's equal-head KDA recurrence. A Triton core override is rejected on
SM103. SM12x retains its existing KDA heuristic; an explicit CuTe override
selects the portable implementation. KDA uses per-coordinate lower-bounded
decay, FP32 working state, BF16 or FP32 checkpoint storage, BF16 recurrent
output, and FP32 gated RMSNorm before final BF16 storage. Triton supplies only
transactional metadata validation for this KDA path. Bound row/column capacities,
strides, scale, lower bound, and live device counts remain runtime arguments.
Tests cover smaller bound views and mutable counts under frozen resolution,
accepted-draft restarts, null slots, malformed metadata, parameter dtypes,
noncontiguous beta, graph allocation, and state offsets beyond 2^31 elements.

Sequential KDA and GDN prefill retain the shared chunked CuTe implementation,
including checkpoints and long sequences spanning workspace windows. SM103
rejects the separate GDN chunk-parallel algorithm; profile generation excludes
its candidates for that target. Existing embedded GDN decode profiles retain
their measured configs with config schema 4; their measurements do not qualify
the CuTe KDA backend. B300 profiles remain unavailable pending GPU measurements.

Dense MLA consumes the compressed cache contract described by its public Caps;
it does not implement GLM Next's distinct 512-wide sparse NSA contract. It
retains FP8/BF16 compute, caller-owned split storage, optional query quantization,
window masking, physical record strides, and Int64 pool offsets. Unquantized
projections retain FP32 accumulation and their existing SIMT/warp-MMA dispatch.
Neither path claims tcgen05 fusion or target performance. Four 1088-wide dense
MLA specializations report stack frames of 8–192 bytes in the SM103 resource
census. These cases require target profiling; compilation and SM120 correctness
do not settle their SM103 latency. Static SMEM reports omit dynamic launch
storage, so the census does not establish achieved occupancy.

## GLM sparse attention

The public sparse-MLA planner selects `SparseMlaConfig(backend="warp")` on
SM103. Supported recipes are GLM NSA with 576-wide queries and GLM Next with
512-wide queries, both with 512-wide values. BF16 queries consume packed FP8
or NVFP4 records directly. FP8 QK uses ordinary warp MMA followed by FP32
group scaling; NVFP4 QK and PV use inline BF16 dequantization. SM12x retains
its native backend unless a policy override selects the portable path.

Decode uses a fixed split count resolved from planned capacity. Extend uses
the shared multigroup prefill pipeline. Caller-owned scratch contains partial
outputs and LSE storage. Tests cover eight-head tails, head-major output,
zero-row bindings, frozen resolution across live row counts, mutated selected
lengths during graph replay, stable addresses, and no replay allocation.
Physical cache strides and pool offsets use Int64; high-pid cases place live
records beyond the 2 GiB byte-offset boundary without repacking the pool.

The SM103 compile corpus covers 30 attention specializations, five merge/LSE
launchers, and ten cache writers. The SM120 regression suite uses the same
public planner and kernels. Synchronization checks cover both the portable
backend and existing shared-prefill users. Producer and consumer branches use
`barrier.cta.sync` for their shared named barrier because they reach different
instruction sites; the aligned `bar.sync` form requires the same instruction.
See NVIDIA's [barrier instruction contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier).

Six sampled sparse-prefill specializations report nonzero stack frames in
SM103 resource reports. Those flags require target inspection and profiling;
SM120 correctness does not establish SM103 resource behavior or performance.
The source-bound evidence is recorded in [the sparse-attention validation
receipt](sm103-sparse-validation.json). The earlier
[portable-operator receipt](sm103-validation.json) retains its own source hash
and describes that historical source revision.

## DSA indexing

The public DSA plan selects `DsaIndexerConfig(backend="warp")` on SM103.
FP8 scoring uses ordinary E4M3 MMA in the paged, fused score/select, and
contiguous-prefill kernels. The FP8 recipe applies its scales outside the dot
product; unit block scales are unnecessary. Exact radix selection and logical
or physical output-index semantics remain shared. Fused eligibility and split
counts are fixed during planning. SM103 inherits conservative shape limits
without a measured target performance claim.

MXFP4 uses the existing CuTe inline BF16 dequantization and warp-MMA scorer
for both decode and prefill. It preserves separate BF16 rounding of the dot,
weighted product, and head sum, plus caller-owned TP reduction before selection.
Full scanning, source block8 candidates, and bounded candidate reindexing use
the shared public `score`/`select` lifecycle. SM12x retains its block-scaled
MXFP4 prefill schedule unless a warp policy override is supplied.

The portable tests exercise the public API with exact selected-index sets,
score oracles, nonzero results, high page IDs, frozen resolution across live
row counts and pool extents, graph mutation, and allocation checks. The fused
cooperative merge publishes every warp's histogram writes before its CTA
arrival announcement. A delayed-warp test checks this ordering directly; the
pre-port implementation fails that test because its announcement can overtake
other warps' writes. Source-bound compilation and regression evidence is in
[the indexer validation receipt](sm103-indexer-validation.json).
Twenty-one sampled indexer specializations report stack frames of 8–312 bytes
and zero reported local storage. The resource flags remain in the receipt;
target occupancy and latency require physical SM103 measurements.

DSA config schema 2 admits the warp backend. Embedded SM12x profiles retain
their measured native configs. Offline profile generation selects candidates
through the component policy and uses candidate contract version 2 for the
merge race. No measured SM103 profile is embedded.

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

## Tensor-scaled and compact block-FP8 projections

Status: **implemented; awaiting physical SM103 qualification**.
`gemm.tensor_fp8_linear` and its `gemm.blockscaled` alias consume E4M3 inputs
and packed tensor-scaled weights. `gemm.blockscaled.mm_block_fp8` consumes
E4M3 inputs with compact FP32 activation scales `[M,K/128]` and weight scales
`[N/128,K/128]`. It rescales each K128 partial sum before accumulation. This
contract is distinct from the planned `gemm.block_fp8_linear` API, which
quantizes BF16/FP16 activations and consumes prepared MXFP8 weights.

The FP8 entry uses ordinary E4M3 warp MMA and the shared TMA pipeline. Tensor
scaling uses only the scalar output multiplier; it allocates and transfers no
unit block scales. Omitted alpha compiles out the scalar load. K is divisible
by 128 after packed-weight padding. Compact block scaling requires N divisible
by 128 and one group. Tensor scaling supports aligned grouped rows and
arbitrary N with one group. Unaligned rows and FP32 outputs use direct stores.

An immutable weight shape and output dtype select a conservative tile. Live M
and activation/output group strides are Int64-addressed launch arguments.
No live count enters kernel resolution. Raw calls accept caller-owned output;
packed functional calls allocate output during capture and retain its address
for replay. Prewarm before freezing resolution or capturing graphs.

```bash
python scripts/compile_sm103.py --component fp8 --output-dir /tmp/sm103-fp8-compile
python scripts/qualify_sm103.py --component fp8 --execute \
  --device-uuid GPU-actual-B300-UUID --output-dir /tmp/sm103-fp8 \
  --sanitizer /path/to/compute-sanitizer
```

The suite checks independent FP32 oracles, scalar and block-scale mutation,
BF16/FP16/FP32 outputs, grouped capacity strides, N/K tails, frozen resolution
at M1/M4/M8/M17/M65/M257, output poisoning, stable addresses, and replay
allocation. Two cases allocate about 4 GiB each to check output rows beyond
2^31 elements. The shared entry runs on physical SM12x for regression without
pretending that device is SM103. See the
[FP8 validation receipt](sm103-fp8-validation.json) for source-bound evidence.

## Planned BF16/FP16 block-FP8 projections

Status: **implemented; awaiting physical SM103 qualification**.
`gemm.block_fp8_linear` retains its public pack/plan/bind/run lifecycle. SM103
planning selects `mxfp8_tcgen05` with a 128x128 tile. The 128x128 checkpoint
recipe and the 32x32 V4.1 recipe expand weight scales once into F8_128x4
storage. Logical K is divisible by 32 and pads to 128; N must be divisible by
eight. V4.1 activation quantization floors each logical K32 group's amax at
1e-4. Padded groups reconstruct as zero. SM12x keeps its existing warp-MMA
tiles and tiny-M fused quantization path.

The SM103 activation quantizer uses eight-lane groups and 128 threads for
every live M. Its cache includes device and architecture identity. Row and
scale-tile offsets use Int64. Bound calls use caller-owned scratch and output;
an explicit stream governs quantization, GEMM, and bias addition. Public
prewarm prepares bound and functional calls before frozen resolution or graph
capture. Functional calls allocate during capture and retain their output
address for replay.

The component owns config schema 3 and candidate contract 2. Embedded SM12x
profiles retain their measured configs with the updated schema; no B300 profile
is fabricated. The generator selects one native candidate on SM103 and races
the existing tiles on SM12x. Its independent Torch oracle quantizes activations
and expands the checkpoint scales without using the production quantizer or
weight packer. Finite/nonzero output, cosine, and relative L2 checks precede
timing. A deliberately wrong reference test verifies that failures are rejected
before timing.

```bash
python scripts/compile_sm103.py --component block_fp8_linear \
  --output-dir /tmp/sm103-block-fp8-compile
python scripts/qualify_sm103.py --component block_fp8_linear --execute \
  --device-uuid GPU-actual-B300-UUID --output-dir /tmp/sm103-block-fp8 \
  --sanitizer /path/to/compute-sanitizer
```

The suite covers BF16/FP16, both checkpoint recipes, K padding, extreme finite
values, zero/tiny groups, independent scale-layout checks, M1/M4/M8/M9/M65/M129
reuse, poisoned output, graph mutation, stable addresses, allocation checks,
stream ordering, and `torch.compile`. Large-offset tests allocate about 6 GiB
and check the source row beginning at 2^31 elements. SM120 regression executes
the shared quantizer and existing public SM12x GEMM path; it cannot qualify
the native SM103 GEMM. See the
[planned block-FP8 validation receipt](sm103-planned-fp8-validation.json).

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
V4.1 compressed attention, mHC and complete MTP execution remain model blockers.

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
| GLM NSA/MLA | `attention/sparse_mla`: physical SM103 qualification of implemented GLM warp paths; compressed DeepSeek sparse MLA remains a separate implementation gap |
| DSA | `attention/dsa_indexer`: physical SM103 qualification of FP8 and MXFP4 score/select paths, cooperative merge, high page IDs, and graph replay |
| KDA/GDN | `sequence/{gdn_decode,kda_prefill,gdn_prefill}`: physical SM103 qualification of implemented CuTe paths; admit chunk-parallel GDN only after its own corpus |
| Dense/draft linears | `gemm/blockscaled/_sm103.py`, `_a16_cute.py`, `_fp8_cute.py`, `gemm/block_fp8_linear`: qualify implemented native block-scaled, A16, tensor/compact FP8, and planned BF16/FP16 block-FP8; implement MXFP6 |
| Trellis experts | `sm103/trellis.py`, `fused_moe/trellis.py`: scale/rotation staging, mixed rates, compressed-to-SMEM pipeline and BF16/FP8 UMMA; no full-model BF16 repack |
| Grace/NIC ordering | `comm/roce/_transport.py`, `_roce_proxy.c`, `_cute_intrinsics.py`: hardware stress, registration and visibility; retain fatal timeout semantics |
| Full serving | existing LIL vLLM per-operation capability routing: complete target hot path before DFlash2 or V4.1 enables b12x globally |

Highest-risk assumptions are TMA descriptor bounds on a single live M row,
TMEM synchronization and lifetime, real-checkpoint calibration/gate order,
BF16 stage tolerances, Grace allocation visibility and NIC-to-GPU ordering.
Compilation cannot resolve those risks.
