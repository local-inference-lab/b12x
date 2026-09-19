# Hierarchical expert residency

The [shared residency subsystem](expert-residency-subsystem.md) owns canonical
placement and optional host cache policy. This guide specifies its SM103
HBM/Grace execution adapter and preparation contracts.


Status: **implemented prototype**. Host contracts, portable Blackwell metadata and
quantization kernels, mapped-host reads, and SM103 cross-compilation are qualified.
The SM103 expert GEMMs, Grace-backed TMA operands, complete operator arithmetic,
and physical performance are **not hardware-qualified**.

`b12x.moe.fused_moe` prepares a static expert placement within the existing
`plan_weights -> plan_execution -> PreparationSession -> bind -> run` lifecycle.
Original expert IDs remain independent of physical tier rows. Placement profiles
are per-layer workload artifacts with versioned hashes and checkpoint identity.
Opt-in [automatic residency](expert-residency-automatic.md) adds device counters,
model-wide byte budgeting, convergence, profile reuse and restart signaling around
these same static plans. The automatic guide includes SM103 startup, configuration,
engine hooks, TP semantics, monitor mode and physical qualification commands.
An independently opted-in [quiescent slot exchange](expert-residency-slots.md)
replaces payloads and updates this map at an engine-owned pause while retaining
captured addresses. Static plans and automatic-profile activation remain unchanged.
An experimental [recent-frequency policy](expert-residency-cache.md) composes the
existing counters and exchange: Grace serves a miss in the observed window,
then a selected promotion makes later executions use HBM. The engine still owns
every pause; no policy or observer is added to static serving.

## Supported contracts

| SM103 routed path | Status |
| --- | --- |
| ModelOpt NVFP4, A4, BF16 SiLU | Existing native materialized backend |
| Trellis and EXL3, A16 | Existing native backend; format-specific eligibility applies |
| Native MXFP4 E2M1/E8M0 K32, A8 | Hierarchical preparation variant implemented |
| MXFP4, A16 routed execution | Unsupported by the hierarchical variant |
| NVFP4 residual W4A8 and MXFP6 routed execution | Unsupported by the hierarchical variant |
| Dense MXFP4 GEMM | Separate existing capability; does not establish routed-MoE support |

The hierarchical variant accepts BF16 activations/output, FP32 routing weights,
contiguous int32 or int64 route IDs, SiLU with an optional clamp, and either W13
(up/gate) or W31 (gate/up) source order. The public weight planner requires H
divisible by 256; this backend additionally requires I divisible by 128. A plan
supports every positive live M and top-k within its declared capacities. Live
counts are launch arguments and never compiler keys.

Weights are packed `[E, 2*I, H/2]` and `[E, H, I/2]`; scales are logical,
**unswizzled** `[E, 2*I, H/32]` and `[E, H, I/32]` E8M0 bytes. A loader must expand
checkpoint row-block scale broadcasting into this existing K32 contract without
changing values. Checkpoint FP4 bytes are copied unchanged. Global/activation
scale metadata must be unit-valued for this MXFP4 recipe. Nonunit values fail
preparation. Biases, SITU, FP16 output, router-weight-on-input, logits routing,
expert parallelism, and concurrent adaptation are unsupported in this variant.

## Declaration and preparation

```python
import torch
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from b12x.moe.fused_moe.residency import read_profiles

placement = next(p for p in read_profiles("placement.json") if p.layer == layer_name)
weight_plan = moe.plan_weights(
    source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
    activation=moe.ActivationSpec(
        mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16,
        swiglu_limit=activation_limit,
    ),
    geometry=moe.MoEGeometry(
        num_experts=num_experts, hidden_size=hidden, intermediate_size=intermediate,
    ),
    constraints=moe.WeightPlanConstraints(required_packing="source_native"),
)
# CPU checkpoint views; the loader verifies the fingerprint before constructing this bundle.
weights = moe.PackedWeights(
    w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
    w13_global_scales=unit_scales, w2_global_scales=unit_scales,
    checkpoint_fingerprint=checkpoint_fingerprint, layer_name=layer_name,
)
plan = moe.plan_execution(
    experts=weight_plan, weights=weights, placement=placement,
    capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=max_top_k),
    memory_budget=moe.ExpertMemoryBudget(
        hbm_bytes=layer_hbm_budget, grace_bytes=layer_grace_budget,
        hbm_safety_bytes=hbm_reserve, grace_safety_bytes=grace_reserve,
        kv_reserved_bytes=kv_reserve,
    ),
)

def prepare_call(state):
    binding = state.bind(a=warmup_x, topk_ids=warmup_ids, topk_weights=warmup_weights)
    return PreparedCall(run=binding.run, output=binding.output, owners=(state, binding))

with PreparationSession(device="cuda:0", autotune=False) as session:
    session.prepare((plan.request(name=layer_name, prepare_call=prepare_call),))
    session.freeze()
    binding = moe.bind(plan, a=x, topk_ids=ids, topk_weights=route_weights)
    output = moe.run(binding=binding)
    # Capture/replay while the session, binding, and their storage owners remain alive.
```

`ExpertResidencyPlan` declares the complete expert partition, physical row order,
layer, model fingerprint, workload, provenance, optional measured selection
counts, and artifact version. Its `expert_map` is original ID -> `(tier, local
row)`, with HBM=0 and Grace=1. Its hash covers all metadata and counts. The source
bundle's checkpoint fingerprint and layer must match the profile. The loader is
responsible for authenticating checkpoint bytes; a matching metadata string is
not a content verification algorithm.

`ResidencyQuery` and `ResidencyConfig` register the `residency` variant of
`moe.fused_moe` in the preparation catalog. The query retains geometry, capacity,
profile/model identity, source order, clamp, and numerical mode. Defaults and
explicit pins use the same real compilation, materialization, and priming hooks.
There is one eligible baseline configuration; it is not a measured tuning winner.
Bindings reject undeclared layouts, aliases, excess capacity, unprepared plans,
and bindings from a released or replaced preparation. Serving never resolves a
kernel, allocates storage, copies checkpoint data, or reads route counts on the
host.

## Storage, admission, and ownership

Preparation allocates one HBM expert slab and one mapped-host expert slab when
those tiers are nonempty, plus a fixed HBM workspace and map. It reuses
`sequence._shared.disk_table.MappedHostAllocation`: `cudaHostAlloc` with mapped,
write-combined storage and an accelerator alias. The CUDA request has the exact
slab byte count; this avoids Torch's pinned-pool size classes. CUDA/OS page
rounding is not asserted to disappear. Source copying and scale swizzling operate
one expert at a time. There is no model-sized HBM staging copy or weight
requantization.

Grace storage requires `probe_platform(...).grace_coherent`: SM103 recognition
alone is insufficient. Nonempty cold tiers fail closed without the existing
host-native-atomic and host-page-table capability checks. The mapped allocation
owner stays alive through prepared state and bindings. Release graphs before
releasing their session/bindings. An execution state owns one workspace and
therefore requires serialized execution; concurrent streams require separate
plans/workspaces.

`ExpertMemoryAccounting` reports HBM expert bytes, mapped expert bytes, scratch,
route-map bytes and optional exchange rollback bytes. `grace_total_bytes` includes
that rollback storage; static plans charge zero. Every slab offset includes alignment. `Plan` memory
requirements report its owned HBM allocation. Admission checks declared budgets
before allocating and free device/host memory during materialization. KV and
safety reservations reduce available capacity. `ModelExpertMemoryBudget` and `ResidencyController` can apportion model-wide
budgets automatically; manual integrations must avoid double-reserving a global
KV pool. All private workspaces remain charged. Shared-lane workspace estimates
are diagnostic and do not authorize aliasing.

For one expert, payload storage is `3*H*I/2` weight bytes plus `3*H*I/32` scale
bytes. Workspace includes route indices/counts, route-major quantized activations,
expanded TMA activation scales, FP32 FC1, BF16 FC2, and output. These intermediates
are intentionally visible in accounting. Borrowed CPU checkpoint views remain
referenced by the declaration; their memory and the bounded CPU scale-swizzle
staging are outside owned-tier accounting. File-backed checkpoint views allow
ordinary source pages to be reclaimed by the OS. The integration must budget
ordinary resident source tensors separately.

## Numerical and execution contracts

1. A single native pass maps valid original IDs into compact tier-local IDs and
   original route indices. Duplicates remain independent routes. Negative and
   out-of-range IDs are omitted without first truncating int64 IDs.
2. BF16 inputs are quantized to E4M3/E8M0 K32 using the existing power-of-two scale
   boundary. FC1 uses SM103 mixed FP8/FP4 tcgen05 with FP32 accumulation/output.
3. Gated SiLU and optional clamp remain FP32 before the second MXFP8 boundary.
   FC2 consumes original MXFP4 weights and stores unweighted BF16 expert outputs
   at their original route indices.
4. One CuTe finalizer loops through the original top-k ranks and issues explicit
   `fma.rn.f32(weight, expert_output, accumulator)`. It casts once to BF16. Invalid
   routes are skipped even when their weight is NaN. Weights are not normalized
   or converted to BF16.

Tier partitioning never changes either quantization boundary or the final
summation order. The declared numerical mode is
`mxfp8_fp32_activation_bf16_expert_ordered_fma`. Adversarial portable tests prove
that separate BF16 tier finalization, separate FP32 multiply/add, and reordered
FP32 sums can differ from this contract. Whole-operator and checkpoint parity
still require physical SM103 tests; the finalizer proof does not establish full
model quality or equality with every FlashInfer backend.

Empty tiers are omitted at preparation. For a mixed placement, each projection
still launches both tier programs with a bounded grid. A CTA-uniform device-count
guard skips inactive slots before reading compact metadata, operands, barriers,
or TMEM. This avoids full-capacity cold GEMM work, but does not eliminate the
empty CUDA launch or its CTA scheduling cost. The route partition baseline uses
one lane for a deterministic scan. Shared input quantization, smaller M tiles,
persistent scheduling, demand caching, and overlap remain optimization work.

## Offline profiles

Routing JSONL records have the form:

```json
{"layer":"layer.7","phase":"decode","expert_ids":[17,3,17,-1]}
```

The geometry file maps layer names to expert counts. Counts preserve duplicates,
skip `-1`, and reject malformed IDs. Each layer ranks independently; ties prefer
the original expert index. Both per-layer Python budgets and CLI count/byte
budgets are supported. Byte budgets divide by the prepared per-expert byte cost;
whole-plan scratch and reserves remain subject to preparation admission.

```bash
python scripts/build_expert_residency_profile.py routes.jsonl placement.json \
  --geometry experts-per-layer.json --hot-count 295 --phase decode \
  --model-fingerprint CHECKPOINT_SHA256 --workload agent-prose \
  --provenance TRACE_SHA256
```

The count in this example is a qualification choice, not a contract. The artifact
contains each profile hash and selection counts; the command reports expected
cold-selection fractions. Zero observed selections report an unknown fraction.
`profiles_from_trace`, `profile_from_counts`, `read_profiles`, and `write_profiles`
also support programmatic use. Static profiles are the default. Automatic profiling uses explicit prepared
counter nodes only in opted-in calibration or monitor graphs; normal serving
adds no telemetry. Model artifact schema 2 adds geometry/recipe compatibility and
atomic workload-specific storage. Automatic placement uses balanced cold-start
coverage, joint HBM/Grace feasibility and convergence-required activation by
default; bounded experiments are saved separately from accepted profiles. `read_profiles` can extract static layer plans
from either artifact schema; automatic reuse performs the complete validation
specified in the [automatic guide](expert-residency-automatic.md).

## Qualification commands

See [the engineering ledger](expert-residency-ledger.md) for completed evidence
and rejected experiments. Raw compiler and service receipts belong outside the
repository.

```bash
python -m pytest tests/moe/test_expert_residency.py
python scripts/compile_sm103_prepared.py --output-dir /evidence/prepared --workers 2
python scripts/compile_sm103.py --component residency \
  --hidden 5120 --intermediate 2304 --experts 384 --hot-experts 295 \
  --capacity 128 --top-k 6 --swiglu-limit 10 \
  --nvdisasm /cuda/bin/nvdisasm --cuobjdump /cuda/bin/cuobjdump \
  --output-dir /evidence/residency-compile
```

On physical B300/GB300 with verified coherent Grace memory:

```bash
python -m pytest tests/moe/test_sm103_residency.py -v
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest tests/moe/test_sm103_residency.py -v
compute-sanitizer --tool synccheck --error-exitcode 99 \
  python -m pytest tests/moe/test_sm103_residency.py -v
python benchmarks/moe/expert_residency.py --output /evidence/residency.json \
  --hidden 5120 --intermediate 2304 --experts 384 --hot-experts 295 \
  --top-k 6 --swiglu-limit 10 --tokens 1 2 4 8 16 32 64 128 \
  --cold-fraction 0.0155
nsys profile --trace=cuda,nvtx,osrt -o /evidence/residency-systems \
  python benchmarks/moe/expert_residency.py --output /evidence/residency-nsys.json
ncu --set full --target-processes all -o /evidence/residency-compute \
  python benchmarks/moe/expert_residency.py --output /evidence/residency-ncu.json \
  --tokens 1 8 128 --iterations 1 --samples 1
```

The benchmark uses the public preparation/bind/run path, rejects non-SM103
hardware, verifies bitwise agreement against an all-HBM placement, mutates
activations/routes across replay, checks allocator events, then records total
and per-launch graph samples. Stage results separate partition, input quantization,
hot/cold FC1, activation quantization, hot/cold FC2, and finalization. Isolated
stage graphs contain repeated device operations to exclude Python enqueue
gaps; isolated stage sums remain distinct from full-operator time. Ratios are
**tiered latency / all-HBM latency**. Source hashes, worktree/commit, device UUID,
driver/mode, toolchain, checkpoint identity, memory, and actual cold fraction are
recorded; failed attempts retain a receipt.

Repeat with cold fractions 0 and 1 and a reduced geometry. A positional placement
and a trace-derived profile must use identical checkpoint inputs and routing
traces for a placement comparison. Synthetic controlled routing is not a
workload benchmark. `--weights` accepts a CPU `PackedWeights` tensor bundle with
`--checkpoint-sha256`; `--profile` additionally checks its model/layer identity.
This is an operator probe, not an end-to-end checkpoint loader or quality harness.
Explicit worker hooks are provided in
`b12x.integration.vllm.expert_residency`; the companion vLLM serving port and engine
wiring remain required. Greedy checkpoint requests, layer probes, C2C
traffic, achieved occupancy, tensor/TMA utilization, HBM throughput, stalls, power,
and overlap are deferred to physical qualification. No performance benefit is
claimed from compilation or portable tests.

The [serving integration and workspace audit](expert-residency-integration.md)
identifies the maintained companion PreparationSession hooks, CPU checkpoint
ownership required for greater-than-HBM loading, and execution-lane conditions
for future scratch sharing. Those integrations and shared arenas are not
implemented by the orchestration API; private scratch remains fully charged.
