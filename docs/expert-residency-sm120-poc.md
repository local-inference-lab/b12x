# SM120 mapped-host expert cache experiment

Status: **research-only, single-layer prototype**. The experiment composes
native prepared SM120 W4A16 operators, fixed VRAM and mapped-host slots, prepared
routing counters and the shared recent-frequency cache policy. It does not add
a production residency backend, automatic calibration or a vLLM serving lane.
The [recorded test spectrum](expert-residency-sm120-spectrum.md) compares static
and adaptive execution across token counts, residency budgets, cold-route
fractions and reuse windows, including transaction pauses and retained raw data.

The runner is
[`benchmarks/moe/sm120_residency_poc.py`](../benchmarks/moe/sm120_residency_poc.py).
It accepts separate ModelOpt NVFP4 gate/up/down expert tensors, including exports
without a safetensors index. The Qwen3.8-Flash-Next-NVFP4 checkpoint is one tested
input. Geometry comes from tensor shapes; expert count, initial hot count, top-k,
layer prefix and planned token capacity are command-line inputs. Activations and
routes are synthetic. This is an operator experiment, not a model quality test.

## Execution and ownership

1. Read one expert layer on the CPU. Validate E2M1 packed weight bytes, logical
   E4M3 K16 scales and positive per-expert FP32 global scales. Concatenate up/gate
   rows and swizzle scale bytes without requantization. Different gate/up global
   scales are unsupported and fail before preparation.
2. Populate a contiguous VRAM slab and an exact-size `cudaHostAllocMapped` slab.
   Each contains weights, block scales and both global scales. Keep allocation
   owners alive until all captured graphs are released. Check the CUDA mapped
   host-memory capability; SM120 recognition alone is insufficient.
3. Declare native A16 weights with
   `WeightPlanConstraints(required_packing="source_native")`. Uniform NVFP4 A16
   still defaults to MMA packing. Prepare both tier operations and the counter
   through `PreparationSession`; retain scratch and compiled callables. Assert
   that the native consumers retain all six original slab field pointers.
4. At execution, a small CuTe helper reads canonical `(tier,row)` entries into
   the native operations' local-row maps and validates IDs before narrowing to
   int32. Each native operation runs against its own physical slab. Cold experts
   execute directly from mapped host memory for that invocation.
5. Discard the operations' separate final outputs. A CuTe reduction selects the
   weighted BF16 FC2 contribution for each original route, adds contributions
   in top-k order in FP32, then casts once to BF16. The W4A16 activation and
   weight-dequantization boundaries stay intact. This differs from the SM103
   MXFP4/A8 ordered-FMA recipe, which must not be substituted here.
6. At a serialized control boundary, snapshot existing device counters, let
   `b12x.moe.residency.ResidencyCacheController` propose a pair, then run the
   existing journaled `_SlotUpdates.exchange` transaction. The runner has no
   concurrent graph submitters. It drains the device, journals both payloads,
   exchanges all fields and publishes the canonical map after copies complete.
   Subsequent replays use the same captured addresses and updated contents.

The benchmark helpers compile before capture and remain outside the public
component catalog. Native compute uses the ordinary preparation API. Private
route-output and transaction interfaces are deliberate experiment dependencies;
a production backend needs its own registered storage/workspace contract.

CUDA mapped-host access on these discrete GPUs travels over PCIe. It does not
establish Grace coherence, Grace-backed TMA legality or B300 performance. See
NVIDIA's [mapped-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

## Run

Use a matching CUDA/CUTLASS environment on one physical SM120 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.moe.sm120_residency_poc \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --prefix model.language_model.layers.0.mlp.experts \
  --experts 16 --hot-experts 8 --top-k 10 --live 1 2 4 \
  --source-revision "$(git rev-parse HEAD)" --receipt /tmp/sm120-cache.json

# Use all experts in the same layer, with an explicit experimental 50/50 split.
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.moe.sm120_residency_poc \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --experts 512 --hot-experts 256 --top-k 10 --live 1 2 4 \
  --source-revision "$(git rev-parse HEAD)" --receipt /tmp/sm120-cache-full-layer.json

python -m pytest tests/moe/test_sm120_residency_poc.py -q
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest tests/moe/test_sm120_residency_poc.py -q
```

The initial hot count is an experiment input, not an automatically derived or
recommended deployment choice. Both tiers must be nonempty. Each device runs an
independent copy of the experiment; this is not tensor parallelism. The runner
also keeps an all-VRAM native control, so VRAM usage exceeds the resident slab.
The journal retains two expert payloads plus before/after maps in pinned memory.

The JSON receipt includes source identity, loaded-field hash, geometry, physical
device identity, slab/journal bytes, canonical count deltas, decisions, committed
generations, copy-byte accounting and observed exchange wall time. Exchange wall
time includes Python/control-plane work; it is not isolated PCIe copy latency.
No throughput advantage is asserted.

## Numerical acceptance

The policy loop requires bitwise equality to the all-VRAM native control for
repeated-expert workloads before and after exchange. Additional mixed, all-hot,
all-cold and invalid-route cases independently reconstruct an ordered FP32 sum
from the native BF16 route outputs and require bitwise equality to that sum.

Splitting native GEMM work into different expert groups does not guarantee
bitwise equality to a single all-VRAM invocation. The full-layer qualification
exposed differences already present in the FC1 activation and FC2 route outputs,
before the experimental finalizer. The mixed-route gate requires relative L2
error at most 0.005 and cosine at least 0.9999 against the native control; the
cosine gate matches existing W4A16 reference tests. A reduced synthetic geometry
also checks the independent dequantization oracle. No weight requantization,
activation quantization or separately rounded tier addition is used to pass.

The [engineering ledger](expert-residency-ledger.md#sm120-mapped-host-cache-proof-of-concept)
records tested source exports, physical cards, exact outcomes and the retained
bitwise-parity failure. These numerical gates establish a bounded operator proof
of concept, not whole-checkpoint quality or a throughput improvement.

## Limits and next gate

The prototype deliberately retains two full operator launches, redundant tier
finalization, one helper map pass and one counter launch. It does not optimize
PCIe traffic, implement asynchronous replacement or run a complete checkpoint.
The selected expert layer fits in VRAM on an RTX PRO 4000, so this does not
establish whole-model memory admission for a model larger than VRAM.

The SM103 static, automatic-profile and optional quiescent-exchange paths keep
their existing behavior and physical qualification gates. B300 qualification
must first validate native static execution and Grace-backed TMA, then compare
the same captured graph before/after exchange. PCIe results cannot replace that
evidence. See the [slot contract](expert-residency-slots.md) and
[shared subsystem](expert-residency-subsystem.md).
