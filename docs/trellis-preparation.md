# Trellis preparation across checkpoint containers

Use `TrellisSource` for encoding and source coordinates, `TrellisWeights` for
tensor data, and `prepare_weights` to create GPU-resident `PreparedExperts`.
The execution API does not dispatch on a checkpoint filename or manifest type.

For an EXL3 checkpoint, the adapter interprets its manifest and returns these
common objects without requantizing the payload. This example prepares eight
32-channel slots from layer 1 of a Kimi-K3 EXL3 checkpoint on GPU 0:

```python
import torch
from b12x.moe import fused_moe
from b12x.moe.checkpoints.exl3 import (
    read_exl3_manifest, read_exl3_layer, trellis_from_exl3,
)

root = "/data/kimi-k3-exl3-2bit"
manifest = read_exl3_manifest(root)
layer = read_exl3_layer(root, manifest, 1, first_slot=0, slot_count=8)
source, weights = trellis_from_exl3(layer)
plan = fused_moe.plan_weights(
    source=source,
    activation=fused_moe.ActivationSpec(
        mode="a16", nonlinearity="situ", io_dtype=torch.bfloat16,
        rotation_dtype=torch.float16,
    ),
    geometry=fused_moe.MoEGeometry(
        num_experts=manifest.geometry.num_experts,
        hidden_size=manifest.geometry.hidden_size,
        intermediate_size=layer.local_intermediate_size,
    ),
)
experts = fused_moe.prepare_weights(
    plan=plan, weights=weights, device="cuda:0",
    staging=fused_moe.TrellisStaging(max_experts=64, max_bytes=64 * 1024 * 1024),
)
```

The caller assigns a legal extent to each tensor-parallel rank before this
step. B12X validates source bounds, tensor shapes, rates, and supported
transforms. This API implementation requires GPU qualification before it
replaces an existing serving deployment; CPU layout tests alone do not verify
CUDA execution or its performance.

## Encoding, data, and arithmetic

- `TrellisSource.config` reuses the native version-2 `TrellisConfig` schema.
  Existing native callers may still pass `TrellisConfig` directly; the planner
  wraps it in a source descriptor. No checkpoint JSON migration is required.
- `TrellisSource.uniform_bits` declares a uniform K value before preparation.
  The actual uint8 rate tensor must agree. Omitting this field retains the
  native K3 LUT or projection-tiered MCG preparation contract. The EXL3 adapter
  sets it from the manifest, including K2 for the converted Kimi-K3 model.
- `TrellisExtent` records global intermediate width and the rank's first/count
  of 32-channel slots. Nonzero Hadamard sign patterns are generated globally
  and sliced, not regenerated from local width. Native callers without global
  extent metadata retain their zero-sign-pattern restriction.
- `TrellisWeights` contains uint8 codes, rate bytes, FP16 scale vectors and
  optional gains, and optional per-expert sign-pattern IDs. There is no
  `Exl3Weights` or `Exl3Source` branch in canonical planning.
- `ActivationSpec.rotation_dtype=torch.float16` makes the FP16 full-rotation
  arithmetic explicit while retaining BF16 public inputs/outputs. This is the
  arithmetic contract used by the EXL3 serving integration, not an additional
  quantization step. Its default is `None`, which retains the I/O dtype for
  existing callers. It is rejected for non-trellis or projection-tiered MCG
  preparation.

The adapter checks zero row padding and removes it through a view. It maps
the two physical FC1 projections to the appropriate hidden-axis scale table
for the interleaved intermediate transform. Codewords and FP16 scale values
are copied or permuted, never decoded and re-encoded.

## Staging and lifetime

`prepare_weights(device=...)` accepts CPU uniform-codeword inputs and moves
them to the specified CUDA device. CUDA inputs must already belong to that
device. CPU weights do not remain in the inference path.

`TrellisStaging` defaults to at most 64 experts and 64 MiB of codeword transfer
data per batch. The smaller limit wins. For a 32 MiB transfer budget, pass
`TrellisStaging(max_bytes=32 * 1024 * 1024)`. A single expert that exceeds the
budget is rejected before final weight allocation; increase the budget or
use a smaller legal source extent. This setting does not limit total model
memory: final weights, scale/sign tables, allocator overhead, and at most one
projection-sized copy operand are separate allocations.

Preparation completes before CUDA Graph capture. Prepared experts own final
GPU storage; bind/run/replay do not read checkpoint files, transfer source
weights, or choose another kernel. Projection-tiered native MCG retains its
CUDA-resident input contract; bounded CPU staging applies to uniform rates.

The low-level EXL3 entry point forwards uniform inputs into common preparation.
Its asymmetric-pair implementation remains separate because its rate and
record contracts differ. The common adapter rejects pair-rate containers
rather than flattening their rates.

## Validation

`tests/moe/test_trellis_adapter.py` checks byte-exact word assembly, FP16 tables,
source-global signs, staging bounds, corrupt inputs, and safetensors reads.
`tests/moe/test_trellis_config.py` checks the native schema and projection layout.
`tests/moe/test_exl3_prepare.py` compares CUDA preparation against independent
word/rotation assembly and exercises execution and graph replay. Run all three
plus `tests/moe/test_exl3_schema.py` in a B12X CUDA environment before deployment.
