# Lossless FP4 scale compression during weight preparation

`fused_moe.prepare_weights` can compress native MXFP4 or ModelOpt NVFP4
expert block scales into CSF at load time. The packed E2M1 weight bytes and
activation precision retain their native meaning. The encoder operates on
scale bytes; it does not round, requantize, or calibrate scale values.

Select compression with `WeightPlanConstraints(scale_compression="csf")`.
The default, `None`, retains ordinary native preparation. Supply two separate
CUDA byte buffers with the logical source-scale shapes. Allocate these once
for a group of layers with identical geometry and serialized execution:

```python
from b12x.moe import fused_moe as moe
import torch

# packed is the ordinary PackedWeights bundle. NVFP4 block scales already
# use native F8_128x4 storage; MXFP4 block scales use logical E8M0 grids.
e, h, n = geometry.num_experts, geometry.hidden_size, geometry.intermediate_size
group_size = 16 if source.format.value == "modelopt_nvfp4" else 32
scale_scratch = (
    torch.empty((e, 2 * n, h // group_size), dtype=torch.uint8,
                device=packed.w13.device),
    torch.empty((e, h, n // group_size), dtype=torch.uint8,
                device=packed.w13.device),
)
plan = moe.plan_weights(
    source=source,
    activation=activation,
    geometry=geometry,
    constraints=moe.WeightPlanConstraints(scale_compression="csf"),
)
experts = moe.prepare_weights(plan=plan, weights=packed,
                              scale_scratch=scale_scratch)
```

Preparation consumes native projection order, including gate/up (`w31`).
It may normalize weights in place, as ordinary native preparation does.
`experts.plan._impl.discards_source_parameters` tells the integration to drop
its source parameter references after the prepared owner has been installed.
Keeping the original block-scale tensors alive defeats the resident-memory
saving. The prepared owner retains the compressed bytes and scale scratch;
execution reconstructs the scales required by its selected decoder.

The buffers must not alias source tensors or each other. Share them only
between serialized layer executions on one CUDA stream. Independent model
execution lanes need independent buffers. Preparation allocates, synchronizes
once to size each compact exception stream, and may compile kernels. Complete
it before graph capture. Replay does not run the encoder or allocate storage.

Supported arithmetic contracts are MXFP4 with A8 or A16 activations and
ModelOpt NVFP4 with A4 or A16 activations, with a gated nonlinearity. The
existing CSF decoder's geometry restrictions still apply. In particular,
MXFP4 A16 preparation supports the Kimi and DS4.1 shard geometries listed by
`prepare_w4a16_x4t_weights`. MXFP4 requires unit global weight factors and,
for A8, unit activation factors. NVFP4 global factors remain unchanged.
Automatic precision switching and an A16 token cutoff are unsupported.

A row stores the base of the interval containing the most source bytes,
with the lowest base winning ties. MXFP4 uses one-bit offsets within a
two-byte interval; NVFP4 uses four-bit offsets within a sixteen-byte interval.
Exceptions retain their original bytes. Highly dispersed scales can produce
a representation larger than the input; enabling compression does not
promise a fixed compression ratio.

See [the validation report](../validation/csf/online-scales.md) for exact
correctness coverage, encoding cost, and serving limits. B12X consumes tensors
and geometry; checkpoint discovery, reading, TP slicing, and parameter
lifetime belong to the integration.
