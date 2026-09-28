# IQ2_XS dense GEMM

Status: implemented on SM120 and SM121. [IQ2_XXS](iq2-xxs.md) shares this
engine with compile-time codec specialization and 66-byte packed blocks.

`b12x.gemm.blockscaled` accepts IQ2_XS safetensors block payloads through
`pack_weight(blocks, recipe="iq2_xs")` and the prepared `mm` API. Inputs are
contiguous CUDA uint8 tensors with shape `[N, K/256, 74]`. K must be divisible
by 256 and N by 8. There is no GGUF container loader.

Each block contains a little-endian FP16 base, 32 uint16 descriptors, and
eight bytes of paired four-bit subscales. A descriptor selects eight
magnitudes and a parity-coded sign pattern. Reconstruction is
`signed_magnitude * (FP32(base) * (subscale + 0.5) * 0.25)`, rounded once
to BF16 before MMA. Activations and output are BF16; accumulation is FP32.
Nonfinite block bases are rejected during packing. Activation quantization
and external activation or weight scales are unsupported.

Packing produces `IQ2XSLinearWeight` with byte descriptors `[N,K/4]` and
byte metadata `[ceil(N/128),K/256,1280]`. Each metadata tile contains 128 FP16
bases followed by eight planes of 128 packed scale pairs. Only metadata in
the final N tile is padded. N128-aligned matrices retain exactly 74 bytes
per 256 weights, or 2.3125 bits per weight. Packing allocates owned compact
storage and leaves the input unchanged; no expanded weight matrix is retained.

`DenseGemmKernel` stages descriptors and metadata asynchronously, then
decodes BF16 pairs directly into MMA registers. The prepared state owns a
reference to the shared process-lifetime 8 KiB magnitude table. GEMM, output
handling, and optional split-K reduction use the existing dense A16 path.
This execution requires no expert routing or MoE operation.

## Prepared execution

Packing must finish before capture. Planning uses the existing
`gemm.blockscaled_precision` component; `activation_mode="auto"` selects
A16 for IQ2_XS at every capacity. Preparation races eligible launch
configurations before capture when autotuning is enabled.
`activation_mode="quantized"` fails during planning.

```python
import torch
from b12x.gemm import blockscaled
from b12x.preparation import PreparationSession, PreparedCall

# blocks: CUDA uint8 [N,K/256,74] read from safetensors.
weight = blockscaled.pack_weight(blocks, recipe="iq2_xs")
x = torch.empty((512, weight.in_features), device=blocks.device, dtype=torch.bfloat16)
out = torch.empty((512, weight.out_features), device=blocks.device, dtype=torch.bfloat16)
query = blockscaled.query_from_call(x, weight, out=out)
plan = blockscaled.plan(query)

def prepare_call(state):
    return PreparedCall(run=lambda: state.run(
        x, weight.values, weight.metadata, None, out=out,
    ))

with PreparationSession(device=x.device, autotune=False) as session:
    session.prepare((plan.request(name="shared-expert-up", prepare_call=prepare_call),))
    session.freeze()
    # Fill x[:live_rows] before execution; live_rows may vary up to 512.
    y = blockscaled.mm(x[:live_rows], weight, out=out[:live_rows], plan=plan)
```

Live row counts are launch arguments and do not enter kernel compile keys.
The plan reserves split-K capacity and materializes the lookup table before
binding. `workspace_size(plan)` reports scratch requirements after preparation;
caller-provided workspace is declared through `query_from_call`. Graph replay
uses fixed addresses and performs no device allocation.

## Checkpoint validation

The checkpoint validation harness compares dense IQ2_XS execution with an
independent CPU block decoder and FP32 matmul oracle, including changed-input
CUDA graph replay:

```bash
PYTHONPATH=. python validation/iq2_xs/qualify_dense.py \
  --model-path /path/to/checkpoint \
  --device 0 --evidence /tmp/iq2-xs-dense-qualification.jsonl
```

Select the assigned device and a local safetensors checkpoint. The harness
records source and payload hashes, device identity, toolchain, launch
configurations, numerical metrics, and graph invariants. It does not collect
performance measurements or execute checkpoint Python code.
