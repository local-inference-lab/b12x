# Qwen3.8 dense-MLP QSRT K5 serving

Status: **research-only**. The loader and packed operators are implemented.
SM120 numerical, CUDA-graph, full-model quality, and latency qualification are
required before deployment.

## Checkpoint contract

The `b12x_qsrt` quantization method accepts a TP1 Qwen3.8-27B checkpoint whose
`config.json` declares all 192 decoder dense-MLP source modules. Every official
BF16 `gate_proj.weight`, `up_proj.weight`, and `down_proj.weight` is absent from
the tensor index and is replaced by these tensors under the same source-module
prefix:

| Suffix | Dtype | Semantic role |
| --- | --- | --- |
| `qsrt_trellis` | I16 | Native `[K/16,N/16,80]` uniform-K5 trellis payload |
| `qsrt_suh` | FP16 | Input outer-rotation vector |
| `qsrt_svh` | FP16 | Output outer-rotation vector |
| `qsrt_a` | BF16 | `[K,16]` additive factor |
| `qsrt_b` | BF16 | `[N,16]` additive factor |

The reconstruction profile is `sqg_fp16_d3l`, exposed to B12X as the
`sqg_fp16` codebook. The plugin rejects another bitrate, profile descriptor,
rank, tensor-parallel size, module inventory, shared-workspace row capacity,
tensor shape, or dtype. The
checkpoint manifest must bind the K5 artifact, rank panel, completed recovery
report, and selected factor overlay. Manifest, `config.json`, and tensor-index
hashes are checked before a claimed linear is constructed.

All attention, gated-DeltaNet, normalization, embedding, vision, MTP, and
`lm_head` tensors remain BF16 and use vLLM's unquantized methods. A tensor is
never decoded to a dense matrix inside the packed method, and an unsupported
packed input raises instead of selecting a Torch or decoded-weight fallback.

## Execution contract

`gate_proj` and `up_proj` retain independent K5 payloads and outer rotations.
Their two rank-16 A projections share one native BF16 tensor-core launch, and
their two B projections share one native output launch. The packed base
operators execute in source order and copy their results into one stable
`gate_up_proj` output before the paired correction is added. `down_proj` uses
one A launch and one B launch. A complete 64-layer decoder therefore dispatches
256 factor kernels instead of the 384 kernels produced by independent
per-projection adapters.

Every CUDA-graph-visible output and scratch tensor has a stable owner. The
checkpoint's `workspace_capacity_rows` value bounds one buffer pool allocated
during weight processing. Exact-row views of that fixed pool serve eager and
CUDA-graph execution without device allocation or cache growth. Qwen3.8 decoder layers
execute in source order: the following activation consumes a gate/up output,
and the following decoder layer's input normalization consumes a down output,
before the same storage address is written again. CUDA stream ordering retains
that lifetime rule during graph capture and replay. Capture with an unprepared
factor geometry fails closed, and a request above the declared row capacity is
rejected. A device outside SM120/SM121, TP
greater than one, activation dtype other than BF16, bias, non-contiguous input,
and capture sizes above the declared workspace capacity are unsupported.

## vLLM registration and launch

Installing B12X registers the entry point:

```toml
[project.entry-points."vllm.general_plugins"]
b12x_qsrt = "b12x.integration.vllm.qsrt_plugin:register_b12x_qsrt"
```

Launch the materialized checkpoint by local path so every spawned worker can
validate its manifest:

```bash
export B12X_QSRT_MODEL_DIR=/models/Qwen3.8-27B-QSRT-K5-R16
vllm serve "$B12X_QSRT_MODEL_DIR" \
  --tensor-parallel-size 1 \
  --dtype bfloat16
```

`benchmarks/benchmark_qwen38_qsrt_k5_dense.py` validates a real sealed layer.
It checks packed output and input-vector Jacobian parity against independently
decoded controls, immutable payloads and factors, bit-exact eager/graph replay,
pointer and allocation stability, and interleaved hot/cold latency. The paired
gate/up case compares against one fused BF16 base GEMM plus a rank-32
block-diagonal control correction. Pass `--recovery-root` with
`--quality-selection-report` when packaging selected an optimizer boundary from
an external KLD panel. The benchmark validates the recovery report, the complete
candidate population, the KLD ranking, and the selected overlay content before
loading any factor.
