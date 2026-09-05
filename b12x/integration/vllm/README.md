# B12X vLLM quantization plugins

Thin vLLM adapters for B12X MX-FP6 (W6A6/W6A8) and Qwen3.8 dense-MLP QSRT K5
checkpoints. The MX-FP6 adapter mirrors
how the maintainer's private fork wires NVFP4: the glue lives in
``b12x/integration/`` and calls the public ``plan`` / ``bind`` / ``run``
APIs — b12x itself does not auto-load into vLLM.

## Files

| File | Role |
|---|---|
| `fp6_serving.py` | Framework-agnostic quant methods (`B12XFP6MoEMethod`, `B12XFP6LinearMethod`) |
| `plugin.py` | vLLM ``QuantizationConfig`` + weight loaders + entry point |
| `qsrt_plugin.py` | TP1 Qwen3.8 K5/rank-16 config, exact weight loaders, bounded workspace, and entry point |

## Install into a vLLM fork

### 1. Install b12x

```bash
pip install -e /path/to/b12x-fp6
```

### 2. Register the plugin entry point

Add to your vLLM fork's ``pyproject.toml`` (or use b12x's optional entry
point if you install b12x with vLLM present):

```toml
[project.entry-points."vllm.general_plugins"]
b12x_fp6 = "b12x.integration.vllm.plugin:register_b12x_fp6"
b12x_qsrt = "b12x.integration.vllm.qsrt_plugin:register_b12x_qsrt"
```

### 3. Launch

```bash
export B12X_ENABLE_FP6=1
export B12X_FP6_MODEL_DIR=/path/to/fp6-checkpoint

# MoE TP>1 on Blackwell: disable vLLM's broken custom all-reduce
vllm serve "$B12X_FP6_MODEL_DIR" \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  ...
```

``--quantization b12x_fp6`` is optional when the checkpoint's
``config.json`` carries ``quant_method=modelopt`` + ``quant_algo=W6A6``.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `B12X_ENABLE_FP6` | off | Master gate |
| `B12X_FP6_MODEL_DIR` | — | Checkpoint path for spawned workers |
| `B12X_QSRT_MODEL_DIR` | — | QSRT checkpoint path used to locate `b12x-qsrt-manifest.json` |
| `B12X_MOE_WARM_MS` | auto | Override MoE decode warm-run token counts |
| `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` | off | Deterministic MoE combine (KLD scoring) |
| `B12X_DISABLE_BF16_GEMV` | off | Disable small-N bf16 GEMV routing |
| `TORCH_COMPILE_DISABLE=1` | — | Required for bit-identical KLD (vLLM Inductor) |

The Qwen3.8 QSRT checkpoint contract and launch are specified in
[qwen38-qsrt-vllm-integration.md](../../../docs/qwen38-qsrt-vllm-integration.md).

See [docs/mxfp6-vllm-integration.md](../../../docs/mxfp6-vllm-integration.md) for
the full lifecycle and maintainer drop-in instructions.
