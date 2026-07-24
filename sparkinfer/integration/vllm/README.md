# sparkinfer FP6 vLLM integration

Thin vLLM adapter for sparkinfer MX-FP6 (W6A6/W6A8) checkpoints.  This mirrors
how the maintainer's private fork wires NVFP4: the glue lives in
``sparkinfer/integration/`` and calls the public ``plan`` / ``bind`` / ``run``
APIs — sparkinfer itself does not auto-load into vLLM.

## Files

| File | Role |
|---|---|
| `fp6_serving.py` | Framework-agnostic quant methods (`SparkInferFP6MoEMethod`, `SparkInferFP6LinearMethod`) |
| `plugin.py` | vLLM ``QuantizationConfig`` + weight loaders + entry point |

## Install into a vLLM fork

### 1. Install sparkinfer

```bash
pip install -e /path/to/sparkinfer-fp6
```

### 2. Register the plugin entry point

Add to your vLLM fork's ``pyproject.toml`` (or use sparkinfer's optional entry
point if you install sparkinfer with vLLM present):

```toml
[project.entry-points."vllm.general_plugins"]
sparkinfer_fp6 = "sparkinfer.integration.vllm.plugin:register_sparkinfer_fp6"
```

The legacy alias ``register_b12x_fp6`` is also exported for forks still wired to
the old B12X entry-point name.

### 3. Launch

```bash
export SPARKINFER_ENABLE_FP6=1
export SPARKINFER_FP6_MODEL_DIR=/path/to/fp6-checkpoint

# MoE TP>1 on Blackwell: disable vLLM's broken custom all-reduce
vllm serve "$SPARKINFER_FP6_MODEL_DIR" \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  ...
```

``--quantization sparkinfer_fp6`` is optional when the checkpoint's
``config.json`` carries ``quant_method=modelopt`` + ``quant_algo=W6A6``.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SPARKINFER_ENABLE_FP6` | off | Master gate (legacy `B12X_ENABLE_FP6` accepted) |
| `SPARKINFER_FP6_MODEL_DIR` | — | Checkpoint path for spawned workers |
| `SPARKINFER_MOE_WARM_MS` | auto | Override MoE decode warm-run token counts |
| `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT` | off | Deterministic MoE combine (KLD scoring) |
| `SPARKINFER_DISABLE_BF16_GEMV` | off | Disable small-N bf16 GEMV routing |
| `TORCH_COMPILE_DISABLE=1` | — | Required for bit-identical KLD (vLLM Inductor) |

See [docs/mxfp6-vllm-integration.md](../../docs/mxfp6-vllm-integration.md) for
the full lifecycle and maintainer drop-in instructions.
