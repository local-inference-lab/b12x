# CSF loading and weight-preparation qualification

The [MXFP8 expert and serving report](mxfp8-serving.md) covers the DS4.1
activation-policy repair and the composed loading/preparation serving tests.
The [NVFP4 serving report](nvfp4-serving.md) covers GLM/Qwen batch decode and
the [prepared scale operands](nvfp4-scale-operands.md) used by dynamic MoE.
Independent [native routing and preparation invariants](dynamic-moe-correctness.md)
have their own reproducers and correctness limits.
The loading comparisons below retain their original source and precision scope.

vLLM reads compressed FP4 checkpoints, validates their manifests and model
inventories, and slices expert tensors for the selected TP rank. It supplies
packed weights and canonical CPU scale planes to the ordinary B12X
`plan_weights` / `prepare_weights` interface. B12X uploads the scale planes,
partitions exceptions and prepares the weight and scale layouts consumed by its
kernels. File names, model tensor names and shard lifetimes belong to vLLM.

**Implemented:** `CsfScalePlanes` supplies one uint8 fixed tensor and one uint32
exception tensor per expert. `Mxfp4CsfWeights` and `Nvfp4CsfWeights` accept those
planes or already resident batches. Batch construction runs inside
`prepare_weights`; the existing W4A16 and ModelOpt preparation paths retain
normalization, projection order, shared scratch and decoder preparation.
The B12X CSF checkpoint modules and their separate loading API are removed.

This boundary requires installing the paired B12X/vLLM changes. Canonical CSF
checkpoint schemas, tensor bytes, launch flags, compute kernels and supported
execution geometries are unchanged. CPU loading of an arbitrary aligned tensor
geometry does not qualify that geometry for expert computation.

## Conditions and measurement

The control readers are from B12X
`1b32d927bab8d354718bfb559fa925f23e6266d2`; the vLLM integration starts from
`096c7037ee3f526434ee4816fc555b5cd3767715`. Both arms read the same immutable
checkpoint, complete expert layer and TP rank, then use the same weight plan
and `prepare_weights`. The control reader supplies resident compressed scales;
the vLLM reader supplies CPU scale planes.

The comparison covers prepared FP4 weight bytes, compressed scale streams,
exception tables, FP32 global/input calibration, activation and packing metadata.
All tensor comparisons use raw bytes. Expanded scale scratch is not initialized
by MXFP4 or NVFP4 A4 loading, so its contents are excluded; both prepared arms
must retain the exact caller-owned storage. Compiled program objects are excluded
from tensor comparison. Component tests separately exercise execution and replay.

The environment is Frank1 GPU 10, RTX PRO 6000 Blackwell 96 GB,
UUID `GPU-6171baff-cc22-608e-4029-507f67c392ff`, 600 W. Tests use CUTLASS DSL
4.7.1 and native extensions from image
`sha256:18b00b38c792463e8a9ba886d68b9f58256544776856d67c5146d9b2c872cf3f`,
with the review worktrees on `PYTHONPATH`. Clocks are dynamic; these are
correctness checks with no throughput or latency claim.

## Results and limits

**Qualified:** all ten full-layer/rank comparisons are byte-identical:
132 tensors totaling 13,011,266,748 compared bytes.

| Published checkpoint | HF main revision | Layer | TP | Ranks | Preparation |
| --- | --- | ---: | ---: | --- | --- |
| DeepSeek-V4.1-Flash-MXFP4-CSF | `872da235166458bd6ffa9ee3f3c5c4771b63159c` | 0 | 4 | 0, 3 | A16, SiLU, clamp 10 |
| GLM-5.3-Flash-NVFP4-CSF | `20f4777422f833c48b67bb0e554e1bc61dc55ca1` | 3 | 2 | 0, 1 | A4, SiLU, clamp 10 |
| Qwen3.8-Flash-Next-NVFP4-CSF | `4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e` | 0 | 2 | 0, 1 | A4, SiLU |
| Kimi-K3-MXFP4-CSF | `8b7d43b0f7141c4ff04a5f87d26c8675c30ff7a0` | 1 | 16 | 0, 15 | A16, SiTU |
| Kimi-K3-MXFP4-CSF | `8b7d43b0f7141c4ff04a5f87d26c8675c30ff7a0` | 1 | 12 | 0, 11 | A16, SiTU |

Checkpoint IDs use the `local-inference-lab` HF organization. The Kimi check
uses its complete layer-1 source shard, authenticated against the published
manifest. Its identity is recorded in the qualification receipt.

**Qualified components:** 69 B12X tests pass. These cover GPU scale decoding,
native NVFP4 A4/A16 output, CPU-plane and resident-batch preparation, MXFP4
native and MMA-packed preparation, both FC1 projection orders where supported,
shared scratch, poisoned CUDA graph replay and zero replay allocations.
All 56 vLLM tests pass, covering TP slicing and exception rebasing, FP32
calibration under BF16 defaults, projection order, rejected inventories,
manifest validation, retained tensors, shard reuse and failure cleanup.

Applicable vLLM staged-file hooks and manual `mypy-3.12` pass. Changed B12X
code passes Ruff; `fused_moe/api.py` retains three B008 diagnostics for
`FrozenMapping()` defaults, reproduced on the unchanged control. These
immutable defaults are unrelated to CSF preparation and were not changed.
The diagnostics are recorded alongside the test logs.

**Unsupported by this qualification:** full-model generation, full-vocabulary
KLD, loader latency and serving throughput for this refactored package pair.
No serving container was changed. The layer comparisons establish identical
persistent state after preparation; they do not establish a speedup or extend
the serving matrix. Existing whole-model timing evidence applies only to the
exact source/image identities in its reports and remains **research-only**.

[Qualification identities](preparation-evidence/qualification.json) record
runtime source hashes, commands, environment, comparison scope and per-case
receipts. The [tensor-source receipts](tensor-source-evidence/README.md) describe
a historical API and do not qualify the CPU-plane preparation interface.

## Reproduction

Use a matching CUDA/B12X/vLLM environment. Run the component suites:

```bash
# B12X repository
.venv/bin/python -m pytest -q \
  tests/quantization/test_nvfp4_csf.py \
  tests/moe/test_nvfp4_csf.py \
  tests/quantization/test_x4t_packed_scales.py

# vLLM repository
.venv/bin/python -m pytest -q \
  tests/quantization/test_mxfp4_csf.py \
  tests/quantization/test_nvfp4_csf.py
```

To compare Qwen TP2 prepared state, extract the control reader from the B12X
revision above, and set `--checkpoint` to the immutable snapshot directory:

```bash
git show 1b32d927bab8d354718bfb559fa925f23e6266d2:b12x/moe/checkpoints/nvfp4_csf.py > /tmp/nvfp4-csf-reference.py
.venv/bin/python validation/csf/compare_tensor_loaders.py \
  --reference-reader /tmp/nvfp4-csf-reference.py \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4-CSF \
  --checkpoint-id local-inference-lab/Qwen3.8-Flash-Next-NVFP4-CSF@4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e \
  --codec nvfp4 --layer 0 --num-experts 512 \
  --hidden-size 2560 --intermediate-size 640 --tp 2 --ranks 0 1 \
  --output /tmp/qwen-csf-prepared-parity.json
```

The output path must not exist. The comparison reads checkpoint files without
modifying them and performs no full-model inference.
