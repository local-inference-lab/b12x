# Third-party notices

This repository records experiments that interoperate with third-party models
and software. Those components are not relicensed by the ShapleyMCG License.

- GLM-5.3-Flash BF16: `zai-org/GLM-5.3-Flash-BF16`; model terms apply.
- GLM-5.3-Flash NVFP4 carrier: `local-inference-lab/GLM-5.3-Flash-NVFP4`;
  model terms apply.
- vLLM: Apache-2.0.
- B12X: Apache-2.0; the license accompanying the modified runtime sources is
  retained at `runtime_patch/b12x_h16/LICENSE.b12x`.
- InstantTensor: upstream https://github.com/scitix/InstantTensor, used through
  the voipmonitor/InstantTensor fork (0.1.9); upstream license applies.
- Humming NVFP4 MoE backend: shipped inside the local-inference-lab/vllm fork
  (Apache-2.0, vLLM contributors).
- FlashInfer: Apache-2.0, used through the voipmonitor/flashinfer fork.
- LibertAIDAI/GLM-5.3-Flash-NVFP4: earlier bring-up stock checkpoint referenced
  in the documentation; model terms apply.
- PyTorch: BSD-style license.
- NVIDIA CUDA, ModelOpt, and container components: NVIDIA terms apply.
- ExLlamaV3 / ExL3: Copyright turboderp and contributors, MIT License. The
  procedural trellis constants and bitstream layout used by the research codec
  are attributed to ExLlamaV3; the retained terms are in
  `LICENSE.exllamav3` and must accompany reuse.
- b12x: upstream https://github.com/lukealonso/b12x by Luke Alonso, with
  Martin Vit and other contributors named in `CITATIONS.md`; Apache-2.0.
- KQuant / QSRT reference snapshot: commit
  `104dd9233f850a3955f4991bea68b07dd34deeb8`, attributed to Luke Alonso and
  contributors. The audited snapshot did not contain a `LICENSE` file, so this
  repository does not assert a license for copied KQuant/QSRT material.
- The vendored B12X `w4a8_trellis` and `w4a8_trellis_decode` paths implement a
  QSRT SQG-XOR-Cheb-T12 decoder. Reuse of its pipeline, lane mapping, or T12
  table must be described as a port and retain the KQuant/QSRT attribution and
  unverified-license notice above.
- The separate `runtime_patch/p4/` endpoint ports the ExLlamaV3 cyclic K4
  stream, MCG constants, and tensor-core tile convention through this
  repository's KQuant/QSRT codec. Its producer/consumer organization follows
  the vendored B12X `w4a8_trellis` design. The integer P4 decoder, CUDA C launch
  ABI, and E2M1/E4M3/16 staging are new; no T12 table or P8 K32 operand
  permutation is used. `glm53_nvfp4/p4_reference.py` independently implements
  the same frozen law/layout for CPU verification. Retain all notices above.
- The GLM P4 serving adapter in `runtime_patch/p4_glm_serving.py` and the
  `b12x_h16/.../p4_native.py` backend entry are new integration work against
  vLLM's existing factory/modular-method contracts. The v2 P4 decoder adopts
  the matrix codec's native RNE and signed-zero semantics. Native activation
  prepass reuse and shape/stream workspace caching are new implementation;
  the underlying ExLlamaV3/KQuant/QSRT/B12X port attribution still applies.

- The P8 narrow-FC1 implementation in
  `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py`
  is a port and modification of B12X's materialized phase-1 pipeline, not an
  independently invented pipeline. It preserves this campaign's MCG decoder,
  FP32 clipped SwiGLU and BF16 rounding boundary while changing output-tile
  ownership and staging ranges. The aligned scratch-arena layout and serving
  selector plumbing are new integration work; all underlying notices above
  continue to apply.

The `runtime_patch/b12x_h16` files are modifications of the pinned B12X
runtime sources. Their upstream notices and the ShapleyMCG attribution must be
retained when redistributed.

Academic and method lineage, including the papers whose ideas this work
builds on, is recorded in `CITATIONS.md`.
