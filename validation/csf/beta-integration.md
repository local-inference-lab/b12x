# CSF integration for the Karmic Kraken beta channel

## Behavior and compatibility

**Implemented:** the `integration/karmic-kraken-beta` composition combines
vLLM-owned MXFP4-CSF/NVFP4-CSF checkpoint loading, ordinary B12X scale
preparation, native DS4.1 MXFP8 expert activations and cooperative NVFP4 scale
reconstruction. Checkpoint schemas and weight values are unchanged. Shared
scale scratch requires serialized layer execution.

B12X integrates #450 and #459 through #456. Paired vLLM integrates #956
through #963. The MoE tuning contract retains the beta native MXFP8 backend
restrictions and Trellis decode-table controls together with compressed-scale
choices.

Only Trellis weights reserve shared memory for a Trellis lookup table.
Ordinary FP4 and block-codec kernels retain their own scale/table layouts.
This preserves resource-valid IQ2_XS and Q8_0 tuning candidates without
changing Trellis lookup selection or reducing candidate coverage.

## Component validation

**Qualified:** the component suites below on two RTX PRO 6000 Blackwell Max-Q
GPUs (`GPU-f93cb3ac-bea7-0586-9f2c-c0307696716e`,
`GPU-5a4ea3b7-bb73-9876-6f45-65769efc3066`), CUDA 13.4 and CUTLASS DSL 4.7.1.
The B12X source tree runs over the installed beta runtime through
`PYTHONPATH`. Public Docker startup and serving throughput require separate
validation after the channel build.

| Conditions and measurement | Result | Conclusion |
|---|---|---|
| CSF scale, MoE, routing, X4T and indexed-reconstruction suites | 228 passed | Checkpoint CSF decoding and operand reconstruction pass |
| W4A16 packed format, tile selection, end-to-end, NVFP4 loader, route-pack and reference suites | 330 passed, 68 failed, 17 skipped | The same 68 cases fail on the prior beta composition in the same image; no new failure |

The CSF command is:

```bash
python -m pytest -q -p no:cacheprovider tests/moe/test_nvfp4_csf.py \
  tests/moe/test_nvfp4_csf_dispatch.py tests/moe/test_mxfp4_csf.py \
  tests/quantization/test_nvfp4_csf.py tests/quantization/test_nvfp4_csf_inline.py \
  tests/quantization/test_csf_routing.py tests/quantization/test_mxfp4_csf.py \
  tests/quantization/test_x4t_scales.py tests/quantization/test_x4t_packed_scales.py
```

The W4A16 command deselects
`tests/moe/test_w4a16_e2e.py::test_w4a16_small_m_direct_barrier_modes_eager_and_graph`:

```bash
python -m pytest -q -p no:cacheprovider tests/moe/test_w4a16_packed_format.py \
  tests/moe/test_w4a16_tile_selection.py tests/moe/test_w4a16_e2e.py \
  tests/moe/test_w4a16_nvfp4_loader.py tests/moe/test_w4a16_route_pack.py \
  tests/moe/test_w4a16_reference.py --deselect \
  tests/moe/test_w4a16_e2e.py::test_w4a16_small_m_direct_barrier_modes_eager_and_graph
```

Raw logs (SHA-256 `dae0ffc3e13324fe0261e372642e823aac36bd534a096eff8980669a93bad8b4`
and `a829073424aa49159fec60ca6583a6b25d55b2f92e8f73e46fb8b6b019022382`) are
retained at `/root/glmqad-spark/results/rw` on the 192.168.66.4 test host.
**Unsupported by this validation:** full-model KLD, maximum-context stress and
unmeasured serving configurations.
