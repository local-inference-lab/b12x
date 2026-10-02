# CSF integration for the Karmic Kraken beta channel

## Behavior and compatibility

**Implemented:** the `integration/karmic-kraken-beta` composition combines
vLLM-owned MXFP4-CSF/NVFP4-CSF checkpoint loading, ordinary B12X scale
preparation, native DS4.1 MXFP8 expert activations, cooperative NVFP4 scale
reconstruction, and native-checkpoint online compression. Enable online
compression with `VLLM_B12X_MOE_FP4_CSF=1`; its default remains `0`.
Checkpoint schemas and weight values are unchanged. Shared scale scratch
requires serialized layer execution.

B12X integrates #450, #458, and #459 through #456. Paired vLLM integrates
#956 and #964 through #963. The MoE tuning contract retains the beta native
MXFP8 backend restrictions and Trellis decode-table controls together with
compressed-scale choices. Query schema 17 and configuration schema 10
invalidate incompatible cached configurations.

Only Trellis weights reserve shared memory for a Trellis lookup table.
Ordinary FP4 and block-codec kernels retain their own scale/table layouts.
This preserves resource-valid IQ2_XS and Q8_0 tuning candidates without
changing Trellis lookup selection or reducing candidate coverage.

## Component validation

**Qualified:** the component contracts below, on RTX PRO 6000 Blackwell,
CUDA 13.4, and CUTLASS DSL 4.7.1. The component environment uses the exact
source composition over an installed runtime. Public Docker startup and
serving throughput require separate validation after its CI build.

| Conditions and measurement | Result | Conclusion |
|---|---|---|
| Combined scale, MoE, tuning, native MXFP8, and Trellis configuration suites | 466 passed; the two block-codec resource failures also reproduce on the frozen beta-port parent | CSF correctness passes; the parent resource defect has an independent reproducer |
| Block-codec tuning, Trellis direct lookup, MXFP4/NVFP4-CSF experts, and bounded dense-MLA cases with the table-ownership correction | 139 passed | Resource-valid candidates and affected GPU paths pass with the correction |
| vLLM MXFP4-CSF/NVFP4-CSF loader suites | 61 passed | Inventory, TP slicing, calibration, precision and prepared ownership contracts pass |
| vLLM compressed-tensors staging, online CSF, and native MXFP8 adapters with repository distributed fixtures | 29 passed | The combined adapter supports online ownership and existing native MXFP8 paths |
| vLLM merge-conflict files | Applicable pre-commit hooks, including mypy, passed | Environment declarations and adapter imports satisfy repository checks |
| Release fragments in both component trees | All cross-component dependencies resolve; published fragments are unchanged | The channel publisher can describe the paired changes |

The independent resource reproducer is
`tests/preparation/test_block_moe_tuning.py::test_every_candidate_constructs_resource_valid_kernel`
on B12X parent `8f0f01829168f51cca424ac2f57c51b7a25a7b67`. IQ2_XS reserves
103,296 bytes and Q8_0 101,504 bytes against a 101,376-byte capacity when an
unused Trellis table is included. The corrected test retains every candidate
and verifies that non-Trellis kernels do not allocate that table.

The affected B12X command is:

```bash
python -m pytest -q tests/preparation/test_block_moe_tuning.py \
  tests/moe/test_trellis_direct_lut_decode.py tests/moe/test_mxfp4_csf.py \
  tests/moe/test_nvfp4_csf.py tests/attention/test_dense_mla.py \
  -k 'not page_ids_past_int32'
```

The two excluded large-page attention cases exercise unchanged addressing
code; this validation does not extend their prior qualification. vLLM's
adapter command needs the root test fixtures and their test dependencies:

```bash
python -m pytest -q --confcutdir=tests tests/kernels/moe/test_b12x.py \
  -k 'compressed_tensors_mxfp4_preserves or online_csf or mxfp8'
```

[Evidence JSON](beta-integration-evidence.json) records source/image
identities, suite counts, and raw receipt hashes. The raw files are retained
at `/data/trellis-quant/csf-beta-integration-20261002/premerge` on Frank1.

**Research-only:** the separately pinned performance results linked from
[online preparation](online-preparation-performance.md) and
[Qwen serving](qwen-serving.md). Those results do not qualify a different
Docker composition. **Unsupported by this validation:** full-model KLD,
maximum-context stress, and unmeasured serving configurations.
