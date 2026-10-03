# Dynamic MoE route and preparation invariants

Dynamic MoE execution must retain a claimed routing batch until every CTA warp
has read it, and must distinguish token indices from routed-pair indices.
These contracts apply to native weights as well as compressed scale storage.
Preparation must also declare the reduction and scale-decoder programs that a
selected execution plan can call before kernel resolution is frozen.

**Implemented:** the shared work-claim broadcast has a read-completion barrier;
shared W4A8 gathers recover token indices from deterministic route metadata;
preparation retains both scale decoders and the deterministic top-k reduction.
Checkpoint bytes, activation quantization and supported output dtypes do not
change. Correcting an invalid activation address can change previously incorrect
deterministic W4A8 outputs.

## Shared routing broadcast

The producer-loop leader publishes a claimed batch in shared memory. A CTA
barrier orders readers after that write. A second CTA barrier is required after
the read: inactive tail warps can otherwise reach the following iteration and
overwrite the slot while another warp still reads it.

**Qualified:** Compute Sanitizer reports 16 hazards on the native control and
zero hazards after the additional barrier for three A4 expert tests. The test
domain has eight experts, hidden dimension 256, intermediate dimension 128 and
top-2 routing, including partial token batches. Native/CSF output equality alone
did not detect this race. No full-model numerical effect is attributed to it.

Receipts are `nv-control-racecheck-1.log` and
`nv-route-sync-racecheck-1.log` under the evidence root specified below. The
matching source is retained in `nv-route-sync-runtime/b12x-beta` under the
investigation workspace.

## Deterministic W4A8 shared activation gather

Atomic scatter stores token indices in `token_map`. Deterministic scatter
stores `token * top_k + slot`, because its output is reduced separately in a
fixed order. Shared activation payloads and scales remain indexed by token in
both modes. The monolithic M32 materialized path must divide a deterministic
map entry by top-k before gathering either payloads or scales. Global row-stride
products use Int64.

Minimal reproduction conditions are eight experts, hidden dimension 512,
intermediate dimension 256, 128 tokens, top-2 routing, SiLU, MXFP8 activations,
deterministic output and dynamic M32 grouped execution. Filling CUDA
`empty`/`empty_like` allocations with byte `0x7f` exposes the invalid read:
all 64 output rows corresponding to tokens 64 through 127 become nonfinite.
The same failure occurs with native weights and the unchanged control loader.

**Qualified:** converting routed-pair indices to token indices removes the
minimal failure. The retained GPU oracle also covers 16 experts, hidden
dimension 4096, intermediate dimension 1024, top-2 routing, M32/128 tokens and
M64/4096 tokens. It compares against the Torch W4A8 oracle, mutates live inputs,
poisons packed activations, scales and intermediate scratch, and replays CUDA
graphs in both deterministic and atomic modes. Existing numerical thresholds
are unchanged. Together with native/CSF MXFP8 parity cases, 26 tests pass in
`nv-corrected-tests-1.log`.

Reproduce the retained oracle with:

```bash
python -m pytest -q \
  tests/moe/test_w4a8_migration_corpus.py::test_w4a8_materialized_routing_phase1_phase2_matches_oracle_under_graph \
  tests/moe/test_mxfp4_csf.py
```

## Frozen preparation declarations

Short-route indexed scale expansion is compiled during weight preparation, but
the serving preparation manifest must also retain it. Declaring only the slab
decoder rejects the indexed launch as an undeclared program. The scale-program
payload now records whether indexed planes exist and declares both routing
integer ABIs for every available decoder.

Dynamic deterministic output also invokes a separate top-k reduction. That
program is declared with the selected token capacity, top-k count, hidden
dimension and element dtype. Retaining only the dynamic expert kernel is
insufficient.

**Qualified:** the strict preparation test passes for atomic and deterministic
output with the external scale decoder selected. It subsequently checks exact
native/CSF output, mutated inputs, poisoned scratch, frozen CUDA graphs and
zero replay allocations. Receipt: `nv-indexed-declarations-3.log`.

```bash
python -m pytest -q \
  tests/moe/test_nvfp4_csf.py::test_indexed_scale_programs_are_declared_for_preparation
```

## Evidence identity and limits

Evidence root: `/data/trellis-quant/csf-batch-performance-20261001`.
Investigation workspace: `/root/vllm/kimi/csf-batch-performance-20261001`.
The allocator-poison control is `api-runtime/b12x-beta`; its minimal reproducer
is `reproduce_mxfp8_poison.py` in the evidence root. The failing control and fixed
receipts are `nv-mx-poison-control-1.log` and
`nv-mx-poison-corrected-1.log`. Diagnostic tensor dumps preserve the failing
binding's maps, activations, scales and outputs.

Checks use RTX PRO 6000 Blackwell on Frank1 and CUTLASS DSL 4.7.1 from image
`sha256:5501c32fa048b25d53004ea9cd391477467eae9b4e9d9c2dcccc851086dac46b`.
The complete publication-source component run passes 408 tests in
`nv-publication-tests-2.log`; the two strict declaration cases additionally
qualify the preparation-manifest corrections. These are correctness claims.
Timing observations elsewhere remain **research-only**. Full-model logit
parity, KLD and unmeasured geometries are **unsupported** by this report.
