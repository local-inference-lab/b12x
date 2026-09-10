# TrellisMX r27 measurements and overlay reconciliation

These are external measured-runtime regression references for the owner-review
PRs. They are not GPU measurements of these PR heads or proof of a source-build
port on a newer Jovian/B12X base. All measurements use already-opened conditional-
fit development data, not protected final qualification.

## KLD identities

| Runtime | KV dtype | Attention | Mean true-decode KLD | Window BCa 95% |
| --- | --- | --- | ---: | --- |
| Historical TP4/DCP1, image H | nvfp4_ds_mla | B12X_MLA_SPARSE | 0.03418114591027796 | [0.0291483518, 0.0409784257] |
| Historical TP4/DCP1, image H | fp8_ds_mla | FLASHINFER_MLA_SPARSE_SM120 | 0.03180776125099182 | [0.0267419020, 0.0387478446] |
| Current TP4/DCP4, image C | nvfp4_ds_mla | B12X | 0.03500781830024326 | [0.0292403806, 0.0430987171] |
| Current TP4/DCP4, image C | fp8 | B12X | 0.031057476694042765 | [0.0263411936, 0.0374509346] |

H is local Docker image ID
`sha256:c121590d3371b406c1e456076b1fc9cf9b596aa425b4401c692d142cbc425cd4`.
C is capture-only local Docker image ID
`sha256:a7fde8169ec24fff3d7f1a7a8a50375a90e6c9e8cf71254680a281f54be37ca6`,
derived from published serving image registry manifest digest
`verdictai/trellismx@sha256:1c8a10d2b21bd6ed5a7ca4a29bcc3900d29acc3ce42e1d722b9ebaa74357de3f`.
The historical image is not the RC5 release image. The capture wrapper is not
part of the public serving recipe.

All four rows use the same 32 windows in the same order and the same reference
token sequences, with 2,048 input tokens and 2,047 prediction rows per window.
Row zero is one-token prefill; the score averages the remaining 2,046 true-decode
rows, then averages windows equally. MTP is disabled for KLD. Teacher-model
weights are BF16, stored teacher logits F32, scoring CPU FP64 KL(teacher||student).
The historical row-count correction changes wording only, not numerical scores.

The current pair changes KV representation/backend specialization within one
fixed r27 image and DCP4 configuration. FP8 minus NVFP4 is -0.003950341606200488,
paired-window BCa95 [-0.00925104571170184, -0.0018107907382052357], lower in 26/32
windows. One server start per arm, fixed NVFP4-then-FP8 order; window intervals
do not measure run-to-run variability. Current prefix caching stays enabled
for hybrid-cache alignment with zero hits at every measured window boundary.

Historical-to-current FP8 also changes FlashInfer to B12X, image, topology and
runtime configuration. It does not establish a DCP4-only accuracy benefit.
The retained historical FP8 score is 0.0318077613; the historical NVFP4 score
is 0.0341811459.
Both historical KV modes were measured. The old FP8 interval uses NumPy
`default_rng(20260902)`; the current pair uses legacy `RandomState(20260902)`.
Both use 20,000 BCa resamples. Retained historical score arrays reproduce the
full float64 mean exactly; both CI endpoints replay within 1e-15.

Current receipts: `comparison.json` and `audit.json`. Historical FP8 retained-score
and input-hash audit: `historical-fp8-audit.json`. Raw logits, teacher tensors,
input token arrays and score NPZs are not included. This audit was performed by
the same operator, not independent empirical reproduction. Two current startup
attempts failed before any scored windows; both failures are disclosed in the
current comparison. No scored windows were excluded or silently replaced.

## Overlay ownership and PR coverage

Measured deployed integration: vLLM `c88fb8847dc8bc18bca56640759f137198750085`,
B12X `7093ad77849cf181bcf0c30b897c54fd32dac40e`. Full published source and hash
inventory are pinned at [HF recipe commit53d93df](https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8/tree/53d93dfbc9002df7b73178327dfe99773efb680e/runtime).

- B12X PR head before this review: `fc9bd550d4dd008ab53aa7810578cdf16a55c39e`,
  base `6483963275dcf32eb2eec6d100e644d1ea647ed6`. P8-specific modules,
  shared intrinsics and phase1/phase2 additions match deployed source.
  The remaining `dynamic.py` difference is inherited from r27 B12X base
  `e8ad299b174f16e2e8fb5879bea272f4efbb53f2`, including upstream split-phase/
  low-smem changes. It is documented as inherited, not copied into the P8 diff.
- vLLM owner PR base remains `9a6b4fb3a6f5598fd2fb68cf0de92bfe145294c1`.
  The inherited ModelOpt carrier-loader capability and both RoutedExperts gates,
  plus regression tests, are ported from deployed commit `c88fb8847`.
- Persistent NCCL DCP query-gather and AG/RS workspaces and their CPU/GPU tests
  are ported separately from `19c1e047e650c1b34b307d1aeb2ec6372ed63cd4`,
  originally sourced from `local-inference-lab/vllm@000a28d19c64898a26628a2f09a0f05d3f257b4f`.
  The port preserves this PR base's `direct_cp_enabled` eligibility policy;
  measured r27 uses `direct_cp_peer_access_enabled`. No wholesale cp_common
  replacement or silent policy backport. This adaptation is not GPU-qualified
  by the r27 measurements and remains an explicit before-promotion gate.
- Frozen r27 Compose/serve scripts are separately named under the vLLM example;
  the existing DCP1 source-build recipe remains separate. The frozen recipe
  runs the published image, not a build of the changed PR head.
- The logits-capture seam is evaluation instrumentation, not a serving patch.
  It remains separately identified by capture image ID and plan seal. No
  capture hooks, encoder, model weights or teacher tensors are bundled here.

No speculative C16 tuning was added. Long-response uninstrumented C16 zero-
context measured651.02 aggregate tokens/s versus252.60 atC2. Short-output
turnover introduced repeated prefill. Four balanced traces attribute residual
cost to grouped TrellisMX FC1/FC2 and collectives; all tensor fills total about
1.99ms per C16 step. Timings overlap and collective time includes waiting;
no compact-buffer or scheduler change is selected by this evidence.
[Full bounded scaling report](https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8/blob/2beda904e659f7bbbc91b8ca3e430d169b687fac/results/r27-scaling-20260908/REPORT.md).

## Validation and review boundary

Focused CPU loader/method, DCP and B12X contracts are run with no GPU devices.
The receipt checker verifies public artifact hashes, matched windows, score
aggregation and historical FP8 runtime labels and current capture-image identity; it does not execute the model or turn
external KLD into a passing GPU test of this PR head.

Codex and the local GLM-5.3-Flash TrellisMX reviewer cross-reviewed the proposed
scope, source diff and evidence. Their agreement is automated review, not human
approval or independent experimental replication. Both drafts remain on the
owner forks. No upstream submission, merge, production restart or GPU job is
part of this update. Existing attribution and licenses are retained unchanged.
