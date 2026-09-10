# TrellisMX P8 native execution — owner review draft

This is an additive port against B12X
`6483963275dcf32eb2eec6d100e644d1ea647ed6`, for review on Brandon's fork only.
It is not an upstream submission or a GPU-qualified release.

## Scope and ownership

Adds procedural MCG K3/K4/K5 reconstruction to native E4M3 with separately
stored UE8M0/32 scales, plus the coupled H512/H128 input/output-scale kernels
used by the GLM-5.3-Flash 17-K5/25-K4 checkpoint. The internal runtime entry
is `b12x.moe._shared.trellismx.p8_native_kernel.P8NativeTPMoE`.
The companion vLLM fork owns ModelOpt integration and checkpoint inventory;
B12X owns these kernels, geometry, compilation and scratch handling.

Only P8 additions and shared-kernel deltas are included. No duplicate B12X
tree, W4A16 implementation, compiler, attention, collective, encoder, model
weights, calibration data or teacher logits are bundled. Existing B12X
dependencies and launch primitives are reused in their original namespace.
This initial integration supports TP4; dormant TP2 research paths are not
an advertised serving interface.

## Source reconciliation

The historical P8 source came from RC5 image
`verdictai/trellismx@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129`,
whose B12X ancestor was `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8`.
Shared intrinsic, dynamic, phase-1 and phase-2 changes were three-way composed
against current B12X. These changes are disjoint from the newer base's intent;
current code was retained rather than replaced with an old runtime snapshot.
The P8-specific modules were ported back from the isolated review namespace
to `b12x`. No arithmetic or schedule constants were intentionally retuned.
The constructor additionally rejects SQG K5, which the existing SQG decoder
does not support; K5 is admitted only for MCG.

Excluded: historical W4A16 route/tile changes, BTX adoption and unrelated
dynamic-launch heuristics. The current fused-MoE launcher is reused unchanged.
The complete historical runtime remains in vLLM draft commit
`8433d53c19bce3462a2a68f95a315e1a3e3e55bb` for comparison, not as a dependency.

Related open work: upstream B12X PRs 275 (coupled QSRT W4A8), 243/245
(MCG/trellis decoding), and 293 (rank-sliced trellis). This draft is specifically
the P8 E4M3/UE8M0 coupled GLM path and credits their shared lineage; it is not
a claim to originate general trellis decoding or coupled transforms.
Current model guidance was checked at rtx6kpro
`94b71ac2a5f9c75f6b60dd1b6e6dffda492a4942`, `models/glm-5.3-flash.md`.

## Validation and limits

CPU-only tests: `tests/moe/test_trellismx_contract.py` checks rate/law rejection,
existing SQG constructor coverage, workspace alignment/non-overlap, and task
ownership. Runtime imports use current B12X, not an isolated dependency copy.
These tests do not execute the decoder or MMA and do not prove bit-exactness.

Before promotion: native compilation on SM120; K4/K5 device decoder and
activation-carrier closure; graph replay and five-run determinism; policy
registration/public plan-bind-run integration review; current-base attention
and collective compatibility; all-layer output/KLD and speed measurements.
The internal adapter is not yet a general public B12X planned-op API.
No GPU jobs or production services were changed to prepare this draft.

Historical checkpoint CF32 KLD is 0.0341811459, window BCa95
[0.0291483518, 0.0409784257], at 4.6587417643 routed bpw with B12X attention,
NVFP4 MLA KV and native P8 E4M3 activations. It is opened development evidence,
not a measurement of this port. P8 uses twice NVFP4's MMA issue count at equal
dimensions; it is not native FP4-rate arithmetic.
[Historical results and receipts](https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/blob/a0c3407e79228ac06a1abf6a79484f38f38bd90a/results/RC5_RELEASE_RESULTS_20260907.md).

## Licensing and review

Credit Brandon M. Music, Luke Alonso and B12X contributors, ExLlamaV3,
KQuant, QSRT and w4a8_trellis. P8 kernel pipeline/lane reuse is a port, not
independent invention. Existing upstream code retains its license. The P8
additions retain SHAPLEYMCG and original third-party terms under
`licenses/trellismx/`; they are not implicitly relicensed Apache-2.0.
The retained notices include the unresolved KQuant/QSRT snapshot-license
boundary. Upstream license acceptance is unresolved and blocks promotion.
OpenAI assistance was used; human review is requested, not asserted.

## September 8 runtime evidence and overlay review

See the [four-row KLD matrix and overlay reconciliation](trellismx/evidence-r27-20260908/README.md).
The historical FP8/DCP1 score is 0.0318077613; historical NVFP4/DCP1 is 0.0341811459.
Current r27 DCP4 scores are 0.0350078183 (NVFP4) and 0.0310574767 (FP8).
These are external measured-image references, not GPU qualification of this PR head.

Focused CPU checks : 31 loader/method tests passed; 38 DCP tests
passed with 21 GPU tests skipped; 14 B12X contract tests passed.
