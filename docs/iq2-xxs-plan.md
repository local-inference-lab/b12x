**Plan: extend the existing packed A16 paths to IQ2_XXS and subsequent codecs.**

Inspected `master` at `555aaa83791c4105267ea54f5b29e4fb1f06e1b8`.
This records the original design. See [IQ2_XXS support](iq2-xxs.md) for the
implemented contract.

The requirements are to minimize duplicated code, retain IQ2_XXS at 66 bytes
per 256 weights, and losslessly retile checkpoint blocks into execution layouts.
The next requested codec is Q8_0. The shared interface should accommodate it without assuming
that every codec uses IQ2 descriptors, a lookup table, or 256-weight blocks.

The current XS implementation already separates packed weights from BF16 MMA,
routing, activation, reduction, and prepared execution. Extend those existing
paths. Keep the dense and MoE layout choices local to their consumers: they
already use different swizzles, and do not need a single universal layout.

The XXS format was checked against GGML revision
`64302f42eb959556633c89ae651915d8157dcd63`:
[block definitions and tables](https://github.com/ggml-org/ggml/blob/64302f42eb959556633c89ae651915d8157dcd63/src/ggml-common.h)
and [reference dequantization](https://github.com/ggml-org/ggml/blob/64302f42eb959556633c89ae651915d8157dcd63/src/ggml-quants.c).

| Format property | IQ2_XS today | IQ2_XXS |
| --- | --- | --- |
| Weights per block | 256 | 256 |
| FP16 base | 2 bytes | 2 bytes |
| Encoded payload | 64 bytes | 64 bytes |
| Separate subscales | 8 bytes | None |
| Total block storage | 74 bytes | 66 bytes |
| Magnitude vectors | 512 | 256, using a distinct table |
| Subscale sharing | 16 weights | 32 weights |

Each XXS 32-weight record is two 32-bit words: four eight-bit grid indices,
then four seven-bit sign codes and one four-bit subscale. Its magnitudes,
parity-coded signs, and scale arithmetic allow reuse of the existing BF16
conversion math. A direct table comparison found 94 XXS vectors absent from
the XS table, so codebook selection must remain explicit.

The proposed implementation sequence is:

1. **Introduce a small static codec description around the existing helpers.**
   Centralize format identity, block weight count, source block bytes,
   payload bytes, metadata layout, alignment, and optional execution-table
   requirements under `b12x/_lib/quant/`. Use a frozen descriptor or equivalent
   constants selected during preparation and kernel construction. Consumer
   layout and fragment helpers use compile-time specialization, not runtime
   dispatch. Keep the initial interface limited to what these paths need.

   A codec supplies packed fragments to the common BF16 compute engine.
   IQ2 codecs can share descriptor/sign/scaling helpers; a future scalar
   eight-bit codec can decode to the same BF16 fragment contract without a
   codebook. Block size, payload density, scale cadence, and LUT allocation
   must therefore be independent properties.

2. **Parameterize the current packers and weight owners.**
   Generalize
   [`pack_iq2_xs_matrix`](../b12x/moe/_shared/kernels/w4a16/iq2_xs.py)
   and the [dense packer](../b12x/gemm/blockscaled/_iq2_xs.py).
   Both codecs can use the existing permutation of bytes `2:66`: its 32-bit
   slots carry descriptor pairs for XS, and alternating grid/sign-scale words
   for XXS. The XXS base plane remains two bytes per block; omit allocation
   and copying of the XS subscale plane. Preserve bounded MoE packing chunks,
   projection ordering, input immutability, and finite-base validation.

   Use a shared packed-block weight owner per API with explicit codec identity;
   preserve the existing XS entry points as compatibility wrappers. Expose
   `recipe="iq2_xxs"` for dense and `PackedSource(format="iq2_xxs")` for MoE.
   Validate that source format, weight owner, and prepared layout agree.

   Assert owned weight storage is exactly `66 * block_count`, excluding the
   separately accounted process-wide LUT and execution workspace. Dense XS
   currently pads metadata to N128; XXS needs compact base storage and masked
   tail loads so N8-aligned shapes do not silently exceed that byte contract.

3. **Specialize fragment loading and reuse IQ2 decoding math.**
   In the [MoE fragment loader](../b12x/moe/_shared/kernels/w4a16/kernel.py),
   read the paired slots belonging to the same K32 record. Select the two
   eight-weight groups for the current K16 fragment. For each group, form
   `descriptor = grid_index | (sign_code << 9)` in registers, with bit 8 zero,
   and extract `subscale = signs_scale_word >> 28`. Return the existing
   descriptor-pair/base/subscale interface. The existing
   [BF16 decode intrinsics](../b12x/_lib/intrinsics.py) can consume it with
   the XXS table and preserve their rounding and sign behavior.

   Apply the same extraction helper in
   [`DenseGemmKernel._load_a16_b_fragment`](../b12x/_lib/dense_gemm.py)
   and the [existing CuTe GEMV path](../b12x/gemm/blockscaled/_iq2_xs_gemv.py).
   Normalization is register-local; resident weights and staged records stay
   compact. Parameterize stage bytes, metadata transfers, and table sizes.
   The XXS magnitude and selector tables require 4 KiB and 2 KiB respectively.
   Preserve the barrier after the final compact-record reads.

   For MoE, the payload stage formula remains `tile_k * tile_n / 4` bytes.
   Bases remain `ceil(tile_k / 256) * tile_n * 2` bytes, and XXS removes the
   `tile_k / 32 * tile_n` XS subscale region. Derive resource accounting from
   the same codec properties used by allocation and launch construction.

4. **Wire specialization through the existing preparation contracts.**
   Generalize shared IQ handling in dense `_linear.py`, `_a16.py`,
   `_preparation.py`, and `_tuning.py`; and MoE `source.py`, `weights.py`,
   `planning.py`, `_shared/execution.py`, `_impl.py`, `_preparation.py`,
   `_tuning.py`, and the W4A16 host/kernel wrappers. Distinguish shared
   execution capabilities from format-specific payload interpretation.
   Keep routing, MMA, activations, split-K, and ordered route summation shared.

   Preserve the current supported scope: SM120/SM121, BF16 activations,
   dense plus fused SiLU/ReLU² MoE, existing direct-route restrictions,
   and aligned TP cuts. Prime every reachable specialization before capture.
   Include static codec/layout identity in relevant compile and LUT cache
   keys; live counts remain runtime arguments. Version affected query
   semantics, candidate eligibility, and changed compiled ABIs. Change config
   schema versions only when their serialized contract changes.

   At this revision, the actual inventory is
   [`b12x/preparation/catalog.py`](../b12x/preparation/catalog.py), with
   `TuningContract` and `PreparationSession` documented in
   [`docs/gpu-profiles.md`](gpu-profiles.md). AGENTS.md references to the older
   `b12x/policy` inventory and offline generator do not match this checkout.
   Reconcile that guidance when implementing; extend the existing registered
   dense/MoE preparation contracts rather than creating another policy system.

5. **Parameterize checkpoint tooling and regression tests.**
   Extend the existing checkpoint readers and canonical dense/MoE benchmarks
   with codec metadata instead of copying them. Validate XXS `quant_algo`,
   256-weight groups, 66-byte payloads, GGML packing, and tensor shapes.
   Keep the independent XXS raw-block oracle separate from production
   packing and extraction helpers. Give it pinned table provenance.

   Parameterize the existing packing, fragment, dense, MoE, staging, routing,
   checkpoint, and preparation tests over XS/XXS. Add XXS-specific coverage
   for all grid/sign/subscale combinations, both K16 halves of a K32 record,
   mixed adjacent records, signed zero and extreme finite bases, malformed
   payloads, exact byte accounting, and non-N128 dense tails. Verify exact
   BF16 decoding before GEMM/MoE error tolerances. Keep offsets scaled from
   global row/block/expert indices in Int64.

6. **Qualify the shared production paths before tuning.**
   Reuse the existing graph qualification harnesses for changed inputs/routes,
   poisoned outputs/scratch, mapped experts, fixed addresses, no replay
   allocation, and multiple live counts under frozen kernel resolution.
   Run XS regressions alongside XXS so parameterization preserves the current
   path. Check preparation memory formulas against actual allocations.

   Benchmark real XXS checkpoint plans across decode, boundary, and prefill
   sizes on the selected hardware, recording source/weight identities,
   commands, GPU UUID/mode, correctness, raw samples, and ratio direction.
   Inspect registers, shared memory, and repeated record loads before adding
   more specializations. The smaller payload alone is not a latency result.

A host-side bit-extraction check over 10,000 generated XXS records verified
that the proposed register values fit the existing descriptor/sign ABI.
Packing round trips, GPU correctness, generated resources, and performance
remain implementation acceptance work.
