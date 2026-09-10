"""Serialized block-FP8 (DeepSeek-style) linear via native MXFP8 GEMM.

Planned lifecycle: ``pack_weight`` packs E4M3 weights with 128x128 DSV4 or
32x32 DSV4.1 block scales into the dense-GEMM MXFP8 layout (one-time).
UE8M0 scales preserve the weight bytes exactly; only legacy arbitrary FP32
128x128 scales require requantization. ``plan(Caps)`` sizes caller-owned scratch;
``bind`` maps allocation-free views; ``run`` quantizes BF16/FP16 input to
E4M3/UE8M0 per K32 and launches native CuTe GEMM.

Bindings default ``expected_m`` to ``Caps.max_tokens``. An explicit row bound
takes precedence, including when separate captured regimes share scratch.
Pass ``block_size=(32, 32)`` to both ``pack_weight`` and ``Caps`` for V4.1.
``quantize_input`` accepts the same recipe to select its 1e-4 amax floor.
``prewarm`` derives the recipe from the packed weight and compiles both the
functional and caller-owned paths with the serving ``expected_m`` bound.

Example:
    from b12x.gemm import block_fp8_linear as bfl

    weight  = bfl.pack_weight(w_fp8, w_scale)                      # one-time
    plan    = bfl.plan(bfl.Caps(device="cuda", max_tokens=M,
                                in_features=K, out_features=N))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = bfl.bind(plan, scratch=scratch, source=x,
                       packed_weight=weight, output=out, expected_m=M)
    y       = bfl.run(binding=binding)     # one-shot form: bfl.run(x, weight)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="block_fp8_linear",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Weight",
        "BlockFp8LinearConfig",
        "BlockFp8LinearQuery",
        "plan",
        "bind",
        "run",
        "pack_weight",
        "quantize_input",
        "prewarm",
        "is_supported",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/block_fp8_linear.py",),
    ),
    test_path="tests/gemm/test_block_fp8_linear.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        BlockFp8LinearConfig,
        BlockFp8LinearQuery,
        Caps,
        Plan,
        Weight,
        bind,
        is_supported,
        pack_weight,
        plan,
        prewarm,
        quantize_input,
        run,
    )

install_lazy_api(globals(), META)
