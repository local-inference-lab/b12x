"""Public surface for gemm.weight_first_gemv (docs in the op ``__init__``)."""

from __future__ import annotations

import os

import torch

from ..._lib.gating import default_is_supported
from . import META
from ._kernel import BRICKS, MAX_ROWS, STAGE_DEPTHS, brick_for  # noqa: F401
from ._kernel import compile_weight_first_gemv, get_cached_weight_first_gemv
from ._kernel import weight_first_gemv  # noqa: F401  (registers the op; alias)

#: Staging depth used when the caller does not choose one: ``cp.async``
#: groups (4 KB per CTA each) left in flight per CTA while a brick is staged
#: beside the allreduce; 0 issues the whole brick at once. Unbounded staging
#: competes with the peers' reads of this GPU's memory and slows the
#: allreduce.
DEFAULT_STAGE_DEPTH = 1

#: (n, k, nt, kt, device index) geometries compiled and warm-run in this
#: process.
_PRECOMPILED: set[tuple[int, int, int, int, int]] = set()


def is_disabled() -> bool:
    """True when ``B12X_DISABLE_WEIGHT_FIRST_GEMV`` turns the op off
    (debug isolation switch)."""
    return os.environ.get("B12X_DISABLE_WEIGHT_FIRST_GEMV", "").lower() in (
        "1",
        "true",
        "yes",
    )


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with the required CUTLASS DSL, unless disabled via
    ``B12X_DISABLE_WEIGHT_FIRST_GEMV``."""
    if is_disabled():
        return False
    return default_is_supported(device, requires=META.requires)


def supports(weights: list[torch.Tensor], x: torch.Tensor | None = None) -> bool:
    """Whether the kernel serves ``weights`` (and ``x`` when given).

    Weights: one or two two-dimensional contiguous BF16 tensors on one CUDA
    device the op supports (``is_supported``: SM120/SM121 and not disabled)
    with a common ``K`` that is a multiple of a brick width (a brick exists
    for ``K`` a multiple of 256). Activation: two-dimensional BF16 ``(M, K)``
    on the weights' device with ``1 <= M <= MAX_ROWS``. Bias handling is the
    caller's responsibility; the op has no bias input.
    """
    if not 1 <= len(weights) <= 2:
        return False
    first = weights[0]
    if first.dim() != 2 or first.dtype != torch.bfloat16 or not first.is_cuda:
        return False
    k = int(first.shape[1])
    for w in weights:
        if (
            w.dim() != 2
            or w.dtype != torch.bfloat16
            or w.device != first.device
            or int(w.shape[1]) != k
            or not w.is_contiguous()
        ):
            return False
    try:
        brick_for(sum(int(w.shape[0]) for w in weights), k)
    except ValueError:
        return False
    if not is_supported(first.device):
        return False
    if x is None:
        return True
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.device == first.device
        and int(x.shape[1]) == k
        and 1 <= int(x.shape[0]) <= MAX_ROWS
    )


def precompile(n: int, k: int, device: torch.device, log=None) -> tuple[int, int]:
    """Compile and warm-run the kernel for a concatenated weight ``(n, k)``.

    Called at weight-load time so the serving path never compiles or loads a
    module inside a CUDA-graph capture. The warm run covers one and
    ``MAX_ROWS`` rows against zero inputs; every row count in between reuses
    the same compiled callable. Returns the brick ``(nt, kt)``.
    """
    if log is None:
        import logging

        log = logging.getLogger("b12x.weight_first_gemv")
    nt, kt = brick_for(n, k)
    # Compile, allocate and warm-run on ``device`` whatever the current
    # device: the compiled callable binds the current stream at each call,
    # and the module load the warm run triggers is per device.
    with torch.cuda.device(device):
        key = (int(n), int(k), nt, kt, torch.cuda.current_device())
        if key in _PRECOMPILED:
            return nt, kt
        log.info(
            "weight-first GEMV precompile: n=%d k=%d brick %dx%d device=%d",
            n,
            k,
            nt,
            kt,
            key[4],
        )
        launch = compile_weight_first_gemv(n, k, nt, kt)
        weight = torch.zeros(n, k, dtype=torch.bfloat16, device=device)
        partial = torch.zeros(k // kt, 16, n, dtype=torch.float32, device=device)
        counters = torch.zeros((n + nt - 1) // nt, dtype=torch.int32, device=device)
        for m in (1, MAX_ROWS):
            x = torch.zeros(m, k, dtype=torch.bfloat16, device=device)
            y0 = torch.empty(m, n, dtype=torch.bfloat16, device=device)
            for pdl in (False, True):
                launch(
                    x, weight, y0, y0, partial, counters, m, n, DEFAULT_STAGE_DEPTH, pdl
                )
        torch.cuda.synchronize(device)
        del weight, partial, counters
    _PRECOMPILED.add(key)
    return nt, kt


class WeightFirstProjection:
    """One or two BF16 weights ``(N_i, K)`` served as one weight-first
    projection: ``proj(x)`` returns one ``(M, N_i)`` output per weight.

    The weights are concatenated along ``N`` into one contiguous buffer and
    each parameter's storage is replaced by a view into it, so no extra
    weight memory is held. The fp32 split partials and the n-tile counters
    are allocated once per instance and have stable addresses; a launch
    allocates only its outputs, which the caller's allocator serves from the
    graph pool during capture.

    ``depth`` is the staging depth (``DEFAULT_STAGE_DEPTH`` when ``None``)
    and ``pdl`` sets the programmatic-stream-serialization attribute on the
    launch. Both are runtime launch arguments, not compile keys.
    """

    def __init__(
        self,
        weights: list[torch.Tensor],
        *,
        depth: int | None = None,
        pdl: bool = True,
        precompile_kernel: bool = True,
    ):
        if not supports(weights):
            raise ValueError(
                "weight-first projection needs one or two contiguous BF16 [N, K] "
                "weights on one supported CUDA device (SM120/SM121, op not "
                "disabled) with K a multiple of 256"
            )
        self.n_parts = [int(w.shape[0]) for w in weights]
        self.n = sum(self.n_parts)
        self.k = int(weights[0].shape[1])
        self.n0 = self.n_parts[0]
        self.device = weights[0].device
        cat = torch.cat([w.detach() for w in weights], dim=0).contiguous()
        offset = 0
        for w, ni in zip(weights, self.n_parts, strict=True):
            w.data = cat[offset : offset + ni]
            offset += ni
        self.weight = cat
        self.nt, self.kt = brick_for(self.n, self.k)
        self.partial = torch.zeros(
            self.k // self.kt, 16, self.n, dtype=torch.float32, device=self.device
        )
        self.counters = torch.zeros(
            (self.n + self.nt - 1) // self.nt, dtype=torch.int32, device=self.device
        )
        self.depth = DEFAULT_STAGE_DEPTH if depth is None else int(depth)
        if self.depth not in STAGE_DEPTHS:
            raise ValueError(
                f"stage depth {self.depth} is not honored by the kernel; use 0 "
                f"(unbounded) or one of {sorted(STAGE_DEPTHS - {0})}"
            )
        self.pdl = bool(pdl)
        if precompile_kernel:
            precompile(self.n, self.k, self.device)

    @property
    def compiled(self) -> bool:
        """Whether the kernel for this geometry is compiled in this process."""
        return (
            get_cached_weight_first_gemv(self.n, self.k, self.nt, self.kt) is not None
        )

    def supports(self, x: torch.Tensor) -> bool:
        """Whether ``x`` takes the kernel rather than the cuBLAS fallback."""
        return (
            x.dim() == 2
            and x.dtype == torch.bfloat16
            and x.device == self.device
            and int(x.shape[1]) == self.k
            and 1 <= int(x.shape[0]) <= MAX_ROWS
        )

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y0, y1 = torch.ops.b12x.weight_first_gemv(
            x,
            self.weight,
            self.partial,
            self.counters,
            self.n0,
            self.nt,
            self.kt,
            self.depth,
            self.pdl,
        )
        return (y0, y1) if len(self.n_parts) == 2 else (y0,)


__all__ = [
    "WeightFirstProjection",
    "weight_first_gemv",
    "brick_for",
    "supports",
    "precompile",
    "is_supported",
    "is_disabled",
    "MAX_ROWS",
    "DEFAULT_STAGE_DEPTH",
    "STAGE_DEPTHS",
    "BRICKS",
]
