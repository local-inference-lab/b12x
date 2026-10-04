"""Prepared standalone M-grouped FP8 GEMM (masked and contiguous modes).

Ports the DeepGEMM-style grouped contracts onto the b12x planned-op spine
with E4M3 A/B values and UE8M0 block scales, BF16 output, FP32 accumulate:

- ``masked_mm((a, sfa), (b, sfb), d, masked_m, ...)``: per-group row masks,
  A ``(G, m_cap, k)`` with gran-128 f32 scales, D ``(G, m_cap, n)``;
  ``masked_m`` is a device int32 ``(G,)`` tensor the host never reads.
  Callers must keep values in ``[0, m_cap]``; values are not range-validated.
  A, B and D must be contiguous.
  The masked scheduler builds device-side prefix sums of live M-tile counts
  from ``masked_m`` and visits only live tiles, without a host count read.
  ``masked_compact`` packs B scales as compact gran-128 UE8M0 bytes and expands
  them in the kernel for block-scaled MMA; A scales retain the expanded layout.
  The ``single`` fallback uses expanded B scales with the same live-tile
  scheduler. Rows beyond the mask remain contract-undefined.
- ``contiguous_mm((a, sfa), (b, sfb), d, labels, ...)``: label-run layout,
  A ``(m_total, k)`` with gran-32 f32 scales, D ``(m_total, n)``. Each M tile
  selects its group from the device ``labels`` tensor at the tile's first
  row (the ``mgroup_labels`` variant of the dense engine); whole padding
  tiles are skipped, and straddling padding rows (label -1) are zero-filled
  after the GEMM. Labels must be -1 or in [0, G). Non-padding runs start at
  multiples of 128. Packing validates these conditions on the GPU; violations
  cause an asynchronous CUDA device error, including during graph replay.
  Contiguous scale workspaces have fixed planned capacity. Joint plans mark
  active groups on-device before packing and rewrite only reachable B scales
  on each call; inactive groups' bytes are not consumed by GEMM.

``plan(query)`` declares static geometry and planned capacity only; live
masked/label/m_total values are runtime scalars and never enter compile or
cache keys. ``query_from_call`` derives the static query from operand
metadata without reading device tensors. A captured graph keeps tensor shapes
fixed; live activity within that capacity changes through labels or masks.
Unseen tensor row lengths can reuse a prepared plan outside that capture.
Both modes own private capacity-sized scale buffers and are not reentrant:
serialize calls and graph replays. Concurrent execution requires separate
plans and output buffers. Masked packing retains the current tensor row stride
within its capacity allocation.

``joint_v1`` uses GPU selection between full BK64 and narrow BK128 bodies,
compact B scales, and a band-local/half-line B cache policy. Its full body uses
packed scale-factor registers with MMA byte selectors; the narrow body does not.
Direct shared-scale publication uses all-lane arrivals in both producer/full
and consumer/empty directions. Supported MXFP8 tiles have N64 or N128 width.
See ``docs/grouped-fp8.md`` for bounded measurements and qualification limits.
The optimized defaults require the exact NVIDIA RTX PRO 6000 Blackwell Server Edition
identity (normalized vendor/product name, SM120, 188 SMs): masked mode selects
``masked_compact``; contiguous mode selects ``joint_v1`` for capacity
<= 131072, ceil(N / 128) >= 16 and K >= 2048, or for 64 groups with
(N, K) in {(1024, 4096), (4096, 512)} and 4096 < capacity <= 65536,
or for 96 groups with (N, K) in {(1152, 5120), (5120, 640)} and the same
capacity bounds, or for 384 groups with (N, K) = (576, 5120) and
4096 < capacity <= 131072.
Other cases use ``single``. These static dispatch bounds do not guarantee
performance for every live label distribution.
Configuration validation rejects ``joint_v1`` and ``masked_compact`` on SM121.
SM121 has only the single-body fallback; neither functionality nor performance
of this op has been qualified there. Availability and default selection do not
establish full-matrix performance parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mgroup_fp8_gemm",
    group="gemm",
    api_style="planned",
    entry_points=(
        "MGroupFP8GemmQuery",
        "MGroupFP8GemmConfig",
        "plan",
        "query_from_call",
        "masked_mm",
        "contiguous_mm",
        "is_supported",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="5c50b153",
        paths=("b12x/_lib/dense_gemm.py",),
    ),
    test_path="tests/gemm/test_mgroup_fp8_gemm.py",
    since="1.3.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        MGroupFP8GemmConfig,
        MGroupFP8GemmQuery,
        contiguous_mm,
        is_supported,
        masked_mm,
        plan,
        query_from_call,
    )

install_lazy_api(globals(), META)
