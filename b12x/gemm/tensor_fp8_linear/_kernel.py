from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass

import cutlass.cute as cute
import torch

from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8_SCALE_K_TILE,
    MXFP8_SCALE_ROW_TILE,
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    pack_mxfp8_scales_for_dense_gemm,
)

# ---------------------------------------------------------------------------
# Graph-safe bounded admission cache for unit-scale MMA tensors.
#
# Each entry is a CUDA tensor whose raw SFA pointer may be embedded in an
# active CUDA graph.  Therefore entries are immutable and process-lifetime
# stable: there is no automatic eviction and no public clear operation that
# could free a graph-owned pointer.  When the cache reaches its entry-count or
# byte-budget ceiling, new geometries are rejected with an explicit error so
# the caller can raise the limits or prune model geometries before capture.
#
# Public admission contract
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The unit-scale MMA tensor cache is a per-device, process-lifetime table.
# Once a geometry is admitted its backing storage is never freed, ensuring
# that CUDA-graph-captured SFA pointers remain valid for the life of the
# process.  Two environment variables control the per-device ceiling and must
# be set **before** importing this module:
#
#   B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_ENTRIES  (default 32, hard max 256)
#   B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_BYTES   (default 64 MiB, hard max 256 MiB)
#
# Invalid, zero, or negative values fall back to the defaults; values above
# the hard maxima are clamped.  When a new geometry would exceed the per-device
# entry count or byte budget, a ``RuntimeError`` is raised.  Because admitted
# entries cannot be evicted, serving integrations should call ``prewarm`` with
# all expected token-count geometries during startup.  ``prewarm`` is
# **non-atomic**: it admits geometries one at a time in sorted order, so a
# failure after partial admission leaves all earlier canonical geometries
# permanently admitted until the process is restarted; the remaining
# geometries stay unadmittable.  Plan capacity accordingly.
#
# Row keys are canonicalized to the physical 128-row scale tile: token counts
# 1–128 (or 129–256, etc.) that produce identical packed all-127 storage share
# a single immutable allocation, reducing cardinality.
# ---------------------------------------------------------------------------

# Hard ceilings that cannot be exceeded even via environment configuration.
_UNIT_SCALE_HARD_MAX_ENTRIES = 256
_UNIT_SCALE_HARD_MAX_BYTES = 256 * 1024 * 1024


def _parse_cache_limit(raw: str | None, default: int, hard_max: int) -> int:
    """Parse a cache limit from an env var, falling back to *default*."""
    if raw is None:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        return default
    if value <= 0:
        return default
    return min(value, hard_max)


_UNIT_SCALE_CACHE_MAX_ENTRIES = _parse_cache_limit(
    os.environ.get("B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_ENTRIES"),
    32,
    _UNIT_SCALE_HARD_MAX_ENTRIES,
)
_UNIT_SCALE_CACHE_MAX_BYTES = _parse_cache_limit(
    os.environ.get("B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_BYTES"),
    64 * 1024 * 1024,
    _UNIT_SCALE_HARD_MAX_BYTES,
)


def _unit_scale_packed_bytes(rows: int, width: int) -> int:
    """Compute the exact backing-storage byte count of the packed unit-scale
    MMA tensor **without** allocating it.

    The packer rounds *rows* up to ``MXFP8_SCALE_ROW_TILE`` (128) and
    ``sf_k = width // 32`` up to ``MXFP8_SCALE_K_TILE`` (4).  The contiguous
    physical tensor has ``num_groups * m_tiles * k_tiles * 32 * 4 * 4`` bytes
    (each element is ``float8_e8m0fnu`` = 1 byte).  With ``num_groups = 1``
    this simplifies to ``m_tiles * k_tiles * 512``.
    """
    m_tiles = _align_up(rows, MXFP8_SCALE_ROW_TILE) // MXFP8_SCALE_ROW_TILE
    sf_k = width // MXFP8_SCALE_VEC_SIZE
    k_tiles = _align_up(sf_k, MXFP8_SCALE_K_TILE) // MXFP8_SCALE_K_TILE
    return m_tiles * k_tiles * 32 * 4 * 4


def _canonical_rows(rows: int) -> int:
    """Round *rows* up to the physical 128-row scale tile boundary.

    All row counts within the same tile produce identical packed all-127
    storage, so they share one immutable cache entry.
    """
    return _align_up(int(rows), MXFP8_SCALE_ROW_TILE)


@dataclass
class _DeviceUnitScaleCache:
    """Per-device immutable entry table with byte accounting."""

    entries: dict[tuple[int, int], torch.Tensor]
    total_bytes: int
    lock: threading.Lock
    prewarmed_launches: set[tuple[int, int, int, torch.dtype, int]]


_unit_scale_caches: dict[tuple[str, int | None], _DeviceUnitScaleCache] = {}
_unit_scale_caches_guard = threading.Lock()


def _get_device_cache(
    device_key: tuple[str, int | None],
) -> _DeviceUnitScaleCache:
    """Return (creating if needed) the per-device admission cache."""
    with _unit_scale_caches_guard:
        dc = _unit_scale_caches.get(device_key)
        if dc is None:
            dc = _DeviceUnitScaleCache(
                entries={},
                total_bytes=0,
                prewarmed_launches=set(),
                lock=threading.Lock(),
            )
            _unit_scale_caches[device_key] = dc
    return dc


def _reset_unit_scale_cache() -> None:
    """Drop every per-device cache entry.  For tests only — never call while
    CUDA graphs that reference cached tensors may still be replayed."""
    with _unit_scale_caches_guard:
        _unit_scale_caches.clear()


@dataclass(frozen=True)
class TensorFP8LinearWeight:
    """Serialized E4M3 weight and static scale metadata for dense GEMM."""

    values: torch.Tensor
    scale_mma: torch.Tensor
    output_scale: torch.Tensor
    in_features: int
    padded_in_features: int
    out_features: int


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"tensor FP8 linear output must be bf16/fp16, got {dtype}")


def _source_2d(source: torch.Tensor) -> torch.Tensor:
    if source.ndim < 2:
        raise ValueError(f"source must have at least 2 dims, got {tuple(source.shape)}")
    return source.reshape(-1, source.shape[-1]).contiguous()


def _pad_k(tensor: torch.Tensor, padded_k: int) -> torch.Tensor:
    rows, width = map(int, tensor.shape)
    if width == padded_k:
        return tensor.contiguous()
    padded = tensor.new_zeros((rows, padded_k))
    padded[:, :width] = tensor
    return padded.contiguous()


def _unit_scale_mma(rows: int, width: int, device: torch.device) -> torch.Tensor:
    scale_rows = torch.full(
        (rows, width // MXFP8_SCALE_VEC_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    return pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=rows,
        k=width,
        num_groups=1,
    )


def _cached_unit_scale_mma(
    device_type: str,
    device_index: int | None,
    rows: int,
    width: int,
) -> torch.Tensor:
    """Return a unit-scale MMA tensor from a bounded per-device admission cache.

    Entries are immutable and process-lifetime stable: once a geometry is
    admitted, the same tensor object is returned for every subsequent call
    with the same key, preserving the stable-allocation guarantee required
    for CUDA graph replay.  No entry is ever evicted, so a graph-captured
    pointer remains valid for the lifetime of the process.

    Row counts are canonicalized to the physical 128-row scale tile, so token
    counts 1–128 (or 129–256, etc.) that produce identical packed all-127
    storage share one immutable allocation.

    When the per-device entry count or byte budget would be exceeded, the new
    geometry is rejected with a :class:`RuntimeError` **before** any CUDA
    allocation is attempted.

    Construction is single-flight under a per-device lock.  On cold creation
    the concrete device is synchronized before the entry is published, so
    cross-thread streams cannot consume incomplete data.
    """
    device_key = (device_type, device_index)
    # Canonicalize rows to the physical 128-row tile boundary.
    canon_rows = _canonical_rows(rows)
    entry_key = (canon_rows, int(width))
    dc = _get_device_cache(device_key)

    # Single-flight construction under the per-device lock.
    with dc.lock:
        existing = dc.entries.get(entry_key)
        if existing is not None:
            return existing

        if device_type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "tensor-FP8 unit-scale geometry was not prewarmed before "
                "CUDA graph capture"
            )
        # Admission control: reject if at capacity.
        if len(dc.entries) >= _UNIT_SCALE_CACHE_MAX_ENTRIES:
            raise RuntimeError(
                f"tensor-FP8 unit-scale cache for device {device_key} is full "
                f"({len(dc.entries)}/{_UNIT_SCALE_CACHE_MAX_ENTRIES} entries). "
                "Increase B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_ENTRIES (hard "
                f"max {_UNIT_SCALE_HARD_MAX_ENTRIES}) or reduce the number of "
                "distinct token-count/width geometries before CUDA graph "
                "capture.  Admitted entries are process-lifetime; restart the "
                "process to reset the cache."
            )

        # Precompute exact backing bytes before allocating.
        tensor_bytes = _unit_scale_packed_bytes(canon_rows, int(width))

        # Reject a single entry that exceeds the entire byte budget.
        if tensor_bytes > _UNIT_SCALE_CACHE_MAX_BYTES:
            raise RuntimeError(
                f"tensor-FP8 unit-scale tensor for rows={rows}, width={width} "
                f"requires {tensor_bytes} bytes, exceeding the per-device byte "
                f"budget of {_UNIT_SCALE_CACHE_MAX_BYTES} (hard max "
                f"{_UNIT_SCALE_HARD_MAX_BYTES}). Increase "
                "B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_BYTES or reduce "
                "the padded K width."
            )

        if dc.total_bytes + tensor_bytes > _UNIT_SCALE_CACHE_MAX_BYTES:
            raise RuntimeError(
                f"admitting tensor-FP8 unit-scale tensor for rows={rows}, "
                f"width={width} ({tensor_bytes} bytes) would exceed the "
                f"per-device byte budget ({dc.total_bytes}/"
                f"{_UNIT_SCALE_CACHE_MAX_BYTES} bytes already used, hard max "
                f"{_UNIT_SCALE_HARD_MAX_BYTES}). Increase "
                "B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_BYTES or reduce the "
                "number of distinct token-count/width geometries before CUDA "
                "graph capture.  Admitted entries are process-lifetime; "
                "restart the process to reset the cache."
            )

        device = torch.device(device_type, device_index)
        tensor = _unit_scale_mma(canon_rows, width, device)

        # Synchronize the concrete device so the tensor data is visible to
        # any stream before publication.
        if device_type == "cuda":
            torch.cuda.synchronize(device)

        dc.entries[entry_key] = tensor
        dc.total_bytes += tensor_bytes

    return tensor


def _launch_key(
    *,
    tokens: int,
    padded_in_features: int,
    out_features: int,
    out_dtype: torch.dtype,
    expected_m: int,
) -> tuple[int, int, int, torch.dtype, int]:
    return (
        int(tokens),
        int(padded_in_features),
        int(out_features),
        out_dtype,
        int(expected_m),
    )


def _require_capture_launch_prewarmed(
    device: torch.device,
    launch_key: tuple[int, int, int, torch.dtype, int],
) -> None:
    if device.type != "cuda" or not torch.cuda.is_current_stream_capturing():
        return
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    dc = _get_device_cache((device.type, device_index))
    with dc.lock:
        if launch_key not in dc.prewarmed_launches:
            raise RuntimeError(
                "tensor-FP8 launch geometry was not prewarmed before "
                "CUDA graph capture"
            )


def _prewarm_unit_scales_atomic(
    device: torch.device,
    rows: Iterable[int],
    width: int,
) -> None:
    """Admit a complete prewarm batch before any serving-shape compilation."""
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    device_key = (device.type, device_index)
    dc = _get_device_cache(device_key)
    entry_keys = sorted({(_canonical_rows(row), int(width)) for row in rows})

    with dc.lock:
        missing = [key for key in entry_keys if key not in dc.entries]
        if not missing:
            return
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "tensor-FP8 unit-scale geometries must be prewarmed before "
                "CUDA graph capture"
            )
        if len(dc.entries) + len(missing) > _UNIT_SCALE_CACHE_MAX_ENTRIES:
            raise RuntimeError(
                f"tensor-FP8 unit-scale cache for device {device_key} is full: "
                f"{len(dc.entries)} existing + {len(missing)} requested exceeds "
                f"{_UNIT_SCALE_CACHE_MAX_ENTRIES}"
            )
        sizes = {
            key: _unit_scale_packed_bytes(key[0], key[1]) for key in missing
        }
        requested_bytes = sum(sizes.values())
        if (
            any(size > _UNIT_SCALE_CACHE_MAX_BYTES for size in sizes.values())
            or dc.total_bytes + requested_bytes > _UNIT_SCALE_CACHE_MAX_BYTES
        ):
            raise RuntimeError(
                f"tensor-FP8 unit-scale prewarm for device {device_key} "
                f"requires {requested_bytes} bytes with {dc.total_bytes} "
                f"already admitted, exceeding {_UNIT_SCALE_CACHE_MAX_BYTES}"
            )

        created = {
            key: _unit_scale_mma(key[0], key[1], device) for key in missing
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dc.entries.update(created)
        dc.total_bytes += requested_bytes


def _activation_scale_mma(
    source: torch.Tensor,
    rows: int,
    width: int,
) -> torch.Tensor:
    device_index = source.device.index
    if source.device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_unit_scale_mma(
        source.device.type,
        device_index,
        int(rows),
        int(width),
    )


def _dense_gemm_kwargs_for_n(out_features: int) -> dict[str, object]:
    if int(out_features) < 64:
        return {"mma_tiler_mn": (64, 32), "swap_ab": True}
    return {}


def is_tensor_fp8_linear_supported() -> tuple[bool, str | None]:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        return False, "CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op"
    return True, None


def pack_tensor_fp8_linear_weight(
    weight: torch.Tensor,
    output_scale: torch.Tensor,
) -> TensorFP8LinearWeight:
    """Pack an E4M3 ``[N,K]`` weight with one combined dequantization scale."""

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("output_scale", output_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if output_scale.dtype != torch.float32 or output_scale.numel() != 1:
        raise ValueError(
            "output_scale must be one float32 value, got "
            f"dtype={output_scale.dtype}, shape={tuple(output_scale.shape)}"
        )
    if output_scale.device != weight.device:
        raise ValueError("weight and output_scale must be on the same device")
    if not bool(torch.isfinite(output_scale).all()) or bool((output_scale < 0).any()):
        raise ValueError("output_scale must be finite and non-negative")

    out_features, in_features = map(int, weight.shape)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "tensor FP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    padded_in_features = _align_up(in_features, 128)
    values = _pad_k(weight, padded_in_features)
    scale_mma = _unit_scale_mma(
        out_features,
        padded_in_features,
        weight.device,
    )
    return TensorFP8LinearWeight(
        values=values,
        scale_mma=scale_mma,
        output_scale=output_scale.reshape(1).contiguous(),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


@torch.library.custom_op(
    "b12x::tensor_fp8_linear_fused",
    mutates_args=(),
)
def _tensor_fp8_linear_fused_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    output_scale: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
) -> torch.Tensor:
    tokens = int(source_2d.shape[0])
    # Guard all CUDA work inside the source device context so compilation
    # and stream resolution use the correct device, not the thread's current
    # device which may differ in multi-GPU serving.
    with torch.cuda.device(source_2d.device):
        source_padded = _pad_k(source_2d, int(padded_in_features))
        source_scale_mma = _activation_scale_mma(
            source_padded,
            tokens,
            int(padded_in_features),
        )
        return dense_gemm(
            (
                source_padded.reshape(tokens, padded_in_features, 1),
                source_scale_mma,
            ),
            (
                weight_values.reshape(out_features, padded_in_features, 1),
                weight_scale_mma,
            ),
            alpha=output_scale,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype=_output_dtype_name(out_dtype),
            sf_vec_size=MXFP8_SCALE_VEC_SIZE,
            expected_m=expected_m,
            stream=stream_int,
            plain_fp8=True,
            **_dense_gemm_kwargs_for_n(out_features),
        )[:, :, 0]


@_tensor_fp8_linear_fused_op.register_fake
def _tensor_fp8_linear_fused_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    output_scale: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
) -> torch.Tensor:
    del weight_values, weight_scale_mma, output_scale
    del padded_in_features, expected_m, stream_int
    return torch.empty(
        (source_2d.shape[0], out_features),
        dtype=out_dtype,
        device=source_2d.device,
    )


def tensor_fp8_linear(
    source: torch.Tensor,
    packed_weight: TensorFP8LinearWeight,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run static per-tensor E4M3 operands through the SM12x dense GEMM.

    ``stream`` must be a ``torch.cuda.Stream`` on the same device as the
    operands, or ``None`` to use the current stream for the operand device.
    Raw integer handles are **not** accepted; the caller must pass a proper
    ``torch.cuda.Stream`` so that device and ordering can be validated.

    All materialization (source reshape, K padding, cache access, output
    allocation, raw GEMM launch, and bias addition) executes under
    ``torch.cuda.device(operand_device)`` plus ``torch.cuda.stream(target)``
    so that temporary tensors are allocated on the target stream.

    **Stream lifetime:** caller-owned tensors (``source``, packed weight
    fields, and ``bias``) are recorded on the target stream via
    ``record_stream`` so the caching allocator does not recycle their
    memory while the raw GEMM launch is still reading them on the target
    stream.  The caller is responsible for ensuring that producers of
    these tensors have completed before the target stream consumes them
    (e.g. by waiting on the producer stream's event on the target stream).
    The returned ``output`` tensor is allocated on the target stream.  The
    caller must ensure the target stream has completed before reading the
    output from a different stream, either by calling
    ``target_stream.synchronize()`` or by having the consumer stream call
    ``consumer.wait_stream(target_stream)`` (or wait on an event recorded on
    the target stream).  Separately, if the caller intends to read the
    output from a different stream, ``output.record_stream(consumer_stream)``
    protects the allocator lifetime of the output tensor.

    Each distinct ``(canonical_rows, padded_K, device)`` geometry admitted by
    this call is cached permanently for CUDA graph replay safety.  See the
    module-level admission contract for env-var limits, defaults, and
    restart semantics.
    """

    _check_gpu_tensor("source", source)
    if not isinstance(packed_weight, TensorFP8LinearWeight):
        raise TypeError("packed_weight must be a TensorFP8LinearWeight")
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(
            "stream must be a torch.cuda.Stream or None; raw integer handles "
            "are not accepted"
        )
    if stream is not None and stream.device != source.device:
        raise ValueError(
            f"stream is on device {stream.device} but source is on "
            f"{source.device}"
        )

    device = source.device
    if device.type == "cuda":
        target_stream = stream if stream is not None else torch.cuda.current_stream(device)
        device_ctx = torch.cuda.device(device)
        stream_ctx = torch.cuda.stream(target_stream)
    else:
        target_stream = None
        device_ctx = nullcontext()
        stream_ctx = nullcontext()

    # Record caller-owned tensors on the target stream so the caching
    # allocator does not recycle their memory while the raw GEMM launch
    # (which reads them via raw pointers on target_stream) is still in
    # flight.  Tensors allocated inside the stream context below
    # (source_2d, padded source) are already on the target stream.  The
    # cached unit-scale tensor is process-lifetime stable.
    if target_stream is not None:
        current_stream = torch.cuda.current_stream(device)
        if target_stream != current_stream:
            target_stream.wait_stream(current_stream)
        source.record_stream(target_stream)
        packed_weight.values.record_stream(target_stream)
        packed_weight.scale_mma.record_stream(target_stream)
        packed_weight.output_scale.record_stream(target_stream)
        if bias is not None:
            bias.record_stream(target_stream)

    with device_ctx, stream_ctx:
        source_2d = _source_2d(source)
        tokens, in_features = map(int, source_2d.shape)
        if source_2d.dtype != torch.float8_e4m3fn:
            raise ValueError(f"source must be float8_e4m3fn, got {source_2d.dtype}")
        if in_features != int(packed_weight.in_features):
            raise ValueError(
                f"input K={in_features} does not match packed weight K="
                f"{packed_weight.in_features}"
            )
        if packed_weight.values.device != source_2d.device:
            raise ValueError("source and packed weight must be on the same device")
        if expected_m is not None and int(expected_m) <= 0:
            raise ValueError("expected_m must be positive when provided")
        _output_dtype_name(out_dtype)
        launch_key = _launch_key(
            tokens=tokens,
            padded_in_features=packed_weight.padded_in_features,
            out_features=packed_weight.out_features,
            out_dtype=out_dtype,
            expected_m=int(expected_m) if expected_m is not None else tokens,
        )
        _require_capture_launch_prewarmed(device, launch_key)

        out_features = int(packed_weight.out_features)
        if bias is not None:
            _check_gpu_tensor("bias", bias)
            if bias.device != source_2d.device:
                raise ValueError("bias must be on the same device as source")
            if bias.dtype != out_dtype or bias.shape != (out_features,):
                raise ValueError(
                    f"bias must have shape {(out_features,)} and dtype {out_dtype}, "
                    f"got shape={tuple(bias.shape)}, dtype={bias.dtype}"
                )
        if tokens == 0:
            output = torch.empty(
                (0, out_features),
                dtype=out_dtype,
                device=source_2d.device,
            )
        else:
            output = torch.ops.b12x.tensor_fp8_linear_fused(
                source_2d,
                packed_weight.values,
                packed_weight.scale_mma,
                packed_weight.output_scale,
                packed_weight.padded_in_features,
                packed_weight.out_features,
                int(expected_m) if expected_m is not None else tokens,
                out_dtype,
                int(target_stream.cuda_stream) if target_stream is not None else None,
            )
        if bias is not None:
            output = output + bias
    return output.view(*source.shape[:-1], out_features)


def prewarm_tensor_fp8_linear(
    packed_weight: TensorFP8LinearWeight,
    token_counts: Iterable[int],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> int:
    """Compile and cache serving shapes before CUDA graph capture.

    The full normalized geometry set is checked against the permanent
    per-device admission limits before any scale allocation or GEMM
    compilation. Invalid stream arguments are rejected before dummy tensors
    are allocated.
    """
    if not isinstance(packed_weight, TensorFP8LinearWeight):
        raise TypeError("packed_weight must be a TensorFP8LinearWeight")
    device = packed_weight.values.device
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(
            "stream must be a torch.cuda.Stream or None; raw integer handles "
            "are not accepted"
        )
    if stream is not None and stream.device != device:
        raise ValueError(
            f"stream is on device {stream.device} but packed weight is on {device}"
        )

    normalized_tokens = sorted(
        {int(value) for value in token_counts if int(value) > 0}
    )
    if not normalized_tokens:
        return 0
    if stream is not None:
        target_stream = stream
    elif device.type == "cuda":
        target_stream = torch.cuda.current_stream(device)
    else:
        target_stream = None

    device_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    stream_ctx = (
        torch.cuda.stream(target_stream)
        if target_stream is not None
        else nullcontext()
    )
    with torch.inference_mode(), device_ctx, stream_ctx:
        _prewarm_unit_scales_atomic(
            device,
            normalized_tokens,
            int(packed_weight.padded_in_features),
        )
        for tokens in normalized_tokens:
            source = torch.zeros(
                (tokens, packed_weight.in_features),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            tensor_fp8_linear(
                source,
                packed_weight,
                out_dtype=out_dtype,
                expected_m=tokens,
                stream=stream,
            )
            device_index = device.index
            if device.type == "cuda" and device_index is None:
                device_index = torch.cuda.current_device()
            dc = _get_device_cache((device.type, device_index))
            launch_key = _launch_key(
                tokens=tokens,
                padded_in_features=packed_weight.padded_in_features,
                out_features=packed_weight.out_features,
                out_dtype=out_dtype,
                expected_m=tokens,
            )
            with dc.lock:
                dc.prewarmed_launches.add(launch_key)
    return len(normalized_tokens)


__all__ = [
    "TensorFP8LinearWeight",
    "is_tensor_fp8_linear_supported",
    "pack_tensor_fp8_linear_weight",
    "prewarm_tensor_fp8_linear",
    "tensor_fp8_linear",
]
