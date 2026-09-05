"""Public API for :mod:`b12x.gemm.trellis_linear`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..._lib.gating import default_is_supported
from ...moe._shared.kernels.w4a16.host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    packed_gemm_scratch_elements,
)
from ...moe._shared.kernels.w4a16.kernel import (
    clear_w4a16_kernel_cache,
    run_trellis256_dense,
)
from ...moe._shared.kernels.w4a16.prepare import (
    PreparedTrellis256DenseWeight,
    prepare_trellis256_dense_weight,
    prepare_trellis256_pair_dense_weight,
)
from . import META
from ._low_rank import (
    clear_low_rank_caches,
    run_low_rank_additive,
    run_low_rank_pair_add,
    run_low_rank_pair_project,
)

PreparedWeight = PreparedTrellis256DenseWeight


def _has_16_byte_alignment(tensor: torch.Tensor) -> bool:
    """Check real storage while allowing fake-tensor graph tracing."""
    return torch.compiler.is_compiling() or int(tensor.data_ptr()) % 16 == 0


@dataclass(frozen=True)
class PreparedAdditiveWeight:
    """Packed Trellis weight plus an additive BF16 rank-16 correction.

    ``a_t`` is the preparation-time rank-major copy of the checkpoint's
    ``[in_features, rank]`` A factor. ``b`` retains the checkpoint's
    ``[out_features, rank]`` B factor by reference. Execution evaluates
    ``base(x) + (x @ a_t.T) @ b.T`` without materializing a decoded base matrix.
    """

    base: PreparedWeight
    a_t: torch.Tensor
    b: torch.Tensor
    rank: int


@dataclass(frozen=True)
class PreparedAdditivePair:
    """Two same-shaped packed weights with jointly executed rank-16 factors."""

    left: PreparedWeight
    right: PreparedWeight
    a_t: torch.Tensor
    b: torch.Tensor
    rank: int


@dataclass(frozen=True)
class Buffers:
    """Caller-owned storage for one fixed-shape dense Trellis execution."""

    output: torch.Tensor
    gemm_output: torch.Tensor
    c_tmp: torch.Tensor
    rotated_f16: torch.Tensor
    input_f16: Optional[torch.Tensor] = None
    rotated_compute: Optional[torch.Tensor] = None
    gemm_output_f16: Optional[torch.Tensor] = None
    output_f16: Optional[torch.Tensor] = None
    low_rank_hidden: Optional[torch.Tensor] = None

    def run_kwargs(self) -> dict[str, Optional[torch.Tensor]]:
        """Return keyword arguments accepted by :func:`run`."""
        return {
            "output": self.output,
            "gemm_output": self.gemm_output,
            "c_tmp": self.c_tmp,
            "input_f16": self.input_f16,
            "rotated_f16": self.rotated_f16,
            "rotated_compute": self.rotated_compute,
            "gemm_output_f16": self.gemm_output_f16,
            "output_f16": self.output_f16,
        }


@dataclass(frozen=True)
class PairBuffers:
    """Caller-owned storage for one fixed-shape paired dense execution."""

    base: Buffers
    output: torch.Tensor
    low_rank_hidden: torch.Tensor


def view_buffers(buffers: Buffers, *, size_m: int) -> Buffers:
    """Return exact-row views backed by one preallocated buffer capacity."""

    if not isinstance(buffers, Buffers):
        raise TypeError("dense Trellis buffer views require Buffers")
    compiling = torch.compiler.is_compiling()
    capacity = int(buffers.output.shape[0])
    if not compiling:
        size_m = int(size_m)
    if not compiling and (size_m <= 0 or size_m > capacity):
        raise ValueError(
            f"dense Trellis row count must be in [1, {capacity}], got {size_m}"
        )

    def rows(name: str, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if not compiling and (
            tensor.ndim != 2 or int(tensor.shape[0]) != capacity
        ):
            raise ValueError(
                f"dense Trellis {name} does not share the {capacity}-row capacity"
            )
        return tensor[:size_m]

    def required_rows(name: str, tensor: torch.Tensor) -> torch.Tensor:
        result = rows(name, tensor)
        assert result is not None
        return result

    if not compiling and size_m == capacity:
        return buffers
    return Buffers(
        output=required_rows("output", buffers.output),
        gemm_output=required_rows("gemm_output", buffers.gemm_output),
        c_tmp=buffers.c_tmp,
        input_f16=rows("input_f16", buffers.input_f16),
        rotated_f16=required_rows("rotated_f16", buffers.rotated_f16),
        rotated_compute=rows("rotated_compute", buffers.rotated_compute),
        gemm_output_f16=rows("gemm_output_f16", buffers.gemm_output_f16),
        output_f16=rows("output_f16", buffers.output_f16),
        low_rank_hidden=rows("low_rank_hidden", buffers.low_rank_hidden),
    )


def view_pair_buffers(buffers: PairBuffers, *, size_m: int) -> PairBuffers:
    """Return exact-row paired views backed by one preallocated capacity."""

    if not isinstance(buffers, PairBuffers):
        raise TypeError("paired Trellis buffer views require PairBuffers")
    compiling = torch.compiler.is_compiling()
    if not compiling and (
        buffers.output.ndim != 2 or buffers.low_rank_hidden.ndim != 3
    ):
        raise ValueError("paired Trellis capacity storage has incompatible ranks")
    capacity = int(buffers.output.shape[0])
    if not compiling:
        size_m = int(size_m)
    if not compiling and (
        size_m <= 0
        or size_m > capacity
        or tuple(buffers.low_rank_hidden.shape[:2]) != (2, capacity)
        or int(buffers.base.output.shape[0]) != capacity
    ):
        raise ValueError(
            f"paired Trellis row count must be in [1, {capacity}], got {size_m}"
        )
    if not compiling and size_m == capacity:
        return buffers
    rank = int(buffers.low_rank_hidden.shape[2])
    hidden = buffers.low_rank_hidden.view(-1, rank)[: 2 * size_m]
    return PairBuffers(
        base=view_buffers(buffers.base, size_m=size_m),
        output=buffers.output[:size_m],
        low_rank_hidden=hidden.view(2, size_m, rank),
    )


def prepare_additive_weight(
    base: PreparedWeight,
    a: torch.Tensor,
    b: torch.Tensor,
) -> PreparedAdditiveWeight:
    """Bind BF16 factors and create the native rank-major A layout."""
    if not isinstance(base, PreparedTrellis256DenseWeight):
        raise TypeError("additive Trellis preparation requires a prepared dense weight")
    rank = int(a.shape[1]) if isinstance(a, torch.Tensor) and a.ndim == 2 else 0
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.ndim != 2
        or b.ndim != 2
        or tuple(a.shape) != (int(base.in_features), rank)
        or tuple(b.shape) != (int(base.out_features), rank)
        or rank != 16
    ):
        raise ValueError(
            "low-rank factors must have shapes [in_features, 16] and [out_features, 16]"
        )
    if base.params_dtype != torch.bfloat16:
        raise TypeError("native additive Trellis execution requires BF16 compute")
    for name, factor in (("a", a), ("b", b)):
        if (
            factor.device != base.trellis.device
            or factor.dtype != base.params_dtype
            or not factor.is_contiguous()
            or int(factor.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"low-rank factor {name} must be contiguous, 16-byte-aligned "
                f"{base.params_dtype} storage on {base.trellis.device}"
            )
    a_t = a.T.contiguous()
    if int(a_t.data_ptr()) % 16 != 0:
        raise RuntimeError("rank-major low-rank factor storage is not aligned")
    return PreparedAdditiveWeight(base=base, a_t=a_t, b=b, rank=rank)


def prepare_additive_pair(
    left: PreparedWeight,
    right: PreparedWeight,
    a: torch.Tensor,
    b: torch.Tensor,
) -> PreparedAdditivePair:
    """Bind two equal-geometry packed bases to stacked gate/up factors."""
    if not isinstance(left, PreparedTrellis256DenseWeight) or not isinstance(
        right, PreparedTrellis256DenseWeight
    ):
        raise TypeError("paired additive preparation requires two dense weights")
    if (
        left.in_features != right.in_features
        or left.out_features != right.out_features
        or left.params_dtype != torch.bfloat16
        or right.params_dtype != torch.bfloat16
        or left.trellis.device != right.trellis.device
        or left.trellis_bits != right.trellis_bits
        or left.trellis_codebook != right.trellis_codebook
    ):
        raise ValueError("paired additive bases must share BF16 geometry and device")
    expected_a = (2, int(left.in_features), 16)
    expected_b = (2, int(left.out_features), 16)
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or tuple(a.shape) != expected_a
        or tuple(b.shape) != expected_b
    ):
        raise ValueError(
            f"paired factors must have shapes {expected_a} and {expected_b}"
        )
    for name, factor in (("a", a), ("b", b)):
        if (
            factor.device != left.trellis.device
            or factor.dtype != torch.bfloat16
            or not factor.is_contiguous()
            or int(factor.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"paired low-rank factor {name} must be contiguous, aligned BF16 "
                f"storage on {left.trellis.device}"
            )
    a_t = a.transpose(1, 2).contiguous()
    if int(a_t.data_ptr()) % 16 != 0:
        raise RuntimeError("paired rank-major factor storage is not aligned")
    return PreparedAdditivePair(
        left=left,
        right=right,
        a_t=a_t,
        b=b,
        rank=16,
    )


def make_buffers(
    weight: PreparedWeight | PreparedAdditiveWeight,
    *,
    size_m: int,
    input_dtype: torch.dtype,
) -> Buffers:
    """Allocate all storage needed by one fixed-shape eager or graph run."""
    if isinstance(weight, PreparedAdditiveWeight):
        base = weight.base
        rank: int | None = weight.rank
    elif isinstance(weight, PreparedTrellis256DenseWeight):
        base = weight
        rank = None
    else:
        raise TypeError("dense Trellis buffers require a prepared weight")
    size_m = int(size_m)
    if size_m <= 0:
        raise ValueError(f"dense Trellis M must be positive, got {size_m}")
    if input_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"dense Trellis input must be fp16 or bf16, got {input_dtype}")
    if rank is not None and input_dtype != torch.bfloat16:
        raise TypeError(
            "additive Trellis execution requires input and factor dtypes to match"
        )
    device = base.trellis.device
    size_k = int(base.in_features)
    size_n = int(base.out_features)
    compute_dtype = base.params_dtype

    def empty(shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)

    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    c_tmp_elements = max(
        packed_gemm_scratch_elements(
            size_n=size_n,
            route_slots=((size_m + block_size - 1) // block_size) * block_size,
            moe_block_size=block_size,
            sms=sms,
        )
        for block_size in _W4A16_ALLOWED_ROUTED_SIZES
    )

    return Buffers(
        output=empty((size_m, size_n), input_dtype),
        gemm_output=empty((size_m, size_n), compute_dtype),
        c_tmp=torch.empty(
            (c_tmp_elements,),
            dtype=torch.float32,
            device=device,
        ),
        input_f16=(
            empty((size_m, size_k), torch.float16)
            if input_dtype == torch.bfloat16
            else None
        ),
        rotated_f16=empty((size_m, size_k), torch.float16),
        rotated_compute=(
            empty((size_m, size_k), torch.bfloat16)
            if compute_dtype == torch.bfloat16
            else None
        ),
        gemm_output_f16=(
            empty((size_m, size_n), torch.float16)
            if compute_dtype == torch.bfloat16
            else None
        ),
        output_f16=(
            empty((size_m, size_n), torch.float16)
            if input_dtype == torch.bfloat16
            else None
        ),
        low_rank_hidden=(
            empty((size_m, rank), input_dtype) if rank is not None else None
        ),
    )


def make_pair_buffers(
    weight: PreparedAdditivePair,
    *,
    size_m: int,
    input_dtype: torch.dtype,
) -> PairBuffers:
    """Allocate stable storage for one paired gate/up execution."""
    if not isinstance(weight, PreparedAdditivePair):
        raise TypeError("paired Trellis buffers require a prepared additive pair")
    if input_dtype != torch.bfloat16:
        raise TypeError("paired additive Trellis execution requires BF16 input")
    base = make_buffers(weight.left, size_m=size_m, input_dtype=input_dtype)
    device = weight.left.trellis.device
    return PairBuffers(
        base=base,
        output=torch.empty(
            (int(size_m), 2 * int(weight.left.out_features)),
            dtype=input_dtype,
            device=device,
        ),
        low_rank_hidden=torch.empty(
            (2, int(size_m), weight.rank),
            dtype=input_dtype,
            device=device,
        ),
    )


def prepare_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Validate one native EXL3 dense weight and retain zero-copy views."""
    return prepare_trellis256_dense_weight(
        trellis,
        suh,
        svh,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
        params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )


def prepare_pair_weight(
    payload: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    pair_kind: str,
    rate_axis: str,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Prepare one compact TP12 P24/P33 pair for the SM12x decoder."""
    return prepare_trellis256_pair_dense_weight(
        payload,
        suh,
        svh,
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
        params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )


def run(
    x: torch.Tensor,
    weight: PreparedWeight,
    *,
    output: Optional[torch.Tensor] = None,
    gemm_output: Optional[torch.Tensor] = None,
    c_tmp: Optional[torch.Tensor] = None,
    input_f16: Optional[torch.Tensor] = None,
    rotated_f16: Optional[torch.Tensor] = None,
    rotated_compute: Optional[torch.Tensor] = None,
    gemm_output_f16: Optional[torch.Tensor] = None,
    output_f16: Optional[torch.Tensor] = None,
    hadamard_128=None,
    _moe_block_size: int = 64,
    _force_tile_config: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Execute Trellis GEMM, optionally reusing all capture-time storage."""
    return run_trellis256_dense(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        c_tmp=c_tmp,
        input_f16=input_f16,
        rotated_f16=rotated_f16,
        rotated_compute=rotated_compute,
        gemm_output_f16=gemm_output_f16,
        output_f16=output_f16,
        hadamard_128=hadamard_128,
        _moe_block_size=_moe_block_size,
        _force_tile_config=_force_tile_config,
    )


def run_additive(
    x: torch.Tensor,
    weight: PreparedAdditiveWeight,
    *,
    low_rank_hidden: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    gemm_output: Optional[torch.Tensor] = None,
    c_tmp: Optional[torch.Tensor] = None,
    input_f16: Optional[torch.Tensor] = None,
    rotated_f16: Optional[torch.Tensor] = None,
    rotated_compute: Optional[torch.Tensor] = None,
    gemm_output_f16: Optional[torch.Tensor] = None,
    output_f16: Optional[torch.Tensor] = None,
    hadamard_128=None,
    _moe_block_size: int = 64,
    _force_tile_config: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Execute packed Trellis GEMM and add a fixed low-rank branch.

    The operation is inference-only. Supplying the buffers returned by
    :func:`make_buffers` avoids allocation and pointer changes during replay.
    """
    if not isinstance(weight, PreparedAdditiveWeight):
        raise TypeError("run_additive requires a prepared additive weight")
    if x.dtype != weight.a_t.dtype:
        raise TypeError("additive Trellis input and factor dtypes must match")
    compiling = torch.compiler.is_compiling()
    rows = x.shape[0] if compiling else int(x.shape[0])
    expected_hidden = (rows, weight.rank) if x.ndim == 2 else ()
    if low_rank_hidden is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Trellis dense low_rank_hidden is not initialized for CUDA graph "
                "capture; provide caller-owned storage"
            )
        low_rank_hidden = torch.empty(
            expected_hidden,
            dtype=x.dtype,
            device=x.device,
        )
    elif (
        (not compiling and tuple(low_rank_hidden.shape) != expected_hidden)
        or low_rank_hidden.dtype != x.dtype
        or low_rank_hidden.device != x.device
        or not low_rank_hidden.is_contiguous()
        or not _has_16_byte_alignment(low_rank_hidden)
    ):
        raise ValueError(
            "low_rank_hidden must be contiguous, 16-byte-aligned storage with "
            f"shape {expected_hidden}, dtype {x.dtype}, and device {x.device}"
        )
    result = run(
        x,
        weight.base,
        output=output,
        gemm_output=gemm_output,
        c_tmp=c_tmp,
        input_f16=input_f16,
        rotated_f16=rotated_f16,
        rotated_compute=rotated_compute,
        gemm_output_f16=gemm_output_f16,
        output_f16=output_f16,
        hadamard_128=hadamard_128,
        _moe_block_size=_moe_block_size,
        _force_tile_config=_force_tile_config,
    )
    run_low_rank_additive(
        x,
        weight.a_t,
        weight.b,
        low_rank_hidden,
        result,
    )
    return result


def run_additive_pair(
    x: torch.Tensor,
    weight: PreparedAdditivePair,
    *,
    buffers: PairBuffers,
    hadamard_128=None,
    _moe_block_size: int = 64,
    _force_tile_config: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Execute two packed bases with a two-launch paired rank-16 branch."""
    if not isinstance(weight, PreparedAdditivePair):
        raise TypeError("run_additive_pair requires a prepared additive pair")
    if not isinstance(buffers, PairBuffers):
        raise TypeError("run_additive_pair requires caller-owned pair buffers")
    if x.ndim != 2:
        raise ValueError("paired additive input must be a matrix")
    compiling = torch.compiler.is_compiling()
    rows = x.shape[0] if compiling else int(x.shape[0])
    expected_output = (rows, 2 * int(weight.left.out_features))
    expected_hidden = (2, rows, weight.rank)
    if (
        x.dtype != torch.bfloat16
        or x.device != weight.left.trellis.device
        or not x.is_contiguous()
        or (not compiling and tuple(buffers.output.shape) != expected_output)
        or (not compiling and tuple(buffers.low_rank_hidden.shape) != expected_hidden)
    ):
        raise ValueError(
            "paired additive input or caller-owned storage is incompatible"
        )
    run_low_rank_pair_project(
        x,
        weight.a_t,
        weight.b,
        buffers.low_rank_hidden,
        buffers.output,
    )
    kwargs = buffers.base.run_kwargs()
    left = run(
        x,
        weight.left,
        hadamard_128=hadamard_128,
        _moe_block_size=_moe_block_size,
        _force_tile_config=_force_tile_config,
        **kwargs,
    )
    width = int(weight.left.out_features)
    buffers.output[:, :width].copy_(left)
    right = run(
        x,
        weight.right,
        hadamard_128=hadamard_128,
        _moe_block_size=_moe_block_size,
        _force_tile_config=_force_tile_config,
        **kwargs,
    )
    buffers.output[:, width:].copy_(right)
    run_low_rank_pair_add(
        x,
        weight.a_t,
        weight.b,
        buffers.low_rank_hidden,
        buffers.output,
    )
    return buffers.output


def is_supported(device=None) -> bool:
    """True when the SM120/SM121 Trellis kernel stack is available."""
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    """Clear compiled W4A16 and additive low-rank specializations."""
    clear_w4a16_kernel_cache()
    clear_low_rank_caches()


__all__ = list(META.entry_points)
