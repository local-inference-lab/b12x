"""Local NVFP4/BF16 weight preparation for the CuTeDSL W4A16 path."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from b12x._lib.quant.sqg_fp16_d3l import SQG_FP16_D3L

from b12x.moe._shared.kernels.w4a16.host import (
    W4A16PackedBuffers,
    make_w4a16_packed_buffers as _make_w4a16_packed_buffers,
    unswizzle_expert_scales,
    validate_activation,
    validate_w4a16_packed_inputs,
)


_PACKED_TILE_SIZE = 16
_PACKED_TILE_N_SIZE = 64
_PACK_FACTOR_4BIT = 8
_MODEL_OPT_W13_LAYOUTS = {"w13", "w31"}
_SOURCE_FORMATS = {
    "modelopt_nvfp4": "modelopt_nvfp4",
    "fp4_e8m0_k32": "fp4_e8m0_k32",
    "compressed_tensors": "compressed_tensors",
}
_E8M0_K32_BF16_MAX_SCALE_BYTE = 247
_E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT = 64
_QSRT_ATOM_CHANNELS = 32
_QSRT_ATOMS_PER_PAIR = 8
_QSRT_ATOMS_PER_EXPERT = 96
_QSRT_MATRIX_ATOM_TRELLIS_BYTES = 43_008
_QSRT_MATRIX_ATOM_SCALE_BYTES = 64
_QSRT_ATOM_TRELLIS_BYTES = 129_024
_QSRT_ATOM_BUNDLE_BYTES = 129_216
_QSRT_MATRIX_TRELLIS_OFFSETS = (0, 43_008, 86_016)
_QSRT_MATRIX_SCALE_OFFSETS = (129_024, 129_088, 129_152)
_QSRT_EXPERT_ROTATION_MULTIPLIER = 5
_QSRT_V2_P33_MATRIX_TRELLIS_BYTES = 43_008
_QSRT_V2_P43_MATRIX_TRELLIS_BYTES = 50_176
_QSRT_V2_P44_MATRIX_TRELLIS_BYTES = 57_344
_QSRT_V2_P33_ATOM_BUNDLE_BYTES = 129_216
_QSRT_V2_P43_ATOM_BUNDLE_BYTES = 150_720
_QSRT_V2_COUPLED_H308_P43_P33_ATOM_BUNDLE_BYTES = 143_552
_QSRT_V2_COUPLED_H308_P43_P44_ATOM_BUNDLE_BYTES = 157_888
_QSRT_V2_P22_MATRIX_TRELLIS_BYTES = 28_672
_QSRT_V2_P22_ATOM_BUNDLE_BYTES = 86_208
_QSRT_V2_PROFILE_H308 = "k3x22_k4x2"
_QSRT_V2_PROFILE_COUPLED_K2 = "k2_coupled_h512_h128"
_QSRT_V2_PROFILE_COUPLED_H308 = "k3x22_k4x2_coupled_h512_h128"
# Canonical W13 layout names are "w13"/"w31"; accept the physical FC1-half
# spellings as aliases. Logical checkpoint order "w13" arrives up/gate and
# needs a swap before the kernel's SwiGLU; "w31" is already kernel-native
# gate/up order.
_W13_LAYOUTS = {
    "w13": "w13",
    "w31": "w31",
    "up_gate": "w13",
    "gate_up": "w31",
}
_MODEL_OPT_NVFP4_FORMATS = {"modelopt_nvfp4"}
# Scale convention: weights decode to the FULL-precision
# bf16 codebook value, the K/32 scale byte encodes t_s * 2**-4 (e4m3-style,
# E4=0 arm), and the per-tensor global_scale carries the matching 2**116.


@dataclass(frozen=True)
class W4A16PackedWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    source_format: str = "modelopt_nvfp4"
    w13_layout: str = "w13"
    weight_layout: str = "packed"
    scale_format: str = "e4m3_k16"
    x4t_w13_scale: object | None = None
    x4t_w2_scale: object | None = None
    x4t_w13_row_rotation: int = 0


@dataclass(frozen=True)
class W4A16ModelOptWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    source_format: str = "modelopt_nvfp4"
    weight_layout: str = "modelopt"
    scale_format: str = "e4m3_k16"
    micro_w13_scale: torch.Tensor | None = None
    micro_w13_global_scale: torch.Tensor | None = None
    micro_w2_scale: torch.Tensor | None = None
    micro_w2_global_scale: torch.Tensor | None = None
    # Physical order of the two fused FC1 halves in source W13. "w13" (logical,
    # == "up_gate" physical) needs a row rotation before W4A16 SwiGLU; "w31"
    # (== "gate_up") is already in the kernel-native order.
    w13_layout: str = "w13"


@dataclass(frozen=True)
class W4A16FC2Weights:
    """Native MXFP4 down-projection weights for the FC2-only path."""

    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    params_dtype: torch.dtype
    source_format: str = "fp4_e8m0_k32"
    weight_layout: str = "modelopt"
    scale_format: str = "e8m0_k32"


@dataclass(frozen=True)
class PreparedW4A16MoeWeights:
    """Runtime native-codebook W4A16 expert weights.

    EXL3 Trellis keeps native codebook tiles and persistent full-rotation
    tables, sharing the W4A16 host ABI and retaining the tile configuration
    used at preparation.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    fc1_tile_n: int
    fc2_tile_n: int
    source_format: str = "qsrt_sqg_e4m3"
    w13_layout: str = "packed"
    weight_layout: str = "trellis3_t256"
    scale_format: str = "e4m3_k32"
    # Native tiles use MCG, K2/K3/K4 SQG-XOR-Cheb-T12, or K5/K6 FP16-D3L.
    trellis_codebook: str | None = None
    trellis_bits: int = 3
    # Optional compact fixed-payload pair specialization.  FC1's pair lies on
    # N (one pair in each projection); FC2's lies on K.  The two kinds may be
    # selected separately while every matrix remains exactly three bpw.
    fc1_trellis_pair_kind: str | None = None
    fc2_trellis_pair_kind: str | None = None
    # The fixed-size K3/K3 versus K2/K4 representation uses one int32 mode per
    # local expert: zero selects K3/K3 and one selects K2/K4. The K3/K3 versus
    # K4/K3 representation uses one int64 descriptor per local expert. Bit
    # zero selects K4/K3 and the remaining bits store the exact uint32 offset
    # into that matrix's compact payload pool.
    fc1_trellis_pair_modes: torch.Tensor | None = None
    fc2_trellis_pair_modes: torch.Tensor | None = None
    # Projection-specific EXL3 input incoherence scales.  They remain optional
    # for non-trellis and synthetic oracle preparation, but the coherent
    # projection-major runtime binds both so gate/up can stage distinct rotated
    # A operands without copying either table.
    gate_suh: torch.Tensor | None = None
    up_suh: torch.Tensor | None = None
    intermediate_rotations: torch.Tensor | None = None
    down_svh: torch.Tensor | None = None
    # Coupled transforms add an exact activation-boundary reparameterization
    # around the per-matrix incoherence transforms. The rotation rows contain
    # [gate I, up I, down I, preactivation 2I, postactivation I]. The 512-wide
    # residual transforms use a fixed all-positive draw and need no table.
    coupled_hadamard: bool = False
    tile_config: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class PreparedTrellis256DenseWeight:
    """Zero-copy native QSRT tensor plus its outer rotations.

    ``trellis`` is the flattened int32 view of one native
    ``[K/16,N/16,16*bits]i16`` payload. ``suh`` and ``svh`` retain the
    checkpoint tensors by reference.
    """

    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    workspace: torch.Tensor
    in_features: int
    out_features: int
    params_dtype: torch.dtype
    trellis_bits: int
    trellis_codebook: str
    mcg: torch.Tensor | None = None
    mul1_e4m3: torch.Tensor | None = None
    num_experts: int = 1
    weight_layout: str = "trellis3_t256"
    scale_format: str = "e4m3_k32"
    w13_layout: str = "packed"
    # Optional fixed-size 256-channel pair container. The payload still averages
    # three bits/weight, but its two 128-channel records are either K3/K3 or
    # K2/K4 and the rate axis is explicit.  The decoder maps both records to a
    # balanced LLHH compute schedule inside each N-axis CTA.  The epilogue
    # restores record0 || record1 order before the outer Hadamard, so this is
    # not a stored or semantic neuron permutation.  N-axis payloads are
    # swizzled once into K16-major pair spans; K-axis payloads already have the
    # required record-major storage order.
    trellis_pair_kind: str | None = None
    trellis_rate_axis: str | None = None


def _make_workspace(
    device: torch.device,
    *,
    max_blocks_per_sm: int = 4,
    min_elements: int = 0,
) -> torch.Tensor:
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    return torch.zeros(
        (max(sms * int(max_blocks_per_sm) + 2, int(min_elements)),),
        dtype=torch.int32,
        device=device,
    )


def _scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _e8m0_logical_tail_scale_n(size_n: int) -> int:
    return (
        (int(size_n) + _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT - 1)
        // _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT
    ) * _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT


def _permute_packed_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int,
    output_size_n: int | None = None,
) -> torch.Tensor:
    scale_perm, scale_perm_single = _scale_perms()
    if group_size < size_k and group_size != -1:
        block = len(scale_perm)
        if output_size_n is not None or int(size_n) % block != 0:
            padded_n = (
                ((int(size_n) + block - 1) // block) * block
                if output_size_n is None
                else int(output_size_n)
            )
            if padded_n < int(size_n) or padded_n % block != 0:
                raise ValueError(
                    f"output_size_n must be a multiple of {block} and >= size_n"
                )
            padded = scales.new_zeros((int(scales.shape[0]), padded_n))
            padded[:, : int(size_n)] = scales
            scales = padded.reshape((int(scales.shape[0]), -1, block))
            scales = scales[:, :, scale_perm].reshape((int(scales.shape[0]), padded_n))
            if output_size_n is None:
                scales = scales[:, : int(size_n)]
            return scales.contiguous()
        scales = scales.reshape((-1, block))[:, scale_perm]
    else:
        block = len(scale_perm_single)
        if output_size_n is not None or int(size_n) % block != 0:
            padded_n = (
                ((int(size_n) + block - 1) // block) * block
                if output_size_n is None
                else int(output_size_n)
            )
            if padded_n < int(size_n) or padded_n % block != 0:
                raise ValueError(
                    f"output_size_n must be a multiple of {block} and >= size_n"
                )
            padded = scales.new_zeros((int(scales.shape[0]), padded_n))
            padded[:, : int(size_n)] = scales
            scales = padded.reshape((int(scales.shape[0]), -1, block))
            scales = scales[:, :, scale_perm_single].reshape(
                (int(scales.shape[0]), padded_n)
            )
            if output_size_n is None:
                scales = scales[:, : int(size_n)]
            return scales.contiguous()
        scales = scales.reshape((-1, block))[:, scale_perm_single]
    return scales.reshape((-1, size_n)).contiguous()


def _nvfp4_compute_scale_factor(
    packed_scales: torch.Tensor,
    a_dtype: torch.dtype,
) -> float:
    if a_dtype == torch.float16:
        return 1.0
    max_scalar = 0.0
    for expert in range(int(packed_scales.shape[0])):
        ws_float = packed_scales[expert].float() * (2**7)
        nonzero_mask = ws_float > 0
        if bool(nonzero_mask.any().item()):
            max_scalar = max(max_scalar, float(ws_float[nonzero_mask].max().item()))
    if max_scalar > 0.0 and max_scalar < 448 * (2**7):
        return float(2 ** math.floor(math.log2((448 * (2**7)) / max_scalar)))
    return 1.0


def _process_nvfp4_packed_scales(
    packed_scales: torch.Tensor,
    *,
    scale_factor: float,
) -> torch.Tensor:
    packed_scales = packed_scales.to(torch.float16)
    packed_scales = packed_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        packed_scales.size(0),
        -1,
    )
    if scale_factor > 1.0:
        packed_scales = (packed_scales.float() * scale_factor).to(torch.float16)
    packed_scales = packed_scales * (2**7)
    packed_scales[packed_scales < 2] = 0
    packed_scales = packed_scales.view(torch.int16) << 1
    packed_scales = packed_scales.view(torch.float8_e4m3fn)
    return packed_scales[:, 1::2].contiguous()


def _process_nvfp4_packed_global_scale(
    global_scale: torch.Tensor,
    *,
    a_dtype: torch.dtype,
) -> torch.Tensor:
    if a_dtype == torch.float16:
        target_exponent = 5
    elif a_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise TypeError(f"unsupported W4A16 activation dtype {a_dtype}")
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def _process_nvfp4_micro_global_scale_from_packed(
    packed_global_scale: torch.Tensor,
    *,
    a_dtype: torch.dtype,
) -> torch.Tensor:
    if a_dtype == torch.float16:
        target_exponent = 5
    elif a_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise TypeError(f"unsupported W4A16 activation dtype {a_dtype}")
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return (
        (packed_global_scale * (2.0 ** (-exponent_bias))).to(torch.float32).contiguous()
    )


def _normalize_source_format(source_format: str) -> str:
    if source_format.lower() == "mxfp4_native":
        raise ValueError(
            "source_format='mxfp4_native' has been removed; use "
            "source_format='fp4_e8m0_k32' for E8M0 K/32 scales, or add "
            "a real MXFP4 source contract"
        )
    try:
        return _SOURCE_FORMATS[source_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "source_format must be one of 'modelopt_nvfp4', "
            "'fp4_e8m0_k32', or 'compressed_tensors', "
            f"got {source_format!r}"
        ) from exc


def _normalize_w13_layout(w13_layout: str) -> str:
    try:
        return _W13_LAYOUTS[w13_layout.lower()]
    except KeyError as exc:
        raise ValueError(
            "w13_layout must be one of 'w13'/'w31' (or the 'up_gate'/'gate_up' "
            f"aliases), got {w13_layout!r}"
        ) from exc


def _source_global_scale(
    global_scale: torch.Tensor, *, source_format: str
) -> torch.Tensor:
    if source_format == "compressed_tensors":
        return (1.0 / global_scale).to(torch.float32).contiguous()
    return global_scale.contiguous()


def _validate_e8m0_k32_scales(
    scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
    name: str,
    allow_k_tail: bool = False,
) -> torch.Tensor:
    """Validate source E8M0 K/32 scale tensor shape and dtype."""
    if scales.ndim != 3:
        raise ValueError(
            f"{name} must be [E, N, ceil(K/32)], got {tuple(scales.shape)}"
        )
    if allow_k_tail:
        if int(cols) % 8 != 0:
            raise ValueError(
                f"{name} compact E8M0 K-tail requires K divisible by 8, got {int(cols)}"
            )
        expected_cols = (int(cols) + 31) // 32
    elif int(cols) % 32 != 0:
        raise ValueError(f"{name} requires K divisible by 32, got {int(cols)}")
    else:
        expected_cols = int(cols) // 32
    if tuple(scales.shape[1:]) != (int(rows), expected_cols):
        raise ValueError(
            f"{name} must have shape [E, {int(rows)}, {expected_cols}] for "
            f"E8M0 K/32 scales, got {tuple(scales.shape)}"
        )
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scales.dtype == torch.uint8:
        return scales.view(e8m0_dtype) if e8m0_dtype is not None else scales
    if e8m0_dtype is not None and scales.dtype == e8m0_dtype:
        return scales
    raise TypeError(f"{name} must be torch.uint8 or torch.float8_e8m0fnu")


def _pack_e8m0_k32_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    reuse_input_storage: bool = False,
    allow_k_tail: bool = False,
    padded_size_n: int | None = None,
) -> torch.Tensor:
    if allow_k_tail:
        if int(size_k) % 8 != 0:
            raise ValueError(
                f"compact E8M0 K-tail requires K divisible by 8, got {size_k}"
            )
        scale_cols = (int(size_k) + 31) // 32
    elif int(size_k) % 32 != 0:
        raise ValueError(f"E8M0 K/32 scales require K divisible by 32, got {size_k}")
    else:
        scale_cols = int(size_k) // 32
    if tuple(scales.shape[1:]) != (int(size_n), scale_cols):
        raise ValueError(
            f"expected E8M0 scale shape [E, {int(size_n)}, {scale_cols}], "
            f"got {tuple(scales.shape)}"
        )
    output_size_n = int(size_n) if padded_size_n is None else int(padded_size_n)
    if output_size_n < int(size_n):
        raise ValueError(
            f"padded_size_n must be >= size_n, got {output_size_n} < {int(size_n)}"
        )
    source = scales.view(torch.uint8)
    if reuse_input_storage:
        if output_size_n != int(size_n):
            raise ValueError("reuse_input_storage requires unpadded E8M0 scales")
        if allow_k_tail:
            raise ValueError(
                "reuse_input_storage is not supported for compact E8M0 K-tail scales"
            )
        if not source.is_contiguous():
            raise ValueError("reuse_input_storage requires contiguous E8M0 scales")
        source.clamp_(max=_E8M0_K32_BF16_MAX_SCALE_BYTE)
        packed = source.reshape(
            int(source.shape[0]),
            scale_cols,
            output_size_n,
        )
    else:
        source = source.clamp(max=_E8M0_K32_BF16_MAX_SCALE_BYTE)
        packed = torch.empty(
            (int(source.shape[0]), scale_cols, output_size_n),
            dtype=torch.uint8,
            device=scales.device,
        )
    for expert in range(int(source.shape[0])):
        expert_source = source[expert]
        if row_rotation is not None:
            expert_source = torch.cat(
                [expert_source[row_rotation:], expert_source[:row_rotation]],
                dim=0,
            )
        expert_packed = _permute_packed_scales(
            expert_source.T.contiguous(),
            size_k=size_k,
            size_n=size_n,
            group_size=32,
            output_size_n=output_size_n,
        )
        expert_packed = (
            expert_packed.view(-1, 4)[:, [0, 2, 1, 3]]
            .reshape_as(expert_packed)
            .contiguous()
        )
        packed[expert].copy_(expert_packed)
    return packed.view(scales.dtype) if scales.dtype != torch.uint8 else packed


def _repack_4bit_no_perm(
    qweight_i32: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    out: torch.Tensor | None = None,
    flat_scratch: torch.Tensor | None = None,
    gather_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack 4-bit weights into the W4A16 A16 kernel layout."""
    if qweight_i32.dtype != torch.int32:
        raise TypeError("qweight_i32 must be torch.int32")
    if tuple(qweight_i32.shape) != (size_k // _PACK_FACTOR_4BIT, size_n):
        raise ValueError(
            f"expected qweight shape {(size_k // _PACK_FACTOR_4BIT, size_n)}, "
            f"got {tuple(qweight_i32.shape)}"
        )
    if size_k % _PACKED_TILE_SIZE != 0 or size_n % _PACKED_TILE_N_SIZE != 0:
        raise ValueError(
            f"W4A16 repack requires K,N multiples of 16,64; got {size_k},{size_n}"
        )

    k_tiles = size_k // _PACKED_TILE_SIZE
    n_tiles = size_n // _PACKED_TILE_N_SIZE
    packed_shape = (k_tiles, n_tiles, 128)
    if out is not None and (
        out.dtype != torch.int32 or tuple(out.shape) != packed_shape
    ):
        raise ValueError(
            f"out must be int32 with shape {packed_shape}, got "
            f"{out.dtype} {tuple(out.shape)}"
        )
    if flat_scratch is not None and (
        flat_scratch.dtype != torch.int32 or tuple(flat_scratch.shape) != packed_shape
    ):
        raise ValueError(
            f"flat_scratch must be int32 with shape {packed_shape}, got "
            f"{flat_scratch.dtype} {tuple(flat_scratch.shape)}"
        )
    if gather_scratch is not None and (
        gather_scratch.dtype != torch.int32
        or tuple(gather_scratch.shape) != packed_shape
    ):
        raise ValueError(
            f"gather_scratch must be int32 with shape {packed_shape}, got "
            f"{gather_scratch.dtype} {tuple(gather_scratch.shape)}"
        )

    tiles = qweight_i32.view(
        k_tiles,
        2,
        n_tiles,
        _PACKED_TILE_N_SIZE,
    )
    if flat_scratch is None:
        flat = tiles.permute(0, 2, 1, 3).reshape(
            k_tiles,
            n_tiles,
            2 * _PACKED_TILE_N_SIZE,
        )
    else:
        flat_scratch.view(k_tiles, n_tiles, 2, _PACKED_TILE_N_SIZE).copy_(
            tiles.permute(0, 2, 1, 3)
        )
        flat = flat_scratch

    device = qweight_i32.device
    out_pos = torch.arange(128, device=device, dtype=torch.long)
    th_id = out_pos // 4
    warp_id = out_pos % 4
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    offsets = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long)
    pack_idx = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device, dtype=torch.long)

    elem = tc_row[:, None] + offsets[None, :]
    row = elem // _PACK_FACTOR_4BIT
    pos = elem % _PACK_FACTOR_4BIT
    col1 = (warp_id * 16 + tc_col)[:, None].expand(-1, 4)
    col2 = col1 + 8
    source_index = torch.cat(
        [row * _PACKED_TILE_N_SIZE + col1, row * _PACKED_TILE_N_SIZE + col2],
        dim=1,
    )[:, pack_idx]
    source_shift = torch.cat([pos, pos], dim=1)[:, pack_idx] * 4

    result = (
        torch.empty(packed_shape, device=device, dtype=torch.int32)
        if out is None
        else out
    )
    result.zero_()
    for slot in range(8):
        gather_index = (
            source_index[:, slot]
            .view(1, 1, 128)
            .expand(
                k_tiles,
                n_tiles,
                128,
            )
        )
        shift = source_shift[:, slot].view(1, 1, 128)
        if gather_scratch is None:
            gathered = flat.gather(2, gather_index)
            nibble = (gathered >> shift) & 0xF
            result |= nibble << (slot * 4)
        else:
            torch.gather(flat, 2, gather_index, out=gather_scratch)
            torch.bitwise_right_shift(gather_scratch, shift, out=gather_scratch)
            torch.bitwise_and(gather_scratch, 0xF, out=gather_scratch)
            if slot:
                torch.bitwise_left_shift(
                    gather_scratch,
                    slot * 4,
                    out=gather_scratch,
                )
            torch.bitwise_or(result, gather_scratch, out=result)

    return result.reshape(k_tiles, n_tiles * 128).contiguous()


def _repack_weight(
    weight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    reuse_input_storage: bool = False,
) -> torch.Tensor:
    num_experts = int(weight.shape[0])
    if tuple(weight.shape[1:]) != (size_n, size_k // 2):
        raise ValueError(
            f"expected packed weight shape {(num_experts, size_n, size_k // 2)}, "
            f"got {tuple(weight.shape)}"
        )
    if size_k % _PACKED_TILE_SIZE != 0 or size_n % _PACKED_TILE_N_SIZE != 0:
        raise ValueError(
            f"W4A16 repack requires K,N multiples of 16,64; got {size_k},{size_n}"
        )

    packed_shape = (
        num_experts,
        size_k // _PACKED_TILE_SIZE,
        (size_n // _PACKED_TILE_N_SIZE) * 128,
    )
    if reuse_input_storage:
        if not weight.is_contiguous():
            raise ValueError("reuse_input_storage requires contiguous packed weights")
        packed = weight.view(torch.int32).reshape(packed_shape)
    else:
        packed = torch.empty(packed_shape, device=weight.device, dtype=torch.int32)

    k_tiles = size_k // _PACKED_TILE_SIZE
    n_tiles = size_n // _PACKED_TILE_N_SIZE
    qweight_scratch = torch.empty(
        (size_k // _PACK_FACTOR_4BIT, size_n),
        device=weight.device,
        dtype=torch.int32,
    )
    flat_scratch = torch.empty(
        (k_tiles, n_tiles, 128),
        device=weight.device,
        dtype=torch.int32,
    )
    gather_scratch = torch.empty_like(flat_scratch)

    for expert in range(num_experts):
        expert_weight = weight[expert].view(torch.int32)
        if row_rotation is not None:
            rotated_rows = int(size_n) - int(row_rotation)
            qweight_scratch[:, :rotated_rows].copy_(expert_weight[row_rotation:].T)
            qweight_scratch[:, rotated_rows:].copy_(expert_weight[:row_rotation].T)
        else:
            qweight_scratch.copy_(expert_weight.T)
        _repack_4bit_no_perm(
            qweight_scratch,
            size_k=size_k,
            size_n=size_n,
            out=packed[expert].view(k_tiles, n_tiles, 128),
            flat_scratch=flat_scratch,
            gather_scratch=gather_scratch,
        )
    return packed


def _permute_nvfp4_scales(
    scales: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    a_dtype: torch.dtype,
    row_rotation: int | None = None,
    output_size_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_scale_factor = _nvfp4_compute_scale_factor(scales, a_dtype)
    packed_scales: torch.Tensor | None = None
    for expert in range(scales.shape[0]):
        expert_source = scales[expert].to(a_dtype)
        if row_rotation is not None:
            expert_source = torch.cat(
                [expert_source[row_rotation:], expert_source[:row_rotation]],
                dim=0,
            )
        expert_scales = _permute_packed_scales(
            expert_source.T,
            size_k=size_k,
            size_n=size_n,
            group_size=16,
            output_size_n=output_size_n,
        )
        expert_packed = _process_nvfp4_packed_scales(
            expert_scales,
            scale_factor=combined_scale_factor,
        )
        if packed_scales is None:
            packed_scales = torch.empty(
                (int(scales.shape[0]), *expert_packed.shape),
                dtype=expert_packed.dtype,
                device=expert_packed.device,
            )
        packed_scales[expert].copy_(expert_packed)
    if packed_scales is None:
        packed_size_n = int(size_n) if output_size_n is None else int(output_size_n)
        packed_scales = torch.empty(
            (0, size_k // _PACKED_TILE_SIZE, packed_size_n // 2),
            dtype=torch.float8_e4m3fn,
            device=scales.device,
        )
    packed_global = _process_nvfp4_packed_global_scale(
        global_scales,
        a_dtype=a_dtype,
    ).to(torch.float32)
    packed_global = packed_global / combined_scale_factor
    return packed_scales, packed_global.contiguous()


def _prepare_w4a16_packed_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    source_format: str,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    source_format = _normalize_source_format(source_format)
    w13_layout = _normalize_w13_layout(w13_layout)
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated

    w13 = w13_fp4
    w13_scale = unswizzle_expert_scales(
        w13_blockscale,
        rows=w13_rows,
        cols=hidden_size,
    )
    w13_row_rotation = None
    if is_gated and w13_layout == "w13":
        # In-place: the half-swap is folded into the repack via row_rotation;
        # never materialize a second copy of the weights/scales.
        w13_row_rotation = intermediate_size

    w2_scale = unswizzle_expert_scales(
        w2_blockscale,
        rows=hidden_size,
        cols=intermediate_size,
    )

    packed_w13 = _repack_weight(
        w13 if reuse_input_storage else w13.contiguous(),
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w2 = _repack_weight(
        w2_fp4 if reuse_input_storage else w2_fp4.contiguous(),
        size_k=intermediate_size,
        size_n=hidden_size,
        reuse_input_storage=reuse_input_storage,
    )
    native_w13_global_scale = _source_global_scale(
        w13_global_scale,
        source_format=source_format,
    )
    native_w2_global_scale = _source_global_scale(
        w2_global_scale,
        source_format=source_format,
    )
    packed_w13_scale, packed_w13_global_scale = _permute_nvfp4_scales(
        w13_scale,
        native_w13_global_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        a_dtype=params_dtype,
        row_rotation=w13_row_rotation,
    )
    packed_w2_scale, packed_w2_global_scale = _permute_nvfp4_scales(
        w2_scale,
        native_w2_global_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        a_dtype=params_dtype,
    )

    return W4A16PackedWeights(
        w13=packed_w13,
        w13_scale=packed_w13_scale,
        w13_global_scale=packed_w13_global_scale,
        w2=packed_w2,
        w2_scale=packed_w2_scale,
        w2_global_scale=packed_w2_global_scale,
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format=source_format,
        w13_layout=w13_layout,
        scale_format="e4m3_k16",
    )


def prepare_w4a16_modelopt_nvfp4_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    """Prepare ModelOpt NVFP4 tensors into the W4A16 packed runtime layout.

    The per-block scales are the normal NVFP4 K/16 scale grid in b12x swizzled
    storage. The global scales are raw ModelOpt weight global scales; activation
    input scales are not folded into W4A16 weight preparation. For gated
    activations, ``w13_layout`` describes whether fused W13 rows arrive in
    checkpoint/logical W13 order or already swapped W31 order.
    """
    return _prepare_w4a16_packed_weights(
        w13_fp4,
        w13_blockscale,
        w13_global_scale,
        w2_fp4,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
        source_format="modelopt_nvfp4",
        w13_layout=w13_layout,
        reuse_input_storage=reuse_input_storage,
    )


def prepare_w4a16_modelopt_native_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
) -> W4A16ModelOptWeights:
    """Prepare W4A16 metadata while keeping ModelOpt FP4 weights native.

    This is the memory-safe path for GLM serving that needs A4 prefill and A16
    decode in the same process. It keeps the checkpoint FP4 tensors resident
    instead of materializing a second full W4A16 packed copy.
    """
    source_format = _normalize_source_format(source_format)
    if source_format not in _MODEL_OPT_NVFP4_FORMATS:
        raise ValueError(
            "native W4A16 ModelOpt weights require source_format 'modelopt_nvfp4'"
        )
    w13_layout = _normalize_w13_layout(w13_layout)

    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated

    w13_scale = unswizzle_expert_scales(
        w13_blockscale,
        rows=w13_rows,
        cols=hidden_size,
    )
    w2_scale = unswizzle_expert_scales(
        w2_blockscale,
        rows=hidden_size,
        cols=intermediate_size,
    )
    native_w13_global_scale = _source_global_scale(
        w13_global_scale,
        source_format=source_format,
    )
    native_w2_global_scale = _source_global_scale(
        w2_global_scale,
        source_format=source_format,
    )

    # The W4A16 activation consumes FC1 output in gate/up logical order.
    # Checkpoint-native ModelOpt GLM tensors are up/gate, while vLLM/FI can
    # hand over gate/up tensors after its own W13 reorder. Keep that physical
    # order explicit so source_format never implies a layout transformation.
    w13_row_rotation = intermediate_size if is_gated and w13_layout == "w13" else None
    packed_w13_scale, packed_w13_global_scale = _permute_nvfp4_scales(
        w13_scale,
        native_w13_global_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        a_dtype=params_dtype,
        row_rotation=w13_row_rotation,
    )
    packed_w2_scale, packed_w2_global_scale = _permute_nvfp4_scales(
        w2_scale,
        native_w2_global_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        a_dtype=params_dtype,
    )
    micro_w13_scale = packed_w13_scale
    micro_w13_global_scale = packed_w13_global_scale
    if w13_rows % _PACKED_TILE_N_SIZE != 0:
        # The direct micro reader uses the native 64-row scale permutation.
        # Keep the final tile intact: truncating it after permutation discards
        # logical rows whose permuted columns land above ``w13_rows``.
        micro_scale_n = (
            (w13_rows + _PACKED_TILE_N_SIZE - 1) // _PACKED_TILE_N_SIZE
        ) * _PACKED_TILE_N_SIZE
        micro_w13_scale, micro_w13_global_scale = _permute_nvfp4_scales(
            w13_scale,
            native_w13_global_scale,
            size_k=hidden_size,
            size_n=w13_rows,
            a_dtype=params_dtype,
            row_rotation=w13_row_rotation,
            output_size_n=micro_scale_n,
        )

    return W4A16ModelOptWeights(
        w13=w13_fp4,
        w13_scale=packed_w13_scale,
        w13_global_scale=packed_w13_global_scale,
        w2=w2_fp4,
        w2_scale=packed_w2_scale,
        w2_global_scale=packed_w2_global_scale,
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format=source_format,
        micro_w13_scale=micro_w13_scale,
        micro_w13_global_scale=_process_nvfp4_micro_global_scale_from_packed(
            micro_w13_global_scale,
            a_dtype=params_dtype,
        ),
        micro_w2_scale=packed_w2_scale,
        micro_w2_global_scale=_process_nvfp4_micro_global_scale_from_packed(
            packed_w2_global_scale,
            a_dtype=params_dtype,
        ),
        w13_layout=w13_layout,
    )


def prepare_w4a16_e8m0_native_weights(
    w13_fp4: torch.Tensor,
    w13_e8m0_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
) -> W4A16ModelOptWeights:
    """Prepare native MXFP4 (E8M0 K/32) weights for the W4A16 path.

    Keeps the FP4 weights resident as a single copy (``weight_layout="modelopt"``)
    and carries two small scale forms so one object serves both kernels:
    ``w13_scale``/``w2_scale`` are the packed E8M0 grid the main W4A16 GEMM reads
    at med/large M, and ``micro_*`` are packed E8M0 grids in the native row order
    that the small-M micro decode kernel reads. ``run_w4a16_moe`` routes small M
    to micro and the rest to the main W4A16 kernel automatically.
    """
    w13_layout = _normalize_w13_layout(w13_layout)
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated
    allow_w2_k_tail = intermediate_size % 32 != 0
    # Packed E8M0 scale rows are permuted in 64-row blocks independently of
    # whether W2 has a compact K tail.  Ungated 32-aligned intermediates such
    # as I=224 therefore still need W13's 224 scale rows padded to 256; tying
    # this padding to ``allow_w2_k_tail`` advertised the direct shape and then
    # rejected it during preparation.
    padded_w13_scale_n = _e8m0_logical_tail_scale_n(w13_rows)

    w13_scale = _validate_e8m0_k32_scales(
        w13_e8m0_scale,
        rows=w13_rows,
        cols=hidden_size,
        name="w13_e8m0_scale",
    )
    w2_scale = _validate_e8m0_k32_scales(
        w2_e8m0_scale,
        rows=hidden_size,
        cols=intermediate_size,
        name="w2_e8m0_scale",
        allow_k_tail=allow_w2_k_tail,
    )
    # Main-GEMM (med/large M) packed E8M0 scales. The "w13" (up_gate) layout
    # needs the FC1 half-swap folded into the scale grid; the kernel applies the
    # matching source_n_rotation to the native weights. micro reads the un-rotated
    # grid and handles the layout itself (w13_gate_first).
    w13_row_rotation = intermediate_size if (is_gated and w13_layout == "w13") else None
    packed_w13_scale = _pack_e8m0_k32_scales(
        w13_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        padded_size_n=padded_w13_scale_n,
    )
    micro_w13_scale = (
        packed_w13_scale
        if w13_row_rotation is None
        else _pack_e8m0_k32_scales(
            w13_scale,
            size_k=hidden_size,
            size_n=w13_rows,
            padded_size_n=padded_w13_scale_n,
        )
    )
    packed_w2_scale = _pack_e8m0_k32_scales(
        w2_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        allow_k_tail=allow_w2_k_tail,
    )
    # Storage-compatible single grid: micro reads the SAME packed _pack_e8m0_k32
    # scales the main GEMM reads (no separate K/16 micro grid).
    w13_global = w13_global_scale.contiguous()
    w2_global = w2_global_scale.contiguous()
    return W4A16ModelOptWeights(
        w13=w13_fp4,
        w13_scale=packed_w13_scale,
        w13_global_scale=w13_global,
        w2=w2_fp4,
        w2_scale=packed_w2_scale,
        w2_global_scale=w2_global,
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format="fp4_e8m0_k32",
        scale_format="e8m0_k32",
        micro_w13_scale=micro_w13_scale,
        micro_w13_global_scale=w13_global,
        micro_w2_scale=packed_w2_scale,
        micro_w2_global_scale=w2_global,
        w13_layout=w13_layout,
    )


def prepare_w4a16_fc2_e8m0_weights(
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    *,
    params_dtype: torch.dtype = torch.bfloat16,
) -> W4A16FC2Weights:
    """Prepare a native MXFP4 down projection without an FC1 placeholder.

    Args:
        w2_fp4: Packed E2M1 values with shape ``[E, H, I / 2]``.
        w2_e8m0_scale: Logical UE8M0 scale bytes with shape
            ``[E, H, I / 32]``.
        params_dtype: Activation and output dtype. The FC2-only kernel uses
            BF16, matching the W4A16 contract.

    Returns:
        Prepared source-native weights and caller-owned launch workspace.
    """

    if params_dtype != torch.bfloat16:
        raise TypeError("W4A16 FC2-only weights require torch.bfloat16")
    if w2_fp4.ndim != 3 or w2_fp4.dtype != torch.uint8:
        raise TypeError("w2_fp4 must be rank-3 torch.uint8")
    if not w2_fp4.is_cuda or not w2_fp4.is_contiguous():
        raise ValueError("w2_fp4 must be contiguous CUDA storage")
    num_experts, hidden_size, packed_intermediate = map(int, w2_fp4.shape)
    intermediate_size = 2 * packed_intermediate
    if num_experts <= 0 or hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("W4A16 FC2-only dimensions must be positive")
    if hidden_size % 128 != 0 or intermediate_size % 32 != 0:
        raise ValueError(
            "W4A16 FC2-only weights require H divisible by 128 and I by 32"
        )
    scale = _validate_e8m0_k32_scales(
        w2_e8m0_scale,
        rows=hidden_size,
        cols=intermediate_size,
        name="w2_e8m0_scale",
    )
    if scale.device != w2_fp4.device:
        raise ValueError("w2 scale and values must be on the same CUDA device")
    packed_scale = _pack_e8m0_k32_scales(
        scale,
        size_k=intermediate_size,
        size_n=hidden_size,
    )
    global_scale = torch.ones(
        (num_experts,), dtype=torch.float32, device=w2_fp4.device
    )
    return W4A16FC2Weights(
        w2=w2_fp4,
        w2_scale=packed_scale,
        w2_global_scale=global_scale,
        workspace=_make_workspace(w2_fp4.device, max_blocks_per_sm=1),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        params_dtype=params_dtype,
    )


def prepare_w4a16_compressed_tensors_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    """Prepare CompressedTensors NVFP4 tensors into the W4A16 packed runtime layout.

    The per-block scales are the normal NVFP4 K/16 scale grid in b12x swizzled
    storage. The CT global scales are stored inverted relative to the ModelOpt
    weight global scale convention, so they are inverted before packing.
    """
    return _prepare_w4a16_packed_weights(
        w13_fp4,
        w13_blockscale,
        w13_global_scale,
        w2_fp4,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
        source_format="compressed_tensors",
        w13_layout=w13_layout,
        reuse_input_storage=reuse_input_storage,
    )


def prepare_w4a16_fp4_e8m0_k32_weights(
    w13_fp4: torch.Tensor,
    w13_e8m0_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    """Prepare FP4 weights with E8M0 K/32 scales for W4A16.

    The per-block source scales are [E, N, K/32] E8M0 bytes. They are only
    saturated to the BF16 kernel's supported byte range and rearranged for
    kernel access; they are not expanded to K/16 or folded into global scales.
    """
    w13_layout = _normalize_w13_layout(w13_layout)
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated

    w13 = w13_fp4
    w13_scale = _validate_e8m0_k32_scales(
        w13_e8m0_scale,
        rows=w13_rows,
        cols=hidden_size,
        name="w13_e8m0_scale",
    )
    w13_row_rotation = None
    if is_gated and w13_layout != "w31":
        w13_row_rotation = intermediate_size

    w2_scale = _validate_e8m0_k32_scales(
        w2_e8m0_scale,
        rows=hidden_size,
        cols=intermediate_size,
        name="w2_e8m0_scale",
    )

    packed_w13 = _repack_weight(
        w13 if reuse_input_storage else w13.contiguous(),
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w2 = _repack_weight(
        w2_fp4 if reuse_input_storage else w2_fp4.contiguous(),
        size_k=intermediate_size,
        size_n=hidden_size,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w13_scale = _pack_e8m0_k32_scales(
        w13_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w2_scale = _pack_e8m0_k32_scales(
        w2_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        reuse_input_storage=reuse_input_storage,
    )

    return W4A16PackedWeights(
        w13=packed_w13,
        w13_scale=packed_w13_scale,
        w13_global_scale=w13_global_scale.contiguous(),
        w2=packed_w2,
        w2_scale=packed_w2_scale,
        w2_global_scale=w2_global_scale.contiguous(),
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format="fp4_e8m0_k32",
        w13_layout=w13_layout,
        scale_format="e8m0_k32",
    )


def prepare_w4a16_x4t_weights(
    w13_fp4: torch.Tensor,
    w13_x4t,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_x4t,
    w2_global_scale: torch.Tensor,
    w13_scale_scratch: torch.Tensor,
    w2_scale_scratch: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
) -> W4A16PackedWeights:
    """Prepare exact-nibble/X4T weights for routed predecode.

    The X4T payloads remain compressed and persistent. ``w13_scale_scratch``
    and ``w2_scale_scratch`` are caller-owned reusable scale grids; they may be
    shared by every MoE layer because each layer's decode and GEMM are ordered
    on one CUDA stream. Only experts routed by the current call are expanded.
    """

    from b12x._lib.quant.x4t_scales import X4TScaleBatch

    if not isinstance(w13_x4t, X4TScaleBatch) or not isinstance(
        w2_x4t, X4TScaleBatch
    ):
        raise TypeError("X4T preparation requires X4TScaleBatch scale planes")
    w13_x4t.validate()
    w2_x4t.validate()
    w13_layout = _normalize_w13_layout(w13_layout)
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    if (
        hidden_size != 3584
        or intermediate_size != 256
        or w13_rows != 512
        or not shape.is_gated
    ):
        raise ValueError(
            "initial X4T W4A16 execution supports only Kimi-K3's qualified "
            "local geometry (H=3584, I=256, gated FC1)"
        )
    if w13_x4t.num_experts != num_experts or w2_x4t.num_experts != num_experts:
        raise ValueError("X4T scale expert counts must match the weight tensors")
    if (w13_x4t.rows, w13_x4t.columns) != (w13_rows, hidden_size // 32):
        raise ValueError("X4T FC1 scale geometry does not match Kimi-K3")
    if (w2_x4t.rows, w2_x4t.columns) != (hidden_size, intermediate_size // 32):
        raise ValueError("X4T FC2 scale geometry does not match Kimi-K3")
    device = w13_fp4.device
    if w13_x4t.fixed.device != device or w2_x4t.fixed.device != device:
        raise ValueError("X4T scales and FP4 weights must share one CUDA device")

    def scale_scratch(
        name: str,
        tensor: torch.Tensor,
        tail_shape: tuple[int, int],
    ) -> torch.Tensor:
        expected_tail = tuple(map(int, tail_shape))
        if (
            tensor.dtype != torch.uint8
            or tensor.device != device
            or tensor.ndim != 3
            or tuple(tensor.shape[1:]) != expected_tail
            or int(tensor.shape[0]) < num_experts
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous uint8 [>=E,{expected_tail[0]},"
                f"{expected_tail[1]}] on {device}"
            )
        return tensor[:num_experts]

    packed_w13_scale = scale_scratch(
        "w13_scale_scratch", w13_scale_scratch, (hidden_size // 32, w13_rows)
    )
    packed_w2_scale = scale_scratch(
        "w2_scale_scratch", w2_scale_scratch, (intermediate_size // 32, hidden_size)
    )
    w13_row_rotation = intermediate_size if w13_layout == "w13" else 0
    packed_w13 = _repack_weight(
        w13_fp4.contiguous(),
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation or None,
    )
    packed_w2 = _repack_weight(
        w2_fp4.contiguous(),
        size_k=intermediate_size,
        size_n=hidden_size,
    )
    return W4A16PackedWeights(
        w13=packed_w13,
        w13_scale=packed_w13_scale,
        w13_global_scale=w13_global_scale.contiguous(),
        w2=packed_w2,
        w2_scale=packed_w2_scale,
        w2_global_scale=w2_global_scale.contiguous(),
        workspace=_make_workspace(device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=True,
        params_dtype=params_dtype,
        source_format="fp4_e8m0_k32",
        w13_layout=w13_layout,
        weight_layout="packed",
        scale_format="e8m0_k32",
        x4t_w13_scale=w13_x4t,
        x4t_w2_scale=w2_x4t,
        x4t_w13_row_rotation=w13_row_rotation,
    )


def prepare_w4a16_packed_weights(
    *args,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    **kwargs,
) -> W4A16PackedWeights:
    source_format = _normalize_source_format(source_format)
    w13_layout = _normalize_w13_layout(w13_layout)
    if source_format == "modelopt_nvfp4":
        return prepare_w4a16_modelopt_nvfp4_weights(
            *args, w13_layout=w13_layout, **kwargs
        )
    if source_format == "compressed_tensors":
        return prepare_w4a16_compressed_tensors_weights(
            *args, w13_layout=w13_layout, **kwargs
        )
    if source_format == "fp4_e8m0_k32":
        return prepare_w4a16_fp4_e8m0_k32_weights(
            *args, w13_layout=w13_layout, **kwargs
        )
    raise AssertionError(f"unhandled W4A16 source_format {source_format!r}")


def make_w4a16_packed_buffers(
    prepared: W4A16PackedWeights | W4A16ModelOptWeights,
    *,
    m: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    route_num_experts: int | None = None,
) -> W4A16PackedBuffers:
    return _make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=dtype,
        device=device,
        route_num_experts=route_num_experts,
    )


_TRELLIS256_W13_LAYOUTS = {"packed", "trellis3_t256_proj"}
_TRELLIS256_CODEBOOKS = {
    "mcg": "mcg",
    "sqg_xor_cheb_t12": "sqg_xor_cheb_t12",
    SQG_FP16_D3L: SQG_FP16_D3L,
}
_TRELLIS256_CODEBOOK_SENTINELS = {
    0xCBAC1FED: "mcg",
}


def _normalize_trellis256_codebook(codebook: str | int) -> str:
    if isinstance(codebook, int):
        normalized = _TRELLIS256_CODEBOOK_SENTINELS.get(
            int(codebook) & 0xFFFFFFFF
        )
        if normalized is None:
            raise ValueError(
                "unsupported trellis256 codebook sentinel "
                f"{int(codebook) & 0xFFFFFFFF:#010x}; expected MCG "
                "0xcbac1fed"
            )
        return normalized
    normalized = _TRELLIS256_CODEBOOKS.get(str(codebook).strip().lower())
    if normalized is None:
        raise ValueError(
            f"unsupported trellis256 codebook {codebook!r}; expected "
            "'mcg', 'sqg_xor_cheb_t12', or 'sqg_fp16_d3l'"
        )
    return normalized


def _validate_trellis256_codebook_bits(codebook: str, bits: int) -> None:
    if codebook == "sqg_xor_cheb_t12" and bits not in (2, 3, 4):
        raise ValueError("sqg_xor_cheb_t12 is defined only for K2/K3/K4")
    if codebook == SQG_FP16_D3L and bits not in (5, 6):
        raise ValueError("sqg_fp16_d3l is defined only for uniform K5/K6")


def _trellis256_random_native_tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate all 32-bit native tile words without changing their bit pattern."""
    raw = torch.randint(
        0,
        1 << 32,
        shape,
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    raw = torch.where(raw >= 2**31, raw - 2**32, raw)
    return raw.to(torch.int32).contiguous()


def _trellis256_bits_from_native_tensor(tensor: torch.Tensor, *, name: str) -> int:
    """Recover the EXL3 bitrate from the native tile's final dimension."""
    if tensor.ndim < 1:
        raise ValueError(f"trellis3_t256 {name} must have at least one dimension")
    words_per_bit = 16 if tensor.dtype == torch.int16 else 8
    if tensor.dtype not in (torch.int16, torch.int32):
        raise TypeError(
            f"trellis3_t256 {name} must use native int16 or int32 storage, "
            f"got {tensor.dtype}"
        )
    last = int(tensor.shape[-1])
    if last % words_per_bit != 0:
        raise ValueError(
            f"trellis3_t256 {name} final dimension {last} is not a native "
            f"{tensor.dtype} tile width"
        )
    bits = last // words_per_bit
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError(
            f"trellis3_t256 {name} encodes unsupported {bits}-bpw storage; "
            "expected 2, 3, 4, 5, or 6"
        )
    return bits


def _trellis256_flat_native_view(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_prefix_shape: tuple[int, ...],
    trellis_bits: int,
    device: torch.device,
) -> torch.Tensor:
    trellis_bits = int(trellis_bits)
    expected_i16_shape = (*expected_prefix_shape, 16 * trellis_bits)
    expected_i32_shape = (*expected_prefix_shape, 8 * trellis_bits)
    if tensor.device != device:
        raise ValueError(
            f"trellis3_t256 {name} must be on {device}, got {tensor.device}"
        )
    if tensor.dtype == torch.int16:
        expected_shape = expected_i16_shape
    elif tensor.dtype == torch.int32:
        expected_shape = expected_i32_shape
    else:
        raise TypeError(
            f"trellis3_t256 {name} must be native int16 tiles ({16 * trellis_bits} words) "
            f"or the identical int32 view ({8 * trellis_bits} words), got {tensor.dtype}"
        )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"trellis3_t256 {name} requires native {trellis_bits}-bit EXL3 shape "
            f"{expected_shape} for dtype {tensor.dtype}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"trellis3_t256 {name} must be contiguous")
    if int(tensor.data_ptr()) % 16 != 0:
        raise ValueError(
            f"trellis3_t256 {name} must be at least 16-byte aligned for cp.async"
        )
    return tensor.view(torch.int32).reshape(-1)


def prepare_trellis256_moe_weights(
    w13: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    fc1_tile_n: int,
    fc2_tile_n: int,
    device: torch.device | str | int | None = None,
    seed: int = 0,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "packed",
    trellis_bits: int | None = None,
    dummy_scale: torch.Tensor | None = None,
    codebook: str | int = "mcg",
    gate_suh: torch.Tensor | None = None,
    up_suh: torch.Tensor | None = None,
    intermediate_rotations: torch.Tensor | None = None,
    down_svh: torch.Tensor | None = None,
    tile_config: tuple[int, int, int, int] | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    """Wrap or synthesize native EXL3 tiles for ``trellis3_t256``.

    Supplying both ``w13`` and ``w2`` is the production path: no bytes are
    copied or permuted; each tensor is only viewed as contiguous int32 words and
    flattened.  Omitting both tensors is the deterministic full-GEMM-oracle
    path selected by ``device`` and ``seed``.  ``gate_suh`` and ``up_suh`` are
    optional zero-copy bindings for the two projection-specific EXL3 input
    rotations; when supplied, both must be contiguous fp16 ``[E,H]`` tensors on
    the weight device.

    Plain FC1 storage is expert-major
    ``[E,H/16,FC1_N/16,16*bits]i16``. ``trellis3_t256_proj`` instead requires one
    projection-major backing ``[2,E,H/16,I/16,16*bits]i16`` so gate/up fallback views
    can continue to alias the same live storage.  FC2 is always the plain native
    ``[E,I/16,H/16,16*bits]i16`` layout.
    """
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    num_experts = int(num_experts)
    fc1_tile_n = int(fc1_tile_n)
    fc2_tile_n = int(fc2_tile_n)
    if params_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("trellis3_t256 W4A16 weights require fp16 or bf16 activations")
    requested_trellis_bits = None if trellis_bits is None else int(trellis_bits)
    if requested_trellis_bits is not None and requested_trellis_bits not in (
        2,
        3,
        4,
        5,
        6,
    ):
        raise ValueError(
            "trellis3_t256 bits must be one of 2, 3, 4, 5, or 6, "
            f"got {requested_trellis_bits}"
        )
    if num_experts <= 0:
        raise ValueError(f"trellis3_t256 requires num_experts > 0, got {num_experts}")
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError(
            "trellis3_t256 requires positive hidden_size and intermediate_size, "
            f"got H={hidden_size} I={intermediate_size}"
        )
    if hidden_size % 16 != 0 or intermediate_size % 16 != 0:
        raise ValueError(
            "native EXL3 tiles require hidden_size and intermediate_size to be "
            f"multiples of 16, got H={hidden_size} I={intermediate_size}"
        )
    if hidden_size % 32 != 0 or intermediate_size % 32 != 0:
        raise ValueError(
            "trellis3_t256 uses E4M3 K/32 kernel plumbing and therefore requires "
            "hidden_size and intermediate_size to be multiples of 32; "
            f"got H={hidden_size} I={intermediate_size}"
        )
    for name, tile_n in (("fc1_tile_n", fc1_tile_n), ("fc2_tile_n", fc2_tile_n)):
        if tile_n < 64 or tile_n % 16 != 0:
            raise ValueError(
                f"trellis3_t256 {name} must be a multiple of 16 and at least "
                f"64 for the current W4A16 kernel, got {tile_n}"
            )

    is_gated = validate_activation(activation)
    w13_rows = intermediate_size * (2 if is_gated else 1)
    if w13_layout not in _TRELLIS256_W13_LAYOUTS:
        raise ValueError(
            f"unsupported trellis3_t256 w13_layout {w13_layout!r}; expected "
            "'packed' or 'trellis3_t256_proj'"
        )
    if w13_layout == "trellis3_t256_proj":
        if not is_gated:
            raise ValueError(
                "trellis3_t256_proj requires a gated activation with separate "
                "gate/up FC1 projections"
            )
        if intermediate_size % fc1_tile_n != 0:
            raise ValueError(
                "trellis3_t256_proj requires each FC1 projection to contain an "
                f"integral number of CTA N tiles: I={intermediate_size}, "
                f"fc1_tile_n={fc1_tile_n}"
            )
    elif w13_rows % fc1_tile_n != 0:
        raise ValueError(
            "trellis3_t256 has no FC1 logical-tail path: "
            f"FC1_N={w13_rows} must be divisible by fc1_tile_n={fc1_tile_n}"
        )
    if hidden_size % fc2_tile_n != 0:
        raise ValueError(
            "trellis3_t256 has no FC2 logical-tail path: "
            f"FC2_N={hidden_size} must be divisible by fc2_tile_n={fc2_tile_n}"
        )
    normalized_codebook = _normalize_trellis256_codebook(codebook)

    have_w13 = w13 is not None
    have_w2 = w2 is not None
    if have_w13 != have_w2:
        raise ValueError("trellis3_t256 requires both w13 and w2, or neither")
    synthetic = not have_w13
    if synthetic:
        resolved_trellis_bits = requested_trellis_bits or 3
        if device is None:
            raise ValueError(
                "device is required when synthesizing trellis3_t256 oracle weights"
            )
        if isinstance(device, int):
            resolved_device = torch.device("cuda", int(device))
        else:
            resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError(
                "trellis3_t256 W4A16 weights require a CUDA device, got "
                f"{resolved_device}"
            )
        if resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(int(seed))
        if w13_layout == "trellis3_t256_proj":
            w13_i32_shape = (
                2,
                num_experts,
                hidden_size // 16,
                intermediate_size // 16,
                8 * resolved_trellis_bits,
            )
        else:
            w13_i32_shape = (
                num_experts,
                hidden_size // 16,
                w13_rows // 16,
                8 * resolved_trellis_bits,
            )
        w2_i32_shape = (
            num_experts,
            intermediate_size // 16,
            hidden_size // 16,
            8 * resolved_trellis_bits,
        )
        packed_w13 = _trellis256_random_native_tensor(
            w13_i32_shape, device=resolved_device, generator=generator
        ).reshape(-1)
        packed_w2 = _trellis256_random_native_tensor(
            w2_i32_shape, device=resolved_device, generator=generator
        ).reshape(-1)
    else:
        assert w13 is not None and w2 is not None
        w13_bits = _trellis256_bits_from_native_tensor(w13, name="w13")
        w2_bits = _trellis256_bits_from_native_tensor(w2, name="w2")
        if w13_bits != w2_bits:
            raise ValueError(
                f"trellis3_t256 w13/w2 bitrate mismatch: {w13_bits} vs {w2_bits}"
            )
        resolved_trellis_bits = w13_bits
        if resolved_trellis_bits not in (2, 3, 4, 5, 6):
            raise ValueError(
                "trellis3_t256 tensors must encode 2, 3, 4, 5, or "
                f"6 bpw, got {resolved_trellis_bits}"
            )
        if (
            requested_trellis_bits is not None
            and requested_trellis_bits != resolved_trellis_bits
        ):
            raise ValueError(
                "explicit trellis_bits disagrees with native tensor shape: "
                f"requested={requested_trellis_bits}, inferred={resolved_trellis_bits}"
            )
        resolved_device = w13.device
        if resolved_device.type != "cuda":
            raise ValueError(
                "trellis3_t256 W4A16 weights require CUDA storage, got "
                f"{resolved_device}"
            )
        if device is not None:
            requested_device = (
                torch.device("cuda", int(device))
                if isinstance(device, int)
                else torch.device(device)
            )
            device_matches = requested_device.type == resolved_device.type and (
                requested_device.index is None
                or requested_device.index == resolved_device.index
            )
            if not device_matches:
                raise ValueError(
                    "explicit trellis3_t256 device does not match supplied weights: "
                    f"device={requested_device}, weights={resolved_device}"
                )
        if w13_layout == "trellis3_t256_proj":
            expected_w13_prefix = (
                2,
                num_experts,
                hidden_size // 16,
                intermediate_size // 16,
            )
        else:
            expected_w13_prefix = (
                num_experts,
                hidden_size // 16,
                w13_rows // 16,
            )
        expected_w2_prefix = (
            num_experts,
            intermediate_size // 16,
            hidden_size // 16,
        )
        packed_w13 = _trellis256_flat_native_view(
            w13,
            name="w13",
            expected_prefix_shape=expected_w13_prefix,
            trellis_bits=resolved_trellis_bits,
            device=resolved_device,
        )
        packed_w2 = _trellis256_flat_native_view(
            w2,
            name="w2",
            expected_prefix_shape=expected_w2_prefix,
            trellis_bits=resolved_trellis_bits,
            device=resolved_device,
        )

    have_gate_suh = gate_suh is not None
    have_up_suh = up_suh is not None
    if have_gate_suh != have_up_suh:
        raise ValueError(
            "trellis3_t256 projection input scales require both gate_suh and up_suh"
        )
    if have_gate_suh:
        if w13_layout != "trellis3_t256_proj":
            raise ValueError(
                "trellis3_t256 gate_suh/up_suh bindings require "
                "w13_layout='trellis3_t256_proj'"
            )
        assert gate_suh is not None and up_suh is not None
        for name, scale in (("gate_suh", gate_suh), ("up_suh", up_suh)):
            if scale.device != resolved_device:
                raise ValueError(
                    f"trellis3_t256 {name} must be on {resolved_device}, got {scale.device}"
                )
            if scale.dtype != torch.float16:
                raise TypeError(
                    f"trellis3_t256 {name} must be torch.float16, got {scale.dtype}"
                )
            # (1, hidden_size) is a broadcast row shared by all experts
            # (kquant shared-su artifacts); kernels index it with expert
            # stride 0.
            if tuple(scale.shape) not in (
                (num_experts, hidden_size),
                (1, hidden_size),
            ):
                raise ValueError(
                    f"trellis3_t256 {name} must have shape "
                    f"{(num_experts, hidden_size)} or {(1, hidden_size)}, "
                    f"got {tuple(scale.shape)}"
                )
            if not scale.is_contiguous():
                raise ValueError(f"trellis3_t256 {name} must be contiguous")
        if (gate_suh.shape[0] == 1) != (up_suh.shape[0] == 1):
            raise ValueError(
                "trellis3_t256 gate_suh and up_suh must both be per-expert "
                "or both broadcast"
            )

    have_full_rotation = any(
        value is not None
        for value in (gate_suh, up_suh, intermediate_rotations, down_svh)
    )
    if have_full_rotation and not all(
        value is not None
        for value in (gate_suh, up_suh, intermediate_rotations, down_svh)
    ):
        raise ValueError(
            "full-rotation trellis3_t256 requires gate_suh, up_suh, "
            "intermediate_rotations, and down_svh"
        )
    if have_full_rotation:
        assert intermediate_rotations is not None and down_svh is not None
        # Projection order is part of the execution ABI: gate/up are FC1
        # output scales and down is the FC2 input scale.  In particular,
        # SiTU's nonlinear up branch makes the last two blocks noncommutative.
        # intermediate_rotations = [gate_svh, up_svh, down_suh].
        for name, scale, shapes in (
            (
                "intermediate_rotations",
                intermediate_rotations,
                ((num_experts, 3 * intermediate_size),),
            ),
            (
                "down_svh",
                down_svh,
                ((num_experts, hidden_size), (1, hidden_size)),
            ),
        ):
            if scale.device != resolved_device:
                raise ValueError(
                    f"trellis3_t256 {name} must be on {resolved_device}, got {scale.device}"
                )
            if scale.dtype != torch.float16:
                raise TypeError(
                    f"trellis3_t256 {name} must be torch.float16, got {scale.dtype}"
                )
            if tuple(scale.shape) not in shapes:
                raise ValueError(
                    f"trellis3_t256 {name} must have shape {shapes}, "
                    f"got {tuple(scale.shape)}"
                )
            if not scale.is_contiguous():
                raise ValueError(f"trellis3_t256 {name} must be contiguous")

    if tile_config is not None:
        tile_config = tuple(int(value) for value in tile_config)
        if len(tile_config) != 4 or any(
            value <= 0 or value % 16 != 0 for value in tile_config
        ):
            raise ValueError(
                "trellis3_t256 tile_config must contain four positive multiples of 16"
            )
        fc1_tile_k, configured_fc1_n, fc2_tile_k, configured_fc2_n = tile_config
        if configured_fc1_n != fc1_tile_n or configured_fc2_n != fc2_tile_n:
            raise ValueError(
                "trellis3_t256 tile_config N dimensions disagree with preparation: "
                f"tile_config={tile_config}, fc1_tile_n={fc1_tile_n}, "
                f"fc2_tile_n={fc2_tile_n}"
            )
        if hidden_size % fc1_tile_k != 0 or intermediate_size % fc2_tile_k != 0:
            raise ValueError(
                "trellis3_t256 tile_config K dimensions must divide the model geometry"
            )

    _validate_trellis256_codebook_bits(
        normalized_codebook, resolved_trellis_bits
    )
    if dummy_scale is None:
        dummy_scale = torch.zeros(4, dtype=torch.uint8, device=resolved_device)
    else:
        if dummy_scale.device != resolved_device:
            raise ValueError(
                "trellis3_t256 dummy_scale must share the weight device, got "
                f"{dummy_scale.device} and {resolved_device}"
            )
        if dummy_scale.dtype != torch.uint8:
            raise TypeError(
                f"trellis3_t256 dummy_scale must be torch.uint8, got {dummy_scale.dtype}"
            )
        if not dummy_scale.is_contiguous() or tuple(dummy_scale.shape) != (4,):
            raise ValueError(
                "trellis3_t256 dummy_scale must be a contiguous four-byte "
                f"tensor with shape (4,), got {tuple(dummy_scale.shape)}"
            )
        if int(dummy_scale.data_ptr()) % 16 != 0:
            raise ValueError(
                "trellis3_t256 dummy_scale must be at least 16-byte aligned"
            )

    global_scale = torch.ones(
        (num_experts,), dtype=torch.float32, device=resolved_device
    )
    if workspace is None:
        workspace = _make_workspace(resolved_device, max_blocks_per_sm=4)
    elif workspace.device != resolved_device or workspace.dtype != torch.int32:
        raise ValueError("trellis3_t256 workspace must be int32 on the weight device")
    return PreparedW4A16MoeWeights(
        w13=packed_w13,
        w13_scale=dummy_scale,
        w13_global_scale=global_scale,
        w2=packed_w2,
        w2_scale=dummy_scale,
        w2_global_scale=global_scale,
        workspace=workspace,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_n=fc2_tile_n,
        source_format=(
            "exl3_trellis_mcg"
            if normalized_codebook == "mcg"
            else "qsrt_sqg_e4m3"
            if normalized_codebook == "sqg_xor_cheb_t12"
            else SQG_FP16_D3L
        ),
        w13_layout=w13_layout,
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        trellis_codebook=normalized_codebook,
        trellis_bits=resolved_trellis_bits,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )


def _trellis256_marker_codebook(
    *,
    mcg: torch.Tensor | None,
    mul1_e4m3: torch.Tensor | None,
    codebook: str | None,
) -> str:
    if mul1_e4m3 is not None:
        raise ValueError(
            "the MUL1 runtime marker is not supported; use legacy EXL3 MCG or "
            "the QSRT SQG-XOR-Cheb-T12 codebook"
        )
    marker_codebook: str | None = None
    if mcg is not None:
        if not isinstance(mcg, torch.Tensor):
            marker_value = int(mcg) & 0xFFFFFFFF
        else:
            if mcg.numel() != 1 or mcg.dtype not in (torch.int32, torch.uint32):
                raise ValueError(
                    "trellis256 mcg marker must be a scalar int32/uint32 tensor"
                )
            marker_value = int(mcg.item()) & 0xFFFFFFFF
        marker_codebook = _normalize_trellis256_codebook(marker_value)
    explicit = None if codebook is None else _normalize_trellis256_codebook(codebook)
    if (
        marker_codebook is not None
        and explicit is not None
        and marker_codebook != explicit
    ):
        raise ValueError(
            "trellis256 codebook metadata disagrees: "
            f"marker={marker_codebook!r}, codebook={explicit!r}"
        )
    normalized = marker_codebook or explicit
    if normalized is None:
        raise ValueError(
            "trellis256 dense preparation requires mcg= or explicit codebook="
        )
    return normalized


def prepare_trellis256_dense_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    mcg: torch.Tensor | None = None,
    mul1_e4m3: torch.Tensor | None = None,
    codebook: str | None = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: torch.Tensor | None = None,
) -> PreparedTrellis256DenseWeight:
    """Prepare one native EXL3 linear for the dense trellis256 entry point.

    The native payload is ``[K/16,N/16,16*bits]i16`` (or the byte-identical
    ``[...,8*bits]i32`` view), optionally with a leading singleton expert axis.
    The bitrate is inferred from that final dimension. No trellis or rotation
    bytes are copied, permuted, stacked, or concatenated.
    """
    if params_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("trellis3_t256 dense compute requires fp16 or bf16 MMA inputs")
    if trellis.ndim not in (3, 4):
        raise ValueError(
            "trellis3_t256 dense payload must have rank 3 or a leading E=1 axis"
        )
    if trellis.ndim == 4:
        if int(trellis.shape[0]) != 1:
            raise ValueError(
                f"trellis3_t256 dense payload requires E=1, got {int(trellis.shape[0])}"
            )
        k16, n16 = int(trellis.shape[1]), int(trellis.shape[2])
    else:
        k16, n16 = int(trellis.shape[0]), int(trellis.shape[1])
    trellis_bits = _trellis256_bits_from_native_tensor(trellis, name="dense trellis")
    in_features = k16 * 16
    out_features = n16 * 16
    if in_features <= 0 or out_features <= 0:
        raise ValueError(
            f"trellis3_t256 dense dimensions must be positive, got {in_features}x{out_features}"
        )
    if in_features % 128 != 0 or out_features % 128 != 0:
        raise ValueError(
            "trellis3_t256 dense rotations require K and N divisible by 128; "
            f"got K={in_features} N={out_features}"
        )
    expected_prefix_shape = (1, k16, n16) if trellis.ndim == 4 else (k16, n16)
    device = trellis.device
    if device.type != "cuda":
        raise ValueError(
            f"trellis3_t256 dense weights require CUDA storage, got {device}"
        )
    packed = _trellis256_flat_native_view(
        trellis,
        name="dense trellis",
        expected_prefix_shape=expected_prefix_shape,
        trellis_bits=trellis_bits,
        device=device,
    )
    for name, scale, width in (
        ("suh", suh, in_features),
        ("svh", svh, out_features),
    ):
        if scale.device != device:
            raise ValueError(f"trellis3_t256 dense {name} must be on {device}")
        if scale.dtype != torch.float16:
            raise TypeError(
                f"trellis3_t256 dense {name} must be torch.float16, got {scale.dtype}"
            )
        if tuple(scale.shape) != (width,):
            raise ValueError(
                f"trellis3_t256 dense {name} must have shape {(width,)}, "
                f"got {tuple(scale.shape)}"
            )
        if not scale.is_contiguous():
            raise ValueError(f"trellis3_t256 dense {name} must be contiguous")
    normalized_codebook = _trellis256_marker_codebook(
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
    )
    _validate_trellis256_codebook_bits(normalized_codebook, trellis_bits)
    if dummy_scale is None:
        dummy_scale = torch.zeros(4, dtype=torch.uint8, device=device)
    else:
        if (
            dummy_scale.device != device
            or dummy_scale.dtype != torch.uint8
            or tuple(dummy_scale.shape) != (4,)
            or not dummy_scale.is_contiguous()
            or int(dummy_scale.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                "trellis3_t256 dense dummy_scale must be a contiguous, 16-byte-"
                "aligned four-byte uint8 tensor on the weight device"
            )
    return PreparedTrellis256DenseWeight(
        trellis=packed,
        suh=suh,
        svh=svh,
        scale=dummy_scale,
        global_scale=torch.ones(1, dtype=torch.float32, device=device),
        workspace=_make_workspace(
            device,
            max_blocks_per_sm=4,
            min_elements=out_features // 16,
        ),
        in_features=in_features,
        out_features=out_features,
        params_dtype=params_dtype,
        trellis_bits=trellis_bits,
        trellis_codebook=normalized_codebook,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
    )


def prepare_trellis256_pair_dense_weight(
    payload: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    pair_kind: str,
    rate_axis: str,
    mcg: torch.Tensor | int | None = None,
    mul1_e4m3: torch.Tensor | int | None = None,
    codebook: str | None = "sqg_xor_cheb_t12",
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: torch.Tensor | None = None,
) -> PreparedTrellis256DenseWeight:
    """Prepare one compact P24/P33 pair for the dense SM12x decoder.

    ``payload`` uses the reference container order ``record0 || record1`` and
    contains exactly 256 channels on ``rate_axis``.  Preparation preserves the
    compact three-bpw byte count and record0 || record1 channel semantics.  For
    an N-axis pair it performs the one-time record-to-K16-major swizzle needed
    for contiguous GEMM staging; the kernel balances heterogeneous decode work
    internally and restores record order before its output Hadamard.  A K-axis
    pair is already in the required order and remains zero-copy.
    """

    pair_kind = str(pair_kind).upper()
    if pair_kind not in {"P24", "P33"}:
        raise ValueError(f"trellis pair_kind must be P24 or P33, got {pair_kind!r}")
    rate_axis = str(rate_axis).lower()
    if rate_axis not in {"k", "n"}:
        raise ValueError(f"trellis pair rate_axis must be 'k' or 'n', got {rate_axis!r}")
    if params_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("trellis3_t256 pair compute requires fp16 or bf16 MMA inputs")
    if payload.dtype != torch.int16:
        raise TypeError(f"trellis pair payload must use torch.int16, got {payload.dtype}")
    if payload.ndim != 1 or not payload.is_contiguous():
        raise ValueError("trellis pair payload must be a contiguous one-dimensional tensor")
    if payload.device.type != "cuda":
        raise ValueError(f"trellis pair payload requires CUDA storage, got {payload.device}")
    device = payload.device
    for name, scale in (("suh", suh), ("svh", svh)):
        if scale.device != device:
            raise ValueError(f"trellis pair {name} must be on {device}")
        if scale.dtype != torch.float16:
            raise TypeError(f"trellis pair {name} must be torch.float16, got {scale.dtype}")
        if scale.ndim != 1 or not scale.is_contiguous():
            raise ValueError(f"trellis pair {name} must be a contiguous vector")
        if not bool(torch.all(torch.isfinite(scale))):
            raise ValueError(f"trellis pair {name} contains non-finite values")

    in_features = int(suh.numel())
    out_features = int(svh.numel())
    if rate_axis == "n":
        if out_features != 256:
            raise ValueError(
                f"N-axis trellis pair requires 256 output features, got {out_features}"
            )
        orthogonal_features = in_features
    else:
        if in_features != 256:
            raise ValueError(
                f"K-axis trellis pair requires 256 input features, got {in_features}"
            )
        orthogonal_features = out_features
    if orthogonal_features <= 0 or orthogonal_features % 128:
        raise ValueError(
            "the trellis-pair orthogonal axis must be a positive multiple of 128, "
            f"got {orthogonal_features}"
        )

    low_bits, high_bits = (2, 4) if pair_kind == "P24" else (3, 3)
    orthogonal_tiles = orthogonal_features // 16
    low_words = orthogonal_tiles * 8 * 16 * low_bits
    high_words = orthogonal_tiles * 8 * 16 * high_bits
    expected_words = low_words + high_words
    if payload.numel() != expected_words:
        raise ValueError(
            "trellis pair payload length mismatch: "
            f"expected {expected_words} int16 words, got {payload.numel()}"
        )

    if rate_axis == "n":
        # Reference storage is record-major.  GEMM stages one K16 row at a
        # time, so retain the same bytes while interleaving the two complete
        # record spans at K16 granularity.
        low = payload[:low_words].reshape(orthogonal_tiles, 8 * 16 * low_bits)
        high = payload[low_words:].reshape(
            orthogonal_tiles, 8 * 16 * high_bits
        )
        prepared_i16 = torch.cat((low, high), dim=1).contiguous().reshape(-1)
    else:
        prepared_i16 = payload
    if int(prepared_i16.data_ptr()) % 16:
        raise ValueError("trellis pair payload must be at least 16-byte aligned")

    normalized_codebook = _trellis256_marker_codebook(
        mcg=(
            mcg
            if isinstance(mcg, torch.Tensor)
            else (
                None
                if mcg is None
                else torch.tensor(mcg, dtype=torch.uint32, device=device)
            )
        ),
        mul1_e4m3=(
            mul1_e4m3
            if isinstance(mul1_e4m3, torch.Tensor)
            else (
                None
                if mul1_e4m3 is None
                else torch.tensor(
                    mul1_e4m3, dtype=torch.uint32, device=device
                )
            )
        ),
        codebook=codebook,
    )
    if dummy_scale is None:
        dummy_scale = torch.zeros(4, dtype=torch.uint8, device=device)
    elif (
        dummy_scale.device != device
        or dummy_scale.dtype != torch.uint8
        or tuple(dummy_scale.shape) != (4,)
        or not dummy_scale.is_contiguous()
        or int(dummy_scale.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            "trellis pair dummy_scale must be a contiguous, 16-byte-aligned "
            "four-byte uint8 tensor on the weight device"
        )

    return PreparedTrellis256DenseWeight(
        trellis=prepared_i16.view(torch.int32).reshape(-1),
        suh=suh,
        svh=svh,
        scale=dummy_scale,
        global_scale=torch.ones(1, dtype=torch.float32, device=device),
        workspace=_make_workspace(device, max_blocks_per_sm=4),
        in_features=in_features,
        out_features=out_features,
        params_dtype=params_dtype,
        trellis_bits=3,
        trellis_codebook=normalized_codebook,
        mcg=mcg if isinstance(mcg, torch.Tensor) else None,
        mul1_e4m3=(
            mul1_e4m3 if isinstance(mul1_e4m3, torch.Tensor) else None
        ),
        trellis_pair_kind=pair_kind,
        trellis_rate_axis=rate_axis,
    )


def prepare_qsrt_pair_moe_weights(
    w13_payload: torch.Tensor,
    w2_payload: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    fc1_pair_kind: torch.Tensor,
    fc2_pair_kind: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    params_dtype: torch.dtype = torch.float16,
    codebook: str = "sqg_xor_cheb_t12",
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    """Prepare the compact P24/P33 reference view used by QSRT tests.

    Checkpoint loading uses :func:`prepare_qsrt_atom_moe_weights`.  This
    internal adapter exists only to compare that canonical atom path with the
    former pair-shaped representation.  It accepts only the active
    SQG-XOR-Cheb-T12 codebook and is independent of legacy EXL3.
    """

    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    num_experts = int(num_experts)
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError(
            "trellis pairs require hidden_size to be a positive multiple of 128"
        )
    if intermediate_size != 256:
        raise ValueError("trellis pairs require intermediate_size=256")
    if num_experts <= 0 or not validate_activation(activation):
        raise ValueError("trellis pairs require gated experts")
    if params_dtype != torch.float16:
        raise ValueError("trellis pairs require fp16 operands")
    if _normalize_trellis256_codebook(codebook) != "sqg_xor_cheb_t12":
        raise ValueError("W4A8 trellis pairs support only sqg_xor_cheb_t12")
    if w13_payload.dtype != torch.int16 or w2_payload.dtype != torch.int16:
        raise TypeError("trellis-pair payloads must use torch.int16")
    if not w13_payload.is_cuda or not w2_payload.is_cuda:
        raise ValueError("trellis-pair payloads require CUDA storage")
    if w13_payload.device != w2_payload.device:
        raise ValueError("w13_payload and w2_payload must share one CUDA device")
    if not w13_payload.is_contiguous() or not w2_payload.is_contiguous():
        raise ValueError("trellis-pair payloads must be contiguous")
    device = w13_payload.device

    def _modes(name: str, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a per-expert mode table tensor")
        if value.device != device or tuple(value.shape) != (num_experts,):
            raise ValueError(
                f"{name} mode table must have shape {(num_experts,)} on {device}"
            )
        if value.dtype not in {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(f"{name} mode table must use an integer or bool dtype")
        result = value.to(dtype=torch.int32).contiguous()
        if not bool(torch.all((result == 0) | (result == 1))):
            raise ValueError(f"{name} mode table values must be 0=P33 or 1=P24")
        return result

    fc1_modes = _modes("fc1_pair_kind", fc1_pair_kind)
    fc2_modes = _modes("fc2_pair_kind", fc2_pair_kind)
    hidden_tiles = hidden_size // 16
    pair_words = hidden_tiles * 8 * 16 * 6
    expected_w13 = (2, num_experts, pair_words)
    expected_w2 = (num_experts, pair_words)
    if tuple(w13_payload.shape) != expected_w13:
        raise ValueError(
            f"w13_payload must have shape {expected_w13}, got {tuple(w13_payload.shape)}"
        )
    if tuple(w2_payload.shape) != expected_w2:
        raise ValueError(
            f"w2_payload must have shape {expected_w2}, got {tuple(w2_payload.shape)}"
        )
    if int(w13_payload.data_ptr()) % 16 or int(w2_payload.data_ptr()) % 16:
        raise ValueError("trellis-pair payloads must be at least 16-byte aligned")

    prepared_w13 = torch.empty_like(w13_payload)
    for mode, (low_bits, high_bits) in ((0, (3, 3)), (1, (2, 4))):
        ids = torch.nonzero(fc1_modes == mode, as_tuple=False).flatten()
        if int(ids.numel()) == 0:
            continue
        selected = w13_payload.index_select(1, ids)
        low_words = hidden_tiles * 8 * 16 * low_bits
        low = selected[..., :low_words].reshape(
            2, ids.numel(), hidden_tiles, 8 * 16 * low_bits
        )
        high = selected[..., low_words:].reshape(
            2, ids.numel(), hidden_tiles, 8 * 16 * high_bits
        )
        swizzled = torch.cat((low, high), dim=-1).reshape(
            2, ids.numel(), pair_words
        )
        prepared_w13.index_copy_(1, ids, swizzled)

    for name, scale, shapes in (
        ("gate_suh", gate_suh, ((1, hidden_size), (num_experts, hidden_size))),
        ("up_suh", up_suh, ((1, hidden_size), (num_experts, hidden_size))),
        (
            "intermediate_rotations",
            intermediate_rotations,
            ((num_experts, 3 * intermediate_size),),
        ),
        ("down_svh", down_svh, ((1, hidden_size), (num_experts, hidden_size))),
    ):
        if (
            scale.device != device
            or scale.dtype != torch.float16
            or tuple(scale.shape) not in shapes
            or not scale.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous fp16 {shapes} on {device}; got "
                f"{tuple(scale.shape)}/{scale.dtype}/{scale.device}"
            )
        if not bool(torch.all(torch.isfinite(scale))):
            raise ValueError(f"{name} contains non-finite values")
    if (gate_suh.shape[0] == 1) != (up_suh.shape[0] == 1):
        raise ValueError("gate_suh and up_suh must both be broadcast or per-expert")

    tile_config = tuple(int(value) for value in tile_config)
    if len(tile_config) != 4:
        raise ValueError("tile_config must contain fc1_k, fc1_n, fc2_k, fc2_n")
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = tile_config
    if fc1_tile_n != 256:
        raise ValueError("FC1 pair decode requires fc1_tile_n=256")
    if fc1_tile_k <= 0 or hidden_size % fc1_tile_k:
        raise ValueError("fc1_tile_k must divide hidden_size")
    if fc2_tile_k <= 0 or fc2_tile_k > 128 or 128 % fc2_tile_k:
        raise ValueError("fc2_tile_k must be a positive divisor of 128")
    if fc2_tile_n <= 0 or hidden_size % fc2_tile_n:
        raise ValueError("fc2_tile_n must divide hidden_size")
    if (fc1_tile_k * fc1_tile_n) // 64 != (fc2_tile_k * fc2_tile_n) // 64:
        raise ValueError("FC1 and FC2 pair tiles must use the same CTA thread count")

    if dummy_scale is None:
        dummy_scale = torch.zeros(4, dtype=torch.uint8, device=device)
    elif (
        dummy_scale.device != device
        or dummy_scale.dtype != torch.uint8
        or tuple(dummy_scale.shape) != (4,)
        or not dummy_scale.is_contiguous()
        or int(dummy_scale.data_ptr()) % 16
    ):
        raise ValueError(
            "dummy_scale must be contiguous aligned uint8[4] on the weight device"
        )
    if workspace is None:
        workspace = _make_workspace(device, max_blocks_per_sm=4)
    elif (
        workspace.device != device
        or workspace.dtype != torch.int32
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be contiguous int32 on the weight device")

    global_scale = torch.ones((num_experts,), dtype=torch.float32, device=device)
    return PreparedW4A16MoeWeights(
        w13=prepared_w13.reshape(-1).view(torch.int32),
        w13_scale=dummy_scale,
        w13_global_scale=global_scale,
        w2=w2_payload.reshape(-1).view(torch.int32),
        w2_scale=dummy_scale,
        w2_global_scale=global_scale,
        workspace=workspace,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=True,
        params_dtype=params_dtype,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_n=fc2_tile_n,
        source_format="qsrt_sqg_e4m3",
        w13_layout="trellis3_t256_proj",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        trellis_codebook="sqg_xor_cheb_t12",
        trellis_bits=3,
        fc1_trellis_pair_kind="PDYNAMIC",
        fc2_trellis_pair_kind="PDYNAMIC",
        fc1_trellis_pair_modes=fc1_modes,
        fc2_trellis_pair_modes=fc2_modes,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )


def _prepare_qsrt_p33_p43_moe_weights(
    w13_payload: torch.Tensor,
    w2_payload: torch.Tensor,
    *,
    pair_offsets_u32: torch.Tensor,
    pair_is_p43: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    params_dtype: torch.dtype = torch.float16,
    codebook: str = "sqg_xor_cheb_t12",
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    """Bind the P33/P43 pools decoded from one atoms-v2 extent.

    ``w13_payload`` is projection-major and ``w2_payload`` is expert-major.
    Within every pool, each expert owns one complete 256-channel pair in
    physical record order.  P33 entries contain six bits per coefficient pair;
    P43 entries contain seven.  ``pair_offsets_u32`` gives the exact start of
    each expert in one W13 projection plane and in W2.  No expert is padded.

    The caller must already have applied one common whole-record placement to
    gate, up, and down.  This routine validates compact geometry; it never
    changes logical funding or permutes channels.
    """

    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    num_experts = int(num_experts)
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError(
            "QSRT P33/P43 preparation requires hidden_size to be a positive "
            "multiple of 128"
        )
    if intermediate_size != 256:
        raise ValueError("one prepared QSRT P33/P43 extent must own 256 channels")
    if num_experts <= 0 or not validate_activation(activation):
        raise ValueError("QSRT P33/P43 preparation requires gated experts")
    if params_dtype != torch.float16:
        raise ValueError("QSRT P33/P43 preparation requires fp16 operands")
    if _normalize_trellis256_codebook(codebook) != "sqg_xor_cheb_t12":
        raise ValueError("QSRT P33/P43 preparation supports only sqg_xor_cheb_t12")
    if w13_payload.dtype != torch.int16 or w2_payload.dtype != torch.int16:
        raise TypeError("QSRT P33/P43 payloads must use torch.int16")
    if not w13_payload.is_cuda or not w2_payload.is_cuda:
        raise ValueError("QSRT P33/P43 preparation requires CUDA payloads")
    if w13_payload.device != w2_payload.device:
        raise ValueError("w13_payload and w2_payload must share one CUDA device")
    if not w13_payload.is_contiguous() or not w2_payload.is_contiguous():
        raise ValueError("QSRT P33/P43 payloads must be contiguous")
    if w13_payload.ndim != 1 or w2_payload.ndim != 1:
        raise ValueError("QSRT P33/P43 payload pools must be flat")
    if int(w13_payload.data_ptr()) % 16 or int(w2_payload.data_ptr()) % 16:
        raise ValueError("QSRT P33/P43 payloads must be at least 16-byte aligned")
    device = w13_payload.device

    for name, value in (
        ("pair_offsets_u32", pair_offsets_u32),
        ("pair_is_p43", pair_is_p43),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.device != device or tuple(value.shape) != (num_experts,):
            raise ValueError(f"{name} must have shape {(num_experts,)} on {device}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if pair_offsets_u32.dtype != torch.int64:
        raise TypeError("pair_offsets_u32 must use int64")
    if pair_is_p43.dtype not in {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("pair_is_p43 must use an integer or bool dtype")
    modes = pair_is_p43.to(dtype=torch.int64).contiguous()
    if not bool(torch.all((modes == 0) | (modes == 1))):
        raise ValueError("pair_is_p43 values must be 0=P33 or 1=P43")

    hidden_tiles = hidden_size // 16
    p33_u32 = hidden_tiles * 8 * 8 * 6
    p43_u32 = hidden_tiles * 8 * 8 * 7
    lengths_u32 = torch.where(
        modes != 0,
        torch.full_like(modes, p43_u32),
        torch.full_like(modes, p33_u32),
    )
    if not bool(torch.all(pair_offsets_u32 >= 0)):
        raise ValueError("pair_offsets_u32 must be non-negative")
    order = torch.argsort(pair_offsets_u32)
    sorted_offsets = pair_offsets_u32.index_select(0, order)
    sorted_lengths = lengths_u32.index_select(0, order)
    expected_offsets = torch.empty_like(sorted_lengths)
    expected_offsets[0] = 0
    if num_experts > 1:
        expected_offsets[1:] = torch.cumsum(sorted_lengths[:-1], dim=0)
    if not bool(torch.equal(sorted_offsets, expected_offsets)):
        raise ValueError(
            "pair_offsets_u32 must describe a gap-free, non-overlapping compact "
            "packing of the exact P33/P43 payload lengths"
        )
    total_u32 = int((sorted_offsets[-1] + sorted_lengths[-1]).item())
    if w13_payload.numel() != 4 * total_u32:
        raise ValueError(
            "projection-major w13_payload must contain two compact planes of "
            f"{total_u32} uint32 values"
        )
    if w2_payload.numel() != 2 * total_u32:
        raise ValueError(
            f"w2_payload must contain {total_u32} compact uint32 values"
        )
    descriptors = ((pair_offsets_u32 << 1) | modes).contiguous()

    for name, scale, shapes in (
        ("gate_suh", gate_suh, ((1, hidden_size), (num_experts, hidden_size))),
        ("up_suh", up_suh, ((1, hidden_size), (num_experts, hidden_size))),
        (
            "intermediate_rotations",
            intermediate_rotations,
            ((num_experts, 3 * intermediate_size),),
        ),
        ("down_svh", down_svh, ((1, hidden_size), (num_experts, hidden_size))),
    ):
        if (
            scale.device != device
            or scale.dtype != torch.float16
            or tuple(scale.shape) not in shapes
            or not scale.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous fp16 {shapes} on {device}; got "
                f"{tuple(scale.shape)}/{scale.dtype}/{scale.device}"
            )
        if not bool(torch.all(torch.isfinite(scale))):
            raise ValueError(f"{name} contains non-finite values")
    if (gate_suh.shape[0] == 1) != (up_suh.shape[0] == 1):
        raise ValueError("gate_suh and up_suh must both be broadcast or per-expert")

    tile_config = tuple(int(value) for value in tile_config)
    if len(tile_config) != 4:
        raise ValueError("tile_config must contain fc1_k, fc1_n, fc2_k, fc2_n")
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = tile_config
    if fc1_tile_n != 256:
        raise ValueError("QSRT P33/P43 preparation requires fc1_tile_n=256")
    if fc1_tile_k <= 0 or hidden_size % fc1_tile_k:
        raise ValueError("fc1_tile_k must divide hidden_size")
    if fc2_tile_k <= 0 or fc2_tile_k > 128 or 128 % fc2_tile_k:
        raise ValueError("fc2_tile_k must be a positive divisor of 128")
    if fc2_tile_n <= 0 or hidden_size % fc2_tile_n:
        raise ValueError("fc2_tile_n must divide hidden_size")
    if (fc1_tile_k * fc1_tile_n) // 64 != (fc2_tile_k * fc2_tile_n) // 64:
        raise ValueError("FC1 and FC2 pair tiles must use the same CTA thread count")

    if dummy_scale is None:
        dummy_scale = torch.zeros(4, dtype=torch.uint8, device=device)
    elif (
        dummy_scale.device != device
        or dummy_scale.dtype != torch.uint8
        or tuple(dummy_scale.shape) != (4,)
        or not dummy_scale.is_contiguous()
        or int(dummy_scale.data_ptr()) % 16
    ):
        raise ValueError(
            "dummy_scale must be contiguous aligned uint8[4] on the weight device"
        )
    if workspace is None:
        workspace = _make_workspace(device, max_blocks_per_sm=4)
    elif (
        workspace.device != device
        or workspace.dtype != torch.int32
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be contiguous int32 on the weight device")

    global_scale = torch.ones((num_experts,), dtype=torch.float32, device=device)
    return PreparedW4A16MoeWeights(
        w13=w13_payload.view(torch.int32),
        w13_scale=dummy_scale,
        w13_global_scale=global_scale,
        w2=w2_payload.view(torch.int32),
        w2_scale=dummy_scale,
        w2_global_scale=global_scale,
        workspace=workspace,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=True,
        params_dtype=params_dtype,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_n=fc2_tile_n,
        source_format="qsrt_sqg_e4m3",
        w13_layout="trellis3_t256_proj",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        trellis_codebook="sqg_xor_cheb_t12",
        trellis_bits=3,
        fc1_trellis_pair_kind="P33_P43",
        fc2_trellis_pair_kind="P33_P43",
        fc1_trellis_pair_modes=descriptors,
        fc2_trellis_pair_modes=descriptors,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )


def _qsrt_coupled_rotation_signs(
    length: int,
    *,
    draw: int,
    axis: int,
) -> torch.Tensor:
    """Reproduce kquant's frozen CPU Rademacher draw exactly."""

    if length <= 0 or not 0 <= draw < 8:
        raise ValueError("QSRT coupled rotation draw is outside its contract")
    if draw == 0:
        return torch.ones(length, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (
            0x6A09E667F3BCC909 * int(draw)
            + 0xBB67AE8584CAA73B * int(axis)
        )
        & ((1 << 63) - 1)
    )
    return (
        torch.randint(0, 2, (length,), generator=generator)
        .mul_(2)
        .sub_(1)
        .float()
    )


def _prepare_qsrt_p22_atom_v2_moe_weights(
    atom_payload: torch.Tensor,
    *,
    first_atom_slot: int,
    rotation_draws: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    params_dtype: torch.dtype,
    codebook: str,
    tile_config: tuple[int, int, int, int],
    dummy_scale: torch.Tensor | None,
    workspace: torch.Tensor | None,
) -> PreparedW4A16MoeWeights:
    """Restore one balanced pure-K2 coupled-Hadamard atom extent."""

    atom_count = int(atom_payload.shape[0])
    if atom_count <= 0 or first_atom_slot < 0 or first_atom_slot + atom_count > 96:
        raise ValueError("pure-K2 atom extent lies outside the 96-slot axis")
    if first_atom_slot % 4 or atom_count % 4:
        raise ValueError(
            "pure-K2 atom extents must close 128-channel postactivation blocks"
        )
    pre_half_atoms = 48
    if first_atom_slot < pre_half_atoms < first_atom_slot + atom_count:
        raise ValueError(
            "pure-K2 atom extents must not cross the two transformed "
            "preactivation halves"
        )
    source_intermediate_size = atom_count * 32
    if (
        intermediate_size < source_intermediate_size
        or intermediate_size % 128
    ):
        raise ValueError(
            "pure-K2 runtime intermediate size must contain its atom extent "
            "and close 128-channel postactivation blocks"
        )
    if atom_payload.shape[1] < num_experts * _QSRT_V2_P22_ATOM_BUNDLE_BYTES:
        raise ValueError("pure-K2 atom row is shorter than its expert bundles")
    payload_bytes = num_experts * _QSRT_V2_P22_ATOM_BUNDLE_BYTES
    if bool(torch.any(atom_payload[:, payload_bytes:] != 0)):
        raise ValueError("pure-K2 atom-row padding must be zero")
    if (
        rotation_draws.dtype != torch.uint8
        or tuple(rotation_draws.shape) != (num_experts,)
        or not rotation_draws.is_contiguous()
        or rotation_draws.device.type != "cpu"
        or bool(torch.any(rotation_draws > 7))
    ):
        raise ValueError(
            "pure-K2 rotation_draws must be contiguous CPU uint8[num_experts]"
        )

    device = gate_suh.device
    source = atom_payload[:, :payload_bytes].reshape(
        atom_count, num_experts, _QSRT_V2_P22_ATOM_BUNDLE_BYTES
    )
    hidden_tiles = hidden_size // 16
    source_local_tiles = source_intermediate_size // 16
    local_tiles = intermediate_size // 16
    w13 = torch.zeros(
        (2, num_experts, hidden_tiles, local_tiles, 32),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.zeros(
        (num_experts, local_tiles, hidden_tiles, 32),
        dtype=torch.int16,
        device=device,
    )
    intermediate_rotations = torch.zeros(
        (num_experts, 3 * intermediate_size),
        dtype=torch.float16,
        device=device,
    )
    scale_base = 3 * _QSRT_V2_P22_MATRIX_TRELLIS_BYTES

    for first_expert in range(0, num_experts, 64):
        count = min(64, num_experts - first_expert)
        for matrix_index in range(3):
            begin = matrix_index * _QSRT_V2_P22_MATRIX_TRELLIS_BYTES
            raw = source[
                :,
                first_expert : first_expert + count,
                begin : begin + _QSRT_V2_P22_MATRIX_TRELLIS_BYTES,
            ]
            values = (
                raw.contiguous()
                .to(device=device)
                .view(torch.int16)
                .reshape(atom_count, count, 2, hidden_tiles, 32)
            )
            if matrix_index < 2:
                restored = (
                    values.permute(1, 3, 0, 2, 4)
                    .reshape(count, hidden_tiles, source_local_tiles, 32)
                )
                w13[
                    matrix_index,
                    first_expert : first_expert + count,
                    :,
                    :source_local_tiles,
                ].copy_(restored)
            else:
                restored = (
                    values.permute(1, 0, 2, 3, 4)
                    .reshape(count, source_local_tiles, hidden_tiles, 32)
                )
                w2[
                    first_expert : first_expert + count,
                    :source_local_tiles,
                ].copy_(restored)

            scale_begin = scale_base + matrix_index * 64
            scales = (
                source[
                    :,
                    first_expert : first_expert + count,
                    scale_begin : scale_begin + 64,
                ]
                .contiguous()
                .to(device=device)
                .view(torch.float16)
                .permute(1, 0, 2)
                .reshape(count, source_intermediate_size)
            )
            intermediate_rotations[
                first_expert : first_expert + count,
                matrix_index * intermediate_size : matrix_index
                * intermediate_size
                + source_intermediate_size,
            ].copy_(scales)

    # The preactivation signs index the interleaved length-2I coordinate and
    # the postactivation signs index the ordinary I coordinate.  Atom-v2's
    # coupled placement makes both slices contiguous for every balanced rank.
    pre_begin = 2 * first_atom_slot * 32
    pre_count = 2 * source_intermediate_size
    post_begin = first_atom_slot * 32
    signs = torch.ones(
        (num_experts, 3 * intermediate_size), dtype=torch.float16
    )
    for draw in sorted(set(int(value) for value in rotation_draws.tolist())):
        rows = torch.nonzero(rotation_draws == draw, as_tuple=False).flatten()
        pre = _qsrt_coupled_rotation_signs(2 * 3072, draw=draw, axis=1)[
            pre_begin : pre_begin + pre_count
        ]
        post = _qsrt_coupled_rotation_signs(3072, draw=draw, axis=2)[
            post_begin : post_begin + source_intermediate_size
        ]
        draw_signs = torch.ones(
            (rows.numel(), 3 * intermediate_size), dtype=torch.float16
        )
        draw_signs[:, :pre_count].copy_(
            pre.to(torch.float16).expand(rows.numel(), -1)
        )
        draw_signs[
            :,
            2 * intermediate_size : 2 * intermediate_size
            + source_intermediate_size,
        ].copy_(post.to(torch.float16).expand(rows.numel(), -1))
        signs.index_copy_(0, rows, draw_signs)
    coupled_signs = signs.to(device=device, non_blocking=True).contiguous()

    # Each balanced rank owns a contiguous slice of the transformed length-2I
    # preactivation axis.  Both physical FC1 slots in that slice therefore use
    # the same ordinary input-side scale table: the first 48 atoms derive from
    # the first stored half, and the final 48 from the second.
    source_suh = gate_suh if first_atom_slot < pre_half_atoms else up_suh

    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        activation=activation,
        params_dtype=params_dtype,
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        dummy_scale=dummy_scale,
        codebook=codebook,
        gate_suh=source_suh,
        up_suh=source_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
        workspace=workspace,
    )
    return replace(
        prepared,
        coupled_hadamard=True,
        intermediate_rotations=torch.cat(
            (intermediate_rotations, coupled_signs), dim=1
        ).contiguous(),
    )


def _prepare_qsrt_coupled_h308_atom_v2_moe_weights(
    atom_payload: torch.Tensor,
    *,
    first_atom_slot: int,
    rotation_draws: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    params_dtype: torch.dtype,
    codebook: str,
    tile_config: tuple[int, int, int, int],
    dummy_scale: torch.Tensor | None,
    workspace: torch.Tensor | None,
) -> PreparedW4A16MoeWeights:
    """Restore one balanced coupled-Hadamard H308 atom extent."""

    atom_count = int(atom_payload.shape[0])
    if atom_count != 8 or first_atom_slot % 8 or not 0 <= first_atom_slot <= 88:
        raise ValueError(
            "coupled H308 extents must contain one aligned eight-atom pair"
        )
    if intermediate_size != 256:
        raise ValueError("coupled H308 requires local intermediate_size=256")
    physical_pair = first_atom_slot // 8
    if physical_pair == 5:
        fc1_kind, fc2_kind = "P43", "P33"
        fc1_matrix_bytes = _QSRT_V2_P43_MATRIX_TRELLIS_BYTES
        fc2_matrix_bytes = _QSRT_V2_P33_MATRIX_TRELLIS_BYTES
        bundle_bytes = _QSRT_V2_COUPLED_H308_P43_P33_ATOM_BUNDLE_BYTES
    elif physical_pair == 11:
        fc1_kind, fc2_kind = "P43", "P44"
        fc1_matrix_bytes = _QSRT_V2_P43_MATRIX_TRELLIS_BYTES
        fc2_matrix_bytes = _QSRT_V2_P44_MATRIX_TRELLIS_BYTES
        bundle_bytes = _QSRT_V2_COUPLED_H308_P43_P44_ATOM_BUNDLE_BYTES
    else:
        fc1_kind = fc2_kind = "P33"
        fc1_matrix_bytes = fc2_matrix_bytes = _QSRT_V2_P33_MATRIX_TRELLIS_BYTES
        bundle_bytes = _QSRT_V2_P33_ATOM_BUNDLE_BYTES
    if atom_payload.shape[1] < num_experts * bundle_bytes:
        raise ValueError("coupled H308 atom row is shorter than its expert bundles")
    payload_bytes = num_experts * bundle_bytes
    if bool(torch.any(atom_payload[:, payload_bytes:] != 0)):
        raise ValueError("coupled H308 atom-row padding must be zero")
    if (
        rotation_draws.dtype != torch.uint8
        or tuple(rotation_draws.shape) != (num_experts,)
        or not rotation_draws.is_contiguous()
        or rotation_draws.device.type != "cpu"
        or bool(torch.any(rotation_draws > 7))
    ):
        raise ValueError(
            "coupled H308 rotation_draws must be contiguous CPU uint8[num_experts]"
        )
    if tuple(int(value) for value in tile_config) != (64, 256, 64, 256):
        raise ValueError(
            "coupled H308 currently requires tile_config=(64, 256, 64, 256)"
        )

    device = gate_suh.device
    if device.type != "cuda":
        raise ValueError("coupled H308 preparation requires CUDA output tensors")
    for name, value, shapes in (
        ("gate_suh", gate_suh, ((1, hidden_size), (num_experts, hidden_size))),
        ("up_suh", up_suh, ((1, hidden_size), (num_experts, hidden_size))),
        ("down_svh", down_svh, ((1, hidden_size), (num_experts, hidden_size))),
    ):
        if (
            value.device != device
            or value.dtype != torch.float16
            or tuple(value.shape) not in shapes
            or not value.is_contiguous()
            or not bool(torch.all(torch.isfinite(value)))
        ):
            raise ValueError(
                f"{name} must be finite contiguous fp16 {shapes} on {device}"
            )
    if (gate_suh.shape[0] == 1) != (up_suh.shape[0] == 1):
        raise ValueError("gate_suh and up_suh must both be broadcast or per-expert")

    source = atom_payload[:, :payload_bytes].reshape(
        atom_count, num_experts, bundle_bytes
    )
    hidden_tiles = hidden_size // 16
    pair_bits = {"P33": (3, 3), "P43": (4, 3), "P44": (4, 4)}

    def restore_matrix(
        begin: int,
        matrix_bytes: int,
        kind: str,
        *,
        fc1: bool,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        low_bits, high_bits = pair_bits[kind]
        low_bytes = hidden_tiles * 32 * low_bits
        if low_bytes + hidden_tiles * 32 * high_bits != matrix_bytes:
            raise AssertionError("coupled H308 matrix geometry drifted")
        elements_per_expert = atom_count * matrix_bytes // 2
        if out is None:
            out = torch.empty(
                (num_experts, elements_per_expert),
                dtype=torch.int16,
                device=device,
            )
        elif (
            out.dtype != torch.int16
            or out.device != device
            or tuple(out.shape) != (num_experts, elements_per_expert)
            or not out.is_contiguous()
        ):
            raise ValueError("coupled H308 restore destination has invalid geometry")
        for first_expert in range(0, num_experts, 64):
            count = min(64, num_experts - first_expert)
            raw = source[
                :,
                first_expert : first_expert + count,
                begin : begin + matrix_bytes,
            ]
            low = (
                raw[..., :low_bytes]
                .contiguous()
                .to(device=device)
                .view(torch.int16)
                .reshape(atom_count, count, hidden_tiles, 16 * low_bits)
            )
            high = (
                raw[..., low_bytes:]
                .contiguous()
                .to(device=device)
                .view(torch.int16)
                .reshape(atom_count, count, hidden_tiles, 16 * high_bits)
            )
            if fc1:
                low = low.permute(1, 2, 0, 3).reshape(count, hidden_tiles, -1)
                high = high.permute(1, 2, 0, 3).reshape(count, hidden_tiles, -1)
                restored = torch.cat((low, high), dim=2).reshape(count, -1)
            else:
                low = low.permute(1, 0, 2, 3).reshape(count, -1)
                high = high.permute(1, 0, 2, 3).reshape(count, -1)
                restored = torch.cat((low, high), dim=1)
            out[first_expert : first_expert + count].copy_(restored)
        return out

    matrix_offsets = (0, fc1_matrix_bytes, 2 * fc1_matrix_bytes)
    fc1_elements_per_expert = atom_count * fc1_matrix_bytes // 2
    w13_storage = torch.empty(
        (2, num_experts, fc1_elements_per_expert),
        dtype=torch.int16,
        device=device,
    )
    restore_matrix(
        matrix_offsets[0],
        fc1_matrix_bytes,
        fc1_kind,
        fc1=True,
        out=w13_storage[0],
    )
    restore_matrix(
        matrix_offsets[1],
        fc1_matrix_bytes,
        fc1_kind,
        fc1=True,
        out=w13_storage[1],
    )
    w13 = w13_storage.reshape(-1)
    w2 = restore_matrix(
        matrix_offsets[2], fc2_matrix_bytes, fc2_kind, fc1=False
    ).reshape(-1)

    scale_base = 2 * fc1_matrix_bytes + fc2_matrix_bytes
    intermediate_scales = torch.empty(
        (num_experts, 3 * intermediate_size),
        dtype=torch.float16,
        device=device,
    )
    for matrix_index in range(3):
        begin = scale_base + matrix_index * 64
        for first_expert in range(0, num_experts, 64):
            count = min(64, num_experts - first_expert)
            values = (
                source[
                    :,
                    first_expert : first_expert + count,
                    begin : begin + 64,
                ]
                .contiguous()
                .to(device=device)
                .view(torch.float16)
                .reshape(atom_count, count, 32)
            )
            restored = torch.cat(
                (
                    values[:, :, :16].permute(1, 0, 2).reshape(count, 128),
                    values[:, :, 16:].permute(1, 0, 2).reshape(count, 128),
                ),
                dim=1,
            )
            intermediate_scales[
                first_expert : first_expert + count,
                matrix_index
                * intermediate_size : (matrix_index + 1)
                * intermediate_size,
            ].copy_(restored)

    pre_begin = 2 * first_atom_slot * 32
    post_begin = first_atom_slot * 32
    signs = torch.empty((num_experts, 3 * intermediate_size), dtype=torch.float16)
    for draw in sorted(set(int(value) for value in rotation_draws.tolist())):
        rows = torch.nonzero(rotation_draws == draw, as_tuple=False).flatten()
        pre = _qsrt_coupled_rotation_signs(6144, draw=draw, axis=1)[
            pre_begin : pre_begin + 2 * intermediate_size
        ]
        post = _qsrt_coupled_rotation_signs(3072, draw=draw, axis=2)[
            post_begin : post_begin + intermediate_size
        ]
        signs.index_copy_(
            0,
            rows,
            torch.cat((pre, post)).to(torch.float16).expand(rows.numel(), -1),
        )
    rotations = torch.cat(
        (intermediate_scales, signs.to(device=device, non_blocking=True)), dim=1
    ).contiguous()

    if dummy_scale is None:
        dummy_scale = torch.zeros(4, dtype=torch.uint8, device=device)
    if (
        dummy_scale.device != device
        or dummy_scale.dtype != torch.uint8
        or tuple(dummy_scale.shape) != (4,)
        or not dummy_scale.is_contiguous()
        or dummy_scale.data_ptr() % 16
    ):
        raise ValueError(
            "dummy_scale must be contiguous aligned uint8[4] on the weight device"
        )
    if workspace is None:
        workspace = _make_workspace(device, max_blocks_per_sm=4)
    if (
        workspace.device != device
        or workspace.dtype != torch.int32
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be contiguous int32 on the weight device")
    normalized_codebook = _trellis256_marker_codebook(
        mcg=None, mul1_e4m3=None, codebook=codebook
    )
    global_scale = torch.ones((num_experts,), dtype=torch.float32, device=device)
    source_suh = gate_suh if physical_pair < 6 else up_suh
    return PreparedW4A16MoeWeights(
        w13=w13.view(torch.int32),
        w13_scale=dummy_scale,
        w13_global_scale=global_scale,
        w2=w2.view(torch.int32),
        w2_scale=dummy_scale,
        w2_global_scale=global_scale,
        workspace=workspace,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=True,
        params_dtype=params_dtype,
        fc1_tile_n=256,
        fc2_tile_n=256,
        source_format="qsrt_sqg_e4m3",
        w13_layout="trellis3_t256_proj",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        trellis_codebook=normalized_codebook,
        trellis_bits=3,
        fc1_trellis_pair_kind=fc1_kind,
        fc2_trellis_pair_kind=fc2_kind,
        gate_suh=source_suh,
        up_suh=source_suh,
        intermediate_rotations=rotations,
        down_svh=down_svh,
        coupled_hadamard=True,
        tile_config=(64, 256, 64, 256),
    )


def prepare_qsrt_atom_v2_moe_weights(
    atom_payload: torch.Tensor,
    *,
    first_atom_slot: int,
    layer_index: int,
    profile: str = _QSRT_V2_PROFILE_H308,
    rotation_draws: torch.Tensor | None = None,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    params_dtype: torch.dtype = torch.float16,
    codebook: str = "sqg_xor_cheb_t12",
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    """Prepare one balanced rank extent from a QSRT atoms-v2 profile."""

    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    num_experts = int(num_experts)
    first_atom_slot = int(first_atom_slot)
    layer_index = int(layer_index)
    if num_experts != 896:
        raise ValueError("QSRT atoms-v2 currently requires all 896 experts")
    if hidden_size != 3584:
        raise ValueError("QSRT atoms-v2 currently requires hidden_size=3584")
    if profile not in {
        _QSRT_V2_PROFILE_H308,
        _QSRT_V2_PROFILE_COUPLED_K2,
        _QSRT_V2_PROFILE_COUPLED_H308,
    }:
        raise ValueError(f"unsupported QSRT atoms-v2 profile {profile!r}")
    if not validate_activation(activation) or params_dtype != torch.float16:
        raise ValueError("QSRT atoms-v2 requires gated fp16 preparation")
    if _normalize_trellis256_codebook(codebook) != "sqg_xor_cheb_t12":
        raise ValueError("QSRT atoms-v2 supports only sqg_xor_cheb_t12")
    if (
        atom_payload.dtype != torch.uint8
        or atom_payload.ndim != 2
        or atom_payload.shape[0] <= 0
        or not atom_payload.is_contiguous()
        or atom_payload.device.type not in {"cpu", "cuda"}
    ):
        raise ValueError(
            "atom_payload must be contiguous CPU/CUDA uint8 [atoms, row_stride]"
        )
    if not 1 <= layer_index <= 92:
        raise ValueError("layer_index must identify a Kimi-K3 MoE layer")

    if profile == _QSRT_V2_PROFILE_COUPLED_K2:
        if rotation_draws is None:
            raise ValueError("pure-K2 atoms-v2 requires coupled rotation draws")
        return _prepare_qsrt_p22_atom_v2_moe_weights(
            atom_payload,
            first_atom_slot=first_atom_slot,
            rotation_draws=rotation_draws,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            activation=activation,
            gate_suh=gate_suh,
            up_suh=up_suh,
            down_svh=down_svh,
            params_dtype=params_dtype,
            codebook=codebook,
            tile_config=tile_config,
            dummy_scale=dummy_scale,
            workspace=workspace,
        )
    if profile == _QSRT_V2_PROFILE_COUPLED_H308:
        if rotation_draws is None:
            raise ValueError("coupled H308 atoms-v2 requires rotation draws")
        return _prepare_qsrt_coupled_h308_atom_v2_moe_weights(
            atom_payload,
            first_atom_slot=first_atom_slot,
            rotation_draws=rotation_draws,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            activation=activation,
            gate_suh=gate_suh,
            up_suh=up_suh,
            down_svh=down_svh,
            params_dtype=params_dtype,
            codebook=codebook,
            tile_config=tile_config,
            dummy_scale=dummy_scale,
            workspace=workspace,
        )

    if intermediate_size != 256:
        raise ValueError("H308 atoms-v2 preparation requires 256 local channels")
    if atom_payload.shape[0] != _QSRT_ATOMS_PER_PAIR:
        raise ValueError("H308 atoms-v2 requires exactly eight atom rows")
    if first_atom_slot % _QSRT_ATOMS_PER_PAIR or not 0 <= first_atom_slot < 96:
        raise ValueError("H308 atoms-v2 requires a pair-aligned atom extent")
    if rotation_draws is not None:
        raise ValueError("H308 atoms-v2 must not carry coupled rotation draws")

    device = gate_suh.device
    if device.type != "cuda":
        raise ValueError("QSRT atoms-v2 preparation requires CUDA output tensors")
    physical_pair = first_atom_slot // _QSRT_ATOMS_PER_PAIR
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    rotation = (_QSRT_EXPERT_ROTATION_MULTIPLIER * expert_ids + layer_index) % 12
    base_pair = (physical_pair - rotation) % 12
    modes = ((base_pair == 0) | (base_pair == 6)).to(torch.int64)
    p33_ids = torch.nonzero(modes == 0, as_tuple=False).flatten()
    p43_ids = torch.nonzero(modes == 1, as_tuple=False).flatten()
    count33 = int(p33_ids.numel())
    count43 = int(p43_ids.numel())
    p33_bytes = count33 * _QSRT_V2_P33_ATOM_BUNDLE_BYTES
    p43_bytes = count43 * _QSRT_V2_P43_ATOM_BUNDLE_BYTES
    if atom_payload.shape[1] < p33_bytes + p43_bytes:
        raise ValueError("QSRT atoms-v2 row is shorter than its compact groups")
    if bool(torch.any(atom_payload[:, p33_bytes + p43_bytes :] != 0)):
        raise ValueError("QSRT atoms-v2 row padding must be zero")

    groups: list[tuple[torch.Tensor, torch.Tensor, bool, int]] = []
    for ids, p43, begin, bundle in (
        (p33_ids, False, 0, _QSRT_V2_P33_ATOM_BUNDLE_BYTES),
        (p43_ids, True, p33_bytes, _QSRT_V2_P43_ATOM_BUNDLE_BYTES),
    ):
        count = int(ids.numel())
        view = atom_payload[:, begin : begin + count * bundle].reshape(
            _QSRT_ATOMS_PER_PAIR, count, bundle
        )
        groups.append((ids, view, p43, bundle))

    hidden_tiles = hidden_size // 16

    def _restore_group_matrix_into(
        source: torch.Tensor,
        matrix_index: int,
        destination: torch.Tensor,
        *,
        p43: bool,
        fc1: bool,
    ) -> None:
        low_bits, high_bits = ((4, 3) if p43 else (3, 3))
        matrix_bytes = (
            _QSRT_V2_P43_MATRIX_TRELLIS_BYTES
            if p43
            else _QSRT_V2_P33_MATRIX_TRELLIS_BYTES
        )
        count = source.shape[1]
        words_per_expert = destination.numel() // count
        destination = destination.reshape(count, words_per_expert)
        # The canonical slab is close to 1 GiB per rank.  Stream bounded
        # expert groups through the GPU so loading never needs the canonical
        # slab and its equally-sized prepared view resident at once.
        for first_expert in range(0, count, 64):
            chunk_count = min(64, count - first_expert)
            raw = source[
                :,
                first_expert : first_expert + chunk_count,
                matrix_index * matrix_bytes : (matrix_index + 1) * matrix_bytes,
            ]
            selected = (
                raw.contiguous()
                .to(device=device)
                .view(torch.int16)
                .reshape(_QSRT_ATOMS_PER_PAIR, chunk_count, -1)
                .permute(1, 0, 2)
            )
            low_words = hidden_tiles * 16 * low_bits
            low = selected[..., :low_words].reshape(
                chunk_count,
                _QSRT_ATOMS_PER_PAIR,
                hidden_tiles,
                16 * low_bits,
            )
            high = selected[..., low_words:].reshape(
                chunk_count,
                _QSRT_ATOMS_PER_PAIR,
                hidden_tiles,
                16 * high_bits,
            )
            target = destination[first_expert : first_expert + chunk_count]
            if fc1:
                target = target.reshape(chunk_count, hidden_tiles, -1)
                target[..., : _QSRT_ATOMS_PER_PAIR * 16 * low_bits].reshape(
                    chunk_count,
                    hidden_tiles,
                    _QSRT_ATOMS_PER_PAIR,
                    16 * low_bits,
                ).copy_(low.permute(0, 2, 1, 3))
                target[..., _QSRT_ATOMS_PER_PAIR * 16 * low_bits :].reshape(
                    chunk_count,
                    hidden_tiles,
                    _QSRT_ATOMS_PER_PAIR,
                    16 * high_bits,
                ).copy_(high.permute(0, 2, 1, 3))
            else:
                low_words_per_expert = low.numel() // chunk_count
                target[:, :low_words_per_expert].copy_(
                    low.reshape(chunk_count, -1)
                )
                target[:, low_words_per_expert:].copy_(
                    high.reshape(chunk_count, -1)
                )

    group_matrix_words = []
    for ids, _source, p43, _bundle in groups:
        matrix_bytes = (
            _QSRT_V2_P43_MATRIX_TRELLIS_BYTES
            if p43
            else _QSRT_V2_P33_MATRIX_TRELLIS_BYTES
        )
        group_matrix_words.append(
            int(ids.numel()) * _QSRT_ATOMS_PER_PAIR * matrix_bytes // 2
        )
    matrix_words = sum(group_matrix_words)
    w13_payload = torch.empty(2 * matrix_words, dtype=torch.int16, device=device)
    w2_payload = torch.empty(matrix_words, dtype=torch.int16, device=device)

    for matrix_index in range(2):
        group_offset = matrix_index * matrix_words
        for group_words, (_ids, source, p43, _bundle) in zip(
            group_matrix_words, groups
        ):
            _restore_group_matrix_into(
                source,
                matrix_index,
                w13_payload.narrow(0, group_offset, group_words),
                p43=p43,
                fc1=True,
            )
            group_offset += group_words
    group_offset = 0
    for group_words, (_ids, source, p43, _bundle) in zip(
        group_matrix_words, groups
    ):
        _restore_group_matrix_into(
            source,
            2,
            w2_payload.narrow(0, group_offset, group_words),
            p43=p43,
            fc1=False,
        )
        group_offset += group_words

    p33_u32 = hidden_tiles * 8 * 8 * 6
    p43_u32 = hidden_tiles * 8 * 8 * 7
    offsets = torch.empty(num_experts, dtype=torch.int64, device=device)
    offsets.index_copy_(0, p33_ids, torch.arange(count33, device=device) * p33_u32)
    offsets.index_copy_(
        0,
        p43_ids,
        count33 * p33_u32 + torch.arange(count43, device=device) * p43_u32,
    )

    intermediate_rotations = torch.empty(
        (num_experts, 3 * intermediate_size), dtype=torch.float16, device=device
    )
    for ids, source, p43, _bundle in groups:
        matrix_bytes = (
            _QSRT_V2_P43_MATRIX_TRELLIS_BYTES
            if p43
            else _QSRT_V2_P33_MATRIX_TRELLIS_BYTES
        )
        count = int(ids.numel())
        matrices = []
        for matrix_index in range(3):
            begin = 3 * matrix_bytes + matrix_index * _QSRT_MATRIX_ATOM_SCALE_BYTES
            raw = source.narrow(2, begin, _QSRT_MATRIX_ATOM_SCALE_BYTES)
            values = (
                raw.contiguous()
                .to(device=device)
                .view(torch.float16)
                .reshape(_QSRT_ATOMS_PER_PAIR, count, _QSRT_ATOM_CHANNELS)
                .permute(1, 0, 2)
            )
            matrices.append(
                torch.cat(
                    (
                        values[..., :16].reshape(count, -1),
                        values[..., 16:].reshape(count, -1),
                    ),
                    dim=1,
                )
            )
        intermediate_rotations.index_copy_(0, ids, torch.cat(tuple(matrices), dim=1))

    return _prepare_qsrt_p33_p43_moe_weights(
        w13_payload,
        w2_payload,
        pair_offsets_u32=offsets,
        pair_is_p43=modes,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        activation=activation,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations.contiguous(),
        down_svh=down_svh,
        params_dtype=params_dtype,
        codebook=codebook,
        tile_config=tile_config,
        dummy_scale=dummy_scale,
        workspace=workspace,
    )


def prepare_qsrt_atom_moe_weights(
    atom_payload: torch.Tensor,
    *,
    first_atom_slot: int,
    layer_index: int,
    expert_ids: torch.Tensor,
    format_codes: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    params_dtype: torch.dtype = torch.float16,
    codebook: str | None = "sqg_xor_cheb_t12",
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    """Prepare one canonical QSRT atom extent for the fused decoder.

    ``atom_payload`` is the de-padded checkpoint tensor
    ``[atom_slots, E, 129216]u8``.  Storage is independent of a deployment's
    shard count: every first-axis entry owns a complete 32-channel physical
    atom.  The currently qualified fused kernel consumes one aligned extent of
    eight atoms (256 channels), so this load-time transform restores its local
    P24/P33 view without decoding or re-encoding any trellis symbols.

    Expert format bytes encode ``r13`` in the high nibble and ``r2`` in the
    low nibble.  The physical atom rotation is inverted from ``layer_index``
    and the global ``expert_ids``; no rank-local pair-mode side table exists in
    the checkpoint.
    """

    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    num_experts = int(num_experts)
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError(
            "QSRT atoms require hidden_size to be a positive multiple "
            f"of 128, got {hidden_size}"
        )
    if intermediate_size != _QSRT_ATOMS_PER_PAIR * _QSRT_ATOM_CHANNELS:
        raise ValueError(
            "the current QSRT fused kernel requires one eight-atom extent "
            "(intermediate_size=256), "
            f"got {intermediate_size}"
        )
    if num_experts <= 0:
        raise ValueError("QSRT atoms require num_experts > 0")
    if not validate_activation(activation):
        raise ValueError("QSRT atoms require a gated expert activation")
    if params_dtype != torch.float16:
        raise ValueError("full-rotation QSRT requires fp16 MMA operands")
    if atom_payload.dtype != torch.uint8:
        raise TypeError("QSRT atom payloads must use torch.uint8")
    if not atom_payload.is_cuda:
        raise ValueError("QSRT atom preparation requires CUDA storage")
    expected_atoms = (
        _QSRT_ATOMS_PER_PAIR,
        num_experts,
        _QSRT_ATOM_BUNDLE_BYTES,
    )
    if tuple(atom_payload.shape) != expected_atoms:
        raise ValueError(
            f"atom_payload must have shape {expected_atoms}, got "
            f"{tuple(atom_payload.shape)}"
        )
    expected_inner_strides = (_QSRT_ATOM_BUNDLE_BYTES, 1)
    if tuple(atom_payload.stride()[1:]) != expected_inner_strides or int(
        atom_payload.stride(0)
    ) < num_experts * _QSRT_ATOM_BUNDLE_BYTES:
        raise ValueError(
            "QSRT atom payloads must be expert-major within each atom row; "
            "the row stride may include checkpoint alignment padding"
        )
    device = atom_payload.device
    first_atom_slot = int(first_atom_slot)
    layer_index = int(layer_index)
    if not 0 <= first_atom_slot < _QSRT_ATOMS_PER_EXPERT:
        raise ValueError(
            f"first_atom_slot must be in 0..{_QSRT_ATOMS_PER_EXPERT - 1}"
        )
    if first_atom_slot % _QSRT_ATOMS_PER_PAIR:
        raise ValueError("the current QSRT kernel requires a pair-aligned atom extent")
    if not 1 <= layer_index <= 92:
        raise ValueError("layer_index must identify a Kimi-K3 MoE layer in 1..92")

    def _normalize_vector(name: str, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.device != device or tuple(value.shape) != (num_experts,):
            raise ValueError(
                f"{name} must have shape {(num_experts,)} on {device}"
            )
        if value.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(f"{name} must use an integer dtype")
        return value.to(dtype=torch.int32).contiguous()

    expert_ids_i32 = _normalize_vector("expert_ids", expert_ids)
    format_codes_i32 = _normalize_vector("format_codes", format_codes)
    if not bool(torch.all((expert_ids_i32 >= 0) & (expert_ids_i32 < 896))):
        raise ValueError("expert_ids must lie in 0..895")
    r13 = format_codes_i32 >> 4
    r2 = format_codes_i32 & 0xF
    if not bool(torch.all((r13 >= 0) & (r13 <= 2) & (r2 >= 0) & (r2 <= 2))):
        raise ValueError("compressed QSRT format codes must encode R0/R1/R2")
    physical_pair = first_atom_slot // _QSRT_ATOMS_PER_PAIR
    rotation = (
        _QSRT_EXPERT_ROTATION_MULTIPLIER * expert_ids_i32 + layer_index
    ) % 12
    logical_pair = (physical_pair - rotation) % 12
    fc1_pair_modes = (logical_pair < r13).to(dtype=torch.int32).contiguous()
    fc2_pair_modes = (logical_pair < r2).to(dtype=torch.int32).contiguous()
    fc1_pair_kind = "PDYNAMIC"
    fc2_pair_kind = "PDYNAMIC"

    hidden_tiles = hidden_size // 16
    pair_words = hidden_tiles * 8 * 16 * 6
    words_per_atom = _QSRT_MATRIX_ATOM_TRELLIS_BYTES // 2

    def _matrix_words(matrix_index: int) -> torch.Tensor:
        begin = _QSRT_MATRIX_TRELLIS_OFFSETS[matrix_index]
        raw = atom_payload.narrow(
            2, begin, _QSRT_MATRIX_ATOM_TRELLIS_BYTES
        ).contiguous()
        return raw.view(torch.int16).reshape(
            _QSRT_ATOMS_PER_PAIR, num_experts, words_per_atom
        ).permute(1, 0, 2)

    def _restore_matrix(
        matrix_index: int, modes: torch.Tensor, *, fc1: bool
    ) -> torch.Tensor:
        source = _matrix_words(matrix_index)
        output = torch.empty(
            (num_experts, pair_words), dtype=torch.int16, device=device
        )
        for mode, (low_bits, high_bits) in ((0, (3, 3)), (1, (2, 4))):
            ids = torch.nonzero(modes == mode, as_tuple=False).flatten()
            if int(ids.numel()) == 0:
                continue
            selected = source.index_select(0, ids).narrow(
                2, 0, hidden_tiles * 16 * 6
            )
            low_words = hidden_tiles * 16 * low_bits
            low = selected[..., :low_words].reshape(
                -1, _QSRT_ATOMS_PER_PAIR, hidden_tiles, 16 * low_bits
            )
            high = selected[..., low_words:].reshape(
                -1, _QSRT_ATOMS_PER_PAIR, hidden_tiles, 16 * high_bits
            )
            if fc1:
                low = low.permute(0, 2, 1, 3).reshape(
                    ids.numel(), hidden_tiles, -1
                )
                high = high.permute(0, 2, 1, 3).reshape(
                    ids.numel(), hidden_tiles, -1
                )
                # FC1 places both 128-channel records under each K16 tile.
                restored = torch.cat((low, high), dim=-1).reshape(
                    ids.numel(), -1
                )
            else:
                # FC2 retains its K-major low-plane/high-plane ordering.
                restored = torch.cat(
                    (
                        low.reshape(ids.numel(), -1),
                        high.reshape(ids.numel(), -1),
                    ),
                    dim=1,
                )
            output.index_copy_(0, ids, restored)
        return output

    prepared_w13_i16 = torch.stack(
        (
            _restore_matrix(0, fc1_pair_modes, fc1=True),
            _restore_matrix(1, fc1_pair_modes, fc1=True),
        )
    ).reshape(-1)
    prepared_w2_i16 = _restore_matrix(
        2, fc2_pair_modes, fc1=False
    ).reshape(-1)

    def _local_scale(matrix_index: int) -> torch.Tensor:
        begin = _QSRT_MATRIX_SCALE_OFFSETS[matrix_index]
        raw = atom_payload.narrow(
            2, begin, _QSRT_MATRIX_ATOM_SCALE_BYTES
        ).contiguous()
        values = raw.view(torch.float16).reshape(
            _QSRT_ATOMS_PER_PAIR, num_experts, _QSRT_ATOM_CHANNELS
        ).permute(1, 0, 2)
        return torch.cat(
            (
                values[..., :16].reshape(num_experts, -1),
                values[..., 16:].reshape(num_experts, -1),
            ),
            dim=1,
        )

    intermediate_rotations = torch.cat(
        (_local_scale(0), _local_scale(1), _local_scale(2)), dim=1
    ).contiguous()

    normalized_codebook = _trellis256_marker_codebook(
        mcg=None,
        mul1_e4m3=None,
        codebook=codebook,
    )

    for name, scale, shapes in (
        ("gate_suh", gate_suh, ((1, hidden_size), (num_experts, hidden_size))),
        ("up_suh", up_suh, ((1, hidden_size), (num_experts, hidden_size))),
        (
            "intermediate_rotations",
            intermediate_rotations,
            ((num_experts, 3 * intermediate_size),),
        ),
        ("down_svh", down_svh, ((1, hidden_size), (num_experts, hidden_size))),
    ):
        if (
            scale.device != device
            or scale.dtype != torch.float16
            or tuple(scale.shape) not in shapes
            or not scale.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous fp16 {shapes} on {device}; got "
                f"{tuple(scale.shape)}/{scale.dtype}/{scale.device}"
            )
        if not bool(torch.all(torch.isfinite(scale))):
            raise ValueError(f"{name} contains non-finite values")
    if (gate_suh.shape[0] == 1) != (up_suh.shape[0] == 1):
        raise ValueError("gate_suh and up_suh must both be broadcast or per-expert")

    tile_config = tuple(int(value) for value in tile_config)
    if len(tile_config) != 4:
        raise ValueError("tile_config must contain fc1_k, fc1_n, fc2_k, fc2_n")
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = tile_config
    if fc1_tile_n != 256:
        raise ValueError("QSRT pair decode requires fc1_tile_n=256")
    if fc1_tile_k <= 0 or hidden_size % fc1_tile_k:
        raise ValueError("fc1_tile_k must divide hidden_size")
    if fc2_tile_k <= 0 or fc2_tile_k > 128 or 128 % fc2_tile_k:
        raise ValueError("fc2_tile_k must be a positive divisor of 128")
    if fc2_tile_n <= 0 or hidden_size % fc2_tile_n:
        raise ValueError("fc2_tile_n must divide hidden_size")
    if (fc1_tile_k * fc1_tile_n) // 64 != (fc2_tile_k * fc2_tile_n) // 64:
        raise ValueError("FC1 and FC2 pair tiles must use the same CTA thread count")

    if dummy_scale is None:
        dummy_scale = torch.zeros(4, dtype=torch.uint8, device=device)
    elif (
        dummy_scale.device != device
        or dummy_scale.dtype != torch.uint8
        or tuple(dummy_scale.shape) != (4,)
        or not dummy_scale.is_contiguous()
        or int(dummy_scale.data_ptr()) % 16
    ):
        raise ValueError(
            "dummy_scale must be contiguous aligned uint8[4] on the weight device"
        )
    if workspace is None:
        workspace = _make_workspace(device, max_blocks_per_sm=4)
    elif (
        workspace.device != device
        or workspace.dtype != torch.int32
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be contiguous int32 on the weight device")

    global_scale = torch.ones((num_experts,), dtype=torch.float32, device=device)
    return PreparedW4A16MoeWeights(
        w13=prepared_w13_i16.view(torch.int32),
        w13_scale=dummy_scale,
        w13_global_scale=global_scale,
        w2=prepared_w2_i16.view(torch.int32),
        w2_scale=dummy_scale,
        w2_global_scale=global_scale,
        workspace=workspace,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=True,
        params_dtype=params_dtype,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_n=fc2_tile_n,
        source_format="qsrt_sqg_e4m3",
        w13_layout="trellis3_t256_proj",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        trellis_codebook=normalized_codebook,
        trellis_bits=3,
        fc1_trellis_pair_kind=fc1_pair_kind,
        fc2_trellis_pair_kind=fc2_pair_kind,
        fc1_trellis_pair_modes=fc1_pair_modes,
        fc2_trellis_pair_modes=fc2_pair_modes,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )


__all__ = [
    "PreparedW4A16MoeWeights",
    "PreparedTrellis256DenseWeight",
    "W4A16PackedBuffers",
    "W4A16FC2Weights",
    "W4A16ModelOptWeights",
    "W4A16PackedWeights",
    "make_w4a16_packed_buffers",
    "prepare_trellis256_moe_weights",
    "prepare_trellis256_dense_weight",
    "prepare_trellis256_pair_dense_weight",
    "prepare_qsrt_atom_moe_weights",
    "prepare_qsrt_atom_v2_moe_weights",
    "prepare_w4a16_compressed_tensors_weights",
    "prepare_w4a16_e8m0_native_weights",
    "prepare_w4a16_fc2_e8m0_weights",
    "prepare_w4a16_fp4_e8m0_k32_weights",
    "prepare_w4a16_x4t_weights",
    "prepare_w4a16_modelopt_native_weights",
    "prepare_w4a16_modelopt_nvfp4_weights",
    "prepare_w4a16_packed_weights",
]
