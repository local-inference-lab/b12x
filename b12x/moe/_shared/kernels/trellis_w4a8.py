"""End-to-end QSRT W4A8 MoE path.

The route-major path keeps the encoded weights compact throughout: activation
transforms feed MXFP8 operands, and the fixed-rate trellis payload is decoded
directly into E4M3 MMA registers. The caller owns all scratch so the complete
sequence is CUDA-graph safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    empty_mxfp8_rows_for_dense_gemm,
)
from b12x.moe._shared.kernels.trellis_w4a8_pair import (
    run_trellis_w4a8_fc1_routes,
    run_trellis_w4a8_fc2_routes,
)
from b12x.moe._shared.kernels.trellis_w4a8_uniform import (
    run_trellis_w4a8_uniform_fc1,
    run_trellis_w4a8_uniform_fc2,
)
from b12x.moe._shared.kernels.trellis_w4a8_transform import (
    run_trellis_w4a8_activation_rotation_quant,
    run_trellis_w4a8_coupled_activation_quant,
    run_trellis_w4a8_coupled_input_quant,
    run_trellis_w4a8_input_rotation_quant,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _w4a16_topk_sum_launch_flat,
)


@dataclass(frozen=True)
class TrellisW4A8MoeScratch:
    """Caller-owned intermediate storage for one fixed ``(M, top-k)`` shape."""

    route_experts: torch.Tensor
    # Broadcast checkpoints quantize one FC1 operand per token.  Per-expert
    # checkpoints quantize one operand per route because each route applies a
    # different input rotation.
    gate_quantized: MXFP8Rows
    up_quantized: MXFP8Rows
    gate_fc1: torch.Tensor
    up_fc1: torch.Tensor
    activated_quantized: MXFP8Rows
    fc2: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class TrellisW4A8PreparedInput:
    """Route map and H-dimensional MXFP8 operands shared by compatible parts."""

    route_experts: torch.Tensor
    gate_quantized: MXFP8Rows
    up_quantized: MXFP8Rows
    m: int
    topk: int
    hidden_size: int
    num_experts: int
    shared_suh: bool
    coupled_hadamard: bool
    gate_suh: torch.Tensor
    up_suh: torch.Tensor


def make_trellis_w4a8_moe_scratch(
    *,
    m: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device | str,
    shared_suh: bool = True,
    coupled_hadamard: bool = False,
) -> TrellisW4A8MoeScratch:
    """Allocate the fixed buffers needed by :func:`run_trellis_w4a8_moe`."""

    m = int(m)
    topk = int(topk)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    shared_suh = bool(shared_suh)
    coupled_hadamard = bool(coupled_hadamard)
    if m <= 0 or topk <= 0:
        raise ValueError("m and topk must be positive")
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError("hidden_size must be a positive multiple of 128")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("intermediate_size must be a positive multiple of 128")
    if coupled_hadamard and not shared_suh:
        raise ValueError("coupled W4A8 requires shared_suh=True")
    device = torch.device(device)
    routes = m * topk
    input_rows = m if shared_suh else routes
    gate_fc1 = torch.empty(
        (routes, intermediate_size), dtype=torch.float16, device=device
    )
    up_fc1 = torch.empty_like(gate_fc1)
    fc2 = torch.empty(
        (routes, hidden_size), dtype=torch.float16, device=device
    )
    gate_quantized = empty_mxfp8_rows_for_dense_gemm(
        input_rows, hidden_size, device=device
    )
    up_quantized = (
        gate_quantized
        if coupled_hadamard
        else empty_mxfp8_rows_for_dense_gemm(
            input_rows, hidden_size, device=device
        )
    )
    return TrellisW4A8MoeScratch(
        route_experts=torch.empty((routes,), dtype=torch.int32, device=device),
        gate_quantized=gate_quantized,
        up_quantized=up_quantized,
        gate_fc1=gate_fc1,
        up_fc1=up_fc1,
        activated_quantized=empty_mxfp8_rows_for_dense_gemm(
            routes, intermediate_size, device=device
        ),
        fc2=fc2,
        output=torch.empty((m, hidden_size), dtype=torch.float32, device=device),
    )


def _mxfp8_rows_prefix(rows: MXFP8Rows, m: int, *, name: str) -> MXFP8Rows:
    if rows.values.ndim != 2 or int(rows.values.shape[0]) < m:
        raise ValueError(f"{name} does not have capacity for {m} rows")
    if rows.scale_rows.ndim != 3 or int(rows.scale_rows.shape[1]) < m:
        raise ValueError(f"{name}.scale_rows does not have capacity for {m} rows")
    m_tiles = (m + 127) // 128
    if rows.scale_mma.ndim != 6 or int(rows.scale_mma.shape[2]) < m_tiles:
        raise ValueError(f"{name}.scale_mma does not have capacity for {m} rows")
    return MXFP8Rows(
        values=rows.values[:m],
        scale_rows=rows.scale_rows[:, :m],
        scale_mma=rows.scale_mma[:, :, :m_tiles],
    )


def view_trellis_w4a8_moe_scratch(
    scratch: TrellisW4A8MoeScratch,
    *,
    m: int,
    topk: int,
    shared_suh: bool = True,
    coupled_hadamard: bool = False,
) -> TrellisW4A8MoeScratch:
    """Return allocation-free prefixes of a capacity-sized scratch binding."""

    m = int(m)
    topk = int(topk)
    shared_suh = bool(shared_suh)
    coupled_hadamard = bool(coupled_hadamard)
    if m <= 0 or topk <= 0:
        raise ValueError("m and topk must be positive")
    routes = m * topk
    input_rows = m if shared_suh else routes

    def prefix(name: str, tensor: torch.Tensor, rows: int) -> torch.Tensor:
        if tensor.ndim == 0 or int(tensor.shape[0]) < rows:
            raise ValueError(f"scratch.{name} does not have capacity for {rows} rows")
        return tensor[:rows]
    gate_quantized = _mxfp8_rows_prefix(
        scratch.gate_quantized, input_rows, name="scratch.gate_quantized"
    )
    up_quantized = (
        gate_quantized
        if coupled_hadamard
        else _mxfp8_rows_prefix(
            scratch.up_quantized, input_rows, name="scratch.up_quantized"
        )
    )

    return TrellisW4A8MoeScratch(
        route_experts=prefix("route_experts", scratch.route_experts, routes),
        gate_quantized=gate_quantized,
        up_quantized=up_quantized,
        gate_fc1=prefix("gate_fc1", scratch.gate_fc1, routes),
        up_fc1=prefix("up_fc1", scratch.up_fc1, routes),
        activated_quantized=_mxfp8_rows_prefix(
            scratch.activated_quantized,
            routes,
            name="scratch.activated_quantized",
        ),
        fc2=prefix("fc2", scratch.fc2, routes),
        output=prefix("output", scratch.output, m),
    )


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_mxfp8_rows(
    name: str,
    rows: MXFP8Rows,
    *,
    m: int,
    k: int,
    device: torch.device,
) -> None:
    _check_tensor(
        f"{name}.values",
        rows.values,
        shape=(m, k),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _check_tensor(
        f"{name}.scale_rows",
        rows.scale_rows,
        shape=(1, m, k // 32),
        dtype=torch.float8_e8m0fnu,
        device=device,
    )
    expected_scale_mma = (
        32,
        4,
        (m + 127) // 128,
        4,
        (k + 127) // 128,
        1,
    )
    if tuple(rows.scale_mma.shape) != expected_scale_mma:
        raise ValueError(
            f"{name}.scale_mma must have shape {expected_scale_mma}, "
            f"got {tuple(rows.scale_mma.shape)}"
        )
    if rows.scale_mma.dtype != torch.float8_e8m0fnu:
        raise TypeError(
            f"{name}.scale_mma must have dtype torch.float8_e8m0fnu, "
            f"got {rows.scale_mma.dtype}"
        )
    if rows.scale_mma.device != device:
        raise ValueError(
            f"{name}.scale_mma must be on {device}, got {rows.scale_mma.device}"
        )


def _validate_prepared_part(
    prepared: object,
    *,
    hidden_size: int,
    device: torch.device,
) -> tuple[int, bool]:
    if int(prepared.hidden_size) != hidden_size:
        raise ValueError(
            f"source hidden size {hidden_size} != prepared {prepared.hidden_size}"
        )
    intermediate_size = int(prepared.intermediate_size)
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("W4A8 requires intermediate_size divisible by 128")
    trellis_codebook = str(getattr(prepared, "trellis_codebook", "")).lower()
    if trellis_codebook != "sqg_xor_cheb_t12":
        raise NotImplementedError(
            "native W4A8 trellis execution requires SQG-XOR-Cheb-T12, "
            f"got {trellis_codebook!r}"
        )
    activation = str(getattr(prepared, "activation", "")).lower()
    if activation not in {"silu", "situ"}:
        raise ValueError(
            "native W4A8 trellis execution requires activation 'silu' or "
            f"'situ', got {activation!r}"
        )
    coupled_hadamard = bool(getattr(prepared, "coupled_hadamard", False))
    if coupled_hadamard and activation != "situ":
        raise ValueError("coupled W4A8 requires activation 'situ'")
    if coupled_hadamard:
        if hidden_size % 512:
            raise ValueError("coupled W4A8 requires hidden_size divisible by 512")
        if prepared.gate_suh.data_ptr() != prepared.up_suh.data_ptr():
            raise ValueError("coupled W4A8 requires one shared gate/up input scale")
        if int(prepared.intermediate_rotations.shape[1]) != 6 * intermediate_size:
            raise ValueError(
                "coupled W4A8 requires six intermediate rotation vectors"
            )
    pair_fc1 = getattr(prepared, "fc1_trellis_pair_kind", None) is not None
    pair_fc2 = getattr(prepared, "fc2_trellis_pair_kind", None) is not None
    if pair_fc1 != pair_fc2:
        raise ValueError("W4A8 requires both matrices to use pair or uniform storage")
    if coupled_hadamard and pair_fc1:
        raise ValueError("coupled W4A8 requires native fixed-rate storage")
    num_experts = int(prepared.num_experts)
    for name in ("gate_suh", "up_suh", "down_svh"):
        scale = getattr(prepared, name, None)
        if (
            not isinstance(scale, torch.Tensor)
            or scale.ndim != 2
            or int(scale.shape[0]) not in (1, num_experts)
            or int(scale.shape[1]) != hidden_size
            or scale.dtype != torch.float16
            or scale.device != device
            or not scale.is_contiguous()
        ):
            raise ValueError(
                f"prepared.{name} must be contiguous fp16 [1|E,{hidden_size}] "
                f"on {device}"
            )
    shared_suh = getattr(prepared, "shared_suh", None)
    if type(shared_suh) is not bool:
        raise ValueError("prepared weights omit the explicit shared_suh contract")
    gate_suh_rows = int(prepared.gate_suh.shape[0])
    up_suh_rows = int(prepared.up_suh.shape[0])
    expected_suh_rows = 1 if shared_suh else num_experts
    if gate_suh_rows != expected_suh_rows or up_suh_rows != expected_suh_rows:
        raise ValueError(
            "prepared gate_suh/up_suh rows disagree with shared_suh; "
            f"got {gate_suh_rows}, {up_suh_rows}, expected {expected_suh_rows}"
        )
    if coupled_hadamard and not shared_suh:
        raise ValueError("coupled W4A8 requires shared_suh=True")
    return num_experts, shared_suh


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.data_ptr() == right.data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
        and left.device == right.device
    )


def _tensor_views_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    left_start = int(left.data_ptr())
    right_start = int(right.data_ptr())
    left_end = left_start + left.numel() * left.element_size()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def _validate_prepared_parts(
    source: torch.Tensor,
    prepared_parts: tuple[object, ...],
) -> tuple[int, bool]:
    if not prepared_parts:
        raise ValueError("prepared_parts must contain at least one W4A8 part")
    if source.ndim != 2 or source.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("source must be contiguous BF16/FP16 [M,H]")
    if not source.is_contiguous():
        raise ValueError("source must be contiguous")
    hidden_size = int(source.shape[1])
    first = prepared_parts[0]
    num_experts, shared_suh = _validate_prepared_part(
        first,
        hidden_size=hidden_size,
        device=source.device,
    )
    first_intermediate_size = int(first.intermediate_size)
    coupled_hadamard = bool(getattr(first, "coupled_hadamard", False))
    for index, prepared in enumerate(prepared_parts[1:], start=1):
        part_num_experts, part_shared_suh = _validate_prepared_part(
            prepared,
            hidden_size=hidden_size,
            device=source.device,
        )
        if part_num_experts != num_experts:
            raise ValueError(
                "W4A8 prepared parts have incompatible expert geometry: "
                f"part 0 has {num_experts}, part {index} has {part_num_experts}"
            )
        if part_shared_suh is not shared_suh:
            raise ValueError("W4A8 prepared parts disagree on shared_suh")
        if int(prepared.intermediate_size) != first_intermediate_size:
            raise ValueError("W4A8 prepared parts disagree on intermediate_size")
        if bool(getattr(prepared, "coupled_hadamard", False)) != coupled_hadamard:
            raise ValueError("W4A8 prepared parts disagree on coupled_hadamard")
        for name in ("gate_suh", "up_suh", "down_svh"):
            if not _same_tensor_view(getattr(first, name), getattr(prepared, name)):
                raise ValueError(
                    f"W4A8 prepared part {index} does not share {name} identity"
                )
    return num_experts, shared_suh


def prepare_trellis_w4a8_moe_input(
    source: torch.Tensor,
    prepared: object,
    topk_ids: torch.Tensor,
    scratch: TrellisW4A8MoeScratch,
    *,
    expert_map: torch.Tensor | None = None,
) -> TrellisW4A8PreparedInput:
    """Materialize routes and quantize the shared H-dimensional operands once."""

    num_experts, shared_suh = _validate_prepared_parts(source, (prepared,))
    coupled_hadamard = bool(getattr(prepared, "coupled_hadamard", False))
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be contiguous int32/int64 [M,top-k]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.device != source.device:
        raise ValueError("source and routing tensors must share one CUDA device")
    m, hidden_size = map(int, source.shape)
    topk = int(topk_ids.shape[1])
    if int(topk_ids.shape[0]) != m:
        raise ValueError(
            "topk_ids must have one row per source token, "
            f"got {topk_ids.shape[0]} and {m}"
        )
    routes = m * topk
    input_rows = m if shared_suh else routes
    _check_tensor(
        "scratch.route_experts",
        scratch.route_experts,
        shape=(routes,),
        dtype=torch.int32,
        device=source.device,
    )
    _check_mxfp8_rows(
        "scratch.gate_quantized",
        scratch.gate_quantized,
        m=input_rows,
        k=hidden_size,
        device=source.device,
    )
    _check_mxfp8_rows(
        "scratch.up_quantized",
        scratch.up_quantized,
        m=input_rows,
        k=hidden_size,
        device=source.device,
    )
    if expert_map is not None and (
        expert_map.ndim != 1
        or expert_map.dtype != torch.int32
        or expert_map.device != source.device
        or not expert_map.is_contiguous()
    ):
        raise TypeError(
            "expert_map must be contiguous int32 [global experts] on source device"
        )
    if coupled_hadamard:
        if expert_map is None:
            scratch.route_experts.copy_(topk_ids.reshape(-1))
        else:
            torch.index_select(
                expert_map,
                0,
                topk_ids.reshape(-1),
                out=scratch.route_experts,
            )
        run_trellis_w4a8_coupled_input_quant(
            source,
            prepared,
            scratch.gate_quantized,
        )
    else:
        run_trellis_w4a8_input_rotation_quant(
            source,
            topk_ids,
            scratch.route_experts,
            prepared,
            scratch.gate_quantized,
            scratch.up_quantized,
            expert_map=expert_map,
            topk=topk,
        )
    return TrellisW4A8PreparedInput(
        route_experts=scratch.route_experts,
        gate_quantized=scratch.gate_quantized,
        up_quantized=scratch.up_quantized,
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        shared_suh=shared_suh,
        coupled_hadamard=coupled_hadamard,
        gate_suh=prepared.gate_suh,
        up_suh=prepared.up_suh,
    )


def run_trellis_w4a8_moe_prepared(
    prepared_input: TrellisW4A8PreparedInput,
    prepared: object,
    topk_weights: torch.Tensor,
    scratch: TrellisW4A8MoeScratch,
    *,
    fast_math: bool = False,
) -> torch.Tensor:
    """Run one compact E4M3 trellis part from a prepared routed input."""

    m = prepared_input.m
    topk = prepared_input.topk
    hidden_size = prepared_input.hidden_size
    routes = m * topk
    intermediate_size = int(prepared.intermediate_size)
    num_experts, shared_suh = _validate_prepared_part(
        prepared,
        hidden_size=hidden_size,
        device=prepared_input.route_experts.device,
    )
    if (
        num_experts != prepared_input.num_experts
        or shared_suh is not prepared_input.shared_suh
        or bool(getattr(prepared, "coupled_hadamard", False))
        is not prepared_input.coupled_hadamard
    ):
        raise ValueError("prepared part has incompatible route/input geometry")
    for name in ("gate_suh", "up_suh"):
        if not _same_tensor_view(
            getattr(prepared_input, name),
            getattr(prepared, name),
        ):
            raise ValueError(f"prepared part does not share {name} identity")
    if topk_weights.shape != (m, topk) or topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be contiguous FP32 with routing shape")
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    if topk_weights.device != prepared_input.route_experts.device:
        raise ValueError("prepared input and topk_weights must share one CUDA device")
    if (
        scratch.route_experts is not prepared_input.route_experts
        or scratch.gate_quantized is not prepared_input.gate_quantized
        or scratch.up_quantized is not prepared_input.up_quantized
    ):
        raise ValueError("prepared input must be consumed with its source scratch")
    _check_mxfp8_rows(
        "scratch.activated_quantized",
        scratch.activated_quantized,
        m=routes,
        k=intermediate_size,
        device=prepared_input.route_experts.device,
    )
    for name, tensor, shape in (
        ("gate_fc1", scratch.gate_fc1, (routes, intermediate_size)),
        ("up_fc1", scratch.up_fc1, (routes, intermediate_size)),
        ("fc2", scratch.fc2, (routes, hidden_size)),
    ):
        _check_tensor(
            f"scratch.{name}",
            tensor,
            shape=shape,
            dtype=torch.float16,
            device=prepared_input.route_experts.device,
        )
    _check_tensor(
        "scratch.output",
        scratch.output,
        shape=(m, hidden_size),
        dtype=torch.float32,
        device=prepared_input.route_experts.device,
    )
    coupled_hadamard = bool(getattr(prepared, "coupled_hadamard", False))
    pair_fc1 = getattr(prepared, "fc1_trellis_pair_kind", None) is not None
    pair_fc2 = getattr(prepared, "fc2_trellis_pair_kind", None) is not None
    if coupled_hadamard and scratch.gate_quantized is not scratch.up_quantized:
        raise ValueError(
            "coupled W4A8 scratch must share one quantized FC1 operand"
        )
    if pair_fc1:
        run_trellis_w4a8_fc1_routes(
            prepared_input.gate_quantized,
            prepared_input.up_quantized,
            prepared,
            prepared_input.route_experts,
            scratch.gate_fc1,
            scratch.up_fc1,
            topk=topk,
            shared_input=shared_suh,
            decode_both=False,
            sqg_direct_lut=True,
        )
    else:
        run_trellis_w4a8_uniform_fc1(
            prepared_input.gate_quantized,
            prepared_input.up_quantized,
            prepared,
            prepared_input.route_experts,
            scratch.gate_fc1,
            scratch.up_fc1,
            topk=topk,
            shared_input=shared_suh,
        )
    if coupled_hadamard:
        run_trellis_w4a8_coupled_activation_quant(
            scratch.gate_fc1,
            scratch.up_fc1,
            prepared_input.route_experts,
            prepared,
            scratch.activated_quantized,
            fast_math=fast_math,
        )
    else:
        run_trellis_w4a8_activation_rotation_quant(
            scratch.gate_fc1,
            scratch.up_fc1,
            prepared_input.route_experts,
            prepared,
            scratch.activated_quantized,
            fast_math=fast_math,
        )
    if pair_fc2:
        run_trellis_w4a8_fc2_routes(
            scratch.activated_quantized,
            prepared,
            prepared_input.route_experts,
            scratch.fc2,
            sqg_direct_lut=True,
        )
    else:
        run_trellis_w4a8_uniform_fc2(
            scratch.activated_quantized,
            prepared,
            prepared_input.route_experts,
            scratch.fc2,
        )
    _w4a16_topk_sum_launch_flat(
        scratch.fc2,
        scratch.output,
        m,
        topk,
        hidden_size,
        "fp16",
        int(cuda_stream_to_int(torch.cuda.current_stream())),
        full_rotation=True,
        num_experts=num_experts,
        topk_weights=topk_weights,
        route_expert_ids=prepared_input.route_experts,
        svh_table=prepared.down_svh,
        coupled_hadamard=coupled_hadamard,
    )
    return scratch.output


def run_trellis_w4a8_moe_parts(
    source: torch.Tensor,
    prepared_parts: tuple[object, ...],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scratch: TrellisW4A8MoeScratch,
    *,
    output_accum: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    fast_math: bool = False,
) -> torch.Tensor:
    """Run compatible trellis parts with one shared-input preparation."""

    _validate_prepared_parts(source, prepared_parts)
    m, hidden_size = map(int, source.shape)
    if topk_ids.ndim != 2:
        raise TypeError("topk_ids must be contiguous int32/int64 [M,top-k]")
    expected_routing_shape = (m, int(topk_ids.shape[1]))
    if (
        topk_weights.shape != expected_routing_shape
        or topk_weights.dtype != torch.float32
    ):
        raise TypeError("topk_weights must be contiguous FP32 with routing shape")
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    if topk_weights.device != source.device:
        raise ValueError("source and routing tensors must share one CUDA device")
    if len(prepared_parts) > 1 and output_accum is None:
        raise ValueError("multipart W4A8 execution requires output_accum")
    if output_accum is not None:
        _check_tensor(
            "output_accum",
            output_accum,
            shape=(m, hidden_size),
            dtype=torch.float32,
            device=source.device,
        )
        if len(prepared_parts) > 1 and _tensor_views_overlap(
            output_accum, scratch.output
        ):
            raise ValueError("multipart output_accum must not overlap scratch.output")
    prepared_input = prepare_trellis_w4a8_moe_input(
        source,
        prepared_parts[0],
        topk_ids,
        scratch,
        expert_map=expert_map,
    )
    part_output = None
    for index, prepared in enumerate(prepared_parts):
        part_output = run_trellis_w4a8_moe_prepared(
            prepared_input,
            prepared,
            topk_weights,
            scratch,
            fast_math=fast_math,
        )
        if output_accum is not None:
            if index == 0:
                output_accum.copy_(part_output)
            else:
                output_accum.add_(part_output)
    if part_output is None:
        raise RuntimeError("prepared W4A8 partition is empty")
    return output_accum if output_accum is not None else part_output


def run_trellis_w4a8_moe(
    source: torch.Tensor,
    prepared: object,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scratch: TrellisW4A8MoeScratch,
    *,
    expert_map: torch.Tensor | None = None,
    fast_math: bool = False,
) -> torch.Tensor:
    """Run one compact E4M3 trellis part through native W4A8 MMA."""

    return run_trellis_w4a8_moe_parts(
        source,
        (prepared,),
        topk_weights,
        topk_ids,
        scratch,
        expert_map=expert_map,
        fast_math=fast_math,
    )


__all__ = [
    "TrellisW4A8MoeScratch",
    "TrellisW4A8PreparedInput",
    "make_trellis_w4a8_moe_scratch",
    "prepare_trellis_w4a8_moe_input",
    "run_trellis_w4a8_moe",
    "run_trellis_w4a8_moe_parts",
    "run_trellis_w4a8_moe_prepared",
    "view_trellis_w4a8_moe_scratch",
]
