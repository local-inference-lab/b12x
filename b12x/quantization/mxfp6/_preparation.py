"""Session-prepared native MX-FP6 dense plan."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.fp6 import as_grouped_mxfp6_scale_view
from b12x._lib.intrinsics import align_up
from b12x.preparation import (
    FrozenMapping, MemoryRequirements, PersistentMemory, Plan,
    current_plan, current_prepared_state,
)
from ._tuning import Mxfp6DenseConfig, Mxfp6DenseQuery, TUNING
from b12x.preparation.types import _owned_tensor_nbytes


def _storage_nbytes(m: int, k: int) -> int:
    padded_m = align_up(m, 128)
    source_nbytes = 0 if m <= 16 else padded_m * k * 2
    return (
        padded_m * k
        + padded_m * align_up(k // 32, 4)
        + 4
        + padded_m * (4 + 2)
        + source_nbytes
    )


def _lowering(query, device):
    from b12x._lib.dense_gemm import _lower_dense_gemm
    m, k, n = query.max_tokens, query.in_features, query.out_features
    a_dtype = torch.float8_e4m3fn if query.activation_format == "e4m3" else torch.uint8
    a = torch.empty_strided((m, k, 1), (k, 1, m * k), dtype=a_dtype, device="meta")
    b_k = 3 * k // 4 if query.weight_storage == "packed" else k
    b = torch.empty_strided((n, b_k, 1), (b_k, 1, n * b_k), dtype=torch.uint8, device="meta")
    sfa = as_grouped_mxfp6_scale_view(torch.empty((1, align_up(m, 128) * align_up(k // 32, 4)), dtype=torch.uint8, device="meta"), m, k)
    sfb = as_grouped_mxfp6_scale_view(torch.empty((1, align_up(n, 128) * align_up(k // 32, 4)), dtype=torch.uint8, device="meta"), n, k)
    out = torch.empty_strided((m, n, 1), (n, 1, m * n), dtype=torch.bfloat16, device="meta")
    row_scale = torch.empty((m,), dtype=torch.bfloat16, device="meta") if query.per_row_global_scale else None
    return _lower_dense_gemm(
        (a, sfa), (b, sfb), out, ab_dtype=f"float6_{query.weight_format}fn",
        sf_dtype="float8_e8m0fnu", c_dtype="bfloat16", sf_vec_size=32,
        sm_count=device.sm_count, expected_m=m,
        alpha=torch.empty((1,), dtype=torch.float32, device="meta"),
        a_preexpanded=True, b_preexpanded=query.weight_storage == "expanded",
        b_packed=query.weight_storage == "packed",
        a_fmt=query.activation_format, b_fmt=query.weight_format, row_scale=row_scale,
    )


@dataclass(frozen=True)
class _CompiledMxfp6Dense:
    lowering: object
    dense_programs: object
    quantize: object
    row_gs: object | None


def compile_mxfp6_dense(query_payload, config_payload, ordinal, sm_count):
    """Compile every launcher consumed by one exact prepared FP6 invocation."""
    from b12x._lib import dense_gemm as dense
    from . import compile_bf16_to_fp6_tma
    from .bf16_to_fp6_small_m import compile_bf16_to_fp6_small_m
    from .fp6_row_gs import compile_fp6_row_gs
    query = Mxfp6DenseQuery(**dict(query_payload))
    config = Mxfp6DenseConfig(**dict(config_payload))
    del config
    class Device:
        def __init__(self): self.sm_count = sm_count
    lowering = _lowering(query, Device())
    padded_m = align_up(query.max_tokens, 128)
    with torch.cuda.device(ordinal):
        programs = dense._compile_dense_lowering(lowering.to_dict(), ordinal)
        if query.max_tokens <= 16:
            quantize = compile_bf16_to_fp6_small_m(query.max_tokens, query.in_features, query.activation_format, query.per_row_global_scale)
            row_gs = None
        else:
            quantize = compile_bf16_to_fp6_tma(padded_m, query.in_features, query.activation_format, "bytes", query.per_row_global_scale)
            row_gs = compile_fp6_row_gs(padded_m, query.in_features, query.activation_format) if query.per_row_global_scale else None
    return attach_programs(
        _CompiledMxfp6Dense(lowering, programs, quantize, row_gs),
        programs, quantize, row_gs,
    )


@dataclass(frozen=True)
class _Mxfp6DenseState:
    query: Mxfp6DenseQuery
    device: torch.device
    dense: object
    quantize: object
    row_gs: object | None
    codes: torch.Tensor
    scales: torch.Tensor
    alpha: torch.Tensor
    row_scales: torch.Tensor
    inverse_row_scales: torch.Tensor
    source_storage: torch.Tensor | None

    def run(self, x: torch.Tensor, weight: torch.Tensor, weight_scales: torch.Tensor,
            global_scale: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        if (x.shape != (self.query.max_tokens, self.query.in_features) or x.dtype != torch.bfloat16
                or x.device != self.device or not x.is_contiguous()):
            raise ValueError("FP6 source differs from prepared exact invocation")
        if weight.device != self.device or weight_scales.device != self.device or global_scale.device != self.device:
            raise ValueError("FP6 weight tensors differ from prepared device")
        if out is None:
            out = torch.empty((self.query.max_tokens, self.query.out_features, 1), dtype=torch.bfloat16, device=self.device)
        quant_source = x
        if self.source_storage is not None:
            self.source_storage[:self.query.max_tokens].copy_(x)
            if self.source_storage.shape[0] > self.query.max_tokens:
                self.source_storage[self.query.max_tokens:].zero_()
            quant_source = self.source_storage
        if self.query.max_tokens <= 16:
            self.quantize(
                quant_source, global_scale.reshape(1), self.codes.view(-1),
                self.scales.view(-1), self.alpha, self.inverse_row_scales,
            )
        elif self.row_gs is not None:
            self.row_gs(
                quant_source, global_scale.reshape(1), self.row_scales,
                self.inverse_row_scales, self.alpha,
            )
            self.quantize(
                quant_source, self.row_scales, self.codes.view(-1),
                self.scales.view(-1),
            )
        else:
            # The non per-row recipe retains the existing scalar scale arithmetic.
            from b12x._lib.fp6 import mx_gs_numerator
            scale = (
                mx_gs_numerator(self.query.activation_format)
                / quant_source.float().abs().amax().clamp_min_(1e-6).double()
            ).float().reshape(1)
            self.alpha.copy_(torch.reciprocal(scale * global_scale.reshape(1)))
            self.quantize(
                quant_source, scale, self.codes.view(-1), self.scales.view(-1),
            )
        a_values = self.codes[:self.query.max_tokens]
        if self.query.activation_format == "e4m3":
            a_values = a_values.view(torch.float8_e4m3fn)
        a_scales = as_grouped_mxfp6_scale_view(
            self.scales.view(1, -1), self.query.max_tokens, self.query.in_features
        )
        b_scales = as_grouped_mxfp6_scale_view(
            weight_scales.view(1, -1), self.query.out_features, self.query.in_features
        )
        self.dense.run(
            (a_values.unsqueeze(-1), a_scales),
            (weight if weight.ndim == 3 else weight.unsqueeze(-1), b_scales),
            out=out, alpha=self.alpha,
            row_scale=(
                self.inverse_row_scales[:self.query.max_tokens]
                if self.query.per_row_global_scale else None
            ),
        )
        return out[:, :, 0]


@dataclass(frozen=True)
class _CompiledMxfp6Sm103:
    scales: object
    quantize: object
    gemm: object


def compile_mxfp6_dense_sm103(query_payload, config_payload, ordinal):
    """Compile the native SM103 activation quantizer and FP6 GEMM for one invocation."""
    from b12x.gemm.blockscaled._fp6 import compile_kernel
    from . import _rows
    query = Mxfp6DenseQuery(**dict(query_payload))
    Mxfp6DenseConfig(**dict(config_payload))
    k, fmt, per_row = query.in_features, query.activation_format, query.per_row_global_scale
    packed = fmt != "e4m3"
    with torch.cuda.device(ordinal):
        scales = _rows.compile_scales(k, fmt, per_row, ordinal, "sm_103a")
        quantize = _rows.compile_quantizer(k, fmt, per_row, packed, ordinal, "sm_103a")
        gemm = compile_kernel(
            query.out_features, k, 1, fmt, query.weight_format, not packed,
            query.weight_storage == "expanded", "bfloat16", False, per_row, ordinal,
        )
    return attach_programs(_CompiledMxfp6Sm103(scales, quantize, gemm), scales, quantize, gemm)


def _sm103_storage_nbytes(query) -> int:
    m, k = query.max_tokens, query.in_features
    stored_k = k if query.activation_format == "e4m3" else 3 * k // 4
    return (
        m * stored_k + ((m + 127) // 128) * (k // 128) * 512
        + 4 * (m if query.per_row_global_scale else 1) + 2 * m + 4
    )


@dataclass(frozen=True)
class _Mxfp6Sm103State:
    """SM103 ``linear_with_workspace`` arithmetic through retained programs."""

    query: Mxfp6DenseQuery
    device: torch.device
    compiled: _CompiledMxfp6Sm103
    workspace: object

    def run(self, x: torch.Tensor, weight: torch.Tensor, weight_scales: torch.Tensor,
            global_scale: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        import cuda.bindings.driver as cuda
        import cutlass
        from b12x.gemm.blockscaled._fp6 import execute
        from b12x.gemm.blockscaled._sm103 import pointer
        q, w = self.query, self.workspace
        m, k, n = q.max_tokens, q.in_features, q.out_features
        if x.shape != (m, k) or x.dtype != torch.bfloat16 or x.device != self.device or not x.is_contiguous():
            raise ValueError("FP6 source differs from prepared exact invocation")
        if weight.ndim == 3 and weight.shape[-1] == 1:
            weight = weight[..., 0]
        stored_k = k if q.weight_storage == "expanded" else 3 * k // 4
        if tuple(weight.shape) != (n, stored_k) or weight.device != self.device:
            raise ValueError("FP6 weights differ from the prepared storage and device")
        if weight_scales.device != self.device or global_scale.device != self.device:
            raise ValueError("FP6 weight tensors differ from prepared device")
        if out is None:
            out = torch.empty((m, n, 1), dtype=torch.bfloat16, device=self.device)
        w._validate(x, global_scale)
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        self.compiled.scales(*(pointer(t, v) for t, v in zip(
            (cutlass.BFloat16, cutlass.Float32, cutlass.Float32, cutlass.BFloat16, cutlass.Float32),
            (x, global_scale, w.global_scales, w.inverse_scales, w.alpha), strict=True,
        )), cutlass.Int32(m), stream)
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.compiled.quantize(
            pointer(cutlass.BFloat16, x), pointer(cutlass.Float32, w.global_scales),
            pointer(cutlass.Uint8, w.values), pointer(cutlass.Uint8, w.scale_storage),
            cutlass.Int32(m), cutlass.Int32(min((m * (k // 32) + 127) // 128, sms * 4)), stream,
        )
        execute(
            (w.values[:m, :, None], w.scale_view(m)),
            (weight.reshape(n, stored_k, 1), as_grouped_mxfp6_scale_view(weight_scales.view(1, -1), n, k)),
            out, alpha=w.alpha, ab_dtype=f"float6_{q.weight_format}fn",
            sf_dtype="float8_e8m0fnu", sf_vec_size=32, c_dtype="bfloat16",
            a_fmt=q.activation_format, b_fmt=q.weight_format, a_preexpanded=not w.packed,
            b_preexpanded=q.weight_storage == "expanded", b_packed=q.weight_storage == "packed",
            row_scale=w.inverse_scales[:m] if q.per_row_global_scale else None,
            compiled=self.compiled.gemm,
        )
        return out[:, :, 0]


def _is_sm103(device) -> bool:
    return tuple(device.identity.compute_capability) == (10, 3)


def plan(query: Mxfp6DenseQuery, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(query, Mxfp6DenseQuery):
        raise TypeError("plan requires Mxfp6DenseQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("MX-FP6 invocation semantics belong in Mxfp6DenseQuery")
    def jobs(config, device):
        if _is_sm103(device):
            return (CompileJob.create("b12x.quantization.mxfp6._preparation:compile_mxfp6_dense_sm103", TUNING.encode_query(query), TUNING.encode_config(config), device.ordinal),)
        return (CompileJob.create("b12x.quantization.mxfp6._preparation:compile_mxfp6_dense", TUNING.encode_query(query), TUNING.encode_config(config), device.ordinal, device.identity.sm_count),)
    def memory(config, device):
        del config
        state = current_prepared_state()
        if _is_sm103(device):
            required, resident = _sm103_storage_nbytes(query), 0
            if isinstance(state, _Mxfp6Sm103State):
                w = state.workspace
                required = resident = _owned_tensor_nbytes((
                    w.values, w.scale_storage, w.global_scales, w.inverse_scales, w.alpha,
                ))
        else:
            required, resident = _storage_nbytes(query.max_tokens, query.in_features), 0
            if isinstance(state, _Mxfp6DenseState):
                required = resident = _owned_tensor_nbytes((
                    state.codes, state.scales, state.alpha, state.row_scales,
                    state.inverse_row_scales, state.source_storage,
                ))
        return MemoryRequirements(persistent=(PersistentMemory(
            ("quantization.mxfp6", current_plan()),
            required, resident,
        ),))
    def materialize(selection, device):
        target = torch.device("cuda", device.ordinal)
        if _is_sm103(device):
            from ._linear_workspace import allocate_fp6_linear_workspace
            compiled = compile_mxfp6_dense_sm103(
                TUNING.encode_query(query), TUNING.encode_config(selection.config), device.ordinal,
            )
            return _Mxfp6Sm103State(query, target, compiled, allocate_fp6_linear_workspace(
                query.max_tokens, query.in_features, device=target,
                act_fmt=query.activation_format, per_row=query.per_row_global_scale,
            ))
        compiled = compile_mxfp6_dense(
            TUNING.encode_query(query), TUNING.encode_config(selection.config),
            device.ordinal, device.identity.sm_count,
        )
        padded_m = align_up(query.max_tokens, 128)
        dense = __import__("b12x._lib.dense_gemm", fromlist=["_DenseExecutionState"])
        return _Mxfp6DenseState(
            query, target,
            dense._DenseExecutionState(
                compiled.lowering, target, compiled.dense_programs["gemm"],
                compiled.dense_programs.get("reduce"), None,
            ),
            compiled.quantize, compiled.row_gs,
            torch.zeros((padded_m, query.in_features), dtype=torch.uint8, device=target),
            torch.zeros((padded_m * align_up(query.in_features // 32, 4),), dtype=torch.uint8, device=target),
            torch.empty((1,), dtype=torch.float32, device=target),
            torch.empty((padded_m,), dtype=torch.float32, device=target),
            torch.empty((padded_m,), dtype=torch.bfloat16, device=target),
            None if query.max_tokens <= 16 else torch.zeros(
                (padded_m, query.in_features), dtype=torch.bfloat16, device=target,
            ))
    return Plan(contract=TUNING, query=query, invocation=invocation, override=override, _compile_jobs=jobs, _memory_requirements=memory, _materialize=materialize)


__all__ = ["Mxfp6DenseQuery", "Mxfp6DenseConfig", "TUNING", "compile_mxfp6_dense", "compile_mxfp6_dense_sm103", "plan"]
