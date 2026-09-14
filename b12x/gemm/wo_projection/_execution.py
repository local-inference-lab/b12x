"""Bound WO execution with caller-owned intermediates and native tcgen05 GEMMs."""

from __future__ import annotations

import torch

from b12x._lib.utils import cuda_stream_to_int


def _span(tensor):
    if tensor.numel() == 0:
        return tensor.data_ptr(), tensor.data_ptr()
    elements = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    return tensor.data_ptr(), tensor.data_ptr() + elements * tensor.element_size()


def validate_binding(source, weights, x_q, tmp, tmp_q, output, *, extra_reads=(), native=True):
    if native:
        from ._quant_cute import _grouped_source_stride
        if source.dtype != torch.bfloat16:
            raise ValueError("SM103 WO requires BF16 input")
        _grouped_source_stride(source, source.shape[0], weights.groups * weights.group_width)
    writes = (x_q.values, x_q.scale_rows, x_q.scale_mma, tmp,
              tmp_q.values, tmp_q.scale_rows, tmp_q.scale_mma, output)
    reads = (source, weights.wo_a.values, weights.wo_a.scale_rows, weights.wo_a.scale_mma,
             weights.wo_b.values, weights.wo_b.scale_rows, weights.wo_b.scale_mma, *extra_reads)
    reads += tuple(t for t in (weights.wo_a.values_tiled, weights.wo_b.values_tiled) if t is not None)
    if any(t.device != source.device for t in (*reads, *writes)):
        raise ValueError("WO inputs, weights, and scratch must share one device")
    for i, tensor in enumerate(writes):
        lo, hi = _span(tensor)
        if any(lo < _span(t)[1] and _span(t)[0] < hi for t in (*reads, *writes[:i])):
            raise ValueError("WO scratch and output must not overlap inputs or each other")


def run(binding, *, stream=None):
    inverse = hasattr(binding, "o")
    source = binding.o if inverse else binding.source_tgd
    if binding.backend != "mxfp8_tcgen05":
        raise ValueError("native WO execution requires the tcgen05 plan backend")
    _execute(
        source, binding.positions if inverse else None,
        binding.cos_sin_cache if inverse else None,
        binding.weights.wo_a.values, binding.weights.wo_a.scale_mma,
        binding.weights.wo_b.values, binding.weights.wo_b.scale_mma,
        binding.x_q.values, binding.x_q.scale_rows, binding.x_q.scale_mma,
        binding.tmp, binding.tmp_q.values, binding.tmp_q.scale_rows,
        binding.tmp_q.scale_mma, binding.output,
        binding.weights.groups, binding.weights.group_width, binding.weights.rank,
        binding.nope_dim if inverse else 0, binding.rope_dim if inverse else 0,
        cuda_stream_to_int(stream),
    )
    return binding.output if binding.return_3d else binding.output[:, :, 0]


@torch.library.custom_op(
    "b12x::wo_projection_tcgen05",
    mutates_args=("x_values", "x_rows", "x_mma", "tmp", "tmp_values",
                  "tmp_rows", "tmp_mma", "output"),
)
def _execute(
    source: torch.Tensor, positions: torch.Tensor | None, cos_sin: torch.Tensor | None,
    wa: torch.Tensor, wa_mma: torch.Tensor, wb: torch.Tensor, wb_mma: torch.Tensor,
    x_values: torch.Tensor, x_rows: torch.Tensor, x_mma: torch.Tensor,
    tmp: torch.Tensor, tmp_values: torch.Tensor, tmp_rows: torch.Tensor,
    tmp_mma: torch.Tensor, output: torch.Tensor,
    groups: int, group_width: int, rank: int, nope_dim: int, rope_dim: int,
    stream_int: int | None,
) -> None:
    from b12x.gemm.blockscaled._sm103 import execute
    from ._quant_cute import quantize_wo_grouped_rows_cute, quantize_wo_group_major_rows_cute

    m = source.shape[0]
    quantize_wo_grouped_rows_cute(
        source, x_values, x_rows, x_mma, m=m, groups=groups, group_width=group_width,
        positions=positions, cos_sin_cache=cos_sin, head_dim=nope_dim + rope_dim,
        nope_dim=nope_dim, rope_dim=rope_dim, stream=stream_int,
    )
    execute(
        (x_values.unsqueeze(-1) if x_values.ndim == 2 else x_values, x_mma),
        (wa.unsqueeze(-1) if wa.ndim == 2 else wa, wa_mma), tmp,
        ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16",
        sf_vec_size=32, stream=stream_int,
    )
    quantize_wo_group_major_rows_cute(
        tmp, tmp_values, tmp_rows, tmp_mma, m=m, groups=groups, rank=rank, stream=stream_int,
    )
    execute(
        (tmp_values.unsqueeze(-1), tmp_mma), (wb.unsqueeze(-1), wb_mma), output,
        ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16",
        sf_vec_size=32, stream=stream_int,
    )


@_execute.register_fake
def _execute_fake(source, positions, cos_sin, wa, wa_mma, wb, wb_mma,
                  x_values, x_rows, x_mma, tmp, tmp_values, tmp_rows, tmp_mma,
                  output, groups, group_width, rank, nope_dim, rope_dim, stream_int):
    return None
