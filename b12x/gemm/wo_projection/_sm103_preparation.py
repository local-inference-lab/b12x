"""Retained CuTe packing launchers for SM103 WO preparation."""
from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import attach_programs


@dataclass(frozen=True)
class Quantizers:
    groups: int
    group_width: int
    rank: int
    grouped: object
    group_major: object

    def quantize_a(self, source, out):
        from ._quant_cute import quantize_wo_grouped_rows_cute
        quantize_wo_grouped_rows_cute(
            source, out.values, out.scale_rows, out.scale_mma,
            m=source.shape[0], groups=self.groups, group_width=self.group_width,
            compiled=self.grouped,
        )

    def quantize_a_inv_rope(self, source, positions, cos_sin, out, *,
                            groups, heads_per_group, nope_dim, rope_dim):
        from ._quant_cute import quantize_wo_grouped_rows_cute
        quantize_wo_grouped_rows_cute(
            source, out.values, out.scale_rows, out.scale_mma,
            m=source.shape[0], groups=groups, group_width=self.group_width,
            positions=positions, cos_sin_cache=cos_sin,
            head_dim=nope_dim + rope_dim, nope_dim=nope_dim, rope_dim=rope_dim,
            compiled=self.grouped,
        )

    def quantize_b(self, source, out):
        from ._quant_cute import quantize_wo_group_major_rows_cute
        quantize_wo_group_major_rows_cute(
            source, out.values, out.scale_rows, out.scale_mma,
            m=source.shape[0], groups=self.groups, rank=self.rank,
            compiled=self.group_major,
        )


def compile_quantizers(payload, ordinal):
    from ._quant_cute import _get_compiled_wo_quant
    q = dict(payload)
    inverse = q["operation"] == "inv_rope"
    with torch.cuda.device(ordinal):
        grouped = _get_compiled_wo_quant(
            "grouped", q["groups"] * q["group_width"], q["group_width"],
            getattr(torch, q["dtype"]), inverse,
            q["nope_dim"] + q["rope_dim"] if inverse else 0,
            q["nope_dim"] if inverse else 0, q["rope_dim"] if inverse else 0,
            getattr(torch, q["positions_dtype"]), getattr(torch, q["cos_sin_dtype"]),
            ordinal, "sm_103a",
        )
        group_major = _get_compiled_wo_quant(
            "group_major", q["groups"] * q["rank"], q["rank"],
            torch.bfloat16, False, 0, 0, 0, torch.int64, torch.bfloat16,
            ordinal, "sm_103a",
        )
    return attach_programs(
        Quantizers(q["groups"], q["group_width"], q["rank"], grouped, group_major),
        grouped, group_major,
    )
