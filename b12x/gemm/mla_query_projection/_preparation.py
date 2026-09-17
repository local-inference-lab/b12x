"""Declarative MLA query-projection preparation."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._tuning import ProjectionQuery, TUNING


def compile_bf16(query_payload, device_ordinal):
    from . import _bf16
    query = ProjectionQuery(**query_payload)
    output_fp8 = query.output_dtype == "float8_e4m3fn"
    with torch.cuda.device(device_ordinal):
        return _bf16._mla_query_projection_bf16_kernel.warmup(
            torch.bfloat16, torch.bfloat16, torch.bfloat16,
            torch.float32 if output_fp8 else torch.bfloat16,
            torch.float8_e4m3fn if output_fp8 else torch.bfloat16,
            query.max_rows, query.max_rows * 192, 192, 192 * 512, 512,
            query.heads * 64, 64, query.heads * 576, 576,
            OUTPUT_FP8=output_fp8, NOPE_DIM=192, LATENT_DIM=512, ROPE_DIM=64,
            BLOCK_M=_bf16.block_m(query.max_rows), BLOCK_N=32, BLOCK_K=64,
            num_warps=4, num_stages=2, grid=(16, query.heads, 1),
        )


def compile_mxfp8(query_payload, device_ordinal):
    from b12x.gemm._shared import mxfp8_bmm as kernels
    query = ProjectionQuery(**query_payload)
    return kernels._compile_mla_query_projection(
        b_major=kernels._coerce_b_major(query.b_major), groups=query.heads,
        m=query.max_rows, n=512, k=192,
        output_fp8=query.output_dtype == "float8_e4m3fn",
        device=torch.device("cuda", device_ordinal),
    )


@dataclass(frozen=True)
class _ProjectionExecutionState:
    query: ProjectionQuery
    device: torch.device
    launch: object

    def run(self, q_nope, weight, q_pe, out, *, q_scale=None, stream=None):
        if q_nope.device != self.device:
            raise ValueError(
                f"MLA query plan is prepared on {self.device}, got operands on {q_nope.device}"
            )
        rows = int(q_nope.shape[1]) if q_nope.ndim == 3 else None
        if rows is not None and rows not in self.query.served_rows:
            if self.query.weight_format == "mxfp8":
                raise ValueError(
                    f"MXFP8 MLA query plan is compiled for exact M={self.query.max_rows}, got M={rows}; "
                    "declare a plan with max_rows equal to the live token count"
                )
            raise ValueError(f"BF16 MLA query plan serves 1<=M<={self.query.max_rows}, got M={rows}")
        if self.query.weight_format == "mxfp8":
            from b12x.gemm._shared import mxfp8_bmm as kernels
            if not isinstance(weight, tuple):
                raise ValueError("prepared MXFP8 MLA plan requires native MXFP8 weight")
            return kernels._run_mla_query_prepared(q_nope, *kernels._rhs_tensors(weight), q_pe, q_scale, out,
                b_major=self.query.b_major, sf_axis=self.query.sf_axis, launch=self.launch, stream=stream)
        if not isinstance(weight, torch.Tensor):
            raise ValueError("prepared BF16 MLA plan requires BF16 weight")
        from . import _bf16
        return _bf16._run_prepared(
            q_nope, weight, q_pe, q_scale, out, launcher=self.launch,
            max_rows=self.query.max_rows,
            output_fp8=self.query.output_dtype == "float8_e4m3fn", stream=stream,
        )


def plan(query: ProjectionQuery, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(query, ProjectionQuery): raise TypeError("MLA query plan requires ProjectionQuery")
    invocation = FrozenMapping(invocation)
    if invocation: raise ValueError("MLA invocation semantics belong in ProjectionQuery")
    def compile_jobs(config, device):
        del config
        factory = "b12x.gemm.mla_query_projection._preparation:compile_bf16" if query.weight_format == "bf16" else "b12x.gemm.mla_query_projection._preparation:compile_mxfp8"
        return (CompileJob.create(factory, TUNING.encode_query(query), device.ordinal),)
    def materialize(selection, device):
        compiler = compile_bf16 if query.weight_format == "bf16" else compile_mxfp8
        return _ProjectionExecutionState(query, torch.device("cuda", device.ordinal), compiler(TUNING.encode_query(query), device.ordinal))
    return Plan(contract=TUNING, query=query, invocation=invocation, override=override,
                _compile_jobs=compile_jobs, _memory_requirements=lambda config, device: MemoryRequirements(),
                _materialize=materialize)
