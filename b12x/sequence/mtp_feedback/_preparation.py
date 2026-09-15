"""Complete capacity-shaped MTP projection and normalization preparation."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.compile_plan import attach_programs
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._impl import Caps, _bind, _materialize_layout
from ._tuning import MtpFeedbackConfig, MtpFeedbackQuery, TUNING


def _alignment(tensor):
    pointer = tensor.data_ptr()
    return min(16, pointer & -pointer) if pointer else 16


def invocation_from_tensors(*, token_embedding, multi_state, token_norm_weight, state_norm_weight, **unused):
    return FrozenMapping({"input_alignments": tuple(_alignment(tensor) for tensor in (
        token_embedding, multi_state, token_norm_weight, state_norm_weight,
    ))})


@dataclass(frozen=True)
class _CompilePointer:
    dtype: torch.dtype
    alignment: int

    def data_ptr(self):
        return self.alignment


def compile_feedback(query_payload, config_payload, ordinal):
    from . import _kernels
    from ._cute_prefill import compile_mtp_prefill_capacity

    query = MtpFeedbackQuery(**dict(query_payload))
    config = MtpFeedbackConfig.from_config(FrozenMapping(config_payload))
    device = torch.device("cuda", ordinal)
    caps = Caps(device=device, max_tokens=query.max_tokens, hidden_size=query.hidden_size, streams=query.streams, contract=query.contract)
    if query.contract != "qwen_multistream":
        from b12x.preparation.device import detect_device
        from . import _concat, _fp8
        backend = _concat if query.contract == "rms_concat" else _fp8
        layout = backend.plan(caps, config, detect_device(device).identity)
        programs = layout._backend_plan
        dependencies = [programs.norms, programs.projection]
        if query.contract == "rms_streams_fp8":
            dependencies.extend((programs.quant, programs.add))
        return {"layout": attach_programs(layout, *dependencies)}
    layout = _materialize_layout(caps, config)
    embedding, multi_state, token_weight, state_weight = (
        _CompilePointer(torch.bfloat16, value) for value in query.input_alignments
    )
    with torch.cuda.device(device):
        projections = compile_mtp_prefill_capacity(
            layout.token_projection_rows, layout.state_projection_rows, caps.hidden_size,
            device=device, streams=caps.streams,
        )
        token_norm = _kernels._token_norm_kernel.warmup(
            embedding, token_weight, torch.bfloat16, 1e-6,
            HIDDEN_SIZE=caps.hidden_size, BLOCK_H=config.norm_block_h,
            num_warps=config.norm_num_warps, num_stages=1, grid=(caps.max_tokens,),
        )
        partial = _kernels._state_partial_sum_kernel.warmup(
            multi_state, torch.float32, HIDDEN_SIZE=caps.hidden_size, BLOCK_H=config.norm_block_h,
            num_warps=config.norm_num_warps, num_stages=1, grid=(caps.max_tokens * caps.streams,),
        )
        state_norm = _kernels._state_norm_kernel.warmup(
            multi_state, torch.float32, state_weight, torch.bfloat16, 1e-6,
            STREAMS=caps.streams, HIDDEN_SIZE=caps.hidden_size,
            BLOCK_S=config.norm_block_s, BLOCK_H=config.norm_block_h,
            num_warps=config.norm_num_warps, num_stages=1, grid=(caps.max_tokens, caps.streams),
        )
        return {"projections": projections, "token_norm": token_norm, "partial": partial, "state_norm": state_norm}


@dataclass(frozen=True)
class _MtpState:
    query: MtpFeedbackQuery
    layout: object
    programs: object

    def _check(self, tensors):
        for tensor, alignment in zip(tensors, self.query.input_alignments):
            if tensor.device != self.layout.caps.device or tensor.dtype != torch.bfloat16:
                raise ValueError("MTP input dtype/device differs from preparation")
            if tensor.numel() and _alignment(tensor) != alignment:
                raise ValueError("MTP input alignment differs from preparation")

    def bind(self, **kwargs):
        if self.layout._backend_plan is not None:
            plan = kwargs.pop("_plan", None)
            binding = self.layout._backend_plan.bind(self.layout, **kwargs)
            return replace(binding, plan=plan)
        binding = _bind(self.layout, **kwargs)
        self._check((binding.token_embedding, binding.multi_state,
                     binding.token_norm_weight, binding.state_norm_weight))
        return binding

    def run(self, binding, *, eps=1e-6):
        if binding._state is not self.layout:
            raise ValueError("binding belongs to another prepared MTP layout")
        if self.layout._backend_plan is not None:
            if not math.isfinite(float(eps)) or float(eps) <= 0:
                raise ValueError("MTP epsilon must be finite and positive")
            return self.layout._backend_plan.run(binding, eps=eps)
        self.run_tensors(
            binding.token_embedding, binding.multi_state, binding.token_norm_weight,
            binding.state_norm_weight, binding.embedding_fc_weight, binding.hidden_fc_weight,
            binding.scratch, binding.output, eps=eps,
        )
        return binding.output

    def run_tensors(self, token_embedding, multi_state, token_norm_weight, state_norm_weight,
                    embedding_fc_weight, hidden_fc_weight, scratch, output, *, eps):
        from ._kernels import _launch_mtp_feedback
        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("MTP epsilon must be finite and positive")
        self._check((token_embedding, multi_state, token_norm_weight, state_norm_weight))
        with torch.cuda.device(self.layout.caps.device):
            _launch_mtp_feedback(
                token_embedding, multi_state, token_norm_weight, state_norm_weight,
                embedding_fc_weight, hidden_fc_weight, scratch, output, eps, self,
            )


def make_plan(caps, *, invocation, override):
    invocation = FrozenMapping(invocation)
    if set(invocation) - {"input_alignments"}:
        raise ValueError("unknown MTP invocation metadata")
    query = MtpFeedbackQuery(
        dtype=str(caps.dtype).removeprefix("torch."), max_tokens=caps.max_tokens,
        hidden_size=caps.hidden_size, streams=caps.streams, contract=caps.contract, **dict(invocation),
    )
    layouts = {}

    def layout(config, device):
        if config not in layouts:
            if caps.contract == "qwen_multistream":
                layouts[config] = _materialize_layout(caps, config)
            else:
                from . import _concat, _fp8
                backend = _concat if caps.contract == "rms_concat" else _fp8
                layouts[config] = backend.plan(caps, config, device.identity, compile_launches=False)
        return layouts[config]

    def compile_jobs(config, device):
        return (CompileJob.create(
            "b12x.sequence.mtp_feedback._preparation:compile_feedback",
            TUNING.encode_query(query), config.to_dict(), device.ordinal,
        ),)

    def memory(config, device):
        return MemoryRequirements(scratch=layout(config, device).scratch_specs())

    def materialize(selection, device):
        programs = compile_feedback(TUNING.encode_query(query), selection.config.to_dict(), device.ordinal)
        return attach_programs(_MtpState(query, programs.get("layout") or layout(selection.config, device), MappingProxyType(programs)), programs)

    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=compile_jobs, _memory_requirements=memory, _materialize=materialize,
        _device=caps.device,
    )
