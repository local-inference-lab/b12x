"""Opaque execution of prepared concatenated and block-FP8 MTP feedback."""
from __future__ import annotations

import torch
from b12x.preparation import plan_from_handle, require_prepared


@torch.library.custom_op("b12x::mtp_prepared_feedback", mutates_args=("scratch", "output"))
def _run(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor | None,
    hidden_fc_weight: torch.Tensor | None,
    combined_fc_weight: torch.Tensor | None,
    embedding_fc_scale: torch.Tensor | None,
    hidden_fc_scale: torch.Tensor | None,
    positions: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "sequence.mtp_feedback", token_embedding.device)
    weights = (dict(combined_fc_weight=combined_fc_weight)
               if state.layout.caps.contract == "rms_concat" else
               dict(embedding_fc_weight=embedding_fc_weight, hidden_fc_weight=hidden_fc_weight,
                    embedding_fc_scale=embedding_fc_scale, hidden_fc_scale=hidden_fc_scale))
    binding = state.bind(
        _plan=plan, token_embedding=token_embedding, multi_state=multi_state,
        token_norm_weight=token_norm_weight, state_norm_weight=state_norm_weight,
        positions=positions, scratch=scratch, output=output, **weights,
    )
    state.run(binding, eps=eps)


@_run.register_fake
def _fake(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor | None,
    hidden_fc_weight: torch.Tensor | None,
    combined_fc_weight: torch.Tensor | None,
    embedding_fc_scale: torch.Tensor | None,
    hidden_fc_scale: torch.Tensor | None,
    positions: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    return None


def run(binding, *, eps):
    torch.ops.b12x.mtp_prepared_feedback(
        binding.token_embedding, binding.multi_state, binding.token_norm_weight,
        binding.state_norm_weight, binding.embedding_fc_weight, binding.hidden_fc_weight,
        binding.combined_fc_weight, binding.embedding_fc_scale, binding.hidden_fc_scale,
        binding.positions, binding.scratch, binding.output, eps, binding.plan.handle,
    )
    return binding.output
