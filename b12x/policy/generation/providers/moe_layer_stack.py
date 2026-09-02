"""Generic transformer layers around a fused MoE decode candidate.

Each layer: an MXFP8 attention-out neighbour writes the MoE input, the
router's top-k ids/weights are refreshed in-graph (host-synthesized route
patterns, L2-warm like a real router output), the fused MoE runs as the
tested slot on that layer's own expert weights (so weights stream cold),
and a closing MXFP8 neighbour plus residual/RMSNorm hand the hidden state
to the next layer.
"""

from __future__ import annotations

from .layer_stack import GenericLayerStack


class _MoeLayerStack(GenericLayerStack):
    def __init__(
        self,
        *,
        geometry,
        experts_layers,
        tokens: int,
        top_k: int,
        x_dtype,
        output_dtype,
        topk_ids,
        topk_weights,
        device,
        generator,
    ) -> None:
        import torch

        hidden = int(geometry.hidden_size)
        super().__init__(
            hidden=hidden,
            in_width=hidden,
            out_width=hidden,
            tokens=tokens,
            layers=len(experts_layers),
            device=device,
            generator=generator,
        )
        self.geometry = geometry
        self.experts_layers = list(experts_layers)
        self.top_k = int(top_k)
        self.ids_src = topk_ids
        self.weights_src = topk_weights
        self.x = [
            torch.empty((self.tokens, hidden), dtype=x_dtype, device=device)
            for _ in range(self.layers)
        ]
        self.outputs = [
            torch.empty((self.tokens, hidden), dtype=output_dtype, device=device)
            for _ in range(self.layers)
        ]
        self.topk_ids = [topk_ids.clone() for _ in range(self.layers)]
        self.topk_weights = [topk_weights.clone() for _ in range(self.layers)]
        self.bindings: dict[str, tuple[list, list]] = {}

    def prepare_candidate(self, candidate_id: str, *, policy, capacity) -> None:
        """Bind the candidate's plan on every layer's own experts."""
        import torch

        from b12x.moe import fused_moe

        if candidate_id in self.bindings:
            return
        bindings, owners = [], []
        for layer in range(self.layers):
            plan = fused_moe.plan_execution(
                experts=self.experts_layers[layer],
                capacity=capacity,
                policy=policy,
            )
            fused_moe.prewarm(plan)
            scratch = {
                spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                for spec in plan.scratch_specs()
            }
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=self.x[layer],
                experts=self.experts_layers[layer],
                topk_weights=self.topk_weights[layer],
                topk_ids=self.topk_ids[layer],
                output=self.outputs[layer],
                input_scales_static=True,
            )
            bindings.append(binding)
            owners.append((plan, scratch))
        self.bindings[candidate_id] = (bindings, owners)

    def release_candidate(self, candidate_id: str) -> None:
        self.bindings.pop(candidate_id, None)

    def produce(self, layer: int, activation) -> None:
        self.x[layer].copy_(activation)
        self.topk_ids[layer].copy_(self.ids_src)
        self.topk_weights[layer].copy_(self.weights_src)

    def tested(self, layer: int, slot):
        from b12x.moe import fused_moe

        fused_moe.run(binding=self.bindings[slot][0][layer])
        return self.outputs[layer]

    def consume(self, layer: int, output):
        import torch

        if output.dtype == torch.bfloat16:
            return output
        return output.to(torch.bfloat16)


__all__ = ["_MoeLayerStack"]
