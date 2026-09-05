"""Transformer-layer context for ranking mHC post/pre schedules.

The captured chain mirrors the DeepSeek V4 call order.  Each simulated
decoder layer runs an attention-like MXFP8 projection pair, mHC post/pre for
the FFN input, an FFN-like MXFP8 projection pair, and mHC post/pre for the
next layer's attention input.  Every boundary owns distinct mHC parameters
and every projection owns distinct weights, so replay establishes the cache
and launch context of a layer stack without an artificial L2 flush.
"""

from __future__ import annotations

from dataclasses import dataclass

MHC_CONTEXT_TRANSFORMER_LAYERS = 2
MHC_CONTEXT_BOUNDARIES = 2 * MHC_CONTEXT_TRANSFORMER_LAYERS


@dataclass(frozen=True)
class _MhcLayerParameters:
    fn: object
    fn_bf16: object | None
    scale: object
    bias: object
    norm_weight: object


def _packed_mxfp8_weight(hidden_size: int, *, device, generator):
    import torch

    from b12x.gemm import blockscaled

    values = torch.randn(
        (hidden_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(hidden_size**-0.5)
    values = values.to(torch.float8_e4m3fn)
    scales = torch.full(
        (hidden_size, hidden_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    return blockscaled.pack_weight(values, scales)


class MhcLayerStack:
    """Two decoder layers with four in-place mHC post/pre boundaries."""

    def __init__(
        self,
        *,
        mhc,
        initial_y,
        initial_residual,
        initial_post,
        initial_comb,
        tokens: int,
        hidden_size: int,
        device,
        generator,
    ) -> None:
        import torch

        from b12x.gemm import blockscaled

        self.mhc = mhc
        self.tokens = int(tokens)
        self.hidden_size = int(hidden_size)
        self.device = device
        self.boundaries = MHC_CONTEXT_BOUNDARIES
        self.transformer_layers = MHC_CONTEXT_TRANSFORMER_LAYERS
        self._blockscaled = blockscaled
        self.initial_y = initial_y.clone()
        self.initial_residual = initial_residual.clone()
        self.initial_post = initial_post.clone()
        self.initial_comb = initial_comb.clone()

        # Two projections stand in for attention or FFN work before each mHC
        # boundary.  A final pair consumes the last normalized activation.
        self.projection_weights = [
            (
                _packed_mxfp8_weight(
                    self.hidden_size,
                    device=device,
                    generator=generator,
                ),
                _packed_mxfp8_weight(
                    self.hidden_size,
                    device=device,
                    generator=generator,
                ),
            )
            for _ in range(self.boundaries + 1)
        ]
        self.parameters = []
        for boundary in range(self.boundaries):
            fn = torch.randn(
                (24, 4 * self.hidden_size),
                dtype=torch.float32,
                device=device,
                generator=generator,
            ).mul_(1.0 / 64.0)
            self.parameters.append(
                _MhcLayerParameters(
                    fn=fn,
                    # vLLM supplies the BF16 mirror at the attention-to-FFN
                    # boundary and the FP32 parameter at the FFN-to-attention
                    # boundary.  Planned backends may select either input.
                    fn_bf16=(fn.to(torch.bfloat16) if boundary % 2 == 0 else None),
                    scale=torch.randn(
                        (3,),
                        dtype=torch.float32,
                        device=device,
                        generator=generator,
                    ).mul_(1.0 / 3.0),
                    bias=torch.randn(
                        (24,),
                        dtype=torch.float32,
                        device=device,
                        generator=generator,
                    ).mul_(1.0 / 5.0),
                    norm_weight=(
                        1.0
                        + torch.randn(
                            (self.hidden_size,),
                            dtype=torch.float32,
                            device=device,
                            generator=generator,
                        ).mul_(0.05)
                    ).to(torch.bfloat16),
                )
            )

        self.residual_outputs = [
            torch.empty(
                (self.tokens, 4, self.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(self.boundaries)
        ]
        self.y_outputs = [
            torch.empty(
                (self.tokens, self.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(self.boundaries)
        ]
        self.post_outputs = [
            torch.empty(
                (self.tokens, 4),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(self.boundaries)
        ]
        self.comb_outputs = [
            torch.empty(
                (self.tokens, 4, 4),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(self.boundaries)
        ]
        self.bindings: dict[str, tuple[object, ...]] = {}
        self.owners: dict[str, tuple[object, ...]] = {}

    def prepare_candidate(self, candidate_id: str, plan) -> None:
        import torch

        if candidate_id in self.bindings:
            return
        bindings = []
        owners = []
        for boundary in range(self.boundaries):
            scratch = tuple(
                torch.empty(shape, dtype=dtype, device=self.device)
                for shape, dtype in plan.shapes_and_dtypes()
            )
            binding = self.mhc.bind(
                plan,
                scratch=scratch,
                tokens=self.tokens,
                expected_m=self.tokens,
                y=self.y_outputs[boundary],
                post=self.post_outputs[boundary],
                comb=self.comb_outputs[boundary],
                out=self.residual_outputs[boundary],
            )
            bindings.append(binding)
            owners.extend((scratch, binding))
        self.bindings[candidate_id] = tuple(bindings)
        self.owners[candidate_id] = (plan, *owners)

    def _project(self, source, weights):
        hidden = self._blockscaled.mm(
            source,
            weights[0],
            expected_m=self.tokens,
        )
        return self._blockscaled.mm(
            hidden,
            weights[1],
            expected_m=self.tokens,
        )

    def run(self, candidate_id: str, events=None):
        bindings = self.bindings[candidate_id]
        y = self.initial_y
        residual = self.initial_residual
        post = self.initial_post
        comb = self.initial_comb
        for boundary in range(self.boundaries):
            x = self._project(y, self.projection_weights[boundary])
            if events is not None:
                events[boundary][0].record()
            parameters = self.parameters[boundary]
            self.mhc.run_post_pre(
                x,
                residual,
                post,
                comb,
                parameters.fn,
                parameters.scale,
                parameters.bias,
                rms_eps=1.0e-6,
                hc_eps=1.0e-6,
                sinkhorn_iters=20,
                norm_weight=parameters.norm_weight,
                norm_eps=1.0e-6,
                fn_bf16=parameters.fn_bf16,
                binding=bindings[boundary],
            )
            if events is not None:
                events[boundary][1].record()
            residual = self.residual_outputs[boundary]
            y = self.y_outputs[boundary]
            post = self.post_outputs[boundary]
            comb = self.comb_outputs[boundary]
        return self._project(y, self.projection_weights[-1])

    def mhc_outputs(self) -> tuple[object, ...]:
        return (
            *self.residual_outputs,
            *self.y_outputs,
            *self.post_outputs,
            *self.comb_outputs,
        )

    def caller_owned_tensors(self, candidate_id: str) -> tuple[object, ...]:
        tensors = [
            self.initial_y,
            self.initial_residual,
            self.initial_post,
            self.initial_comb,
            *self.mhc_outputs(),
        ]
        for owner in self.owners[candidate_id]:
            if isinstance(owner, tuple):
                tensors.extend(owner)
        return tuple(tensors)


__all__ = [
    "MHC_CONTEXT_BOUNDARIES",
    "MHC_CONTEXT_TRANSFORMER_LAYERS",
    "MhcLayerStack",
]
