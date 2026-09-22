"""CPU checkpoint ownership for deferred expert-cache preparation."""

from dataclasses import dataclass
import os

import torch

from .planning import WeightPlan
from .weights import PackedWeights
from b12x.moe.residency.storage import (
    BackingMode, ExpertRepresentation, ExpertShard, ExpertStorageContract,
)


NVFP4_CACHE_ADAPTER = "modelopt_nvfp4_canonical_w4a16"
NVFP4_CACHE_RECIPE = "nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum"


@dataclass(frozen=True, kw_only=True)
class ExpertWeightSource:
    """Retain logical, unswizzled CPU rows until the prepared cache is released.

    The loader owns TP slicing and checkpoint identity verification. Gate/up
    order is declared by ``plan.source.w13_layout``. Tensor and mmap owners must
    remain immutable; no device copy or value inspection occurs at declaration.
    Preparation permutes scale bytes one expert at a time without requantizing.
    """

    plan: WeightPlan
    weights: PackedWeights
    owners: tuple = ()
    shard: ExpertShard | None = None

    def __post_init__(self):
        from .planning import ActivationMode
        from .source import PackedSource, PackedSourceFormat
        from .weights import WeightPacking

        p, w = self.plan, self.weights
        if not isinstance(p, WeightPlan) or not isinstance(w, PackedWeights):
            raise TypeError("expert source requires WeightPlan and PackedWeights")
        if (
            not isinstance(p.source, PackedSource)
            or p.source.format != PackedSourceFormat.MODELOPT_NVFP4
            or p.activation.mode != ActivationMode.A16
            or p.activation.io_dtype != torch.bfloat16
            or p.activation.nonlinearity != "silu"
            or any(
                getattr(p.activation, n) is not None
                for n in ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
            )
            or p.prepared_format.packing != WeightPacking.SOURCE_NATIVE
        ):
            raise ValueError(
                "canonical cache requires explicit BF16 W4A16 source-native ModelOpt NVFP4 SiLU"
            )
        e, h, i = (
            p.geometry.num_experts,
            p.geometry.hidden_size,
            p.geometry.intermediate_size,
        )
        if h % 128 or i % 128:
            raise ValueError("canonical cache requires H and local I divisible by 128")
        shard = self.shard or ExpertShard(
            experts=e, hidden=h, intermediate=i, global_intermediate=i
        )
        if not isinstance(shard, ExpertShard) or (shard.experts, shard.hidden, shard.intermediate) != (e, h, i):
            raise ValueError("storage shard differs from the weight plan geometry")
        object.__setattr__(self, "shard", shard)
        for name, shape, dtype in (
            ("w13", (e, 2 * i, h // 2), torch.uint8),
            ("w2", (e, h, i // 2), torch.uint8),
            ("w13_block_scales", (e, 2 * i, h // 16), torch.float8_e4m3fn),
            ("w2_block_scales", (e, h, i // 16), torch.float8_e4m3fn),
            ("w13_global_scales", (e,), torch.float32),
            ("w2_global_scales", (e,), torch.float32),
        ):
            value = getattr(w, name)
            if (
                value.device.type != "cpu"
                or value.shape != shape
                or value.dtype != dtype
                or not value.is_contiguous()
            ):
                raise ValueError(f"{name} requires contiguous CPU {dtype} {shape}")
        if not w.checkpoint_fingerprint or not w.layer_name:
            raise ValueError(
                "expert source requires verified checkpoint and layer identity"
            )
        object.__setattr__(self, "owners", tuple(self.owners))
        if any(
            value.device.type != "cpu"
            for value in (*vars(w).values(), *self.owners)
            if isinstance(value, torch.Tensor)
        ):
            raise ValueError(
                "expert source metadata and tensor owners must remain on CPU"
            )

    @property
    def storage(self):
        g = self.shard
        matrices = 3 * g.hidden * g.intermediate
        payload = matrices // 2 + matrices // 16 + 8
        canonical = ExpertRepresentation(
            encoding="modelopt_nvfp4_swizzled_k16", bytes_per_expert=payload,
            alignment=256,
        )
        return ExpertStorageContract(
            adapter=NVFP4_CACHE_ADAPTER,
            checkpoint=self.weights.checkpoint_fingerprint,
            layer=self.weights.layer_name,
            recipe=NVFP4_CACHE_RECIPE,
            shard=g,
            source=ExpertRepresentation(
                encoding="modelopt_nvfp4", bytes_per_expert=self.source_bytes // g.experts,
                shared_bytes=self.source_bytes % g.experts,
            ),
            backing=canonical, resident=canonical, mode=BackingMode.PREPARED,
            direct_cold_execution=True, source_transform="swizzle_k16_scales",
            rollback="canonical_restore",
        )

    @property
    def source_bytes(self):
        storages = {
            v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
            for v in (*vars(self.weights).values(), *self.owners)
            if isinstance(v, torch.Tensor)
        }
        return sum(storages.values())

    def validate_values(self):
        for name in ("w13_global_scales", "w2_global_scales"):
            value = getattr(self.weights, name)
            if not bool(torch.isfinite(value).all() and (value > 0).all()):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("w13_block_scales", "w2_block_scales"):
            # One expert at a time bounds temporary FP32 validation storage.
            for row in getattr(self.weights, name):
                value = row.float()
                if not bool(torch.isfinite(value).all() and (value >= 0).all()):
                    raise ValueError(f"{name} must be nonnegative and finite")

    def row(self, expert):
        from ._residency_storage import _swizzle_scale

        if type(expert) is not int or not 0 <= expert < self.shard.experts:
            raise ValueError("expert ID is outside canonical layer geometry")
        w = self.weights
        return dict(
            w13=w.w13[expert],
            w2=w.w2[expert],
            s13=_swizzle_scale(w.w13_block_scales[expert]),
            s2=_swizzle_scale(w.w2_block_scales[expert]),
            g13=w.w13_global_scales[expert : expert + 1].view(torch.uint8),
            g2=w.w2_global_scales[expert : expert + 1].view(torch.uint8),
        )


def checkpoint_fingerprint(directory):
    """Hash local checkpoint/config contents with bounded CPU memory.

    An explicit B12X_CHECKPOINT_IDENTITY receipt avoids rescanning immutable
    files across engine trials. File replacement or modification fails closed.
    Download revision labels alone are not content verification.
    """
    from b12x.integration.vllm.checkpoint_identity import checkpoint_identity

    return checkpoint_identity(
        directory, receipt=os.environ.get("B12X_CHECKPOINT_IDENTITY")
    )["fingerprint"]
