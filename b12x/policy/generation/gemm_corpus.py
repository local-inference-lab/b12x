"""Reviewed dense-linear and W_o projection geometries for profile generation.

Shapes are derived from the production checkpoints served on SM12x parts
(DeepSeek-V4-Flash, GLM-5.3 NVFP4 / Flash, Qwen3.8-Flash-Next and the
Qwen3.5 NVFP4 family) sharded for the tensor-parallel sizes those recipes
are served at. Column-parallel layers shard ``out_features``; row-parallel
layers shard ``in_features``; replicated layers keep both.
"""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy.generation.sweep import SweepCase

# Decode band densely (every M through 8 plus the spec-verify widths), then
# the anchors the reducer interpolates between.
DENSE_TOKEN_LADDER: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
)
WO_TOKEN_LADDER: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
)
DENSE_MAX_TOKENS_BOUNDS: tuple[int, int] = (1, 8192)


@dataclass(frozen=True)
class LinearLayer:
    name: str
    in_features: int
    out_features: int
    parallel: str  # "col" | "row" | "rep"


@dataclass(frozen=True)
class ModelLinears:
    model_id: str
    recipe: str
    tp_sizes: tuple[int, ...]
    layers: tuple[LinearLayer, ...]


def _col(name: str, k: int, n: int) -> LinearLayer:
    return LinearLayer(name, k, n, "col")


def _row(name: str, k: int, n: int) -> LinearLayer:
    return LinearLayer(name, k, n, "row")


def _rep(name: str, k: int, n: int) -> LinearLayer:
    return LinearLayer(name, k, n, "rep")


MODEL_LINEARS: tuple[ModelLinears, ...] = (
    # Qwen3.8-Flash-Next: MXFP8 GDN / QSA projections and shared expert.
    ModelLinears(
        model_id="qwen3.8-flash-next-180b",
        recipe="mxfp8",
        tp_sizes=(1, 2, 4),
        layers=(
            _col("gdn.in_proj_qkvz", 2560, 16384),
            _col("gdn.in_proj_ba", 2560, 96),
            _row("gdn.out_proj", 6144, 2560),
            _col("qsa.q_proj", 2560, 12288),
            _col("qsa.kv_proj", 2560, 1024),
            _row("qsa.o_proj", 6144, 2560),
            _rep("qsa.indexer_qk", 2560, 640),
            _col("shared_expert.gate_up", 2560, 1280),
            _row("shared_expert.down", 640, 2560),
        ),
    ),
    # GLM-5.3 Flash: MXFP8 MLA projections and shared expert.
    ModelLinears(
        model_id="glm-5.3-flash",
        recipe="mxfp8",
        tp_sizes=(1, 2, 4),
        layers=(
            _rep("mla.q_a", 4096, 1536),
            _col("mla.q_b", 1536, 16384),
            _rep("mla.kv_a", 4096, 576),
            _col("mla.kv_b", 512, 32768),
            _row("mla.o", 16384, 4096),
            _col("shared_expert.gate_up", 4096, 4096),
            _row("shared_expert.down", 2048, 4096),
        ),
    ),
    # DeepSeek-V4-Flash-0731: 128x128 block-FP8 attention / shared expert.
    ModelLinears(
        model_id="deepseek-v4-flash",
        recipe="block_fp8",
        tp_sizes=(1, 2, 4, 8),
        layers=(
            _rep("attn.wq_a", 4096, 1024),
            _col("attn.wq_b", 1024, 36864),
            _rep("attn.wkv", 4096, 576),
            _col("attn.indexer.wq_b", 1024, 8192),
            _col("shared_experts.w1w3", 4096, 4096),
            _row("shared_experts.w2", 2048, 4096),
        ),
    ),
    # GLM-5.3 NVFP4: NVFP4 MLA projections and shared expert.
    ModelLinears(
        model_id="glm-5.3-nvfp4",
        recipe="nvfp4",
        tp_sizes=(1, 2, 4, 8),
        layers=(
            _rep("mla.q_a", 6144, 2048),
            _col("mla.q_b", 2048, 16384),
            _rep("mla.kv_a", 6144, 576),
            _col("mla.kv_b", 512, 28672),
            _row("mla.o", 16384, 6144),
            _col("shared_expert.gate_up", 6144, 4096),
            _row("shared_expert.down", 2048, 6144),
        ),
    ),
    # Qwen3.5-397B NVFP4: attention, GDN and shared expert projections.
    ModelLinears(
        model_id="qwen3.5-397b-nvfp4",
        recipe="nvfp4",
        tp_sizes=(1, 2, 4, 8),
        layers=(
            _col("attn.q_proj", 4096, 16384),
            _col("attn.kv_proj", 4096, 1024),
            _row("attn.o_proj", 8192, 4096),
            _col("gdn.in_proj_qkvz", 4096, 20480),
            _row("gdn.out_proj", 8192, 4096),
            _col("shared_expert.gate_up", 4096, 2048),
            _row("shared_expert.down", 1024, 4096),
        ),
    ),
    # Qwen3.5-122B / 35B NVFP4 (hidden 3072 / 2048 variants).
    ModelLinears(
        model_id="qwen3.5-122b-nvfp4",
        recipe="nvfp4",
        tp_sizes=(1, 2, 4),
        layers=(
            _col("attn.q_proj", 3072, 16384),
            _col("attn.kv_proj", 3072, 1024),
            _row("attn.o_proj", 8192, 3072),
            _col("gdn.in_proj_qkvz", 3072, 20480),
            _row("gdn.out_proj", 8192, 3072),
            _col("shared_expert.gate_up", 3072, 2048),
            _row("shared_expert.down", 1024, 3072),
        ),
    ),
    ModelLinears(
        model_id="qwen3.5-35b-nvfp4",
        recipe="nvfp4",
        tp_sizes=(1, 2),
        layers=(
            _col("attn.q_proj", 2048, 8192),
            _col("attn.kv_proj", 2048, 1024),
            _row("attn.o_proj", 4096, 2048),
            _col("gdn.in_proj_qkvz", 2048, 12288),
            _row("gdn.out_proj", 4096, 2048),
            _col("shared_expert.gate_up", 2048, 1024),
            _row("shared_expert.down", 512, 2048),
        ),
    ),
    # Generic square/wide shards keep the MXFP4 and tensor-FP8 recipes covered
    # until a production checkpoint pins their geometry.
    ModelLinears(
        model_id="generic-mxfp4",
        recipe="mxfp4",
        tp_sizes=(1, 2),
        layers=(
            _col("proj.wide", 4096, 12288),
            _rep("proj.square", 4096, 4096),
            _row("proj.down", 12288, 4096),
        ),
    ),
    ModelLinears(
        model_id="generic-tensor-fp8",
        recipe="tensor_fp8",
        tp_sizes=(1, 2),
        layers=(
            _col("proj.wide", 4096, 12288),
            _rep("proj.square", 4096, 4096),
            _row("proj.down", 12288, 4096),
        ),
    ),
)


@dataclass(frozen=True)
class DenseLinearShape:
    recipe: str
    in_features: int
    out_features: int
    model_ids: tuple[str, ...]
    tp_sizes: tuple[int, ...]


def _k_alignment(recipe: str) -> int:
    return 128 if recipe == "block_fp8" else 32


def dense_linear_shapes(
    models: tuple[ModelLinears, ...] = MODEL_LINEARS,
) -> tuple[DenseLinearShape, ...]:
    """Distinct (recipe, K, N) shards across every reviewed model and TP."""
    found: dict[tuple[str, int, int], tuple[set[str], set[int]]] = {}
    for model in models:
        for tp in model.tp_sizes:
            for layer in model.layers:
                k, n = layer.in_features, layer.out_features
                if layer.parallel == "col":
                    if n % tp:
                        continue
                    n //= tp
                elif layer.parallel == "row":
                    if k % tp:
                        continue
                    k //= tp
                if k % _k_alignment(model.recipe) or n % 16 or n <= 0:
                    continue
                if model.recipe == "block_fp8" and n % 128:
                    # The K128 block recipe needs whole 128x128 scale blocks.
                    continue
                entry = found.setdefault((model.recipe, k, n), (set(), set()))
                entry[0].add(model.model_id)
                entry[1].add(tp)
    return tuple(
        DenseLinearShape(
            recipe=recipe,
            in_features=k,
            out_features=n,
            model_ids=tuple(sorted(models_)),
            tp_sizes=tuple(sorted(tps)),
        )
        for (recipe, k, n), (models_, tps) in sorted(found.items())
    )


def dense_linear_cases(
    *,
    recipes: tuple[str, ...] | None = None,
    token_ladder: tuple[int, ...] = DENSE_TOKEN_LADDER,
) -> tuple[SweepCase, ...]:
    cases = []
    for shape in dense_linear_shapes():
        if recipes is not None and shape.recipe not in recipes:
            continue
        group_id = f"{shape.recipe}-k{shape.in_features}-n{shape.out_features}"
        for tokens in token_ladder:
            cases.append(
                SweepCase.create(
                    group_id=group_id,
                    query={
                        "recipe": shape.recipe,
                        "output_dtype": "bfloat16",
                        "in_features": shape.in_features,
                        "out_features": shape.out_features,
                        "max_tokens": int(tokens),
                    },
                    metadata={
                        "models": list(shape.model_ids),
                        "tp_sizes": list(shape.tp_sizes),
                    },
                )
            )
    return tuple(cases)


@dataclass(frozen=True)
class WoProjectionGeometry:
    model_id: str
    groups: int
    group_width: int
    rank: int
    hidden: int
    inv_rope: bool
    nope_dim: int
    rope_dim: int
    tp_size: int


def wo_projection_geometries() -> tuple[WoProjectionGeometry, ...]:
    geometries = []
    # DeepSeek-V4-Flash: 8 output groups of rank 1024; every group spans 8
    # heads of 512 (vLLM: group_width = heads_per_group * head_dim = 4096),
    # served through the fused inverse-RoPE chain.
    for tp in (1, 2, 4, 8):
        geometries.append(
            WoProjectionGeometry(
                model_id="deepseek-v4-flash",
                groups=8 // tp,
                group_width=4096,
                rank=1024,
                hidden=4096,
                inv_rope=True,
                nope_dim=448,
                rope_dim=64,
                tp_size=tp,
            )
        )
    # The previously qualified 24-group / rank-512 composite (plain path).
    for tp in (1, 2, 4, 8):
        geometries.append(
            WoProjectionGeometry(
                model_id="qwen3.8-flash-next-180b",
                groups=24 // tp,
                group_width=512,
                rank=512,
                hidden=2560,
                inv_rope=False,
                nope_dim=448,
                rope_dim=64,
                tp_size=tp,
            )
        )
    return tuple(geometries)


def wo_projection_cases(
    *,
    token_ladder: tuple[int, ...] = WO_TOKEN_LADDER,
) -> tuple[SweepCase, ...]:
    cases = []
    for geometry in wo_projection_geometries():
        group_id = f"wo-{geometry.model_id}-tp{geometry.tp_size}"
        for tokens in token_ladder:
            cases.append(
                SweepCase.create(
                    group_id=group_id,
                    query={
                        "dtype": "bfloat16",
                        "groups": geometry.groups,
                        "group_width": geometry.group_width,
                        "rank": geometry.rank,
                        "hidden": geometry.hidden,
                        "max_tokens": int(tokens),
                    },
                    metadata={
                        "model": geometry.model_id,
                        "tp_size": geometry.tp_size,
                        "inv_rope": geometry.inv_rope,
                        "nope_dim": geometry.nope_dim,
                        "rope_dim": geometry.rope_dim,
                    },
                )
            )
    return tuple(cases)


__all__ = [
    "DENSE_MAX_TOKENS_BOUNDS",
    "DENSE_TOKEN_LADDER",
    "MODEL_LINEARS",
    "WO_TOKEN_LADDER",
    "DenseLinearShape",
    "LinearLayer",
    "ModelLinears",
    "WoProjectionGeometry",
    "dense_linear_cases",
    "dense_linear_shapes",
    "wo_projection_cases",
    "wo_projection_geometries",
]
