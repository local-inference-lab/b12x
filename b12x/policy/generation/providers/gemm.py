"""Measured providers for planned GEMM composites."""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from contextlib import AbstractContextManager

from b12x.policy.components import (
    BF16_VOCAB_PROJECTION,
    BLOCK_FP8_LINEAR,
    DENSE_LINEAR,
    WO_PROJECTION,
)
from b12x.policy.generation.gemm_corpus import (
    DENSE_MAX_TOKENS_BOUNDS,
    dense_linear_cases,
    wo_projection_cases,
)
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import (
    _bounded_repetitions,
    _cuda_event_samples_us,
    _l2_flush_fn,
    _median_of_group_medians,
)


_BLOCK_FP8_TILES = (
    (16, 64),
    (16, 128),
    (32, 64),
    (32, 128),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)

_VOCAB_PROJECTION_GEOMETRIES = (
    ("qwen3.8-flash-next-180b", 2_560, (248_320,)),
    ("qwen3.8-27b", 5_120, (248_320,)),
    ("glm-5.3", 6_144, (154_880,)),
    ("glm-5.3-flash", 4_096, (154_880,)),
    ("glm-5.2", 6_144, (163_840, 163_968)),
)
_VOCAB_PROJECTION_TP_SIZES = (1, 2, 4, 8, 16)


def _bf16_vocab_projection_cases() -> tuple[SweepCase, ...]:
    cases = []
    for model_id, in_features, global_vocab_sizes in _VOCAB_PROJECTION_GEOMETRIES:
        for global_vocab_size in global_vocab_sizes:
            for tp_size in _VOCAB_PROJECTION_TP_SIZES:
                if global_vocab_size % tp_size:
                    continue
                out_features = global_vocab_size // tp_size
                cases.append(
                    SweepCase.create(
                        group_id=(f"{model_id}-v{global_vocab_size}-tp{tp_size}"),
                        query={
                            "dtype": "bfloat16",
                            "max_tokens": 1,
                            "in_features": in_features,
                            "out_features": out_features,
                        },
                        scenario=f"{model_id}-tp{tp_size}",
                        metadata={
                            "model_id": model_id,
                            "global_vocab_size": global_vocab_size,
                            "tp_size": tp_size,
                        },
                    )
                )
    return tuple(cases)


class _Bf16VocabProjectionSession(
    AbstractContextManager["_Bf16VocabProjectionSession"]
):
    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_Bf16VocabProjectionSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        in_features = int(case.query["in_features"])
        direct_block = 1 << (in_features - 1).bit_length()
        configs = [
            {
                "backend": "torch",
                "algorithm": "torch",
                "block_k": 0,
                "num_warps": 0,
            }
        ]
        configs.extend(
            {
                "backend": "triton",
                "algorithm": "row",
                "block_k": direct_block,
                "num_warps": num_warps,
            }
            for num_warps in (1, 2, 4, 8)
        )
        configs.extend(
            {
                "backend": "triton",
                "algorithm": "loop",
                "block_k": block_k,
                "num_warps": num_warps,
            }
            for block_k in (256, 512, 1_024)
            for num_warps in (4, 8)
        )
        return tuple(SweepCandidate.create(config) for config in configs)

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.gemm import bf16_vocab_projection as projection
        from b12x.gemm.bf16_vocab_projection._policy import (
            Bf16VocabProjectionConfig,
        )
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        in_features = int(case.query["in_features"])
        out_features = int(case.query["out_features"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            source = torch.randn(
                (1, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            weight = torch.randn(
                (out_features, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.125)
            expected = torch_functional.linear(source, weight)
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            base_policy = PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            )
            measurements = []
            for candidate in candidates:
                try:
                    config = Bf16VocabProjectionConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(
                        BF16_VOCAB_PROJECTION,
                        config,
                    )
                    planned = projection.plan(
                        projection.Caps(
                            device=device,
                            max_tokens=1,
                            in_features=in_features,
                            out_features=out_features,
                        ),
                        policy=policy,
                    )
                    binding = projection.bind(
                        planned,
                        source=source,
                        weight=weight,
                    )

                    def run():
                        return projection.run(binding)

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    actual.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    cosine = float(
                        torch_functional.cosine_similarity(
                            actual.float(),
                            expected.float(),
                        ).item()
                    )
                    finite = bool(torch.isfinite(actual).all().item())
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * settings.repetitions,
                        device=device,
                        flush=flush,
                    )
                    allocated_after = torch.cuda.memory_allocated(device)
                    latency_us = _median_of_group_medians(
                        samples,
                        groups=settings.groups,
                        repetitions=settings.repetitions,
                    )
                    transferred_bytes = 2 * (
                        out_features * in_features + in_features + out_features
                    )
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=latency_us,
                            correct=(
                                finite
                                and cosine >= settings.minimum_cosine
                                and allocated_after <= allocated_before
                            ),
                            metrics={
                                "cosine": cosine,
                                "finite": finite,
                                "replay_allocation_bytes": (
                                    allocated_after - allocated_before
                                ),
                                "effective_bandwidth_gbps": (
                                    transferred_bytes / latency_us / 1_000.0
                                ),
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed configs survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)


class _Bf16VocabProjectionFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("vocabulary projection allocation groups contain one case")
        return _Bf16VocabProjectionSession(context)


class Bf16VocabProjectionGenerator(DiscreteSweepGenerator):
    """Race production BF16 vocabulary projection paths over common models."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=BF16_VOCAB_PROJECTION,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=(
                "dtype",
                "max_tokens",
                "in_features",
                "out_features",
            ),
            range_fields=frozenset({"out_features"}),
            cases=(_bf16_vocab_projection_cases() if cases is None else cases),
            benchmark_factory=_Bf16VocabProjectionFactory(),
            coverage={},
            candidate_contract_version=1,
            nearest_range_bounds={"out_features": (1, 248_320)},
        )


def _block_fp8_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"m{tokens}-k{in_features}-n{out_features}",
            query={
                "max_tokens": tokens,
                "in_features": in_features,
                "out_features": out_features,
                "output_dtype": "bfloat16",
            },
        )
        for tokens in (4, 32)
        for in_features, out_features in (
            (2_560, 2_560),
            (2_560, 10_240),
        )
    )


class _BlockFp8Session(AbstractContextManager["_BlockFp8Session"]):
    _CANDIDATES = tuple(
        SweepCandidate.create({"backend": "mxfp8", "tile_m": tile_m, "tile_n": tile_n})
        for tile_m, tile_n in _BLOCK_FP8_TILES
    )

    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_BlockFp8Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.gemm import block_fp8_linear as block_fp8
        from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
        from b12x.gemm.block_fp8_linear._policy import BlockFp8LinearConfig
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        tokens = int(case.query["max_tokens"])
        in_features = int(case.query["in_features"])
        out_features = int(case.query["out_features"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            source = torch.randn(
                (tokens, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            weight = (
                torch.randn(
                    (out_features, in_features),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                .mul_(0.125)
                .to(torch.float8_e4m3fn)
            )
            scale = torch.ones(
                (out_features // 128, in_features // 128),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )
            packed = block_fp8.pack_weight(weight, scale)
            source_q = block_fp8.quantize_input(source)
            source_dequantized = dequantize_mxfp8_rows_torch(
                source_q.values,
                source_q.scale_rows,
            )
            weight_dequantized = dequantize_mxfp8_rows_torch(
                packed.weight.values,
                packed.weight.scale_rows,
            )
            expected = source_dequantized @ weight_dequantized.T
            del source_q, source_dequantized, weight_dequantized
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            base_policy = PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            )
            measurements = []
            for candidate in candidates:
                try:
                    config = BlockFp8LinearConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(BLOCK_FP8_LINEAR, config)
                    plan = block_fp8.plan(
                        block_fp8.Caps(
                            device=device,
                            max_tokens=tokens,
                            in_features=in_features,
                            out_features=out_features,
                            output_dtype=torch.bfloat16,
                        ),
                        policy=policy,
                    )
                    (scratch_spec,) = plan.scratch_specs()
                    scratch = torch.empty(
                        scratch_spec.shape,
                        dtype=scratch_spec.dtype,
                        device=scratch_spec.device,
                    )
                    output = torch.empty(
                        (tokens, out_features, 1),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    binding = block_fp8.bind(
                        plan,
                        scratch=scratch,
                        source=source,
                        packed_weight=packed,
                        output=output,
                        expected_m=tokens,
                    )

                    def run() -> None:
                        block_fp8.run(binding=binding)

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    binding.output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    actual = binding.output[:, :, 0]
                    cosine = float(
                        torch_functional.cosine_similarity(
                            actual.float().reshape(1, -1),
                            expected.float().reshape(1, -1),
                        ).item()
                    )
                    finite = bool(torch.isfinite(actual).all().item())
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * settings.repetitions,
                        device=device,
                        flush=flush,
                    )
                    allocated_after = torch.cuda.memory_allocated(device)
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=_median_of_group_medians(
                                samples,
                                groups=settings.groups,
                                repetitions=settings.repetitions,
                            ),
                            correct=(
                                finite
                                and cosine >= settings.minimum_cosine
                                and allocated_after <= allocated_before
                            ),
                            metrics={
                                "cosine": cosine,
                                "finite": finite,
                                "replay_allocation_bytes": (
                                    allocated_after - allocated_before
                                ),
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed tiles survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)


class _BlockFp8Factory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("block-FP8 allocation groups contain one case")
        return _BlockFp8Session(context)


class BlockFp8LinearGenerator(DiscreteSweepGenerator):
    """Race the production block-FP8 linear MMA tiles."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=BLOCK_FP8_LINEAR,
            query_schema_version=1,
            config_schema_version=2,
            query_fields=(
                "max_tokens",
                "in_features",
                "out_features",
                "output_dtype",
            ),
            range_fields=frozenset({"max_tokens", "in_features", "out_features"}),
            cases=_block_fp8_cases() if cases is None else cases,
            benchmark_factory=_BlockFp8Factory(),
            coverage={},
        )


# ---------------------------------------------------------------------------
# gemm.dense_linear: race the dense SM120 GEMM launch plans per recipe/shape.
# ---------------------------------------------------------------------------

# Decode/mid/prefill tile menus for the FP8 recipes; FP4 menus come from the
# policy tables because the FP4 MMA only accepts 64/128-row tiles.
_FP8_DECODE_TILES = ((16, 64), (16, 128), (32, 64), (32, 128), (64, 64), (64, 128))
_FP8_MID_TILES = (
    (16, 128),
    (32, 64),
    (32, 128),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)
_FP8_PREFILL_TILES = ((64, 64), (64, 128), (128, 64), (128, 128))
_DENSE_TILE_K_RECIPES = frozenset({"nvfp4", "mxfp8"})
_DENSE_SPLIT_K_RECIPES = frozenset({"mxfp8", "block_fp8"})
_DENSE_SWAP_RECIPES = frozenset({"mxfp8"})


# A profile plan must beat the built-in plan by this fraction to replace it.
_GEMM_BASELINE_MARGIN = 0.03


def _device_identity(context):
    """Identity the built-in heuristics key on (SM count decides the Spark rules)."""
    from b12x.policy import DeviceIdentity, PolicyContext, PolicyMode

    device = context.device
    if isinstance(device, DeviceIdentity):
        return device
    return PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY).device


def _env_filter(name: str) -> tuple[str, ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _default_dense_linear_cases() -> tuple[SweepCase, ...]:
    """Reviewed corpus, optionally narrowed for incremental generation runs.

    ``B12X_GEMM_PROFILE_RECIPES`` (comma list) keeps only those recipes and
    ``B12X_GEMM_PROFILE_MODELS`` (comma list of substrings) keeps only shapes
    contributed by matching models. Case IDs are stable, so a narrowed run
    checkpoints normally and a later wider run reuses its measurements.
    """
    recipes = _env_filter("B12X_GEMM_PROFILE_RECIPES")
    models = _env_filter("B12X_GEMM_PROFILE_MODELS")
    cases = dense_linear_cases(recipes=recipes)
    if models:
        cases = tuple(
            case
            for case in cases
            if any(
                any(needle in model for needle in models)
                for model in case.metadata.get("models", ())
            )
        )
    if not cases:
        raise ValueError("dense linear corpus filters removed every case")
    return cases


def _dense_linear_candidate_configs(
    *,
    recipe: str,
    in_features: int,
    max_tokens: int,
) -> tuple[dict[str, object], ...]:
    """Enumerate launch plans worth racing for one (recipe, K, capacity).

    Only combinations ``DenseGemmKernel.can_implement`` accepts are emitted;
    the K-tile and split-K axes are explored independently of each other.
    """
    from b12x.gemm.dense_linear._policy import (
        FP4_NARROW_TILES,
        FP4_TILES,
        FP8_SWAPPED_TILES,
    )

    decode = max_tokens <= 16
    tiles: list[tuple[tuple[int, int], bool]]
    if recipe == "mxfp4":
        tiles = [(tile, False) for tile in FP4_TILES]
    elif recipe == "nvfp4":
        tiles = [(tile, False) for tile in FP4_TILES]
        if max_tokens <= 256:
            tiles += [(tile, True) for tile in FP4_NARROW_TILES]
    else:
        if decode:
            base = _FP8_DECODE_TILES
        elif max_tokens <= 256:
            base = _FP8_MID_TILES
        else:
            base = _FP8_PREFILL_TILES
        tiles = [(tile, False) for tile in base]
        if decode and recipe in _DENSE_SWAP_RECIPES:
            tiles += [(tile, True) for tile in FP8_SWAPPED_TILES]
    load_paths = ("tma", "cpasync") if (recipe == "nvfp4" and decode) else ("tma",)
    tile_ks: tuple[int, ...] = ()
    if recipe in _DENSE_TILE_K_RECIPES:
        tile_ks = tuple(
            tk
            for tk in (128, 256, 512)
            if in_features % tk == 0 and (tk != 512 or max_tokens > 256)
        )
    split_ks: tuple[int, ...] = ()
    if recipe in _DENSE_SPLIT_K_RECIPES and decode and in_features >= 2048:
        split_ks = (2, 4)
    axes = [(0, 0)] + [(tk, 0) for tk in tile_ks] + [(0, sk) for sk in split_ks]
    configs: list[dict[str, object]] = []
    for (tile_m, tile_n), swap_ab in tiles:
        for load_path in load_paths:
            for tile_k, split_k in axes:
                configs.append(
                    {
                        "backend": "dense",
                        "tile_m": tile_m,
                        "tile_n": tile_n,
                        "tile_k": tile_k,
                        "load_path": load_path,
                        "swap_ab": swap_ab,
                        "split_k": split_k,
                    }
                )
    return tuple(configs)


def _random_block_scales(
    rows: int,
    k: int,
    *,
    vec_size: int,
    scale_dtype,
    generator,
    device,
):
    """Random-but-valid block scales as (row-major rows, swizzled storage)."""
    import torch

    from b12x._lib.intrinsics import swizzle_block_scale

    if scale_dtype == torch.float8_e8m0fnu:
        low, high = 122, 132  # 2^-5 .. 2^5
    else:
        low, high = 0x30, 0x40  # E4M3 values in [0.5, 2.0]
    raw = torch.randint(
        low,
        high,
        (rows, k // vec_size),
        dtype=torch.int64,
        device=device,
        generator=generator,
    ).to(torch.uint8)
    return raw, swizzle_block_scale(raw).view(scale_dtype)


_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_fp4(packed, scale_rows, *, vec_size: int, scale_dtype):
    """Expand packed E2M1 nibbles and per-block scales to a float32 matrix."""
    import torch

    magnitudes = torch.tensor(_FP4_MAGNITUDES, device=packed.device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(1)
    values = magnitudes[(nibbles & 0x7).long()]
    values = values * torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    if scale_dtype == torch.float8_e8m0fnu:
        scale = torch.exp2(scale_rows.float() - 127.0)
    else:
        scale = scale_rows.view(torch.float8_e4m3fn).float()
    return values * scale.repeat_interleave(vec_size, dim=1)


def _dequantize_mxfp8(values, scale_bytes):
    import torch

    scale = torch.exp2(scale_bytes.float() - 127.0)
    return values.float() * scale.repeat_interleave(32, dim=1)


class _DenseLinearOperands:
    """Synthetic packed weight + activation for one (recipe, K, N).

    ``reference`` dequantizes the same operands in float32 so correctness is
    gated against the arithmetic definition of the recipe, not against the
    built-in planner's own launch.
    """

    def __init__(
        self, *, recipe: str, in_features: int, out_features: int, device, generator
    ) -> None:
        import torch

        from b12x.gemm import mxfp8_linear, tensor_fp8_linear

        self.recipe = recipe
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        k, n = in_features, out_features
        self.alpha = None
        self.output_scale = 1.0
        if recipe == "mxfp8":
            weight = torch.randn(
                (n, k), dtype=torch.bfloat16, device=device, generator=generator
            ).mul_(0.125)
            values, scales = _quantize_bf16_mxfp8(weight)
            self.packed = mxfp8_linear.pack_weight(values, scales)
            self.weight_dequant = _dequantize_mxfp8(values, scales)
        elif recipe == "tensor_fp8":
            weight = torch.randn(
                (n, k), dtype=torch.bfloat16, device=device, generator=generator
            ).mul_(0.125)
            self.output_scale = 1.0 / 32.0
            output_scale = torch.tensor(
                [self.output_scale], dtype=torch.float32, device=device
            )
            weight_fp8 = weight.to(torch.float8_e4m3fn)
            self.packed = tensor_fp8_linear.pack_weight(weight_fp8, output_scale)
            self.weight_dequant = weight_fp8.float()
        elif recipe in ("nvfp4", "mxfp4"):
            self.weight = torch.randint(
                0,
                256,
                (n, k // 2),
                dtype=torch.int64,
                device=device,
                generator=generator,
            ).to(torch.uint8)
            vec = 16 if recipe == "nvfp4" else 32
            sdt = torch.float8_e4m3fn if recipe == "nvfp4" else torch.float8_e8m0fnu
            scale_rows, self.weight_scale = _random_block_scales(
                n, k, vec_size=vec, scale_dtype=sdt, generator=generator, device=device
            )
            self.weight_dequant = _dequantize_fp4(
                self.weight, scale_rows, vec_size=vec, scale_dtype=sdt
            )
            if recipe == "nvfp4":
                self.output_scale = 1.0 / 64.0
                self.alpha = torch.tensor(
                    [self.output_scale], dtype=torch.float32, device=device
                )
            self.packed = None
        elif recipe == "block_fp8":
            self.weight = (
                torch.randn(
                    (n, k), dtype=torch.bfloat16, device=device, generator=generator
                )
                .mul_(0.125)
                .to(torch.float8_e4m3fn)
            )
            self.weight_scale = (
                torch.rand(
                    (n // 128, k // 128),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                .mul_(0.5)
                .add_(0.75)
            )
            self.weight_dequant = (
                self.weight.float()
                * self.weight_scale.repeat_interleave(128, dim=0).repeat_interleave(
                    128, dim=1
                )
            )
            self.packed = None
        else:
            raise ValueError(f"unsupported dense linear recipe {recipe!r}")

    def activation(self, tokens: int, generator):
        """Return (kernel operands, float32 dequantized activation)."""
        import torch

        k = self.in_features
        recipe = self.recipe
        if recipe == "mxfp8":
            source = torch.randn(
                (tokens, k),
                dtype=torch.bfloat16,
                device=self.device,
                generator=generator,
            ).mul_(0.25)
            values, scales = _quantize_bf16_mxfp8(source)
            return (source,), _dequantize_mxfp8(values, scales)
        if recipe == "tensor_fp8":
            source = (
                torch.randn(
                    (tokens, k),
                    dtype=torch.bfloat16,
                    device=self.device,
                    generator=generator,
                )
                .mul_(0.25)
                .to(torch.float8_e4m3fn)
            )
            return (source,), source.float()
        if recipe in ("nvfp4", "mxfp4"):
            values = torch.randint(
                0,
                256,
                (tokens, k // 2),
                dtype=torch.int64,
                device=self.device,
                generator=generator,
            ).to(torch.uint8)
            vec = 16 if recipe == "nvfp4" else 32
            sdt = torch.float8_e4m3fn if recipe == "nvfp4" else torch.float8_e8m0fnu
            scale_rows, storage = _random_block_scales(
                tokens,
                k,
                vec_size=vec,
                scale_dtype=sdt,
                generator=generator,
                device=self.device,
            )
            return (values, storage), _dequantize_fp4(
                values, scale_rows, vec_size=vec, scale_dtype=sdt
            )
        if recipe == "block_fp8":
            values = (
                torch.randn(
                    (tokens, k),
                    dtype=torch.bfloat16,
                    device=self.device,
                    generator=generator,
                )
                .mul_(0.25)
                .to(torch.float8_e4m3fn)
            )
            scale = (
                torch.rand(
                    (tokens, k // 128),
                    dtype=torch.float32,
                    device=self.device,
                    generator=generator,
                )
                .mul_(0.5)
                .add_(0.75)
            )
            return (values, scale), values.float() * scale.repeat_interleave(128, dim=1)
        raise ValueError(recipe)

    def reference(self, activation_dequant):
        return (activation_dequant @ self.weight_dequant.T) * self.output_scale

    def run(self, activation, *, plan, expected_m: int):
        from b12x.gemm import dense_linear

        recipe = self.recipe
        if recipe in ("mxfp8", "tensor_fp8"):
            return dense_linear.mm(
                activation[0], self.packed, plan=plan, expected_m=expected_m
            )
        if recipe == "nvfp4":
            return dense_linear.mm_serialized(
                (activation[0], activation[1]),
                (self.weight, self.weight_scale),
                alpha=self.alpha.reshape(1),
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16",
                sf_vec_size=16,
                plan=plan,
                expected_m=expected_m,
            )
        if recipe == "mxfp4":
            return dense_linear.mm_serialized(
                (activation[0], activation[1]),
                (self.weight, self.weight_scale),
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16",
                sf_vec_size=32,
                plan=plan,
                expected_m=expected_m,
            )
        if recipe == "block_fp8":
            return dense_linear.mm_serialized(
                (activation[0], activation[1]),
                (self.weight, self.weight_scale),
                ab_dtype="float8_e4m3fn",
                sf_dtype="float32",
                c_dtype="bfloat16",
                sf_vec_size=128,
                block_fp8=True,
                plan=plan,
                expected_m=expected_m,
            )
        raise ValueError(recipe)


def _quantize_bf16_mxfp8(weight):
    import torch

    n, k = weight.shape
    blocked = weight.to(torch.float32).reshape(n, k // 32, 32)
    amax = blocked.abs().amax(dim=-1).clamp_min(2.0**-126)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0)
    values = (
        (blocked / torch.exp2(exponent)[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(n, k)
    )
    return values.contiguous(), (exponent + 127.0).to(torch.uint8).contiguous()


def _graph_race(
    *,
    run,
    output_of,
    baseline,
    settings,
    device,
    flush,
):
    """Warm up, capture, poison, replay, gate on cosine, then time replays."""
    import torch
    import torch.nn.functional as torch_functional

    for _ in range(settings.warmup):
        run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    output = output_of(captured)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(output).all().item())
    cosine = float(
        torch_functional.cosine_similarity(
            output.float().reshape(1, -1), baseline.float().reshape(1, -1)
        ).item()
    )
    pilot = _cuda_event_samples_us(graph.replay, count=1, device=device, flush=flush)[0]
    repetitions = _bounded_repetitions(settings, pilot_us=pilot)
    allocated_before = torch.cuda.memory_allocated(device)
    samples = _cuda_event_samples_us(
        graph.replay,
        count=settings.groups * repetitions,
        device=device,
        flush=flush,
    )
    allocated_after = torch.cuda.memory_allocated(device)
    latency = _median_of_group_medians(
        samples, groups=settings.groups, repetitions=repetitions
    )
    correct = (
        finite
        and cosine >= settings.minimum_cosine
        and allocated_after <= allocated_before
    )
    return (
        latency,
        correct,
        {
            "cosine": cosine,
            "finite": finite,
            "replay_allocation_bytes": allocated_after - allocated_before,
            "repetitions": repetitions,
        },
    )


class _DenseLinearSession(AbstractContextManager["_DenseLinearSession"]):
    def __init__(self, context, cases: tuple[SweepCase, ...]) -> None:
        self._context = context
        self._cases = cases
        self._operands: _DenseLinearOperands | None = None

    def __enter__(self) -> "_DenseLinearSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        self._operands = None
        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        return tuple(
            SweepCandidate.create(config)
            for config in _dense_linear_candidate_configs(
                recipe=str(case.query["recipe"]),
                in_features=int(case.query["in_features"]),
                max_tokens=int(case.query["max_tokens"]),
            )
        )

    def _operands_for(self, case: SweepCase, device):
        import torch

        recipe = str(case.query["recipe"])
        k = int(case.query["in_features"])
        n = int(case.query["out_features"])
        if self._operands is None or (
            self._operands.recipe,
            self._operands.in_features,
            self._operands.out_features,
        ) != (recipe, k, n):
            generator = torch.Generator(device=device).manual_seed(
                self._context.settings.seed + int(case.case_id[-8:], 16) % 1_000_003
            )
            self._operands = _DenseLinearOperands(
                recipe=recipe,
                in_features=k,
                out_features=n,
                device=device,
                generator=generator,
            )
        return self._operands

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.gemm import dense_linear
        from b12x.gemm.dense_linear._policy import DenseLinearConfig
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        tokens = int(case.query["max_tokens"])
        with torch.cuda.device(self._context.device_ordinal):
            operands = self._operands_for(case, device)
            generator = torch.Generator(device=device).manual_seed(
                settings.seed + int(case.case_id[-8:], 16)
            )
            activation, activation_dequant = operands.activation(tokens, generator)
            baseline = operands.reference(activation_dequant)
            del activation_dequant
            torch.cuda.synchronize(device)
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            base_policy = PolicyContext.for_device(
                device, mode=PolicyMode.HEURISTIC_ONLY
            )
            caps = dense_linear.Caps(
                device=device,
                recipe=operands.recipe,
                in_features=operands.in_features,
                out_features=operands.out_features,
                max_tokens=tokens,
                output_dtype=torch.bfloat16,
            )
            measurements = []
            for candidate in candidates:
                try:
                    config = DenseLinearConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(DENSE_LINEAR, config)
                    plan = dense_linear.plan(caps, policy=policy)

                    def run(plan=plan):
                        return operands.run(activation, plan=plan, expected_m=tokens)

                    latency, correct, metrics = _graph_race(
                        run=run,
                        output_of=lambda out: out,
                        baseline=baseline,
                        settings=settings,
                        device=device,
                        flush=flush,
                    )
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=latency,
                            correct=correct,
                            metrics=metrics,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed plans survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)


class _DenseLinearFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        return _DenseLinearSession(context, tuple(cases))


class DenseLinearGenerator(DiscreteSweepGenerator):
    """Race the dense SM120 GEMM launch plans over the reviewed shape corpus."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=DENSE_LINEAR,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=(
                "recipe",
                "output_dtype",
                "in_features",
                "out_features",
                "max_tokens",
            ),
            range_fields=frozenset({"in_features", "out_features", "max_tokens"}),
            cases=_default_dense_linear_cases() if cases is None else cases,
            benchmark_factory=_DenseLinearFactory(),
            coverage={"corpus": "gemm_corpus.MODEL_LINEARS"},
            candidate_contract_version=1,
            nearest_range_bounds={"max_tokens": DENSE_MAX_TOKENS_BOUNDS},
            baseline_margin=_GEMM_BASELINE_MARGIN,
        )

    def baseline_config(self, case, context):
        from b12x.gemm.dense_linear._policy import DenseLinearQuery, _heuristic

        query = DenseLinearQuery(
            recipe=str(case.query["recipe"]),
            in_features=int(case.query["in_features"]),
            out_features=int(case.query["out_features"]),
            max_tokens=int(case.query["max_tokens"]),
            output_dtype=str(case.query["output_dtype"]),
        )
        return _heuristic(query, _device_identity(context)).to_dict()


# ---------------------------------------------------------------------------
# gemm.wo_projection: race the WO-A / WO-B decode chain launch options.
# ---------------------------------------------------------------------------

_WO_SMALL_TILES = ((0, 0), (16, 64), (16, 128), (32, 64))
_WO_LARGE_TILES = ((0, 0), (64, 128))


def _wo_projection_candidate_configs(max_tokens: int) -> tuple[dict[str, object], ...]:
    if max_tokens <= 16:
        wo_a_tiles = _WO_SMALL_TILES
        wo_b_tiles = _WO_SMALL_TILES
    else:
        wo_a_tiles = ((0, 0),)
        wo_b_tiles = _WO_LARGE_TILES
    fused_options = (False, True) if max_tokens <= 8 else (False,)
    quantized_options = (False, True) if max_tokens <= 16 else (False,)
    configs = []
    for wo_a in wo_a_tiles:
        for wo_b in wo_b_tiles:
            for fused in fused_options:
                for quantized in quantized_options:
                    if fused and quantized:
                        continue
                    configs.append(
                        {
                            "backend": "mxfp8",
                            "wo_a_tile_m": wo_a[0],
                            "wo_a_tile_n": wo_a[1],
                            "wo_b_tile_m": wo_b[0],
                            "wo_b_tile_n": wo_b[1],
                            "wo_b_fused_quant": fused,
                            "quantized_intermediate": quantized,
                        }
                    )
    return tuple(configs)


class _WoProjectionSession(AbstractContextManager["_WoProjectionSession"]):
    def __init__(self, context, cases: tuple[SweepCase, ...]) -> None:
        self._context = context
        self._cases = cases

    def __enter__(self) -> "_WoProjectionSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        return tuple(
            SweepCandidate.create(config)
            for config in _wo_projection_candidate_configs(
                int(case.query["max_tokens"])
            )
        )

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from benchmarks.benchmark_wo_projection import make_case
        from b12x.gemm import wo_projection
        from b12x.gemm.wo_projection._policy import WoProjectionConfig
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        tokens = int(case.query["max_tokens"])
        groups = int(case.query["groups"])
        group_width = int(case.query["group_width"])
        rank = int(case.query["rank"])
        hidden = int(case.query["hidden"])
        inv_rope = bool(case.metadata.get("inv_rope", False))
        nope_dim = int(case.metadata.get("nope_dim", 448))
        rope_dim = int(case.metadata.get("rope_dim", 64))
        with torch.cuda.device(self._context.device_ordinal):
            data = make_case(
                tokens=tokens,
                groups=groups,
                group_width=group_width,
                rank=rank,
                hidden=hidden,
                seed=settings.seed + int(case.case_id[-8:], 16) % 1_000_003,
                inv_rope=inv_rope,
                context_length=max(4096, tokens),
                nope_dim=nope_dim,
                rope_dim=rope_dim,
            )
            weights = data["weights"]
            caps = wo_projection.Caps(
                device=device,
                max_tokens=tokens,
                groups=groups,
                group_width=group_width,
                rank=rank,
                hidden=hidden,
            )
            base_policy = PolicyContext.for_device(
                device, mode=PolicyMode.HEURISTIC_ONLY
            )
            reference_plan = wo_projection.plan(caps, policy=base_policy)
            spec = reference_plan.scratch_specs()[0]
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

            def bind(plan):
                if inv_rope:
                    return wo_projection.bind_inv_rope(
                        plan,
                        scratch=scratch,
                        o=data["o"],
                        positions=data["positions"],
                        cos_sin_cache=data["cos_sin_cache"],
                        weights=weights,
                        heads_per_group=int(data["heads_per_group"]),
                        nope_dim=nope_dim,
                        rope_dim=rope_dim,
                        expected_m=tokens,
                    )
                return wo_projection.bind(
                    plan,
                    scratch=scratch,
                    source_tgd=data["x_tgd"],
                    weights=weights,
                    expected_m=tokens,
                )

            # Ground truth from the dequantized operands, not from the kernel
            # chain itself: the built-in plan is exactly what is under test.
            baseline = _wo_projection_reference(
                data, tokens=tokens, groups=groups, rank=rank
            )
            torch.cuda.synchronize(device)
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            measurements = []
            for candidate in candidates:
                try:
                    config = WoProjectionConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(WO_PROJECTION, config)
                    plan = wo_projection.plan(caps, policy=policy)
                    binding = bind(plan)
                    latency, correct, metrics = _graph_race(
                        run=binding.run,
                        output_of=lambda out: out,
                        baseline=baseline,
                        settings=settings,
                        device=device,
                        flush=flush,
                    )
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=latency,
                            correct=correct,
                            metrics=metrics,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed plans survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)


def _wo_projection_reference(data, *, tokens: int, groups: int, rank: int):
    """Float32 two-GEMM reference over the benchmark case's dequantized operands."""
    import torch

    tmp = torch.einsum(
        "tgd,grd->tgr",
        data["x_deq_tgd"].float(),
        data["wo_a_deq_grd"].float(),
    )
    return tmp.reshape(tokens, groups * rank) @ data["wo_b_deq_hgr"].float().T


class _WoProjectionFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        return _WoProjectionSession(context, tuple(cases))


class WoProjectionGenerator(DiscreteSweepGenerator):
    """Race the W_o projection decode-chain launch options per geometry."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=WO_PROJECTION,
            query_schema_version=1,
            config_schema_version=2,
            query_fields=(
                "dtype",
                "groups",
                "group_width",
                "rank",
                "hidden",
                "max_tokens",
            ),
            range_fields=frozenset({"max_tokens"}),
            cases=wo_projection_cases() if cases is None else cases,
            benchmark_factory=_WoProjectionFactory(),
            coverage={"corpus": "gemm_corpus.wo_projection_geometries"},
            candidate_contract_version=1,
            nearest_range_bounds={"max_tokens": DENSE_MAX_TOKENS_BOUNDS},
            baseline_margin=_GEMM_BASELINE_MARGIN,
        )

    def baseline_config(self, case, context):
        from b12x.gemm.wo_projection._policy import WoProjectionQuery, _heuristic

        query = WoProjectionQuery(
            dtype=str(case.query["dtype"]),
            max_tokens=int(case.query["max_tokens"]),
            groups=int(case.query["groups"]),
            group_width=int(case.query["group_width"]),
            rank=int(case.query["rank"]),
            hidden=int(case.query["hidden"]),
        )
        return _heuristic(query, _device_identity(context)).to_dict()


__all__ = [
    "Bf16VocabProjectionGenerator",
    "BlockFp8LinearGenerator",
    "DenseLinearGenerator",
    "WoProjectionGenerator",
]
