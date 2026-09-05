"""Typed component policy for mHC residual planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    MHC,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)

_MHC_MULT = 4
_MHC_MIXES = 24
_PREFILL_TF32_MIN_TOKENS = 384


@dataclass(frozen=True, kw_only=True)
class MhcQuery:
    dtype: str
    max_tokens: int
    hidden_size: int
    split_k: int


@dataclass(frozen=True, kw_only=True)
class MhcConfig:
    backend: str
    native_post_pre_backend: str
    decode_source_splits: int
    decode_tile_n: int
    decode_bf16x2: bool
    decode_partials_per_cta: int
    decode_finalize_threads: int
    decode_finalize_ctas: int
    prefill_block_m: int
    prefill_tile_n: int
    projection_tile_m: int
    projection_tile_n: int
    projection_tile_k: int
    projection_num_stages: int
    projection_num_m_warps: int
    projection_num_n_warps: int
    projection_k_splits: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "MhcConfig":
        expected = frozenset(cls.__dataclass_fields__)
        if frozenset(payload) != expected:
            raise ValueError(f"mHC config fields must be {sorted(expected)}")
        decode_bf16x2 = payload["decode_bf16x2"]
        if not isinstance(decode_bf16x2, bool):
            raise ValueError("mHC decode_bf16x2 profile field must be a bool")
        return cls(
            backend=str(payload["backend"]),
            native_post_pre_backend=str(payload["native_post_pre_backend"]),
            decode_source_splits=int(payload["decode_source_splits"]),
            decode_tile_n=int(payload["decode_tile_n"]),
            decode_bf16x2=decode_bf16x2,
            decode_partials_per_cta=int(payload["decode_partials_per_cta"]),
            decode_finalize_threads=int(payload["decode_finalize_threads"]),
            decode_finalize_ctas=int(payload["decode_finalize_ctas"]),
            prefill_block_m=int(payload["prefill_block_m"]),
            prefill_tile_n=int(payload["prefill_tile_n"]),
            projection_tile_m=int(payload["projection_tile_m"]),
            projection_tile_n=int(payload["projection_tile_n"]),
            projection_tile_k=int(payload["projection_tile_k"]),
            projection_num_stages=int(payload["projection_num_stages"]),
            projection_num_m_warps=int(payload["projection_num_m_warps"]),
            projection_num_n_warps=int(payload["projection_num_n_warps"]),
            projection_k_splits=int(payload["projection_k_splits"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "native_post_pre_backend": self.native_post_pre_backend,
            "decode_source_splits": self.decode_source_splits,
            "decode_tile_n": self.decode_tile_n,
            "decode_bf16x2": self.decode_bf16x2,
            "decode_partials_per_cta": self.decode_partials_per_cta,
            "decode_finalize_threads": self.decode_finalize_threads,
            "decode_finalize_ctas": self.decode_finalize_ctas,
            "prefill_block_m": self.prefill_block_m,
            "prefill_tile_n": self.prefill_tile_n,
            "projection_tile_m": self.projection_tile_m,
            "projection_tile_n": self.projection_tile_n,
            "projection_tile_k": self.projection_tile_k,
            "projection_num_stages": self.projection_num_stages,
            "projection_num_m_warps": self.projection_num_m_warps,
            "projection_num_n_warps": self.projection_num_n_warps,
            "projection_k_splits": self.projection_k_splits,
        }


def _tf32_config(
    *,
    backend: str = "tf32_tma",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    num_m_warps: int,
    num_n_warps: int,
    k_splits: int,
) -> MhcConfig:
    return MhcConfig(
        backend=backend,
        native_post_pre_backend="decode",
        decode_source_splits=0,
        decode_tile_n=0,
        decode_bf16x2=False,
        decode_partials_per_cta=4,
        decode_finalize_threads=0,
        decode_finalize_ctas=1,
        prefill_block_m=0,
        prefill_tile_n=0,
        projection_tile_m=tile_m,
        projection_tile_n=tile_n,
        projection_tile_k=tile_k,
        projection_num_stages=num_stages,
        projection_num_m_warps=num_m_warps,
        projection_num_n_warps=num_n_warps,
        projection_k_splits=k_splits,
    )


def _native_config(
    *,
    post_pre_backend: str,
    decode_source_splits: int = 0,
    decode_tile_n: int = 0,
    decode_bf16x2: bool = False,
    decode_partials_per_cta: int = 4,
    decode_finalize_threads: int = 0,
    decode_finalize_ctas: int = 1,
    prefill_block_m: int = 0,
    prefill_tile_n: int = 0,
) -> MhcConfig:
    return MhcConfig(
        backend="native",
        native_post_pre_backend=post_pre_backend,
        decode_source_splits=decode_source_splits,
        decode_tile_n=decode_tile_n,
        decode_bf16x2=decode_bf16x2,
        decode_partials_per_cta=decode_partials_per_cta,
        decode_finalize_threads=decode_finalize_threads,
        decode_finalize_ctas=decode_finalize_ctas,
        prefill_block_m=prefill_block_m,
        prefill_tile_n=prefill_tile_n,
        projection_tile_m=16,
        projection_tile_n=8,
        projection_tile_k=256,
        projection_num_stages=1,
        projection_num_m_warps=1,
        projection_num_n_warps=1,
        projection_k_splits=1,
    )


def _heuristic(
    query: MhcQuery,
    _device: DeviceIdentity | None,
) -> MhcConfig:
    tokens = int(query.max_tokens)
    hidden_size = int(query.hidden_size)
    backend = (
        "tf32_tma"
        if hidden_size in {4_096, 7_168} and tokens >= _PREFILL_TF32_MIN_TOKENS
        else "native"
    )
    if backend == "native":
        if tokens >= 96:
            return _native_config(
                post_pre_backend="prefill_block_m",
                prefill_block_m=2,
                prefill_tile_n=12 if hidden_size == 7_168 else 24,
            )
        return _native_config(post_pre_backend="decode")
    if hidden_size == 4_096 and tokens >= 8_192:
        return _tf32_config(
            backend=backend,
            tile_m=128,
            tile_n=24,
            tile_k=64,
            num_stages=2,
            num_m_warps=8,
            num_n_warps=1,
            k_splits=4,
        )
    if hidden_size == 4_096 and tokens >= 3_584:
        return _tf32_config(
            backend=backend,
            tile_m=192,
            tile_n=24,
            tile_k=64,
            num_stages=2,
            num_m_warps=12,
            num_n_warps=1,
            k_splits=8,
        )
    if hidden_size == 4_096 and tokens >= 2_304:
        return _tf32_config(
            backend=backend,
            tile_m=64,
            tile_n=24,
            tile_k=64,
            num_stages=2 if tokens >= 3_072 else 3,
            num_m_warps=4,
            num_n_warps=1,
            k_splits=8,
        )
    if hidden_size != 4_096 and tokens >= 4_096:
        return _tf32_config(
            backend=backend,
            tile_m=32,
            tile_n=8,
            tile_k=256,
            num_stages=1,
            num_m_warps=2,
            num_n_warps=1,
            k_splits=1,
        )
    return _tf32_config(
        backend=backend,
        tile_m=16,
        tile_n=8,
        tile_k=256,
        num_stages=1,
        num_m_warps=1,
        num_n_warps=1,
        k_splits=1,
    )


def _encode(query: MhcQuery) -> dict[str, object]:
    return {
        "dtype": query.dtype,
        "max_tokens": query.max_tokens,
        "hidden_size": query.hidden_size,
        "split_k": query.split_k,
    }


def _validate(
    query: MhcQuery,
    config: MhcConfig,
    _device: DeviceIdentity | None,
) -> None:
    if query.dtype != "bfloat16":
        raise ValueError(f"mHC requires bfloat16 outputs, got {query.dtype!r}")
    if query.max_tokens <= 0 or query.hidden_size <= 0 or query.split_k <= 0:
        raise ValueError("mHC query dimensions must be positive")
    if config.backend not in {"native", "tf32_tma"}:
        raise ValueError(f"unsupported mHC backend {config.backend!r}")
    if not isinstance(config.decode_bf16x2, bool):
        raise ValueError("decode_bf16x2 must be a bool")
    if config.native_post_pre_backend not in {"decode", "prefill_block_m"}:
        raise ValueError(
            "native_post_pre_backend must be 'decode' or 'prefill_block_m'"
        )
    if config.native_post_pre_backend == "decode":
        if config.prefill_block_m != 0 or config.prefill_tile_n != 0:
            raise ValueError("decode mHC configs must disable block-M prefill geometry")
        if config.decode_source_splits == 0:
            if config.decode_tile_n != 0:
                raise ValueError("unsplit mHC decode must set decode_tile_n=0")
            if config.decode_bf16x2:
                raise ValueError("unsplit mHC decode must disable decode_bf16x2")
        else:
            if query.hidden_size != 4_096:
                raise ValueError("split mHC decode supports hidden_size=4096")
            if config.decode_source_splits not in {4, 8}:
                raise ValueError("decode_source_splits must be 0, 4, or 8")
            if config.decode_tile_n <= 0 or _MHC_MIXES % config.decode_tile_n:
                raise ValueError("decode_tile_n must divide the 24 mHC projections")
            if config.decode_partials_per_cta != 4:
                raise ValueError(
                    "split mHC decode must use decode_partials_per_cta=4"
                )
        if not 1 <= config.decode_partials_per_cta <= 25:
            raise ValueError("decode_partials_per_cta must be in [1, 25]")
        if config.decode_finalize_threads != 0:
            if (
                config.decode_finalize_threads <= 0
                or config.decode_finalize_threads > 1_024
                or config.decode_finalize_threads % 32
                or query.hidden_size % (2 * config.decode_finalize_threads)
            ):
                raise ValueError(
                    "decode_finalize_threads must be a positive multiple of 32 "
                    "that evenly vectorizes hidden_size"
                )
            if config.decode_finalize_ctas <= 0:
                raise ValueError("decode_finalize_ctas must be positive")
        elif config.decode_finalize_ctas != 1:
            raise ValueError(
                "multi-CTA mHC finalize must set decode_finalize_ctas=1"
            )
    else:
        if (
            config.decode_source_splits != 0
            or config.decode_tile_n != 0
            or config.decode_bf16x2
            or config.decode_partials_per_cta != 4
            or config.decode_finalize_threads != 0
            or config.decode_finalize_ctas != 1
        ):
            raise ValueError(
                "block-M prefill mHC configs must use the inactive decode schedule"
            )
        if config.prefill_block_m <= 0:
            raise ValueError("prefill_block_m must be positive")
        if config.prefill_tile_n <= 0 or _MHC_MIXES % config.prefill_tile_n:
            raise ValueError("prefill_tile_n must divide the 24 mHC projections")
    if config.backend == "native":
        if (
            config.projection_tile_m,
            config.projection_tile_n,
            config.projection_tile_k,
            config.projection_num_stages,
            config.projection_num_m_warps,
            config.projection_num_n_warps,
            config.projection_k_splits,
        ) != (16, 8, 256, 1, 1, 1, 1):
            raise ValueError(
                "native mHC configs must use canonical projection geometry"
            )
        return
    if config.native_post_pre_backend != "decode" or any(
        (
            config.decode_source_splits,
            config.decode_tile_n,
            config.decode_bf16x2,
            config.decode_partials_per_cta != 4,
            config.decode_finalize_threads,
            config.decode_finalize_ctas != 1,
            config.prefill_block_m,
            config.prefill_tile_n,
        )
    ):
        raise ValueError("TF32 mHC configs must use an unsplit decode fallback")
    if query.hidden_size not in {4_096, 7_168}:
        raise ValueError("TF32 mHC projection supports hidden sizes 4096 and 7168")
    if config.projection_num_stages not in range(1, 9):
        raise ValueError("projection_num_stages must be in [1, 8]")
    if config.projection_num_m_warps <= 0 or config.projection_num_n_warps <= 0:
        raise ValueError("projection warp counts must be positive")
    if config.projection_tile_m != 16 * config.projection_num_m_warps:
        raise ValueError("projection_tile_m must equal 16 * projection_num_m_warps")
    if config.projection_tile_n <= 0 or config.projection_tile_n % 8:
        raise ValueError("projection_tile_n must be a positive multiple of 8")
    n_mma_tiles = config.projection_tile_n // 8
    if n_mma_tiles % config.projection_num_n_warps:
        raise ValueError("projection_num_n_warps must divide projection_tile_n / 8")
    total_k = _MHC_MULT * query.hidden_size
    if config.projection_tile_k <= 0 or config.projection_tile_k % 8:
        raise ValueError("projection_tile_k must be a positive multiple of 8")
    if total_k % config.projection_tile_k:
        raise ValueError("projection_tile_k must divide the flattened hidden width")
    k_tiles = total_k // config.projection_tile_k
    if config.projection_k_splits <= 0 or k_tiles % config.projection_k_splits:
        raise ValueError("projection_k_splits must divide the projection K tiles")
    if config.projection_k_splits >= query.split_k:
        raise ValueError("projection_k_splits must be smaller than scratch split_k")
    compute_warps = config.projection_num_m_warps * config.projection_num_n_warps
    if (compute_warps + 1) * 32 > 1_024:
        raise ValueError("projection geometry exceeds the CUDA thread-block limit")


MHC_POLICY = ComponentPolicy(
    component_id=MHC,
    query_schema_version=1,
    config_schema_version=4,
    query_fields=frozenset(MhcQuery.__dataclass_fields__),
    config_fields=frozenset(MhcConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=MhcConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = ["MHC_POLICY", "MhcConfig", "MhcQuery"]
