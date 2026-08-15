from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from b12x.moe._shared.kernels.w4a16.host import (
    route_pack_capacity,
    route_pack_token_capacity,
    route_pack_warmup_token_counts,
)
from b12x.moe.fused_moe._impl import _w4a16_route_pack_prewarm_specs


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [
        (1, (1,)),
        (5, (1, 2, 3, 4, 5)),
        (6, (1, 2, 3, 4, 5, 6)),
        (32, (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)),
        (
            3072,
            (
                1,
                2,
                3,
                4,
                5,
                8,
                9,
                16,
                17,
                32,
                33,
                64,
                65,
                128,
                129,
                256,
                257,
                512,
                513,
                1024,
                1025,
                2048,
                2049,
                3072,
            ),
        ),
    ],
)
def test_route_pack_warmup_covers_capacity_buckets(
    capacity: int, expected: tuple[int, ...]
) -> None:
    assert route_pack_warmup_token_counts(capacity) == expected


def test_route_pack_warmup_rejects_empty_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        route_pack_warmup_token_counts(0)


def test_non_power_of_two_capacity_has_complete_warmup_signatures() -> None:
    capacity = 3072
    topk = 8
    warm_counts = route_pack_warmup_token_counts(capacity)
    warm_signatures = {
        (
            route_pack_token_capacity(
                tokens,
                topk,
                max_tokens=capacity,
            ),
            (tokens * topk) % 16 == 0,
        )
        for tokens in warm_counts
    }

    live_signatures = {
        (
            route_pack_token_capacity(
                tokens,
                topk,
                max_tokens=capacity,
            ),
            (tokens * topk) % 16 == 0,
        )
        for tokens in range(1, capacity + 1)
    }
    assert live_signatures <= warm_signatures


def test_full_rotation_prewarm_specs_cover_dtype_mapping_and_live_alignment() -> None:
    specs = _w4a16_route_pack_prewarm_specs(
        3072,
        8,
        full_rotation=True,
    )
    warm_counts = route_pack_warmup_token_counts(3072)

    assert len(specs) == len(warm_counts) * 4
    assert {(dtype, mapped) for dtype, mapped, _, _ in specs} == {
        (torch.int32, False),
        (torch.int32, True),
        (torch.int64, False),
        (torch.int64, True),
    }
    for dtype in (torch.int32, torch.int64):
        for mapped in (False, True):
            assert [
                (live_tokens, token_capacity)
                for spec_dtype, spec_mapped, live_tokens, token_capacity in specs
                if spec_dtype == dtype and spec_mapped == mapped
            ] == [
                (
                    live_tokens,
                    route_pack_token_capacity(
                        live_tokens,
                        8,
                        max_tokens=3072,
                    ),
                )
                for live_tokens in warm_counts
            ]


@pytest.mark.parametrize("tokens", [2049, 2050, 3071, 3072])
def test_partial_top_bucket_uses_declared_capacity(tokens: int) -> None:
    topk = 8
    routed_capacity, _, _ = route_pack_capacity(
        tokens * topk,
        block_size=64,
        num_experts=256,
        topk=topk,
        max_tokens=3072,
    )
    assert routed_capacity == 3072 * topk


def test_route_pack_capacity_rejects_live_rows_above_limit() -> None:
    with pytest.raises(ValueError, match="below the live token count"):
        route_pack_token_capacity(3073, 8, max_tokens=3072)


def test_mixed_warmup_checks_capture_on_buffer_device(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis

    state = {"device": 0}

    @contextmanager
    def device_guard(device: torch.device):
        previous = state["device"]
        state["device"] = device.index
        try:
            yield
        finally:
            state["device"] = previous

    monkeypatch.setattr(mixed_trellis.torch.cuda, "device", device_guard)
    monkeypatch.setattr(
        mixed_trellis.torch.cuda,
        "is_current_stream_capturing",
        lambda: state["device"] == 1,
    )
    buffers = SimpleNamespace(
        packed_route_indices=SimpleNamespace(device=torch.device("cuda:1"))
    )
    launch = SimpleNamespace(tier0_num_experts=1, tier1_num_experts=1)

    with pytest.raises(RuntimeError, match="cannot run during capture"):
        mixed_trellis.warmup_mixed_trellis_route_pack(
            launch,
            buffers,
            expert_map=object(),
        )


def test_mixed_warmup_retries_failed_sync_and_keys_route_dtype(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis

    @contextmanager
    def device_guard(device: torch.device):
        del device
        yield

    sync_attempts = 0

    def synchronize() -> None:
        nonlocal sync_attempts
        sync_attempts += 1
        if sync_attempts == 1:
            raise RuntimeError("injected sync failure")

    calls: list[tuple[torch.dtype, int]] = []

    def pack_routes(topk_ids, block_size, num_experts, **kwargs) -> None:
        del block_size, num_experts
        calls.append((topk_ids.dtype, kwargs["max_tokens"]))

    monkeypatch.setattr(mixed_trellis, "_ROUTE_PACK_WARMED", set())
    monkeypatch.setattr(mixed_trellis.torch.cuda, "device", device_guard)
    monkeypatch.setattr(
        mixed_trellis.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(mixed_trellis.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        mixed_trellis.torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=synchronize),
    )
    monkeypatch.setattr(
        mixed_trellis.torch,
        "zeros",
        lambda shape, *, dtype, device: SimpleNamespace(
            shape=shape,
            dtype=dtype,
            device=device,
        ),
    )
    monkeypatch.setattr(mixed_trellis, "pack_topk_routes_by_expert", pack_routes)

    buffers = SimpleNamespace(
        packed_route_indices=SimpleNamespace(device=torch.device("cuda:0")),
        block_expert_ids=object(),
        packed_route_count=object(),
        expert_offsets=object(),
        expert_counts=object(),
    )
    launch = SimpleNamespace(
        tier0_num_experts=3,
        tier1_num_experts=2,
        route_ids_dtype=torch.int64,
        size_m=6,
        top_k=8,
        moe_block_size=8,
    )

    with pytest.raises(RuntimeError, match="injected sync failure"):
        mixed_trellis.warmup_mixed_trellis_route_pack(
            launch,
            buffers,
            expert_map=object(),
        )
    assert not mixed_trellis._ROUTE_PACK_WARMED

    assert (
        mixed_trellis.warmup_mixed_trellis_route_pack(
            launch,
            buffers,
            expert_map=object(),
        )
        == 6
    )
    int32_launch = SimpleNamespace(**{**vars(launch), "route_ids_dtype": torch.int32})
    assert (
        mixed_trellis.warmup_mixed_trellis_route_pack(
            int32_launch,
            buffers,
            expert_map=object(),
        )
        == 6
    )
    assert calls == [(torch.int64, 6)] * 12 + [(torch.int32, 6)] * 6
