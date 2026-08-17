import pytest
import torch

from b12x.moe.fused_moe._route_pack_cache import route_pack_prewarm_key


def _key(
    *,
    token_count: int = 3,
    top_k: int = 1,
    packed_route_slots: int = 256,
    route_blocks: int = 4,
) -> tuple[object, ...]:
    return route_pack_prewarm_key(
        "cuda",
        0,
        torch.int32,
        token_count,
        top_k,
        packed_route_slots,
        route_blocks,
        64,
        4,
        False,
    )


def test_route_pack_prewarm_key_includes_each_launch_dimension() -> None:
    baseline = _key()
    assert baseline != _key(token_count=4)
    assert baseline != _key(top_k=2)
    assert baseline != _key(packed_route_slots=320)
    assert baseline != _key(route_blocks=5)


def test_route_pack_prewarm_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        _key(top_k=0)
