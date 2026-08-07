import pytest

from b12x.moe._shared.kernels.w4a16.host import (
    route_pack_warmup_token_counts,
)


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [
        (1, (1,)),
        (5, (1, 2, 4, 5)),
        (32, (1, 2, 4, 8, 16, 32)),
        (3072, (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072)),
    ],
)
def test_route_pack_warmup_covers_capacity_buckets(
    capacity: int, expected: tuple[int, ...]
) -> None:
    assert route_pack_warmup_token_counts(capacity) == expected


def test_route_pack_warmup_rejects_empty_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        route_pack_warmup_token_counts(0)
