"""Allocator counters and cleanup of retired preparation graph pools."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _counter():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("preparation memory accounting requires CUDA headers; set CUDA_HOME")
    return load(
        name="b12x_preparation_memory",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        extra_include_paths=[str(Path(CUDA_HOME) / "include")],
        extra_ldflags=["-lc10_cuda", "-ltorch_cuda"],
        with_cuda=False,
    )


def allocated_bytes(device_ordinal: int) -> int:
    return int(_counter().allocated_bytes(device_ordinal))


def release_graph_pool_cache(pool_id):
    """Return retired graph blocks to CUDA while retaining the default pool cache."""
    _counter().release_graph_pool_cache(*pool_id)
