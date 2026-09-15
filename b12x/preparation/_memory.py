"""Allocator counters and cleanup of retired preparation graph pools."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _counter():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("preparation memory accounting requires CUDA headers; set CUDA_HOME")
    # Key the build directory by source content: torch's staleness check is
    # mtime-based and does not survive a persisted TORCH_EXTENSIONS_DIR, so a
    # changed .cpp next to an old cached .so silently loaded stale symbols.
    source = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha1(source.read_bytes()).hexdigest()[:12]
    return load(
        name=f"b12x_preparation_memory_{digest}",
        sources=[str(source)],
        extra_include_paths=[str(Path(CUDA_HOME) / "include")],
        extra_ldflags=["-lc10_cuda", "-ltorch_cuda"],
        with_cuda=False,
    )


def allocated_bytes(device_ordinal: int) -> int:
    return int(_counter().allocated_bytes(device_ordinal))


def release_graph_pool_cache(pool_id):
    """Return retired graph blocks to CUDA while retaining the default pool cache."""
    _counter().release_graph_pool_cache(*pool_id)
