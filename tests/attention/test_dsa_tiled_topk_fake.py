import torch
import pytest

from b12x._lib.compile_plan import compile_only_launches
from b12x.attention.dsa_indexer import contiguous_kernel, tiled_topk
from b12x.attention.dsa_indexer.tiled_topk import _buffers_alias


def test_indexer_dlpack_helpers_skip_fake_storage(monkeypatch) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    def fail(*args, **kwargs):
        raise AssertionError("fake tensor reached DLPack")

    monkeypatch.setattr(contiguous_kernel, "from_dlpack", fail)
    monkeypatch.setattr(tiled_topk, "from_dlpack", fail)
    with FakeTensorMode(), compile_only_launches():
        values = torch.empty((2, 4), device="cuda")
        contiguous_kernel._to_kernel_tensor(values, contiguous_kernel.cutlass.Float32)
        tiled_topk._to_kernel_tensor(values, tiled_topk.cutlass.Float32)
        assert not torch.cuda.is_initialized()


def test_tiled_topk_alias_check_skips_fake_storage() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        values = torch.empty((1, 2), device="cuda")
        indices = torch.empty((1, 2), dtype=torch.int32, device="cuda")
        assert not _buffers_alias(values, values)
        assert not _buffers_alias(indices, indices)


def test_tiled_topk_alias_check_rejects_real_alias() -> None:
    values = torch.empty((1, 2))
    with pytest.raises(ValueError):
        if _buffers_alias(values, values):
            raise ValueError("aliased buffers")
