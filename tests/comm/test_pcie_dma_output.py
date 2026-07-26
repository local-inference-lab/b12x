from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sparkinfer.comm.pcie.pcie_dma import (
    PCIeDmaAllReduce,
    _persistent_output_view,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_persistent_output_view_reuses_exact_capacity_storage(
    dtype: torch.dtype,
) -> None:
    max_bytes = 4096
    storage = torch.empty(max_bytes, dtype=torch.uint8)
    elements = max_bytes // dtype.itemsize
    inp = torch.empty((16, elements // 16), dtype=dtype)

    output = _persistent_output_view(storage, inp, max_bytes)
    smaller = _persistent_output_view(storage, inp[:1], max_bytes)

    assert output.shape == inp.shape
    assert output.dtype == dtype
    assert output.is_contiguous()
    assert output.data_ptr() == storage.data_ptr()
    assert smaller.data_ptr() == output.data_ptr()
    assert storage.numel() == max_bytes
    assert output.untyped_storage().nbytes() == max_bytes


def test_persistent_output_view_rejects_request_over_capacity() -> None:
    storage = torch.empty(128, dtype=torch.uint8)
    inp = torch.empty(65, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="capacity is 128"):
        _persistent_output_view(storage, inp, 128)


def test_persistent_output_view_rejects_undersized_storage() -> None:
    storage = torch.empty(64, dtype=torch.uint8)
    inp = torch.empty(16, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="storage has 64 bytes"):
        _persistent_output_view(storage, inp, 128)


def test_persistent_output_view_rejects_wrong_storage_layout() -> None:
    inp = torch.empty(16, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="contiguous uint8"):
        _persistent_output_view(torch.empty(128), inp, 128)

    non_contiguous = torch.empty((128, 2), dtype=torch.uint8)[:, 0]
    with pytest.raises(ValueError, match="contiguous uint8"):
        _persistent_output_view(non_contiguous, inp, 128)


def test_close_releases_persistent_output_storage() -> None:
    class FakeCudaRuntime:
        def cudaIpcCloseMemHandle(self, _ptr: int) -> None:
            pass

        def cudaFree(self, _ptr: int) -> None:
            pass

    ring = object.__new__(PCIeDmaAllReduce)
    ring._closed = False
    ring._ipc = FakeCudaRuntime()
    ring._slab = SimpleNamespace(remote_ptrs=(1, 2), local_ptr=3)
    ring._output_storage = torch.empty(128, dtype=torch.uint8)

    ring.close()

    assert ring._closed
    assert ring._output_storage is None
