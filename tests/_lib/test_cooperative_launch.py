"""Cooperative grids respect compiled resources before capture."""

from types import SimpleNamespace

import pytest
import torch

from b12x._lib import cooperative


@pytest.mark.parametrize("resident", [1, 2])
@pytest.mark.parametrize("exact_smem", [False, True])
def test_residency_uses_launch_threads_and_exact_or_conservative_smem(
    monkeypatch, resident, exact_smem
):
    import cuda.bindings

    seen = []
    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        CUfunction_attribute=SimpleNamespace(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES="maximum_smem"
        ),
        CUlibrary=int,
        cuLibraryGetKernel=lambda library, symbol: (0, 23),
        cuKernelGetFunction=lambda kernel: (0, 29),
        cuFuncGetAttribute=lambda attribute, function: (0, 101376),
        cuOccupancyMaxActiveBlocksPerMultiprocessor=lambda function, threads, smem: (
            seen.append((function, threads, smem)) or (0, resident)
        ),
    )
    monkeypatch.setattr(cuda.bindings, "driver", driver)
    metadata = {
        "status": "exact" if exact_smem else "unknown",
        "launch_dynamic_smem_bytes": {"kernel": [49152]},
    }
    compiled = SimpleNamespace(
        kernel_info={"kernel": {}},
        to=lambda device: SimpleNamespace(
            jit_module=SimpleNamespace(cuda_library=[17])
        ),
        _b12x_launch_metadata=metadata,
    )
    assert cooperative._resident_blocks_per_sm(compiled, 192) == resident
    assert seen == [(29, 192, 49152 if exact_smem else 101376)]


def test_grid_residency_is_prepared_per_device_and_reused_during_capture(monkeypatch):
    compiled = SimpleNamespace()
    device, capturing = 0, False
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=70 if index == 0 else 48),
    )

    def residency(obj, threads):
        assert not capturing
        calls.append((obj, threads, device))
        return 1 if device == 0 else 2

    monkeypatch.setattr(cooperative, "_resident_blocks_per_sm", residency)
    assert cooperative.cooperative_grid_limit(compiled, 192) == 70
    capturing = True
    for _ in range(3):
        assert cooperative.cooperative_grid_limit(compiled, 192) == 70
    device = 1
    with pytest.raises(RuntimeError, match="must be warmed before capture"):
        cooperative.cooperative_grid_limit(compiled, 192)
    capturing = False
    assert cooperative.cooperative_grid_limit(compiled, 192) == 96
    assert [(threads, index) for _, threads, index in calls] == [(192, 0), (192, 1)]
