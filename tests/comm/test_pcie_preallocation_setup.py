from __future__ import annotations

import pytest
import torch

import sparkinfer.comm.pcie.pcie_dcp_a2a as dcp
import sparkinfer.comm.pcie.pcie_oneshot as oneshot
import sparkinfer.comm.pcie.pcie_twoshot as twoshot


def _mock_collective_failure(monkeypatch) -> None:
    def exchange(
        local_error,
        *,
        exchange_group,
        phase,
        exports_retained_on_exchange_failure=True,
    ):
        assert local_error is not None
        assert not exports_retained_on_exchange_failure
        return ((f"RuntimeError: {local_error}",), ())

    monkeypatch.setattr(oneshot, "_exchange_setup_failures", exchange)


def _fail_cuda_runtime():
    raise RuntimeError("injected one-rank CUDA runtime load failure")


def test_oneshot_factory_coordinates_preallocation_failure_before_ipc(
    monkeypatch,
) -> None:
    _mock_collective_failure(monkeypatch)
    monkeypatch.setattr(oneshot.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(oneshot.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(oneshot, "_require_full_grid_residency", lambda **kwargs: None)
    monkeypatch.setattr(oneshot, "CudaRTLibrary", _fail_cuda_runtime)
    monkeypatch.setattr(
        oneshot.PCIeOneshotAllReduce,
        "_allocate_eager_channel_buffers",
        lambda *args, **kwargs: pytest.fail("IPC allocation must not start"),
    )

    with pytest.raises(RuntimeError, match="pre-allocation setup") as exc:
        oneshot.PCIeOneshotAllReduce.from_exchange_group(
            exchange_group=object(),
            device=torch.device("cuda:0"),
        )

    assert "exports were retained" not in str(exc.value)


def test_dcp_factory_coordinates_preallocation_failure_before_ipc(monkeypatch) -> None:
    _mock_collective_failure(monkeypatch)
    monkeypatch.setattr(dcp.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dcp.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dcp, "_require_full_grid_residency", lambda **kwargs: None)
    monkeypatch.setattr(dcp, "CudaRTLibrary", _fail_cuda_runtime)
    monkeypatch.setattr(
        oneshot.PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        lambda *args, **kwargs: pytest.fail("IPC allocation must not start"),
    )

    with pytest.raises(RuntimeError, match="pre-allocation setup") as exc:
        dcp.PCIeDCPA2A.from_exchange_group(
            exchange_group=object(),
            device=torch.device("cuda:0"),
            max_batch_size=8,
            total_heads=16,
            head_dim=64,
        )

    assert "exports were retained" not in str(exc.value)


def test_twoshot_factory_coordinates_preallocation_failure_before_ipc(
    monkeypatch,
) -> None:
    _mock_collective_failure(monkeypatch)
    monkeypatch.setattr(twoshot.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(twoshot.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(twoshot, "_require_full_grid_residency", lambda **kwargs: None)
    monkeypatch.setattr(twoshot, "CudaRTLibrary", _fail_cuda_runtime)
    monkeypatch.setattr(
        oneshot.PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        lambda *args, **kwargs: pytest.fail("IPC allocation must not start"),
    )

    with pytest.raises(RuntimeError, match="pre-allocation setup") as exc:
        twoshot.PCIeTwoShotSP.from_exchange_group(
            exchange_group=object(),
            device=torch.device("cuda:0"),
            max_rows=8,
            row_elems=16,
        )

    assert "exports were retained" not in str(exc.value)


@pytest.mark.parametrize("module_name", ("oneshot", "dcp"))
def test_direct_constructor_coordinates_argument_failure_before_residency(
    monkeypatch,
    module_name: str,
) -> None:
    _mock_collective_failure(monkeypatch)
    if module_name == "oneshot":
        monkeypatch.setattr(
            oneshot,
            "_require_full_grid_residency",
            lambda **kwargs: pytest.fail("residency gate must not start"),
        )
        def constructor():
            return oneshot.PCIeOneshotAllReduce(
                rank=0,
                world_size=2,
                device=torch.device("cuda:0"),
                signal_ptrs=(100,),
                exchange_group=object(),
                ext_module=object(),
            )
    else:
        monkeypatch.setattr(
            dcp,
            "_require_full_grid_residency",
            lambda **kwargs: pytest.fail("residency gate must not start"),
        )
        def constructor():
            return dcp.PCIeDCPA2A(
                rank=0,
                world_size=2,
                device=torch.device("cuda:0"),
                signal_ptrs=(100,),
                staging0_ptrs=(200, 300),
                staging1_ptrs=(400, 500),
                max_batch_size=8,
                total_heads=16,
                head_dim=64,
                output_capacity_elems=8 * 16 * 64,
                lse_offset=8 * 16 * 64 * 2,
                lse_capacity=8 * 16,
                exchange_group=object(),
                ext_module=object(),
            )

    with pytest.raises(RuntimeError, match="argument validation"):
        constructor()


@pytest.mark.parametrize(
    ("module", "class_name"),
    (
        (oneshot, "PCIeOneshotAllReduce"),
        (dcp, "PCIeDCPA2A"),
    ),
)
def test_exported_direct_constructor_contains_capability_and_setup_coordination(
    module,
    class_name: str,
) -> None:
    import inspect

    source = inspect.getsource(getattr(module, class_name).__init__)
    assert "_require_full_grid_residency(" in source
    assert "_run_collective_preallocation_setup(" in source
    assert "_finish_collective_unowned_runtime_setup(" in source


def test_oneshot_direct_cuda_constructor_executes_all_collective_gates(
    monkeypatch,
) -> None:
    events = []

    class FakeTensor:
        pass

    class FakeExt:
        def init_custom_ar(self, signal_ptrs, rank_data, rank):
            events.append("native-init")
            return 123

    class FakeIPC:
        def cudaSetDevice(self, device):
            events.append(("set-device", device))

    monkeypatch.setattr(
        oneshot,
        "_require_full_grid_residency",
        lambda **kwargs: events.append("residency"),
    )
    monkeypatch.setattr(
        oneshot,
        "_run_collective_preallocation_setup",
        lambda **kwargs: (events.append("preflight"), kwargs["setup"]())[1],
    )
    monkeypatch.setattr(
        oneshot,
        "_require_collective_contract",
        lambda **kwargs: events.append("contract"),
    )
    monkeypatch.setattr(
        oneshot,
        "_finish_collective_unowned_runtime_setup",
        lambda **kwargs: events.append("native-verdict"),
    )
    monkeypatch.setattr(oneshot.torch, "empty", lambda *args, **kwargs: FakeTensor())

    runtime = oneshot.PCIeOneshotAllReduce(
        rank=0,
        world_size=2,
        device=torch.device("cuda:0"),
        signal_ptrs=(100, 200),
        exchange_group=object(),
        ipc=FakeIPC(),
        ext_module=FakeExt(),
    )

    assert runtime._ptr == 123
    assert events == [
        "preflight",
        "residency",
        "preflight",
        ("set-device", 0),
        "contract",
        "native-init",
        "native-verdict",
    ]
    runtime._coordinated_close_complete = True


def test_dcp_direct_cuda_constructor_executes_all_collective_gates(
    monkeypatch,
) -> None:
    events = []

    class FakeExt:
        def init_dcp_a2a(self, *args):
            events.append("native-init")
            return 456

    monkeypatch.setattr(
        dcp,
        "_require_full_grid_residency",
        lambda **kwargs: events.append("residency"),
    )
    monkeypatch.setattr(
        dcp,
        "_run_collective_preallocation_setup",
        lambda **kwargs: (events.append("preflight"), kwargs["setup"]())[1],
    )
    monkeypatch.setattr(
        dcp,
        "_require_collective_contract",
        lambda **kwargs: events.append("contract"),
    )
    monkeypatch.setattr(
        dcp,
        "_finish_collective_unowned_runtime_setup",
        lambda **kwargs: events.append("native-verdict"),
    )

    runtime = dcp.PCIeDCPA2A(
        rank=0,
        world_size=2,
        device=torch.device("cuda:0"),
        signal_ptrs=(100, 200),
        staging0_ptrs=(300, 400),
        staging1_ptrs=(500, 600),
        max_batch_size=8,
        total_heads=16,
        head_dim=64,
        output_capacity_elems=8 * 16 * 64,
        lse_offset=8 * 16 * 64 * 2,
        lse_capacity=8 * 16,
        exchange_group=object(),
        ext_module=FakeExt(),
    )

    assert runtime._ptr == 456
    assert events == [
        "preflight",
        "residency",
        "preflight",
        "contract",
        "native-init",
        "native-verdict",
    ]
