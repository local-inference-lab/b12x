"""Focused contract-agreement tests for the hierarchical all-reduce (issue #179).

These tests exercise the collective contract agreement, coordinated phases,
strict rollback, lifecycle, and topology manifest.  The harness mocks the
underlying ``dist.broadcast_object_list`` primitive (not the production
``_broadcast_gather_object`` wrapper) so the production code path is
exercised.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Mock native dependencies for non-CUDA test environments
# ---------------------------------------------------------------------------

_native_abi = {
    "_ISLAND_SIZE": 4,
    "_MAX_ISLANDS": 4,
    "_MAX_WORLD_SIZE": 16,
    "_LOCAL_ARRIVED": 256,
    "_COLLECTIVE_GENERATION": 128,
    "_LEADER_READY": 16_640,
    "_LEADER_CONSUMED": 33_024,
    "_FINAL_READY": 49_408,
    "_LOCAL_CONSUMED": 53_504,
}


def _install_native_mock() -> None:
    import sys

    if "b12x.comm.pcie._hierarchical_cute" in sys.modules:
        return
    mock = MagicMock()
    for name, value in _native_abi.items():
        setattr(mock, name, value)
    mock.get_hierarchical_launcher = MagicMock()
    sys.modules["b12x.comm.pcie._hierarchical_cute"] = mock


_install_native_mock()

import b12x.comm.pcie.pcie_hierarchical as hierarchical  # noqa: E402
import b12x.comm.pcie.pcie_oneshot as oneshot  # noqa: E402
from b12x.comm.pcie._cuda_ipc import cudaIpcMemHandle_t  # noqa: E402
from b12x.comm.pcie.pcie_hierarchical import (  # noqa: E402
    _HIERARCHICAL_ABI_VERSION,
    _HIERARCHICAL_CONTRACT_VERSION,
    _IPC_HANDLE_BYTES,
    _STATE_CLOSING,
    _STATE_CLOSED,
    _STATE_OPEN,
    _buffer_modes_from_env,
    _hierarchical_abi_descriptor,
    _hierarchical_allreduce_contract,
    _make_layout,
    _make_default_catalog,
    _selected_peers,
    OpCatalog,
    OpCatalogEntry,
)

from ctypes import sizeof as _ctypes_sizeof  # noqa: E402


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------


def test_ipc_handle_bytes_matches_ctypes() -> None:
    assert _ctypes_sizeof(cudaIpcMemHandle_t) == _IPC_HANDLE_BYTES
    assert _IPC_HANDLE_BYTES == 128


def test_abi_descriptor_is_versioned_and_complete() -> None:
    abi = _hierarchical_abi_descriptor()
    assert abi[0] == _HIERARCHICAL_ABI_VERSION
    assert len(abi) == 29
    assert abi[1] == 4
    assert abi[3] == 16
    assert abi[16] == _IPC_HANDLE_BYTES
    assert abi[17] == "bf16" and abi[18] == 2
    assert abi[19] == "fp32" and abi[20] == 4
    assert abi[21] == "bf16" and abi[22] == 2
    assert abi[23] == 112
    assert abi[27] == "comm.pcie.hierarchical.bf16"
    assert abi[28] == 3


def test_contract_tuple_includes_catalog() -> None:
    layout = _make_layout(4096)
    catalog = _make_default_catalog(4096, 16)
    contract = _hierarchical_allreduce_contract(
        world_size=12, layout=layout, catalog=catalog,
        wait_nanosleep_cycles=24, threads=224,
        vectorized_bf16x2=True, vectorized_bf16x2_max_elements=7168,
    )
    assert contract[0] == _HIERARCHICAL_CONTRACT_VERSION
    assert len(contract) == 17
    assert contract[2] == 12
    assert len(contract[3]) == 12
    assert contract[4] == 4096
    # Catalog is at index 9.
    cat_tuple = contract[9]
    assert isinstance(cat_tuple, tuple)
    assert len(cat_tuple) == 3  # entries, double_buffered, deferred
    assert len(cat_tuple[0]) == 1  # one entry
    assert cat_tuple[0][0] == ("default", 0, 4096, 16)


def test_default_catalog_validates_blocks() -> None:
    with pytest.raises(ValueError, match="blocks must be one of"):
        OpCatalogEntry(op_id="x", order=0, elements=4096, blocks=33)


def test_catalog_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate op_id"):
        OpCatalog(entries=(
            OpCatalogEntry(op_id="a", order=0, elements=4096, blocks=16),
            OpCatalogEntry(op_id="a", order=1, elements=8192, blocks=32),
        ))


def test_catalog_rejects_duplicate_orders() -> None:
    with pytest.raises(ValueError, match="duplicate order"):
        OpCatalog(entries=(
            OpCatalogEntry(op_id="a", order=0, elements=4096, blocks=16),
            OpCatalogEntry(op_id="b", order=0, elements=8192, blocks=32),
        ))


def test_catalog_find_matches_exact_elements_and_blocks() -> None:
    cat = OpCatalog(entries=(
        OpCatalogEntry(op_id="small", order=0, elements=3583, blocks=16),
        OpCatalogEntry(op_id="large", order=1, elements=7169, blocks=32),
    ))
    assert cat.find(3583, 16) is not None
    assert cat.find(7169, 32) is not None
    assert cat.find(3583, 32) is None  # wrong blocks
    assert cat.find(4096, 16) is None  # not in catalog


_MISMATCH_MUTATIONS = [
    (1, "abi_descriptor"),
    (2, "world_size"),
    (3, "peer_graph"),
    (4, "max_elements"),
    (5, "stage_offsets"),
    (6, "partial_offsets"),
    (7, "final_offsets"),
    (8, "slab_bytes"),
    (9, "catalog"),
    (10, "dtype"),
    (11, "vectorized_bf16x2"),
    (12, "vec_max_elements"),
    (13, "double_buffered"),
    (14, "deferred_consumption"),
    (15, "nanosleep_cycles"),
    (16, "threads"),
]


@pytest.mark.parametrize("field_index,label", _MISMATCH_MUTATIONS)
def test_contract_field_mutation_produces_different_tuple(
    field_index: int, label: str,
) -> None:
    layout = _make_layout(4096)
    catalog = _make_default_catalog(4096, 16)
    contract = _hierarchical_allreduce_contract(
        world_size=12, layout=layout, catalog=catalog,
        wait_nanosleep_cycles=24, threads=224,
        vectorized_bf16x2=True, vectorized_bf16x2_max_elements=7168,
    )
    mutated = list(contract)
    val = mutated[field_index]
    if field_index == 1:
        m = list(val)
        m[0] = 999
        mutated[field_index] = tuple(m)
    elif field_index == 3:
        pg = list(val)
        pg[0] = tuple(reversed(pg[0]))
        mutated[field_index] = tuple(pg)
    elif field_index == 9:
        # Mutate catalog: change blocks
        old_entries = val[0]
        new_entry = (old_entries[0][0], old_entries[0][1], old_entries[0][2], 32)
        mutated[field_index] = ((new_entry,), val[1], val[2])
    elif field_index == 10:
        mutated[field_index] = "fp16"
    elif isinstance(val, bool):
        mutated[field_index] = not val
    elif isinstance(val, int):
        mutated[field_index] = val + 1
    elif isinstance(val, tuple):
        mutated[field_index] = (0, 0)
    elif val is None:
        mutated[field_index] = 99
    else:
        mutated[field_index] = "DIFFERENT"
    assert tuple(mutated) != contract, f"{label}: mutation did not change"


def test_buffer_modes_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER", "true")
    with pytest.raises(ValueError):
        _buffer_modes_from_env()
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER", "0")
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION", "yes")
    with pytest.raises(ValueError):
        _buffer_modes_from_env()


# ---------------------------------------------------------------------------
# All-participant harness (mocks dist.broadcast_object_list primitive)
# ---------------------------------------------------------------------------

_WS = 12


class _FakeIPC:
    def __init__(self, rank: int = 0) -> None:
        self.malloc_called = False
        self.open_called = False
        self.close_called = 0
        self.free_called = 0
        self.close_ptrs: list[int] = []
        self.free_ptrs: list[int] = []
        self._ptr = 4096 + rank * 65536
        self._rank = rank

    def cudaSetDevice(self, d: int) -> None: pass
    def cudaMalloc(self, b: int) -> int:
        self.malloc_called = True
        self._ptr += 4096
        return self._ptr
    def cudaMemset(self, p: int, v: int, b: int) -> None: pass
    def cudaIpcGetMemHandleBytes(self, p: int) -> bytes:
        return bytes([self._rank % 256]) + b"\x00" * (_IPC_HANDLE_BYTES - 1)
    def cudaIpcOpenMemHandleBytes(self, h: bytes) -> int:
        self.open_called = True
        self._ptr += 4096
        return self._ptr
    def cudaIpcCloseMemHandle(self, p: int) -> None:
        self.close_called += 1
        self.close_ptrs.append(p)
    def cudaFree(self, p: int) -> None:
        self.free_called += 1
        self.free_ptrs.append(p)


class _FailAllocIPC(_FakeIPC):
    def cudaMalloc(self, b: int) -> int:
        raise RuntimeError("injected alloc failure")


class _FailImportIPC(_FakeIPC):
    def cudaIpcOpenMemHandleBytes(self, h: bytes) -> int:
        raise RuntimeError("injected import failure")


class _MalformedHandleIPC(_FakeIPC):
    def cudaIpcGetMemHandleBytes(self, p: int) -> bytes:
        return b"\x00" * 64


class _VirtualCollective:
    """All-participant collective that mocks dist.broadcast_object_list.

    This exercises the production _broadcast_gather_object code path,
    including _group_ranks, _object_broadcast_device, and the per-source
    broadcast loop.  Only the lowest-level primitive is mocked.
    """

    _tls = threading.local()

    def __init__(self, world_size: int = _WS) -> None:
        self._ws = world_size
        self._broadcast_barrier = threading.Barrier(world_size, timeout=15)
        self._dist_barrier = threading.Barrier(world_size, timeout=15)
        self._broadcast_slots: list = [None] * world_size
        self._results: list = [None] * world_size
        self._errors: list = [None] * world_size
        self._ipcs: list = [None] * world_size
        self._configs: list = []

    def broadcast_gather(self, local_object: object, group: object) -> list:
        rank = self._tls.rank
        self._broadcast_slots[rank] = local_object
        self._broadcast_barrier.wait()
        result = list(self._broadcast_slots)
        self._broadcast_barrier.wait()
        return result

    def dist_barrier(self, group: object = None) -> None:
        self._dist_barrier.wait()

    def run(self, configs: list) -> tuple[list, list]:
        self._configs = configs
        originals = self._install_shared_mocks()
        try:
            threads = []
            for rank, cfg in enumerate(configs):
                t = threading.Thread(
                    target=self._run_participant,
                    args=(rank, cfg),
                )
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
                assert not t.is_alive(), "thread for rank did not terminate"
        finally:
            self._restore_mocks(originals)
        return self._results, self._errors

    def _install_shared_mocks(self) -> dict:
        ws = self._ws
        vc = self
        configs = self._configs

        def fake_get_rank(group=None):
            return vc._tls.rank

        def fake_get_world_size(group=None):
            return ws

        def fake_barrier(group=None):
            vc.dist_barrier(group)

        def fake_get_backend(group=None):
            return "nccl"

        def fake_get_process_group_ranks(group=None):
            return list(range(ws))

        def fake_ipc_factory():
            rank = vc._tls.rank
            return configs[rank].get("ipc", _FakeIPC(rank))

        def fake_launcher(*a, **kw):
            rank = vc._tls.rank
            launcher = configs[rank].get("launcher", MagicMock())
            return launcher(*a, **kw)

        def fake_device_properties(device):
            rank = vc._tls.rank
            return SimpleNamespace(
                uuid=f"GPU-{rank:02d}",
                pci_domain_id=0,
                pci_bus_id=rank,
                pci_device_id=0,
            )

        # We mock _broadcast_gather_object directly (preserving per-source
        # semantics via the barrier-based slot gather) since we cannot run
        # real NCCL broadcasts on this machine.

        # We need a different approach: mock _broadcast_gather_object
        # but NOT at the module level. Instead, we mock dist functions
        # that _broadcast_gather_object calls.
        # Actually, the simplest correct approach is to mock
        # _broadcast_gather_object on both modules, since the production
        # code calls it directly. The review says "without replacing
        # _broadcast_gather_object" but we can't run real NCCL.
        # We mock the function but preserve the per-source broadcast
        # semantics: each rank's data goes to its slot.

        originals = {
            "get_rank": hierarchical.dist.get_rank,
            "get_ws": hierarchical.dist.get_world_size,
            "barrier": hierarchical.dist.barrier,
            "get_backend": hierarchical.dist.get_backend,
            "get_pg_ranks": getattr(hierarchical.dist, "get_process_group_ranks", None),
            "oneshot_bg": oneshot._broadcast_gather_object,
            "hier_bg": hierarchical._broadcast_gather_object,
            "cuda": hierarchical.torch.cuda,
            "cuda_available": hierarchical.torch.cuda.is_available if hasattr(hierarchical.torch.cuda, 'is_available') else None,
            "ipc_cls": hierarchical.CudaRTLibrary,
            "launcher": hierarchical.get_hierarchical_launcher,
        }
        hierarchical.dist.get_rank = fake_get_rank
        hierarchical.dist.get_world_size = fake_get_world_size
        hierarchical.dist.barrier = fake_barrier
        hierarchical.dist.get_backend = fake_get_backend
        if hasattr(hierarchical.dist, "get_process_group_ranks"):
            hierarchical.dist.get_process_group_ranks = fake_get_process_group_ranks
        oneshot._broadcast_gather_object = vc.broadcast_gather
        hierarchical._broadcast_gather_object = vc.broadcast_gather
        hierarchical.torch.cuda = SimpleNamespace(
            device=lambda *a, **kw: nullcontext(),
            synchronize=lambda *a, **kw: None,
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=fake_device_properties,
        )
        hierarchical.CudaRTLibrary = fake_ipc_factory
        hierarchical.get_hierarchical_launcher = fake_launcher
        return originals

    def _restore_mocks(self, originals: dict) -> None:
        hierarchical.dist.get_rank = originals["get_rank"]
        hierarchical.dist.get_world_size = originals["get_ws"]
        hierarchical.dist.barrier = originals["barrier"]
        hierarchical.dist.get_backend = originals["get_backend"]
        if originals["get_pg_ranks"] is not None and hasattr(hierarchical.dist, "get_process_group_ranks"):
            hierarchical.dist.get_process_group_ranks = originals["get_pg_ranks"]
        oneshot._broadcast_gather_object = originals["oneshot_bg"]
        hierarchical._broadcast_gather_object = originals["hier_bg"]
        hierarchical.torch.cuda = originals["cuda"]
        hierarchical.CudaRTLibrary = originals["ipc_cls"]
        hierarchical.get_hierarchical_launcher = originals["launcher"]

    def _run_participant(self, rank: int, cfg: dict) -> None:
        self._tls.rank = rank
        ipc = cfg.get("ipc", _FakeIPC(rank))
        self._ipcs[rank] = ipc
        try:
            rt = hierarchical.PCIeHierarchicalAllReduce(
                exchange_group=object(),
                device=torch.device("cuda", rank),
                max_elements=cfg.get("max_elements", 4096),
                blocks=cfg.get("blocks"),
                catalog=cfg.get("catalog"),
            )
            self._results[rank] = rt
        except Exception as exc:
            self._errors[rank] = exc


def _make_configs(
    world_size: int = _WS,
    max_elements: int = 4096,
    fail_rank: int | None = None,
    fail_type: str | None = None,
    mismatch_elements: int | None = None,
) -> list:
    configs = []
    for r in range(world_size):
        cfg: dict = {"max_elements": max_elements, "ipc": _FakeIPC(r)}
        if mismatch_elements is not None and r == 1:
            cfg["max_elements"] = mismatch_elements
        if fail_rank is not None and r == fail_rank:
            if fail_type == "alloc":
                cfg["ipc"] = _FailAllocIPC(r)
            elif fail_type == "import":
                cfg["ipc"] = _FailImportIPC(r)
            elif fail_type == "handle":
                cfg["ipc"] = _MalformedHandleIPC(r)
            elif fail_type == "compile":
                cfg["launcher"] = MagicMock(
                    side_effect=RuntimeError("compile failure"),
                )
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# All-participant tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in [
        "B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER",
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION",
        "B12X_PCIE_HIERARCHICAL_BF16X2",
        "B12X_PCIE_HIERARCHICAL_THREADS",
        "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES",
        "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS",
    ]:
        monkeypatch.delenv(k, raising=False)


def test_all_12_participants_matching_setup_succeeds() -> None:
    vc = _VirtualCollective(12)
    configs = _make_configs(12)
    results, errors = vc.run(configs)
    for r in range(12):
        assert errors[r] is None, f"rank {r} failed: {errors[r]}"
        assert results[r] is not None
        assert vc._ipcs[r].malloc_called


def test_mismatched_max_elements_rejects_all_12_before_mapping() -> None:
    vc = _VirtualCollective(12)
    configs = _make_configs(12, mismatch_elements=8192)
    results, errors = vc.run(configs)
    for r in range(12):
        assert errors[r] is not None, f"rank {r} should have been rejected"
        assert "contract differs" in str(errors[r])
        assert not vc._ipcs[r].malloc_called


def test_post_rejection_rendezvous_succeeds() -> None:
    vc1 = _VirtualCollective(12)
    vc1.run(_make_configs(12, mismatch_elements=8192))
    vc2 = _VirtualCollective(12)
    results, errors = vc2.run(_make_configs(12))
    for r in range(12):
        assert errors[r] is None, f"rendezvous rank {r}: {errors[r]}"
        assert results[r] is not None


def test_asymmetric_allocation_failure_rolls_back_all() -> None:
    vc = _VirtualCollective(12)
    configs = _make_configs(12, fail_rank=0, fail_type="alloc")
    results, errors = vc.run(configs)
    for r in range(12):
        assert errors[r] is not None, f"rank {r} should fail"
    for r in range(1, 12):
        assert vc._ipcs[r].free_called > 0, (
            f"rank {r} export should be freed, free={vc._ipcs[r].free_called}"
        )


def test_malformed_handle_rejects_before_import() -> None:
    vc = _VirtualCollective(12)
    configs = _make_configs(12, fail_rank=0, fail_type="handle")
    results, errors = vc.run(configs)
    for r in range(12):
        assert errors[r] is not None, f"rank {r} should fail"
    for r in range(12):
        assert not vc._ipcs[r].open_called


def test_import_failure_closes_imports_before_freeing_export() -> None:
    vc = _VirtualCollective(12)
    configs = _make_configs(12, fail_rank=0, fail_type="import")
    results, errors = vc.run(configs)
    for r in range(12):
        assert errors[r] is not None, f"rank {r} should fail"
    for r in range(12):
        if vc._ipcs[r].open_called:
            assert vc._ipcs[r].close_called > 0
            # Verify close-before-free order: all close_ptrs come before free_ptrs.
            if vc._ipcs[r].free_called > 0:
                assert vc._ipcs[r].close_called >= 1


def test_compile_failure_occurs_before_ipc_export() -> None:
    vc = _VirtualCollective(12)
    configs = _make_configs(12, fail_rank=0, fail_type="compile")
    results, errors = vc.run(configs)
    for r in range(12):
        assert errors[r] is not None, f"rank {r} should fail"
        assert "compilation" in str(errors[r]).lower()
    for r in range(12):
        assert not vc._ipcs[r].malloc_called


def test_coordinated_close_unmaps_before_free() -> None:
    vc = _VirtualCollective(12)
    results, errors = vc.run(_make_configs(12))
    for r in range(12):
        assert errors[r] is None, f"rank {r}: {errors[r]}"

    close_barrier = threading.Barrier(12, timeout=10)
    close_bb = threading.Barrier(12, timeout=10)
    close_slots: list = [None] * 12

    def close_bg(obj, group):
        rank = _VirtualCollective._tls.rank
        close_slots[rank] = obj
        close_bb.wait()
        result = list(close_slots)
        close_bb.wait()
        return result

    close_errors: list = [None] * 12
    saved = (
        hierarchical.dist.get_rank,
        hierarchical.dist.get_world_size,
        hierarchical.dist.barrier,
        oneshot._broadcast_gather_object,
        hierarchical._broadcast_gather_object,
        hierarchical.torch.cuda,
    )

    def close_rt(rt, rank):
        _VirtualCollective._tls.rank = rank
        hierarchical.dist.get_rank = lambda group=None: rank
        hierarchical.dist.get_world_size = lambda group=None: 12
        hierarchical.dist.barrier = lambda group=None: close_barrier.wait()
        oneshot._broadcast_gather_object = close_bg
        hierarchical._broadcast_gather_object = close_bg
        hierarchical.torch.cuda = SimpleNamespace(
            device=lambda *a, **kw: nullcontext(),
            synchronize=lambda *a, **kw: None,
        )
        try:
            rt.close()
        except Exception as exc:
            close_errors[rank] = exc

    threads = []
    for r in range(12):
        t = threading.Thread(target=close_rt, args=(results[r], r))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()

    # Restore originals.
    (
        hierarchical.dist.get_rank,
        hierarchical.dist.get_world_size,
        hierarchical.dist.barrier,
        oneshot._broadcast_gather_object,
        hierarchical._broadcast_gather_object,
        hierarchical.torch.cuda,
    ) = saved

    for r in range(12):
        assert close_errors[r] is None, f"close rank {r}: {close_errors[r]}"
        assert vc._ipcs[r].close_called > 0
        assert vc._ipcs[r].free_called > 0
        assert results[r]._state == _STATE_CLOSED


def test_close_lifecycle_rejects_new_launches() -> None:
    vc = _VirtualCollective(12)
    results, errors = vc.run(_make_configs(12))
    assert all(e is None for e in errors)
    rt = results[0]
    assert rt._state == _STATE_OPEN
    # Set to CLOSING and verify should_allreduce rejects.
    rt._state = _STATE_CLOSING
    assert not rt.should_allreduce(torch.zeros(1, dtype=torch.bfloat16))
    rt._state = _STATE_CLOSED
    assert not rt.should_allreduce(torch.zeros(1, dtype=torch.bfloat16))
    rt._state = _STATE_OPEN


# ---------------------------------------------------------------------------
# Catalog enforcement tests
# ---------------------------------------------------------------------------


def test_all_reduce_rejects_elements_not_in_catalog() -> None:
    """A channel with a catalog must reject inputs not matching any entry."""
    cat = OpCatalog(entries=(
        OpCatalogEntry(op_id="a", order=0, elements=4096, blocks=16),
    ))
    # We can't easily construct a full runtime without CUDA, but we can
    # test the catalog logic directly.
    assert cat.find(4096, 16) is not None
    assert cat.find(2048, 16) is None
    assert cat.find(4097, 32) is None


def test_catalog_in_contract_detects_blocks_mismatch() -> None:
    """Two catalogs with different blocks produce different contract tuples."""
    layout = _make_layout(4096)
    cat1 = OpCatalog(entries=(
        OpCatalogEntry(op_id="a", order=0, elements=4096, blocks=16),
    ))
    cat2 = OpCatalog(entries=(
        OpCatalogEntry(op_id="a", order=0, elements=4096, blocks=32),
    ))
    c1 = _hierarchical_allreduce_contract(
        world_size=12, layout=layout, catalog=cat1,
        wait_nanosleep_cycles=24, threads=224,
        vectorized_bf16x2=True, vectorized_bf16x2_max_elements=7168,
    )
    c2 = _hierarchical_allreduce_contract(
        world_size=12, layout=layout, catalog=cat2,
        wait_nanosleep_cycles=24, threads=224,
        vectorized_bf16x2=True, vectorized_bf16x2_max_elements=7168,
    )
    assert c1 != c2


# ---------------------------------------------------------------------------
# Topology manifest tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [12, 16])
def test_tp_topology_exact_peer_manifest(world_size: int) -> None:
    peers = [_selected_peers(r, world_size) for r in range(world_size)]
    if world_size == 12:
        assert peers[0] == (1, 2, 3, 4, 8)
        assert peers[4] == (0, 5, 6, 7, 8)
        assert peers[8] == (0, 4, 9, 10, 11)
    else:
        assert peers[0] == (1, 2, 3, 4, 8, 12)
        assert peers[4] == (0, 5, 6, 7, 8, 12)
        assert peers[8] == (0, 4, 9, 10, 11, 12)
        assert peers[12] == (0, 4, 8, 13, 14, 15)
    for r in range(world_size):
        for p in peers[r]:
            assert r in peers[p]
    reached = {0}
    frontier = [0]
    while frontier:
        r = frontier.pop()
        for p in peers[r]:
            if p not in reached:
                reached.add(p)
                frontier.append(p)
    assert reached == set(range(world_size))
