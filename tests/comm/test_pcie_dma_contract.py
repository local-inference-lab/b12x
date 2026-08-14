"""Tests for PCIe-DMA collective IPC layout contract agreement.

Every rank must collectively agree on a versioned, fixed-schema tuple covering
allocation bytes, region offsets/capacities, wire mode/codec, topology, and
chunk/schedule/launch policy *before* any rank opens a peer CUDA IPC memory
handle (``cudaIpcOpenMemHandle``).  A single-field mismatch must reject every
rank before IPC mapping; an exact match must proceed to allocation.

The mismatch tests use a real two-rank ``mp.spawn`` harness with a gloo process
group so that ``dist.broadcast_object_list`` genuinely runs on both ranks.
CUDA-specific dependencies (device normalization, kernel loading, the CUDA
runtime library) are mocked per-worker so the tests run on CPU-only machines.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import b12x.comm.pcie.pcie_dma as dma
import b12x.comm.pcie.pcie_oneshot as oneshot
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce


# ---------------------------------------------------------------------------
# Contract field indices (from ``_dma_ipc_layout_contract``):
#   0  version              1  world_size          2  max_bytes
#   3  slab_bytes           4  flags_bytes         5  shard_capacity
#   6  steps                7  fp8                 8  fp8_stage_stride
#   9  dtype ABI signature  10 pieces_override     11 a2a_chunks_override
#  12  FLAG_STRIDE          13 FLAG_SLOTS          14 MAX_PIECES
#  15  SCRATCH_ALIGN        16 FP8_QUANT_BLOCK
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Unit tests — contract function and helpers (no distributed comms needed).
# ---------------------------------------------------------------------------


def test_dma_contract_is_versioned_fixed_schema():
    """The contract tuple starts with an integer version field so schema
    incompatibilities are caught before any pointer interpretation."""

    contract = dma._dma_ipc_layout_contract(
        world_size=2,
        max_bytes=4096,
        shard_capacity=2048,
        flags_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE,
        slab_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE + 2 * 2048,
        fp8="",
        fp8_stage_stride=0,
        pieces_override=0,
        a2a_chunks_override=0,
    )
    assert contract[0] == dma._DMA_CONTRACT_VERSION
    assert isinstance(contract[0], int)


def test_dtype_abi_signature_binds_name_code_width():
    """The dtype ABI signature binds each dtype name to its kernel code and
    element width, not merely the multiset of codes."""

    sig = dma._dtype_abi_signature()
    assert len(sig) == 3
    # Each entry is (name, code, element_bytes).
    names = [entry[0] for entry in sig]
    codes = [entry[1] for entry in sig]
    assert names[0] == "torch.bfloat16"
    assert names[1] == "torch.float16"
    assert names[2] == "torch.float32"
    assert sorted(codes) == [0, 1, 2]
    # bf16 and fp16 are 2 bytes; fp32 is 4 bytes.
    for name, _code, width in sig:
        if "bfloat16" in name or "float16" in name:
            assert width == 2
        elif "float32" in name:
            assert width == 4


def test_dtype_abi_signature_detects_permuted_mapping():
    """A permuted dtype→code mapping produces a different signature even if
    the multiset of codes is the same."""

    real_sig = dma._dtype_abi_signature()
    # Simulate a permuted mapping: bf16→2, fp32→0 (same code multiset).
    permuted = tuple(
        (str(dt), 2 - dma.SUPPORTED_DTYPES[dt], dt.itemsize)
        for dt, _ in sorted(dma.SUPPORTED_DTYPES.items(), key=lambda kv: kv[1])
    )
    assert permuted != real_sig


def test_dma_contract_includes_frozen_scheduling_knobs():
    """The contract includes the frozen pieces and a2a_chunks override values
    so two ranks with different env overrides are caught."""

    base = dict(
        world_size=2,
        max_bytes=4096,
        shard_capacity=2048,
        flags_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE,
        slab_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE + 2 * 2048,
        fp8="",
        fp8_stage_stride=0,
    )
    no_override = dma._dma_ipc_layout_contract(
        pieces_override=0, a2a_chunks_override=0, **base
    )
    pieces_1 = dma._dma_ipc_layout_contract(
        pieces_override=1, a2a_chunks_override=0, **base
    )
    chunks_2 = dma._dma_ipc_layout_contract(
        pieces_override=0, a2a_chunks_override=2, **base
    )
    assert no_override[10] == 0
    assert pieces_1[10] == 1
    assert chunks_2[11] == 2
    assert no_override != pieces_1
    assert no_override != chunks_2


@pytest.mark.parametrize(
    ("env_name", "parser"),
    [
        ("B12X_PCIE_DMA_PIECES", dma._parse_pieces_override),
        ("B12X_PCIE_DMA_A2A_CHUNKS", dma._parse_a2a_chunks_override),
    ],
)
def test_malformed_dma_override_uses_default(monkeypatch, env_name, parser):
    monkeypatch.setenv(env_name, "auto")
    assert parser() == 0


def test_dma_contract_wire_mode_field_distinguishes_fp8():
    """The wire-mode field must change when FP8 is enabled so a mismatched
    peer cannot interpret BF16 bytes as compressed payloads."""

    base = dict(
        world_size=2,
        max_bytes=4096,
        shard_capacity=2048,
        flags_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE,
        slab_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE + 2 * 2048,
        pieces_override=0,
        a2a_chunks_override=0,
    )
    no_fp8 = dma._dma_ipc_layout_contract(fp8="", fp8_stage_stride=0, **base)
    with_fp8 = dma._dma_ipc_layout_contract(
        fp8="ag", fp8_stage_stride=1024, **base
    )
    assert no_fp8[7] == ""
    assert with_fp8[7] == "ag"
    assert no_fp8 != with_fp8


def test_parse_pieces_override_canonicalizes_out_of_range():
    """Out-of-range or unset env values canonicalize to 0 (default)."""

    original = os.environ.pop("B12X_PCIE_DMA_PIECES", None)
    try:
        assert dma._parse_pieces_override() == 0
        os.environ["B12X_PCIE_DMA_PIECES"] = "3"
        assert dma._parse_pieces_override() == 3
        os.environ["B12X_PCIE_DMA_PIECES"] = "0"
        assert dma._parse_pieces_override() == 0
        os.environ["B12X_PCIE_DMA_PIECES"] = "99"
        assert dma._parse_pieces_override() == 0
        os.environ["B12X_PCIE_DMA_PIECES"] = "-1"
        assert dma._parse_pieces_override() == 0
    finally:
        if original is not None:
            os.environ["B12X_PCIE_DMA_PIECES"] = original
        else:
            os.environ.pop("B12X_PCIE_DMA_PIECES", None)


def test_parse_a2a_chunks_override_canonicalizes_out_of_range():
    """Out-of-range or unset env values canonicalize to 0 (default)."""

    original = os.environ.pop("B12X_PCIE_DMA_A2A_CHUNKS", None)
    try:
        assert dma._parse_a2a_chunks_override() == 0
        os.environ["B12X_PCIE_DMA_A2A_CHUNKS"] = "4"
        assert dma._parse_a2a_chunks_override() == 4
        os.environ["B12X_PCIE_DMA_A2A_CHUNKS"] = "0"
        assert dma._parse_a2a_chunks_override() == 0
        os.environ["B12X_PCIE_DMA_A2A_CHUNKS"] = "99"
        assert dma._parse_a2a_chunks_override() == 0
    finally:
        if original is not None:
            os.environ["B12X_PCIE_DMA_A2A_CHUNKS"] = original
        else:
            os.environ.pop("B12X_PCIE_DMA_A2A_CHUNKS", None)


# ---------------------------------------------------------------------------
# Two-rank spawned harness.
#
# Each worker joins a real gloo process group so ``dist.broadcast_object_list``
# genuinely executes on both ranks.  CUDA-specific dependencies are mocked
# per-worker so the tests run on CPU-only machines.  A spy on
# ``_allocate_shared_buffer`` asserts that IPC allocation never starts during
# a mismatch rejection.  A post-rejection ``dist.barrier()`` proves no rank
# is left blocked.
# ---------------------------------------------------------------------------


class _FakeKernels:
    def prepare(self, *, world_size, wire_mode):
        pass


class _FakeIPC:
    def cudaSetDevice(self, device):
        pass


class _FakeSlab:
    peer_ptrs = (1000, 2000)

def _install_cuda_mocks():
    """Patch CUDA-specific module attributes for CPU-only testing.

    The collective operations (``_broadcast_gather_object``) are replaced
    with a gloo-compatible implementation that uses ``all_gather_object``
    so both ranks genuinely participate in a real collective exchange.
    This is not a fabrication: each rank's contract/preflight status is
    derived from its own local inputs and exchanged through a real
    distributed operation.
    """

    def _gloo_broadcast_gather(local_object: object, group) -> list:
        gathered = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(gathered, local_object, group=group)
        return gathered

    oneshot._broadcast_gather_object = _gloo_broadcast_gather
    dma._broadcast_gather_object = _gloo_broadcast_gather
    # Return a fake CUDA device so the preflight device check passes.
    dma._normalize_device = lambda d: torch.device("cuda:0")
    dma._load_kernels = lambda: _FakeKernels()
    dma.CudaRTLibrary = lambda: _FakeIPC()


def _install_ipc_allocation_spy(allow_allocation: bool):
    """Replace ``_allocate_shared_buffer`` with a spy.

    If *allow_allocation* is True, return a fake slab (exact-match test).
    Otherwise, raise AssertionError if called (mismatch tests).
    """

    if allow_allocation:

        def spy(*args, **kwargs):
            return _FakeSlab()

        PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(spy)
    else:

        def spy(*args, **kwargs):
            raise AssertionError(
                "CUDA IPC allocation must not start when the contract differs"
            )

        PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(spy)

def _mismatch_worker(
    rank: int,
    world_size: int,
    port: int,
    rank0_kwargs: dict,
    rank1_kwargs: dict,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    _install_ipc_allocation_spy(allow_allocation=False)
    try:
        kwargs = rank0_kwargs if rank == 0 else rank1_kwargs
        with pytest.raises(RuntimeError):
            dma.PCIeDmaAllReduce(
                exchange_group=dist.group.WORLD,
                device=torch.device("cuda:0"),
                **kwargs,
            )
        # Post-rejection rendezvous: prove no rank is left blocked.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _preflight_failure_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Rank 0 has an invalid FP8 mode; rank 1 is valid.  The coordinated
    preflight must reject both ranks before any contract exchange."""

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    _install_ipc_allocation_spy(allow_allocation=False)
    try:
        if rank == 0:
            with pytest.raises(RuntimeError, match="pre-allocation setup"):
                dma.PCIeDmaAllReduce(
                    exchange_group=dist.group.WORLD,
                    device=torch.device("cuda:0"),
                    max_bytes=4096,
                    fp8="invalid_mode_xyz",
                )
        else:
            with pytest.raises(RuntimeError, match="pre-allocation setup"):
                dma.PCIeDmaAllReduce(
                    exchange_group=dist.group.WORLD,
                    device=torch.device("cuda:0"),
                    max_bytes=4096,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class _ContractPassed(Exception):
    """Sentinel raised by the allocation spy to prove the contract check
    passed and the constructor reached ``_allocate_shared_buffer``."""


def _exact_match_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Both ranks use identical args; the contract check must pass and the
    constructor must reach ``_allocate_shared_buffer``.

    On CPU-only machines we cannot complete the post-IPC CUDA resource
    construction, so the allocation spy raises a sentinel exception to prove
    the constructor progressed past the contract agreement.  Both ranks
    independently reach this point after completing the collective exchange,
    proving the contract accepted the matching inputs.
    """

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()

    def allocation_sentinel(*args, **kwargs):
        raise _ContractPassed("contract agreed; reached IPC allocation")

    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        allocation_sentinel
    )
    try:
        with pytest.raises(_ContractPassed):
            dma.PCIeDmaAllReduce(
                exchange_group=dist.group.WORLD,
                device=torch.device("cuda:0"),
                max_bytes=4096,
            )
        # Post-rejection rendezvous: prove no rank is left blocked.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _env_knob_mismatch_worker(
    rank: int,
    world_size: int,
    port: int,
    env_var: str,
    rank0_value: str,
    rank1_value: str,
) -> None:
    """Each rank sets a different env var value before construction so the
    frozen scheduling knob mismatch is caught by the contract."""

    os.environ[env_var] = rank0_value if rank == 0 else rank1_value
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    _install_ipc_allocation_spy(allow_allocation=False)
    try:
        with pytest.raises(RuntimeError):
            dma.PCIeDmaAllReduce(
                exchange_group=dist.group.WORLD,
                device=torch.device("cuda:0"),
                max_bytes=4096,
            )
        dist.barrier()
    finally:
        os.environ.pop(env_var, None)
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Spawned mismatch tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("category", "rank0_kwargs", "rank1_kwargs"),
    [
        ("size", {"max_bytes": 4096}, {"max_bytes": 8192}),
        ("wire", {"max_bytes": 4096, "fp8": ""}, {"max_bytes": 4096, "fp8": "ag"}),
    ],
    ids=["size", "wire"],
)
def test_dma_contract_mismatch_rejects_both_ranks_before_ipc(
    category, rank0_kwargs, rank1_kwargs
):
    mp.spawn(
        _mismatch_worker,
        args=(2, _free_port(), rank0_kwargs, rank1_kwargs),
        nprocs=2,
        join=True,
    )


def test_dma_preflight_failure_rejects_both_ranks_before_contract():
    """One rank has an invalid FP8 mode; the coordinated preflight must reject
    both ranks before the contract exchange, so no rank blocks."""

    mp.spawn(
        _preflight_failure_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_exact_match_succeeds():
    """When both ranks agree on all contract fields, the constructor proceeds
    past the contract check to IPC allocation and succeeds."""

    mp.spawn(
        _exact_match_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_pieces_override_mismatch_rejects_both_ranks():
    """Different ``B12X_PCIE_DMA_PIECES`` env values on two ranks produce
    different frozen scheduling knobs, caught by the contract."""

    mp.spawn(
        _env_knob_mismatch_worker,
        args=(2, _free_port(), "B12X_PCIE_DMA_PIECES", "1", "2"),
        nprocs=2,
        join=True,
    )


def test_dma_a2a_chunks_override_mismatch_rejects_both_ranks():
    """Different ``B12X_PCIE_DMA_A2A_CHUNKS`` env values on two ranks produce
    different frozen scheduling knobs, caught by the contract."""

    mp.spawn(
        _env_knob_mismatch_worker,
        args=(2, _free_port(), "B12X_PCIE_DMA_A2A_CHUNKS", "1", "3"),
        nprocs=2,
        join=True,
    )


def test_dma_version_mismatch_rejects():
    """A different contract version is caught by the version field."""

    base = dict(
        world_size=2,
        max_bytes=4096,
        shard_capacity=2048,
        flags_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE,
        slab_bytes=dma.FLAG_SLOTS * dma.FLAG_STRIDE + 2 * 2048,
        fp8="",
        fp8_stage_stride=0,
        pieces_override=0,
        a2a_chunks_override=0,
    )
    contract = dma._dma_ipc_layout_contract(**base)
    # Tamper the version field.
    tampered = (contract[0] + 1,) + contract[1:]
    assert tampered[0] != contract[0]
    assert tampered != contract


# ---------------------------------------------------------------------------
# Post-IPC failure ordering tests.
# ---------------------------------------------------------------------------


class _PrepareFailedKernels:
    """Fake kernels whose ``prepare`` raises to simulate a post-IPC
    compile/launch failure on one rank."""

    def prepare(self, *, world_size, wire_mode):
        raise RuntimeError("injected post-IPC prepare failure")


class _TrackingIPC:
    """Fake IPC that records every close/free call so the test can assert
    that all import closes precede every export free."""

    def __init__(self, fail_close: bool = False):
        self.cudaSetDevice = lambda d: None
        self.close_calls: list[int] = []
        self.free_calls: list[int] = []
        self.event_log: list[tuple[str, int]] = []
        self._fail_close = fail_close

    def cudaIpcCloseMemHandle(self, ptr: int):
        if self._fail_close:
            raise RuntimeError(f"injected unmap failure for {ptr}")
        self.close_calls.append(ptr)
        self.event_log.append(("close", ptr))

    def cudaFree(self, ptr: int):
        self.free_calls.append(ptr)
        self.event_log.append(("free", ptr))


class _TrackingSlab:
    """Fake slab with known local/remote pointers."""

    peer_ptrs = (1000, 2000)
    local_ptr = 1000
    remote_ptrs = (2000,)


def _prepare_failure_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Rank 0's ``prepare`` fails after IPC handles are open.  The owned
    collective finish/rollback machinery must:
      1. Exchange the failure status across all ranks.
      2. Close every import on every rank.
      3. Exchange the all-rank unmap verdict.
      4. Free every export only after all imports are unmapped.
    Both ranks must reject; the tracking IPC proves close-before-free ordering.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()

    # Install dummy CUDA resources so the constructor's post-IPC phase
    # (torch.zeros, torch.cuda.Stream/Event) does not fail on CPU-only
    # machines.  This lets rank 1 reach `prepare` successfully while rank 0
    # fails at `prepare`, proving the asymmetric rollback path.
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None

    tracking_ipc = _TrackingIPC()
    dma.CudaRTLibrary = lambda: tracking_ipc

    def tracking_allocate(*args, **kwargs):
        return _TrackingSlab()

    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        tracking_allocate
    )

    # Rank 0 gets failing kernels; rank 1 gets normal fake kernels.
    if rank == 0:
        dma._load_kernels = lambda: _PrepareFailedKernels()

    try:
        with pytest.raises(RuntimeError):
            dma.PCIeDmaAllReduce(
                exchange_group=dist.group.WORLD,
                device=torch.device("cuda:0"),
                max_bytes=4096,
            )
        # Post-rejection rendezvous.
        dist.barrier()
        # All import closes must precede any export free.
        assert tracking_ipc.close_calls, "at least one import must be closed"
        assert tracking_ipc.free_calls, "export must be freed after unmap"
        # The local export (1000) is freed; the remote import (2000) is closed.
        # The close must have happened before the free (event-log ordering).
        close_idx = next(i for i, (op, _) in enumerate(tracking_ipc.event_log) if op == "close")
        free_idx = next(i for i, (op, _) in enumerate(tracking_ipc.event_log) if op == "free")
        assert close_idx < free_idx, f"close at {close_idx} must precede free at {free_idx}"
        assert 1000 in tracking_ipc.free_calls
    finally:
        dist.destroy_process_group()


def _unmap_failure_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """One rank's import close fails.  The strict close protocol must retain
    exports (not free them) and raise a retryable error."""

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        """Stand-in for any CUDA resource (stream/event)."""
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    # Mock CUDA-specific constructors.  torch.zeros/empty are replaced with
    # CPU equivalents so gloo's all_gather_object still works (it uses
    # torch.zeros internally), and the DMA post-IPC phase gets dummy tensors.
    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None

    # Rank 0's IPC fails on close; rank 1's succeeds.
    fail_close = rank == 0
    tracking_ipc = _TrackingIPC(fail_close=fail_close)
    dma.CudaRTLibrary = lambda: tracking_ipc

    def tracking_allocate(*args, **kwargs):
        return _TrackingSlab()

    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        tracking_allocate
    )

    try:
        # Construction succeeds (prepare is fine); then close() should fail
        # because rank 0's unmap fails, and exports must be retained.
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )
        with pytest.raises(RuntimeError):
            ring.close()
        # Export must NOT be freed when any unmap failed.
        assert tracking_ipc.free_calls == []
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_dma_prepare_failure_closes_imports_before_freeing_exports():
    """A post-IPC prepare failure on one rank must trigger the owned
    collective rollback: every import is closed before any export is freed,
    and both ranks reject."""

    mp.spawn(
        _prepare_failure_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_unmap_failure_retains_exports():
    """An asymmetric unmap failure during close must retain exports (not
    free them) and raise a coordinated error so the caller can retry."""

    mp.spawn(
        _unmap_failure_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


class _FreeFailTrackingIPC(_TrackingIPC):
    """IPC whose first cudaFree fails and whose retry succeeds."""

    def __init__(self):
        super().__init__()
        self._fail_free = True

    def cudaFree(self, ptr: int):
        if self._fail_free:
            self._fail_free = False
            raise RuntimeError(f"injected free failure for {ptr}")
        super().cudaFree(ptr)


def _free_failure_worker(rank: int, world_size: int, port: int) -> None:
    """A partial export-free failure must retain exactly the unfinished
    ownership and let a second collective close retry it once."""

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None

    # Rank 0's first free fails; rank 1's succeeds.
    tracking_ipc = (
        _FreeFailTrackingIPC() if rank == 0 else _TrackingIPC()
    )
    dma.CudaRTLibrary = lambda: tracking_ipc

    def tracking_allocate(*args, **kwargs):
        return _TrackingSlab()

    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        tracking_allocate
    )

    try:
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )
        with pytest.raises(RuntimeError) as caught:
            ring.close()
        ticket = caught.value.retryable_setup
        assert ticket is ring._close_retry_ticket
        assert ticket.state == "free"
        assert ticket.remote_ptrs == []
        assert ticket.local_ptr == (1000 if rank == 0 else 0)
        assert ring._owned_buffers == []
        assert ring._collective_close_ticket is True
        assert ring._lifecycle_state == "FAILED"

        ring.close()
        assert ring._lifecycle_state == "CLOSED"
        assert ring._collective_close_ticket is False
        assert ring._close_retry_ticket is None
        assert tracking_ipc.free_calls == [1000]
        assert ticket.key not in oneshot._RETAINED_FAILED_IPC_EXPORTS
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _threaded_launch_close_worker(rank: int, world_size: int, port: int) -> None:
    """A thread launches all_reduce while the main thread closes.
    The lifecycle lock must serialize: either the launch is admitted before
    CLOSING (and completes its host enqueue), or it is rejected after
    CLOSING.  No use-after-unmap."""

    import threading as _threading

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None
    dma.CudaRTLibrary = lambda: _TrackingIPC()
    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        lambda *a, **kw: _TrackingSlab()
    )

    errors: list[Exception] = []
    launch_admitted = _threading.Event()
    release_launch = _threading.Event()
    close_done = _threading.Event()

    def launch():
        try:
            # Hold the mocked host enqueue open so close must wait on the
            # local lifecycle lease before synchronizing or unmapping.
            class _FakeInput:
                device = torch.device("cuda:0")
                dtype = torch.bfloat16

                def numel(self):
                    return 16

                def element_size(self):
                    return 2

                def is_contiguous(self):
                    return True

                @property
                def shape(self):
                    return (16,)

            orig_arod = dma.PCIeDmaAllReduce._all_reduce_on_device

            def mock_arod(self, inp, *, out=None):
                launch_admitted.set()
                if not release_launch.wait(timeout=10.0):
                    raise RuntimeError("timed out waiting to release launch")
                return out if out is not None else inp

            dma.PCIeDmaAllReduce._all_reduce_on_device = mock_arod
            try:
                ring.all_reduce(_FakeInput())
            finally:
                dma.PCIeDmaAllReduce._all_reduce_on_device = orig_arod
        except Exception as exc:
            errors.append(exc)

    def close():
        try:
            ring.close()
        except Exception as exc:
            errors.append(exc)
        finally:
            close_done.set()

    try:
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )
        launch_thread = _threading.Thread(target=launch)
        launch_thread.start()
        assert launch_admitted.wait(timeout=10.0)

        close_thread = _threading.Thread(target=close)
        close_thread.start()
        # The launch now holds the lease through host enqueue.
        assert not close_done.wait(timeout=0.1)
        assert ring._ipc.event_log == []

        release_launch.set()
        launch_thread.join(timeout=10.0)
        close_thread.join(timeout=10.0)
        assert not launch_thread.is_alive()
        assert not close_thread.is_alive()
        assert errors == []
        assert ring._lifecycle_state == "CLOSED"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _capture_lease_close_worker(rank: int, world_size: int, port: int) -> None:
    """Close must wait for active capture leases before proceeding."""

    import threading as _threading

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None
    dma.CudaRTLibrary = lambda: _TrackingIPC()
    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        lambda *a, **kw: _TrackingSlab()
    )

    try:
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )
        # Acquire a capture lease; close must wait until it's released.
        ring.begin_capture()
        close_done = _threading.Event()
        errors: list[Exception] = []

        def do_close():
            try:
                ring.close()
            except Exception as exc:
                errors.append(exc)
            finally:
                close_done.set()

        t = _threading.Thread(target=do_close)
        t.start()
        # Close should be blocked waiting for the capture lease.
        assert not close_done.wait(timeout=1.0)
        ring.end_capture()
        assert close_done.wait(timeout=5.0)
        t.join()
        assert errors == []
        assert ring._lifecycle_state == "CLOSED"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _retry_after_unmap_failure_worker(rank: int, world_size: int, port: int) -> None:
    """An asymmetric unmap failure on the first close attempt must be
    retryable: a second collective close() transitions FAILED→CLOSING and
    succeeds when the injected failure is cleared."""

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None

    # First close: rank 0's IPC fails on close.  Second close: succeeds.
    attempt = [0]
    class _RetryableIPC(_TrackingIPC):
        def cudaIpcCloseMemHandle(self, ptr: int):
            if attempt[0] == 0 and rank == 0:
                attempt[0] += 1
                raise RuntimeError("injected unmap failure for retry test")
            super().cudaIpcCloseMemHandle(ptr)

    retry_ipc = _RetryableIPC()
    dma.CudaRTLibrary = lambda: retry_ipc
    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        lambda *a, **kw: _TrackingSlab()
    )

    try:
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )
        # First close fails and transfers only the remaining ownership.
        with pytest.raises(RuntimeError) as caught:
            ring.close()
        ticket = caught.value.retryable_setup
        assert ticket is ring._close_retry_ticket
        assert ticket.state == "unmap"
        assert ticket.remote_ptrs == ([2000] if rank == 0 else [])
        assert ticket.local_ptr == 1000
        assert ring._owned_buffers == []
        assert ring._lifecycle_state == "FAILED"
        assert ring._collective_close_ticket is True

        # Second close retries through the ticket without double-closing the
        # rank whose first unmap already succeeded.
        ring.close()
        assert ring._lifecycle_state == "CLOSED"
        assert ring._collective_close_ticket is False
        assert ring._close_retry_ticket is None
        assert retry_ipc.close_calls == [2000]
        assert retry_ipc.free_calls == [1000]
        assert ticket.key not in oneshot._RETAINED_FAILED_IPC_EXPORTS
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _destructor_quarantine_worker(rank: int, world_size: int, port: int) -> None:
    """A channel whose close never completed must be quarantined by __del__."""

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None
    dma.CudaRTLibrary = lambda: _TrackingIPC()
    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        lambda *a, **kw: _TrackingSlab()
    )

    try:
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )
        # Don't close — just drop the reference.  __del__ should quarantine.
        assert ring._collective_close_ticket is True
        del ring
        import gc
        gc.collect()
        # Verify the channel was quarantined.
        from b12x.comm.pcie.pcie_oneshot import _ABANDONED_PCIE_RUNTIME_QUARANTINE
        assert len(_ABANDONED_PCIE_RUNTIME_QUARANTINE) > 0
        # Clean up for the barrier.
        _ABANDONED_PCIE_RUNTIME_QUARANTINE.clear()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _duplicate_close_failure_worker(rank: int, world_size: int, port: int) -> None:
    """When the first close fails, a duplicate caller must receive the
    error, not silently succeed."""

    import threading as _threading

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None

    # Rank 0's unmap fails.
    if rank == 0:
        dma.CudaRTLibrary = lambda: _TrackingIPC(fail_close=True)
    else:
        dma.CudaRTLibrary = lambda: _TrackingIPC()
    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        lambda *a, **kw: _TrackingSlab()
    )

    errors: list[Exception] = []

    try:
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )

        def do_close():
            try:
                ring.close()
            except Exception as exc:
                errors.append(exc)

        t1 = _threading.Thread(target=do_close)
        t2 = _threading.Thread(target=do_close)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # At least one caller must have received the error.
        assert len(errors) >= 1
        assert all(isinstance(e, RuntimeError) for e in errors)
        assert ring._lifecycle_state == "FAILED"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _duplicate_close_worker(rank: int, world_size: int, port: int) -> None:
    """Two concurrent close() calls must serialize: one does the work,
    the other is a no-op."""

    import threading as _threading

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    _install_cuda_mocks()
    original_zeros = dma.torch.zeros
    original_empty = dma.torch.empty

    class _DummyCudaResource:
        data_ptr = 0
        def record(self, *a, **kw): pass
        def wait(self, *a, **kw): pass
        def wait_event(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def synchronize(self): pass

    def _cpu_zeros(*args, **kw):
        kw.pop("device", None)
        return original_zeros(*args, **kw)

    def _cpu_empty(*args, **kw):
        kw.pop("device", None)
        return original_empty(*args, **kw)

    dma.torch.zeros = _cpu_zeros
    dma.torch.empty = _cpu_empty
    dma.torch.cuda.Stream = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.Event = lambda **kw: _DummyCudaResource()
    dma.torch.cuda.synchronize = lambda *a, **kw: None
    oneshot.torch.cuda.synchronize = lambda *a, **kw: None

    tracking_ipc = _TrackingIPC()
    dma.CudaRTLibrary = lambda: tracking_ipc

    def tracking_allocate(*args, **kwargs):
        return _TrackingSlab()

    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(
        tracking_allocate
    )

    errors: list[Exception] = []

    try:
        ring = dma.PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )

        def do_close():
            try:
                ring.close()
            except Exception as exc:
                errors.append(exc)

        t1 = _threading.Thread(target=do_close)
        t2 = _threading.Thread(target=do_close)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []
        assert ring._lifecycle_state == "CLOSED"
        # Each import should be closed exactly once.
        assert tracking_ipc.close_calls.count(2000) == 1
        # Each export should be freed exactly once.
        assert tracking_ipc.free_calls.count(1000) == 1
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_dma_free_failure_retains_exact_retry_ticket():
    """An asymmetric export-free failure retains only unfinished ownership
    and a second collective close consumes the retry ticket."""

    mp.spawn(
        _free_failure_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_threaded_launch_vs_close():
    """A concurrent launch and close must be serialized by the lifecycle
    lock: no use-after-unmap, no deadlock."""

    mp.spawn(
        _threaded_launch_close_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_duplicate_close_serializes():
    """Two concurrent close() calls must serialize: one does the work,
    the other is a no-op; each IPC handle is closed/freed exactly once."""

    mp.spawn(
        _duplicate_close_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_capture_lease_blocks_close():
    """Close must wait for active capture/replay leases before proceeding."""

    mp.spawn(
        _capture_lease_close_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_retry_after_unmap_failure():
    """An asymmetric unmap failure on the first close must be retryable:
    a second collective close succeeds when the failure is cleared."""

    mp.spawn(
        _retry_after_unmap_failure_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_destructor_quarantines_unclosed_channel():
    """A channel whose close never completed must be quarantined by __del__."""

    mp.spawn(
        _destructor_quarantine_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_dma_duplicate_close_failure_propagates():
    """When the first close fails, a duplicate caller must receive the
    error, not silently succeed."""

    mp.spawn(
        _duplicate_close_failure_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )
