"""Contract tests for the typed tensor IPC wire format (issue #169).

These tests verify that:
  * The PCIe setup path contains no pickle-based object collective.
  * Valid multi-rank metadata round-trips through the typed tensor exchanges.
  * Malformed version, length, rank, world, capacity, and handle bytes are
    rejected **before** ``cudaIpcOpenMemHandle`` is called.
  * The NCCL/CUDA control-device contract is preserved.
  * Coordinated preflight prevents asymmetric hangs on local validation
    failures.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from b12x.comm.pcie import pcie_oneshot
from b12x.comm.pcie.pcie_oneshot import (
    _CUDA_IPC_HANDLE_BYTES,
    _FRAME_SIZE_IPC_HANDLE,
    _IPC_WIRE_MAGIC,
    _IPC_WIRE_VERSION,
    _MAX_CONTRACT_INTS,
    _MAX_GRAPH_BUFFERS,
    _MSG_KIND_IPC_HANDLE,
    _decode_strings_from_int64,
    _encode_strings_to_int64,
    _exchange_capture_contract,
    _exchange_channel_state,
    _exchange_graph_meta,
    _exchange_int_contract,
    _exchange_ipc_handles,
    _exchange_status_strings,
    _pack_bytes_to_int64s,
    _unpack_int64s_to_bytes,
    _validate_envelope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_group():
    return MagicMock()


def _patch_common(monkeypatch, world_size=2, rank=0):
    """Patch dist and CUDA so exchange functions can run on CPU."""
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: world_size)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: rank)
    monkeypatch.setattr("torch.distributed.get_backend", lambda group=None: "nccl")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._resolve_exchange_device",
        lambda group: torch.device("cpu"),
    )


def _patch_all_gather_echo(monkeypatch, world_size=2, rank=0):
    """Patch all_gather to echo local data into every slot with correct rank."""
    _patch_common(monkeypatch, world_size, rank)

    def fake_all_gather(gathered_list, local_tensor, group=None):
        vals = local_tensor.tolist()
        for i, slot in enumerate(gathered_list):
            peer_vals = list(vals)
            if len(peer_vals) > 3:
                peer_vals[3] = i  # set rank field in envelope
            slot.copy_(torch.tensor(peer_vals, dtype=torch.int64))

    monkeypatch.setattr("torch.distributed.all_gather", fake_all_gather)


def _patch_all_gather_per_rank(monkeypatch, world_size, rank, per_rank_values):
    """Patch all_gather to return caller-specified per-rank int64 lists."""
    _patch_common(monkeypatch, world_size, rank)
    call_count = [0]

    def fake_all_gather(gathered_list, local_tensor, group=None):
        # First call is preflight (frame size = _FRAME_SIZE_PREFLIGHT)
        # Second call is the payload
        if call_count[0] == 0:
            # Preflight: preserve the production planned-kind/frame fields.
            preflight = local_tensor.tolist()
            for i, slot in enumerate(gathered_list):
                peer = list(preflight)
                peer[3] = i
                slot.copy_(torch.tensor(peer, dtype=torch.int64))
        else:
            for i, slot in enumerate(gathered_list):
                slot.copy_(torch.tensor(per_rank_values[i], dtype=torch.int64))
        call_count[0] += 1

    monkeypatch.setattr("torch.distributed.all_gather", fake_all_gather)


def _patch_all_gather_with_preflight_failure(monkeypatch, world_size, rank, failed_ranks):
    """Patch all_gather so preflight reports failure for specified ranks."""
    _patch_common(monkeypatch, world_size, rank)

    def fake_all_gather(gathered_list, local_tensor, group=None):
        preflight = local_tensor.tolist()
        for i, slot in enumerate(gathered_list):
            peer = list(preflight)
            peer[3] = i
            peer[-1] = 0 if i in failed_ranks else 1
            slot.copy_(torch.tensor(peer, dtype=torch.int64))

    monkeypatch.setattr("torch.distributed.all_gather", fake_all_gather)


def _make_ipc_handle_frame(rank, world_size, handle_bytes=None, magic=_IPC_WIRE_MAGIC, version=_IPC_WIRE_VERSION):
    """Build a valid IPC handle wire frame."""
    if handle_bytes is None:
        handle_bytes = bytes(range(128))
    packed = _pack_bytes_to_int64s(handle_bytes)
    return [
        magic, version, _MSG_KIND_IPC_HANDLE,
        rank, world_size, _CUDA_IPC_HANDLE_BYTES, _CUDA_IPC_HANDLE_BYTES,
        *packed,
    ]


# ---------------------------------------------------------------------------
# No-object-collective assertion
# ---------------------------------------------------------------------------

def test_setup_path_has_no_object_collective_calls():
    """The pcie_oneshot source must not call any pickle-based object collective."""
    source = inspect.getsource(pcie_oneshot)
    for func_name in (
        "broadcast_object_list(",
        "all_gather_object(",
        "gather_object(",
        "scatter_object(",
    ):
        for lineno, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            code_part = line.split("#")[0] if "#" in line else line
            if func_name in code_part:
                pytest.fail(
                    f"Object collective {func_name!r} found at line {lineno}: {line!r}"
                )


# ---------------------------------------------------------------------------
# Byte packing round-trip
# ---------------------------------------------------------------------------

class TestPackUnpackBytes:
    def test_roundtrip_exact_multiple_of_8(self):
        data = bytes(range(8))
        packed = _pack_bytes_to_int64s(data)
        assert len(packed) == 1
        assert _unpack_int64s_to_bytes(packed, 8) == data

    def test_roundtrip_non_multiple_of_8(self):
        data = bytes(range(10))
        packed = _pack_bytes_to_int64s(data)
        assert _unpack_int64s_to_bytes(packed, 10) == data

    def test_roundtrip_128_bytes(self):
        data = bytes(range(128))
        packed = _pack_bytes_to_int64s(data)
        assert len(packed) == 16
        assert _unpack_int64s_to_bytes(packed, 128) == data

    def test_empty_bytes(self):
        packed = _pack_bytes_to_int64s(b"")
        assert packed == []
        assert _unpack_int64s_to_bytes(packed, 0) == b""


# ---------------------------------------------------------------------------
# String codec
# ---------------------------------------------------------------------------

class TestStringCodec:
    def test_roundtrip_empty(self):
        encoded = _encode_strings_to_int64((), 4, 32)
        assert _decode_strings_from_int64(encoded, 4, 32) == ()

    def test_roundtrip_single(self):
        encoded = _encode_strings_to_int64(("hello",), 4, 32)
        assert _decode_strings_from_int64(encoded, 4, 32) == ("hello",)

    def test_roundtrip_multiple(self):
        strings = ("a", "bb", "ccc", "dddd")
        encoded = _encode_strings_to_int64(strings, 4, 32)
        assert _decode_strings_from_int64(encoded, 4, 32) == strings

    def test_roundtrip_unicode(self):
        strings = ("café", "naïve", "日本語")
        encoded = _encode_strings_to_int64(strings, 4, 64)
        assert _decode_strings_from_int64(encoded, 4, 64) == strings

    def test_oversize_count_rejected_not_truncated(self):
        """Oversize string count must raise, not silently truncate."""
        with pytest.raises(ValueError, match="exceeds max"):
            _encode_strings_to_int64(("a", "b", "c", "d", "e"), 3, 32)

    def test_oversize_length_rejected_not_truncated(self):
        """Oversize string length must raise, not silently truncate."""
        with pytest.raises(ValueError, match="exceeds max"):
            _encode_strings_to_int64(("x" * 100,), 1, 10)

    def test_decode_rejects_negative_count(self):
        max_count, max_len = 4, 32
        encoded = [-1] + [0] * max_count + [0] * ((max_count * max_len + 7) // 8)
        with pytest.raises(ValueError, match="invalid string count"):
            _decode_strings_from_int64(encoded, max_count, max_len)

    def test_decode_rejects_excessive_count(self):
        max_count, max_len = 4, 32
        encoded = [99] + [0] * max_count + [0] * ((max_count * max_len + 7) // 8)
        with pytest.raises(ValueError, match="invalid string count"):
            _decode_strings_from_int64(encoded, max_count, max_len)

    def test_decode_rejects_short_block(self):
        with pytest.raises(ValueError, match="string block too short"):
            _decode_strings_from_int64([], 4, 32)


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------

class TestEnvelopeValidation:
    def test_valid_envelope(self):
        frame = [_IPC_WIRE_MAGIC, _IPC_WIRE_VERSION, _MSG_KIND_IPC_HANDLE, 0, 2, 128, 128]
        _validate_envelope(frame + [0] * 16, 0, 2, _MSG_KIND_IPC_HANDLE, _FRAME_SIZE_IPC_HANDLE)

    def test_wrong_magic(self):
        frame = [0xDEAD, _IPC_WIRE_VERSION, _MSG_KIND_IPC_HANDLE, 0, 2, 128, 128]
        with pytest.raises(RuntimeError, match="magic mismatch"):
            _validate_envelope(frame + [0] * 16, 0, 2, _MSG_KIND_IPC_HANDLE, _FRAME_SIZE_IPC_HANDLE)

    def test_wrong_version(self):
        frame = [_IPC_WIRE_MAGIC, 999, _MSG_KIND_IPC_HANDLE, 0, 2, 128, 128]
        with pytest.raises(RuntimeError, match="version mismatch"):
            _validate_envelope(frame + [0] * 16, 0, 2, _MSG_KIND_IPC_HANDLE, _FRAME_SIZE_IPC_HANDLE)

    def test_wrong_kind(self):
        frame = [_IPC_WIRE_MAGIC, _IPC_WIRE_VERSION, 99, 0, 2, 128, 128]
        with pytest.raises(RuntimeError, match="message kind mismatch"):
            _validate_envelope(frame + [0] * 16, 0, 2, _MSG_KIND_IPC_HANDLE, _FRAME_SIZE_IPC_HANDLE)

    def test_rank_mismatch(self):
        frame = [_IPC_WIRE_MAGIC, _IPC_WIRE_VERSION, _MSG_KIND_IPC_HANDLE, 5, 2, 128, 128]
        with pytest.raises(RuntimeError, match="rank mismatch"):
            _validate_envelope(frame + [0] * 16, 0, 2, _MSG_KIND_IPC_HANDLE, _FRAME_SIZE_IPC_HANDLE)

    def test_world_size_mismatch(self):
        frame = [_IPC_WIRE_MAGIC, _IPC_WIRE_VERSION, _MSG_KIND_IPC_HANDLE, 0, 99, 128, 128]
        with pytest.raises(RuntimeError, match="world size mismatch"):
            _validate_envelope(frame + [0] * 16, 0, 2, _MSG_KIND_IPC_HANDLE, _FRAME_SIZE_IPC_HANDLE)

    def test_wrong_frame_size(self):
        frame = [_IPC_WIRE_MAGIC, _IPC_WIRE_VERSION, _MSG_KIND_IPC_HANDLE, 0, 2, 128, 128]
        with pytest.raises(RuntimeError, match="frame size mismatch"):
            _validate_envelope(frame + [0] * 10, 0, 2, _MSG_KIND_IPC_HANDLE, _FRAME_SIZE_IPC_HANDLE)


# ---------------------------------------------------------------------------
# IPC handle exchange
# ---------------------------------------------------------------------------

class TestExchangeIpcHandles:
    def test_valid_roundtrip(self, monkeypatch):
        local = bytes(range(128))
        _patch_all_gather_echo(monkeypatch)
        handles = _exchange_ipc_handles(local, _make_group())
        assert len(handles) == 2
        assert handles[0] == local
        assert len(handles[1]) == _CUDA_IPC_HANDLE_BYTES

    def test_wrong_local_handle_size_rejected_via_preflight(self, monkeypatch):
        """A wrong-size local handle causes preflight failure, not a hang."""
        _patch_all_gather_with_preflight_failure(monkeypatch, 2, 0, failed_ranks={0})
        with pytest.raises(RuntimeError, match="preflight failed"):
            _exchange_ipc_handles(b"\x00" * 64, _make_group())

    def test_rank_mismatch_in_payload_rejected(self, monkeypatch):
        """A peer reporting wrong rank in the payload envelope is rejected."""
        local = bytes(range(128))
        frame0 = _make_ipc_handle_frame(0, 2, local)
        frame1 = _make_ipc_handle_frame(1, 2, b"\x00" * 128)
        frame1[3] = 99  # wrong rank
        _patch_all_gather_per_rank(monkeypatch, 2, 0, [frame0, frame1])
        with pytest.raises(RuntimeError, match="rank mismatch"):
            _exchange_ipc_handles(local, _make_group())

    def test_version_mismatch_rejected_before_open(self, monkeypatch):
        local = bytes(range(128))
        frame0 = _make_ipc_handle_frame(0, 2, local)
        frame1 = _make_ipc_handle_frame(1, 2, b"\x00" * 128, version=999)
        _patch_all_gather_per_rank(monkeypatch, 2, 0, [frame0, frame1])
        with pytest.raises(RuntimeError, match="version mismatch"):
            _exchange_ipc_handles(local, _make_group())

    def test_magic_mismatch_rejected(self, monkeypatch):
        local = bytes(range(128))
        frame0 = _make_ipc_handle_frame(0, 2, local)
        frame1 = _make_ipc_handle_frame(1, 2, b"\x00" * 128, magic=0xDEAD)
        _patch_all_gather_per_rank(monkeypatch, 2, 0, [frame0, frame1])
        with pytest.raises(RuntimeError, match="magic mismatch"):
            _exchange_ipc_handles(local, _make_group())

    def test_kind_mismatch_rejected(self, monkeypatch):
        local = bytes(range(128))
        frame0 = _make_ipc_handle_frame(0, 2, local)
        frame1 = _make_ipc_handle_frame(1, 2, b"\x00" * 128)
        frame1[2] = 99  # wrong message kind
        _patch_all_gather_per_rank(monkeypatch, 2, 0, [frame0, frame1])
        with pytest.raises(RuntimeError, match="kind mismatch"):
            _exchange_ipc_handles(local, _make_group())


# ---------------------------------------------------------------------------
# Integer contract exchange
# ---------------------------------------------------------------------------

class TestExchangeIntContract:
    def test_valid_roundtrip(self, monkeypatch):
        local = [1, 2, 3, 4]
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_int_contract(local, _make_group())
        assert len(gathered) == 2
        assert gathered[0] == local
        assert gathered[1] == local

    def test_too_many_fields_rejected_via_preflight(self, monkeypatch):
        _patch_all_gather_with_preflight_failure(monkeypatch, 2, 0, failed_ranks={0})
        with pytest.raises(RuntimeError, match="preflight failed"):
            _exchange_int_contract(list(range(_MAX_CONTRACT_INTS + 1)), _make_group())


# ---------------------------------------------------------------------------
# Status string exchange
# ---------------------------------------------------------------------------

class TestExchangeStatusStrings:
    def test_valid_roundtrip(self, monkeypatch):
        local = ("RuntimeError: boom",)
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_status_strings(local, _make_group())
        assert isinstance(gathered, tuple)
        assert len(gathered) == 2
        assert gathered[0] == local
        assert gathered[1] == local

    def test_empty_strings_roundtrip(self, monkeypatch):
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_status_strings((), _make_group())
        assert gathered == ((), ())

    def test_multiple_strings_roundtrip(self, monkeypatch):
        local = ("Error 1: bad", "Error 2: worse", "Error 3: worst")
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_status_strings(local, _make_group())
        assert gathered[0] == local


# ---------------------------------------------------------------------------
# Graph meta exchange
# ---------------------------------------------------------------------------

class TestExchangeGraphMeta:
    def test_valid_roundtrip_independent_lengths(self, monkeypatch):
        """Handles and offsets have independent cardinalities."""
        local_h = [100, 200, 300]
        local_o = [0, 64]  # different length from handles
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_graph_meta(local_h, local_o, _make_group())
        assert len(gathered) == 2
        assert gathered[0] == (local_h, local_o)

    def test_empty_meta_roundtrip(self, monkeypatch):
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_graph_meta([], [], _make_group())
        assert gathered == [([], []), ([], [])]

    def test_too_many_handles_rejected_via_preflight(self, monkeypatch):
        _patch_all_gather_with_preflight_failure(monkeypatch, 2, 0, failed_ranks={0})
        with pytest.raises(RuntimeError, match="preflight failed"):
            _exchange_graph_meta(list(range(_MAX_GRAPH_BUFFERS + 1)), [], _make_group())


# ---------------------------------------------------------------------------
# Channel state exchange
# ---------------------------------------------------------------------------

class TestExchangeChannelState:
    def test_valid_roundtrip(self, monkeypatch):
        normalized = ("chan_a", "chan_b")
        existing = ("chan_a",)
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_channel_state(normalized, existing, _make_group())
        assert len(gathered) == 2
        assert gathered[0] == (normalized, existing)

    def test_empty_state_roundtrip(self, monkeypatch):
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_channel_state((), (), _make_group())
        assert gathered == [((), ()), ((), ())]


# ---------------------------------------------------------------------------
# Capture contract exchange
# ---------------------------------------------------------------------------

class TestExchangeCaptureContract:
    def test_valid_roundtrip(self, monkeypatch):
        logical_id = "graph:target"
        catalog = ("chan_a", "chan_b", "chan_c")
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_capture_contract(logical_id, catalog, _make_group())
        assert len(gathered) == 2
        assert gathered[0] == (logical_id, catalog)

    def test_empty_catalog_roundtrip(self, monkeypatch):
        _patch_all_gather_echo(monkeypatch)
        gathered = _exchange_capture_contract("single", (), _make_group())
        assert gathered == [("single", ()), ("single", ())]


# ---------------------------------------------------------------------------
# Coordinated preflight
# ---------------------------------------------------------------------------

class TestCoordinatedPreflight:
    def test_local_validation_failure_raises_before_payload(self, monkeypatch):
        """If one rank's local data is malformed, preflight raises before
        any rank enters the payload exchange."""
        _patch_all_gather_with_preflight_failure(monkeypatch, 2, 0, failed_ranks={1})
        with pytest.raises(RuntimeError, match="preflight failed"):
            _exchange_ipc_handles(b"\x00" * 128, _make_group())


# ---------------------------------------------------------------------------
# NCCL / control-device contract
# ---------------------------------------------------------------------------

class TestControlDeviceContract:
    def test_nccl_backend_required(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.get_backend", lambda group=None: "gloo")
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
        monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
        with pytest.raises(RuntimeError, match="requires an NCCL process group"):
            _exchange_ipc_handles(b"\x00" * 128, _make_group())

    def test_cuda_required(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.get_backend", lambda group=None: "nccl")
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
        monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
        with pytest.raises(RuntimeError, match="requires CUDA"):
            _exchange_ipc_handles(b"\x00" * 128, _make_group())
