"""PCIe two-shot sequence-parallel collectives with fp8 transport.

Pull-based reduce_scatter / all_gather over IPC-mapped peer slabs for
TP sequence parallelism: values are quantized exactly once at the
source (per-token e4m3 scales), moved as fp8, and dequantized fused
with the fp32 reduction (RS) or the bf16 store (AG). Staging goes
through alternating eager slots so both ops are CUDA-graph-capturable;
a runtime instance must not be shared concurrently across CUDA streams.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _finish_collective_runtime_setup,
    _raise_local_cleanup_errors,
    _align_up,
    _coordinated_close_channels,
    _cuda_device_index,
    _normalize_device,
    _OwnedSharedBuffer,
    _require_collective_contract,
    _require_full_grid_residency,
    _run_collective_preallocation_setup,
)

SUPPORTED_WORLD_SIZES = (2, 4, 8)
FP8_MAX = 448.0
TWOSHOT_REQUIRED_SMS = 64


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_twoshot.cu")
    verbose = os.getenv("SPARKINFER_PCIE_TWOSHOT_VERBOSE_BUILD", "0") == "1"
    return load(
        name="sparkinfer_pcie_twoshot_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def quantize_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-row e4m3 quantization (tests / non-fused callers)."""
    assert x.dim() == 2
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_(min=1e-12)
    scale = amax / FP8_MAX
    payload = (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return payload, scale.squeeze(-1).contiguous()


class PCIeTwoShotSP:
    """Two-shot fp8-transport reduce_scatter / all_gather runtime."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("use PCIeTwoShotSP.from_exchange_group()")

    @classmethod
    def _from_prepared_factory(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        ext_module,
        fptr: int,
        owned_buffers: Sequence[_OwnedSharedBuffer],
        ipc: CudaRTLibrary,
        exchange_group: ProcessGroup,
        max_rows: int,
        row_elems: int,
    ) -> "PCIeTwoShotSP":
        self = object.__new__(cls)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self._ext = ext_module
        self._fptr = fptr
        self._owned_buffers = list(owned_buffers)
        self._ipc = ipc
        self.exchange_group = exchange_group
        self.max_rows = max_rows
        self.row_elems = row_elems
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._closed_ipc_import_indices: set[tuple[int, int]] = set()
        return self

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        row_elems: int,
        ext_module=None,
    ) -> "PCIeTwoShotSP":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)

        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            normalized_max_rows = int(max_rows)
            normalized_row_elems = int(row_elems)
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {world_size}")
            if device_obj.type != "cuda":
                raise ValueError("PCIe twoshot requires a CUDA device")
            if normalized_max_rows <= 0:
                raise ValueError("max_rows must be positive")
            if normalized_row_elems <= 0 or normalized_row_elems % 16 != 0:
                raise ValueError("row_elems must be a positive multiple of 16")
            if normalized_max_rows % world_size != 0:
                raise ValueError("max_rows must be divisible by world size")
            return device_obj, normalized_max_rows, normalized_row_elems

        device_obj, max_rows, row_elems = _run_collective_preallocation_setup(
            owner="PCIe twoshot argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )

        _require_full_grid_residency(
            owner="PCIe twoshot",
            required_sms=TWOSHOT_REQUIRED_SMS,
            device=device_obj,
            exchange_group=exchange_group,
        )

        def prepare():
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(_cuda_device_index(device_obj))
            prepared_ext = ext_module or _load_extension()

            # Per-slot staging: [world][pack_stride] Fp8Packs then
            # [world][scale_stride] fp32 scales, regions 256B-aligned.
            max_rows_per_rank = max_rows // world_size
            packs_per_row = row_elems // 16
            pack_stride = _align_up(max_rows_per_rank * packs_per_row, 16)
            payload_bytes = world_size * pack_stride * 16
            scale_offset = _align_up(payload_bytes, IPC_SLAB_ALIGNMENT)
            scale_stride = _align_up(max_rows_per_rank, 64)
            slot_bytes = _align_up(
                scale_offset + world_size * scale_stride * 4, IPC_SLAB_ALIGNMENT
            )
            signal_bytes = _align_up(int(prepared_ext.meta_size()), IPC_SLAB_ALIGNMENT)
            slab_bytes = signal_bytes + 2 * slot_bytes
            return (
                prepared_ipc,
                prepared_ext,
                pack_stride,
                scale_offset,
                scale_stride,
                signal_bytes,
                slot_bytes,
                slab_bytes,
            )

        (
            ipc,
            ext,
            pack_stride,
            scale_offset,
            scale_stride,
            signal_bytes,
            slot_bytes,
            slab_bytes,
        ) = _run_collective_preallocation_setup(
            owner="PCIe twoshot",
            exchange_group=exchange_group,
            setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe twoshot channel layout",
            exchange_group=exchange_group,
            contract=(
                int(max_rows),
                int(row_elems),
                int(pack_stride),
                int(scale_offset),
                int(scale_stride),
                int(signal_bytes),
                int(slab_bytes),
            ),
        )

        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            slab_bytes,
            zero_fill=True,
            ipc=ipc,
        )
        peer_ptrs = list(shared.peer_ptrs)
        signal_ptrs = peer_ptrs
        staging0 = [p + signal_bytes for p in peer_ptrs]
        staging1 = [p + signal_bytes + slot_bytes for p in peer_ptrs]

        fptr = 0
        init_error: BaseException | None = None
        try:
            fptr = ext.init_twoshot(
                signal_ptrs,
                staging0,
                staging1,
                pack_stride,
                scale_offset,
                scale_stride,
                rank,
            )
        except Exception as exc:
            init_error = exc

        def abort_native_runtime() -> None:
            nonlocal fptr
            if fptr:
                ext.dispose(fptr)
                fptr = 0

        _finish_collective_runtime_setup(
            owner="PCIe twoshot",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=shared,
            local_error=init_error,
            local_cleanup=abort_native_runtime,
        )
        return cls._from_prepared_factory(
            rank=rank,
            world_size=world_size,
            device=device_obj,
            ext_module=ext,
            fptr=fptr,
            owned_buffers=[shared],
            ipc=ipc,
            exchange_group=exchange_group,
            max_rows=max_rows,
            row_elems=row_elems,
        )

    def _check(self, payload: torch.Tensor, scale: torch.Tensor, rows: int) -> None:
        if self._closed:
            raise RuntimeError("PCIeTwoShotSP is closed")
        if payload.shape != (rows, self.row_elems):
            raise ValueError(
                f"payload shape {tuple(payload.shape)} != ({rows}, {self.row_elems})"
            )
        if scale.numel() != rows:
            raise ValueError(f"scale numel {scale.numel()} != {rows}")

    def reduce_scatter_fp8(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        """Sum per-token-quantized partials; return the local row shard."""
        rows = payload.shape[0]
        self._check(payload, scale, rows)
        if rows % self.world_size != 0:
            raise ValueError("rows must be divisible by world size")
        if out is None:
            out = torch.empty(
                rows // self.world_size,
                self.row_elems,
                dtype=torch.bfloat16,
                device=self.device,
            )
        self._ext.reduce_scatter_fp8(
            self._fptr, payload, scale, out, threads, block_limit
        )
        return out

    def all_gather_fp8(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        """Gather per-token-quantized shards; return bf16 full width."""
        rows = payload.shape[0]
        self._check(payload, scale, rows)
        if out is None:
            out = torch.empty(
                rows * self.world_size,
                self.row_elems,
                dtype=torch.bfloat16,
                device=self.device,
            )
        self._ext.all_gather_fp8(self._fptr, payload, scale, out, threads, block_limit)
        return out

    def _closed_import_indices(self) -> set[tuple[int, int]]:
        closed = getattr(self, "_closed_ipc_import_indices", None)
        if closed is None:
            closed = set()
            self._closed_ipc_import_indices = closed
        return closed

    def _all_python_ipc_imports_closed(self, closed: set[tuple[int, int]]) -> bool:
        return all(
            (buffer_index, remote_index) in closed
            for buffer_index, shared in enumerate(self._owned_buffers)
            for remote_index, _ in enumerate(shared.remote_ptrs)
        )

    def _close_ipc_imports_strict(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        failures: list[tuple[str, Exception]] = []
        if self._fptr:
            try:
                self._ext.dispose(self._fptr)
            except Exception as exc:
                failures.append(("native runtime", exc))
            else:
                self._fptr = 0

        closed = self._closed_import_indices()
        for buffer_index, shared in enumerate(self._owned_buffers):
            for remote_index, ptr in enumerate(shared.remote_ptrs):
                key = (buffer_index, remote_index)
                if key in closed:
                    continue
                try:
                    self._ipc.cudaIpcCloseMemHandle(ptr)
                except Exception as exc:
                    failures.append((f"CUDA IPC import {ptr}", exc))
                else:
                    closed.add(key)

        if (
            not failures
            and not self._fptr
            and self._all_python_ipc_imports_closed(closed)
        ):
            self._ipc_imports_closed = True
        if failures:
            _raise_local_cleanup_errors("PCIe twoshot", "IPC import close", failures)

    def _free_ipc_exports_strict(self) -> None:
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports_strict()
        failures: list[tuple[str, Exception]] = []
        remaining = []
        for shared in self._owned_buffers:
            try:
                self._ipc.cudaFree(shared.local_ptr)
            except Exception as exc:
                remaining.append(shared)
                failures.append((f"CUDA IPC export {shared.local_ptr}", exc))
        self._owned_buffers = remaining
        if not remaining:
            self._ipc_exports_freed = True
        if failures:
            _raise_local_cleanup_errors("PCIe twoshot", "IPC export free", failures)

    def close(self) -> None:
        if getattr(self, "_coordinated_close_complete", False):
            return
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        # GC cannot prove queued CUDA work is complete.  Explicit close() is the
        # only path allowed to synchronize, unmap imports, and free exports.
        if getattr(self, "_coordinated_close_complete", False):
            return
        if getattr(self, "_fptr", 0) or getattr(self, "_owned_buffers", ()):
            _quarantine[id(self)] = self
