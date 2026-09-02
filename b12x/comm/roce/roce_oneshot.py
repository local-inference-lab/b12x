"""RoCEnante: one-shot RoCE all-reduce and all-gather runtime for multi-node tensor parallelism.

Designed for DGX Spark clusters, whose integrated GPU can read pinned host
memory in place and whose ConnectX-7 can RDMA-write into the same memory.
Each rank owns one pinned region (see ``_roce_proxy.c``); every all-reduce is
one kernel launch that stages the input, rings the proxy thread, waits for
every peer's RDMA-written payload, and reduces in fixed rank order.

Constraints of this first runtime:

* one all-reduce in flight per runtime (single channel, one stream);
* all-reduce messages up to ``max_size`` bytes and all-gather shards up to
  ``max_gather_bytes``, both multiples of 16 bytes;
* every rank of the exchange group must construct the runtime collectively.
"""

from __future__ import annotations

import contextlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from . import _allgather_cute
from ._oneshot_cute import PACK_BYTES, get_launcher, is_launcher_prepared
from ._proxy import Layout, Proxy

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SUPPORTED_WORLD_SIZES = tuple(range(2, 17))
DEFAULT_MAX_SIZE = 2 * 1024 * 1024
DEFAULT_MAX_GATHER_BYTES = 16 * 1024 * 1024
DEFAULT_THREADS = 512
DEFAULT_BLOCKS = 8
DEFAULT_GID_INDEX = 3
# Polls of a peer flag before the kernel gives up (each poll is a system-scope
# load of host memory, roughly a microsecond): about 20 s.
DEFAULT_SPIN_LIMIT = 20_000_000
_SLOT_ALIGNMENT = 4096
_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _env_list(*names: str) -> tuple[str, ...]:
    for name in names:
        raw = os.getenv(name)
        if raw:
            items = []
            for item in raw.split(","):
                item = item.strip().lstrip("=^")
                if item:
                    items.append(item.split(":")[0])
            if items:
                return tuple(items)
    return ()


def _env_int(*names: str, default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw:
            return int(raw)
    return default


def default_gid_index() -> int:
    """``B12X_ROCE_GID_INDEX``, else NCCL's ``NCCL_IB_GID_INDEX``, else 3."""

    return _env_int("B12X_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX", default=DEFAULT_GID_INDEX)


def discover_hcas(gid_index: Optional[int] = None) -> tuple[str, ...]:
    """Return the RDMA devices to use, at most two.

    ``B12X_ROCE_HCA`` (or NCCL's ``NCCL_IB_HCA``) selects explicitly; otherwise
    every active device with a populated GID at ``gid_index`` is used.
    """

    explicit = _env_list("B12X_ROCE_HCA", "NCCL_IB_HCA")
    if explicit:
        return explicit[:2]
    gid_index = default_gid_index() if gid_index is None else int(gid_index)
    found = []
    root = Path("/sys/class/infiniband")
    for dev in sorted(root.glob("*")):
        state = dev / "ports" / "1" / "state"
        gid = dev / "ports" / "1" / "gids" / str(gid_index)
        try:
            if "ACTIVE" not in state.read_text():
                continue
            if gid.read_text().strip().replace(":", "").strip("0") == "":
                continue
        except OSError:
            continue
        found.append(dev.name)
    return tuple(found[:2])


def is_supported(device: torch.device | int | str | None = None) -> bool:
    """True on an integrated GPU with at least one active RDMA device.

    The kernel reads pinned host memory in place, which needs an integrated
    (unified-memory) GPU such as the DGX Spark GB10.
    """

    if not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device() if device is None else torch.device(device).index
    props = torch.cuda.get_device_properties(index if index is not None else 0)
    if not getattr(props, "is_integrated", False):
        return False
    return len(discover_hcas()) > 0


def _normalize_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, int):
        device = torch.device("cuda", device)
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("RoCE all-reduce requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _exchange(local: object, group: ProcessGroup) -> list[object]:
    gathered: list[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local, group=group)
    return gathered


class RoceOneshotAllReduce:
    """One-shot RDMA all-reduce over the DGX Spark 200 GbE fabric."""

    algorithm = "rocenante"

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int = DEFAULT_MAX_SIZE,
        max_gather_bytes: int = DEFAULT_MAX_GATHER_BYTES,
        hca_names: Optional[Sequence[str]] = None,
        gid_index: Optional[int] = None,
        threads: int = DEFAULT_THREADS,
        blocks: int = DEFAULT_BLOCKS,
    ) -> None:
        self.device = _normalize_device(device)
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self._group = exchange_group
        self._closed = False
        self._proxy: Optional[Proxy] = None
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"unsupported RoCE all-reduce world size {self.world_size}; "
                f"supported sizes are {SUPPORTED_WORLD_SIZES}"
            )
        if int(max_size) < PACK_BYTES:
            raise ValueError("max_size must hold at least one 16-byte pack")
        if int(threads) % 32 != 0 or int(threads) < 32 or int(threads) > 1024:
            raise ValueError("threads must be a multiple of 32 between 32 and 1024")
        if int(blocks) < 1:
            raise ValueError("blocks must be positive")
        self.max_size = int(max_size)
        self.max_gather_bytes = int(max_gather_bytes)
        self._threads = int(threads)
        self._blocks = int(blocks)
        self.gid_index = default_gid_index() if gid_index is None else int(gid_index)
        self.spin_limit = _env_int("B12X_ROCE_SPIN_LIMIT", default=DEFAULT_SPIN_LIMIT)
        names = tuple(hca_names) if hca_names else discover_hcas(self.gid_index)
        if not names:
            raise RuntimeError("no active RDMA device found for the RoCE all-reduce")
        self.hca_names = names[:2]

        slot_bytes = _align_up(max(self.max_size, self.max_gather_bytes), _SLOT_ALIGNMENT)
        self._layout = Layout(self.world_size, slot_bytes)
        self._slot_bytes = slot_bytes

        with torch.cuda.device(self.device):
            # Pinned, zero-initialised: flags and the control record start at 0
            # and the first sequence number is 1.
            self._region = torch.zeros(
                self._layout.total_bytes, dtype=torch.uint8, pin_memory=True
            )
            self._counters = torch.zeros(4, dtype=torch.int32, device=self.device)
        host_ptr = self._region.data_ptr()
        device_ptr = self._device_pointer(host_ptr)
        if device_ptr != host_ptr:
            raise RuntimeError(
                "RoCE all-reduce needs host pointers that are directly device "
                "accessible (integrated GPU with unified addressing)"
            )
        self._recv_base = host_ptr + self._layout.recv_off
        self._flag_base = host_ptr + self._layout.flag_off
        self._send_base = host_ptr + self._layout.send_off
        self._ctrl_base = host_ptr + self._layout.ctrl_off
        # ctrl record (kernel-written): seq, nbytes, error seq, missing peer,
        # nbytes per slot (the proxy uses these when it has to catch up)
        self._ctrl_words = self._region[
            self._layout.ctrl_off : self._layout.ctrl_off + 24
        ].view(torch.int32)
        self._error_word = self._ctrl_words[2:3]
        self._epoch_address = self._counters.data_ptr()

        error: Optional[str] = None
        try:
            self._proxy = Proxy(
                world_size=self.world_size,
                rank=self.rank,
                hca_names=self.hca_names,
                gid_index=self.gid_index,
                region_ptr=host_ptr,
                region_bytes=self._layout.total_bytes,
                slot_bytes=slot_bytes,
            )
            blob = self._proxy.local_blob()
        except Exception as exc:  # noqa: BLE001 - reported collectively below
            error = str(exc)
            blob = b""
        statuses = _exchange((error, blob), exchange_group)
        failures = [f"rank {i}: {s[0]}" for i, s in enumerate(statuses) if s[0] is not None]
        if failures:
            self.close()
            raise RuntimeError("RoCE all-reduce setup failed: " + "; ".join(failures))
        try:
            self._proxy.connect([s[1] for s in statuses])
            self._proxy.start()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        verdicts = _exchange(error, exchange_group)
        failures = [f"rank {i}: {v}" for i, v in enumerate(verdicts) if v is not None]
        if failures:
            self.close()
            raise RuntimeError("RoCE all-reduce connect failed: " + "; ".join(failures))
        if self.rank == 0:
            logger.info(
                "RoCEnante ready: world=%d hcas=%s gid_index=%d max_size=%d",
                self.world_size,
                ",".join(self.hca_names),
                self.gid_index,
                self.max_size,
            )

    @staticmethod
    def _device_pointer(host_ptr: int) -> int:
        from cuda.bindings import runtime as cudart

        err, ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        return int(ptr)

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_gather_bytes: int = DEFAULT_MAX_GATHER_BYTES,
        **_ignored: Any,
    ) -> "RoceOneshotAllReduce":
        """Mirror ``comm.pcie.AllReduce.from_exchange_group``; PCIe-only knobs are ignored."""

        capacity = max(int(max_size), int(eager_buffer_bytes or 0))
        return cls(
            exchange_group=exchange_group,
            device=device,
            max_size=capacity,
            max_gather_bytes=max_gather_bytes,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int = DEFAULT_MAX_SIZE,
        max_input_bytes: Optional[int] = None,
        **_ignored: Any,
    ) -> "RoceOneshotAllReduce":
        capacity = max(int(max_size), int(max_input_bytes or 0))
        return cls(exchange_group=process_group, device=device, max_size=capacity)

    # -- policy -----------------------------------------------------------------

    @property
    def supports_all_peer_auxiliary(self) -> bool:
        return False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed or self._proxy is None:
            return False
        if inp.dtype not in SUPPORTED_DTYPES or not inp.is_cuda:
            return False
        if inp.device != self.device or not inp.is_contiguous():
            return False
        # The kernel moves 16-byte packs, so the buffer must be pack-aligned.
        if inp.data_ptr() % PACK_BYTES != 0:
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_size and nbytes % PACK_BYTES == 0

    # -- channels / streams (single channel runtime) ---------------------------

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        return None

    def for_stream(self, stream: object = None, *, channel_id: Optional[str] = None):
        return self

    # -- compilation ------------------------------------------------------------

    def _launcher_key(self, dtype: torch.dtype) -> tuple[object, ...]:
        return (
            _DTYPE_NAMES[dtype],
            self.world_size,
            self.rank,
            self._threads,
            self._layout.slots,
            self._layout.flag_stride,
            self.device.index,
        )

    def _gather_launcher_key(self) -> tuple[object, ...]:
        return (
            self.world_size,
            self.rank,
            self._threads,
            self._layout.slots,
            self._layout.flag_stride,
            self.device.index,
        )

    def prepare(self, dtypes: Sequence[torch.dtype] = (torch.bfloat16,)) -> None:
        """Compile the all-reduce launchers for ``dtypes`` and the all-gather launcher."""

        with torch.cuda.device(self.device):
            for dtype in dtypes:
                get_launcher(*self._launcher_key(dtype))
            _allgather_cute.get_launcher(*self._gather_launcher_key())

    def prepare_graph_all_reduce(self, inp: torch.Tensor, *, stream: object = None) -> None:
        self.prepare((inp.dtype,))

    # -- execution ----------------------------------------------------------------

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if not self.should_allreduce(inp):
            raise ValueError("input is not eligible for the RoCE one-shot all-reduce")
        if out is None:
            out = torch.empty_like(inp)
        elif (
            out.shape != inp.shape
            or out.dtype != inp.dtype
            or not out.is_contiguous()
            or out.data_ptr() % PACK_BYTES != 0
        ):
            raise ValueError("out must be a contiguous, 16-byte-aligned tensor matching the input")
        key = self._launcher_key(inp.dtype)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not is_launcher_prepared(*key):
            raise RuntimeError(
                "RoCE all-reduce launcher must be prepared before CUDA graph capture"
            )
        launcher = get_launcher(*key)
        nbytes = inp.numel() * inp.element_size()
        context = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with torch.cuda.device(self.device), context:
            launcher(
                inp.data_ptr(),
                out.data_ptr(),
                nbytes // PACK_BYTES,
                nbytes,
                self._recv_base,
                self._flag_base,
                self._send_base,
                self._ctrl_base,
                self._slot_bytes,
                self._epoch_address,
                self.spin_limit,
                self._blocks,
            )
        if not capturing:
            self.check_health()
        return out

    def check_health(self) -> None:
        """Raise if the proxy thread died or a kernel wait timed out.

        Cheap (two host memory reads); call it after graph replays, which
        cannot check inline.
        """

        if self._proxy is not None and self._proxy.failed():
            raise RuntimeError(f"RoCE proxy failed: {self._proxy.error()}")
        failed_seq = int(self._error_word.item())
        if failed_seq != 0:
            peer = int(self._ctrl_words[3].item())
            raise RuntimeError(
                f"RoCE all-reduce on rank {self.rank} timed out waiting for rank {peer} "
                f"at sequence {failed_seq}; that rank's proxy or kernel is dead "
                "(rank data is no longer trustworthy)"
            )

    # -- all-gather ---------------------------------------------------------------

    @staticmethod
    def _normalize_dim(inp: torch.Tensor, dim: int) -> int:
        if dim < 0:
            dim += inp.dim()
        return dim

    def should_all_gather(self, inp: torch.Tensor, dim: int = -1) -> bool:
        """Eligible: contiguous CUDA tensor of any dtype, concat along dim 0 or the last dim.

        16-byte-aligned rows take the direct-layout kernel; anything else goes
        through a padded contiguous gather plus a torch reshape, so shape never
        forces a fallback to another backend.
        """

        if self._closed or self._proxy is None or not inp.is_cuda or inp.dim() == 0:
            return False
        if inp.device != self.device or not inp.is_contiguous():
            return False
        if inp.is_complex() or inp.is_sparse or inp.dtype == torch.bool:
            return False
        dim = self._normalize_dim(inp, dim)
        if dim not in (0, inp.dim() - 1):
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_gather_bytes

    def _direct_gather_layout(self, inp: torch.Tensor, dim: int) -> bool:
        nbytes = inp.numel() * inp.element_size()
        if nbytes % PACK_BYTES != 0 or inp.data_ptr() % PACK_BYTES != 0:
            return False
        return dim == 0 or (inp.shape[-1] * inp.element_size()) % PACK_BYTES == 0

    def all_gather(
        self,
        inp: torch.Tensor,
        *,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
    ) -> torch.Tensor:
        """Concatenate every rank's ``inp`` along ``dim`` (0 or the last dim).

        With 16-byte-aligned rows the kernel writes the concatenated layout
        directly (no reshape or copy afterwards).  Otherwise the shards are
        gathered contiguously with 16-byte padding and finished with a torch
        reshape, which still keeps the collective on RDMA.
        """

        if not self.should_all_gather(inp, dim):
            raise ValueError("input is not eligible for the RoCE all-gather")
        dim = self._normalize_dim(inp, dim)
        shape = list(inp.shape)
        shape[dim] *= self.world_size
        context = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with torch.cuda.device(self.device), context:
            if self._direct_gather_layout(inp, dim):
                if out is None:
                    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
                elif list(out.shape) != shape or out.dtype != inp.dtype or not out.is_contiguous():
                    raise ValueError("out must be a contiguous tensor of the gathered shape")
                nbytes = inp.numel() * inp.element_size()
                row_packs = (
                    nbytes // PACK_BYTES
                    if dim == 0
                    else (inp.shape[-1] * inp.element_size()) // PACK_BYTES
                )
                self._launch_gather(inp.data_ptr(), out.data_ptr(), nbytes, row_packs)
                return out
            # General path: pad each shard to a whole number of packs, gather
            # contiguously, then let torch produce the requested layout.
            nbytes = inp.numel() * inp.element_size()
            padded = _align_up(nbytes, PACK_BYTES)
            staged = torch.empty(padded, dtype=torch.uint8, device=inp.device)
            staged[:nbytes].copy_(inp.reshape(-1).view(torch.uint8))
            gathered = torch.empty(self.world_size * padded, dtype=torch.uint8, device=inp.device)
            self._launch_gather(staged.data_ptr(), gathered.data_ptr(), padded, padded // PACK_BYTES)
            stacked = (
                gathered.view(self.world_size, padded)[:, :nbytes]
                .reshape(-1)
                .view(inp.dtype)
                .reshape(self.world_size, *inp.shape)
            )
            result = stacked.movedim(0, dim).reshape(shape)
            if out is None:
                return result.contiguous()
            out.copy_(result)
            return out

    def _launch_gather(self, input_address: int, output_address: int, nbytes: int, row_packs: int) -> None:
        key = self._gather_launcher_key()
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not _allgather_cute.is_launcher_prepared(*key):
            raise RuntimeError(
                "RoCE all-gather launcher must be prepared before CUDA graph capture"
            )
        launcher = _allgather_cute.get_launcher(*key)
        launcher(
            input_address,
            output_address,
            nbytes // PACK_BYTES,
            nbytes,
            row_packs,
            self._recv_base,
            self._flag_base,
            self._send_base,
            self._ctrl_base,
            self._slot_bytes,
            self._epoch_address,
            self.spin_limit,
            self._blocks,
        )
        if not capturing:
            self.check_health()

    @contextmanager
    def capture(self, stream: object = None, *, channel_id: Optional[str] = None):
        yield self

    # -- diagnostics / lifecycle --------------------------------------------------

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "world_size": self.world_size,
            "rank": self.rank,
            "hcas": list(self.hca_names),
            "max_size": self.max_size,
            "max_gather_bytes": self.max_gather_bytes,
            "slot_bytes": self._slot_bytes,
            "epoch": int(self._counters[0].item()),
            "error_seq": int(self._error_word.item()),
            "error_peer": int(self._ctrl_words[3].item()),
            "ctrl_seq": int(self._ctrl_words[0].item()),
            "spin_limit": self.spin_limit,
        }
        if self._proxy is not None:
            info.update(self._proxy.stats())
            info["peer_hca"] = {
                peer: self._proxy.peer_hca(peer)
                for peer in range(self.world_size)
                if peer != self.rank
            }
        return info

    def close(self) -> None:
        """Stop the proxy and release the transport.

        Waits for the device first so no in-flight kernel still reads the
        pinned region or rings the doorbell after the proxy is gone.  Peers
        that write to this rank afterwards see a remote-access completion
        error and raise on their side, which is the intended shutdown signal.
        """

        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            torch.cuda.synchronize(self.device)
        if self._proxy is not None:
            self._proxy.close()
            self._proxy = None

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        with contextlib.suppress(Exception):
            self.close()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


__all__ = [
    "DEFAULT_MAX_GATHER_BYTES",
    "DEFAULT_MAX_SIZE",
    "SUPPORTED_DTYPES",
    "SUPPORTED_WORLD_SIZES",
    "RoceOneshotAllReduce",
    "default_gid_index",
    "discover_hcas",
    "is_supported",
]
