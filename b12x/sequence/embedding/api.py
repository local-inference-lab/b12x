"""Exact unquantized embedding lookup without output or replay allocation."""
import torch

from ..._lib.gating import default_is_supported
from . import META
from ._kernel import _compile, launch


def _check_weight(weight):
    if not weight.is_cuda or weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("embedding weight must be CUDA BF16 or FP32")
    if weight.ndim != 2 or weight.shape[1] <= 0:
        raise ValueError("embedding weight must be [table_rows, positive width]")
    if weight.stride(1) != 1 or weight.stride(0) < weight.shape[1]:
        raise ValueError("embedding weight must have contiguous nonoverlapping rows")


@torch.library.custom_op("b12x::embedding_out", mutates_args=("out",))
def _embedding_out(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                   num_rows: torch.Tensor | None = None) -> None:
    _check_weight(weight)
    if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous():
        raise TypeError("embedding IDs must be contiguous Int32 or Int64")
    if ids.device != weight.device or out.device != weight.device:
        raise ValueError("embedding tensors must share the CUDA device")
    if out.shape != (*ids.shape, weight.shape[1]) or out.dtype != weight.dtype:
        raise ValueError("embedding output must have ids.shape + (width,) and weight dtype")
    if not out.is_contiguous() or ids.numel() >= 2**31:
        raise ValueError("embedding requires contiguous output and an Int32 launch count")
    if torch._C._overlaps(out, weight) or torch._C._overlaps(out, ids):
        raise ValueError("embedding output must not alias inputs")
    if num_rows is not None:
        if (num_rows.device != weight.device or num_rows.dtype != torch.int32
                or num_rows.numel() != 1 or not num_rows.is_contiguous()):
            raise ValueError("num_rows must be a CUDA Int32 scalar on the weight device")
        if ids.numel() == 0:
            raise ValueError("device-count lookup requires positive output capacity")
        if torch._C._overlaps(out, num_rows):
            raise ValueError("embedding output must not alias num_rows")
    launch(weight, ids, out, num_rows)


@_embedding_out.register_fake
def _embedding_out_fake(weight, ids, out, num_rows=None):
    return None


def run(weight, ids, *, out, num_rows=None):
    """Copy exact rows into caller-owned ``out``; preserve the weight dtype.

    IDs may have any contiguous shape (including a scalar). The output shape
    is ``(*ids.shape, weight.shape[1])``. A strided table is allowed when each
    row is contiguous. ``num_rows`` optionally supplies a device Int32 live
    count bounded by ``ids.numel()``; only that flattened prefix is written,
    leaving inactive output/IDs untouched. IDs and counts can change in graph
    replay. Invalid live IDs or counts trap on the device, never read outside
    the table or silently return zeros. CUDA errors surface asynchronously.
    """
    _embedding_out(weight, ids, out, num_rows)
    return out


def precompile(weight, *, id_dtype=torch.int64):
    """Compile and warm one width/type specialization after loading, before capture."""
    _check_weight(weight)
    if id_dtype not in (torch.int32, torch.int64):
        raise TypeError("embedding IDs must be Int32 or Int64")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("embedding precompile must run before CUDA graph capture")
    _compile(weight.shape[1], weight.dtype, id_dtype, weight.device.index)
    ids = torch.zeros(1, dtype=id_dtype, device=weight.device)
    out = torch.empty((1, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    # A zero live count warms the module without reading an empty/unloaded row.
    count = torch.zeros((), dtype=torch.int32, device=weight.device)
    run(weight, ids, out=out, num_rows=count)


def is_supported(device=None):
    return default_is_supported(device, requires=META.requires)


def clear_caches():
    _compile.cache_clear()
