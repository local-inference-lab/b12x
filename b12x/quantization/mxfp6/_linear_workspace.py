"""Caller-owned FP6 linear workspace and shared capacity-bound execution."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import torch

from b12x._lib.fp6 import as_grouped_mxfp6_scale_view
from b12x.gemm.blockscaled._sm103 import pointer
from . import _rows


def _geometry(capacity, k, fmt, packed):
    if not isinstance(capacity, int) or not isinstance(k, int) or not 0 < capacity < 2**31 or not 0 < k < 2**31 or k % 128:
        raise ValueError("FP6 workspace requires positive capacity and K divisible by 128")
    if fmt not in ("e2m3", "e3m2", "e4m3") or packed and fmt == "e4m3":
        raise ValueError("FP6 workspace requires a valid activation format and storage recipe")


def _tensor(tensor, shape, dtype, device, name):
    if (tensor.device != device or device.type != "cuda" or tuple(tensor.shape) != shape
            or tensor.dtype != dtype or not tensor.is_contiguous()
            or tensor.numel() and tensor.data_ptr() % 16):
        raise ValueError(f"{name} requires aligned contiguous {dtype} storage of shape {shape} on {device}")


def _disjoint(writes, reads=()):
    for index, tensor in enumerate(writes):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        for other in (*writes[index + 1:], *reads):
            if tensor.numel() and other.numel() and start < other.data_ptr() + other.numel() * other.element_size() and other.data_ptr() < end:
                raise ValueError("FP6 writable storage must not overlap other buffers or inputs")


@dataclass(frozen=True)
class FP6LinearWorkspace:
    max_tokens: int
    in_features: int
    act_fmt: str
    per_row: bool
    packed: bool
    values: torch.Tensor
    scale_storage: torch.Tensor
    global_scales: torch.Tensor
    inverse_scales: torch.Tensor
    alpha: torch.Tensor

    def _validate(self, source, weight_scale):
        _geometry(self.max_tokens, self.in_features, self.act_fmt, self.packed)
        if source.ndim != 2 or source.shape[1] != self.in_features or source.shape[0] > self.max_tokens:
            raise ValueError("FP6 source must have shape [M,K] within workspace capacity")
        device, k, capacity = source.device, self.in_features, self.max_tokens
        _tensor(source, tuple(source.shape), torch.bfloat16, device, "FP6 source")
        _tensor(weight_scale, (1,), torch.float32, device, "FP6 weight global scale")
        specs = (
            (self.values, (capacity, k * 3 // 4 if self.packed else k), torch.uint8, "FP6 values"),
            (self.scale_storage, (((capacity + 127) // 128) * (k // 128) * 512,), torch.uint8, "FP6 block scales"),
            (self.global_scales, (capacity if self.per_row else 1,), torch.float32, "FP6 global scales"),
            (self.inverse_scales, (capacity,), torch.bfloat16, "FP6 inverse scales"),
            (self.alpha, (1,), torch.float32, "FP6 alpha"),
        )
        for tensor, shape, dtype, name in specs:
            _tensor(tensor, shape, dtype, device, name)
        buffers = tuple(spec[0] for spec in specs)
        _disjoint(buffers, (source, weight_scale))
        return buffers

    def quantize(self, source, weight_scale):
        self._validate(source, weight_scale)
        m, k = source.shape
        if not m:
            return
        with torch.cuda.device(source.device):
            props = torch.cuda.get_device_properties(source.device)
            architecture = f"sm_{props.major}{props.minor}a"
            stream = cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream)
            scales = _rows.compile_scales(k, self.act_fmt, self.per_row, source.device.index, architecture)
            quantizer = _rows.compile_quantizer(k, self.act_fmt, self.per_row, self.packed, source.device.index, architecture)
            scales(*(pointer(t, v) for t, v in zip((cutlass.BFloat16, cutlass.Float32,
                cutlass.Float32, cutlass.BFloat16, cutlass.Float32),
                (source, weight_scale, self.global_scales, self.inverse_scales, self.alpha), strict=True)), cutlass.Int32(m), stream)
            grid = min((m * (k // 32) + 127) // 128, props.multi_processor_count * 4)
            quantizer(pointer(cutlass.BFloat16, source), pointer(cutlass.Float32, self.global_scales),
                      pointer(cutlass.Uint8, self.values), pointer(cutlass.Uint8, self.scale_storage),
                      cutlass.Int32(m), cutlass.Int32(grid), stream)

    def scale_view(self, rows):
        if not 0 <= rows <= self.max_tokens:
            raise ValueError("FP6 scale view exceeds workspace capacity")
        size = ((rows + 127) // 128) * (self.in_features // 128) * 512
        return as_grouped_mxfp6_scale_view(self.scale_storage[:size].view(1, -1), rows, self.in_features)


def allocate_fp6_linear_workspace(max_tokens, in_features, *, device="cuda", act_fmt="e3m2",
                                  per_row=True, packed=None):
    """Allocate fixed quantization capacity before warmup and graph capture.

    SM103 defaults to packed FP6 activations; SM12x defaults to byte containers.
    E4M3 activations always use bytes. Storage belongs to the caller and must
    not be shared by overlapping launches or independent captured graphs.
    """
    _geometry(max_tokens, in_features, act_fmt, packed)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("FP6 linear workspace requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if packed is None:
        packed = torch.cuda.get_device_capability(device) == (10, 3) and act_fmt != "e4m3"
    if packed and act_fmt == "e4m3":
        raise ValueError("E4M3 activations require byte storage")
    stored_k = in_features * 3 // 4 if packed else in_features
    return FP6LinearWorkspace(
        max_tokens, in_features, act_fmt, per_row, packed,
        torch.empty((max_tokens, stored_k), device=device, dtype=torch.uint8),
        torch.empty(((max_tokens + 127) // 128) * (in_features // 128) * 512, device=device, dtype=torch.uint8),
        torch.empty(max_tokens if per_row else 1, device=device, dtype=torch.float32),
        torch.empty(max_tokens, device=device, dtype=torch.bfloat16),
        torch.empty(1, device=device, dtype=torch.float32),
    )


def quantize_fp6_weight(source, fmt, global_scale):
    """Prepare packed checkpoint values with the fixed supplied global scale."""
    if source.ndim != 2 or fmt not in ("e2m3", "e3m2"):
        raise ValueError("FP6 weights require a matrix and an FP6 format")
    _tensor(source, tuple(source.shape), torch.bfloat16, source.device, "FP6 weight source")
    _tensor(global_scale, (1,), torch.float32, source.device, "FP6 weight global scale")
    m, k = source.shape
    workspace = allocate_fp6_linear_workspace(m, k, device=source.device,
        act_fmt=fmt, per_row=False, packed=True)
    with torch.cuda.device(source.device):
        props = torch.cuda.get_device_properties(source.device)
        architecture = f"sm_{props.major}{props.minor}a"
        fn = _rows.compile_quantizer(k, fmt, False, True, source.device.index, architecture)
        fn(pointer(cutlass.BFloat16, source), pointer(cutlass.Float32, global_scale),
           pointer(cutlass.Uint8, workspace.values), pointer(cutlass.Uint8, workspace.scale_storage),
           cutlass.Int32(m), cutlass.Int32(min((m * (k // 32) + 127) // 128, props.multi_processor_count * 4)),
           cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream))
    return workspace.values, workspace.scale_storage


def linear_with_workspace(source, weight, scale_storage, global_scale, fmt, n, k,
                          *, out=None, act_fmt=None, workspace=None, per_row=True,
                          expected_m=None, stream=None):
    from b12x._lib.dense_gemm import dense_gemm
    from b12x.gemm.blockscaled._a16 import _stream_context
    act_fmt = act_fmt or fmt
    if weight.ndim == 3 and weight.shape[-1] == 1:
        weight = weight[..., 0]
    if source.ndim != 2 or source.shape[1] != k or n <= 0 or n % 8:
        raise ValueError("FP6 linear requires [M,K] input and N divisible by eight")
    if expected_m is not None and (expected_m <= 0 or expected_m < source.shape[0]):
        raise ValueError("expected_m must be a positive capacity covering live rows")
    m = source.shape[0]
    if source.device.type != "cuda" or fmt not in ("e2m3", "e3m2"):
        raise ValueError("FP6 linear requires CUDA input and FP6 weights")
    with torch.cuda.device(source.device), _stream_context(stream, source.device):
        if workspace is None:
            source = source.to(torch.bfloat16).contiguous()
            workspace = allocate_fp6_linear_workspace(max(1, expected_m or m), k,
                device=source.device, act_fmt=act_fmt, per_row=per_row)
        if not isinstance(workspace, FP6LinearWorkspace) or workspace.act_fmt != act_fmt or workspace.per_row != per_row:
            raise ValueError("FP6 workspace must match the activation format and scaling contract")
        if expected_m is not None and expected_m != workspace.max_tokens:
            raise ValueError("expected_m must match the FP6 workspace capacity")
        native = torch.cuda.get_device_capability(source.device) == (10, 3)
        if workspace.packed and not native:
            raise ValueError("SM12x linear workspaces require byte-container activations")
        if weight.ndim != 2 or weight.shape[1] not in (k, k * 3 // 4):
            raise ValueError("FP6 weights require K byte containers or 3K/4 packed bytes")
        _tensor(weight, (n, weight.shape[1]), torch.uint8, source.device, "FP6 weights")
        _tensor(scale_storage, (((n + 127) // 128) * (k // 128) * 512,), torch.uint8, source.device, "FP6 weight block scales")
        if out is None:
            out = torch.empty((m, n), device=source.device, dtype=torch.bfloat16)
        if out.device != source.device or out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n) or not out.is_contiguous():
            raise ValueError("FP6 linear output must be contiguous BF16 [M,N] on the source device")
        _tensor(out, (m, n), torch.bfloat16, source.device, "FP6 output")
        buffers = workspace._validate(source, global_scale)
        _disjoint((*buffers, out), (source, weight, scale_storage, global_scale))
        workspace.quantize(source, global_scale)
        dense_gemm(
            (workspace.values[:m, :, None], workspace.scale_view(m)),
            (weight.reshape(n, weight.shape[1], 1), as_grouped_mxfp6_scale_view(scale_storage.view(1, -1), n, k)),
            out=out[..., None], alpha=workspace.alpha, ab_dtype=f"float6_{fmt}fn",
            sf_dtype="float8_e8m0fnu", sf_vec_size=32, c_dtype="bfloat16",
            a_fmt=act_fmt, b_fmt=fmt, a_preexpanded=not workspace.packed,
            b_preexpanded=weight.shape[1] == k, b_packed=weight.shape[1] != k,
            row_scale=workspace.inverse_scales[:m] if per_row else None,
            expected_m=workspace.max_tokens,
        )
    return out
