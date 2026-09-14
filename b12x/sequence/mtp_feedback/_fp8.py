"""Per-stream ordinary RMS feedback with separate compact K128 FP8 projections."""

from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch

from b12x._lib.architecture import UnsupportedArchitectureError, supports_architecture
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, fabs_f32, fmax_f32, warp_reduce
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import align_up, materialize_scratch_view
from b12x._lib.utils import current_cuda_stream
from b12x._lib.fp8_gemm import compile_kernel as compile_projection
from ._concat import _pointer, compile_norm


class QuantizeRows:
    """One warp quantizes 128 BF16 values with an ordinary FP32 scale."""

    def __init__(self, hidden):
        self.hidden = hidden
        self.groups = hidden // 128

    @cute.jit
    def __call__(self, source, output, scales, rows: c.Int32, stream: cuda.CUstream):
        self.kernel(source, output, scales).launch(
            grid=(rows * self.groups, 1, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source, output, scales):
        group, _, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        offset = c.Int64(group) * 128 + c.Int64(lane) * 4
        x0 = c.Float32(source[offset])
        x1 = c.Float32(source[offset + 1])
        x2 = c.Float32(source[offset + 2])
        x3 = c.Float32(source[offset + 3])
        maximum = fmax_f32(
            fmax_f32(fabs_f32(x0), fabs_f32(x1)), fmax_f32(fabs_f32(x2), fabs_f32(x3))
        )
        maximum = warp_reduce(maximum, fmax_f32)
        scale = fmax_f32(maximum, c.Float32(1e-10)) / c.Float32(448)
        if lane == 0:
            scales[c.Int64(group)] = scale
        output[c.Int64(group) * 32 + c.Int64(lane)] = cvt_f32x4_to_e4m3x4(
            x0 / scale,
            x1 / scale,
            x2 / scale,
            x3 / scale,
        )


class AddEmbedding:
    """Add two BF16 projection results with one BF16 output rounding."""

    def __init__(self, hidden, streams):
        self.hidden = hidden
        self.streams = streams

    @cute.jit
    def __call__(self, embedding, output, tokens: c.Int32, stream: cuda.CUstream):
        elements = c.Int64(tokens) * self.streams * self.hidden
        self.kernel(embedding, output, elements).launch(
            grid=(cute.ceil_div(elements, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, embedding, output, elements: c.Int64):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        offset = c.Int64(block) * 256 + c.Int64(thread)
        if offset < elements:
            token = offset // (self.streams * self.hidden)
            column = offset % self.hidden
            output[offset] = c.BFloat16(
                c.Float32(output[offset])
                + c.Float32(embedding[token * self.hidden + column])
            )


@lru_cache(maxsize=None)
def compile_aux(hidden, streams, device_index, architecture):
    from b12x._lib.utils import make_ptr

    def ptr(dtype):
        return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)

    with torch.cuda.device(device_index):
        quant = compile_cute(
            QuantizeRows(hidden),
            ptr(c.BFloat16),
            ptr(c.Uint32),
            ptr(c.Float32),
            c.Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "sequence.mtp_feedback.fp8_quantize",
                1,
                ("hidden", hidden),
                ("architecture", architecture),
            ),
        )
        add = compile_cute(
            AddEmbedding(hidden, streams),
            ptr(c.BFloat16),
            ptr(c.BFloat16),
            c.Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "sequence.mtp_feedback.fp8_add",
                1,
                ("hidden", hidden),
                ("streams", streams),
                ("architecture", architecture),
            ),
        )
    return quant, add


@dataclass(frozen=True)
class BoundLaunches:
    norm: object
    norm_args: tuple
    quant_args: tuple
    projection_args: tuple
    add_args: tuple


@dataclass
class BackendPlan:
    layout: dict
    norms: dict
    quant: object
    projection: object
    add: object
    warmed: bool = False

    def bind(
        self,
        planned,
        *,
        scratch,
        token_embedding,
        multi_state,
        token_norm_weight,
        state_norm_weight,
        embedding_fc_weight,
        hidden_fc_weight,
        embedding_fc_scale,
        hidden_fc_scale,
        positions,
        output,
        tokens,
    ):
        from ._impl import Binding, _overlaps, _require_tensor

        caps = planned.caps
        if not isinstance(token_embedding, torch.Tensor) or token_embedding.ndim != 2:
            raise ValueError("token_embedding must have shape [rows, hidden_size]")
        available = token_embedding.shape[0]
        live = planned._live_tokens(available if tokens is None else tokens)
        if not live <= available <= caps.max_tokens:
            raise ValueError(
                "FP8 feedback input rows must cover live tokens within capacity"
            )
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 3
            or not live <= output.shape[0] <= caps.max_tokens
        ):
            raise ValueError(
                "FP8 feedback output must have shape [rows, streams, hidden_size] within capacity"
            )
        inputs = dict(
            token_embedding=token_embedding,
            multi_state=multi_state,
            token_norm_weight=token_norm_weight,
            state_norm_weight=state_norm_weight,
        )
        shapes = dict(
            token_embedding=(available, caps.hidden_size),
            multi_state=(available, caps.streams, caps.hidden_size),
            token_norm_weight=(caps.hidden_size,),
            state_norm_weight=(caps.hidden_size,),
        )
        for name, tensor in inputs.items():
            _require_tensor(name, tensor, shape=shapes[name], caps=caps)
        _require_tensor(
            "output",
            output,
            shape=(output.shape[0], caps.streams, caps.hidden_size),
            caps=caps,
        )
        for name, tensor, dtype, shape in (
            (
                "embedding_fc_weight",
                embedding_fc_weight,
                torch.float8_e4m3fn,
                (caps.hidden_size, caps.hidden_size),
            ),
            (
                "hidden_fc_weight",
                hidden_fc_weight,
                torch.float8_e4m3fn,
                (caps.hidden_size, caps.hidden_size),
            ),
            (
                "embedding_fc_scale",
                embedding_fc_scale,
                torch.float32,
                (caps.hidden_size // 128, caps.hidden_size // 128),
            ),
            (
                "hidden_fc_scale",
                hidden_fc_scale,
                torch.float32,
                (caps.hidden_size // 128, caps.hidden_size // 128),
            ),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != caps.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"{name} must be aligned contiguous {dtype} {shape} on the plan device"
                )
            inputs[name] = tensor
        if (
            not isinstance(positions, torch.Tensor)
            or positions.shape != (available,)
            or positions.dtype not in (torch.int32, torch.int64)
            or positions.device != caps.device
            or not positions.is_contiguous()
        ):
            raise ValueError(
                "positions must be contiguous Int32/Int64 positions on the plan device"
            )
        if output.numel() and output.data_ptr() % 16:
            raise ValueError("FP8 feedback output must be 16-byte aligned")
        inputs["positions"] = positions
        storage = scratch_tensor(
            scratch, planned.scratch_specs(), owner="FP8 MTP feedback"
        )
        for mutable in (storage, output):
            if any(_overlaps(mutable, t) for t in inputs.values()):
                raise ValueError(
                    "FP8 feedback scratch/output must not overlap read-only inputs"
                )
        if _overlaps(storage, output):
            raise ValueError("FP8 feedback scratch and output must not overlap")
        views = {
            name: materialize_scratch_view(
                storage, offset_bytes=offset, shape=shape, dtype=dtype
            )[0]
            for name, (offset, shape, dtype) in self.layout.items()
        }
        norm_args = tuple(
            _pointer(t)
            for t in (
                token_embedding,
                multi_state,
                token_norm_weight,
                state_norm_weight,
            )
        )
        norm_args += (
            _pointer(positions, c.Int64 if positions.dtype == torch.int64 else c.Int32),
            _pointer(views["embedding_norm"]),
        )
        quant_args = tuple(
            (
                _pointer(views[f"{path}_norm"]),
                _pointer(views[f"{path}_quant"], c.Uint32),
                _pointer(views[f"{path}_scale"], c.Float32),
            )
            for path in ("embedding", "state")
        )
        projection_args = tuple(
            (
                _pointer(views[f"{path}_quant"], c.Float8E4M3FN),
                _pointer(weight, c.Float8E4M3FN),
                _pointer(views[f"{path}_scale"], c.Float32),
                _pointer(scale, c.Float32),
                _pointer(target),
                _pointer(scale, c.Float32),
            )
            for path, weight, scale, target in (
                (
                    "embedding",
                    embedding_fc_weight,
                    embedding_fc_scale,
                    views["embedding_projection"],
                ),
                ("state", hidden_fc_weight, hidden_fc_scale, output),
            )
        )
        bound = BoundLaunches(
            self.norms[positions.dtype],
            norm_args,
            quant_args,
            projection_args,
            (_pointer(views["embedding_projection"]), _pointer(output)),
        )
        return Binding(
            plan=planned,
            tokens=live,
            scratch=storage,
            token_normalized=views["embedding_norm"][:live],
            state_partial_sums=None,
            state_normalized=views["state_norm"][:live],
            token_path=views["embedding_projection"][:live],
            token_embedding=token_embedding[:live],
            multi_state=multi_state[:live],
            token_norm_weight=token_norm_weight,
            state_norm_weight=state_norm_weight,
            embedding_fc_weight=embedding_fc_weight,
            hidden_fc_weight=hidden_fc_weight,
            embedding_fc_scale=embedding_fc_scale,
            hidden_fc_scale=hidden_fc_scale,
            positions=positions[:live],
            output=output[:live],
            _backend_binding=bound,
        )

    def run(self, binding, *, eps):
        if torch.cuda.is_current_stream_capturing() and not self.warmed:
            raise RuntimeError("FP8 MTP feedback must be warm-run before graph capture")
        bound = binding._backend_binding
        caps = binding.plan.caps
        stream = current_cuda_stream()
        run_compiled(
            bound.norm,
            (*bound.norm_args, c.Float32(eps), c.Int32(binding.tokens), stream),
        )
        for rows, quant_args, projection_args in zip(
            (binding.tokens, binding.tokens * caps.streams),
            bound.quant_args,
            bound.projection_args,
            strict=True,
        ):
            run_compiled(self.quant, (*quant_args, c.Int32(rows), stream))
            run_compiled(
                self.projection,
                (
                    *projection_args,
                    c.Int32(rows),
                    c.Int64(rows * caps.hidden_size),
                    c.Int64(rows * caps.hidden_size),
                    stream,
                ),
            )
        run_compiled(self.add, (*bound.add_args, c.Int32(binding.tokens), stream))
        self.warmed = True
        return binding.output


def plan(caps, resolution):
    from ._impl import Plan

    identity = resolution.device
    if identity is None or not supports_architecture(
        identity.compute_capability, ("sm103a", "sm120a", "sm121a")
    ):
        raise UnsupportedArchitectureError(
            "FP8 MTP feedback requires an implemented Blackwell architecture"
        )
    architecture = (
        f"sm_{identity.compute_capability[0]}{identity.compute_capability[1]}a"
    )
    config = resolution.config
    norm_capacity = align_up(caps.max_tokens, 4)
    norms = {
        dtype: compile_norm(
            caps.hidden_size,
            config.norm_block_h,
            config.norm_num_warps,
            dtype,
            caps.device.index,
            identity.compute_capability,
            caps.streams,
            norm_capacity,
        )
        for dtype in (torch.int32, torch.int64)
    }
    quant, add = compile_aux(
        caps.hidden_size, caps.streams, caps.device.index, architecture
    )
    projection = compile_projection(
        caps.hidden_size,
        caps.hidden_size,
        1,
        "bfloat16",
        True,
        True,
        caps.device.index,
        identity.sm_count,
        architecture,
    )
    layout = {}
    offset = 0
    for name, shape, dtype in (
        ("embedding_norm", (norm_capacity, caps.hidden_size), torch.bfloat16),
        (
            "state_norm",
            (caps.max_tokens, caps.streams, caps.hidden_size),
            torch.bfloat16,
        ),
        ("embedding_quant", (caps.max_tokens, caps.hidden_size), torch.float8_e4m3fn),
        (
            "state_quant",
            (caps.max_tokens * caps.streams, caps.hidden_size),
            torch.float8_e4m3fn,
        ),
        ("embedding_scale", (caps.max_tokens, caps.hidden_size // 128), torch.float32),
        (
            "state_scale",
            (caps.max_tokens * caps.streams, caps.hidden_size // 128),
            torch.float32,
        ),
        ("embedding_projection", (caps.max_tokens, caps.hidden_size), torch.bfloat16),
    ):
        # Padding embedding rows keeps the norm arrays adjacent at the shared
        # scratch materializer's 1024-byte alignment boundary.
        offset = align_up(offset, 1024)
        layout[name] = (offset, shape, dtype)
        elements = 1
        for dimension in shape:
            elements *= dimension
        offset += elements * dtype.itemsize
    spec = scratch_buffer_spec("mtp_feedback", nbytes=offset, device=caps.device)
    return Plan(
        caps=caps,
        token_normalized_offset_bytes=layout["embedding_norm"][0],
        state_partial_sums_offset_bytes=None,
        state_normalized_offset_bytes=layout["state_norm"][0],
        token_path_offset_bytes=layout["embedding_projection"][0],
        _scratch_specs=(spec,),
        token_projection_rows=caps.max_tokens,
        state_projection_rows=caps.max_tokens * caps.streams,
        norm_block_h=config.norm_block_h,
        norm_block_s=config.norm_block_s,
        norm_num_warps=config.norm_num_warps,
        policy_resolution=resolution,
        _backend_plan=BackendPlan(layout, norms, quant, projection, add),
    )
