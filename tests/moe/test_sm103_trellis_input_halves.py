"""Intermediate-Hadamard checkpoint extents with two input transforms in one FC1 tile."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import pytest
import torch

from b12x._lib.architecture import architecture_for
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.sm103.launch import pointer
from b12x.moe._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm
from b12x.moe.fused_moe._sm103_trellis import _mixed_contract
from tests._reference.trellis_reference import moe_reference
from tests.moe.test_sm103_trellis_moe import prepare_experts
from tests.moe.test_sm103_trellis_mixed_moe import prepare_mixed


class InspectColumnSelection:
    """Exercise the production epilogue predicate with distinguishable MMA rows."""

    def __init__(self):
        self.projection = RoutedTrellisGemm(
            384, 512, 3, 8, bits=3, codebook="lut_e4m3", dual_input=True
        )

    @cute.jit
    def __call__(
        self, values, output, split: c.Int64, origin: c.Int64,
        live: c.Int32, stream: cuda.CUstream
    ):
        self.kernel(values, output, split, origin).launch(
            grid=(live * 3, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, values, output, split: c.Int64, origin: c.Int64):
        lane, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        route, col = c.Int64(block // 3), c.Int64(block % 3) * 128 + c.Int64(lane)
        source = cute.make_tensor(values, cute.make_layout(8 * 2 * 384))
        target = cute.make_tensor(output, cute.make_layout(8 * 384))
        cutoff = self.projection.output_split(
            (None, None, split), origin + c.Int64(block % 3) * 128
        )
        for row in c.range_constexpr(2):
            if self.projection.output_row(cutoff, row, c.Int32(lane)):
                target[route * 384 + col] = source[(route * 2 + row) * 384 + col]


def test_intermediate_hadamard_column_selection_changes_inside_tiles_and_replays():
    if not torch.cuda.is_available():
        pytest.skip("portable CUDA epilogue probe requires a GPU")
    torch.manual_seed(212)
    values = torch.randn(8, 2, 384, device="cuda", dtype=torch.float16)
    output = torch.empty(8, 384, device="cuda", dtype=torch.float16)
    args = (pointer(c.Float16, values), pointer(c.Float16, output))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    fn = cute.compile(
        InspectColumnSelection(),
        *args,
        c.Int64(192),
        c.Int64(0),
        c.Int32(1),
        stream,
        options=f"--gpu-arch={target}",
    )

    def expected(split, live):
        return torch.cat((values[:live, 0, :split], values[:live, 1, split:]), dim=1)

    with patch.object(cute, "compile", side_effect=AssertionError("resolution frozen")):
        for origin in (0, 2**31 + 128, 2**40):
            for split in (64, 128, 192, 256, 320):
                for live in (8, 1, 4, 3):
                    output.fill_(float("nan"))
                    fn(*args, c.Int64(origin + split), c.Int64(origin), c.Int32(live), stream)
                    torch.testing.assert_close(
                        output[:live], expected(split, live), atol=0, rtol=0
                    )
                    assert torch.isnan(output[live:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(
                *args,
                c.Int64(192),
                c.Int64(0),
                c.Int32(8),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        values[:, 1].neg_()
        changed = expected(192, 8)
        addresses = values.data_ptr(), output.data_ptr()
        allocated = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocated_bytes.all.allocated"] == allocated
        assert addresses == (values.data_ptr(), output.data_ptr())
        torch.testing.assert_close(output, changed, atol=0, rtol=0)


@pytest.mark.parametrize("mixed,bits", [(False, 2), (False, 3), (False, 4), (True, 3)])
@pytest.mark.parametrize("width,offset", [(384, 0), (256, 128)])
def test_canonical_cross_half_preparation_and_oracle(mixed, bits, width, offset):
    if not torch.cuda.is_available():
        pytest.skip("canonical preparation requires CUDA")
    torch.manual_seed(77)
    kwargs = dict(
        intermediate_hadamard=True,
        dtype=torch.bfloat16,
        sign_pattern=5,
        global_intermediate_size=384,
        intermediate_offset=offset,
        distinct_input_scales=True,
    )
    if mixed:
        owner, _, _ = prepare_mixed(
            **kwargs,
            experts=3,
            hidden=512,
            width=width,
            activation="situ",
            per_expert_scales=True,
        )
    else:
        owner = prepare_experts(
            **kwargs, bits=bits, device="cuda", geometry=(3, 512, width)
        )
    payload = owner._impl.representation_for("w4a16")
    state = (
        _mixed_contract(
            SimpleNamespace(weight_E=3, k=512, n=width, device=payload.w13.device),
            payload,
        )[0]
        if mixed
        else payload.trellis
    )
    assert state.input_scale_split == 192 - offset
    assert not torch.equal(state.gate_suh, state.up_suh)
    assert state.gate_suh.data_ptr() != state.up_suh.data_ptr()
    source = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16) * 0.01
    ids = torch.tensor([[0, 1], [2, -1], [1, 0], [0, 2]], device="cuda")
    routing = torch.rand(4, 2, device="cuda")
    oracle = moe_reference(source, payload, ids, routing, activation_kind="situ")
    assert torch.isfinite(oracle).all() and torch.count_nonzero(oracle)
    # A single-transform execution must disagree for these unequal scales.
    if mixed:
        wrong = replace(
            payload,
            input_scale_split=None,
            rotations=replace(payload.rotations, up_suh=payload.rotations.gate_suh),
        )
    else:
        wrong = replace(
            payload,
            trellis=replace(state, input_scale_split=None, up_suh=state.gate_suh),
        )
    incorrect = moe_reference(source, wrong, ids, routing, activation_kind="situ")
    assert (
        torch.linalg.vector_norm(oracle - incorrect) / torch.linalg.vector_norm(oracle)
        > 0.01
    )
    if torch.cuda.get_device_capability() in ((12, 0), (12, 1)):
        # Declarations are device-free. SM12x never executes a split extent:
        # binding rejects the split, and planning already rejects geometries
        # and activations its kernels do not implement.
        from b12x.moe.fused_moe import _impl as impl

        rejected = (
            "distinct input-scale halves|integral number of CTA N tiles|requires silu"
        )
        with pytest.raises((NotImplementedError, ValueError), match=rejected):
            plan = impl.plan_tp_moe_scratch(impl.TPMoEScratchCaps(
                max_tokens=4, core_token_counts=(4,), num_topk=2, route_num_experts=3,
                device=source.device, weight_plan=owner._impl.plan, quant_mode="w4a16",
                w4a16_block_size_m=64,
                decode_config=impl.MoeDecodeConfig(
                    backend="w4a16", route_planner="internal", max_active_clusters=None,
                    w4a16_route_mode="packed",
                ),
            ))
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=source.device)
                for spec in plan.scratch_specs()
            )
            plan.bind(
                scratch=scratch, a=source, experts=owner._impl,
                topk_weights=routing, topk_ids=ids.clamp(min=0).to(torch.int32),
                output=torch.empty_like(source),
            )
