"""Portable GPU qualification of complete Trellis transform boundaries."""

from unittest.mock import patch

import pytest
import torch

from tests._reference import trellis_transforms as reference


def _require_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("physical Blackwell GPU required")


@pytest.mark.parametrize(
    "coupled,kind", [(False, "silu"), (False, "situ"), (True, "situ")]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("broadcast", [False, True])
def test_transform_oracles_live_counts_and_graph(coupled, kind, dtype, broadcast):
    _require_gpu()
    import cuda.bindings.driver as cuda
    import cutlass as c
    import cutlass.cute as cute
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis_transforms import (
        InputRotation,
        IntermediateRotation,
        OutputRotation,
    )

    torch.manual_seed(826)
    tokens, top_k, experts = 8, 3, 3
    routes, hidden, width = tokens * top_k, 1024, 256
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.randn(tokens, hidden, device=device, dtype=dtype) * 0.1
    ids = torch.arange(routes, device=device, dtype=torch.int64) % experts
    ids[1], ids[3] = -1, -1

    def scales(shape):
        return (0.875 + torch.rand(shape, device=device) * 0.25).half()

    suh, svh = (
        scales((1 if broadcast else experts, hidden)),
        scales((1 if broadcast else experts, hidden)),
    )
    rotations = scales((experts, width * (6 if coupled else 3)))
    if coupled:
        rotations[:, 3 * width :] = (
            torch.randint(0, 2, (experts, 3 * width), device=device).mul_(2).sub_(1)
        )
    gate, up = (
        torch.randn(routes, width, device=device, dtype=torch.float16) * 0.1
        for _ in range(2)
    )
    down = torch.randn(routes, hidden, device=device, dtype=torch.float16) * 0.1
    weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    a = torch.empty(routes, hidden, device=device, dtype=torch.float16)
    h = torch.empty_like(gate)
    out = torch.empty(tokens, hidden, device=device, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    source_type = c.Float16 if dtype == torch.float16 else c.BFloat16

    def ptrs(items):
        return tuple(pointer(t, v) for t, v in items)

    input_args = ptrs(
        ((source_type, source), (c.Int64, ids), (c.Float16, suh), (c.Float16, a))
    ) + (c.Int64(0 if broadcast else hidden),)
    middle_args = ptrs(
        (
            (c.Float16, gate),
            (c.Float16, up),
            (c.Int64, ids),
            (c.Float16, rotations),
            (c.Float16, h),
        )
    )
    output_args = ptrs(
        (
            (c.Float16, down),
            (c.Int64, ids),
            (c.Float16, svh),
            (c.Float32, weights),
            (c.Float32, out),
        )
    ) + (c.Int64(0 if broadcast else hidden),)
    fi = cute.compile(
        InputRotation(hidden, experts, top_k, routes, coupled=coupled),
        *input_args,
        c.Int32(routes),
        stream,
        options=f"--gpu-arch={target}",
    )
    fm = cute.compile(
        IntermediateRotation(width, experts, routes, coupled=coupled, activation=kind),
        *middle_args,
        c.Int32(routes),
        stream,
        options=f"--gpu-arch={target}",
    )
    fo = cute.compile(
        OutputRotation(hidden, experts, top_k, tokens, coupled=coupled),
        *output_args,
        c.Int32(tokens),
        stream,
        options=f"--gpu-arch={target}",
    )

    def expected():
        return (
            reference.input_rotation(source, ids, suh, top_k, coupled),
            reference.intermediate_rotation(gate, up, ids, rotations, coupled, kind),
            reference.output_rotation(down, ids, svh, weights, coupled),
        )

    expected_a, expected_h, expected_out = expected()
    with patch.object(
        cute, "compile", side_effect=AssertionError("kernel resolution is frozen")
    ):
        for live in (8, 1, 4, 3):
            a.fill_(float("nan"))
            h.fill_(float("nan"))
            out.fill_(float("nan"))
            fi(*input_args, c.Int32(live * top_k), stream)
            fm(*middle_args, c.Int32(live * top_k), stream)
            fo(*output_args, c.Int32(live), stream)
            torch.testing.assert_close(
                a[: live * top_k], expected_a[: live * top_k], atol=0.0005, rtol=0.002
            )
            torch.testing.assert_close(
                h[: live * top_k], expected_h[: live * top_k], atol=0.0001, rtol=0.003
            )
            torch.testing.assert_close(
                out[:live], expected_out[:live], atol=0.000001, rtol=0.0001
            )
            assert torch.isnan(a[live * top_k :]).all()
            assert torch.isnan(h[live * top_k :]).all()
            assert torch.isnan(out[live:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            current = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            fi(*input_args, c.Int32(routes), current)
            fm(*middle_args, c.Int32(routes), current)
            fo(*output_args, c.Int32(tokens), current)
        source.mul_(0.5)
        gate.neg_()
        up.mul_(0.75)
        down.neg_()
        ids.copy_(torch.arange(routes, device=device) % experts)
        changed_a, changed_h, changed_out = expected()
        assert not torch.equal(changed_out, expected_out)
        addresses = tuple(t.data_ptr() for t in (source, ids, rotations, a, h, out))
        allocated = torch.cuda.memory_allocated()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert addresses == tuple(
            t.data_ptr() for t in (source, ids, rotations, a, h, out)
        )
        torch.testing.assert_close(a, changed_a, atol=0.0005, rtol=0.002)
        torch.testing.assert_close(h, changed_h, atol=0.0001, rtol=0.003)
        torch.testing.assert_close(out, changed_out, atol=0.000001, rtol=0.0001)
        assert torch.isfinite(out).all() and torch.count_nonzero(out)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "mapped,output_mapped", [(False, False), (True, False), (False, True), (True, True)]
)
def test_route_mapping_preserves_large_ids_and_masks(id_dtype, mapped, output_mapped):
    _require_gpu()
    import cuda.bindings.driver as cuda
    import cutlass as c
    import cutlass.cute as cute
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis_transforms import MapRoutes

    ids = torch.tensor(
        [-1, 0, 1, 2, 3, 4, 5, 2**32 + 1 if id_dtype == torch.int64 else 2**31 - 1],
        device="cuda",
        dtype=id_dtype,
    )
    routes = torch.tensor([2, -1, 1, 0, 2], device="cuda", dtype=torch.int32)
    outputs = torch.tensor([-1, 2, 0, 1, 3], device="cuda", dtype=torch.int32)
    local, out = (
        torch.empty_like(ids, dtype=torch.int64),
        torch.empty_like(ids, dtype=torch.int64),
    )
    params = [
        pointer(t, v)
        for t, v in (
            (c.Int64 if id_dtype == torch.int64 else c.Int32, ids),
            (c.Int32, routes),
            (c.Int32, outputs),
            (c.Int64, local),
            (c.Int64, out),
        )
    ]
    params += [
        c.Int32(ids.numel()),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    fn = cute.compile(
        MapRoutes(ids.numel(), 3, 5, mapped=mapped, output_mapped=output_mapped),
        *params,
        options=f"--gpu-arch={target}",
    )
    fn(*params)
    expected_local, expected_out = [], []
    for value in ids.cpu().tolist():
        a = int(routes[value]) if mapped and 0 <= value < 5 else value
        b = int(outputs[value]) if output_mapped and 0 <= value < 5 else a
        a = a if 0 <= value < 5 and 0 <= a < 3 else -1
        b = b if a >= 0 and 0 <= b < 3 else -1
        expected_local.append(a)
        expected_out.append(b)
    assert local.cpu().tolist() == expected_local
    assert out.cpu().tolist() == expected_out


def test_large_expert_scale_offsets():
    _require_gpu()
    import cuda.bindings.driver as cuda
    import cutlass as c
    import cutlass.cute as cute
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis_transforms import (
        InputRotation,
        IntermediateRotation,
        OutputRotation,
    )

    hidden, width = 512, 128
    experts = 2**31 // (3 * width) + 9
    required = experts * (hidden + 3 * width) * 2
    if torch.cuda.mem_get_info()[0] < required + 2**30:
        pytest.skip("large-offset Trellis regression requires 11 GiB free GPU memory")
    scales = torch.empty((experts, hidden), dtype=torch.float16, device="cuda")
    rotations = torch.empty((experts, 3 * width), dtype=torch.float16, device="cuda")
    scales[-1].fill_(1)
    rotations[-1].fill_(1)
    ids = torch.tensor([experts - 1], dtype=torch.int64, device="cuda")
    assert (experts - 1) * hidden > 2**31 and (experts - 1) * 3 * width > 2**31
    source = torch.randn(1, hidden, dtype=torch.float16, device="cuda") * 0.01
    gate = torch.randn(1, width, dtype=torch.float16, device="cuda") * 0.01
    up = torch.randn_like(gate)
    weights = torch.ones(1, 1, device="cuda")
    result_a, result_middle = torch.empty_like(source), torch.empty_like(gate)
    result = torch.empty(1, hidden, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target

    def launch(kernel, tensors, scalars):
        args = (
            [pointer(dtype, tensor) for dtype, tensor in tensors] + scalars + [stream]
        )
        fn = cute.compile(kernel, *args, options=f"--gpu-arch={target}")
        fn(*args)

    launch(
        InputRotation(hidden, experts, 1, 1, coupled=False),
        [
            (c.Float16, source),
            (c.Int64, ids),
            (c.Float16, scales),
            (c.Float16, result_a),
        ],
        [c.Int64(hidden), c.Int32(1)],
    )
    launch(
        IntermediateRotation(width, experts, 1, coupled=False, activation="silu"),
        [
            (c.Float16, gate),
            (c.Float16, up),
            (c.Int64, ids),
            (c.Float16, rotations),
            (c.Float16, result_middle),
        ],
        [c.Int32(1)],
    )
    launch(
        OutputRotation(hidden, experts, 1, 1, coupled=False),
        [
            (c.Float16, source),
            (c.Int64, ids),
            (c.Float16, scales),
            (c.Float32, weights),
            (c.Float32, result),
        ],
        [c.Int64(hidden), c.Int32(1)],
    )
    torch.testing.assert_close(
        result_a,
        reference.input_rotation(source, ids, scales, 1, False),
        atol=0.0001,
        rtol=0.002,
    )
    torch.testing.assert_close(
        result_middle,
        reference.intermediate_rotation(gate, up, ids, rotations, False, "silu"),
        atol=0.0001,
        rtol=0.003,
    )
    torch.testing.assert_close(
        result,
        reference.output_rotation(source, ids, scales, weights, False),
        atol=0.00001,
        rtol=0.001,
    )


def test_prepared_expert_map_composition_live_counts_and_graph():
    _require_gpu()
    import cuda.bindings.driver as cuda
    import cutlass as c
    import cutlass.cute as cute
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis_transforms import ComposeExpertMaps

    experts, capacity = 384, 259
    mapping = (
        torch.arange(experts, dtype=torch.int32, device="cuda").flip(0).contiguous()
    )
    mapping[1], mapping[3] = -1, experts
    original = torch.arange(capacity, dtype=torch.int64, device="cuda") % experts
    original[0], original[6], original[8] = experts - 1, -1, 2**32 + 1
    original_output = original.flip(0).contiguous()
    ids, out_ids = original.clone(), original_output.clone()
    args = (pointer(c.Int64, ids), pointer(c.Int64, out_ids), pointer(c.Int32, mapping))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(
        ComposeExpertMaps(capacity, experts),
        *args,
        c.Int32(1),
        stream,
        options=f"--gpu-arch={architecture_for(torch.cuda.get_device_capability()).compilation_target}",
    )

    def expected(values):
        result = mapping[values.clamp(0, experts - 1)].long()
        return result.masked_fill(
            (values < 0) | (values >= experts) | (result < 0) | (result >= experts), -1
        )

    with patch.object(
        cute, "compile", side_effect=AssertionError("resolution is frozen")
    ):
        for live in (capacity, 1, 128, 129):
            ids.copy_(original)
            out_ids.copy_(original_output)
            fn(*args, c.Int32(live), stream)
            ei = expected(original[:live])
            eo = expected(original_output[:live]).masked_fill(ei < 0, -1)
            torch.testing.assert_close(ids[:live], ei)
            torch.testing.assert_close(out_ids[:live], eo)
            torch.testing.assert_close(ids[live:], original[live:])
            torch.testing.assert_close(out_ids[live:], original_output[live:])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(
                *args,
                c.Int32(capacity),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        mapping[4] = 0
        ids.copy_(original)
        out_ids.copy_(original_output)
        ei = expected(original)
        eo = expected(original_output).masked_fill(ei < 0, -1)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(ids, ei)
        torch.testing.assert_close(out_ids, eo)
