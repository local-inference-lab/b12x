"""Public MXFP4 indexing contracts through the portable warp backend."""
from b12x._lib.runtime_control import kernel_resolution_guard

import pytest
import torch

import b12x

from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer._tuning import DsaIndexerConfig
from b12x.preparation import FrozenMapping
from tests.conftest import require_sm103_or_sm12x
from . import test_dsa_indexer_mxfp4 as contracts

for _name, _test in vars(contracts).items():
    if _name.startswith("test_") and callable(_test):
        globals()[_name] = _test


@pytest.fixture(autouse=True)
def warp_backend(monkeypatch):
    require_sm103_or_sm12x()
    original = dsa_indexer.plan
    def plan(caps, *, invocation=FrozenMapping(), override=None):
        return original(caps, invocation=invocation, override=override if caps.cache_format == "mxfp4" else DsaIndexerConfig(backend="warp"))
    monkeypatch.setattr(dsa_indexer, "plan", plan)



@pytest.mark.parametrize("mode", ["decode", "decode_tiled", "prefill"])
@pytest.mark.parametrize("high_pages", [False, True])
@torch.inference_mode()
def test_fp8_public_plan_reuses_live_rows_and_pool_views(mode, high_pages, monkeypatch):
    if mode == "decode_tiled":
        monkeypatch.setenv("B12X_FUSED_INDEXER", "0")
        mode = "decode"
    from b12x._lib import compiler
    from b12x.attention.dsa_indexer.paged import (
        pack_paged_index_k_cache_reference,
        paged_index_logits_reference,
    )

    device = require_sm103_or_sm12x()
    torch.manual_seed(10351)
    capacity, heads, width, topk = 8, 32, 2048, 512
    keys = torch.randn((width, 128), device=device, dtype=torch.bfloat16)
    packed = pack_paged_index_k_cache_reference(keys)
    base = 2**31 // 8448 + 17 if high_pages else 3
    pool = torch.empty((base + 64, 8448), device=device, dtype=torch.uint8)
    pool[base : base + 32].copy_(packed)
    logical = torch.arange(32, device=device, dtype=torch.int32).repeat(capacity, 1)
    pages = logical + base
    q = torch.randn((capacity, heads, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.rand((capacity, heads), device=device, dtype=torch.float32) / 32
    lengths = torch.full((capacity,), width, device=device, dtype=torch.int32)
    active = torch.tensor([width], device=device, dtype=torch.int32)
    output = torch.empty((capacity, topk), device=device, dtype=torch.int32)
    scores = torch.empty((capacity, topk), device=device, dtype=torch.float32)
    caps = dsa_indexer.Caps(device=device, num_q_heads=heads, max_q_rows=capacity,
                            max_page_table_width=32, topk=topk, mode=mode)
    plan = dsa_indexer.plan(caps, invocation=dsa_indexer.invocation_from_tensors(
        caps, q_fp8=q, query_weights=weights, index_k_cache=pool,
        page_table=pages[:1].expand(capacity, -1) if mode == "prefill" else pages,
        cache_lengths=lengths, active_width=active, output_indices=output, output_scores=scores,
    ))
    indices_plan = dsa_indexer.plan(caps, invocation=dsa_indexer.invocation_from_tensors(
        caps, q_fp8=q, query_weights=weights, index_k_cache=pool,
        page_table=pages[:1].expand(capacity, -1) if mode == "prefill" else pages,
        cache_lengths=lengths, active_width=active, output_indices=output,
        output_scores=None,
    ))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, device=device, dtype=spec.dtype)

    def bind(rows, compact=False, *, indices_only=False):
        return dsa_indexer.bind(
            indices_plan if indices_only else plan,
            scratch=scratch,
            q_fp8=q[:rows],
            query_weights=weights[:rows],
            index_k_cache=pool[: base + 32] if compact else pool,
            page_table=pages[:1].expand(rows, -1)
            if mode == "prefill"
            else pages[:rows],
            cache_lengths=lengths[:rows],
            active_width=active,
            output_indices=output[:rows],
            output_scores=None if indices_only else scores[:rows],
        )

    def check(rows):
        expected = paged_index_logits_reference(
            q_fp8=q[:rows],
            weights=weights[:rows],
            index_k_cache=packed,
            real_page_table=logical[:rows],
            seqlens_per_query=lengths[:rows],
            query_row_to_batch=torch.arange(rows, device=device, dtype=torch.int32),
        )
        wanted = expected.topk(topk, dim=-1)
        assert torch.equal(
            output[:rows].sort(-1).values, wanted.indices.int().sort(-1).values
        )
        torch.testing.assert_close(
            scores[:rows].sort(-1).values,
            wanted.values.sort(-1).values,
            atol=0.01,
            rtol=0,
        )

    dsa_indexer.run(bind(capacity))
    dsa_indexer.run(bind(capacity, indices_only=True))
    identities = {id(v) for v in compiler._MEMORY_CACHE.values()}
    with kernel_resolution_guard("DSA live rows and pool views retain their compiled kernels"):
        for rows in (1, 3, capacity):
            dsa_indexer.run(bind(rows, compact=True))
            check(rows)
            assert {id(v) for v in compiler._MEMORY_CACHE.values()} == identities
        from dataclasses import replace
        with pytest.raises(ValueError, match="output_scores presence"):
            dsa_indexer.run(replace(bind(3), output_scores=None))
        indices_only = bind(3, indices_only=True)
        dsa_indexer.run(indices_only)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)
        dsa_indexer.run(indices_only)
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
            assert before[key] == after[key]
        binding = bind(3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dsa_indexer.run(binding)
        pointers = (output.data_ptr(), scores.data_ptr(), scratch.data_ptr())
        lengths[:3].fill_(1024)
        active.fill_(1024)
        output.fill_(-99)
        scores.fill_(float("nan"))
        before = torch.cuda.memory_stats(device)
        graph.replay()
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        assert pointers == (output.data_ptr(), scores.data_ptr(), scratch.data_ptr())
        for key in ("allocation.all.allocated", "allocation.all.freed"):
            assert before[key] == after[key]
        check(3)


def test_cooperative_barrier_waits_for_every_publishing_warp():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import from_dlpack
    from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
    from b12x._lib.utils import current_cuda_stream
    from b12x.attention.dsa_indexer.fused_indexer import (
        _COOP_STATE_WORDS,
        _fused_group_barrier,
    )

    class DelayedPublish:
        @cute.jit
        def __call__(
            self,
            values: cute.Tensor,
            state: cute.Tensor,
            errors: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(values, state, errors).launch(
                grid=(4, 1, 1),
                block=(1024, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, values: cute.Tensor, state: cute.Tensor, errors: cute.Tensor):
            tid, _, _ = cute.arch.thread_idx()
            cta, _, _ = cute.arch.block_idx()
            if tid >= 992:
                start = cute.arch.clock64()
                while cute.arch.clock64() - start < Int64(cta + 1) * 200_000:
                    pass
                if tid == 1023:
                    values[cta] = Int32(1)
            _fused_group_barrier(state, Int32(0), Int32(0), Int32(4), Int32(tid))
            if tid == 0:
                missing = Int32(0)
                for other in cutlass.range_constexpr(4):
                    missing += Int32(values[other] != Int32(1))
                errors[cta] = missing

    device = require_sm103_or_sm12x()
    values = torch.zeros(4, device=device, dtype=torch.int32)
    state = torch.zeros(_COOP_STATE_WORDS, device=device, dtype=torch.int32)
    errors = torch.empty(4, device=device, dtype=torch.int32)
    tensors = tuple(from_dlpack(t, assumed_align=4) for t in (values, state, errors))
    kernel = compile_kernel(
        DelayedPublish(),
        *tensors,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "test.dsa.cooperative_barrier", 1, (4, 1024)
        ),
    )
    for _ in range(8):
        values.zero_()
        state.zero_()
        errors.fill_(-1)
        kernel(*tensors, current_cuda_stream())
        assert torch.equal(errors, torch.zeros_like(errors))
