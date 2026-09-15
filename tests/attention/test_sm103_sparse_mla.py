"""GLM sparse-MLA contracts through the planned ordinary-MMA backend."""
from b12x._lib.runtime_control import kernel_resolution_guard

import pytest
import torch
import b12x

from b12x.attention import sparse_mla
from b12x.attention.sparse_mla._tuning import SparseMlaConfig
from b12x.preparation import FrozenMapping
from tests.conftest import require_sm103_or_sm12x
from . import test_glm_next_mla as contracts
from . import test_sparse_mla_decode_regimes as nsa_contracts

# Reuse the numerical, cache-address, and graph contracts verbatim. Their
# module globals resolve the same public API patched by this fixture.
for _name, _test in vars(contracts).items():
    if _name.startswith("test_") and callable(_test):
        globals()[_name] = _test
for _name, _test in vars(nsa_contracts).items():
    if _name.startswith("test_") and callable(_test):
        globals()["test_glm_nsa_" + _name.removeprefix("test_")] = _test


@pytest.fixture(autouse=True)
def ordinary_mma_backend(monkeypatch):
    original = sparse_mla.plan
    def plan(caps, *, invocation=FrozenMapping(), override=None):
        splits = min(4, caps.max_chunks_per_row, max(1, (caps.max_width + 63) // 64))
        config = SparseMlaConfig(backend="warp", num_splits=splits if caps.mode == "decode" else 1)
        return original(caps, invocation=invocation, override=config)
    monkeypatch.setattr(sparse_mla, "plan", plan)
    monkeypatch.setattr(contracts, "require_sm120", require_sm103_or_sm12x)



@pytest.mark.parametrize("mode,with_sink", [("decode", False), ("decode", True), ("extend", False)])
@pytest.mark.parametrize("head_major", [False, True])
@torch.inference_mode()
def test_live_rows_reuse_callables_and_replay(mode, with_sink, head_major):
    from b12x.attention.sparse_mla import _sm103

    device = require_sm103_or_sm12x()
    heads, width, capacity, records = 24, 129, 8, 192
    torch.manual_seed(103)
    latent = torch.randn((records, 512), device=device, dtype=torch.bfloat16) / 4
    cache = torch.empty((3, 64, 528), device=device, dtype=torch.uint8)
    writer_plan = sparse_mla.plan_cache_writer(latent, cache, torch.arange(records, device=device, dtype=torch.int64))
    sparse_mla.concat_and_cache_glm_next_mla(
        latent,
        cache,
        torch.arange(records, device=device, dtype=torch.int64), plan=writer_plan)
    q = torch.randn((capacity, heads, 512), device=device, dtype=torch.bfloat16) / 4
    selected = torch.arange(width, device=device, dtype=torch.int32).repeat(capacity, 1)
    lengths = torch.full((capacity,), records, device=device, dtype=torch.int32)
    active = torch.full((capacity,), width, device=device, dtype=torch.int32)
    sink = torch.linspace(3.0, 6.0, heads, device=device) if with_sink else None
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=capacity,
            max_width=width,
            softmax_scale=256**-0.5,
            kv_dtype=torch.uint8,
            head_dim=512,
            v_head_dim=512,
            model_type=sparse_mla.ModelType.GLM_NEXT,
            page_size=64,
            mode=mode,
            head_major_output=head_major,
            return_lse=True,
            has_attention_sink=with_sink,
            lse_scale="natural",
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, device=device, dtype=spec.dtype)

    def bind(rows):
        return sparse_mla.bind(
            plan,
            scratch=scratch,
            q=q[:rows],
            kv_cache=cache,
            selected_indices=selected[:rows],
            cache_lengths=lengths[:rows],
            selected_lengths=active[:rows],
            attention_sink=sink,
        )

    sparse_mla.run(bind(capacity))
    warmed = tuple(id(item) for group in _sm103._CACHE.values() for item in group)
    with kernel_resolution_guard("SM103 sparse MLA live-row reuse"):
        for rows in (1, 3, capacity):
            binding = bind(rows)
            actual, lse = sparse_mla.run(binding)
            expected, expected_lse = contracts.sparse_mla_reference(
                q_all=q[:rows],
                kv_cache=cache.view(records, 1, 528),
                page_table_1=selected[:rows],
                active_token_counts=active[:rows],
                sm_scale=256**-0.5,
                v_head_dim=512,
                return_lse=True,
            )
            if sink is not None:
                natural = expected_lse * torch.log(torch.tensor(2.0))
                total = torch.logaddexp(natural, sink)
                expected = (expected.float() * torch.exp(natural - total)[..., None]).to(expected.dtype)
                expected_lse = total / torch.log(torch.tensor(2.0))
            contracts._assert_glm_next_attention_close(actual, expected)
            torch.testing.assert_close(
                lse, expected_lse * torch.log(torch.tensor(2.0)), atol=0.05, rtol=0
            )
            assert (
                tuple(id(item) for group in _sm103._CACHE.values() for item in group)
                == warmed
            )
        binding = bind(3)
        compiled = torch.compile(
            lambda: sparse_mla.run(binding), backend="eager", fullgraph=True
        )
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, _ = sparse_mla.run(binding)
        active[:3].fill_(65)
        expected, expected_lse = contracts.sparse_mla_reference(
            q_all=q[:3],
            kv_cache=cache.view(records, 1, 528),
            page_table_1=selected[:3],
            active_token_counts=active[:3],
            sm_scale=256**-0.5,
            v_head_dim=512,
            return_lse=True,
        )
        if sink is not None:
            natural = expected_lse * torch.log(torch.tensor(2.0))
            total = torch.logaddexp(natural, sink)
            expected = (expected.float() * torch.exp(natural - total)[..., None]).to(expected.dtype)
        before = contracts._allocator_counters(device)
        graph.replay()
        torch.cuda.synchronize(device)
        assert contracts._allocator_counters(device) == before
        contracts._assert_glm_next_attention_close(output, expected)
        assert sparse_mla.run(bind(0))[0].shape == (0, heads, 512)
