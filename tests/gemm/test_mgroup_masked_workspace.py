from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from b12x.gemm import mgroup_fp8_gemm as mgg
from b12x.gemm.mgroup_fp8_gemm import _preparation as preparation
from b12x.gemm.mgroup_fp8_gemm._tuning import TUNING
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import _plan_scope


def query():
    return mgg.MGroupFP8GemmQuery(mode="masked", num_groups=3, n=136, k=640,
                                  m_capacity=257, a_sf_gran=128)


def config(implementation):
    return mgg.MGroupFP8GemmConfig(backend="cutedsl", tile_m=128, tile_n=128,
                                   tile_k=64, implementation=implementation)


@pytest.mark.parametrize("implementation", ["single", "masked_compact"])
def test_masked_capacity_memory_is_private_and_counts_resident_bytes(implementation):
    q, cfg = query(), config(implementation)
    sizes = preparation._masked_workspace_sizes(q, cfg)
    assert sizes == (3 * 3 * 5 * 512, 3 * 2 * (2 if implementation == "masked_compact" else 5) * 512)
    plans = [mgg.plan(q, override=cfg) for _ in range(2)]
    assert all(not p.shared for p in plans)
    memory = []
    for p in plans:
        with _plan_scope(p):
            memory.append(p._memory_requirements(cfg, SimpleNamespace(ordinal=0)))
    own = [next(x for x in m.persistent if x.key[0] == "mgroup.masked") for m in memory]
    assert own[0].key != own[1].key
    assert all(x.required_nbytes == sum(sizes) and x.resident_nbytes == 0 for x in own)
    state = SimpleNamespace(sfa_workspace=torch.empty(sizes[0], dtype=torch.uint8),
                            sfb_workspace=torch.empty(sizes[1], dtype=torch.uint8))
    object.__setattr__(plans[0], "_prepared", SimpleNamespace(state=state, closed=False))
    try:
        with _plan_scope(plans[0]):
            resident = plans[0]._memory_requirements(cfg, SimpleNamespace(ordinal=0))
        assert next(x for x in resident.persistent if x.key[0] == "mgroup.masked").resident_nbytes == sum(sizes)
    finally:
        object.__setattr__(plans[0], "_prepared", None)
    assert TUNING.semantic_version == 4
    assert TUNING.candidate_contract_version == 8 and TUNING.config_schema_version == 2


@pytest.mark.parametrize("implementation", ["single", "masked_compact"])
def test_masked_compile_closure_uses_owned_destination(implementation, monkeypatch):
    from b12x._lib import dense_gemm
    from b12x.gemm.mgroup_fp8_gemm import _packing
    q, cfg = query(), config(implementation)
    seen = []
    monkeypatch.setattr(torch.cuda, "device", lambda ordinal: nullcontext())
    monkeypatch.setattr(dense_gemm, "_get_compiled_dense_gemm_masked_mgroup", lambda *a, **k: object())
    def pack(src, dst, **kwargs):
        seen.append((tuple(src.shape), dst.numel(), kwargs))
        return object()
    monkeypatch.setattr(_packing, "pack_grouped_scales_into", pack)
    monkeypatch.setattr(_packing, "pack_grouped_scales_fast", lambda *a, **k: pytest.fail("allocating packer"))
    programs = preparation.compile_mgroup_fp8(TUNING.encode_query(q), TUNING.encode_config(cfg), 0, 188)
    assert set(programs) == {"gemm", "pack_a", "pack_b"}
    assert [x[1] for x in seen] == list(preparation._masked_workspace_sizes(q, cfg))
    assert [x[2]["rows"] for x in seen] == [257, 136]
    assert [x[2]["compact128"] for x in seen] == [False, implementation == "masked_compact"]


@pytest.mark.parametrize("implementation", ["single", "masked_compact"])
def test_masked_owned_workspace_live_stride_graph_and_allocations(implementation, monkeypatch):
    from tests._reference.helpers import require_b12x
    from tests.gemm.test_mgroup_fp8_gemm import _masked_case, _masked_reference, _zero_masked_tail
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x._lib.compile_plan import retained_program_keys
    from b12x.preparation._measurement import no_compilation
    from b12x.gemm.mgroup_fp8_gemm import _packing
    from b12x._lib import compile_plan

    device = require_b12x()
    q, cfg = query(), config(implementation)
    plans = [mgg.plan(q, override=cfg) for _ in range(2)]
    operands = {rows: _masked_case(3, rows, 136, 640, [rows, 0, 17], seed=rows)
                for rows in (129, 257)}
    initial = operands[257]
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(p.request(name=f"masked-{i}", prepare_call=lambda state:
            PreparedCall(run=lambda: state.run_masked(*initial))) for i, p in enumerate(plans)))
        session.freeze()
        state, other = (p.prepared.state for p in plans)
        pointers = state.sfa_workspace.data_ptr(), state.sfb_workspace.data_ptr()
        assert set(pointers).isdisjoint((other.sfa_workspace.data_ptr(), other.sfb_workspace.data_ptr()))
        assert (state.sfa_workspace.numel(), state.sfb_workspace.numel()) == preparation._masked_workspace_sizes(q, cfg)
        payload = plans[0]._prepared
        assert payload.retained is not None and payload.programs <= set(retained_program_keys())
        assert "_pack_ue8m0_mma_kernel" in {p.name for p in payload.programs}
        def forbidden(*args, **kwargs):
            pytest.fail("hot path allocated scale storage")
        monkeypatch.setattr(_packing, "pack_grouped_scales_fast", forbidden)
        seen_rows = []
        real_launch = compile_plan.launch_triton
        def launch(kernel, grid, *args, **kwargs):
            if kernel is _packing._pack_ue8m0_mma_kernel:
                seen_rows.append((args[2], args[4]))
            return real_launch(kernel, grid, *args, **kwargs)
        monkeypatch.setattr(compile_plan, "launch_triton", launch)
        with kernel_resolution_guard("masked owned capacity"), no_compilation():
            for rows in (129, 257, 129):
                lhs, rhs, d, counts = operands[rows]
                def run():
                    return mgg.masked_mm(lhs, rhs, d, counts, plan=plans[0])
                for mutation in ("a", "b", "empty", "restore"):
                    counts.fill_(0 if mutation == "empty" else rows)
                    lhs[1].fill_(2. if mutation == "a" else 1.)
                    rhs[1].fill_(4. if mutation == "b" else 1.)
                    d.fill_(float("nan"))
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    with monkeypatch.context() as guard:
                        guard.setattr(torch, "empty", forbidden)
                        run()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                    expected = _masked_reference(lhs, rhs, counts)
                    torch.testing.assert_close(_zero_masked_tail(d, counts).float(), expected.float(), rtol=.02, atol=.02)
                graph = torch.cuda.CUDAGraph()
                try:
                    with monkeypatch.context() as guard:
                        guard.setattr(torch, "empty", forbidden)
                        with session.capture(), torch.cuda.graph(graph):
                            run()
                    for empty in (False, True, False):
                        counts.fill_(0 if empty else rows)
                        lhs[1].fill_(2.)
                        rhs[1].fill_(1.)
                        d.fill_(float("nan"))
                        before = torch.cuda.memory_stats()["allocation.all.allocated"]
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                        torch.testing.assert_close(_zero_masked_tail(d, counts).float(),
                                                   _masked_reference(lhs, rhs, counts).float(), rtol=.02, atol=.02)
                        assert pointers == (state.sfa_workspace.data_ptr(), state.sfb_workspace.data_ptr())
                finally:
                    graph.reset()
        assert (129, 2) in seen_rows and (257, 3) in seen_rows
        session.release(plans[0])
        assert payload.closed and payload.state is None and payload.retained is None
        assert plans[1].prepared is not None
