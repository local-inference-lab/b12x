from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from b12x.gemm import mla_query_projection
from b12x.gemm.mla_query_projection._preparation import _ProjectionExecutionState
from b12x.gemm.mla_query_projection._tuning import TUNING, ProjectionQuery
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation._measurement import no_compilation
from tests._reference.helpers import require_b12x
from tests.gemm.test_bmm import _make_pack, _rhs_views


def _inputs(*, heads, m, weight_format, seed=31):
    torch.manual_seed(seed)
    q_nope = torch.randn(heads, m, 192, device="cuda", dtype=torch.bfloat16)
    q_full = torch.randn(m, heads, 576, device="cuda", dtype=torch.bfloat16)
    q_pe = q_full[..., 512:]
    q_scale = torch.tensor([.037], device="cuda", dtype=torch.float32)
    if weight_format == "mxfp8":
        values, scales = _make_pack(seed=seed, batch=heads)
        weight = _rhs_views(values, scales, batch=heads)["n"]
    else:
        weight = torch.randn(heads, 192, 512, device="cuda", dtype=torch.bfloat16) * .05
    return q_nope, weight, q_pe, q_scale


def _query(*, heads, max_rows, weight_format, output_dtype="bfloat16"):
    return ProjectionQuery(heads=heads, max_rows=max_rows, weight_format=weight_format,
                           output_dtype=output_dtype, b_major="n", sf_axis="n")


def _warm_call(weight, *, heads, output_dtype):
    def prepare(state):
        m = state.query.max_rows
        q_nope = torch.zeros(heads, m, 192, device="cuda", dtype=torch.bfloat16)
        q_pe = torch.zeros(m, heads, 64, device="cuda", dtype=torch.bfloat16)
        q_scale = torch.ones(1, device="cuda", dtype=torch.float32) if output_dtype == torch.float8_e4m3fn else None
        out = torch.empty(m, heads, 576, device="cuda", dtype=output_dtype)
        return PreparedCall(run=lambda: state.run(q_nope, weight, q_pe, out, q_scale=q_scale))
    return prepare


@contextmanager
def _prepared_plans(weight, rows, *, heads, output_dtype):
    weight_format = "bf16" if isinstance(weight, torch.Tensor) else "mxfp8"
    plans = {m: mla_query_projection.plan(_query(heads=heads, max_rows=m, weight_format=weight_format,
                                                 output_dtype=str(output_dtype).removeprefix("torch.")))
             for m in rows}
    session = PreparationSession(autotune=False)
    result = session.prepare(tuple(plan.request(name=f"mla.m{m}", prepare_call=_warm_call(weight, heads=heads, output_dtype=output_dtype))
                                   for m, plan in plans.items()))
    try: yield session, plans
    finally: result.close(); session.close()


@contextmanager
def _prepared(weight, *, heads, m, output_dtype):
    with _prepared_plans(weight, (m,), heads=heads, output_dtype=output_dtype) as (_, plans):
        yield plans[m]


def _reference(q_nope, weight, q_pe):
    if isinstance(weight, torch.Tensor): projected = torch.bmm(q_nope, weight)
    else:
        values, scales = weight
        physical = values.to(torch.bfloat16) * scales.view(torch.float8_e8m0fnu).to(torch.bfloat16).repeat_interleave(32, dim=-1)
        projected = torch.bmm(q_nope, physical)
    return torch.cat((projected.transpose(0, 1), q_pe), dim=-1)


@pytest.mark.parametrize("weight_format,heads", [("mxfp8", 8), ("mxfp8", 16), ("bf16", 11)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_projection_preserves_weight_forms_and_output_modes(weight_format, heads, output_dtype):
    require_b12x(); m = 4
    q_nope, weight, q_pe, q_scale = _inputs(heads=heads, m=m, weight_format=weight_format)
    out = torch.empty(m, heads, 576, device="cuda", dtype=output_dtype)
    with _prepared(weight, heads=heads, m=m, output_dtype=output_dtype) as plan:
        assert mla_query_projection.run(q_nope, weight, q_pe, out, plan=plan,
                                        q_scale=q_scale if output_dtype == torch.float8_e4m3fn else None) is out
    expected = _reference(q_nope, weight, q_pe)
    if output_dtype == torch.bfloat16: torch.testing.assert_close(out, expected, rtol=.03, atol=.03)
    else: assert torch.equal(out.view(torch.uint8), (expected.float() / q_scale).clamp(-448, 448).to(output_dtype).view(torch.uint8))


@pytest.mark.parametrize("weight_format,heads", [("mxfp8", 8), ("bf16", 11)])
def test_prepared_projection_graph_replays_changed_inputs(weight_format, heads):
    require_b12x(); m = 4
    q_nope, weight, q_pe, q_scale = _inputs(heads=heads, m=m, weight_format=weight_format)
    out = torch.empty(m, heads, 576, device="cuda", dtype=torch.bfloat16)
    with _prepared(weight, heads=heads, m=m, output_dtype=torch.bfloat16) as plan:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): mla_query_projection.run(q_nope, weight, q_pe, out, plan=plan)
        fresh_nope, fresh_pe = torch.randn_like(q_nope), torch.randn_like(q_pe)
        q_nope.copy_(fresh_nope); q_pe.copy_(fresh_pe); graph.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(out, _reference(fresh_nope, weight, fresh_pe), rtol=.03, atol=.03)


def test_projection_rejects_unprepared_execution():
    require_b12x()
    q_nope, weight, q_pe, _ = _inputs(heads=8, m=1, weight_format="mxfp8")
    out = torch.empty(1, 8, 576, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        mla_query_projection.run(q_nope, weight, q_pe, out)


def test_query_row_contract_is_bf16_capacity_and_mxfp8_exact_m():
    """BF16 ``max_rows`` is a capacity; MXFP8 compiles M. Neither changes the tuning identity."""
    assert mla_query_projection.ProjectionQuery is ProjectionQuery
    bf16 = _query(heads=11, max_rows=16, weight_format="bf16")
    mxfp8 = _query(heads=8, max_rows=16, weight_format="mxfp8")
    assert bf16.served_rows == range(1, 17)
    assert mxfp8.served_rows == range(16, 17)
    assert 1 in bf16.served_rows and 16 in bf16.served_rows and 17 not in bf16.served_rows
    assert 15 not in mxfp8.served_rows and 17 not in mxfp8.served_rows
    assert set(TUNING.encode_query(bf16)) == TUNING.query_fields
    assert "served_rows" not in TUNING.query_fields


@pytest.mark.parametrize("m,admitted", [(1, True), (3, True), (16, True), (17, False)])
def test_bf16_execution_admits_rows_up_to_capacity(monkeypatch, m, admitted):
    from b12x.gemm.mla_query_projection import _bf16
    calls = []
    monkeypatch.setattr(_bf16, "_run_prepared", lambda *args, **kwargs: calls.append(kwargs["max_rows"]))
    device = torch.device("meta")
    state = _ProjectionExecutionState(_query(heads=11, max_rows=16, weight_format="bf16"), device, launch=object())
    q_nope = torch.empty(11, m, 192, device=device, dtype=torch.bfloat16)
    weight = torch.empty(11, 192, 512, device=device, dtype=torch.bfloat16)
    q_pe = torch.empty(m, 11, 64, device=device, dtype=torch.bfloat16)
    out = torch.empty(m, 11, 576, device=device, dtype=torch.bfloat16)
    if admitted:
        state.run(q_nope, weight, q_pe, out)
        assert calls == [16]
    else:
        with pytest.raises(ValueError, match=r"BF16 MLA query plan serves 1<=M<=16, got M=17"):
            state.run(q_nope, weight, q_pe, out)
        assert calls == []


@pytest.mark.parametrize("m,admitted", [(3, False), (15, False), (16, True), (17, False)])
def test_mxfp8_execution_still_requires_exact_m(monkeypatch, m, admitted):
    from b12x.gemm._shared import mxfp8_bmm
    calls = []
    monkeypatch.setattr(mxfp8_bmm, "_rhs_tensors", lambda weight: weight)
    monkeypatch.setattr(mxfp8_bmm, "_run_mla_query_prepared", lambda *args, **kwargs: calls.append(args[0].shape[1]))
    device = torch.device("meta")
    state = _ProjectionExecutionState(_query(heads=8, max_rows=16, weight_format="mxfp8"), device, launch=object())
    q_nope = torch.empty(8, m, 192, device=device, dtype=torch.bfloat16)
    weight = (torch.empty(8, 192, 512, device=device, dtype=torch.float8_e4m3fn),
              torch.empty(8, 192, 16, device=device, dtype=torch.uint8))
    q_pe = torch.empty(m, 8, 64, device=device, dtype=torch.bfloat16)
    out = torch.empty(m, 8, 576, device=device, dtype=torch.bfloat16)
    if admitted:
        state.run(q_nope, weight, q_pe, out)
        assert calls == [16]
    else:
        with pytest.raises(ValueError, match=rf"MXFP8 MLA query plan is compiled for exact M=16, got M={m}"):
            state.run(q_nope, weight, q_pe, out)
        assert calls == []


@pytest.mark.parametrize("capacity,rows", [(16, (1, 3, 7, 15, 16)), (32, (1, 3, 7, 15, 16, 17, 31, 32))])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_bf16_capacity_plan_matches_exact_m_plans(capacity, rows, output_dtype):
    """One BF16 plan serves every M up to its capacity, bit-identical to exact-M plans and graph-replayable."""
    require_b12x(); heads = 11
    fp8 = output_dtype == torch.float8_e4m3fn
    q_nope_cap, weight, q_pe_cap, q_scale = _inputs(heads=heads, m=capacity, weight_format="bf16")
    q_scale = q_scale if fp8 else None
    out_cap = torch.empty(capacity, heads, 576, device="cuda", dtype=output_dtype)
    sentinel = torch.full_like(out_cap.view(torch.uint8), 0x5A)

    def expected(q_nope, q_pe):
        reference = _reference(q_nope, weight, q_pe)
        return (reference.float() / q_scale).clamp(-448, 448).to(output_dtype) if fp8 else reference

    with _prepared_plans(weight, sorted({capacity, *rows}), heads=heads, output_dtype=output_dtype) as (session, plans):
        session.freeze()
        capacity_plan = plans[capacity]
        for m in rows:
            q_nope, q_pe, out = q_nope_cap[:, :m], q_pe_cap[:m], out_cap[:m]
            out_cap.view(torch.uint8).copy_(sentinel)
            exact_out = torch.empty(m, heads, 576, device="cuda", dtype=output_dtype)
            with no_compilation():
                mla_query_projection.run(q_nope, weight, q_pe, out, plan=capacity_plan, q_scale=q_scale)
                mla_query_projection.run(q_nope.contiguous(), weight, q_pe.contiguous(), exact_out,
                                         plan=plans[m], q_scale=q_scale)
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint8), exact_out.view(torch.uint8)), m
            assert torch.equal(out_cap[m:].view(torch.uint8), sentinel[m:]), m
            if fp8: assert torch.equal(out.view(torch.uint8), expected(q_nope, q_pe).view(torch.uint8)), m
            else: torch.testing.assert_close(out, expected(q_nope, q_pe), rtol=.03, atol=.03)
        with pytest.raises(ValueError, match=rf"serves 1<=M<={capacity}, got M={capacity + 1}"):
            wide = torch.zeros(heads, capacity + 1, 192, device="cuda", dtype=torch.bfloat16)
            mla_query_projection.run(wide, weight, torch.zeros(capacity + 1, heads, 64, device="cuda", dtype=torch.bfloat16),
                                     torch.empty(capacity + 1, heads, 576, device="cuda", dtype=output_dtype),
                                     plan=capacity_plan, q_scale=q_scale)

        m = rows[len(rows) // 2]
        q_nope, q_pe, out = q_nope_cap[:, :m], q_pe_cap[:m], out_cap[:m]
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            mla_query_projection.run(q_nope, weight, q_pe, out, plan=capacity_plan, q_scale=q_scale)
        fresh_nope, fresh_pe = torch.randn_like(q_nope), torch.randn_like(q_pe)
        q_nope.copy_(fresh_nope); q_pe.copy_(fresh_pe); out.view(torch.uint8).zero_()
        graph.replay(); torch.cuda.synchronize()
    if fp8: assert torch.equal(out.view(torch.uint8), expected(fresh_nope, fresh_pe).view(torch.uint8))
    else: torch.testing.assert_close(out, expected(fresh_nope, fresh_pe), rtol=.03, atol=.03)
