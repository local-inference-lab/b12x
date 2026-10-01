"""Tests for gemm.mgroup_fp8_gemm (masked + contiguous grouped FP8 GEMM).

The first section is CPU-only (query validation, plan construction, capacity
checks). The GPU section is gated on require_b12x() and covers both modes
against a dequantized torch reference, compile-key hygiene across live
masked_m sets, CUDA-graph capture/replay fidelity, and a >2**31-byte B stack
for the grouped addressing path.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from b12x.gemm import mgroup_fp8_gemm as mgg
from b12x.gemm.mgroup_fp8_gemm._preparation import _MGroupFP8ExecutionState
from tests._reference.helpers import require_b12x


def _query(**overrides) -> mgg.MGroupFP8GemmQuery:
    fields = dict(
        mode="masked", num_groups=4, n=128, k=256, m_capacity=128, a_sf_gran=128,
    )
    fields.update(overrides)
    return mgg.MGroupFP8GemmQuery(**fields)


def test_query_defaults_and_validation() -> None:
    query = _query()
    assert query.mode == "masked"
    assert query.b_sf_gran == 128
    assert query.c_dtype == "bfloat16"
    from b12x.gemm.mgroup_fp8_gemm._tuning import TUNING

    TUNING.validate_query(query, None)
    assert TUNING.component_id == "gemm.mgroup_fp8_gemm"


@pytest.mark.parametrize("mode,gran", [("masked", 32), ("contiguous", 128)])
def test_query_rejects_mode_mismatched_a_granularity(mode: str, gran: int) -> None:
    from b12x.gemm.mgroup_fp8_gemm._tuning import TUNING

    with pytest.raises(ValueError, match="gran"):
        TUNING.validate_query(_query(mode=mode, a_sf_gran=gran), None)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(mode="batched"),
        dict(num_groups=0),
        dict(n=-8),
        dict(k=0),
        dict(m_capacity=-1),
        dict(k=96),  # K must be divisible by 128
        dict(b_sf_gran=32),
        dict(c_dtype="float16"),
        dict(expected_m=0),
        dict(expected_m=129),  # regime hint must stay within capacity
    ],
)
def test_query_rejects_bad_geometry(overrides) -> None:
    from b12x.gemm.mgroup_fp8_gemm._tuning import TUNING

    with pytest.raises((TypeError, ValueError)):
        TUNING.validate_query(_query(**overrides), None)


def test_plan_constructs_and_config_round_trips() -> None:
    from b12x.gemm.mgroup_fp8_gemm._tuning import TUNING

    query = _query()
    declaration = mgg.plan(query)
    assert declaration.component_id == "gemm.mgroup_fp8_gemm"
    assert declaration.query is query
    config = declaration.contract.default_config(query, None)
    TUNING.validate_config(query, config, None)
    assert config == TUNING.decode_config(TUNING.config_payload(config))
    encoded = TUNING.encode_query(query)
    assert set(encoded) == TUNING.query_fields
    assert encoded["m_capacity"] == 128


def test_plan_rejects_foreign_query() -> None:
    with pytest.raises(TypeError, match="MGroupFP8GemmQuery"):
        mgg.plan(object())


def test_verbs_reject_foreign_plans() -> None:
    from b12x.gemm import block_fp8_linear as bfl

    foreign = bfl.plan(bfl.Caps(
        device="cpu", max_tokens=8, in_features=128, out_features=256,
    ))
    for verb in (mgg.masked_mm, mgg.contiguous_mm):
        with pytest.raises(ValueError, match="plan belongs to"):
            verb(None, None, None, None, plan=foreign)


def _masked_operands(g=4, m_cap=96, n=128, k=256):
    a = torch.zeros((g, m_cap, k), dtype=torch.float8_e4m3fn)
    sfa = torch.ones((g, m_cap, k // 128), dtype=torch.float32)
    b = torch.zeros((g, n, k), dtype=torch.float8_e4m3fn)
    sfb = torch.ones((g, n, k // 128), dtype=torch.float32)
    d = torch.zeros((g, m_cap, n), dtype=torch.bfloat16)
    masked_m = torch.full((g,), m_cap // 2, dtype=torch.int32)
    return (a, sfa), (b, sfb), d, masked_m


def _contiguous_operands(m_total=96, g=4, n=128, k=256):
    a = torch.zeros((m_total, k), dtype=torch.float8_e4m3fn)
    sfa = torch.ones((m_total, k // 32), dtype=torch.float32)
    b = torch.zeros((g, n, k), dtype=torch.float8_e4m3fn)
    sfb = torch.ones((g, n, k // 128), dtype=torch.float32)
    d = torch.zeros((m_total, n), dtype=torch.bfloat16)
    labels = torch.zeros((m_total,), dtype=torch.int32)
    return (a, sfa), (b, sfb), d, labels


def _state(query) -> _MGroupFP8ExecutionState:
    from b12x.gemm.mgroup_fp8_gemm._tuning import default_config

    return _MGroupFP8ExecutionState(
        query, default_config(query, None), torch.device("cpu"), None, None
    )


def test_masked_state_validates_then_needs_cuda() -> None:
    # Valid operands pass all host-side capacity/geometry checks and reach the
    # CUDA-only scale-packing boundary (kernels need a GPU; see the GPU section).
    state = _state(_query())
    lhs, rhs, d, masked_m = _masked_operands()
    with pytest.raises(ValueError, match="CUDA"):
        state.run_masked(lhs, rhs, d, masked_m)


def test_masked_state_enforces_capacity() -> None:
    state = _state(_query(m_capacity=64))
    lhs, rhs, d, masked_m = _masked_operands(m_cap=96)
    with pytest.raises(ValueError, match="exceeds planned m_capacity"):
        state.run_masked(lhs, rhs, d, masked_m)


def test_masked_state_rejects_bad_mask() -> None:
    state = _state(_query())
    lhs, rhs, d, masked_m = _masked_operands()
    with pytest.raises(ValueError, match="masked_m"):
        state.run_masked(lhs, rhs, d, masked_m.to(torch.int64))


def test_contiguous_state_validates_then_needs_cuda() -> None:
    state = _state(_query(mode="contiguous", a_sf_gran=32, m_capacity=128))
    lhs, rhs, d, labels = _contiguous_operands(m_total=96)
    with pytest.raises(ValueError, match="CUDA"):
        state.run_contiguous(lhs, rhs, d, labels)


def test_contiguous_state_enforces_capacity() -> None:
    state = _state(_query(mode="contiguous", a_sf_gran=32, m_capacity=64))
    lhs, rhs, d, labels = _contiguous_operands(m_total=96)
    with pytest.raises(ValueError, match="exceeds planned m_capacity"):
        state.run_contiguous(lhs, rhs, d, labels)


def test_contiguous_tile_m_must_divide_label_alignment() -> None:
    from b12x.gemm.mgroup_fp8_gemm._tuning import (
        MGroupFP8GemmConfig,
        TUNING,
    )

    query = _query(mode="contiguous", a_sf_gran=32)
    config = MGroupFP8GemmConfig(backend="cutedsl", tile_m=32, tile_n=128, tile_k=128)
    with pytest.raises(ValueError, match="128-aligned"):
        TUNING.validate_config(query, config, None)
    assert TUNING.default_config(query, None).tile_m == 64
    big = _query(mode="contiguous", a_sf_gran=32, m_capacity=8192)
    assert TUNING.default_config(big, None).tile_m == 128


def test_query_from_call_masked_derives_static_geometry() -> None:
    lhs, rhs, d, masked_m = _masked_operands(g=4, m_cap=96, n=128, k=256)
    query = mgg.query_from_call(lhs, rhs, d)
    assert query.mode == "masked"
    assert (query.num_groups, query.n, query.k) == (4, 128, 256)
    assert query.m_capacity == 96
    assert (query.a_sf_gran, query.b_sf_gran) == (128, 128)
    assert query.expected_m is None


def test_query_from_call_contiguous_with_extra_capacity() -> None:
    lhs, rhs, d, labels = _contiguous_operands(m_total=96, g=8, n=128, k=256)
    d_cap = torch.zeros((128, 128), dtype=torch.bfloat16)
    query = mgg.query_from_call(lhs, rhs, d_cap, m_capacity=128, expected_m=64)
    assert query.mode == "contiguous"
    assert query.m_capacity == 128
    assert query.num_groups == 8
    assert query.a_sf_gran == 32
    assert query.expected_m == 64


def test_query_from_call_rejects_wrong_sfa_granularity() -> None:
    lhs, rhs, d, _ = _masked_operands()
    a, _ = lhs
    bad_sfa = torch.ones((4, 96, 8), dtype=torch.float32)  # gran-32 shape in masked mode
    with pytest.raises(ValueError, match="SFA"):
        mgg.query_from_call((a, bad_sfa), rhs, d)


# ---------------------------------------------------------------------------
# GPU section (require_b12x)
# ---------------------------------------------------------------------------


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact UE8M0 ceil (DeepGEMM parity, see test_fp8_quant_deepgemm_parity)."""
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _per_token_cast_fp8(x: torch.Tensor, gran_k: int):
    """DeepGEMM per_token_cast_to_fp8 parity helper (gran_k granularity)."""
    m, n = x.shape
    assert n % gran_k == 0
    xv = x.view(m, n // gran_k, gran_k)
    amax = xv.abs().float().amax(dim=2).clamp(1e-4)
    sf = _ceil_to_ue8m0(amax / 448.0)
    fp8 = (xv * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n)
    return fp8, sf  # sf: fp32 power-of-two, [m, n//gran_k]


def _dequant(values: torch.Tensor, scales_f32: torch.Tensor, gran_k: int) -> torch.Tensor:
    rows, k = values.shape
    return values.to(torch.float32) * scales_f32.repeat_interleave(gran_k, dim=1).view(rows, k)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item()


@contextmanager
def _prepared_mgroup(query, *operand_args, override=None):
    """plan + PreparationSession with a real priming call over the given operands."""
    from b12x.preparation import PreparationSession, PreparedCall

    plan = mgg.plan(query, override=override)
    if query.mode == "masked":
        lhs, rhs, d, masked_m = operand_args
        run_args = (lhs, rhs, d, masked_m)
        verb = lambda state: state.run_masked(*run_args)
    else:
        lhs, rhs, d, labels = operand_args
        run_args = (lhs, rhs, d, labels)
        verb = lambda state: state.run_contiguous(*run_args)

    def call(state):
        return PreparedCall(run=lambda: verb(state))

    with PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="mgroup", prepare_call=call),))
        session.freeze()
        yield plan


def _masked_case(groups, m_cap, n, k, lengths, *, seed=0, expected_m=None):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    a_f32 = torch.randn((groups * m_cap, k), device=dev) * 0.5
    a_fp8, sfa = _per_token_cast_fp8(a_f32, gran_k=128)
    a_fp8, sfa = a_fp8.view(groups, m_cap, k), sfa.view(groups, m_cap, k // 128)
    b_parts, sb_parts = [], []
    for g in range(groups):
        bg, sbg = _per_token_cast_fp8(torch.randn((n, k), device=dev) * 0.5, gran_k=128)
        b_parts.append(bg)
        sb_parts.append(sbg)
    b, sfb = torch.stack(b_parts), torch.stack(sb_parts)
    d = torch.empty((groups, m_cap, n), dtype=torch.bfloat16, device=dev)
    masked_m = torch.tensor(lengths, dtype=torch.int32, device=dev)
    return (a_fp8, sfa), (b, sfb), d, masked_m


def _masked_reference(lhs, rhs, masked_m) -> torch.Tensor:
    (a, sfa), (b, sfb) = lhs, rhs
    groups, m_cap, k = a.shape
    n = b.shape[1]
    d = torch.zeros((groups, m_cap, n), dtype=torch.float32, device=a.device)
    lens = masked_m.cpu().tolist()
    for g in range(groups):
        a_dec = _dequant(a[g], sfa[g], 128)
        b_dec = _dequant(b[g], sfb[g], 128)
        ln = lens[g]
        if ln:
            d[g, :ln] = a_dec[:ln] @ b_dec.T
    return d.to(torch.bfloat16)


def _zero_masked_tail(d: torch.Tensor, masked_m: torch.Tensor) -> torch.Tensor:
    """Compare on the contract region only: rows >= masked_m are undefined."""
    d = d.clone()
    for g, ln in enumerate(masked_m.cpu().tolist()):
        d[g, ln:] = 0
    return d


def test_masked_mm_matches_reference_odd_cap_zero_group() -> None:
    require_b12x()
    groups, m_cap, n, k = 4, 35, 192, 256
    lengths = [0, 17, 35, 5]  # zero group, partial rows, full cap
    lhs, rhs, d, masked_m = _masked_case(groups, m_cap, n, k, lengths, seed=7)
    query = mgg.MGroupFP8GemmQuery(
        mode="masked", num_groups=groups, n=n, k=k, m_capacity=m_cap,
        a_sf_gran=128, expected_m=32,
    )
    with _prepared_mgroup(query, lhs, rhs, d, masked_m) as plan:
        d.zero_()
        out = mgg.masked_mm(lhs, rhs, d, masked_m, plan=plan, expected_m=32)
    ref = _masked_reference(lhs, rhs, masked_m)
    # Dead tiles are skipped outright: rows >= masked_m keep whatever D held,
    # so finite/value assertions must stay inside the contract region.
    out_c = _zero_masked_tail(out, masked_m)
    assert torch.isfinite(out_c.float()).all()
    assert _cosine(out_c, ref) >= 1 - 2e-3
    torch.testing.assert_close(out_c.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_masked_mm_expected_m_1_keeps_capacity_semantics() -> None:
    """bs=1 masked decode carries expected_m=1; the dense engine must NOT see
    it (its small-M tactics assume live M == the hint, breaking m_cap rows)."""
    require_b12x()
    groups, m_cap, n, k = 8, 128, 512, 512
    lengths = [1, 0, 1, 0, 1, 1, 0, 1]
    lhs, rhs, d, masked_m = _masked_case(groups, m_cap, n, k, lengths, seed=41)
    query = mgg.MGroupFP8GemmQuery(
        mode="masked", num_groups=groups, n=n, k=k, m_capacity=m_cap,
        a_sf_gran=128, expected_m=1,
    )
    with _prepared_mgroup(query, lhs, rhs, d, masked_m) as plan:
        out = mgg.masked_mm(lhs, rhs, d, masked_m, plan=plan, expected_m=1)
    ref = _masked_reference(lhs, rhs, masked_m)
    out_c = _zero_masked_tail(out, masked_m)
    assert torch.isfinite(out_c.float()).all()
    assert _cosine(out_c, ref) >= 1 - 2e-3
    torch.testing.assert_close(out_c.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_masked_two_live_mask_sets_reuse_one_plan() -> None:
    require_b12x()
    groups, m_cap, n, k = 8, 35, 2112, 512
    lhs, rhs, d, masked_m_a = _masked_case(
        groups, m_cap, n, k, [32, 17, 5, 0, 32, 1, 24, 8], seed=11,
    )
    masked_m_b = torch.tensor(
        [1, 32, 0, 9, 13, 32, 4, 27], dtype=torch.int32, device="cuda",
    )
    query = mgg.MGroupFP8GemmQuery(
        mode="masked", num_groups=groups, n=n, k=k, m_capacity=m_cap,
        a_sf_gran=128, expected_m=32,
    )
    with _prepared_mgroup(query, lhs, rhs, d, masked_m_a) as plan:
        d.zero_()
        out_a = mgg.masked_mm(lhs, rhs, d, masked_m_a, plan=plan, expected_m=32).clone()
        d.zero_()
        out_b = mgg.masked_mm(lhs, rhs, d, masked_m_b, plan=plan, expected_m=32)
    ref_a = _zero_masked_tail(_masked_reference(lhs, rhs, masked_m_a), masked_m_a)
    ref_b = _zero_masked_tail(_masked_reference(lhs, rhs, masked_m_b), masked_m_b)
    assert _cosine(_zero_masked_tail(out_a, masked_m_a), ref_a) >= 1 - 2e-3
    assert _cosine(_zero_masked_tail(out_b, masked_m_b), ref_b) >= 1 - 2e-3


def _contiguous_case(groups, n, k, lengths, alignment, *, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    intervals, end = [], 0
    for ln in lengths:
        start = -(-end // alignment) * alignment
        end = start + ln
        intervals.append((start, end))
    m_total = end
    a_f32 = torch.randn((m_total, k), device=dev) * 0.5
    a_fp8, sfa = _per_token_cast_fp8(a_f32, gran_k=32)
    b_parts, sb_parts = [], []
    for g in range(groups):
        bg, sbg = _per_token_cast_fp8(torch.randn((n, k), device=dev) * 0.5, gran_k=128)
        b_parts.append(bg)
        sb_parts.append(sbg)
    b, sfb = torch.stack(b_parts), torch.stack(sb_parts)
    d = torch.empty((m_total, n), dtype=torch.bfloat16, device=dev)
    labels = torch.full((m_total,), -1, dtype=torch.int32)
    for g, (s, e) in enumerate(intervals):
        labels[s:e] = g
    return (a_fp8, sfa), (b, sfb), d, labels.to(dev), intervals


def _contiguous_reference(lhs, rhs, intervals) -> torch.Tensor:
    (a, sfa), (b, sfb) = lhs, rhs
    m_total, k = a.shape
    n = b.shape[1]
    d = torch.zeros((m_total, n), dtype=torch.float32, device=a.device)
    a_dec = _dequant(a, sfa, 32)
    for g, (s, e) in enumerate(intervals):
        if e > s:
            b_dec = _dequant(b[g], sfb[g], 128)
            d[s:e] = a_dec[s:e] @ b_dec.T
    return d.to(torch.bfloat16)


def test_contiguous_mm_matches_reference_empty_and_tail() -> None:
    require_b12x()
    groups, n, k = 5, 192, 256
    # leading empty group, mid-array empty group, non-128-multiple tail
    lengths = [0, 128, 96, 0, 64]
    lhs, rhs, d, labels, intervals = _contiguous_case(groups, n, k, lengths, 128, seed=13)
    query = mgg.MGroupFP8GemmQuery(
        mode="contiguous", num_groups=groups, n=n, k=k, m_capacity=d.shape[0],
        a_sf_gran=32,
    )
    with _prepared_mgroup(query, lhs, rhs, d, labels) as plan:
        d.fill_(float("nan"))
        out = mgg.contiguous_mm(lhs, rhs, d, labels, plan=plan)
    ref = _contiguous_reference(lhs, rhs, intervals)
    assert torch.isfinite(out.float()).all()
    # padding rows are zero-filled by the op (its contract, not the pack's)
    padding = labels < 0
    assert padding.any()
    assert (out[padding] == 0).all()
    assert _cosine(out, ref) >= 1 - 2e-3
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_masked_mm_replays_under_cuda_graph() -> None:
    require_b12x()
    groups, m_cap, n, k = 4, 32, 256, 256
    lhs, rhs, d, masked_m = _masked_case(groups, m_cap, n, k, [1, 5, 0, 32], seed=21)
    query = mgg.MGroupFP8GemmQuery(
        mode="masked", num_groups=groups, n=n, k=k, m_capacity=m_cap,
        a_sf_gran=128, expected_m=16,
    )
    with _prepared_mgroup(query, lhs, rhs, d, masked_m) as plan:
        mgg.masked_mm(lhs, rhs, d, masked_m, plan=plan, expected_m=16)
        eager = d.clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                mgg.masked_mm(lhs, rhs, d, masked_m, plan=plan, expected_m=16)
            pointer = d.data_ptr()
            allocated = torch.cuda.memory_allocated()
            for _ in range(3):
                d.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert d.data_ptr() == pointer
            torch.testing.assert_close(
                _zero_masked_tail(d, masked_m), _zero_masked_tail(eager, masked_m),
                rtol=0, atol=0,
            )
        finally:
            graph.reset()


def test_contiguous_mm_replays_under_cuda_graph() -> None:
    require_b12x()
    groups, n, k = 4, 256, 256
    lengths = [64, 0, 128, 32]
    lhs, rhs, d, labels, intervals = _contiguous_case(groups, n, k, lengths, 128, seed=22)
    query = mgg.MGroupFP8GemmQuery(
        mode="contiguous", num_groups=groups, n=n, k=k, m_capacity=d.shape[0],
        a_sf_gran=32,
    )
    with _prepared_mgroup(query, lhs, rhs, d, labels) as plan:
        mgg.contiguous_mm(lhs, rhs, d, labels, plan=plan)
        eager = d.clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                mgg.contiguous_mm(lhs, rhs, d, labels, plan=plan)
            pointer = d.data_ptr()
            allocated = torch.cuda.memory_allocated()
            for _ in range(3):
                d.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert d.data_ptr() == pointer
            torch.testing.assert_close(d, eager, rtol=0, atol=0)
        finally:
            graph.reset()


def test_contiguous_mm_large_group_offsets() -> None:
    """B stack past 2**31 bytes: late groups must address through TMA's 64-bit
    descriptor path (group id * n * k overflows Int32 element offsets)."""
    require_b12x()
    groups, n, k = 128, 4096, 5120  # B: 128*4096*5120 = 2.5 GiB of e4m3
    dev = torch.device("cuda")
    torch.manual_seed(31)
    lengths = [0] * groups
    lengths[3], lengths[120] = 128, 128
    lhs, rhs, d, labels, intervals = _contiguous_case(groups, n, k, lengths, 128, seed=31)
    query = mgg.MGroupFP8GemmQuery(
        mode="contiguous", num_groups=groups, n=n, k=k, m_capacity=d.shape[0],
        a_sf_gran=32,
    )
    with _prepared_mgroup(query, lhs, rhs, d, labels) as plan:
        d.fill_(float("nan"))
        out = mgg.contiguous_mm(lhs, rhs, d, labels, plan=plan)
    ref = _contiguous_reference(lhs, rhs, intervals)
    assert _cosine(out, ref) >= 1 - 2e-3


def test_pack_grouped_scales_into_matches_torch_reference() -> None:
    """The owned-workspace Triton packer is byte-exact vs the torch reference."""
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._packing import (
        pack_grouped_scales,
        pack_grouped_scales_into,
    )

    dev = torch.device("cuda")
    torch.manual_seed(53)
    for gran, groups, rows, k in (
        (128, 4, 96, 256),    # masked SFA: partial 128-row atom, REP=4
        (128, 3, 192, 512),   # grouped SFB shape, partial row atom
        (128, 8, 128, 2176),  # exact rows, 17 K128 tiles
        (32, 1, 152, 256),    # contiguous SFA: gran-32 straight through
        (128, 1, 35, 256),    # single group, odd rows
    ):
        shape = (rows, k // gran) if groups == 1 else (groups, rows, k // gran)
        scales = torch.exp2(
            torch.randint(-8, 8, shape, device=dev).float()
        ).contiguous()
        ref = pack_grouped_scales(
            scales, rows=rows, k=k, num_groups=groups, gran=gran
        )
        # ref is the (32, 4, MT, 4, KT, G) atom view; its storage order is
        # (G, MT, KT, 32, 4, 4) — the exact byte order the packer writes.
        phys = ref.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8).flatten()
        dst = torch.full((phys.numel(),), 0xAB, dtype=torch.uint8, device=dev)
        pack_grouped_scales_into(
            scales, dst, rows=rows, k=k, num_groups=groups, gran=gran
        )
        torch.testing.assert_close(dst, phys, rtol=0, atol=0)


def test_pack_grouped_scales_masked_skips_dead_groups() -> None:
    """masked_m==0 groups are not packed; live groups stay byte-exact."""
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._packing import (
        pack_grouped_scales,
        pack_grouped_scales_fast,
    )

    dev = torch.device("cuda")
    torch.manual_seed(61)
    groups, rows, k = 8, 96, 512
    scales = torch.exp2(
        torch.randint(-8, 8, (groups, rows, k // 128), device=dev).float()
    ).contiguous()
    masked_m = torch.tensor([3, 0, 96, 0, 1, 0, 17, 0], dtype=torch.int32, device=dev)
    ref = pack_grouped_scales(scales, rows=rows, k=k, num_groups=groups, gran=128)
    phys = ref.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8)
    out = pack_grouped_scales_fast(
        scales, rows=rows, k=k, num_groups=groups, gran=128, masked_m=masked_m,
    )
    row_atoms, k_atoms = -(-rows // 128), -(-(k // 32) // 4)
    group_bytes = row_atoms * k_atoms * 512
    out = out.view(groups, group_bytes)
    for g in range(groups):
        if masked_m[g] > 0:
            torch.testing.assert_close(out[g], phys[g].flatten(), rtol=0, atol=0)


def test_masked_mm_tile_k64_matches_reference() -> None:
    """The masked kernel also builds with tile_k=64 (manual BK64 SF staging)."""
    require_b12x()
    groups, m_cap, n, k = 4, 64, 512, 512
    lengths = [3, 64, 0, 17]
    lhs, rhs, d, masked_m = _masked_case(groups, m_cap, n, k, lengths, seed=29)
    query = mgg.MGroupFP8GemmQuery(
        mode="masked", num_groups=groups, n=n, k=k, m_capacity=m_cap,
        a_sf_gran=128,
    )
    override = mgg.MGroupFP8GemmConfig(
        backend="cutedsl", tile_m=128, tile_n=128, tile_k=64,
    )
    with _prepared_mgroup(query, lhs, rhs, d, masked_m, override=override) as plan:
        d.zero_()
        out = mgg.masked_mm(lhs, rhs, d, masked_m, plan=plan)
    ref = _masked_reference(lhs, rhs, masked_m)
    out_c = _zero_masked_tail(out, masked_m)
    assert torch.isfinite(out_c.float()).all()
    assert _cosine(out_c, ref) >= 1 - 2e-3
    torch.testing.assert_close(out_c.float(), ref.float(), rtol=2e-2, atol=2e-2)
