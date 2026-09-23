"""paged_decode parity with the paged reference through the public lifecycle.

Caches use vLLM's hybrid DiffKV page layout: one allocation of
``[pages, layers, kv_heads, page_size, head_dim_qk + head_dim_vo]`` whose K and
V are head-major-within-page views of one layer.  Batches are ragged (every
request draws its own query length up to ``max_q_per_req``), and q is a
strided view of a wider fused-projection row.
"""

from __future__ import annotations

import pytest
import torch
from b12x.preparation import PreparationSession, PreparedCall

from b12x.attention import paged_decode
from b12x.attention.paged.reference import paged_attention_reference

from ..conftest import require_b12x

# name: (q_heads, kv_heads, dim_qk, dim_vo, window_left, causal, sinks, fp8,
#        max_batch, max_q_per_req, max_seq)
CASES = {
    # Full-context GQA16 192/128 layer: split-KV at small batch.
    "global_verify": (16, 1, 192, 128, -1, True, False, False, 3, 8, 9000),
    "global_decode": (16, 1, 192, 128, -1, True, False, False, 24, 1, 5000),
    # Sliding-window GQA8 layer with sinks: unsplit, GQA group sliced across CTAs.
    "swa_verify": (16, 2, 192, 128, 127, True, True, False, 4, 8, 5000),
    "swa_decode": (16, 2, 192, 128, 127, True, True, False, 8, 1, 3000),
    # The same target layers with FP8 KV (per-layer dequant scales).
    "global_fp8_verify": (16, 1, 192, 128, -1, True, False, True, 3, 8, 9000),
    "global_fp8_decode": (16, 1, 192, 128, -1, True, False, True, 24, 1, 5000),
    "swa_fp8_verify": (16, 2, 192, 128, 127, True, True, True, 4, 8, 5000),
    # Non-causal windowed drafter layer, FP8 KV: split (small batch) and not.
    "draft_fp8_split": (16, 2, 128, 128, 1023, False, True, True, 2, 8, 6000),
    "draft_fp8": (16, 2, 128, 128, 1023, False, True, True, 16, 8, 3000),
    "draft_bf16": (16, 2, 128, 128, 1023, False, True, False, 6, 8, 4000),
}
PAGE = 64


class _Case:
    """Fixed caches for one case; ``draw`` makes a new ragged batch over them."""

    def __init__(self, name: str, seed: int):
        (self.hq, self.hkv, self.dqk, self.dvo, self.window, self.causal, sinks, self.fp8,
         self.max_batch, self.max_q, self.max_seq) = CASES[name]
        self.name = name
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.width = (self.max_seq + PAGE - 1) // PAGE
        self.pool = self.max_batch * self.width + 5
        packed = torch.randn((self.pool, 2, self.hkv, PAGE, self.dqk + self.dvo),
                             generator=self.gen, device="cuda")
        self.k_descale = self.v_descale = None
        if self.fp8:
            view = packed[:, 1].transpose(1, 2)
            kd = float(view[..., :self.dqk].abs().amax()) / 448.0
            vd = float(view[..., self.dqk:].abs().amax()) / 448.0
            packed[:, :, :, :, :self.dqk] /= kd
            packed[:, :, :, :, self.dqk:] /= vd
            packed = packed.to(torch.float8_e4m3fn)
            # One dequant scale per layer, as vLLM keeps them.
            self.k_descale = torch.tensor([kd], dtype=torch.float32, device="cuda")
            self.v_descale = torch.tensor([vd], dtype=torch.float32, device="cuda")
        else:
            packed = packed.to(torch.bfloat16)
        view = packed[:, 1].transpose(1, 2)  # [pages, page, Hkv, Dqk + Dvo]
        self.k_cache, self.v_cache = view[..., :self.dqk], view[..., self.dqk:]
        self.sinks = (
            (torch.randn(self.hq, generator=self.gen, device="cuda") * 2.0).float()
            if sinks else None
        )

    def draw(self, *, q_lens=None):
        batch = self.max_batch
        lens = torch.randint(1, self.max_seq + 1, (batch,), generator=self.gen,
                             device="cuda").tolist()
        lens[0] = self.max_seq  # always cover the longest split partition
        if q_lens is None:
            q_lens = torch.randint(1, self.max_q + 1, (batch,), generator=self.gen,
                                   device="cuda").tolist()
        q_lens = [min(q, s) for q, s in zip(q_lens, lens, strict=True)]
        perm = torch.randperm(self.pool, generator=self.gen, device="cuda").to(torch.int32)
        page_table = perm[: batch * self.width].view(batch, self.width).contiguous()
        total = sum(q_lens)
        fused = torch.randn((total, self.hq * self.dqk + 3 * 64), generator=self.gen,
                            device="cuda").to(torch.bfloat16)
        cu = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu[1:] = torch.tensor(q_lens, device="cuda").cumsum(0)
        return dict(
            q=fused[:, : self.hq * self.dqk].view(total, self.hq, self.dqk),
            k_cache=self.k_cache, v_cache=self.v_cache, page_table=page_table,
            cache_seqlens=torch.tensor(lens, dtype=torch.int32, device="cuda"),
            cu_seqlens_q=cu, attention_sink_bias=self.sinks,
            k_descale=self.k_descale, v_descale=self.v_descale,
        )

    def caps(self):
        return paged_decode.Caps(
            device="cuda", num_q_heads=self.hq, num_kv_heads=self.hkv,
            head_dim_qk=self.dqk, head_dim_vo=self.dvo, page_size=PAGE,
            max_batch=self.max_batch, max_q_per_req=self.max_q,
            kv_dtype=torch.float8_e4m3fn if self.fp8 else torch.bfloat16,
            window_left=self.window, causal=self.causal, has_sinks=self.sinks is not None,
        )

    def reference(self, inputs):
        batch = inputs["cache_seqlens"].shape[0]
        per_request = [None if d is None else d.expand(batch).contiguous()
                       for d in (inputs["k_descale"], inputs["v_descale"])]
        ref, _ = paged_attention_reference(
            inputs["q"].contiguous(), inputs["k_cache"], inputs["v_cache"],
            inputs["page_table"], inputs["cache_seqlens"], inputs["cu_seqlens_q"],
            k_descale=per_request[0], v_descale=per_request[1],
            causal=self.causal, window_left=self.window,
            attention_sink_bias=inputs["attention_sink_bias"],
        )
        return ref


def _output(inputs):
    q, v = inputs["q"], inputs["v_cache"]
    return torch.full((q.shape[0], q.shape[1], v.shape[3]), float("nan"),
                      dtype=torch.bfloat16, device="cuda")


def _scratch(plan):
    return [torch.empty(s.shape, dtype=s.dtype, device="cuda") for s in plan.scratch_specs()]


def _prepare(case: _Case, session: PreparationSession, inputs):
    declaration = paged_decode.plan(case.caps())

    def prepare_call(state):
        out = _output(inputs)
        binding = state.bind(plan=declaration, scratch=_scratch(declaration), output=out,
                             **inputs)
        return PreparedCall(run=lambda: state.run(binding), output=out)

    result = session.prepare((declaration.request(name=case.name, prepare_call=prepare_call),))
    return result, result.plans[case.name]


def _check(out, ref):
    assert torch.isfinite(out).all()
    err = (out.float() - ref.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
    assert err <= 2e-2, err
    assert cos >= 0.9999, cos


def test_default_splits_fill_one_wave():
    """Split-KV grids stay within one wave of one-CTA-per-SM decode CTAs."""
    from b12x.attention.paged_decode._kernel import default_max_splits

    for sms in (188, 170, 84):
        for heads in (1, 2):
            for batch in range(1, sms + 1):
                ctas = batch * heads * default_max_splits(batch, heads, sms, 128)
                assert ctas <= max(sms, batch * heads)
                assert ctas > max(sms, batch * heads) // 2
    assert default_max_splits(32, 1, 188) == 5


@pytest.mark.parametrize("name", sorted(CASES))
def test_paged_decode_matches_reference(name):
    require_b12x()
    case = _Case(name, seed=1)
    inputs = case.draw()
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(case, session, inputs)
        outs = []
        for _ in range(2):
            out = _output(inputs)
            paged_decode.run(paged_decode.bind(plan, scratch=_scratch(plan), output=out,
                                               **inputs))
            outs.append(out)
        result.close()
    _check(outs[0], case.reference(inputs))
    assert torch.equal(outs[0], outs[1]), "paged_decode must be deterministic"


@pytest.mark.parametrize(
    "name", ["global_verify", "swa_verify", "global_fp8_verify", "draft_fp8_split"]
)
def test_paged_decode_graph_replay_with_new_lengths(name):
    """One capture serves new context lengths, pages and query-length mix."""
    require_b12x()
    case = _Case(name, seed=2)
    first = case.draw()
    q_lens = torch.diff(first["cu_seqlens_q"]).tolist()
    total = int(first["q"].shape[0])
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(case, session, first)
        live = {k: (v.clone() if isinstance(v, torch.Tensor) and k in (
            "q", "page_table", "cache_seqlens", "cu_seqlens_q") else v)
            for k, v in first.items()}
        out = _output(first)
        binding = paged_decode.bind(plan, scratch=_scratch(plan), output=out, **live)
        paged_decode.run(binding)
        torch.cuda.synchronize()
        _check(out, case.reference(live))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode.run(binding)
        # Same total rows (the captured extent), reversed per-request lengths.
        second = case.draw(q_lens=q_lens[::-1])
        if int(second["q"].shape[0]) != total:
            pytest.skip("clipped query lengths changed the captured row count")
        for key in ("q", "page_table", "cache_seqlens", "cu_seqlens_q"):
            live[key].copy_(second[key])
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        result.close()
    _check(out, case.reference(live))


@pytest.mark.parametrize("fp8", [False, True])
def test_paged_decode_reads_pages_past_the_int32_element_range(fp8):
    """Recycled high page ids: live pages parked past 2**31 / page-stride elements."""
    require_b12x()
    hq, hkv, dqk, dvo, batch, q_len, seq = 16, 1, 192, 128, 2, 4, 5 * PAGE - 3
    page_elems = 2 * hkv * PAGE * (dqk + dvo)  # hybrid page: two layers
    first_live = (2**31) // page_elems + 1
    width = (seq + PAGE - 1) // PAGE
    pool = first_live + batch * width
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    packed = torch.empty((pool, 2, hkv, PAGE, dqk + dvo), dtype=dtype, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(5)
    live = torch.randn((pool - first_live, 2, hkv, PAGE, dqk + dvo), generator=gen,
                       device="cuda")
    packed[first_live:] = (live / 4).to(dtype)
    view = packed[:, 1].transpose(1, 2)
    k_cache, v_cache = view[..., :dqk], view[..., dqk:]
    ids = torch.arange(first_live, pool, dtype=torch.int32, device="cuda")
    page_table = ids.flip(0).view(batch, width).contiguous()
    q = torch.randn((batch * q_len, hq, dqk), generator=gen, device="cuda").to(torch.bfloat16)
    cu = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device="cuda")
    lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    descale = torch.tensor([4.0], device="cuda") if fp8 else None
    inputs = dict(q=q, k_cache=k_cache, v_cache=v_cache, page_table=page_table,
                  cache_seqlens=lens, cu_seqlens_q=cu, attention_sink_bias=None,
                  k_descale=descale, v_descale=descale)
    caps = paged_decode.Caps(device="cuda", num_q_heads=hq, num_kv_heads=hkv,
                             head_dim_qk=dqk, head_dim_vo=dvo, page_size=PAGE,
                             max_batch=batch, max_q_per_req=8, kv_dtype=dtype)
    declaration = paged_decode.plan(caps)
    out = _output(inputs)

    def prepare_call(state):
        binding = state.bind(plan=declaration, scratch=_scratch(declaration),
                             output=out, **inputs)
        return PreparedCall(run=lambda: state.run(binding), output=out)

    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result = session.prepare((declaration.request(name="high_pids",
                                                      prepare_call=prepare_call),))
        out.fill_(float("nan"))
        paged_decode.run(paged_decode.bind(declaration, scratch=_scratch(declaration),
                                           output=out, **inputs))
        result.close()
    per_request = None if descale is None else descale.expand(batch).contiguous()
    ref, _ = paged_attention_reference(
        q, k_cache, v_cache, page_table, lens, cu,
        k_descale=per_request, v_descale=per_request, causal=True, window_left=-1,
        attention_sink_bias=None,
    )
    _check(out, ref)


@pytest.mark.parametrize("fp8", [False, True])
def test_paged_decode_write_kv_appends_rows_exactly(fp8):
    """write_kv puts each token's K/V rows at its slot, bit-exact, and nothing else.

    Pages sit past 2**31 / page-stride elements (Int64 addressing), key rows are
    strided views into a fused QKV buffer, and negative slots are skipped.
    """
    require_b12x()
    hkv, dqk, dvo, tokens = 2, 192, 128, 37
    page_elems = 2 * hkv * PAGE * (dqk + dvo)  # hybrid page: two layers
    first_live = (2**31) // page_elems + 1
    pool = first_live + 4
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    gen = torch.Generator(device="cuda").manual_seed(11)
    packed = torch.zeros((pool, 2, hkv, PAGE, dqk + dvo), dtype=dtype, device="cuda")
    packed[first_live:] = torch.randn(packed[first_live:].shape, generator=gen,
                                      device="cuda").to(dtype)
    view = packed[:, 1].transpose(1, 2)
    k_cache, v_cache = view[..., :dqk], view[..., dqk:]
    q_size = 16 * dqk
    qkv = torch.randn((tokens, q_size + hkv * (dqk + dvo)), generator=gen, device="cuda") * 60
    qkv = qkv.to(torch.bfloat16)
    key = qkv[:, q_size:q_size + hkv * dqk].view(tokens, hkv, dqk)
    value = qkv[:, q_size + hkv * dqk:].reshape(tokens, hkv, dvo).contiguous()
    slots = torch.randperm(4 * PAGE, generator=gen, device="cuda")[:tokens] + first_live * PAGE
    slots[[3, 20]] = -1
    scale = torch.tensor([3.3], device="cuda")  # not a power of two: rounding matters
    before = packed.clone()

    caps = paged_decode.Caps(device="cuda", num_q_heads=16, num_kv_heads=hkv,
                             head_dim_qk=dqk, head_dim_vo=dvo, page_size=PAGE,
                             max_batch=1, max_q_per_req=8, kv_dtype=dtype)
    declaration = paged_decode.plan(caps)
    q = torch.zeros((8, 16, dqk), dtype=torch.bfloat16, device="cuda")
    inputs = dict(q=q, k_cache=k_cache, v_cache=v_cache,
                  page_table=torch.full((1, 1), first_live, dtype=torch.int32, device="cuda"),
                  cache_seqlens=torch.full((1,), 8, dtype=torch.int32, device="cuda"),
                  cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32, device="cuda"),
                  attention_sink_bias=None,
                  k_descale=scale if fp8 else None, v_descale=scale if fp8 else None)
    out = _output(inputs)

    def prepare_call(state):
        binding = state.bind(plan=declaration, scratch=_scratch(declaration), output=out,
                             **inputs)
        return PreparedCall(run=lambda: state.run(binding), output=out)

    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result = session.prepare((declaration.request(name="write_kv",
                                                      prepare_call=prepare_call),))
        packed.copy_(before)
        paged_decode.write_kv(declaration, key=key, value=value, k_cache=k_cache,
                              v_cache=v_cache, slot_mapping=slots,
                              k_scale=scale if fp8 else None, v_scale=scale if fp8 else None)
        torch.cuda.synchronize()
        result.close()

    expected = before.clone()
    exp_view = expected[:, 1].transpose(1, 2)
    for t in range(tokens):
        slot = int(slots[t])
        if slot < 0:
            continue
        page, offset = divmod(slot, PAGE)
        k, v = key[t].float(), value[t].float()
        if fp8:
            k, v = k / scale, v / scale
        exp_view[page, offset, :, :dqk] = k.to(dtype)
        exp_view[page, offset, :, dqk:] = v.to(dtype)
    assert torch.equal(packed.view(torch.uint8), expected.view(torch.uint8))
