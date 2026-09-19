"""Precision and materialization choices retain their numerical prerequisites."""
from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")


@pytest.mark.parametrize("rows", (1, 8, 128))
def test_dense_nvfp4_auto_races_a4_and_a16(rows):
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=rows, in_features=2560, padded_in_features=2560,
        out_features=512, activation_mode="auto", activation_scale_available=True,
    )
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.mode for config in configs} == {"a16", "quantized"}
    assert TUNING.configure(query, device=DEVICE).default.mode == ("a16" if rows <= 8 else "quantized")
    for mode in ("a16", "quantized"):
        forced = replace(query, activation_mode=mode)
        assert {config.mode for _, config in TUNING.eligible_plan(forced, DEVICE).candidates} == {mode}
        selected = next(config for config in configs if config.mode == mode)
        assert TUNING.configure(query, device=DEVICE, override=selected).default == selected


def test_dense_nvfp4_without_activation_scale_excludes_a4():
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=1, in_features=256, padded_in_features=256,
        out_features=128, activation_mode="auto", activation_scale_available=False,
    )
    assert {config.mode for _, config in TUNING.eligible_plan(query, DEVICE).candidates} == {"a16"}


def _nvfp4_query():
    from b12x.moe.fused_moe._tuning import MoeDecodeQuery

    return MoeDecodeQuery(
        quant_mode="nvfp4", quant_modes=("nvfp4",), source_format="modelopt_nvfp4",
        activation="silu", io_dtype="bfloat16", num_experts=32, hidden_size=512,
        intermediate_size=512, top_k=4, num_tokens=1024, routed_rows=4096,
        route_num_experts=32, route_logits_dtype=None, apply_router_weight_on_input=False,
        collect_activation_amax=False, deterministic_output=False, swiglu_limit=None,
        swiglu_alpha=1.0, swiglu_beta=0.0, w13_layout="w13", weight_layouts=("mma_view",),
        w4a16_weight_layout=None, w4a16_scale_format=None, w4a16_block_size_m=None,
        fast_math=True, numerical_recipe=None, controls=FrozenMapping(), shared_input_scales=True,
    )


def _nvfp4_auto_query(rows=1):
    return replace(
        _nvfp4_query(), quant_mode="nvfp4_auto", quant_modes=("nvfp4", "w4a16"),
        num_experts=256, hidden_size=6144, intermediate_size=256, top_k=8,
        num_tokens=rows, routed_rows=rows * 8, route_num_experts=256,
        weight_layouts=("mma_view", "source_native"), w4a16_weight_layout="modelopt",
        w4a16_scale_format="e4m3_k16",
    )


@pytest.mark.parametrize("capability,sms", (((12, 0), 188), ((12, 1), 48)))
@pytest.mark.parametrize("rows", (1, 2, 4, 8, 9, 16))
def test_moe_auto_promotes_native_decode_and_races_both_precisions(capability, sms, rows):
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_auto_query(rows)
    device = DeviceIdentity("nvidia", capability, sms, "synthetic Blackwell")
    default = TUNING.configure(query, device=device).default
    configs = [config for _, config in TUNING.eligible_plan(query, device).candidates]
    assert (default.backend == "w4a16") is (rows <= 8)
    assert {config.w4a16_route_mode for config in configs if config.backend == "w4a16"} == (
        {"direct", "packed"} if rows <= 8 else {"packed"}
    )
    assert {config.backend for config in configs} >= {"dynamic", "w4a16"}
    assert default in configs
    assert configs[0].backend == "w4a16"
    pinned_a4 = replace(query, quant_mode="nvfp4", quant_modes=("nvfp4",))
    assert all(config.backend != "w4a16" for _, config in TUNING.eligible_plan(pinned_a4, device).candidates)


@pytest.mark.parametrize("changes", (
    {"hidden_size": 192}, {"top_k": 33, "routed_rows": 33},
    {"apply_router_weight_on_input": True}, {"collect_activation_amax": True},
    {"io_dtype": "float16"},
))
def test_moe_native_direct_rejects_unsupported_query_and_override(changes):
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_auto_query()
    direct = TUNING.configure(query, device=DEVICE).default
    query = replace(query, **changes)
    assert TUNING.configure(query, device=DEVICE).default.backend != "w4a16"
    with pytest.raises(ValueError, match="direct routing"):
        TUNING.configure(query, device=DEVICE, override=direct)
    assert not any(config.backend == "w4a16" and config.w4a16_route_mode == "direct"
                   for _, config in TUNING.eligible_plan(query, DEVICE).candidates)


def test_moe_auto_has_no_native_decode_default_on_other_architectures():
    from b12x.moe.fused_moe._tuning import TUNING

    for device in (None, replace(DEVICE, compute_capability=(10, 0))):
        assert TUNING.configure(_nvfp4_auto_query(), device=device, search=False).default.backend != "w4a16"


@pytest.mark.parametrize("a4_latency,a16_latency,old_backend,backend", (
    (6, 6, "micro", "w4a16"), (5, 6, "micro", "micro"), (6, 5, "w4a16", "w4a16"),
))
def test_moe_precision_race_and_revision_invalidate_cached_choice(
    tmp_path, monkeypatch, a4_latency, a16_latency, old_backend, backend,
):
    from types import SimpleNamespace
    from b12x.preparation import DetectedDevice, MemoryRequirements, Plan, PreparedCall
    from b12x.moe.fused_moe._tuning import TUNING
    from .test_session import _deterministic_timer, session

    _deterministic_timer(monkeypatch)
    old_contract = replace(
        TUNING, candidate_contract_version=4,
        knobs=(replace(TUNING.knobs[0], values=("micro", "dynamic", "w4a16")), *TUNING.knobs[1:]),
    )

    def request(contract):
        plan = Plan(
            contract=contract, query=_nvfp4_auto_query(),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(config=selection.config),
        )

        def call(state):
            latency = a16_latency if state.config.backend == "w4a16" else a4_latency
            # The shared timer measures abs(output - 6) + 1 microseconds.
            return PreparedCall(run=lambda: latency + 5, produce=lambda: None)

        return plan.request(
            name="moe-precision",
            prepare_call=call, benchmark_call=call,
        )

    assert TUNING.candidate_contract_version > old_contract.candidate_contract_version
    for contract, source, expected_backend in (
        (old_contract, "tuned", old_backend),
        (TUNING, "tuned", backend),
        (TUNING, "cached", backend),
    ):
        with session(tmp_path, race_batch=2) as engine:
            engine.device = DetectedDevice(ordinal=None, identity=DEVICE)
            result = engine.prepare((request(contract),))
            selection = result.selections["moe-precision"]
            assert (selection.source, selection.config.backend) == (source, expected_backend)


def test_nvfp4_split_materialization_is_a_real_candidate_and_default():
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_query()
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.nvfp4_materialize_intermediate for config in configs} == {False, True}
    for config in configs:
        if config.nvfp4_materialize_intermediate:
            assert config.nvfp4_share_input and config.dynamic_tile_m == 128
    default = TUNING.configure(query, device=DEVICE).default
    assert default.nvfp4_share_input and default.nvfp4_materialize_intermediate
    assert default in configs


@pytest.mark.parametrize("changes", (
    {"shared_input_scales": False}, {"deterministic_output": True}, {"activation": "relu2"},
    {"intermediate_size": 320}, {"controls": FrozenMapping({"dynamic_down_scale": True})},
    {"controls": FrozenMapping({"dynamic_swap_ab": "1"})},
    {"controls": FrozenMapping({"dynamic_work_source": "ready_queue"})},
))
def test_nvfp4_split_materialization_rejects_unsupported_contracts(changes):
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_query()
    split = next(config for _, config in TUNING.eligible_plan(query, DEVICE).candidates
                 if config.nvfp4_materialize_intermediate)
    invalid = replace(query, **changes)
    with pytest.raises(ValueError):
        TUNING.configure(invalid, device=DEVICE, override=split)
    assert all(not config.nvfp4_materialize_intermediate
               for _, config in TUNING.eligible_plan(invalid, DEVICE).candidates)


def test_compact_w4a8_has_only_its_supported_dynamic_route():
    from b12x.moe.fused_moe._tuning import TUNING

    query = replace(_nvfp4_query(), quant_mode="w4a8_mx", quant_modes=("w4a8_mx",),
                    source_format="fp4_e8m0_k32", intermediate_size=576)
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert configs
    assert {(config.backend, config.dynamic_tile_m, config.dynamic_route_mode, config.route_planner)
            for config in configs} == {("dynamic", 16, "grouped", "internal")}


@pytest.mark.parametrize("heads", (1, 12, 16, 20, 32))
@pytest.mark.parametrize("mode", ("decode", "extend"))
def test_v41_precision_candidates_match_native_head_group_contract(heads, mode):
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._tuning import TUNING

    q = torch.empty((3, heads, 512), dtype=torch.bfloat16)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    plan = mla.plan(
        mla.Caps(device="cpu", num_q_heads=heads, max_q_rows=3, max_width=128,
                 swa_width=128, indexed_width=0, cache_format="deepseek_v41", mode=mode),
        invocation=mla.invocation_from_tensors(q=q, swa_k_cache=cache, out=torch.empty_like(q)),
    )
    configs = [config for _, config in TUNING.eligible_plan(plan.query, DEVICE).candidates]
    expected = {("bf16", 16), ("fp8", 16)} if mode == "extend" else {
        ("bf16", 16), ("fp8", 8),
    }
    if mode == "decode" and heads % 16 == 0:
        expected.add(("fp8", 16))
    assert {(config.v41_compute_mode, config.v41_heads_per_block) for config in configs} == expected
    default = TUNING.configure(plan.query, device=DEVICE).default
    assert default in configs and default.v41_compute_mode == "fp8"
    assert default.v41_heads_per_block == (8 if mode == "decode" and heads % 16 else 16)
    for config in configs:
        assert TUNING.configure(plan.query, device=DEVICE, override=config).default == config
    invalid = replace(default, v41_compute_mode="bf16", v41_heads_per_block=8)
    with pytest.raises(ValueError, match="head grouping"):
        TUNING.configure(plan.query, device=DEVICE, override=invalid)
    if mode == "decode" and heads % 16:
        with pytest.raises(ValueError, match="complete 16-head groups"):
            TUNING.configure(plan.query, device=DEVICE, override=replace(default, v41_heads_per_block=16))


@pytest.mark.parametrize("mode", ("decode", "extend"))
def test_compressed_mla_rejects_unaligned_q_before_preparation(mode):
    import torch
    from b12x.attention import compressed_sparse_mla as mla

    storage = torch.empty(3 * 16 * 512 + 1, dtype=torch.bfloat16)
    q = storage[1:].view(3, 16, 512)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Q requires 16-byte alignment"):
        mla.plan(
            mla.Caps(device="cpu", num_q_heads=16, max_q_rows=3, max_width=128,
                     swa_width=128, indexed_width=0, cache_format="deepseek_v41", mode=mode),
            invocation=mla.invocation_from_tensors(q=q, swa_k_cache=cache, out=torch.empty_like(q)),
        )
