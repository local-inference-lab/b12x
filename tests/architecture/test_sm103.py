from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

import b12x
from b12x._lib.architecture import architecture_for, UnsupportedArchitectureError
from b12x._lib import gating
from b12x.moe import fused_moe
from b12x.moe.fused_moe._policy import (
    MOE_DECODE_POLICY,
    MoeDecodeQuery,
    MoeDecodeConfig,
)
from b12x.moe.fused_moe._sm103 import BACKEND, scratch_layout, capacity_regime
from b12x.policy import DeviceIdentity, PolicyContext, PolicyMode, PolicySource

B300 = DeviceIdentity(
    vendor="nvidia",
    product_name="NVIDIA B300",
    compute_capability=(10, 3),
    sm_count=148,
)


def query(**kwargs):
    fields = dict(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        num_experts=288,
        hidden_size=4096,
        intermediate_size=2048,
        top_k=8,
        num_tokens=8,
        routed_rows=64,
    )
    return MoeDecodeQuery(**(fields | kwargs))


def test_architecture_features_and_unknown_device():
    arch = architecture_for((10, 3))
    assert arch.has_tmem and arch.tmem_columns == 512
    assert arch.compilation_target == "sm_103a"
    assert arch.max_smem_per_block == 227 * 1024
    assert arch.mma_family == "tcgen05"
    assert not architecture_for((12, 0)).has_tmem
    assert not architecture_for((12, 1)).has_tmem
    assert not architecture_for((10, 0)).implemented
    assert architecture_for((10, 9)) is None


@pytest.mark.parametrize(
    "capability,recognized,default",
    [
        ((10, 3), True, False),
        ((12, 0), True, True),
        ((12, 1), True, True),
        ((10, 0), False, False),
        (None, False, False),
    ],
)
def test_op_gates_are_distinct_from_architecture_recognition(
    monkeypatch, capability, recognized, default
):
    monkeypatch.setattr(
        gating, "get_compute_capability", lambda device=None: capability
    )
    monkeypatch.setattr(gating, "has_cutlass_dsl", lambda: True)
    monkeypatch.setattr(gating, "has_triton", lambda: True)
    assert gating.is_b12x() is recognized
    assert gating.default_is_supported() is default
    assert fused_moe.is_supported() is recognized


def test_identity_normalizes_and_preserves_sm103():
    assert replace(B300, product_name=" NVIDIA   B300 ") == B300
    assert asdict(B300)["compute_capability"] == (10, 3)
    policy = PolicyContext.for_identity(B300)
    assert policy.profile_id is None
    resolution = policy.resolve(MOE_DECODE_POLICY, query())
    assert resolution.source is PolicySource.HEURISTIC
    assert resolution.config.backend == BACKEND
    assert resolution.device == B300


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 256, 8192, 65536])
def test_policy_capacity_arms(tokens):
    q = query(num_tokens=tokens, routed_rows=tokens * 8)
    result = PolicyContext.for_identity(B300).resolve(MOE_DECODE_POLICY, q)
    assert result.config.backend == BACKEND
    assert capacity_regime(tokens) in {"m1", "m2_m4", "m5_m8", "m9_m32", "prefill"}


@pytest.mark.parametrize(
    "changes",
    [
        dict(quant_mode="w4a16"),
        dict(source_format="b12x_trellis"),
        dict(hidden_size=4128),
        dict(activation="relu2"),
    ],
)
def test_unimplemented_numeric_contracts_rejected(changes):
    with pytest.raises(UnsupportedArchitectureError):
        PolicyContext.for_identity(B300).resolve(MOE_DECODE_POLICY, query(**changes))


def test_sm120_override_never_selected_on_sm103():
    config = MoeDecodeConfig("micro", "internal", None)
    for mode in (PolicyMode.AUTO, PolicyMode.HEURISTIC_ONLY):
        with pytest.raises(UnsupportedArchitectureError):
            PolicyContext.for_identity(B300, mode=mode).resolve(
                MOE_DECODE_POLICY, query(), override=config
            )


@pytest.mark.parametrize("cache_format", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_sm103_compressed_mla_public_plan(cache_format, mode, sm103_context):
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._policy import SparseMlaConfig

    caps = mla.Caps(
        device="cuda:0", num_q_heads=20, max_q_rows=19, max_width=256,
        swa_width=128, indexed_width=128, cache_format=cache_format, mode=mode,
        swa_page_size=64, indexed_page_size=32,
    )
    plan = mla.plan(caps, policy=sm103_context)
    assert plan.policy_resolution.source is PolicySource.HEURISTIC
    assert plan.policy_resolution.config.backend == "warp"
    assert plan.policy_resolution.config.max_chunks_per_row == (4 if mode == "decode" else 1)
    assert plan.backend_key
    assert plan.layout.staged_selections_offset_bytes > 0
    with pytest.raises(ValueError, match="requires the warp backend"):
        mla.plan(caps, policy=sm103_context, config=SparseMlaConfig(max_chunks_per_row=4))


@pytest.mark.parametrize("cache_format", ["fp8", "mxfp4"])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_sm103_public_indexer_plan_selects_warp(cache_format, mode, sm103_context, monkeypatch):
    from b12x.attention import dsa_indexer
    from b12x.attention.dsa_indexer._policy import DsaIndexerConfig
    from b12x.policy import DSA_INDEXER

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(
        major=10, minor=3, multi_processor_count=148,
    ))
    caps = dsa_indexer.Caps(
        device="cuda:0", num_q_heads=32, max_q_rows=8,
        max_page_table_width=128, topk=512, cache_format=cache_format, mode=mode,
    )
    policy = sm103_context
    plan = dsa_indexer.plan(caps, policy=policy)
    assert plan.policy_resolution.config.backend == "warp"
    if cache_format == "fp8" and mode == "decode":
        assert plan.inner.layout.route == "paged_fused"
    assert plan.policy_resolution.source == PolicySource.HEURISTIC
    with pytest.raises(ValueError, match="requires the warp backend"):
        dsa_indexer.plan(caps, policy=policy.with_override(DSA_INDEXER, DsaIndexerConfig(backend="native")))


def test_sm103_indexer_generator_candidates_use_runtime_policy():
    from b12x.attention.dsa_indexer._policy import DSA_INDEXER_POLICY, DsaIndexerQuery
    from b12x.policy.generation.providers.tunable import _DsaIndexerMergeSession, _dsa_indexer_merge_cases

    session = _DsaIndexerMergeSession(SimpleNamespace(device=B300))
    case = _dsa_indexer_merge_cases()[0]
    candidates = session.candidates(case)
    assert {c.config["fused_merge"] for c in candidates} == {"cooperative", "serial"}
    for candidate in candidates:
        config = DSA_INDEXER_POLICY.decode_profile(candidate.config)
        assert config.backend == "warp"
        DSA_INDEXER_POLICY.validate_config(DsaIndexerQuery(**case.query), config, B300)


@pytest.mark.parametrize("recipe", ("kda_decode", "qwen_decode", "kda_prefill", "gdn_prefill", "dense_mla"))
def test_portable_cute_public_plans_on_sm103(sm103_context, monkeypatch, recipe):
    from b12x.sequence import gdn_decode, gdn_prefill, kda_prefill
    from b12x.attention import dense_mla

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        multi_processor_count=B300.sm_count, shared_memory_per_block_optin=227 * 1024))
    if recipe.endswith("decode"):
        heads = 4 if recipe == "kda_decode" else 12
        plan = gdn_decode.plan(gdn_decode.Caps(device="cuda:0", max_tokens=16,
            max_seqs=4, max_state_slots=17, key_heads=4, value_heads=heads,
            state_index_columns=4, gate_activation="sigmoid"), policy=sm103_context)
        assert plan.config.backend == "cutedsl"
    elif recipe == "dense_mla":
        plan = dense_mla.plan(dense_mla.Caps(device="cuda:0", mode="decode",
            kv_dtype=torch.bfloat16, num_q_heads=16, page_size=64, max_total_q=4,
            max_batch=4, max_cache_tokens=4096, max_page_table_width=64,
            num_cache_pages=128), policy=sm103_context)
    else:
        op = kda_prefill if recipe == "kda_prefill" else gdn_prefill
        heads = dict(heads=16) if recipe == "kda_prefill" else dict(key_heads=4, value_heads=12)
        plan = op.plan(op.Caps(device="cuda:0", max_tokens=4096, max_seqs=16,
            max_state_slots=129, **heads), policy=sm103_context)
    assert plan.policy_resolution.device == B300
    assert plan.policy_resolution.source is PolicySource.HEURISTIC
    assert plan.scratch_specs()


def test_sm103_kda_policy_rejects_triton_override(sm103_context):
    from b12x.sequence.gdn_decode._policy import GDN_POLICY, GdnQuery, GdnConfig

    query = GdnQuery(gate_activation="sigmoid", qk_l2norm=True, state_dtype="float32",
                     key_heads=16, value_heads=16, max_seqs=4, max_tokens=16,
                     state_index_columns=4)
    with pytest.raises(ValueError, match="SM103 recurrent compute requires"):
        sm103_context.resolve(GDN_POLICY, query,
                              override=GdnConfig(backend="triton", recurrent_block_v=32))


def test_sm103_gdn_generator_uses_cute_candidates():
    from b12x.policy.generation.attention_corpus import gdn_cases
    from b12x.policy.generation.providers.gpu_workers import GdnBenchmarkFactory

    case = next(c for c in gdn_cases() if c.metadata["decay_recipe"] == "kda")
    session = GdnBenchmarkFactory()(case.group_id, (case,), SimpleNamespace(device=B300))
    candidates = tuple(c.config.to_dict() for c in session.candidates(case))
    assert candidates == tuple(dict(backend="cutedsl", recurrent_block_v=v) for v in (16, 32))


def test_kda_qualification_converts_legacy_case_override_before_sm103_validation(sm103_context, monkeypatch):
    from b12x.policy import GDN_ATTENTION
    from b12x.sequence import gdn_decode
    from tests.sequence.test_gdn_decode_kda_cute import cute_backend

    cute_backend.__wrapped__(monkeypatch)
    caps = gdn_decode.Caps(device="cuda:0", max_tokens=16, max_seqs=4,
        max_state_slots=17, key_heads=4, value_heads=4,
        state_index_columns=4, gate_activation="sigmoid")
    legacy = sm103_context.with_override(GDN_ATTENTION,
        gdn_decode.GdnConfig(backend="triton", recurrent_block_v=16))
    planned = gdn_decode.plan(caps, policy=legacy)
    assert planned.config == gdn_decode.GdnConfig(backend="cutedsl", recurrent_block_v=16)


def test_scratch_layout_disjoint_and_model_size_independent():
    buffers, nbytes = scratch_layout(8, 8, 4096, 2048)
    assert all(b.offset % 1024 == 0 for b in buffers)
    for a, b in zip(buffers[:-1], buffers[1:], strict=True):
        assert a.offset + a.nbytes <= b.offset
    assert buffers[-1].offset + buffers[-1].nbytes <= nbytes
    assert next(b for b in buffers if b.name == "fc1").shape == (64, 4096)
    assert nbytes < 16 * 1024 * 1024


@pytest.fixture
def sm103_context(monkeypatch):
    import b12x.policy.context as context

    monkeypatch.setattr(
        context,
        "detect_device",
        lambda device: SimpleNamespace(identity=B300, ordinal=None),
    )
    return PolicyContext.for_identity(B300)


@pytest.mark.parametrize("model,dim", [(1, 576), (2, 512)])
@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_sm103_sparse_mla_plan_retains_warp_policy(sm103_context, model, dim, mode):
    from b12x.attention import sparse_mla
    from b12x.attention.sparse_mla._policy import SparseMlaConfig
    from b12x.policy import SPARSE_MLA_ATTENTION

    caps = sparse_mla.Caps(
        device="cuda:0", num_q_heads=24, max_q_rows=8, max_width=2051,
        softmax_scale=256**-0.5, kv_dtype=torch.uint8, head_dim=dim,
        v_head_dim=512, model_type=model, mode=mode,
    )
    plan = sparse_mla.plan(caps, policy=sm103_context)
    resolution = plan.policy_resolution
    assert resolution.source is PolicySource.HEURISTIC
    assert resolution.device == B300
    assert resolution.config == SparseMlaConfig(backend="warp", num_splits=4 if mode == "decode" else 1)
    with pytest.raises(ValueError, match="requires the warp backend"):
        sparse_mla.plan(caps, policy=sm103_context.with_override(
            SPARSE_MLA_ATTENTION, SparseMlaConfig(backend="native"),
        ))


def make_experts(*, device="cpu", e=2, k=256, n=256, w13_layout="w13", mode="a4"):
    wp = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout=w13_layout),
        activation=fused_moe.ActivationSpec(
            mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=e, hidden_size=k, intermediate_size=n
        ),
    )
    weights = fused_moe.PackedWeights(
        w13=torch.zeros((e, 2 * n, k // 2), dtype=torch.uint8, device=device),
        w2=torch.zeros((e, k, n // 2), dtype=torch.uint8, device=device),
        w13_block_scales=torch.ones((e, 2 * n, k // 16), device=device).to(
            torch.float8_e4m3fn
        ),
        w2_block_scales=torch.ones((e, k, n // 16), device=device).to(
            torch.float8_e4m3fn
        ),
        w13_global_scales=torch.ones(e, device=device),
        w2_global_scales=torch.ones(e, device=device),
        input_scale=torch.ones(e, device=device),
        intermediate_scale=torch.ones(e, device=device),
    )
    return fused_moe.prepare_weights(plan=wp, weights=weights)


def test_both_public_planners_preserve_storage_and_backend(sm103_context, monkeypatch):
    experts = make_experts()
    execution = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8, top_k=2, warmup_token_counts=(1, 4)
        ),
        policy=sm103_context,
    )
    assert all(v.implementation == BACKEND for v in execution.variants)
    assert execution.variant_for(1).execution.gemm_engine.value == "nvfp4_tcgen05"
    assert execution.variant_for(4).execution.graph_partition.value == "materialized"
    with pytest.raises(RuntimeError, match="prewarmed"):
        fused_moe.bind(execution)
    caps = fused_moe.Caps(
        max_tokens=8,
        num_topk=2,
        device="cpu",
        weight_plan=experts.plan._impl,
        quant_mode="nvfp4",
        policy_context=sm103_context,
    )
    from b12x.moe.fused_moe import _impl

    impl = _impl.plan_tp_moe_scratch(caps, prewarm_launches=False)
    assert isinstance(impl, fused_moe.Plan)
    assert (
        fused_moe.required_nbytes(caps)
        == execution.scratch.nbytes
        == impl.scratch_specs()[0].nbytes
    )
    assert experts.plan.prepared_format.packing.value == "source_native"
    assert impl.launch_plan.policy_resolution.config.backend == BACKEND


def test_binding_capacity_reuse_no_policy_or_compile(sm103_context, monkeypatch):
    from b12x.moe._shared.kernels.sm103 import launch

    calls = []

    def dummy(*args):
        calls.append(args)

    def fake_compile(caps):
        import cutlass

        result = {
            (name, dtype): dummy
            for name in ("fc1", "fc2", "q1", "q2")
            for dtype in (cutlass.Int32, cutlass.Int64)
        }
        return result | {"sum": dummy}

    monkeypatch.setattr(launch, "compile_launches", fake_compile)
    experts = make_experts()
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=8, top_k=2),
        policy=sm103_context,
    )
    fused_moe.prewarm(plan)
    scratch = torch.empty(plan.scratch.nbytes, dtype=torch.uint8)
    monkeypatch.setattr(
        PolicyContext,
        "resolve",
        lambda *args, **kwargs: pytest.fail("bind resolved policy"),
    )
    monkeypatch.setattr(
        launch, "compile_launches", lambda *args: pytest.fail("bind compiled")
    )
    b12x.freeze_kernel_resolution("SM103 contract")
    try:
        for m in (1, 4, 8, 2):
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                experts=experts,
                a=torch.ones((m, 256), dtype=torch.bfloat16),
                topk_ids=torch.zeros((m, 2), dtype=torch.int32),
                topk_weights=torch.ones((m, 2)),
            )
            assert binding._backend_binding is not None
            assert binding.output.data_ptr() >= scratch.data_ptr()
            assert binding.output.shape == (m, 256)
    finally:
        b12x.unfreeze_kernel_resolution()
    assert not calls  # Binding only constructs views and launch arguments.


@pytest.mark.parametrize("mode", ["a4", "auto"])
@pytest.mark.parametrize("override", [False, True])
def test_capacity_resolves_once_and_prewarm_retains_provenance(
    sm103_context, monkeypatch, mode, override
):
    from b12x.moe._shared.kernels.sm103 import launch
    from b12x.moe._shared.kernels.w4a16 import prepare

    # AUTO preparation retains an A16 workspace. This host test exercises
    # planning and provenance; CUDA allocation is qualified by the GPU tests.
    monkeypatch.setattr(
        prepare, "_make_workspace", lambda *args, **kwargs: torch.zeros(8, dtype=torch.int32)
    )

    policy = sm103_context
    if override:
        policy = policy.with_override("moe.decode", MoeDecodeConfig(BACKEND, "internal", None))
    resolved = []
    resolve = PolicyContext.resolve

    def record(context, component, query, **kwargs):
        result = resolve(context, component, query, **kwargs)
        resolved.append((query, result))
        return result

    monkeypatch.setattr(PolicyContext, "resolve", record)
    plan = fused_moe.plan_execution(
        experts=make_experts(mode=mode),
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8, top_k=2, warmup_token_counts=(1, 4)
        ),
        policy=policy,
    )
    assert len(resolved) == 1
    query, resolution = resolved[0]
    assert (query.num_tokens, query.routed_rows) == (8, 16)
    assert query.quant_mode == ("nvfp4_auto" if mode == "auto" else "nvfp4")
    assert resolution.source is (PolicySource.OVERRIDE if override else PolicySource.HEURISTIC)
    lowering = plan._impl.launch_plan
    assert lowering.policy_resolution is resolution
    assert all(variant._impl is lowering for variant in plan.variants)
    if mode == "auto":
        assert plan.precision_resolution is resolution

    monkeypatch.setattr(
        PolicyContext, "resolve", lambda *args, **kwargs: pytest.fail("prewarm resolved policy")
    )
    compilations = []
    launches = {}

    def compile_capacity(caps):
        compilations.append(caps.max_tokens)
        return launches

    monkeypatch.setattr(launch, "compile_launches", compile_capacity)
    specs = plan.scratch_specs()
    fused_moe.prewarm(plan)
    fused_moe.prewarm(plan)
    assert compilations == [8]
    assert plan._impl.launch_plan is lowering
    assert plan.scratch_specs() == specs
    assert plan._impl._backend_plan.launches is launches


def test_compile_gate_rejects_sm120_kernel_before_compiler_resolution(monkeypatch):
    from b12x._lib import compiler

    class WarpKernel:
        __module__ = "b12x.moe._shared.kernels.dynamic"

    monkeypatch.setattr(gating, "get_compute_capability", lambda: (10, 3))
    with pytest.raises(UnsupportedArchitectureError, match="no admitted SM103"):
        compiler.compile(WarpKernel())


def test_sm103_profile_candidate_roundtrip_and_coverage():
    from b12x.policy.generation.providers.moe import _config_covers_query
    from b12x.policy.types import FrozenMapping

    config = PolicyContext.for_identity(B300).resolve(MOE_DECODE_POLICY, query()).config
    serialized = asdict(config)
    assert MoeDecodeConfig.from_profile(FrozenMapping(serialized)) == config
    assert _config_covers_query(asdict(query()), serialized)
    assert not _config_covers_query(
        asdict(query(source_format="b12x_trellis")), serialized
    )


def test_v41_trellis_pipeline_consumes_canonical_weight_plan():
    from b12x.moe._shared.kernels.sm103.trellis import TrellisPipeline
    from tests.moe.test_trellis_config import _k3_config

    source = fused_moe.TrellisConfig.from_dict(_k3_config())
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=384, hidden_size=5120, intermediate_size=2304
        ),
    )
    pipeline = TrellisPipeline.from_weight_plan(plan._impl)
    assert pipeline.coupled_hadamard
    assert pipeline.reconstruction(projection="w2", bits=3).capacity == 384 * 320 * 144
    assert (
        pipeline.reconstruction(projection="w13", bits=4).capacity
        == 2 * 384 * 320 * 144
    )
    from b12x.moe.fused_moe._sm103_trellis import validate_weight_plan
    with pytest.raises(UnsupportedArchitectureError, match="SiTU"):
        validate_weight_plan(plan._impl)


def test_sm103_generator_filters_unsupported_recipes_and_route_capacities():
    from b12x.policy.generation.providers.moe import MoeDecodeGenerator
    from b12x.policy.generation.providers.moe_gpu_worker import _candidates_for_geometry

    generator = MoeDecodeGenerator()
    context = SimpleNamespace(device=B300)
    selected = generator._for_device(context)
    assert selected is not generator
    assert 0 < len(selected._geometries) < len(generator._geometries)
    assert all(
        (g.recipe.quant_mode == "nvfp4"
         and g.hidden_size % 256 == 0 and g.intermediate_size % 256 == 0)
        or (g.recipe.trellis_variant == "k3-sqg-uniform-coupled"
            and g.activation == "situ" and g.hidden_size % 512 == 0
            and g.intermediate_size % 128 == 0)
        or (g.recipe.trellis_variant == "glm-mcg-projection-tiered"
            and g.activation in {"silu", "situ"} and g.hidden_size % 128 == 0
            and g.intermediate_size % 128 == 0)
        for g in selected._geometries
    )
    assert selected._for_device(context) is selected
    assert generator.estimate(context) == selected.estimate(context)
    for g in selected._geometries:
        candidates = _candidates_for_geometry(
            g, sm_count=B300.sm_count, compute_capability=(10, 3)
        )
        expected = BACKEND if g.recipe.quant_mode == "nvfp4" else "tcgen05_trellis"
        assert len(candidates) == 1 and candidates[0].config["backend"] == expected


def test_sm103_route_count_rejects_int32_overflow():
    with pytest.raises(UnsupportedArchitectureError, match="Int32 launch-count"):
        PolicyContext.for_identity(B300).resolve(
            MOE_DECODE_POLICY, query(num_tokens=2**28, routed_rows=2**31)
        )


def test_public_capability_queries_keep_operation_coverage(monkeypatch):
    import b12x
    from b12x._lib import gating
    from b12x.gemm import mxfp8_linear

    assert b12x.supports_architecture((10, 3), mxfp8_linear.META.archs)
    assert not b12x.supports_architecture((10, 3), b12x.find_op('attention.paged').archs)
    monkeypatch.setattr(gating, 'get_compute_capability', lambda *args: (10, 3))
    monkeypatch.setattr(gating, 'has_cutlass_dsl', lambda: True)
    monkeypatch.setattr(gating, 'has_triton', lambda: True)
    assert mxfp8_linear.is_supported()
