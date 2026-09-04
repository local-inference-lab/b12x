from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace

import b12x.policy.generation.attention_corpus as attention_corpus_module
from benchmarks.benchmark_gdn_decode import QWEN38_GDN_CASES
from benchmarks.benchmark_paged_attention import BENCHMARK_PROFILES
from benchmarks.benchmark_qsa import PROFILES as QSA_PROFILES
from b12x.policy import (
    EMBEDDED_REGISTRY,
    DeviceIdentity,
    PolicyContext,
    PolicyMode,
    PolicySource,
    list_profiled_components,
    profile_from_dict,
)
from b12x.policy.generation import (
    CheckpointStore,
    GenerationContext,
    GenerationSettings,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)
from b12x.policy.generation.attention_corpus import (
    ATTENTION_BENCHMARK_PRESETS,
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_SEQUENCE_CAPACITIES,
    GDN_GEOMETRIES,
    GQA_GEOMETRIES,
    GLM53_TP3_KDA_PROFILE_IDS,
    GLM53_TP3_KDA_SERVING_CASES,
    MLA_GEOMETRIES,
    QSA_GEOMETRIES,
    SPARSE_MLA_GEOMETRIES,
    attention_corpus_manifest,
    gdn_cases,
    gqa_cases,
    mla_cases,
    qsa_cases,
    sparse_mla_cases,
)
from b12x.policy.generation.providers import register_builtin_generators
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers.attention import (
    GdnAttentionGenerator,
    _QsaProbe,
)
from b12x.policy.generation.providers.gpu_workers import GdnBenchmarkFactory
from b12x.policy.generation.providers.qualification import (
    _DsaIndexerProbe,
    DsaIndexerGenerator,
    SparseMlaGenerator,
)
from b12x.policy.generation.providers.norm_sequence import (
    MhcGenerator,
    _MhcSession,
    _hyperconnection_cases,
    _mhc_cases,
    _mtp_feedback_cases,
)
from b12x.policy.generation.registry import ComponentGeneratorRegistry
from b12x.sequence.gdn_decode._policy import GDN_POLICY, GdnQuery


class _FixedGdnSession(AbstractContextManager["_FixedGdnSession"]):
    def __init__(self, benchmarked: list[SweepCase]) -> None:
        self._benchmarked = benchmarked

    def __enter__(self) -> "_FixedGdnSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def candidates(self, _case):
        return (SweepCandidate.create({"backend": "triton", "recurrent_block_v": 32}),)

    def measure(self, case, candidates):
        self._benchmarked.append(case)
        return (
            SweepMeasurement(
                candidate=candidates[0],
                latency_us=1.0,
                correct=True,
            ),
        )


class _FixedGdnFactory:
    def __init__(self) -> None:
        self.benchmarked: list[SweepCase] = []

    def __call__(self, _group_id, _cases, _context):
        return _FixedGdnSession(self.benchmarked)


def test_builtin_registry_covers_every_top_level_component() -> None:
    registry = ComponentGeneratorRegistry()

    register_builtin_generators(registry)

    assert registry.component_ids() == tuple(
        str(item.component_id) for item in list_profiled_components()
    )


def test_attention_corpora_have_stable_reviewed_cross_products() -> None:
    assert len(GDN_GEOMETRIES) == 21
    assert len(GQA_GEOMETRIES) == 18
    assert len(MLA_GEOMETRIES) == 1
    assert len(QSA_GEOMETRIES) == 3
    assert len(SPARSE_MLA_GEOMETRIES) == 12
    assert len(gdn_cases()) == 1_468
    assert len(gqa_cases()) == 14_400
    assert len(mla_cases()) == 200
    assert len(qsa_cases()) == 6_348
    assert len(sparse_mla_cases()) == 288
    assert len({case.query for case in gqa_cases()}) == len(gqa_cases())

    all_cases = (
        *gdn_cases(),
        *gqa_cases(),
        *mla_cases(),
        *qsa_cases(),
        *sparse_mla_cases(),
    )
    assert len({case.case_id for case in all_cases}) == len(all_cases)


def test_gdn_corpus_includes_qwen_and_glm_decay_contracts() -> None:
    cases = gdn_cases()
    recipes = {case.metadata["decay_recipe"] for case in cases}
    glm_cases = [case for case in cases if case.metadata["decay_recipe"] == "kda"]

    assert recipes == {"gdn", "kda"}
    assert len(glm_cases) == 816
    assert {case.query["key_heads"] for case in glm_cases} == {4, 8, 16, 22, 32, 64}
    assert all(
        case.query["key_heads"] == case.query["value_heads"] for case in glm_cases
    )
    glm_tp4_capacities = {
        (
            case.query["max_seqs"],
            case.query["max_tokens"],
            case.query["state_index_columns"],
        )
        for case in glm_cases
        if case.query["key_heads"] == 16
    }
    assert (16, 16, 1) in glm_tp4_capacities
    assert (16, 96, 6) in glm_tp4_capacities
    tp3_cases = [
        case
        for case in glm_cases
        if case.metadata.get("model_id") == "glm-5.3-flash-kda-tp3"
    ]
    assert [
        (
            case.metadata["serving_mode"],
            case.query["max_seqs"],
            case.query["max_tokens"],
            case.query["state_index_columns"],
            case.query["key_heads"],
        )
        for case in tp3_cases
    ] == [(*serving_case, 22) for serving_case in GLM53_TP3_KDA_SERVING_CASES]
    assert {tuple(case.metadata["profile_ids"]) for case in tp3_cases} == {
        GLM53_TP3_KDA_PROFILE_IDS
    }
    exercised = {
        case.query
        for case in cases
        if max(case.metadata["query_lengths"]) == int(case.query["state_index_columns"])
    }
    assert exercised == {case.query for case in cases}


def test_embedded_gdn_profiles_cover_every_corpus_query() -> None:
    cases_by_query = {case.query: case for case in gdn_cases()}

    for profile in EMBEDDED_REGISTRY.list_profiles():
        component = profile.component("attention.gdn")
        assert component is not None, profile.profile_id
        for query, case in cases_by_query.items():
            applicable_profiles = case.metadata.get("profile_ids")
            if (
                applicable_profiles is not None
                and profile.profile_id not in applicable_profiles
            ):
                continue
            hit = component.lookup(query)
            assert hit is not None, (profile.profile_id, query.to_dict())
            expected_backend = (
                "triton" if case.metadata["decay_recipe"] == "kda" else "cutedsl"
            )
            assert hit.config["backend"] == expected_backend, (
                profile.profile_id,
                query.to_dict(),
                hit.config,
            )


def test_embedded_norm_sequence_profiles_cover_every_corpus_query() -> None:
    component_cases = {
        "norm.hyperconnection": _hyperconnection_cases(),
        "sequence.mtp_feedback": _mtp_feedback_cases(),
    }

    for profile in EMBEDDED_REGISTRY.list_profiles():
        for component_id, cases in component_cases.items():
            component = profile.component(component_id)
            assert component is not None, (profile.profile_id, component_id)
            for case in cases:
                hit = component.lookup(case.query)
                assert hit is not None, (
                    profile.profile_id,
                    component_id,
                    case.query.to_dict(),
                )
                assert hit.config["backend"] == "cutedsl"


def test_attention_capacity_axes_cover_serving_and_prefill_buckets() -> None:
    expected_sequence_capacities = (
        *range(1, 17),
        32,
        64,
        128,
        256,
    )
    assert expected_sequence_capacities == COMMON_SEQUENCE_CAPACITIES
    assert COMMON_PREFILL_TOKEN_CAPACITIES == (1_024, 2_048, 4_096, 8_192)

    for cases in (_hyperconnection_cases(), _mtp_feedback_cases()):
        capacities = {int(case.query["max_tokens"]) for case in cases}
        assert set(COMMON_SEQUENCE_CAPACITIES) <= capacities
        assert {512, *COMMON_PREFILL_TOKEN_CAPACITIES} <= capacities

    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["max_q_rows"])
        for case in qsa_cases()
        if int(case.query["max_q_rows"]) >= 1_024
    }
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["query_rows"])
        for case in mla_cases()
        if case.query["mode"] == "extend"
    }
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["query_rows"]) for case in sparse_mla_cases()
    }

    qsa_prefill_rows = {
        rows
        for _profile, rows, _context, _dtype, kind in _QsaProbe._CASES
        if kind == "prefill"
    }
    assert qsa_prefill_rows == set(COMMON_PREFILL_TOKEN_CAPACITIES)
    assert {
        (profile, rows, kv_dtype)
        for profile, rows, _context, kv_dtype, kind in _QsaProbe._CASES
        if kind == "prefill"
    } == {
        (profile, rows, kv_dtype)
        for profile in ("tp1", "tp2", "tp4")
        for rows in COMMON_PREFILL_TOKEN_CAPACITIES
        for kv_dtype in ("bf16", "fp8_e4m3")
    }
    assert {
        int(case[0].removeprefix("glm52-extend-m"))
        for case in _DsaIndexerProbe._CASES
        if case[0].startswith("glm52-extend-m")
    } == set(COMMON_PREFILL_TOKEN_CAPACITIES)


def test_gdn_backend_identifies_decay_contract_from_head_geometry() -> None:
    common = {
        "gate_activation": "sigmoid",
        "qk_l2norm": True,
        "state_dtype": "float32",
        "max_seqs": 1,
        "max_tokens": 4,
        "state_index_columns": 4,
    }

    qwen = GdnQuery(key_heads=8, value_heads=24, **common)
    glm = GdnQuery(key_heads=8, value_heads=8, **common)

    assert GDN_POLICY.heuristic(qwen, None).backend == "cutedsl"
    assert GDN_POLICY.heuristic(glm, None).backend == "triton"
    assert GDN_POLICY.heuristic(qwen, None).recurrent_block_v == 32
    assert GDN_POLICY.heuristic(glm, None).recurrent_block_v == 32


def test_rtx_pro_6000_profile_covers_glm_5_3_kda_serving_capacities() -> None:
    """Require planned tiles for four-way tensor-parallel serving graphs."""

    device = DeviceIdentity(
        vendor="NVIDIA",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    )
    profile = EMBEDDED_REGISTRY.get("nvidia.rtx.pro.6000.blackwell")
    component = profile.component("attention.gdn")
    assert component is not None

    serving_capacities = (
        (16, 16, 1),  # One target token without speculative decoding.
        (16, 64, 4),  # Target plus three multi-token-prediction draft tokens.
        (16, 128, 8),  # One target token plus seven DFlash2 draft tokens.
    )
    for max_seqs, max_tokens, state_index_columns in serving_capacities:
        query = GdnQuery(
            gate_activation="sigmoid",
            qk_l2norm=True,
            state_dtype="float32",
            key_heads=16,
            value_heads=16,
            max_seqs=max_seqs,
            max_tokens=max_tokens,
            state_index_columns=state_index_columns,
        )
        leaf = component.lookup(query.profile_fields())
        resolution = PolicyContext.for_identity(
            device,
            mode=PolicyMode.PREPLANNED_ONLY,
        ).resolve(GDN_POLICY, query)

        assert leaf is not None
        assert leaf.config["backend"] == "triton"
        assert leaf.config["recurrent_block_v"] == 16
        assert resolution.source is PolicySource.PREPLANNED
        assert resolution.config.backend == "triton"
        assert resolution.config.recurrent_block_v == 16

    other_device = DeviceIdentity(
        vendor="NVIDIA",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="Synthetic RTX",
    )
    assert GDN_POLICY.heuristic(query, other_device).recurrent_block_v == 32


def test_rtx_pro_6000_profile_preplans_glm_5_3_tp3_kda_capacities() -> None:
    device = DeviceIdentity(
        vendor="NVIDIA",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    )
    component = EMBEDDED_REGISTRY.get("nvidia.rtx.pro.6000.blackwell").component(
        "attention.gdn"
    )
    assert component is not None
    expected_block_v = {
        "ordinary": 16,
        "mtp3": 32,
        "dflash7": 32,
        "ordinary-c8": 16,
        "mtp3-c8": 16,
        "dflash7-c8": 16,
    }

    for mode, max_seqs, max_tokens, state_index_columns in GLM53_TP3_KDA_SERVING_CASES:
        query = GdnQuery(
            gate_activation="sigmoid",
            qk_l2norm=True,
            state_dtype="float32",
            key_heads=22,
            value_heads=22,
            max_seqs=max_seqs,
            max_tokens=max_tokens,
            state_index_columns=state_index_columns,
        )
        leaf = component.lookup(query.profile_fields())
        resolution = PolicyContext.for_identity(
            device,
            mode=PolicyMode.PREPLANNED_ONLY,
        ).resolve(GDN_POLICY, query)

        assert leaf is not None
        assert leaf.config["backend"] == "triton"
        assert leaf.config["recurrent_block_v"] == expected_block_v[mode]
        assert resolution.source is PolicySource.PREPLANNED
        assert resolution.config.backend == "triton"
        assert resolution.config.recurrent_block_v == expected_block_v[mode]


def test_generated_gdn_profile_covers_dense_and_sparse_capacity_ranges(
    tmp_path,
) -> None:
    cases = tuple(
        case
        for case in gdn_cases()
        if case.metadata["decay_recipe"] == "kda" and case.query["key_heads"] == 16
    )
    generator = GdnAttentionGenerator(
        benchmark_factory=_FixedGdnFactory(),
        cases=cases,
    )
    device = DeviceIdentity(
        vendor="nvidia",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="Synthetic RTX",
    )
    context = GenerationContext(
        device=device,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="test",
        settings=GenerationSettings(),
    )
    result = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )
    profile = profile_from_dict(
        {
            "profile_id": "synthetic",
            "targets": [
                {
                    "vendor": device.vendor,
                    "compute_capability": list(device.compute_capability),
                    "sm_count": device.sm_count,
                    "product_name": device.product_name,
                }
            ],
            "components": [result.component],
        }
    )
    component = profile.component("attention.gdn")
    assert component is not None

    for max_seqs, columns in ((16, 1), (24, 1), (24, 6), (256, 8)):
        leaf = component.lookup(
            {
                "gate_activation": "sigmoid",
                "qk_l2norm": True,
                "state_dtype": "float32",
                "key_heads": 16,
                "value_heads": 16,
                "max_seqs": max_seqs,
                "max_tokens": max_seqs * columns,
                "state_index_columns": columns,
            }
        )
        assert leaf is not None
        assert leaf.config["backend"] == "triton"
        assert leaf.config["recurrent_block_v"] == 32

    assert (
        component.lookup(
            {
                "gate_activation": "sigmoid",
                "qk_l2norm": True,
                "state_dtype": "float32",
                "key_heads": 16,
                "value_heads": 16,
                "max_seqs": 257,
                "max_tokens": 257,
                "state_index_columns": 1,
            }
        )
        is None
    )


def test_gdn_generator_filters_profile_specific_cases_before_racing(
    tmp_path,
) -> None:
    restricted_cases = tuple(
        case
        for case in gdn_cases()
        if case.metadata.get("model_id") == "glm-5.3-flash-kda-tp3"
    )
    unrestricted_case = next(
        case
        for case in gdn_cases()
        if case.query["key_heads"] == 16 and case.metadata.get("profile_ids") is None
    )
    cases = (*restricted_cases, unrestricted_case)
    device = DeviceIdentity(
        vendor="nvidia",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="Synthetic Blackwell",
    )
    generated = {}
    for profile_id in (
        GLM53_TP3_KDA_PROFILE_IDS[0],
        "nvidia.gb10.48sm",
    ):
        work_dir = tmp_path / profile_id
        factory = _FixedGdnFactory()
        generator = GdnAttentionGenerator(
            benchmark_factory=factory,
            cases=cases,
        )
        context = GenerationContext(
            device=device,
            device_ordinal=0,
            work_dir=work_dir,
            source_revision="test",
            settings=GenerationSettings(),
            profile_id=profile_id,
        )
        estimate = generator.estimate(context)
        result = generator.generate(
            context,
            progress=NullProgressReporter(),
            checkpoints=CheckpointStore(work_dir / "checkpoints"),
        )
        profile = profile_from_dict(
            {
                "profile_id": profile_id,
                "targets": [
                    {
                        "vendor": device.vendor,
                        "compute_capability": list(device.compute_capability),
                        "sm_count": device.sm_count,
                        "product_name": device.product_name,
                    }
                ],
                "components": [result.component],
            }
        )
        component = profile.component("attention.gdn")
        assert component is not None
        generated[profile_id] = (estimate, factory, component)

    rtx_estimate, rtx_factory, rtx_component = generated[GLM53_TP3_KDA_PROFILE_IDS[0]]
    gb10_estimate, gb10_factory, gb10_component = generated["nvidia.gb10.48sm"]
    assert rtx_estimate.case_count == len(cases)
    assert gb10_estimate.case_count == 1
    assert {case.case_id for case in rtx_factory.benchmarked} == {
        case.case_id for case in cases
    }
    assert {case.case_id for case in gb10_factory.benchmarked} == {
        unrestricted_case.case_id
    }
    assert rtx_component.lookup(unrestricted_case.query) is not None
    assert gb10_component.lookup(unrestricted_case.query) is not None
    for case in restricted_cases:
        assert case.query["key_heads"] == 22
        assert rtx_component.lookup(case.query) is not None
        assert gb10_component.lookup(case.query) is None


def test_gdn_generator_invalidates_version_one_candidate_checkpoint(
    tmp_path,
) -> None:
    case = next(
        case
        for case in gdn_cases()
        if case.query["key_heads"] == 16 and case.metadata.get("profile_ids") is None
    )
    factory = _FixedGdnFactory()
    generator = GdnAttentionGenerator(
        benchmark_factory=factory,
        cases=(case,),
    )
    context = GenerationContext(
        device=DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 0),
            sm_count=188,
            product_name="Synthetic Blackwell",
        ),
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="test",
        settings=GenerationSettings(),
    )
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    stale_candidate = SweepCandidate.create({"backend": "triton"})
    checkpoints.save(
        generator.component_id,
        case.case_id,
        {
            "schema_version": 2,
            "candidate_contract_version": 1,
            "generation": context.checkpoint_metadata(),
            "case_id": case.case_id,
            "group_id": case.group_id,
            "query": case.query.to_dict(),
            "scenario": case.scenario,
            "metadata": case.metadata.to_dict(),
            "candidate_ids": [stale_candidate.candidate_id],
            "measurements": [
                SweepMeasurement(
                    candidate=stale_candidate,
                    latency_us=1.0,
                    correct=True,
                ).to_dict()
            ],
        },
    )

    generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )

    assert factory.benchmarked == [case]
    refreshed = checkpoints.load(generator.component_id, case.case_id)
    assert refreshed is not None
    assert refreshed["candidate_contract_version"] == 2
    assert refreshed["measurements"][0]["config"] == {
        "backend": "triton",
        "recurrent_block_v": 32,
    }


def test_gdn_benchmark_factory_accepts_grouped_capacity_cases() -> None:
    group_id = gdn_cases()[0].group_id
    cases = tuple(case for case in gdn_cases() if case.group_id == group_id)

    session = GdnBenchmarkFactory()(group_id, cases, object())

    assert len(cases) > 1
    assert session.candidates(cases[0])[0].config["backend"] == "cutedsl"
    assert session.candidates(cases[0])[0].config["recurrent_block_v"] == 32


def test_gdn_benchmark_factory_races_kda_recurrent_value_tiles() -> None:
    case = next(case for case in gdn_cases() if case.metadata["decay_recipe"] == "kda")
    session = GdnBenchmarkFactory()(case.group_id, (case,), object())

    assert tuple(
        candidate.config.to_dict() for candidate in session.candidates(case)
    ) == (
        {"backend": "triton", "recurrent_block_v": 16},
        {"backend": "triton", "recurrent_block_v": 32},
    )


def test_attention_corpus_manifests_are_content_addressed() -> None:
    expected_schema_versions = {
        "gdn": 2,
        "gqa": 1,
        "mla": 1,
        "qsa": 1,
        "sparse_mla": 1,
    }
    for component, schema_version in expected_schema_versions.items():
        manifest = attention_corpus_manifest(component)

        assert manifest["schema_version"] == schema_version
        assert len(manifest["corpus_sha256"]) == 64


def test_gdn_manifest_hash_tracks_serving_cases_and_profile_ids(monkeypatch) -> None:
    serving_cases = attention_corpus_module.GLM53_TP3_KDA_SERVING_CASES
    profile_ids = attention_corpus_module.GLM53_TP3_KDA_PROFILE_IDS
    original = attention_corpus_manifest("gdn")
    serving_group = "glm-5.3-flash-kda-tp3-serving"
    generated_cases = [case for case in gdn_cases() if case.group_id == serving_group]

    monkeypatch.setattr(
        attention_corpus_module,
        "GLM53_TP3_KDA_SERVING_CASES",
        (*serving_cases, ("future", 16, 256, 16)),
    )
    serving_changed = attention_corpus_manifest("gdn")
    monkeypatch.setattr(
        attention_corpus_module,
        "GLM53_TP3_KDA_SERVING_CASES",
        serving_cases,
    )
    monkeypatch.setattr(
        attention_corpus_module,
        "GLM53_TP3_KDA_PROFILE_IDS",
        (*profile_ids, "nvidia.synthetic"),
    )
    profiles_changed = attention_corpus_manifest("gdn")

    assert original["glm53_tp3_kda_serving_cases"] == [
        {
            "group_id": case.group_id,
            "query": case.query.to_dict(),
            "scenario": case.scenario,
            "metadata": case.metadata.to_dict(),
            "label": case.case_id.rsplit("-", 1)[0],
        }
        for case in generated_cases
    ]
    assert original["glm53_tp3_kda_profile_ids"] == list(profile_ids)
    assert serving_changed["corpus_sha256"] != original["corpus_sha256"]
    assert profiles_changed["corpus_sha256"] != original["corpus_sha256"]


def test_gdn_manifest_hash_tracks_fixed_serving_query_fields(monkeypatch) -> None:
    original_contract = attention_corpus_module._glm53_tp3_kda_serving_case_contract
    original = attention_corpus_manifest("gdn")

    def with_changed_key_heads(
        serving_mode: str,
        max_seqs: int,
        max_tokens: int,
        columns: int,
    ) -> dict[str, object]:
        contract = original_contract(serving_mode, max_seqs, max_tokens, columns)
        query = contract["query"]
        assert isinstance(query, dict)
        return {**contract, "query": {**query, "key_heads": 23}}

    monkeypatch.setattr(
        attention_corpus_module,
        "_glm53_tp3_kda_serving_case_contract",
        with_changed_key_heads,
    )
    changed = attention_corpus_manifest("gdn")
    changed_case = changed["glm53_tp3_kda_serving_cases"][0]
    assert isinstance(changed_case, dict)
    changed_query = changed_case["query"]
    assert isinstance(changed_query, dict)

    assert changed_query["key_heads"] == 23
    assert changed["corpus_sha256"] != original["corpus_sha256"]


def test_mhc_tuner_races_the_medium_prefill_plan() -> None:
    case = next(
        case
        for case in _mhc_cases()
        if case.query["hidden_size"] == 4_096 and case.query["max_tokens"] == 3_072
    )
    configs = tuple(
        candidate.config.to_dict()
        for candidate in _MhcSession(SimpleNamespace(device=None)).candidates(case)
    )

    assert any(config["backend"] == "native" for config in configs)
    assert any(
        config
        == {
            "backend": "tf32_tma",
            "decode_partials_schedule": "default",
            "projection_tile_m": 64,
            "projection_tile_n": 24,
            "projection_tile_k": 64,
            "projection_num_stages": 2,
            "projection_num_m_warps": 4,
            "projection_num_n_warps": 1,
            "projection_k_splits": 8,
        }
        for config in configs
    )


def test_mhc_tuner_races_profiled_decode_partial_grouping() -> None:
    case = next(
        case
        for case in _mhc_cases()
        if case.query["hidden_size"] == 4_096
        and case.query["max_tokens"] == 128
        and case.query["split_k"] == 64
    )
    configs = tuple(
        candidate.config.to_dict()
        for candidate in _MhcSession(SimpleNamespace(device=None)).candidates(case)
    )

    assert tuple(config["decode_partials_schedule"] for config in configs) == (
        "default",
        "hidden4096_m128_v1",
    )


def test_glm_profile_generation_envelope_matches_presets() -> None:
    dsa_queries = DsaIndexerGenerator().reviewed_queries()
    sparse_queries = SparseMlaGenerator().reviewed_queries()
    mhc_queries = MhcGenerator().reviewed_queries()

    assert any(
        query.num_q_heads == 32 and query.top_k == 2_048 for query in dsa_queries
    )
    assert any(query.num_q_heads == 32 and query.top_k == 512 for query in dsa_queries)
    assert {
        (query.qk_head_dim, query.v_head_dim, query.model_type)
        for query in sparse_queries
    } == {(576, 512, None), (512, 512, 2)}
    assert {query.num_q_heads for query in sparse_queries} == {8, 16, 32, 64}
    assert any(
        query.max_tokens == 6 and query.hidden_size == 4_096 and query.split_k == 64
        for query in mhc_queries
    )
    assert {4_096, 7_168} == {query.hidden_size for query in mhc_queries}
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        query.max_tokens for query in mhc_queries
    }
    assert {2_304, 3_072, 3_584} <= {query.max_tokens for query in mhc_queries}
    assert {query.score_mode for query in dsa_queries} == {"dsa", "msa"}
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        query.max_q_rows for query in dsa_queries if query.mode == "prefill"
    }
    assert set(COMMON_SEQUENCE_CAPACITIES) <= {
        query.max_q_rows for query in sparse_queries if query.mode == "decode"
    }


def test_named_attention_benchmark_presets_are_in_the_reviewed_inventory() -> None:
    preset_ids = {preset.preset_id for preset in ATTENTION_BENCHMARK_PRESETS}
    assert {
        name.removeprefix("paged:") for name in preset_ids if name.startswith("paged:")
    } == set(BENCHMARK_PROFILES)
    assert {
        name.removeprefix("qsa:") for name in preset_ids if name.startswith("qsa:")
    } == set(QSA_PROFILES)
    assert {
        name.removeprefix("gdn:") for name in preset_ids if name.startswith("gdn:")
    } == {case.name for case in QWEN38_GDN_CASES}
    assert preset_ids == {
        "compressed-mla:deepseek-v4-flash-default",
        "compressed-mla:vllm-dsv4-trace",
        "dense-mla:kimi-k3",
        "dsa-indexer:glm-5.1-default",
        "gdn:qk16-v48-decode-bs1",
        "gdn:qk2-v6-decode-bs1",
        "gdn:qk4-v12-decode-bs1",
        "gdn:qk8-v24-decode-bs1",
        "gdn:qk8-v24-decode-bs4",
        "gdn:qk8-v24-spec2-bs4",
        "gdn:qk8-v24-spec4-bs1",
        "gdn:qk8-v24-spec4-bs4",
        "gdn:qk8-v24-spec4-uneven",
        "mla:target-dsv4-trace",
        "mla:target-glm52-prefill4k-ctx16k",
        "mla:target-prefill64k-bs1",
        "mla:glm-5.2-default",
        "msa-indexer:minimax-m3-default",
        "paged-msa:minimax-m3-default",
        "paged:minimax-m2.7",
        "paged:qwen-gqa",
        "paged:qwen3.8-27b",
        "paged-indexer:deepseek-v4-flash-default",
        "qsa:tp1",
        "qsa:tp2",
        "qsa:tp4",
        "unified-mla:deepseek-v4-flash-decode",
        "unified-mla:deepseek-v4-flash-prefill",
        "unified-mla:glm-5.1-decode",
        "vllm-paged:minimax-m2.7",
        "vllm-paged:qwen-gqa",
    }
