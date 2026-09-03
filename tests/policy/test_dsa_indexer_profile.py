"""Host-side checks of the DSA indexer corpus, candidate menus and reduction."""

from __future__ import annotations

from contextlib import AbstractContextManager

from b12x.attention.dsa_indexer._policy import DSA_INDEXER_POLICY, DsaIndexerQuery
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.dsa_indexer_corpus import (
    DSA_INDEXER_CONTEXT_TOKENS,
    DSA_INDEXER_DECODE_ROWS,
    DSA_INDEXER_MODEL_LEN_TOKENS,
    DSA_INDEXER_RACED_GEOMETRIES,
    page_table_width,
    qualification_dsa_indexer_cases,
    raced_dsa_indexer_cases,
)
from b12x.policy.generation.providers.dsa_indexer import (
    AUTO_CONFIG,
    DsaIndexerGenerator,
    candidate_configs,
)
from b12x.policy.generation.store import CheckpointStore
from b12x.policy.generation.sweep import SweepMeasurement
from b12x.policy.types import DeviceIdentity


def test_raced_corpus_covers_the_served_c4_ladder() -> None:
    cases = raced_dsa_indexer_cases()
    expected = sum(
        len([c for c in DSA_INDEXER_CONTEXT_TOKENS if c <= model_len])
        for _ in DSA_INDEXER_RACED_GEOMETRIES
        for model_len in DSA_INDEXER_MODEL_LEN_TOKENS
        for _ in DSA_INDEXER_DECODE_ROWS
    )
    assert len(cases) == expected
    assert len({case.case_id for case in cases}) == len(cases)
    assert page_table_width(524_288, compress_ratio=4) == 2048
    widths = {int(case.query["max_page_table_width"]) for case in cases}
    assert widths == {512, 2048}
    for case in cases:
        assert case.metadata["raced"] is True
        assert int(case.metadata["context_tokens"]) <= int(
            case.metadata["model_len_tokens"]
        )
        DSA_INDEXER_POLICY.validate_config(
            DsaIndexerQuery(**dict(case.query)),
            DSA_INDEXER_POLICY.heuristic(DsaIndexerQuery(**dict(case.query)), None),
            None,
        )


def test_qualification_cases_keep_the_production_shapes_with_probes() -> None:
    cases = qualification_dsa_indexer_cases()
    labels = {str(case.metadata["probe"]["label"]) for case in cases}
    assert {
        "glm52-decode-spec4",
        "glm53-pooled-spec6",
        "minimax-m3-msa-decode",
    } <= labels
    for case in cases:
        assert case.metadata["raced"] is False
        assert case.metadata["probe"]["kind"] in {"dsa", "msa"}
    queries = DsaIndexerGenerator().reviewed_queries()
    assert any(q.num_q_heads == 32 and q.top_k == 2_048 for q in queries)
    assert any(q.num_q_heads == 32 and q.top_k == 512 for q in queries)
    assert any(q.score_mode == "msa" for q in queries)


def _raced_case(rows: int):
    return next(
        case
        for case in raced_dsa_indexer_cases()
        if int(case.query["max_q_rows"]) == rows
        and int(case.query["max_page_table_width"]) == 512
        and int(case.query["num_q_heads"]) == 32
    )


def test_candidate_menu_is_device_sized_and_starts_with_auto() -> None:
    small = candidate_configs(
        _raced_case(4).query, sm_count=48, compute_capability=(12, 1)
    )
    large = candidate_configs(
        _raced_case(4).query, sm_count=188, compute_capability=(12, 0)
    )
    assert small[0] == AUTO_CONFIG and large[0] == AUTO_CONFIG
    assert len(small) == len(large)
    fused_small = {
        c["fused_ctas_per_group"] for c in small if c["route"] == "paged_fused"
    }
    fused_large = {
        c["fused_ctas_per_group"] for c in large if c["route"] == "paged_fused"
    }
    assert fused_small == {0, 6, 24}
    assert fused_large == {0, 23, 94}
    tiled = [c for c in small if c["route"] == "paged_tiled"]
    assert {c["supertile_k"] for c in tiled} == {0, 8_192, 16_384}
    assert {c["two_level_fold"] for c in tiled} == {"auto", "off"}
    # Beyond the fused row gate only the tiled menu remains.
    wide = candidate_configs(
        _raced_case(64).query, sm_count=188, compute_capability=(12, 0)
    )
    assert all(c["route"] != "paged_fused" for c in wide[1:])
    for config in (*small, *large, *wide):
        DSA_INDEXER_POLICY.validate_config(
            DsaIndexerQuery(**dict(_raced_case(4).query)),
            DSA_INDEXER_POLICY.decode_profile(config),
            DeviceIdentity(
                vendor="nvidia",
                compute_capability=(12, 0),
                sm_count=188,
                product_name="NVIDIA RTX PRO 6000 Blackwell",
            ),
        )


class _FakeSession(AbstractContextManager):
    """Auto is 2% slower than a fused candidate on one scenario only."""

    def __init__(self, context):
        self._context = context

    def __exit__(self, *_exc):
        return None

    def candidates(self, case):
        from b12x.policy.generation.sweep import SweepCandidate

        if not case.metadata["raced"]:
            return (SweepCandidate.create(AUTO_CONFIG),)
        configs = candidate_configs(case.query, sm_count=48, compute_capability=(12, 1))
        return tuple(SweepCandidate.create(c) for c in configs[:3])

    def measure(self, case, candidates):
        out = []
        for candidate in candidates:
            if candidate.config["route"] == "auto":
                latency = 100.0
            elif candidate.config["fused_ctas_per_group"] == 0:
                latency = 98.5
            else:
                latency = 150.0
            out.append(
                SweepMeasurement(candidate=candidate, latency_us=latency, correct=True)
            )
        return tuple(out)


def test_reduction_keeps_auto_unless_a_candidate_clears_the_margin(tmp_path) -> None:
    cases = [c for c in raced_dsa_indexer_cases() if c.query["max_q_rows"] == 4][:3]
    cases.append(qualification_dsa_indexer_cases()[0])
    generator = DsaIndexerGenerator(
        benchmark_factory=lambda group, cases, context: _FakeSession(context),
        cases=cases,
    )
    context = GenerationContext(
        device=DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 1),
            sm_count=48,
            product_name="NVIDIA GB10",
        ),
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="test",
        settings=GenerationSettings(),
    )

    class Progress:
        def start_component(self, *a, **k):
            pass

        def start_stage(self, *a, **k):
            pass

        def advance(self, *a, **k):
            pass

        def finish_component(self, *a, **k):
            pass

    result = generator.generate(
        context, progress=Progress(), checkpoints=CheckpointStore(tmp_path / "ck")
    )

    def leaves(node):
        if node.get("kind") == "leaf":
            yield node["config"]
        for branch in node.get("branches", ()):
            yield from leaves(branch["node"])
        if "node" in node:
            yield from leaves(node["node"])

    configs = list(leaves(result.component["planner"]))
    assert configs, result.component
    assert all(config["route"] == "auto" for config in configs)
    assert result.evidence["single_candidate_qualification_cases"] == 1


def test_generator_extends_measured_anchors_to_the_served_domain() -> None:
    """A 64k max_model_len (width 256) or a 128-row batch must resolve to the
    nearest measured anchor rather than the heuristic."""
    from b12x.policy.generation.providers.dsa_indexer import (
        INDEXER_ROWS_BOUNDS,
        INDEXER_WIDTH_BOUNDS,
    )

    generator = DsaIndexerGenerator()
    bounds = generator._nearest_range_bounds
    assert bounds["max_page_table_width"] == INDEXER_WIDTH_BOUNDS
    assert bounds["max_q_rows"] == INDEXER_ROWS_BOUNDS
    assert INDEXER_WIDTH_BOUNDS[0] <= 256 and INDEXER_WIDTH_BOUNDS[1] >= 2048
    assert INDEXER_ROWS_BOUNDS[0] <= 1 and INDEXER_ROWS_BOUNDS[1] >= 16_384
