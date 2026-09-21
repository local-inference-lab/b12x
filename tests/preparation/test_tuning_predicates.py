"""Dominance predicates and ladders of the raced parameter spaces.

The first tests cover the predicate machinery itself. The rest hold the four
families whose spaces carry a dominance predicate or a ladder -- blockscaled
A16 precision, fused-MoE decode, HyperConnection and GDN prefill -- against the
served Qwen3.8 Flash Next TP2 corpus under ``tests/preparation/corpora``: every
distinct declared query of those families keeps the recorded candidate count,
and the configuration that start selected for each query stays in its space.

``CANDIDATES`` records both counts, before and after the rules, so a change that
removes work states how much; ``WINNERS`` is the selection cache of that start.
A rule that drops a recorded winner is a rule that raced the wrong space.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, fields, replace

import pytest

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)

CANDIDATES = {
    # gemm.blockscaled_precision
    ('gemm.blockscaled_precision', 'mxfp8 n1152 k1152:1152 m65536 auto'): (13, 13),
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m1 a16'): (16, 16),
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m128 a16'): (16, 16),
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m2 a16'): (16, 16),
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m4 a16'): (16, 16),
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m8 a16'): (16, 16),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m1 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m128 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m2 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m4 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m8 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m1 auto'): (15, 15),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m128 auto'): (15, 15),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m2 auto'): (15, 15),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m4 auto'): (15, 15),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m8 auto'): (15, 15),
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k4608:4608 m16384 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n3456 k1152:1152 m65536 auto'): (9, 9),
    ('gemm.blockscaled_precision', 'mxfp8 n4304 k1152:1152 m65536 auto'): (5, 5),
    ('gemm.blockscaled_precision', 'mxfp8 n4608 k4608:4608 m16384 auto'): (13, 13),
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m1 auto'): (17, 9),
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m128 auto'): (17, 9),
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m2 auto'): (17, 9),
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m4 auto'): (17, 9),
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m8 auto'): (17, 9),
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m1 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m128 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m2 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m4 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m8 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m1 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m128 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m2 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m4 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m8 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m1 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m128 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m2 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m4 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m8 auto'): (17, 17),
    ('gemm.blockscaled_precision', 'nvfp4 n1152 k4304:4320 m65536 a16'): (12, 54),
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m1 a16'): (16, 24),
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m128 a16'): (16, 72),
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m2 a16'): (16, 24),
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m4 a16'): (16, 24),
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m8 a16'): (16, 24),
    # moe.decode
    ('moe.decode', 'nvfp4 rows10'): (195, 13),
    ('moe.decode', 'nvfp4 rows1250'): (4, 4),
    ('moe.decode', 'nvfp4 rows1280'): (4, 4),
    ('moe.decode', 'nvfp4 rows20'): (195, 14),
    ('moe.decode', 'nvfp4 rows40'): (194, 14),
    ('moe.decode', 'nvfp4 rows80'): (194, 15),
    ('moe.decode', 'w4a16 rows10'): (2, 2),
    ('moe.decode', 'w4a16 rows1250'): (1, 1),
    ('moe.decode', 'w4a16 rows1280'): (1, 1),
    ('moe.decode', 'w4a16 rows20'): (2, 2),
    ('moe.decode', 'w4a16 rows40'): (2, 2),
    ('moe.decode', 'w4a16 rows80'): (1, 1),
    # norm.hyperconnection
    ('norm.hyperconnection', 'combine h2560'): (1, 1),
    ('norm.hyperconnection', 'combine_norm h2560'): (1, 1),
    ('norm.hyperconnection', 'gate_mean h2560'): (14, 8),
    ('norm.hyperconnection', 'grouped_rmsnorm h2560'): (8, 8),
    ('norm.hyperconnection', 'scaled_silu h2560'): (1, 1),
    # sequence.gdn_prefill
    ('sequence.gdn_prefill', 't128 s1'): (416, 96),
    ('sequence.gdn_prefill', 't16 s1'): (192, 64),
    ('sequence.gdn_prefill', 't32 s1'): (224, 96),
    ('sequence.gdn_prefill', 't64 s1'): (288, 96),
}

WINNERS = {
    # gemm.blockscaled_precision
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m1 a16'): {'mode': 'a16', 'split_k': 2, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m128 a16'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m2 a16'): {'mode': 'a16', 'split_k': 2, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m4 a16'): {'mode': 'a16', 'split_k': 2, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'mxfp8 n124160 k2560:2560 m8 a16'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m1 auto'): {'mode': 'a16', 'split_k': 4, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m128 auto'): {'mode': 'quantized', 'split_k': None, 'tile_k': None, 'tile_n': None},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m2 auto'): {'mode': 'a16', 'split_k': 4, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m4 auto'): {'mode': 'a16', 'split_k': 4, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k3072:3072 m8 auto'): {'mode': 'a16', 'split_k': 4, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m1 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m128 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m2 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m4 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n2560 k320:384 m8 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m1 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m128 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m2 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m4 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n48 k2560:2560 m8 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m1 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m128 auto'): {'mode': 'a16', 'split_k': 4, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m2 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m4 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n640 k2560:2560 m8 auto'): {'mode': 'a16', 'split_k': 8, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m1 auto'): {'mode': 'quantized', 'split_k': None, 'tile_k': None, 'tile_n': None},
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m128 auto'): {'mode': 'quantized', 'split_k': None, 'tile_k': None, 'tile_n': None},
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m2 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m4 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n6656 k2560:2560 m8 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m1 auto'): {'mode': 'quantized', 'split_k': None, 'tile_k': None, 'tile_n': None},
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m128 auto'): {'mode': 'quantized', 'split_k': None, 'tile_k': None, 'tile_n': None},
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m2 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m4 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'mxfp8 n8192 k2560:2560 m8 auto'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 64},
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m1 a16'): {'mode': 'a16', 'split_k': 4, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m128 a16'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m2 a16'): {'mode': 'a16', 'split_k': 1, 'tile_k': 128, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m4 a16'): {'mode': 'a16', 'split_k': 4, 'tile_k': 64, 'tile_n': 128},
    ('gemm.blockscaled_precision', 'nvfp4 n124160 k2560:2560 m8 a16'): {'mode': 'a16', 'split_k': 1, 'tile_k': 64, 'tile_n': 128},
    # moe.decode
    ('moe.decode', 'nvfp4 rows10'): {'backend': 'micro', 'dynamic_route_mode': None, 'dynamic_tile_m': None, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': None},
    ('moe.decode', 'nvfp4 rows1250'): {'backend': 'dynamic', 'dynamic_route_mode': 'grouped', 'dynamic_tile_m': 32, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': None},
    ('moe.decode', 'nvfp4 rows1280'): {'backend': 'dynamic', 'dynamic_route_mode': 'grouped', 'dynamic_tile_m': 32, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': None},
    ('moe.decode', 'nvfp4 rows20'): {'backend': 'micro', 'dynamic_route_mode': None, 'dynamic_tile_m': None, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': None},
    ('moe.decode', 'nvfp4 rows40'): {'backend': 'micro', 'dynamic_route_mode': None, 'dynamic_tile_m': None, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': None},
    ('moe.decode', 'nvfp4 rows80'): {'backend': 'micro', 'dynamic_route_mode': None, 'dynamic_tile_m': None, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': None},
    ('moe.decode', 'w4a16 rows10'): {'backend': 'w4a16', 'dynamic_route_mode': None, 'dynamic_tile_m': None, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': 'direct'},
    ('moe.decode', 'w4a16 rows20'): {'backend': 'w4a16', 'dynamic_route_mode': None, 'dynamic_tile_m': None, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': 'direct'},
    ('moe.decode', 'w4a16 rows40'): {'backend': 'w4a16', 'dynamic_route_mode': None, 'dynamic_tile_m': None, 'max_active_clusters': None, 'route_planner': 'internal', 'w4a16_route_mode': 'packed'},
    # norm.hyperconnection
    ('norm.hyperconnection', 'gate_mean h2560'): {'pointwise_block': 256, 'reduction_block_h': 4096, 'reduction_num_warps': 4},
    ('norm.hyperconnection', 'grouped_rmsnorm h2560'): {'pointwise_block': 256, 'reduction_block_h': 4096, 'reduction_num_warps': 8},
    # sequence.gdn_prefill
    ('sequence.gdn_prefill', 't128 s1'): {'algorithm': 'sequential', 'backend': 'cutedsl', 'k_split': 2, 'segment_tokens': 256, 'stages': 4, 'v_split': 32, 'window_tiles': 9},
    ('sequence.gdn_prefill', 't16 s1'): {'algorithm': 'sequential', 'backend': 'cutedsl', 'k_split': 1, 'segment_tokens': 256, 'stages': 4, 'v_split': 32, 'window_tiles': 2},
    ('sequence.gdn_prefill', 't32 s1'): {'algorithm': 'sequential', 'backend': 'cutedsl', 'k_split': 1, 'segment_tokens': 256, 'stages': 3, 'v_split': 32, 'window_tiles': 3},
    ('sequence.gdn_prefill', 't64 s1'): {'algorithm': 'sequential', 'backend': 'cutedsl', 'k_split': 2, 'segment_tokens': 256, 'stages': 4, 'v_split': 32, 'window_tiles': 5},
}


CORPUS = pathlib.Path(__file__).parent / "corpora" / "qwen3_8_flash_next_tp2_rank0.jsonl"
# The recorded start ran on a 188-SM SM120 part. The MoE cluster ladder is keyed
# on that SM count and the A16 mode on the compute capability.
IDENTITY = DeviceIdentity("nvidia", (12, 0), 188, "synthetic SM120")



def test_predicates_see_complete_conditional_assignments():
    knobs = (
        Knob(name="backend", values=("native", "tiled")),
        Knob(
            name="tile",
            values=None,
            when=FrozenMapping({"backend": "tiled"}),
            otherwise=0,
        ),
        Knob(
            name="splits",
            values=(1, 2, 4),
            binding=ParameterBinding.RUNTIME,
            when=FrozenMapping({"backend": "tiled"}),
            otherwise=1,
        ),
    )
    space = ParameterSpace.create(
        knobs,
        values={"tile": (8, 16)},
        predicates=(
            lambda p: p["backend"] == "native" or p["tile"] * p["splits"] <= 32,
        ),
    )
    assignments = list(space.configurations())
    assert {(p["backend"], p["tile"], p["splits"]) for p in assignments} == {
        ("native", 0, 1),
        ("tiled", 8, 1),
        ("tiled", 8, 2),
        ("tiled", 8, 4),
        ("tiled", 16, 1),
        ("tiled", 16, 2),
    }
    with pytest.raises(ValueError, match="predicates"):
        space.validate(dict(backend="tiled", tile=16, splits=4))


def test_predicates_filter_before_materialization_and_keep_raw_count():
    @dataclass(frozen=True)
    class Config:
        tile: int
        splits: int

    knobs = (Knob(name="tile", values=(8, 16)), Knob(name="splits", values=(1, 2, 4)))

    def parameters(capacity, device):
        return ParameterSpace.create(
            knobs,
            predicates=(lambda p: p["tile"] * p["splits"] <= capacity,),
        )

    def materialize(capacity, device, choice):
        if choice["tile"] * choice["splits"] > capacity:
            raise AssertionError("filtered assignment reached the constructor")
        return Config(**choice)

    contract = TuningContract(
        component_id="test.predicates", query_schema_version=1, config_schema_version=1,
        query_fields=frozenset({"capacity"}), config_fields=frozenset({"tile", "splits"}),
        encode_query=lambda capacity: {"capacity": capacity}, encode_config=asdict,
        decode_config=lambda payload: Config(**dict(payload)),
        default_config=lambda capacity, device: Config(8, 1),
        validate_query=lambda capacity, device: None,
        validate_config=lambda query, config, device: None,
        knobs=knobs,
        parameters=parameters,
        materialize=materialize,
    )
    plan = contract.eligible_plan(32, None)
    assert plan.cartesian_count == 6
    assert plan.legal_count == 5
    assert {(config.tile, config.splits) for _, config in plan.candidates} == {
        (8, 1),
        (8, 2),
        (8, 4),
        (16, 1),
        (16, 2),
    }
    with pytest.raises(ValueError, match="predicates"):
        contract.lower(32, None, dict(tile=16, splits=4))

    def broken(assignment):
        raise ValueError("planner defect")

    invalid = replace(
        contract,
        parameters=lambda *_: ParameterSpace(knobs=knobs, predicates=(broken,)),
    )
    with pytest.raises(ValueError, match="planner defect"):
        invalid.eligible_plan(32, None)


def test_mhc_predicates_remove_inactive_work_without_rejecting_useful_tails():
    from b12x.norm.mhc import _tuning as component

    query = component.MhcQuery(
        dtype="bfloat16",
        max_tokens=1,
        hidden_size=4096,
        split_k=64,
        operation="post_pre",
        has_norm_weight=True,
        norm_eps=1e-6,
        smem_limit=100 << 10,
    )
    choice = dict(
        backend="tf32_tma",
        lagged_prepare=False,
        projection_tile_n=8,
        projection_tile_k=64,
        projection_num_stages=2,
        projection_num_m_warps=1,
        projection_num_n_warps=1,
        projection_k_splits=8,
    )
    decode = component.TUNING.parameter_space(query, None)
    with pytest.raises(ValueError, match="predicates"):
        decode.validate({**choice, "projection_num_m_warps": 2})
    with pytest.raises(ValueError, match="predicates"):
        decode.validate(
            {**choice, "projection_tile_n": 32, "projection_num_n_warps": 4}
        )
    with pytest.raises(ValueError, match="predicates"):
        decode.validate({**choice, "projection_tile_n": 64})
    with pytest.raises(ValueError, match="predicates"):
        decode.validate(
            {
                **choice,
                "projection_tile_k": 256,
                "projection_k_splits": 32,
                "projection_num_stages": 3,
            }
        )

    boundary = component.TUNING.parameter_space(replace(query, max_tokens=128), None)
    # Forty-eight-row tiles have a partial final CTA at M128, but every warp
    # still does useful work in earlier CTAs. N32 remains a useful N24 overtile.
    tail = {
        **choice,
        "projection_num_m_warps": 3,
        "projection_tile_n": 32,
        "projection_num_n_warps": 2,
    }
    assignments = set(boundary.configurations())
    assert FrozenMapping(tail) in assignments
    assert (
        FrozenMapping({**choice, "projection_tile_n": 24, "projection_num_n_warps": 3})
        in assignments
    )
    with pytest.raises(ValueError, match="predicates"):
        boundary.validate(
            {
                **choice,
                "projection_num_m_warps": 8,
                "projection_tile_n": 24,
                "projection_num_n_warps": 3,
                "projection_tile_k": 256,
                "projection_num_stages": 4,
            }
        )


@pytest.mark.parametrize("tile_k", [8, 16])
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
def test_mhc_tf32_rejects_short_weight_rows(tile_k, operation):
    from b12x.norm.mhc import _tuning as component
    from b12x.norm.mhc._kernels import MHCPrefillTf32ProjectTmaKernel

    query = component.MhcQuery(
        dtype="bfloat16", max_tokens=1, hidden_size=5120, split_k=80,
        operation=operation, has_norm_weight=True, lagged_mix=True,
        expanded_residual=operation == "pre", norm_eps=1e-20,
        rms_eps=1e-20, smem_limit=100 << 10,
    )
    choice = dict(
        backend="tf32_tma", lagged_prepare=False, projection_tile_n=8,
        projection_tile_k=tile_k, projection_num_stages=3,
        projection_num_m_warps=1, projection_num_n_warps=1,
        projection_k_splits=2,
    )
    space = component.TUNING.parameter_space(query, None)
    with pytest.raises(ValueError, match="predicates"):
        space.validate(choice)
    config = component.MhcConfig(projection_tile_m=16, **choice)
    with pytest.raises(ValueError, match="at least 32"):
        component.TUNING.validate_config(query, config, None)
    with pytest.raises(ValueError, match="at least 32"):
        MHCPrefillTf32ProjectTmaKernel(
            hidden_size=5120, split_k=80, tile_m=16, tile_n=8,
            tile_k=tile_k, num_stages=3, num_m_warps=1, num_n_warps=1,
            k_splits=2, split_fp32_fn=True,
        )
    space.validate({**choice, "projection_tile_k": 32})


def _thawed(value):
    if isinstance(value, dict):
        return FrozenMapping({name: _thawed(item) for name, item in value.items()})
    if isinstance(value, list):
        return tuple(_thawed(item) for item in value)
    return value


def _label(component: str, query: dict) -> str:
    """Name a query by the fields that select its candidate space."""
    if component == "gemm.blockscaled_precision":
        return (
            f"{query['recipe']} n{query['out_features']} "
            f"k{query['in_features']}:{query['padded_in_features']} "
            f"m{query['num_tokens']} {query['activation_mode']}"
        )
    if component == "moe.decode":
        return f"{query['quant_mode']} rows{query['routed_rows']}"
    if component == "norm.hyperconnection":
        return f"{query['operation']} h{query['hidden_size']}"
    return f"t{query['max_tokens']} s{query['max_seqs']}"


def _contract_and_query(component: str, payload: dict):
    payload = dict(payload)
    if component == "gemm.blockscaled_precision":
        from b12x.gemm.blockscaled._tuning import TUNING, BlockscaledQuery

        payload["codegen"] = _thawed(payload["codegen"])
        return TUNING, BlockscaledQuery(**payload)
    if component == "moe.decode":
        from b12x.moe.fused_moe._tuning import TUNING, MoeDecodeQuery

        return TUNING, MoeDecodeQuery(**{name: _thawed(item) for name, item in payload.items()})
    if component == "norm.hyperconnection":
        from b12x.norm.hyperconnection._tuning import TUNING, HyperConnectionQuery

        if payload.get("limit") == "+inf":
            payload["limit"] = float("inf")
        return TUNING, HyperConnectionQuery(**payload)
    if component == "sequence.gdn_prefill":
        from b12x.sequence.gdn_prefill._tuning import TUNING, GdnPrefillQuery

        names = {field.name for field in fields(GdnPrefillQuery)}
        return TUNING, GdnPrefillQuery(**{k: v for k, v in payload.items() if k in names})
    raise KeyError(component)


def _corpus_queries():
    """One declaration per distinct candidate space of the four families."""
    families = {component for component, _ in CANDIDATES}
    seen = {}
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            component = record["component"]
            if component not in families:
                continue
            seen.setdefault((component, _label(component, record["query"])), record["query"])
    return seen


DECLARED = _corpus_queries()


def test_corpus_declares_every_recorded_candidate_space():
    assert set(DECLARED) == set(CANDIDATES)


@pytest.mark.parametrize("key", sorted(CANDIDATES))
def test_served_queries_keep_their_recorded_candidate_count(key):
    component, label = key
    before, after = CANDIDATES[key]
    contract, query = _contract_and_query(component, DECLARED[key])
    iterator = contract.iterate(contract.configure(query, device=IDENTITY))
    for _ in iterator:
        pass
    assert iterator.effective_count == after, f"{label}: {before} candidates before the rules"


@pytest.mark.parametrize("key", sorted(WINNERS))
def test_recorded_selections_stay_eligible(key):
    contract, query = _contract_and_query(key[0], DECLARED[key])
    values = dict(WINNERS[key])
    if key[0] == "moe.decode":
        # The recorded programs use unshared input and monolithic NVFP4 execution.
        values.update(nvfp4_share_input=False, nvfp4_materialize_intermediate=False)
    if key[0] == "gemm.blockscaled_precision":
        values["tile_m"] = 16 if values["mode"] == "a16" else None
    assignment = FrozenMapping(values)
    contract.parameter_space(query, IDENTITY).validate(assignment)
    contract.lower(query, IDENTITY, assignment)


def test_gdn_prefill_races_one_segment_plan_and_a_window_ladder():
    from b12x.sequence.gdn_prefill import _tuning as component

    query = component.GdnPrefillQuery(
        key_heads=8, value_heads=24, head_dim=128, model_dtype="bfloat16",
        state_dtype="float32", qk_l2norm=True, checkpoint_export=True,
        max_tokens=128, max_seqs=1, null_state_index=0, dt_bias_dtype="bfloat16",
    )
    space = component.TUNING.parameter_space(query, IDENTITY)
    sequential = dict(
        backend=component.BACKEND, algorithm="sequential", v_split=32, k_split=1,
        stages=3, segment_tokens=256, window_tiles=9,
    )
    windows = {
        assignment["window_tiles"] for assignment in space.configurations()
        if assignment["algorithm"] == "sequential"
    }
    # cap = ceil(128/16) + 1 = 9 tiles, raced as one, two and four windows.
    assert windows == {9, 5, 3}
    space.validate(sequential)
    with pytest.raises(ValueError, match="efficiency predicates"):
        space.validate({**sequential, "window_tiles": 8})
    exhaustive = component.TUNING.parameter_space(replace(query, exhaustive=True), IDENTITY)
    exhaustive.validate({**sequential, "window_tiles": 8})
    exhaustive.validate({**sequential, "algorithm": "chunk_parallel",
                         "segment_tokens": 1024, "window_tiles": None})
    # A 128-token sequence is one 128-token segment: chunk-parallel has no
    # chunk-level parallelism to exploit and is not raced at all.
    assert not any(
        assignment["algorithm"] == "chunk_parallel" for assignment in space.configurations()
    )
    with pytest.raises(ValueError, match="predicates"):
        space.validate({
            **sequential, "algorithm": "chunk_parallel",
            "segment_tokens": 128, "window_tiles": None,
        })

    longer = component.TUNING.parameter_space(replace(query, max_tokens=512), IDENTITY)
    chunked = {
        assignment["segment_tokens"] for assignment in longer.configurations()
        if assignment["algorithm"] == "chunk_parallel"
    }
    # Segments at or above the sequence plan one segment per sequence; 512 is
    # the smallest of those, and 1024 would only widen the shared-memory band.
    assert chunked == {128, 256, 512}


def test_moe_cluster_ladder_stops_at_the_planned_task_queue():
    from b12x.moe.fused_moe import _tuning as component

    query = component.MoeDecodeQuery(
        quant_mode="nvfp4", quant_modes=("nvfp4",), source_format="modelopt_nvfp4",
        activation="silu", io_dtype="bfloat16", num_experts=512, hidden_size=2560,
        intermediate_size=320, top_k=10, num_tokens=1, routed_rows=10,
        route_num_experts=512, route_logits_dtype="float32",
        apply_router_weight_on_input=False, collect_activation_amax=False,
        deterministic_output=False, swiglu_limit=None, swiglu_alpha=1.702,
        swiglu_beta=1.0, w13_layout="fused", weight_layouts=("fused",),
        w4a16_weight_layout=None, w4a16_scale_format=None, w4a16_block_size_m=None,
        fast_math=True, numerical_recipe=None, controls=FrozenMapping(),
    )
    space = component.TUNING.parameter_space(query, IDENTITY)
    clusters = {
        assignment["max_active_clusters"] for assignment in space.configurations()
        if assignment["route_planner"] == "triton"
    }
    # Ten routed rows over three N128 slices plan thirty grouped tasks; wider
    # grids only add clusters that take no task.
    assert clusters == {None, 1, 2, 4, 8, 16, 30}
    triton = dict(
        backend="dynamic", route_planner="triton", dynamic_tile_m=16,
        dynamic_route_mode="grouped", w4a16_route_mode=None, max_active_clusters=30,
        nvfp4_share_input=False, nvfp4_materialize_intermediate=False,
    )
    space.validate(triton)
    for outside in (31, 64, 188):
        with pytest.raises(ValueError, match="eligible values"):
            space.validate({**triton, "max_active_clusters": outside})


def test_a16_wide_tile_needs_more_than_one_n_tile():
    from b12x.gemm.blockscaled._tuning import TUNING, BlockscaledQuery

    narrow = BlockscaledQuery(
        recipe="mxfp8", num_tokens=1, in_features=2560, padded_in_features=2560,
        out_features=48,
    )
    space = TUNING.parameter_space(narrow, IDENTITY)
    assignment = dict(mode="a16", tile_m=16, tile_n=64, tile_k=64, split_k=1)
    space.validate(assignment)
    with pytest.raises(ValueError, match="predicates"):
        space.validate({**assignment, "tile_n": 128})
    wide = TUNING.parameter_space(replace(narrow, out_features=128), IDENTITY)
    wide.validate({**assignment, "tile_n": 128})


def test_iq2_transposed_tiles_are_raced_without_split_k():
    from b12x.gemm.blockscaled._tuning import TUNING, BlockscaledConfig, BlockscaledQuery

    query = BlockscaledQuery(recipe="iq2_xs", num_tokens=8, in_features=768,
                             padded_in_features=768, out_features=136)
    assignment = dict(mode="a16", tile_m=8, tile_n=128, tile_k=256, split_k=1)
    TUNING.parameter_space(query, IDENTITY).validate(assignment)
    TUNING.validate_config(query, BlockscaledConfig(**assignment), IDENTITY)
    with pytest.raises(ValueError, match="split-K"):
        TUNING.validate_config(query, BlockscaledConfig(**{**assignment, "split_k": 2}), IDENTITY)
    for recipe in ("nvfp4", "mxfp8"):
        with pytest.raises(ValueError, match="predicates"):
            TUNING.parameter_space(replace(query, recipe=recipe), IDENTITY).validate(
                {**assignment, "tile_k": 128})


def test_nvfp4_a16_races_k256_tiles():
    from b12x.gemm.blockscaled._tuning import TUNING, BlockscaledConfig, BlockscaledQuery

    query = BlockscaledQuery(recipe="nvfp4", num_tokens=64, in_features=800,
                             padded_in_features=800, out_features=136, activation_mode="a16")
    for tile_m in (16, 32, 64):
        assignment = dict(mode="a16", tile_m=tile_m, tile_n=128, tile_k=256, split_k=4)
        TUNING.parameter_space(query, IDENTITY).validate(assignment)
        TUNING.validate_config(query, BlockscaledConfig(**assignment), IDENTITY)
    with pytest.raises(ValueError, match="predicates"):
        TUNING.parameter_space(replace(query, recipe="mxfp8", global_scale_kind="none"), IDENTITY).validate(
            {**assignment, "tile_m": 16})


def test_gate_mean_partitions_span_one_warp_to_the_covering_block():
    from b12x.norm.hyperconnection._tuning import TUNING, HyperConnectionQuery

    query = HyperConnectionQuery(
        dtype="bfloat16", max_tokens=128, hidden_size=2560, streams=4, lowrank=4,
        operation="gate_mean",
    )
    space = TUNING.parameter_space(query, IDENTITY)
    blocks = {assignment["pointwise_block"] for assignment in space.configurations()}
    assert blocks == {32, 64, 128, 256, 512, 1024, 2048, 4096}
