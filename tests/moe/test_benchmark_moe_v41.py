"""V4.1 benchmark rank geometry and unbiased global routing contracts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks.benchmark_moe import (
    MODEL_PROFILES,
    ModelSpec,
    _slice_v41_tp_shard,
    build_model_spec,
    compute_model_gate_routing,
    make_benchmark_case,
)


def test_v41_checkpoint_shards_intermediate_at_native_width(tmp_path):
    config = {
        "model_type": "deepseek_v41_text", "hidden_size": 5120,
        "moe_intermediate_size": 2304, "n_routed_experts": 384,
        "num_experts_per_tok": 6,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": config}))
    profile = MODEL_PROFILES["deepseek-v4.1-flash"]
    for rank in range(4):
        spec = build_model_spec(tmp_path, profile, tp_rank=rank)
        assert (spec.num_experts, spec.logical_I_tp, spec.I_tp) == (384, 576, 576)
        assert spec.global_intermediate_size == 2304
        assert spec.tp_rank == rank
    with pytest.raises(ValueError):
        build_model_spec(tmp_path, profile, tp_rank=4)
    config["model_type"] = "deepseek_v4"
    path.write_text(json.dumps({"text_config": config}))
    with pytest.raises(ValueError, match="V4.1"):
        build_model_spec(tmp_path, profile)


def test_shared_40_benchmark_uses_identical_oracle_and_timing_routes():
    spec = ModelSpec(16, 128, 384, 6, 4, 0)
    profile = MODEL_PROFILES["deepseek-v4.1-flash"]
    oracle = make_benchmark_case(
        profile, None, spec, 8, 50, torch.device("cpu"),
        routing_workload="shared_40",
    )
    timing = make_benchmark_case(
        profile, None, spec, 8, 50, torch.device("cpu"),
        routing_workload="shared_40",
    )
    for left, right in zip(oracle[:3], timing[:3], strict=True):
        torch.testing.assert_close(left, right)
    assert oracle[1].unique().numel() == 29
    assert oracle[3] is None
    torch.testing.assert_close(oracle[2].sum(-1), torch.ones(8))

def test_v41_tp_shards_keep_global_route_ids_and_unbiased_weights():
    spec = ModelSpec(2, 128, 8, 2, 4, 0)
    gate = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.],
                         [5., 0.], [6., 0.], [7., 0.], [8., 0.]])
    x = torch.tensor([[1., 0.], [-1., 0.]])
    bias = torch.tensor([0., 10., 0., 0., 0., 0., 0., 0.])
    weights = SimpleNamespace(
        spec=spec, gate_weight=gate, gate_score_func="sqrtsoftplus", gate_bias=bias,
        gate_tid2eid=None, gate_norm_topk_prob=True, gate_route_scale=1.5,
    )
    original_scores = torch.nn.functional.softplus(x @ gate.T).sqrt()
    expected_ids = torch.tensor([[1, 7], [1, 0]])
    expected_weights = original_scores.gather(1, expected_ids)
    expected_weights *= 1.5 / expected_weights.sum(-1, keepdim=True)
    for rank in range(4):
        weights.spec = replace(spec, tp_rank=rank)
        ids, route_weights = compute_model_gate_routing(weights, x, seed=0)
        torch.testing.assert_close(ids, expected_ids.to(torch.int32))
        torch.testing.assert_close(route_weights, expected_weights)


def test_v41_checkpoint_shard_offsets_are_exact():
    source_rows = torch.arange(4 * 576, dtype=torch.uint8).reshape(4 * 576, 1)
    source_columns = torch.arange(4 * (576 // 2), dtype=torch.uint8).reshape(1, -1)
    source_scales = torch.arange(4 * (576 // 32), dtype=torch.uint8).reshape(1, -1)
    rank = 1
    row_shard = _slice_v41_tp_shard(
        source_rows, dimension=0, intermediate_size_per_partition=576, tp_rank=rank,
    )
    column_shard = _slice_v41_tp_shard(
        source_columns, dimension=1, intermediate_size_per_partition=576,
        tp_rank=rank, packing=2,
    )
    scale_shard = _slice_v41_tp_shard(
        source_scales, dimension=1, intermediate_size_per_partition=576,
        tp_rank=rank, packing=32,
    )

    torch.testing.assert_close(row_shard, source_rows[rank * 576:(rank + 1) * 576])
    torch.testing.assert_close(column_shard, source_columns[:, rank * 288:(rank + 1) * 288])
    torch.testing.assert_close(scale_shard, source_scales[:, rank * 18:(rank + 1) * 18])
