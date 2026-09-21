"""Model admission and checkpoint-bound learned cache artifacts."""

from dataclasses import replace
import json

import pytest
import torch

from b12x.integration.vllm.expert_cache import (
    ExpertCacheModel,
    ExpertCacheServingConfig,
    digest,
)
from b12x.moe.fused_moe._cache_preparation import ExpertCacheMemory
from b12x.moe.fused_moe.execution import ExecutionCapacity
from b12x.moe.fused_moe.residency import profile_from_counts
from tests.moe.test_prepared_expert_cache import source


def test_checkpoint_receipt_reuses_hash_and_rejects_same_size_mutation(tmp_path, monkeypatch):
    from b12x.integration.vllm.checkpoint_identity import checkpoint_identity

    root = tmp_path / "model"
    root.mkdir()
    shard = root / "model.safetensors"
    shard.write_bytes(b"original")
    receipt = tmp_path / "identity.json"
    first = checkpoint_identity(root, output=receipt)
    assert checkpoint_identity(root, receipt=receipt) == first
    shard.write_bytes(b"mutation")
    with pytest.raises(ValueError, match="stale"):
        checkpoint_identity(root, receipt=receipt)


def test_checkpoint_receipt_rejects_added_inventory_and_does_not_change_digest(tmp_path):
    import hashlib
    from b12x.integration.vllm.checkpoint_identity import checkpoint_identity

    root = tmp_path / "model"
    root.mkdir()
    (root / "a.safetensors").write_bytes(b"weights")
    receipt = tmp_path / "identity.json"
    document = checkpoint_identity(root, output=receipt)
    expected = hashlib.sha256(b"a.safetensors\0" + (7).to_bytes(8, 'little') + b"weights")
    assert document['fingerprint'] == expected.hexdigest()
    (root / "config.json").write_text('{}')
    with pytest.raises(ValueError, match="stale"):
        checkpoint_identity(root, receipt=receipt)


def test_experimental_check_cadence_backs_off_only_on_observed_health():
    from benchmarks.moe.expert_cache_serving import maintenance_check_interval

    interval, history = 32, []
    for health in ("healthy", "healthy", "healthy", "healthy", "pressure", None):
        interval = maintenance_check_interval(
            interval, minimum=32, maximum=256, health=health
        )
        history.append(interval)
    assert history == [64, 128, 256, 256, 32, 32]
    with pytest.raises(ValueError, match="ordered"):
        maintenance_check_interval(16, minimum=32, maximum=256, health="healthy")


def test_trace_alignment_excludes_incomplete_prefill_and_uses_logical_request():
    from benchmarks.moe.compare_execution_traces import sample_rows

    trace = {
        "steps": [
            {
                "step": 0,
                "requests": ["cache-7-deadbeef", "cache-4-1234abcd"],
                "computed": [0, 20],
                "scheduled": [6, 1],
                "prefill_lengths": [18, 20],
            }
        ]
    }
    device = {"step": 0, "logits": torch.zeros(2, 8)}
    rows = sample_rows(trace, [device])
    assert set(rows) == {("cache-4", 1)}
    assert rows["cache-4", 1][2] == 1
    with pytest.raises(ValueError, match="duplicate"):
        sample_rows(trace, [device, device])


def test_trace_comparison_keeps_shape_and_numeric_gates_separate(tmp_path):
    from benchmarks.moe.compare_execution_traces import compare

    paths = [tmp_path / name for name in ("left.json", "right.json")]
    for index, path in enumerate(paths):
        trace = {
            "overflow": False,
            "steps": [
                {
                    "step": 0,
                    "requests": ["cache-4-" + ("deadbeef" if index else "1234abcd")],
                    "computed": [20],
                    "scheduled": [1],
                    "prefill_lengths": [20],
                    "tokens": 15 if index else 57,
                }
            ],
        }
        path.write_text(json.dumps(trace))
        torch.save(
            [
                {
                    "step": 0,
                    "hidden": torch.ones(1, 2),
                    "logits": torch.tensor(
                        [[2.0, 2.01 if index else 1.99, 0, 0, 0, 0]]
                    ),
                }
            ],
            str(path) + ".pt",
        )
    result = compare(*paths)
    assert not result["same_execution_signature"]
    request = result["requests"]["cache-4"]
    assert request["first_hidden_difference"] is None
    assert request["first_argmax_difference"]["output_index"] == 1
    trace["overflow"] = True
    paths[1].write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="truncated"):
        compare(*paths)


def config(tmp_path, **changes):
    return ExpertCacheServingConfig(
        **(
            dict(
                mode="profile",
                activation="w4a16",
                profile_path=str(tmp_path / "profile.json"),
                workload="general",
                expert_device_bytes=150000,
                host_bytes=1 << 30,
                kv_reserved_bytes=100,
                graph_reserved_bytes=100,
                device_safety_bytes=100,
                host_safety_bytes=100,
            )
            | changes
        )
    )


def model(tmp_path, monkeypatch, **changes):
    import b12x.integration.vllm.expert_cache as module

    def memory(q, device, source_bytes):
        return ExpertCacheMemory(
            q.resident * 28000,
            q.experts * 28000,
            source_bytes,
            100,
            100,
            64 if q.max_pairs else 0,
        )

    monkeypatch.setattr(module, "memory_for", memory)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (900000, 1000000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 80000)
    value = ExpertCacheModel(config(tmp_path, **changes), "a" * 64, "cuda:0")
    a = source()
    value.add_source(a)
    value.add_source(replace(a, weights=replace(a.weights, layer_name="second")))
    return value


def test_model_admission_counts_existing_device_use_once_and_bootstraps_fairly(
    tmp_path, monkeypatch
):
    value = model(tmp_path, monkeypatch)
    value.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    assert value.memory.dense_model == 80000 and value.memory.other_device == 20000
    counts = [len(p.hbm_expert_ids) for p in value.placements.values()]
    assert max(counts) - min(counts) <= 1
    assert value.memory.host_sources == sum(
        s.source_bytes for s in value.sources.values()
    )
    assert value.memory.device_bytes <= value.memory.device_capacity
    assert value.counter is not None


def test_calibration_starts_after_warmup_and_saves_only_new_counts(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from b12x.moe.residency.contracts import LayerRoutingCounts, RoutingSnapshot
    import b12x.integration.vllm.expert_cache as module

    value = model(tmp_path, monkeypatch)
    value.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    with pytest.raises(ValueError, match="must start"):
        value.save_profile(quiescent=True)
    with pytest.raises(ValueError, match="prepared, drained"):
        value.start_profile(quiescent=True)
    startup = RoutingSnapshot(epoch=1, rank=0, layers=tuple(
        LayerRoutingCounts(layer=name, phase="decode", counts=(9, 0, 0, 0))
        for name in value.sources))
    counters = Mock(snapshot=Mock(return_value=startup))
    value._counters = counters
    with pytest.raises(ValueError, match="prepared, drained"):
        value.start_profile()
    counters.reset.assert_not_called()
    result = value.start_profile(quiescent=True)
    assert result["discarded_startup_observations"]["layers"][0]["counts"] == (9, 0, 0, 0)
    counters.reset.assert_called_once_with(quiescent=True)
    counters.set_token_limit.assert_called_once_with(0)
    with pytest.raises(RuntimeError, match="already started"):
        value.start_profile(quiescent=True)
    calibrated = replace(startup, epoch=2, layers=tuple(
        replace(row, counts=(0, 3, 2, 10)) for row in startup.layers))
    counters.snapshot.return_value = calibrated
    monkeypatch.setattr(module.moe, "routing_profile_state", lambda _: counters)
    value.save_profile(quiescent=True)
    artifact = json.loads((tmp_path / "profile.json").read_text())
    assert all(p["selection_counts"] == [0, 3, 2, 10]
               for p in artifact["placements"].values())


@pytest.mark.parametrize("mode", ["static", "adaptive"])
def test_calibration_reset_cannot_reset_a_serving_observer(tmp_path, mode):
    from unittest.mock import Mock

    value = ExpertCacheModel(config(tmp_path, mode=mode), "a" * 64, "cuda:0")
    value._counters = Mock()
    with pytest.raises(ValueError, match="explicit profile mode"):
        value.start_profile(quiescent=True)
    value._counters.reset.assert_not_called()


def write_profile(value, hot_count=2):
    value.capacity = ExecutionCapacity(max_tokens=4, top_k=4)
    placements = {
        name: profile_from_counts(
            counts=(1, 2, 20, 30),
            hot_count=hot_count,
            layer=name,
            model_fingerprint="a" * 64,
            workload="general",
            provenance="held-out calibration",
            phase="decode",
        ).to_dict()
        for name in value.sources
    }
    payload = {
        "identity": value._identity(),
        "placements": placements,
        "termination": "explicit_calibration_boundary",
        "converged": False,
    }
    payload["hash"] = digest(payload)
    from pathlib import Path

    Path(value.config.profile_path).write_text(json.dumps(payload))
    return payload


def test_static_and_adaptive_start_from_identical_learned_profile(
    tmp_path, monkeypatch
):
    static = model(tmp_path, monkeypatch, mode="static")
    write_profile(static)
    adaptive = model(tmp_path, monkeypatch, mode="adaptive")
    capacity = ExecutionCapacity(max_tokens=4, top_k=4)
    static.declare(capacity)
    adaptive.declare(capacity)
    assert static.placements == adaptive.placements
    assert static.counter is None and adaptive.counter is not None
    assert all(p.query.max_pairs == 0 for p in static.plans.values())
    assert all(p.query.max_pairs == 2 for p in adaptive.plans.values())


def test_all_resident_static_is_admitted_but_adaptive_is_rejected(
    tmp_path, monkeypatch
):
    static = model(tmp_path, monkeypatch, mode="static", expert_device_bytes=300000)
    write_profile(static, hot_count=4)
    static.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    assert static.counter is None
    assert all(
        p.query.resident == p.query.experts and not p.query.max_pairs
        for p in static.plans.values()
    )
    adaptive = model(tmp_path, monkeypatch, mode="adaptive", expert_device_bytes=300000)
    with pytest.raises(
        ValueError, match="requires at least one layer with nonresident"
    ):
        adaptive.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    assert all(p.prepared is None for p in adaptive.plans.values())


@pytest.mark.parametrize(
    "field,value",
    [("checkpoint", "b" * 64), ("recipe", "a4"), ("workload", "code"), ("version", 2)],
)
def test_profile_identity_is_validated_even_with_recomputed_hash(
    tmp_path, monkeypatch, field, value
):
    m = model(tmp_path, monkeypatch, mode="static")
    payload = write_profile(m)
    payload.pop("hash")
    payload["identity"][field] = value
    payload["hash"] = digest(payload)
    from pathlib import Path

    Path(m.config.profile_path).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mismatch"):
        m.declare(ExecutionCapacity(max_tokens=4, top_k=4))


def test_failed_model_admission_cannot_be_retried_as_prepared(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch, host_bytes=1)
    with pytest.raises(ValueError, match="capacity"):
        m.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    with pytest.raises(RuntimeError, match="admission failed"):
        m.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    with pytest.raises(RuntimeError, match="admission"):
        m.requests()


def test_source_identity_and_preallocation_host_admission(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        m.add_source(source())
    with pytest.raises(ValueError, match="host envelope"):
        m.reserve_source(m.config.host_bytes)


def test_cpu_source_owner_bytes_include_original_scale_storage():
    s = source()
    owner = torch.empty((4, 2), dtype=torch.float32)
    retained = replace(s, owners=(owner, s.weights.w13))
    assert (
        retained.source_bytes == s.source_bytes + owner.numel() * owner.element_size()
    )


def test_routing_top_k_is_part_of_profile_identity(tmp_path, monkeypatch):
    value = model(tmp_path, monkeypatch, mode="static")
    write_profile(value)
    with pytest.raises(ValueError, match="mismatch"):
        value.declare(ExecutionCapacity(max_tokens=4, top_k=2))


def test_adaptive_omits_fully_resident_layers_from_observations(tmp_path, monkeypatch):
    from pathlib import Path

    value = model(tmp_path, monkeypatch, mode="adaptive", expert_device_bytes=300000)
    payload = write_profile(value)
    payload["placements"]["second"] = profile_from_counts(
        counts=(1, 2, 20, 30),
        hot_count=4,
        layer="second",
        model_fingerprint="a" * 64,
        workload="general",
        provenance="fully resident learned control",
        phase="decode",
    ).to_dict()
    payload.pop("hash")
    payload["hash"] = digest(payload)
    Path(value.config.profile_path).write_text(json.dumps(payload))
    value.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    assert value.observed_layers == ("layer",)
    assert value.plans["second"].query.max_pairs == 0
    assert value.counter.query.layers == (("layer", 4),)


@pytest.fixture
def next_checkpoint(tmp_path):
    """Small complete target schema; no real checkpoint data or CUDA needed."""
    import struct
    from b12x.integration.vllm.checkpoint import expected_qwen3_next

    config = dict(
        architectures=["Qwen3NextForCausalLM"],
        model_type="qwen3_next",
        dtype="bfloat16",
        hidden_act="silu",
        norm_topk_prob=True,
        hidden_size=128,
        moe_intermediate_size=128,
        num_experts=2,
        num_hidden_layers=2,
        num_experts_per_tok=1,
        vocab_size=8,
        layer_types=["linear_attention", "full_attention"],
        shared_expert_intermediate_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
    )
    excluded = [
        "lm_head",
        "*.mlp.gate",
        "*.mlp.shared_expert_gate",
        "*.linear_attn.in_proj_ba",
        "*.linear_attn.in_proj_qkvz",
        "*.self_attn.q_proj",
        "*.self_attn.k_proj",
        "*.self_attn.v_proj",
    ]
    scheme = dict(dynamic=False, num_bits=4, type="float", group_size=16)
    config["quantization_config"] = dict(
        quant_method="modelopt",
        quant_algo="NVFP4",
        ignore=excluded,
        config_groups={
            "group_0": dict(
                targets=["Linear"], weights=scheme, input_activations=scheme
            )
        },
    )
    quant = dict(
        quantization=dict(quant_algo="NVFP4", group_size=16, exclude_modules=excluded)
    )
    expected = expected_qwen3_next(config, quant)
    expected["mtp.layers.0.aux.weight"] = dict(shape=[3], dtype="BF16")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant))

    def write(changes=None):
        import math
        from b12x.integration.vllm.checkpoint import DTYPE_BYTES

        records = {k: dict(v) for k, v in expected.items()}
        if changes:
            changes(records)
        payload, header = bytearray(), {}
        for name, item in records.items():
            size = math.prod(item["shape"]) * DTYPE_BYTES[item["dtype"]]
            data = struct.pack("<f", 1.0) if item["dtype"] == "F32" else bytes(size)
            header[name] = dict(
                shape=item["shape"],
                dtype=item["dtype"],
                data_offsets=[len(payload), len(payload) + size],
            )
            payload.extend(data)
        raw = json.dumps(header).encode()
        (tmp_path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(raw)) + raw + payload
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                dict(
                    metadata=dict(total_size=len(payload)),
                    weight_map={k: "model.safetensors" for k in header},
                )
            )
        )

    write()
    return tmp_path, write


def test_metadata_preflight_counts_target_only_and_checks_globals(next_checkpoint):
    from b12x.integration.vllm.checkpoint import audit

    path, _ = next_checkpoint
    result = audit(path, check_values=True)
    assert result["classes"]["optional_mtp"]["bytes"] == 6
    assert result["required_target_bytes"] == sum(
        r["bytes"] for k, r in result["classes"].items() if k != "optional_mtp"
    )
    assert result["routed_source_bytes"] == 2 * 2 * (
        3 * 128 * 128 // 2 + 3 * 128 * 128 // 16 + 24
    )
    assert (
        result["host_expert_lower_bound_bytes"]
        == result["routed_source_bytes"] + result["canonical_mapped_bytes"]
    )
    from b12x.moe.fused_moe._cache_preparation import tier_layout

    assert result["canonical_mapped_bytes"] == 2 * tier_layout(2, 128, 128)[1]
    assert result["global_scale_values"].startswith("all_routed")
    assert result["status"] == "metadata_compatible_execution_unqualified"


def test_local_block_scales_accept_zero_and_reject_nan(next_checkpoint):
    from b12x.integration.vllm.checkpoint import inventory, validate_block_scales

    path, _ = next_checkpoint
    result = validate_block_scales(path)
    assert result['bytes'] == result['zero_scales'] > 0
    tensors, _, _ = inventory(path)
    name = 'model.layers.0.mlp.experts.0.gate_proj.weight_scale'
    with (path / tensors[name]['shard']).open('r+b') as stream:
        stream.seek(tensors[name]['offset'])
        stream.write(bytes([127]))
    with pytest.raises(ValueError, match='nonfinite or negative'):
        validate_block_scales(path)


def test_actual_loader_coverage_requires_callbacks_and_excludes_mtp(next_checkpoint):
    from b12x.integration.vllm.checkpoint import (
        inventory, expected_qwen3_next, read_json, validate_loader_coverage)

    path, _ = next_checkpoint
    tensors, _, _ = inventory(path)
    expected = expected_qwen3_next(read_json(path / 'config.json'), read_json(path / 'hf_quant_config.json'))
    coverage = dict(status='completed', tensors={n: dict(
        shape=t['shape'], bytes=t['bytes'], destinations=[dict(
            parameter=n, device='cpu' if expected[n]['kind'] == 'routed' else 'cuda:0', accepted=True)]
            if n in expected else []) for n,t in tensors.items()})
    assert validate_loader_coverage(path, coverage)['target_tensors'] == len(expected)
    name = next(iter(expected))
    destination = coverage['tensors'][name]['destinations'].pop()
    with pytest.raises(ValueError, match='one accepted'):
        validate_loader_coverage(path, coverage)
    coverage['tensors'][name]['destinations'].append(destination)
    coverage['tensors']['mtp.layers.0.aux.weight']['destinations'].append(destination)
    with pytest.raises(ValueError, match='auxiliary tensor'):
        validate_loader_coverage(path, coverage)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.experts.1.down_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ],
)
def test_metadata_preflight_rejects_missing_target_component(next_checkpoint, name):
    from b12x.integration.vllm.checkpoint import audit

    path, write = next_checkpoint
    write(lambda records: records.pop(name))
    with pytest.raises(ValueError, match="target tensor coverage"):
        audit(path)


def test_metadata_preflight_rejects_layout_and_global_scale_changes(next_checkpoint):
    import struct
    from b12x.integration.vllm.checkpoint import audit, inventory

    path, write = next_checkpoint
    key = "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"
    tensors, _, _ = inventory(path)
    with (path / "model.safetensors").open("r+b") as stream:
        stream.seek(tensors[key]["offset"])
        stream.write(struct.pack("<f", 2.0))
    with pytest.raises(ValueError, match="unequal gate/up"):
        audit(path, check_values=True)
    write(lambda records: records[key].update(shape=[1]))
    with pytest.raises(ValueError, match="layout mismatch"):
        audit(path)


def test_metadata_preflight_rejects_unindexed_data_and_truncation(next_checkpoint):
    from b12x.integration.vllm.checkpoint import audit

    path, write = next_checkpoint
    write(
        lambda records: records.update(
            {"model.unhandled.weight": dict(shape=[1], dtype="BF16")}
        )
    )
    with pytest.raises(ValueError, match="unexpected"):
        audit(path)
    write()
    shard = path / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="shard length"):
        audit(path)


def test_metadata_preflight_rejects_conflicting_quantization(next_checkpoint):
    from b12x.integration.vllm.checkpoint import audit

    path, _ = next_checkpoint
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["ignore"].append("*.mlp.experts.*")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="metadata disagrees"):
        audit(path)


def test_metadata_range_reader_never_accepts_full_shard_fallback(monkeypatch):
    from scripts import inspect_expert_cache_checkpoint as script

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            raise AssertionError("full shard response was read")

    monkeypatch.setattr(script, "urlopen", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="bounded checkpoint range"):
        script.fetch("https://example.invalid/shard", 8, (0, 7))


def test_metadata_export_hash_and_index_bind_every_header(next_checkpoint):
    import hashlib
    from b12x.integration.vllm.checkpoint import audit, read_header

    path, _ = next_checkpoint
    raw = read_header(path / "model.safetensors")
    (path / "headers").mkdir()
    header = path / "headers" / "model.safetensors.json"
    header.write_bytes(raw)
    (path / "header-manifest.json").write_text(
        json.dumps(
            dict(
                repo="owner/model",
                revision="a" * 40,
                shards=[
                    dict(
                        shard="model.safetensors",
                        file_bytes=(path / "model.safetensors").stat().st_size,
                        header_bytes=len(raw),
                        header_sha256=hashlib.sha256(raw).hexdigest(),
                    )
                ],
            )
        )
    )
    (path / "model.safetensors").unlink()
    assert audit(path, headers_only=True)["revision"] == "a" * 40
    with pytest.raises(ValueError, match="requires local checkpoint"):
        audit(path, headers_only=True, check_values=True)
    header.write_bytes(raw.replace(b"BF16", b"FFFF", 1))
    with pytest.raises(ValueError, match="header export identity"):
        audit(path, headers_only=True)


def test_metadata_rejects_unbounded_header_and_shard_traversal(tmp_path):
    import struct
    from b12x.integration.vllm.checkpoint import HEADER_LIMIT, read_header, shard_name

    path = tmp_path / "large.safetensors"
    path.write_bytes(struct.pack("<Q", HEADER_LIMIT + 1))
    with pytest.raises(ValueError, match="bounded read"):
        read_header(path)
    with pytest.raises(ValueError, match="basename"):
        shard_name("../outside.safetensors")


@pytest.fixture
def qwen4_checkpoint(tmp_path):
    """Small main/PLE/vision/MTP inventory with the pinned NVIDIA recipes."""
    import json
    import math
    import struct
    from b12x.integration.vllm.checkpoint import DTYPE_BYTES
    from b12x.integration.vllm.checkpoint_qwen4 import expected_qwen4

    text = dict(
        dtype="bfloat16",
        hidden_act="silu",
        norm_topk_prob=True,
        hidden_size=128,
        moe_intermediate_size=128,
        num_experts=2,
        num_hidden_layers=2,
        num_experts_per_tok=2,
        vocab_size=8,
        layer_types=["linear_attention", "full_attention"],
        shared_expert_intermediate_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        hc_count=4,
        hc_lowrank=16,
        output_gate_type="sigmoid",
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        mtp_num_hidden_layers=1,
        mtp=dict(num_hidden_layers=1, layer_types=["full_attention"]),
        ple_layer_ids=[2],
        ple_embed_dim=128,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
    )
    nvfp4 = dict(dynamic=False, num_bits=4, type="float", group_size=16)
    block = dict(dynamic=False, num_bits=8, type="float", group_size=128)
    routed = [f"model.language_model.layers.{l}.mlp.experts" for l in range(2)]
    ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    mtp = "mtp.layers.0.mlp.experts"
    algorithms = {n: dict(quant_algo="NVFP4", group_size=16) for n in routed}
    algorithms.update(
        {
            ple: dict(quant_algo="FP8"),
            mtp: dict(quant_algo="FP8_BLOCK_SCALES", group_size=128),
        }
    )
    groups = dict(
        routed=dict(targets=routed, weights=nvfp4, input_activations=nvfp4),
        ple=dict(targets=[ple], weights=dict(dynamic=False, num_bits=8, type="float")),
        draft=dict(
            targets=[mtp], weights=block, input_activations=dict(block, dynamic=True)
        ),
    )
    config = dict(
        architectures=["Qwen4ExpForConditionalGeneration"],
        model_type="qwen4_exp",
        dtype="bfloat16",
        text_config=text,
        vision_config=dict(
            hidden_size=16,
            intermediate_size=32,
            depth=1,
            in_channels=3,
            temporal_patch_size=2,
            patch_size=2,
            num_position_embeddings=8,
            spatial_merge_size=2,
            out_hidden_size=128,
        ),
        quantization_config=dict(
            quant_method="modelopt",
            quant_algo="MIXED_PRECISION",
            ignore=[],
            quantized_layers=algorithms,
            config_groups=groups,
        ),
    )
    quant = dict(
        quantization=dict(
            quant_algo="MIXED_PRECISION",
            group_size=16,
            exclude_modules=[],
            quantized_layers=algorithms,
        )
    )
    expected = expected_qwen4(config, quant)

    def write(change=None):
        records = {n: dict(v) for n, v in expected.items()}
        if change:
            change(records)
        payload, header = bytearray(), {}
        for name, v in records.items():
            size = math.prod(v["shape"]) * DTYPE_BYTES[v["dtype"]]
            data = struct.pack("<f", 1.0) if v["dtype"] == "F32" else bytes(size)
            header[name] = dict(
                shape=v["shape"],
                dtype=v["dtype"],
                data_offsets=[len(payload), len(payload) + size],
            )
            payload.extend(data)
        raw = json.dumps(header).encode()
        (tmp_path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(raw)) + raw + payload
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                dict(
                    metadata=dict(total_size=len(payload)),
                    weight_map={n: "model.safetensors" for n in header},
                )
            )
        )
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant))

    write()
    return tmp_path, config, quant, write


def test_qwen4_audit_keeps_ple_and_mtp_outside_routed_cache(qwen4_checkpoint):
    from b12x.integration.vllm.checkpoint import audit
    from b12x.moe.fused_moe._cache_preparation import tier_layout

    path, _, _, _ = qwen4_checkpoint
    r = audit(path, check_values=True)
    assert r["status"] == "metadata_audited_cache_integration_required"
    assert r["routed_weight_lower_bound_bytes"] == 2 * 2 * 3 * 128 * 128 // 2
    assert r["canonical_mapped_bytes"] == 2 * tier_layout(2, 128, 128)[1]
    assert r["promotion_payload_bytes"] == 3 * 128 * 128 * 9 // 16 + 8
    assert r["classes"]["optional_mtp"]["bytes"] > 0
    assert r["classes"]["optional_mtp"]["packed_bytes"] == 0
    assert (
        r["complete_checkpoint_tensor_bytes"]
        == r["required_target_bytes"] + r["classes"]["optional_mtp"]["bytes"]
    )
    assert (
        r["text_only_target_bytes"]
        == r["required_target_bytes"] - r["classes"]["vision"]["bytes"]
    )
    assert (
        r["host_with_ple_lower_bound_bytes"]
        == r["host_expert_lower_bound_bytes"] + r["ple_table_bytes"]
    )
    assert r["global_scale_values"].startswith("all_routed")


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
        "model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale_2",
        "mtp.layers.0.mlp.experts.1.down_proj.weight_scale_inv",
    ],
)
def test_qwen4_inventory_rejects_missing_component(qwen4_checkpoint, name):
    from b12x.integration.vllm.checkpoint import audit

    path, _, _, write = qwen4_checkpoint
    write(lambda records: records.pop(name))
    with pytest.raises(ValueError, match="coverage mismatch"):
        audit(path)


def test_qwen4_mixed_quantization_alias_and_conflict(qwen4_checkpoint):
    from b12x.integration.vllm.checkpoint import audit

    path, config, quant, write = qwen4_checkpoint
    import copy

    config["quantization_config"]["quantized_layers"] = copy.deepcopy(
        quant["quantization"]["quantized_layers"]
    )
    entry = config["quantization_config"]["quantized_layers"][
        "mtp.layers.0.mlp.experts"
    ]
    entry["quant_algo"] = "FP8_PB_WO"
    write()
    assert audit(path)["status"] == "metadata_audited_cache_integration_required"
    entry["group_size"] = 64
    write()
    with pytest.raises(ValueError, match="mixed-precision metadata"):
        audit(path)


def test_qwen4_rejects_fp8_ple_relabelled_as_bf16(qwen4_checkpoint):
    from b12x.integration.vllm.checkpoint import audit

    path, _, _, write = qwen4_checkpoint
    name = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    write(lambda records: records[name].update(dtype="BF16"))
    with pytest.raises(ValueError, match="layout mismatch"):
        audit(path)
