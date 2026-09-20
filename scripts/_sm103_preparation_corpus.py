"""Representative SM103 preparation declarations for offline compiler coverage.

These cases construct production metadata. Their CPU weight fixtures supply
storage contracts only and do not establish numerical correctness.
"""
from __future__ import annotations
from b12x.preparation import DetectedDevice, DeviceIdentity, FrozenMapping

IDENTITY = DeviceIdentity(vendor="nvidia", product_name="NVIDIA B300",
                          compute_capability=(10, 3), sm_count=148)
DEVICE = DetectedDevice(0, IDENTITY, "synthetic-sm103", 227 * 1024, 228 * 1024)
CASES = (
    *(f"dense:{recipe}" for recipe in ("nvfp4", "mxfp4", "mxfp8", "tensor_fp8", "block_fp8",
                                        "mxfp6_e2m3", "mxfp6_e3m2", "w6a8_e2m3", "w6a8_e3m2")),
    "block_linear:32", "block_linear:128", "fp8_workspace:tensor_fp8", "fp8_workspace:block_fp8",
    "trellis:uniform", "trellis:coupled", "trellis:mixed", "trellis:grouped", "trellis:btx", "trellis:btx_coupled",
    "packed:nvfp4", "packed:mxfp8", "packed:mxfp8_fp16", "prefill:gdn", "prefill:kda", "wo:plain", "wo:inv_rope", "vocab",
    "mtp:rms_concat", "mtp:rms_streams_fp8", "mtp:qwen_multistream",
    "gdn:kda", "gdn:qwen", "glm:1", "glm:2", "compressed:deepseek_v4", "compressed:deepseek_v41",
    "mhc:pre", "mhc:post_pre",
    *(f"mhc:{op}:{hidden}:{capacity}:{variant}"
      for op in ("pre", "post_pre") for hidden in (4096, 5120, 7168)
      for capacity in (8, 389) for variant in ("plain", "norm", "lagged")),
    "mhc:post:5120:17:plain", "mhc:collapse:5120:17:plain", "mhc:pre:5120:17:broadcast", "moe:nvfp4", "moe:residency", "moe:routing_profile", "dsa:decode", "dsa:prefill",
    "hyperconnection:grouped_rmsnorm", "hyperconnection:gate_mean", "moe:residency_updates", "moe:routing_profile_extent", "moe:routing_health",
)


def declare(case):
    import torch
    family, _, recipe = case.partition(":")
    if case in ("moe:routing_profile", "moe:routing_profile_extent", "moe:routing_health"):
        from b12x.moe import fused_moe as op
        return op.plan_routing_profile(op.RoutingProfileQuery(layers=(("compile", 384),),
            max_tokens=128, max_top_k=8, phases=("decode",) if case.endswith("_health") else ("decode", "verify"),
            health_summary=case.endswith("_health"),
            runtime_token_limit=case.endswith(("_extent", "_health"))))
    if case in ("moe:residency", "moe:residency_updates"):
        from b12x.moe import fused_moe as op
        weight_plan = op.plan_weights(
            source=op.PackedSource(format="fp4_e8m0_k32"),
            activation=op.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=op.MoEGeometry(num_experts=4, hidden_size=256, intermediate_size=256),
            constraints=op.WeightPlanConstraints(required_packing="source_native"))
        weights = op.PackedWeights(torch.empty(4, 512, 128, dtype=torch.uint8),
            torch.empty(4, 256, 128, dtype=torch.uint8),
            torch.empty(4, 512, 8, dtype=torch.uint8), torch.empty(4, 256, 8, dtype=torch.uint8),
            torch.ones(4), torch.ones(4), checkpoint_fingerprint="synthetic", layer_name="compile")
        return op.plan_execution(experts=weight_plan, weights=weights,
            capacity=op.ExecutionCapacity(max_tokens=17, top_k=3),
            placement=op.ExpertResidencyPlan(total_experts=4, hbm_expert_ids=(0, 2), grace_expert_ids=(1, 3),
                layer="compile", model_fingerprint="synthetic", workload="compile", provenance="compiler corpus"),
            memory_budget=op.ExpertMemoryBudget(hbm_bytes=2**30, grace_bytes=2**30),
            updates=op.ResidencyUpdateCapacity(max_pairs=2) if case.endswith("_updates") else None)
    if family == "dense":
        from b12x.gemm._preparation import plan
        from b12x.gemm._tuning import DenseGemmQuery
        return plan(DenseGemmQuery(
            recipe=recipe, entry_point="gemm.mm", max_rows=17, in_features=256,
            out_features=256, batch=1, output_dtype="bfloat16", output_mode="provided",
            weight_storage="packed" if recipe.startswith(("mxfp6", "w6a8")) else "native",
        ))
    if family == "block_linear":
        from b12x.gemm import block_fp8_linear as op
        block = int(recipe)
        return op.plan(op.Caps(device="cuda:0", max_tokens=17, in_features=256,
                               out_features=256, block_size=(block, block)))
    if family == "fp8_workspace":
        from b12x.gemm.blockscaled import plan, FixedBlockscaledQuery
        return plan(FixedBlockscaledQuery(
            recipe=recipe, call_kind="packed" if recipe == "tensor_fp8" else "serialized",
            max_rows=17, in_features=160 if recipe == "tensor_fp8" else 256,
            padded_in_features=256, out_features=256, input_dtype="float8_e4m3fn",
            output_dtype="float16", expected_m=17, fp8_workspace=True,
            output_mode="provided", workspace_form="provided",
        ))
    if family == "trellis":
        return trellis_compiler_declaration(recipe)
    if family == "packed":
        from b12x.gemm.blockscaled._preparation import plan
        from b12x.gemm.blockscaled._tuning import BlockscaledQuery
        return plan(BlockscaledQuery(recipe="mxfp8" if recipe == "mxfp8_fp16" else recipe,
                                     input_dtype="float16" if recipe == "mxfp8_fp16" else "bfloat16",
                                     num_tokens=8, in_features=256,
                                     padded_in_features=256, out_features=256,
                                     activation_mode="quantized" if recipe == "mxfp8_fp16" else "a16", output_mode="provided"))
    if family == "wo":
        from b12x.gemm import wo_projection as op
        invocation = {"operation": recipe, "dynamic_tokens": True}
        if recipe == "inv_rope":
            invocation.update(heads_per_group=1, nope_dim=96, rope_dim=32)
        return op.plan(op.Caps(device="cuda:0", max_tokens=17, groups=2, group_width=128,
                               rank=128, hidden=256), invocation=FrozenMapping(invocation))
    if family == "vocab":
        from b12x.gemm import bf16_vocab_projection as op
        return op.plan(op.Caps(device="cuda:0", max_tokens=17, in_features=256, out_features=1024))
    if family == "mtp":
        from b12x.sequence import mtp_feedback as op
        streams = {"rms_concat": 1, "rms_streams_fp8": 3, "qwen_multistream": 4}[recipe]
        return op.plan(op.Caps(device="cuda:0", max_tokens=17, streams=streams,
                               hidden_size=2560 if recipe == "qwen_multistream" else 256, contract=recipe))
    if family == "prefill":
        from b12x.sequence import gdn_prefill, kda_prefill
        op = gdn_prefill if recipe == "gdn" else kda_prefill
        heads = dict(key_heads=1, value_heads=3) if recipe == "gdn" else dict(heads=3)
        return op.plan(op.Caps(device="cuda:0", max_tokens=257, max_seqs=4,
                               max_state_slots=31, checkpoint_export=True, **heads))
    if family == "gdn":
        from b12x.sequence import gdn_decode as op
        return op.plan(op.Caps(device="cuda:0", max_tokens=16, max_seqs=4, max_state_slots=31,
                               key_heads=4, value_heads=4 if recipe == "kda" else 12,
                               state_index_columns=4, gate_activation="sigmoid"))
    if family == "glm":
        from b12x.attention import sparse_mla as op
        return op.plan(op.Caps(device="cuda:0", num_q_heads=24, max_q_rows=17, max_width=257,
                               softmax_scale=512**-0.5, kv_dtype=torch.uint8, head_dim=576 if recipe == "1" else 512,
                               v_head_dim=512, model_type=int(recipe), mode="decode"))
    if family == "compressed":
        from b12x.attention import compressed_sparse_mla as op
        from b12x.attention.compressed_sparse_mla._preparation import invocation_from_descriptors
        from b12x.attention._shared.mla.compressed_reference import (
            pack_compressed_sparse_mla_kv_cache_reference, pack_deepseek_v41_cache_reference,
        )
        values = torch.zeros(64, 512, dtype=torch.bfloat16)
        cache = (pack_deepseek_v41_cache_reference(values, page_size=64, cache_kind="swa")
                 if recipe == "deepseek_v41" else pack_compressed_sparse_mla_kv_cache_reference(
                     values[:, :448].contiguous(), values[:, 448:].contiguous(), page_size=64))
        def descriptor(tensor):
            return FrozenMapping(dict(shape=tuple(tensor.shape), stride=tuple(tensor.stride()),
                                      alignment=16, dtype=str(tensor.dtype).removeprefix("torch.")))
        return op.plan(op.Caps(device="cuda:0", num_q_heads=16, max_q_rows=17, max_width=64,
                               swa_width=64, indexed_width=0, swa_page_size=64, cache_format=recipe),
                       invocation=invocation_from_descriptors(
                           q=descriptor(torch.empty(17, 16, 512, dtype=torch.bfloat16)),
                           swa_cache=descriptor(cache), return_lse=True))
    if family == "mhc":
        from b12x.norm import mhc as op
        parts = recipe.split(":")
        operation = parts[0]
        hidden, capacity, variant = (int(parts[1]), int(parts[2]), parts[3]) if len(parts) > 1 else (5120, 17, "norm")
        return op.plan(op.Caps(device="cuda:0", max_tokens=capacity, hidden_size=hidden),
                       invocation=FrozenMapping({"operation": operation, "rms_eps": 1e-6,
                           "hc_eps": 1e-6, "sinkhorn_iters": 20,
                           "has_norm_weight": variant == "norm", "norm_eps": 1e-6,
                           "lagged_mix": variant == "lagged",
                           "expanded_residual": operation == "pre" and variant != "broadcast"}))
    if family == "moe":
        from b12x.moe import fused_moe as op
        return op.plan_execution(experts=make_experts(),
                                 capacity=op.ExecutionCapacity(max_tokens=17, top_k=2))
    if family == "dsa":
        from b12x.attention import dsa_indexer as op
        return op.plan(op.Caps(device="cuda:0", num_q_heads=32, max_q_rows=17,
                               max_page_table_width=64, topk=512, cache_format="mxfp4", mode=recipe))
    if family == "hyperconnection":
        from b12x.norm import hyperconnection as op
        return op.plan(op.Caps(device="cuda:0", max_tokens=17, streams=4, hidden_size=5120),
                       invocation=FrozenMapping({"operation": recipe}))
    raise ValueError(case)



def make_experts(*, device="cpu", e=2, k=256, n=256, w13_layout="w13", mode="a4"):
    import torch
    from b12x.moe import fused_moe
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



def trellis_compiler_declaration(kind):
    """Exercise the production compiler factory from canonical weight-plan metadata.

    Checkpoint byte preparation requires CUDA and is qualified separately by the
    GPU Trellis corpus. This offline declaration supplies no executable weights.
    """
    import torch
    from b12x.moe import fused_moe as op
    e, k, n = 2, 512, 256
    coupled, grouped = kind in {"coupled", "btx_coupled"}, kind == "grouped"
    mixed = kind in {"mixed", "grouped"}
    rate = {"granularity": "per_expert_projection" if mixed else "uniform"}
    if grouped:
        rate["group_size"] = 32
    config = op.TrellisConfig.from_dict({
        "version": 2, "codebook": "mcg" if mixed else "sqg_e4m3", "rate": rate,
        "scale": {name: {"vectors": "per_expert", "gains": "none"}
                  for name in ("input_scales", "intermediate_scales", "output_scales")},
        "transform": {
            "projection": {"kind": "scaled_hadamard", "block_size": 128},
            "expert": {"kind": "coupled_hadamard", "pre_block_size": 512,
                       "post_block_size": 128, "draw_granularity": "per_expert"}
                      if coupled else {"kind": "none"},
        },
    })
    if kind.startswith("btx"):
        from b12x.moe._shared.btx_schema import BtxManifest
        from b12x.moe._shared.kernels.w4a16.btx_synth import BtxSynthConfig, _manifest_dict
        rates = torch.tensor([[0x24, 0x43]], dtype=torch.uint8)
        metadata = _manifest_dict(BtxSynthConfig(
            codebook="sqg_e4m3", num_experts=e, hidden_size=k, intermediate_size=n,
            moe_layer_indices=(0,), rate_tables={0: (rates, rates)}, coupled=coupled,
            pre_block=512 if coupled else None, post_block=128 if coupled else None,
            per_expert_input_rotations=True, extent_alignment_slots=8,
        ))
        # Offline metadata describes an absent checkpoint; it cannot be materialized.
        metadata["layers"] = {"0": {"file": "offline.safetensors", "sha256": "0" * 64}}
        config = op.BtxSource(manifest=BtxManifest.from_dict(metadata))
    plan = op.plan_weights(source=config,
        activation=op.ActivationSpec(mode="a16", nonlinearity="situ" if coupled else "silu",
                                     io_dtype=torch.bfloat16),
        geometry=op.MoEGeometry(num_experts=e, hidden_size=k, intermediate_size=n))
    from types import SimpleNamespace
    from b12x.preparation import Plan, MemoryRequirements
    from b12x._lib.compile_pool import CompileJob
    from b12x.moe.fused_moe._preparation import _weight_payload, _lower_caps
    from b12x.moe.fused_moe._impl import plan_tp_moe_scratch
    from b12x.moe.fused_moe._sm103 import query_for_weight_plan
    from b12x.moe.fused_moe._tuning import TUNING
    query = query_for_weight_plan(plan._impl, quant_mode="w4a16", num_tokens=8, num_topk=2)
    payload = _weight_payload(SimpleNamespace(plan=plan))

    def memory(config, device):
        caps = _lower_caps(query, config, plan._impl, torch.device("cuda", device.ordinal))
        scratch = plan_tp_moe_scratch(caps, prewarm_launches=False)
        return MemoryRequirements(scratch=scratch.scratch_specs())

    def no_execution(*args):
        raise RuntimeError("offline Trellis corpus has no prepared checkpoint weights")

    return Plan(contract=TUNING, query=query, _device=torch.device("cuda", 0),
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.moe.fused_moe._preparation:compile_fused_moe",
            TUNING.encode_query(query), TUNING.encode_config(config), payload, (1, 1), device.ordinal,
        ),), _memory_requirements=memory, _materialize=no_execution)
