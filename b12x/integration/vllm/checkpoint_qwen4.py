"""Metadata inventory for Qwen4Exp mixed-precision checkpoints.

This schema audits storage and numerical prerequisites. It does not enable the
expert-cache loader for ModelOpt mixed precision or qualify PLE/cache ownership.
"""

import fnmatch


def expected_qwen4(config, quant):
    """Describe target, vision and optional draft tensors without device storage."""
    from b12x.sequence.ple_hash.reference import ple_table_geometry

    c = config["text_config"]
    q, external = config["quantization_config"], quant["quantization"]
    if (
        config.get("architectures")
        not in (
            ["Qwen4ExpForConditionalGeneration"],
            ["Qwen3_8FlashNextForConditionalGeneration"],
        )
        or config.get("model_type") not in ("qwen4_exp", "qwen3_8_flash_next")
        or c.get("hidden_act") != "silu"
        or c.get("attention_bias", False)
        or c.get("decoder_sparse_step", 1) != 1
        or c.get("mlp_only_layers", [])
        or config.get("tie_word_embeddings", False)
        or c.get("tie_word_embeddings", False)
        or not c.get("norm_topk_prob")
        or c.get("output_gate_type") != "sigmoid"
        or c.get("dtype", config.get("dtype")) != "bfloat16"
    ):
        raise ValueError("unsupported Qwen4Exp architecture or numerical recipe")
    h, i, e, layers = (
        c[k]
        for k in (
            "hidden_size",
            "moe_intermediate_size",
            "num_experts",
            "num_hidden_layers",
        )
    )
    if (
        min(h, i, e, layers, c["num_experts_per_tok"]) <= 0
        or h % 128
        or i % 128
        or c["num_experts_per_tok"] > e
        or len(c["layer_types"]) != layers
        or c["hc_count"] <= 1
    ):
        raise ValueError("unsupported Qwen4Exp expert or residual geometry")
    mtp_layers = c["mtp_num_hidden_layers"]
    if (
        mtp_layers != c["mtp"]["num_hidden_layers"]
        or c["mtp"]["layer_types"] != ["full_attention"] * mtp_layers
        or c.get("mtp_use_dedicated_embeddings", False)
    ):
        raise ValueError("unsupported optional MTP inventory")
    algorithms = {
        f"model.language_model.layers.{l}.mlp.experts": dict(
            quant_algo="NVFP4", group_size=16
        )
        for l in range(layers)
    }
    algorithms.update(
        {
            f"mtp.layers.{l}.mlp.experts": dict(
                quant_algo="FP8_BLOCK_SCALES", group_size=128
            )
            for l in range(mtp_layers)
        }
    )
    ple_ids = c["ple_layer_ids"]
    if len(set(ple_ids)) != len(ple_ids) or any(not 1 <= l <= layers for l in ple_ids):
        raise ValueError("invalid PLE layer inventory")
    for l in ple_ids:
        algorithms[
            f"model.language_model.layers.{l - 1}.ple.ple_embedding.ngram_embedding"
        ] = dict(quant_algo="FP8")

    def normalize(entries):
        return {
            name: dict(
                info,
                quant_algo={"FP8_PB_WO": "FP8_BLOCK_SCALES"}.get(
                    info["quant_algo"], info["quant_algo"]
                ),
            )
            for name, info in entries.items()
        }

    if (
        q.get("quant_method") != "modelopt"
        or q.get("quant_algo") != "MIXED_PRECISION"
        or external.get("quant_algo") != "MIXED_PRECISION"
        or external.get("group_size") != 16
        or normalize(q.get("quantized_layers", {})) != algorithms
        or normalize(external.get("quantized_layers", {})) != algorithms
        or sorted(q["ignore"]) != sorted(external["exclude_modules"])
    ):
        raise ValueError(
            "Qwen4Exp mixed-precision metadata disagrees with the inventory"
        )
    schemes = {
        "NVFP4": dict(
            weights=dict(dynamic=False, num_bits=4, type="float", group_size=16),
            input_activations=dict(
                dynamic=False, num_bits=4, type="float", group_size=16
            ),
        ),
        "FP8_BLOCK_SCALES": dict(
            weights=dict(dynamic=False, num_bits=8, type="float", group_size=128),
            input_activations=dict(
                dynamic=True, num_bits=8, type="float", group_size=128
            ),
        ),
        "FP8": dict(weights=dict(dynamic=False, num_bits=8, type="float")),
    }
    declared = {}
    for group in q["config_groups"].values():
        for name in group["targets"]:
            if name in declared or name not in algorithms:
                raise ValueError("duplicate or unsupported quantization target")
            declared[name] = {k: v for k, v in group.items() if k != "targets"}
    if declared != {n: schemes[a["quant_algo"]] for n, a in algorithms.items()}:
        raise ValueError("Qwen4Exp quantization schemes disagree")

    expected = {}

    def tensor(name, shape, kind, dtype="BF16"):
        expected[name] = dict(shape=list(shape), dtype=dtype, kind=kind)

    def weight(name, shape, kind, bias=False):
        tensor(name + ".weight", shape, kind)
        if bias:
            tensor(name + ".bias", shape[:1], kind)

    hc, rank = h * c["hc_count"], c["hc_lowrank"]

    def residual(p, kind, inject=True):
        weight(p + ".hc_norm", (hc,), kind)
        weight(p + ".input_mix_weight_down", (rank, hc), kind)
        weight(p + ".input_mix_weight_up", (hc, rank), kind)
        if inject:
            weight(p + ".block_inject_weight", (c["hc_count"], hc), kind)

    def attention(p, kind):
        heads, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        for proj, shape in dict(
            q=(2 * heads * d, h), k=(kv * d, h), v=(kv * d, h), o=(h, heads * d)
        ).items():
            weight(f"{p}.{proj}_proj", shape, kind)
        for proj in ("q", "k"):
            weight(f"{p}.{proj}_norm", (d,), kind)
        kd = c["indexer_head_dim"]
        weight(
            p + ".indexer.index_qk_proj",
            ((c["indexer_n_heads"] + c["indexer_kv_heads"]) * kd, h),
            kind,
        )
        for proj in ("q", "k"):
            weight(f"{p}.indexer.{proj}_layernorm", (kd,), kind)

    weight("model.language_model.embed_tokens", (c["vocab_size"], h), "embedding_head")
    weight("lm_head", (c["vocab_size"], h), "embedding_head")
    residual("model.language_model.hyper_connection_mixer", "hyperconnection", False)
    for draft, count in ((False, layers), (True, mtp_layers)):
        for l in range(count):
            p = f"mtp.layers.{l}" if draft else f"model.language_model.layers.{l}"

            def kind(normal):
                return "optional_mtp" if draft else normal

            for name in ("attn_hyper_connection", "mlp_hyper_connection"):
                residual(p + "." + name, kind("hyperconnection"))
            weight(p + ".mlp.gate", (e, h), kind("router"))
            weight(p + ".mlp.shared_expert_gate", (1, h), kind("shared"))
            si = c["shared_expert_intermediate_size"]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                out, inp = (h, i) if proj == "down_proj" else (i, h)
                shared_shape = (h, si) if proj == "down_proj" else (si, h)
                weight(p + ".mlp.shared_expert." + proj, shared_shape, kind("shared"))
                for expert in range(e):
                    n = f"{p}.mlp.experts.{expert}.{proj}"
                    if draft:
                        tensor(n + ".weight", (out, inp), "optional_mtp", "F8_E4M3")
                        tensor(
                            n + ".weight_scale_inv",
                            ((out + 127) // 128, (inp + 127) // 128),
                            "optional_mtp",
                        )
                    else:
                        tensor(n + ".weight", (out, inp // 2), "routed", "U8")
                        tensor(
                            n + ".weight_scale", (out, inp // 16), "routed", "F8_E4M3"
                        )
                        for suffix in ("weight_scale_2", "input_scale"):
                            tensor(n + "." + suffix, (), "routed", "F32")
            if draft or c["layer_types"][l] == "full_attention":
                attention(p + ".self_attn", kind("qsa"))
            elif c["layer_types"][l] == "linear_attention":
                nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
                kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
                a = p + ".linear_attn"
                for name in ("A_log", "dt_bias"):
                    tensor(a + "." + name, (nv,), "gdn")
                weight(
                    a + ".conv1d",
                    (2 * nk * kd + nv * vd, 1, c["linear_conv_kernel_dim"]),
                    "gdn",
                )
                for name, rows in (
                    ("a", nv),
                    ("b", nv),
                    ("qkv", 2 * nk * kd + nv * vd),
                    ("z", nv * vd),
                ):
                    weight(f"{a}.in_proj_{name}", (rows, h), "gdn")
                weight(a + ".norm", (vd,), "gdn")
                weight(a + ".out_proj", (h, nv * vd), "gdn")
            else:
                raise ValueError("unsupported Qwen4Exp attention type")
    for name, shape in (
        ("fc_embedding", (h, h)),
        ("fc_hidden", (h, h)),
        ("pre_fc_norm_embedding", (h,)),
        ("pre_fc_norm_hidden", (hc,)),
    ):
        weight("mtp." + name, shape, "optional_mtp")
    residual("mtp.hyper_connection_mixer", "optional_mtp", False)

    heads = (c["ngram_size"] - 1) * c["heads_per_ngram"]
    dim, parts = c["ple_embed_dim"], c["split_ngram_parts"]
    if min(heads, parts) <= 0 or dim % heads:
        raise ValueError("unsupported PLE lookup geometry")
    for ordinal, l in enumerate(sorted(ple_ids)):
        p = f"model.language_model.layers.{l - 1}.ple"
        weight(p + ".key_proj", (hc, dim), "ple_aux")
        weight(p + ".value_proj", (h, dim), "ple_aux")
        weight(p + ".conv1d", (hc, 1, c["ple_conv_kernel_size"]), "ple_aux")
        for name in ("norm_conv", "norm_key", "norm_query"):
            weight(p + "." + name, (hc,), "ple_aux")
        for name, n in (
            ("layer_multipliers", c["ngram_size"]),
            ("ngram_heads_offsets", heads),
            ("ngram_heads_vocab_sizes", heads),
        ):
            tensor(p + ".ple_embedding." + name, (n,), "ple_aux", "I64")
        primes, _ = ple_table_geometry(
            base_size=c["ngram_vocab_size_base"],
            dense_layer_ordinal=ordinal,
            total_heads=heads,
        )
        alignment = c["make_ngram_vocab_size_divisible_by"]
        rows = (sum(primes.tolist()) + alignment - 1) // alignment * alignment
        shard_rows = (rows + parts - 1) // parts
        for part in range(parts):
            tensor(
                f"{p}.ple_embedding.ngram_embedding.shard_{part}.weight",
                (min(shard_rows, rows - part * shard_rows), dim // heads),
                "ple_table",
                "F8_E4M3",
            )
        tensor(p + ".ple_embedding.ngram_embedding.weight_scale", (1,), "ple_aux")

    v = config["vision_config"]
    vh, vi = v["hidden_size"], v["intermediate_size"]
    if v.get("deepstack_visual_indexes", []):
        raise ValueError("deepstack vision inventory is not supported by this audit")
    weight(
        "model.visual.patch_embed.proj",
        (
            vh,
            v["in_channels"],
            v["temporal_patch_size"],
            v["patch_size"],
            v["patch_size"],
        ),
        "vision",
        True,
    )
    weight("model.visual.pos_embed", (v["num_position_embeddings"], vh), "vision")
    for l in range(v["depth"]):
        p = f"model.visual.blocks.{l}"
        for name, shape in (
            ("attn.proj", (vh, vh)),
            ("attn.qkv", (3 * vh, vh)),
            ("mlp.linear_fc1", (vi, vh)),
            ("mlp.linear_fc2", (vh, vi)),
            ("norm1", (vh,)),
            ("norm2", (vh,)),
        ):
            weight(p + "." + name, shape, "vision", True)
    merged = vh * v["spatial_merge_size"] ** 2
    for name, shape in (
        ("norm", (vh,)),
        ("linear_fc1", (merged, merged)),
        ("linear_fc2", (v["out_hidden_size"], merged)),
    ):
        weight("model.visual.merger." + name, shape, "vision", True)
    for name, item in expected.items():
        if (
            item["kind"] == "routed"
            and name.endswith(".weight")
            and any(
                fnmatch.fnmatchcase(name.removesuffix(".weight"), p)
                for p in q["ignore"]
            )
        ):
            raise ValueError(
                "routed expert is excluded from declared NVFP4 quantization"
            )
    return expected
