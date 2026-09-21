"""Bounded, CPU-only checkpoint inventory for the Qwen3-Next NVFP4 cache lane.

Header acceptance proves tensor coverage and layout, not execution correctness.
Full checkpoint hashing, native value validation and live memory admission remain
separate loader/preparation gates. Optional MTP tensors never enter target bytes.
"""

from collections import defaultdict
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import struct


HEADER_LIMIT = 64 << 20
DTYPE_BYTES = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint key: {key}")
        result[key] = value
    return result


def decode_json(data):
    return json.loads(data, object_pairs_hook=_unique)


def read_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(HEADER_LIMIT + 1)
    if len(data) > HEADER_LIMIT:
        raise ValueError(f"checkpoint metadata exceeds bounded read: {path}")
    return decode_json(data)


def shard_name(name):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".safetensors")
    ):
        raise ValueError("checkpoint shard must be a safetensors basename")
    return name


def read_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= HEADER_LIMIT:
            raise ValueError("safetensors header exceeds bounded read")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated safetensors header")
    return data


def inventory(directory, *, headers_only=False):
    """Validate indexed shards without mapping or reading weight payloads.

    A header export requires its retained size/hash manifest. Its identity is
    explicitly metadata-only; it cannot attest the checkpoint's weight bytes.
    """
    directory = Path(directory)
    index = read_json(directory / "model.safetensors.index.json")
    mapping = index["weight_map"]
    manifest = read_json(directory / "header-manifest.json") if headers_only else None
    entries = {r["shard"]: r for r in manifest["shards"]} if manifest else {}
    tensors, shards = {}, []
    for name in sorted(set(mapping.values())):
        shard_name(name)
        if headers_only:
            path = directory / "headers" / (name + ".json")
            with path.open("rb") as stream:
                data = stream.read(HEADER_LIMIT + 1)
            entry = entries[name]
            file_size = entry["file_bytes"]
            if (
                len(data) != entry["header_bytes"]
                or hashlib.sha256(data).hexdigest() != entry["header_sha256"]
            ):
                raise ValueError(f"header export identity mismatch: {name}")
        else:
            path = directory / name
            data, file_size = read_header(path), path.stat().st_size
        if len(data) > HEADER_LIMIT:
            raise ValueError("safetensors header exceeds bounded read")
        header = decode_json(data)
        offsets = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in tensors or mapping.get(key) != name:
                raise ValueError(f"checkpoint index/header mismatch: {key}")
            shape, dtype = value["shape"], value["dtype"]
            if dtype not in DTYPE_BYTES or any(
                type(d) is not int or d < 0 for d in shape
            ):
                raise ValueError(f"unsupported tensor layout: {key}")
            begin, end = value["data_offsets"]
            if (
                type(begin) is not int
                or type(end) is not int
                or begin < 0
                or end - begin != math.prod(shape) * DTYPE_BYTES[dtype]
            ):
                raise ValueError(f"invalid tensor extent: {key}")
            offsets.append((begin, end))
            tensors[key] = dict(
                shape=shape,
                dtype=dtype,
                bytes=end - begin,
                shard=name,
                offset=8 + len(data) + begin,
            )
        tail = 0
        for begin, end in sorted(offsets):
            if begin != tail:
                raise ValueError(f"overlapping or incomplete shard offsets: {name}")
            tail = end
        if tail + 8 + len(data) != file_size:
            raise ValueError(f"shard length mismatch: {name}")
        shards.append(
            dict(
                shard=name,
                file_bytes=file_size,
                header_bytes=len(data),
                header_sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    if set(tensors) != set(mapping):
        raise ValueError("checkpoint index names missing from headers")
    if sum(t["bytes"] for t in tensors.values()) != index["metadata"]["total_size"]:
        raise ValueError("checkpoint index byte total mismatch")
    return tensors, shards, manifest


def expected_qwen3_next(config, quant):
    """Describe the target model implemented by the maintained companion.

    Every target tensor is required. Auxiliary MTP tensors are classified
    separately because Qwen3NextForCausalLM explicitly excludes that prefix.
    """
    c = config
    if (
        c.get("architectures") != ["Qwen3NextForCausalLM"]
        or c.get("model_type") != "qwen3_next"
        or c.get("dtype", c.get("torch_dtype")) != "bfloat16"
        or c.get("hidden_act") != "silu"
        or c.get("attention_bias", False)
        or c.get("decoder_sparse_step", 1) != 1
        or c.get("mlp_only_layers", [])
        or c.get("tie_word_embeddings", False)
        or not c.get("norm_topk_prob")
    ):
        raise ValueError(
            "unsupported Qwen3-Next target architecture or numerical recipe"
        )
    q, external = c["quantization_config"], quant["quantization"]
    groups = q.get("config_groups", {})
    schemes = [
        g.get(k, {}) for g in groups.values() for k in ("weights", "input_activations")
    ]
    if (
        q.get("quant_algo") != "NVFP4"
        or external.get("quant_algo") != "NVFP4"
        or external.get("group_size") != 16
        or len(groups) != 1
        or any(g.get("targets") != ["Linear"] for g in groups.values())
        or any(
            s != dict(dynamic=False, num_bits=4, type="float", group_size=16)
            for s in schemes
        )
    ):
        raise ValueError("checkpoint requires native ModelOpt NVFP4 K16")
    if sorted(q["ignore"]) != sorted(external["exclude_modules"]):
        raise ValueError("quantization exclusion metadata disagrees")
    if q.get("quant_method") != "modelopt":
        raise ValueError("checkpoint requires ModelOpt quantization metadata")
    h, i, e = c["hidden_size"], c["moe_intermediate_size"], c["num_experts"]
    layers, topk = c["num_hidden_layers"], c["num_experts_per_tok"]
    if min(h, i, e, layers, topk) <= 0 or h % 128 or i % 128 or topk > e:
        raise ValueError("unsupported routed expert geometry")
    if len(c["layer_types"]) != layers:
        raise ValueError("attention layer inventory does not cover target model")
    expected = {}

    def tensor(name, shape, dtype, kind):
        expected[name] = dict(shape=list(shape), dtype=dtype, kind=kind)

    def nvfp4(name, out, inp, kind):
        if inp % 16:
            raise ValueError("NVFP4 input extent must be divisible by 16")
        tensor(name + ".weight", (out, inp // 2), "U8", kind)
        tensor(name + ".weight_scale", (out, inp // 16), "F8_E4M3", kind)
        for suffix in ("input_scale", "weight_scale_2"):
            tensor(name + "." + suffix, (), "F32", kind)

    tensor("model.embed_tokens.weight", (c["vocab_size"], h), "BF16", "embedding_head")
    tensor("lm_head.weight", (c["vocab_size"], h), "BF16", "embedding_head")
    tensor("model.norm.weight", (h,), "BF16", "norm")
    for layer, attention in enumerate(c["layer_types"]):
        p = f"model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensor(f"{p}.{norm}.weight", (h,), "BF16", "norm")
        tensor(p + ".mlp.gate.weight", (e, h), "BF16", "router")
        tensor(p + ".mlp.shared_expert_gate.weight", (1, h), "BF16", "shared_gate")
        for expert in range(e + 1):
            shared = expert == e
            name = p + (".mlp.shared_expert" if shared else f".mlp.experts.{expert}")
            intermediate = c["shared_expert_intermediate_size"] if shared else i
            kind = "shared" if shared else "routed"
            for proj in ("gate_proj", "up_proj"):
                nvfp4(name + "." + proj, intermediate, h, kind)
            nvfp4(name + ".down_proj", h, intermediate, kind)
        if attention == "full_attention":
            a = p + ".self_attn"
            d, heads, kv = (
                c["head_dim"],
                c["num_attention_heads"],
                c["num_key_value_heads"],
            )
            for proj, rows in (("q", 2 * heads * d), ("k", kv * d), ("v", kv * d)):
                tensor(f"{a}.{proj}_proj.weight", (rows, h), "BF16", "attention")
            for proj in ("q", "k"):
                tensor(f"{a}.{proj}_norm.weight", (d,), "BF16", "attention")
            for proj in ("k", "v"):
                tensor(f"{a}.{proj}_proj.{proj}_scale", (), "F32", "attention")
            nvfp4(a + ".o_proj", h, heads * d, "attention")
        elif attention == "linear_attention":
            a = p + ".linear_attn"
            nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
            kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
            for name in ("A_log", "dt_bias"):
                tensor(a + "." + name, (nv,), "BF16", "recurrent")
            tensor(
                a + ".conv1d.weight",
                (2 * nk * kd + nv * vd, 1, c["linear_conv_kernel_dim"]),
                "BF16",
                "recurrent",
            )
            tensor(a + ".in_proj_ba.weight", (2 * nv, h), "BF16", "recurrent")
            tensor(
                a + ".in_proj_qkvz.weight",
                (2 * nk * kd + 2 * nv * vd, h),
                "BF16",
                "recurrent",
            )
            tensor(a + ".norm.weight", (vd,), "BF16", "recurrent")
            nvfp4(a + ".out_proj", h, nv * vd, "recurrent")
        else:
            raise ValueError(f"unsupported attention kind: {attention}")
    for name, item in expected.items():
        if not name.endswith(".weight"):
            continue
        module = name.removesuffix(".weight")
        excluded = any(fnmatch.fnmatchcase(module, pattern) for pattern in q["ignore"])
        linear = (
            item["kind"] in ("router", "shared_gate")
            or module == "lm_head"
            or module.endswith(
                ("q_proj", "k_proj", "v_proj", "in_proj_ba", "in_proj_qkvz")
            )
        )
        if (item["dtype"] == "U8" and excluded) or (linear and not excluded):
            raise ValueError(
                f"quantization exclusion conflicts with tensor layout: {name}"
            )
    return expected


def audit(directory, *, headers_only=False, check_values=False):
    """Return complete metadata coverage; optionally check local global scales."""
    directory = Path(directory)
    config = read_json(directory / "config.json")
    quant = read_json(directory / "hf_quant_config.json")
    expected = expected_qwen3_next(config, quant)
    tensors, shards, manifest = inventory(directory, headers_only=headers_only)
    target = {n for n in tensors if not n.startswith("mtp.")}
    missing, unexpected = set(expected) - target, target - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"target tensor coverage mismatch: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}"
        )
    totals = defaultdict(
        lambda: dict(tensors=0, bytes=0, packed_bytes=0, scale_bytes=0)
    )
    for name, tensor in tensors.items():
        if name in expected:
            item = expected[name]
            if any(tensor[k] != item[k] for k in ("shape", "dtype")):
                raise ValueError(f"target tensor layout mismatch: {name}")
            kind = item["kind"]
        else:
            kind = "optional_mtp"
        row = totals[kind]
        row["tensors"] += 1
        row["bytes"] += tensor["bytes"]
        if tensor["dtype"] == "U8":
            row["packed_bytes"] += tensor["bytes"]
        if "scale" in name:
            row["scale_bytes"] += tensor["bytes"]
    values = "not_checked"
    if check_values:
        if headers_only:
            raise ValueError(
                "global value validation requires local checkpoint payloads"
            )
        scales = {}
        by_shard = defaultdict(list)
        for name, t in tensors.items():
            if (
                name in expected
                and expected[name]["kind"] == "routed"
                and name.endswith("weight_scale_2")
            ):
                by_shard[t["shard"]].append((name, t["offset"]))
        for shard, records in by_shard.items():
            with (directory / shard).open("rb") as stream:
                for name, offset in records:
                    stream.seek(offset)
                    value = struct.unpack("<f", stream.read(4))[0]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"invalid routed global scale: {name}")
                    scales[name] = value
        for name, value in scales.items():
            if (
                name.endswith("gate_proj.weight_scale_2")
                and value != scales[name.replace("gate_proj", "up_proj")]
            ):
                raise ValueError(
                    f"unequal gate/up global scales; no requantization: {name}"
                )
        values = "all_routed_global_scales_positive_finite_and_gate_up_equal"
    e, h, i, layers = (
        config[k]
        for k in (
            "num_experts",
            "hidden_size",
            "moe_intermediate_size",
            "num_hidden_layers",
        )
    )

    # Six aligned fields per canonical layer; globals retain one scalar per expert.
    def align(n):
        return (n + 255) // 256 * 256

    canonical = layers * sum(
        align(n)
        for n in (
            e * 2 * i * h // 2,
            e * h * i // 2,
            e * 2 * i * h // 16,
            e * h * i // 16,
            e * 4,
            e * 4,
        )
    )
    required = sum(r["bytes"] for n, r in totals.items() if n != "optional_mtp")
    return dict(
        schema=1,
        status="metadata_compatible_execution_unqualified",
        architecture=config["architectures"][0],
        repository=manifest["repo"] if manifest else None,
        revision=manifest["revision"] if manifest else None,
        metadata_sha256={
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "hf_quant_config.json",
                "model.safetensors.index.json",
            )
        },
        shards=shards,
        classes=dict(totals),
        required_target_bytes=required,
        routed_weight_lower_bound_bytes=totals["routed"]["packed_bytes"],
        routed_source_bytes=totals["routed"]["bytes"],
        canonical_mapped_bytes=canonical,
        host_expert_lower_bound_bytes=totals["routed"]["bytes"] + canonical,
        non_routed_target_stored_bytes=required - totals["routed"]["bytes"],
        geometry=dict(
            layers=layers,
            experts=e,
            hidden=h,
            intermediate=i,
            top_k=config["num_experts_per_tok"],
        ),
        global_scale_values=values,
        remaining_gates=[
            "full checkpoint content fingerprint",
            "block scale value validation",
            "live host/device admission including conversion peaks, shared/dense preparation, KV/recurrent state, graphs and safety",
            "actual loader consumption and native layer oracle",
            "source-matched complete-model execution and lifecycle",
        ],
    )
