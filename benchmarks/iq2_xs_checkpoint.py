"""Read routed IQ2_XS expert tensors from a local safetensors snapshot."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

from safetensors import safe_open
import torch

from b12x.moe.fused_moe import IQ2XSWeights


@dataclass(frozen=True)
class IQ2XSLayer:
    weights: IQ2XSWeights
    hidden_size: int
    intermediate_size: int
    route_num_experts: int
    top_k: int
    expert_ids: tuple[int, ...]
    snapshot: Path
    layer: int
    tp_size: int
    tp_rank: int

    def expert_map(self, device: torch.device | str) -> torch.Tensor:
        result = torch.full((self.route_num_experts,), -1, dtype=torch.int32)
        result[list(self.expert_ids)] = torch.arange(
            len(self.expert_ids), dtype=torch.int32
        )
        return result.to(device)


def load_iq2_xs_layer(
    snapshot: str | Path,
    *,
    layer: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    expert_ids: tuple[int, ...] | None = None,
) -> IQ2XSLayer:
    """Load CPU W31 blocks with block-aligned TP cuts and explicit expert order.

    Only routed expert weights are read. Attention, shared experts, vision and
    MTP tensors retain their independent checkpoint formats.
    """
    snapshot = Path(snapshot).expanduser().resolve()
    config = json.loads((snapshot / "config.json").read_text())
    text = config.get("text_config", config)
    h, i, e = (
        int(text[key])
        for key in ("hidden_size", "moe_intermediate_size", "num_experts")
    )
    if not 0 <= layer < int(text["num_hidden_layers"]):
        raise ValueError("layer is outside the checkpoint")
    if (
        tp_size <= 0
        or not 0 <= tp_rank < tp_size
        or i % tp_size
        or (i // tp_size) % 256
        or h % 256
    ):
        raise ValueError("IQ2_XS tensor-parallel cuts must align H and local I to 256")
    selected = tuple(range(e)) if expert_ids is None else tuple(expert_ids)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(not 0 <= value < e for value in selected)
    ):
        raise ValueError("expert_ids must be distinct checkpoint expert indices")
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    quant = json.loads((snapshot / "hf_quant_config.json").read_text())["quantization"]
    recipe = quant["quantized_layers"][prefix]
    expected = {
        "quant_algo": "IQ2_XS",
        "group_size": 256,
        "block_payload_bytes": 74,
        "packing": "ggml",
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError(f"unsupported IQ2_XS block contract: {recipe}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    local_i = i // tp_size
    lo, hi = tp_rank * local_i, (tp_rank + 1) * local_i
    w13 = torch.empty((len(selected), 2 * local_i, h // 256, 74), dtype=torch.uint8)
    w2 = torch.empty((len(selected), h, local_i // 256, 74), dtype=torch.uint8)
    with ExitStack() as stack:
        handles = {}
        for local, expert in enumerate(selected):
            for projection in ("gate", "up", "down"):
                name = f"{prefix}.{expert}.{projection}_proj.weight"
                shard = index[name]
                if not shard.endswith(".safetensors"):
                    raise ValueError("IQ2_XS checkpoint shards must be safetensors")
                if shard not in handles:
                    handles[shard] = stack.enter_context(
                        safe_open(snapshot / shard, framework="pt", device="cpu")
                    )
                source = handles[shard].get_slice(name)
                shape = (h, i // 256, 74) if projection == "down" else (i, h // 256, 74)
                if tuple(source.get_shape()) != shape or source.get_dtype() != "U8":
                    raise ValueError(
                        f"invalid IQ2_XS tensor {name}: expected uint8{shape}"
                    )
                if projection == "down":
                    w2[local].copy_(source[:, lo // 256 : hi // 256, :])
                else:
                    offset = 0 if projection == "gate" else local_i
                    w13[local, offset : offset + local_i].copy_(source[lo:hi, :, :])
    return IQ2XSLayer(
        IQ2XSWeights(w13, w2),
        h,
        local_i,
        e,
        int(text["num_experts_per_tok"]),
        selected,
        snapshot,
        layer,
        tp_size,
        tp_rank,
    )
