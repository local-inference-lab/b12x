"""Independent diagnostic oracles for native Trellis expert execution.

These Torch/NumPy routines belong to offline profile qualification. Production
plans never call them. Weight decoding expands one selected expert at a time.
"""

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=8)
def _hadamard_matrix(device):
    return (
        torch.tensor(
            [
                [-1.0 if (row & col).bit_count() % 2 else 1.0 for col in range(128)]
                for row in range(128)
            ],
            dtype=torch.float32,
            device=device,
        )
        / 128**0.5
    )


def native_weight(payload: torch.Tensor, bits: int, codebook: str) -> torch.Tensor:
    """Return [E,N,K] from native [E,K/16,N/16,16*bits] Int16 records."""
    native = payload.detach().cpu().contiguous().numpy().view(np.uint16)
    experts, k_tiles, n_tiles, _ = native.shape
    halves = native.reshape(experts, k_tiles, n_tiles, 8 * bits, 2).astype(np.uint32)
    words = halves[..., 0] | (halves[..., 1] << np.uint32(16))
    decoded = np.empty((experts, n_tiles, 16, k_tiles, 16), dtype=np.float16)
    table = None
    if codebook == "sqg_e4m3":
        from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_direct_lut_cpu

        table = (
            sqg_xor_cheb_t12_direct_lut_cpu().view(torch.float8_e4m3fn).half().numpy()
        )
        table = table[(bits - 2) * 65536 : (bits - 1) * 65536]
    elif codebook == "sqg_fp16":
        from b12x._lib.quant.sqg_fp16_d3l import sqg_fp16_d3l_direct_lut_cpu

        table = sqg_fp16_d3l_direct_lut_cpu(bits).numpy()
    elif codebook != "mcg":
        raise ValueError(f"unknown test codebook {codebook}")
    for lane in range(32):
        for j in range(8):
            end = (8 * lane + j + 257) * bits
            start = end - 16
            first = words[..., (start // 32) % (8 * bits)].astype(np.uint64)
            last_index = (end - 1) // 32
            last = words[..., last_index % (8 * bits)].astype(np.uint64)
            window = (((first << 32) | last) >> ((last_index + 1) * 32 - end)) & 65535
            if table is None:
                code = ((window * 0xCBAC1FED) & 0xFFFFFFFF).astype(np.uint32)
                code = (code & 0x8FFF8FFF) ^ 0x3B603B60
                low = (code & 65535).astype(np.uint16).view(np.float16)
                high = (code >> 16).astype(np.uint16).view(np.float16)
                values = (low + high).astype(np.float16)
            else:
                values = table[window]
            nn = 2 * (lane // 8) + ((lane >> 2) & 1) + (8 if j >= 4 else 0)
            kk = 2 * (lane % 4) + j % 2 + (8 if j % 4 >= 2 else 0)
            decoded[:, :, nn, :, kk] = values.transpose(0, 2, 1)
    return torch.from_numpy(decoded.reshape(experts, 16 * n_tiles, 16 * k_tiles))


def had128(x):
    return (x.float().reshape(-1, 128) @ _hadamard_matrix(x.device)).reshape_as(x)


def had512(x):
    matrix = (
        torch.tensor(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
            dtype=torch.float32,
            device=x.device,
        )
        * 0.5
    )
    transformed = had128(x).reshape(-1, 4, 128)
    return torch.einsum("rgc,gh->rhc", transformed, matrix).reshape_as(x)


def input_rotation(source, ids, scales, top_k, coupled):
    output = torch.zeros(
        (ids.numel(), source.shape[1]), device=source.device, dtype=torch.float16
    )
    for row, expert in enumerate(ids.cpu().tolist()):
        if expert >= 0:
            x = source[row // top_k].half().float()
            if coupled:
                x = had512(x)
            scale = scales[0 if scales.shape[0] == 1 else expert]
            output[row] = had128((x * scale.float()).half())
    return output


def activation(gate, up, kind):
    if kind == "situ":
        return 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
    return gate * torch.sigmoid(gate) * up


def intermediate_rotation(gate, up, ids, rotations, coupled, kind):
    width = gate.shape[1]
    result = torch.zeros_like(gate)
    for row, expert in enumerate(ids.cpu().tolist()):
        if expert < 0:
            continue
        gscale, uscale, down = rotations[expert, : 3 * width].reshape(3, width).float()
        if coupled:
            raw = torch.stack(
                (gate[row].reshape(-1, 32), up[row].reshape(-1, 32)), 1
            ).flatten()
            scale = torch.stack(
                (gscale.reshape(-1, 32), uscale.reshape(-1, 32)), 1
            ).flatten()
            raw = (
                had128(had128(raw) * scale)
                * rotations[expert, 3 * width : 5 * width].float()
            )
            activated = activation(raw[::2], raw[1::2], "situ")
            activated *= rotations[expert, 5 * width :].float()
            result[row] = had128(had128(activated) * down)
        else:
            g = (had128(gate[row]) * gscale).half().float()
            u = (had128(up[row]) * uscale).half().float()
            activated = activation(g, u, kind).half().float()
            result[row] = had128((activated * down).half())
    return result


def output_rotation(source, ids, scales, weights, coupled):
    result = torch.zeros(
        (weights.shape[0], source.shape[1]), dtype=torch.float32, device=source.device
    )
    for row, expert in enumerate(ids.cpu().tolist()):
        if expert >= 0:
            scale = scales[0 if scales.shape[0] == 1 else expert].float()
            result[row // weights.shape[1]] += (
                had128(source[row]) * scale * weights.flatten()[row]
            )
    return had512(result) if coupled else result


def _moe_reference(
    source,
    prepared,
    topk_ids,
    topk_weights,
    *,
    activation_kind,
    route_expert_map=None,
    output_expert_map=None,
):
    """Evaluate transforms and FP16 projection boundaries from compressed weights."""
    experts, hidden = prepared.num_experts, source.shape[1]
    mixed = getattr(prepared, "weight_layout", None) == "trellis_mixed3"
    internal_map, descriptor_rows = None, None
    if mixed:
        from types import SimpleNamespace

        rotations = prepared.rotations
        state = SimpleNamespace(
            coupled_hadamard=prepared.coupled_hadamard,
            gate_suh=rotations.gate_suh[:experts],
            up_suh=rotations.up_suh[:experts],
            down_svh=rotations.down_svh[:experts],
            intermediate_rotations=rotations.intermediate[:experts],
        )
        internal_map = prepared.global_to_combined.cpu().tolist()
        descriptor_rows = prepared.descriptor_map.cpu().view(3, -1).tolist()
    else:
        state = prepared.trellis
    width = state.intermediate_rotations.shape[1] // (
        6 if state.coupled_hadamard else 3
    )
    route_map = None if route_expert_map is None else route_expert_map.cpu().tolist()
    output_map = None if output_expert_map is None else output_expert_map.cpu().tolist()
    ids, output_ids = [], []
    for raw in topk_ids.flatten().cpu().tolist():
        expert = (
            raw
            if route_map is None
            else (route_map[raw] if 0 <= raw < len(route_map) else -1)
        )
        expert = expert if 0 <= expert < experts else -1
        output_expert = (
            expert
            if output_map is None
            else (output_map[raw] if 0 <= raw < len(output_map) else -1)
        )
        if internal_map is not None:
            expert = internal_map[expert] if 0 <= expert < experts else -1
            output_expert = (
                internal_map[output_expert] if 0 <= output_expert < experts else -1
            )
            expert = expert if 0 <= expert < experts else -1
        ids.append(expert)
        output_ids.append(
            output_expert if expert >= 0 and 0 <= output_expert < experts else -1
        )
    device_ids = torch.tensor(ids, device=source.device, dtype=torch.int64)
    final_ids = torch.tensor(output_ids, device=source.device, dtype=torch.int64)
    gate_input = input_rotation(
        source, device_ids, state.gate_suh, topk_ids.shape[1], state.coupled_hadamard
    )
    up_input = (
        gate_input
        if state.coupled_hadamard
        else input_rotation(source, device_ids, state.up_suh, topk_ids.shape[1], False)
    )
    gate = torch.zeros((len(ids), width), device=source.device, dtype=torch.float16)
    up = torch.zeros_like(gate)

    def projection_weight(projection, expert):
        n, k = (width, hidden) if projection < 2 else (hidden, width)
        if mixed:
            descriptor = descriptor_rows[projection][expert]
            tier = descriptor >> prepared.descriptor_local_bits
            local = descriptor & ((1 << prepared.descriptor_local_bits) - 1)
            if not 0 <= tier < 3:
                return None
            counts = (prepared.gate_counts, prepared.up_counts, prepared.down_counts)[
                projection
            ]
            if not 0 <= local < counts[tier]:
                return None
            bits, codebook = tier + 3, "mcg"
            payload = prepared.tiers[tier]
            if projection < 2:
                if projection == 1:
                    local += prepared.gate_counts[tier]
                native = payload.w13.view(torch.int16).view(
                    -1, k // 16, n // 16, 16 * bits
                )
            else:
                native = payload.w2.view(torch.int16).view(
                    -1, k // 16, n // 16, 16 * bits
                )
        else:
            bits, codebook, local = state.bits, state.codebook, expert
            if projection < 2:
                native = prepared.w13.view(torch.int16).view(
                    2, experts, k // 16, n // 16, 16 * bits
                )[projection]
            else:
                native = prepared.w2.view(torch.int16).view(
                    experts, k // 16, n // 16, 16 * bits
                )
        return native_weight(native[local : local + 1], bits, codebook)[0].to(
            source.device
        )

    selected = sorted(set(ids) - {-1})
    for expert in selected:
        rows = torch.tensor(
            [i for i, value in enumerate(ids) if value == expert], device=source.device
        )
        for projection, inputs, outputs in ((0, gate_input, gate), (1, up_input, up)):
            weight = projection_weight(projection, expert)
            if weight is not None:
                outputs[rows] = (inputs[rows].float() @ weight.float().T).half()
    middle = intermediate_rotation(
        gate,
        up,
        device_ids,
        state.intermediate_rotations,
        state.coupled_hadamard,
        activation_kind,
    )
    down = torch.zeros((len(ids), hidden), device=source.device, dtype=torch.float16)
    for expert in selected:
        rows = torch.tensor(
            [i for i, value in enumerate(ids) if value == expert], device=source.device
        )
        weight = projection_weight(2, expert)
        if weight is not None:
            down[rows] = (middle[rows].float() @ weight.float().T).half()
    return output_rotation(
        down, final_ids, state.down_svh, topk_weights, state.coupled_hadamard
    )


def moe_reference(*args, **kwargs):
    """Evaluate the compressed expert contract with TF32 disabled."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _moe_reference(*args, **kwargs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
