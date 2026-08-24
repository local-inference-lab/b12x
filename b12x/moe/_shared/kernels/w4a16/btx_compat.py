"""Lift frozen QSRT atom containers into in-memory BTX extents.

The `kquant_kimi_k3_qsrt_atoms_v1`/`_v2` containers derive rate placement
arithmetically and embed per-atom rotation spans inside expert bundles.
These lifts re-express one rank extent as a :class:`BtxLayer` — explicit
rate tables, a separate rotations tensor, pure-code-word bundles, and a
synthesized fail-closed manifest — so the declarative BTX preparation path
serves the frozen containers unchanged.

This module is the only remaining holder of the frozen containers' layout
knowledge (the ``(5*expert + layer) % 12`` pair rotation, the embedded
64-byte span offsets, and the rate-class row grouping). Remove it when the
checkpoints it serves have been re-exported as `btx-atoms-v1` and the
re-exports validate.

The unshipped ``qsrt_atoms_v3`` container stores uniform-K3 trellis words
separately from compact sign-and-magnitude reconstruction tables. Its lift
expands only the owner-local intermediate-coordinate extent into the same
``BtxLayer`` boundary tables consumed by the fused kernels.
"""

from __future__ import annotations

import torch

from b12x.moe._shared.btx_schema import (
    ATOMS_PER_PAIR,
    BTX_MANIFEST_KIND,
    BTX_SCHEMA,
    BtxManifest,
    matrix_atom_bytes,
    rate_code,
)
from b12x.moe._shared.kernels.w4a16.btx import BtxLayer
from b12x.moe._shared.trellis_codebooks import SQG_E4M3

# The frozen containers spread each expert's high-rate pairs across ranks
# with this multiplier; the lift materializes the result as tables.
_PAIR_ROTATION_MULTIPLIER = 5
_SPAN_BYTES = 64
_UNIFORM_RECORD_CHANNELS = 128
_ROTATION_SPAN_CHANNELS = 32


def _lift_manifest(
    *,
    num_experts: int,
    hidden_size: int,
    global_intermediate_size: int,
    layer_index: int,
    rates: dict,
    coupled: bool,
    per_expert_input_rotations: bool = False,
) -> BtxManifest:
    hadamard: dict = {
        "coupled": coupled,
        "per_expert_input_rotations": per_expert_input_rotations,
    }
    if coupled:
        hadamard["pre_block"] = 512
        hadamard["post_block"] = 128
    atom_slots = global_intermediate_size // 32
    barriers = [atom_slots // 2] if coupled else []
    return BtxManifest.from_dict(
        {
            "kind": BTX_MANIFEST_KIND,
            "schema": BTX_SCHEMA,
            "codebook": SQG_E4M3,
            "geometry": {
                "num_experts": int(num_experts),
                "hidden_size": int(hidden_size),
                "intermediate_size": int(global_intermediate_size),
                "atom_channels": 32,
                "atom_slots": atom_slots,
                "moe_layer_indices": [int(layer_index)],
            },
            "rates": rates,
            "hadamard": hadamard,
            "layout": {
                "atom_row_alignment": 1,
                "extent_alignment_slots": 4,
                "extent_barriers": barriers,
            },
            "layers": {
                str(int(layer_index)): {
                    "file": "lifted-in-memory",
                    "sha256": "0" * 64,
                }
            },
        }
    )


def _strip_spans(
    bundles: torch.Tensor, *, trellis_bytes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split embedded bundles into code words and per-atom rotation spans.

    ``bundles`` is ``[atoms, experts, trellis_bytes + 3*64]`` uint8. Returns
    the pure code words ``[atoms, experts, trellis_bytes]`` and the rotation
    values ``[atoms, experts, 3, 32]`` fp16 in physical channel order.
    """

    words = bundles[..., :trellis_bytes]
    spans = (
        bundles[..., trellis_bytes:]
        .contiguous()
        .view(torch.float16)
        .reshape(bundles.shape[0], bundles.shape[1], 3, 32)
    )
    return words, spans


def _side_table(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous()


def _uniform_k3_scale_atoms(signs: torch.Tensor) -> torch.Tensor:
    """Map candidate-order scale signs to coupled logical atom order."""

    if signs.dim() != 1 or signs.numel() % _UNIFORM_RECORD_CHANNELS:
        raise ValueError(
            "uniform-K3 intermediate signs must contain complete 128-channel records"
        )
    record_count = signs.numel() // _UNIFORM_RECORD_CHANNELS
    physical_to_logical: list[int] = []
    for low in range((record_count + 1) // 2):
        high = record_count - 1 - low
        physical_to_logical.append(low)
        if high != low:
            physical_to_logical.append(high)
    logical_to_physical = torch.argsort(
        torch.tensor(
            physical_to_logical,
            dtype=torch.long,
            device=signs.device,
        )
    )
    return (
        signs.reshape(record_count, _UNIFORM_RECORD_CHANNELS)
        .index_select(0, logical_to_physical)
        .reshape(-1, _ROTATION_SPAN_CHANNELS)
        .contiguous()
    )


def _per_expert_side_tables(
    *,
    num_experts: int,
    hidden_size: int,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
) -> bool:
    shapes = {
        tuple(gate_suh.shape),
        tuple(up_suh.shape),
        tuple(down_svh.shape),
    }
    shared = (hidden_size,)
    per_expert = (num_experts, hidden_size)
    if shapes == {shared}:
        return False
    if shapes == {per_expert}:
        return True
    raise ValueError(
        "gate_suh, up_suh, and down_svh must all be shared [hidden_size] "
        "or expert-private [num_experts, hidden_size] tables"
    )


def lift_qsrt_atoms_v1_extent(
    atom_payload: torch.Tensor,
    *,
    first_atom_slot: int,
    layer_index: int,
    expert_ids: torch.Tensor,
    format_codes: torch.Tensor,
    hidden_size: int,
    global_intermediate_size: int,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
) -> BtxLayer:
    """Lift one fixed-payload atoms-v1 pair extent.

    ``atom_payload`` is ``[8, num_experts, bundle]`` uint8, one 256-channel
    pair whose per-expert format byte packs the P24 pair counts
    (``r13`` high nibble for FC1, ``r2`` low nibble for FC2).
    """

    if atom_payload.dim() != 3 or atom_payload.shape[0] != ATOMS_PER_PAIR:
        raise ValueError(
            "atoms-v1 extents are [8, num_experts, bundle] uint8 rows"
        )
    num_experts = int(atom_payload.shape[1])
    matrix_bytes = matrix_atom_bytes(hidden_size, 3, 3)
    if int(atom_payload.shape[2]) != 3 * matrix_bytes + 3 * _SPAN_BYTES:
        raise ValueError(
            "atoms-v1 bundle bytes disagree with the declared hidden size"
        )
    expert_ids = expert_ids.to(dtype=torch.int64, device="cpu")
    format_codes = format_codes.to(dtype=torch.int64, device="cpu")
    if expert_ids.shape != (num_experts,) or format_codes.shape != (
        num_experts,
    ):
        raise ValueError(
            "atoms-v1 expert_ids and format_codes must cover the extent's "
            "experts"
        )
    r13 = format_codes >> 4
    r2 = format_codes & 0xF
    if bool(torch.any((r13 < 0) | (r13 > 2) | (r2 < 0) | (r2 > 2))):
        raise ValueError("atoms-v1 format codes must encode R0/R1/R2")
    physical_pair = first_atom_slot // ATOMS_PER_PAIR
    rotation = (_PAIR_ROTATION_MULTIPLIER * expert_ids + layer_index) % 12
    logical_pair = (physical_pair - rotation) % 12
    p33 = rate_code(3, 3)
    p24 = rate_code(2, 4)
    fc1_codes = torch.where(logical_pair < r13, p24, p33).to(torch.uint8)
    fc2_codes = torch.where(logical_pair < r2, p24, p33).to(torch.uint8)

    kinds = {"P33"}
    if bool(torch.any(fc1_codes == p24)) or bool(torch.any(fc2_codes == p24)):
        kinds.add("P24")
    manifest = _lift_manifest(
        num_experts=num_experts,
        hidden_size=hidden_size,
        global_intermediate_size=global_intermediate_size,
        layer_index=layer_index,
        rates={
            "structure": "per_expert_pair",
            "pair_kinds": sorted(kinds),
        },
        coupled=False,
    )
    manifest.validate_extent(first_atom_slot, ATOMS_PER_PAIR)

    words, spans = _strip_spans(atom_payload, trellis_bytes=3 * matrix_bytes)
    pairs = manifest.geometry.atom_slots // ATOMS_PER_PAIR
    rates_fc1 = torch.zeros((pairs, num_experts), dtype=torch.uint8)
    rates_fc2 = torch.zeros((pairs, num_experts), dtype=torch.uint8)
    rates_fc1[physical_pair] = fc1_codes
    rates_fc2[physical_pair] = fc2_codes
    return BtxLayer(
        manifest=manifest,
        layer_index=layer_index,
        first_slot=first_atom_slot,
        slot_count=ATOMS_PER_PAIR,
        atoms=words.reshape(ATOMS_PER_PAIR, -1),
        rotations=spans,
        gate_suh=_side_table(gate_suh),
        up_suh=_side_table(up_suh),
        down_svh=_side_table(down_svh),
        rates_fc1=rates_fc1[physical_pair : physical_pair + 1],
        rates_fc2=rates_fc2[physical_pair : physical_pair + 1],
        rotation_draws=None,
    )


def lift_qsrt_atoms_v2_extent(
    atom_payload: torch.Tensor,
    *,
    profile: str,
    first_atom_slot: int,
    layer_index: int,
    hidden_size: int,
    global_intermediate_size: int,
    num_experts: int,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    rotation_draws: torch.Tensor | None = None,
) -> BtxLayer:
    """Lift one atoms-v2 rank extent for a supported profile.

    ``atom_payload`` is ``[atom_count, row_bytes]`` uint8. Uniform K2/K3
    coupled profiles lift to a coupled extent of any legal length;
    the fixed high-rate profile (``k3x22_k4x2``) lifts one pair with its
    rate-class row grouping restored to expert-major order. The coupled
    high-rate profile has no lift: its pair-kind mixes have no qualified
    BTX execution path, so serving it requires a re-export decision.
    """

    if profile in {"k2_coupled_h512_h128", "k3_coupled_h512_h128"}:
        if rotation_draws is None:
            raise ValueError("coupled uniform profiles carry draws")
        atom_count = int(atom_payload.shape[0])
        bits = 2 if profile == "k2_coupled_h512_h128" else 3
        matrix_bytes = matrix_atom_bytes(hidden_size, bits, bits)
        bundle = 3 * matrix_bytes + 3 * _SPAN_BYTES
        payload = num_experts * bundle
        if int(atom_payload.shape[1]) < payload:
            raise ValueError("uniform rows are shorter than their bundles")
        if bool(torch.any(atom_payload[:, payload:] != 0)):
            raise ValueError("uniform row padding must be zero")
        manifest = _lift_manifest(
            num_experts=num_experts,
            hidden_size=hidden_size,
            global_intermediate_size=global_intermediate_size,
            layer_index=layer_index,
            rates={"structure": "uniform", "bits": bits},
            coupled=True,
            per_expert_input_rotations=_per_expert_side_tables(
                num_experts=num_experts,
                hidden_size=hidden_size,
                gate_suh=gate_suh,
                up_suh=up_suh,
                down_svh=down_svh,
            ),
        )
        manifest.validate_extent(first_atom_slot, atom_count)
        bundles = atom_payload[:, :payload].reshape(
            atom_count, num_experts, bundle
        )
        words, spans = _strip_spans(bundles, trellis_bytes=3 * matrix_bytes)
        return BtxLayer(
            manifest=manifest,
            layer_index=layer_index,
            first_slot=first_atom_slot,
            slot_count=atom_count,
            atoms=words.reshape(atom_count, -1),
            rotations=spans,
            gate_suh=_side_table(gate_suh),
            up_suh=_side_table(up_suh),
            down_svh=_side_table(down_svh),
            rates_fc1=None,
            rates_fc2=None,
            rotation_draws=rotation_draws.to(
                dtype=torch.uint8, device="cpu"
            ).contiguous(),
        )

    if profile == "k3x22_k4x2":
        if rotation_draws is not None:
            raise ValueError("the fixed high-rate profile carries no draws")
        if int(atom_payload.shape[0]) != ATOMS_PER_PAIR:
            raise ValueError("fixed high-rate extents cover one pair")
        physical_pair = first_atom_slot // ATOMS_PER_PAIR
        expert_ids = torch.arange(num_experts, dtype=torch.int64)
        rotation = (
            _PAIR_ROTATION_MULTIPLIER * expert_ids + layer_index
        ) % 12
        base_pair = (physical_pair - rotation) % 12
        modes = (base_pair == 0) | (base_pair == 6)
        p33_ids = torch.nonzero(~modes, as_tuple=False).flatten()
        p43_ids = torch.nonzero(modes, as_tuple=False).flatten()
        p33_bytes = 3 * matrix_atom_bytes(hidden_size, 3, 3) + 3 * _SPAN_BYTES
        p43_bytes = 3 * matrix_atom_bytes(hidden_size, 4, 3) + 3 * _SPAN_BYTES
        payload = (
            int(p33_ids.numel()) * p33_bytes + int(p43_ids.numel()) * p43_bytes
        )
        if int(atom_payload.shape[1]) < payload:
            raise ValueError(
                "fixed high-rate rows are shorter than their compact groups"
            )
        if bool(torch.any(atom_payload[:, payload:] != 0)):
            raise ValueError("fixed high-rate row padding must be zero")
        fc_codes = torch.where(
            modes, rate_code(4, 3), rate_code(3, 3)
        ).to(torch.uint8)
        manifest = _lift_manifest(
            num_experts=num_experts,
            hidden_size=hidden_size,
            global_intermediate_size=global_intermediate_size,
            layer_index=layer_index,
            rates={
                "structure": "per_expert_pair",
                "pair_kinds": sorted({"P33", "P43"}),
            },
            coupled=False,
        )
        manifest.validate_extent(first_atom_slot, ATOMS_PER_PAIR)

        # Restore the rate-class row grouping to expert-major order.
        span_rows = torch.empty(
            (ATOMS_PER_PAIR, num_experts, 3, 32), dtype=torch.float16
        )
        expert_words: list[torch.Tensor | None] = [None] * num_experts
        begin = 0
        for ids, bundle in ((p33_ids, p33_bytes), (p43_ids, p43_bytes)):
            count = int(ids.numel())
            group = atom_payload[
                :, begin : begin + count * bundle
            ].reshape(ATOMS_PER_PAIR, count, bundle)
            words, spans = _strip_spans(
                group, trellis_bytes=bundle - 3 * _SPAN_BYTES
            )
            for position, expert in enumerate(ids.tolist()):
                expert_words[expert] = words[:, position]
                span_rows[:, expert] = spans[:, position]
            begin += count * bundle
        assert all(words is not None for words in expert_words)
        atoms = torch.cat(
            [words for words in expert_words if words is not None], dim=1
        ).reshape(ATOMS_PER_PAIR, -1)
        return BtxLayer(
            manifest=manifest,
            layer_index=layer_index,
            first_slot=first_atom_slot,
            slot_count=ATOMS_PER_PAIR,
            atoms=atoms,
            rotations=span_rows,
            gate_suh=_side_table(gate_suh),
            up_suh=_side_table(up_suh),
            down_svh=_side_table(down_svh),
            rates_fc1=fc_codes.reshape(1, -1),
            rates_fc2=fc_codes.reshape(1, -1),
            rotation_draws=None,
        )

    raise ValueError(
        f"QSRT atoms-v2 profile {profile!r} has no BTX lift; the coupled "
        "high-rate profile's pair-kind mixes have no qualified fused "
        "execution path"
    )


def lift_qsrt_atoms_v3_extent(
    atom_payload: torch.Tensor,
    *,
    first_atom_slot: int,
    layer_index: int,
    hidden_size: int,
    global_intermediate_size: int,
    num_experts: int,
    gate_suh_signs: torch.Tensor,
    up_suh_signs: torch.Tensor,
    down_svh_signs: torch.Tensor,
    gate_svh_signs: torch.Tensor,
    up_svh_signs: torch.Tensor,
    down_suh_signs: torch.Tensor,
    layer_scale_magnitudes: torch.Tensor,
    expert_scale_magnitudes: torch.Tensor,
    rotation_draws: torch.Tensor,
) -> BtxLayer:
    """Lift one compact sign-and-scalar uniform-K3 atoms-v3 extent.

    ``atom_payload`` contains trellis words only. The six sign tables and
    magnitudes reconstruct the three side tables and three owner-local
    intermediate tables without allocating a model-global expert table.
    """

    if atom_payload.dim() != 2 or atom_payload.dtype != torch.uint8:
        raise ValueError("atoms-v3 payload must be uint8[atom_count, row_bytes]")
    atom_count = int(atom_payload.shape[0])
    matrix_bytes = matrix_atom_bytes(hidden_size, 3, 3)
    bundle = 3 * matrix_bytes
    payload = num_experts * bundle
    if int(atom_payload.shape[1]) < payload:
        raise ValueError("atoms-v3 rows are shorter than their trellis bundles")
    if bool(torch.any(atom_payload[:, payload:] != 0)):
        raise ValueError("atoms-v3 row padding must be zero")

    hidden_signs = {
        "gate_suh": gate_suh_signs,
        "up_suh": up_suh_signs,
        "down_svh": down_svh_signs,
    }
    for name, signs in hidden_signs.items():
        if signs.dtype != torch.float16 or tuple(signs.shape) != (hidden_size,):
            raise ValueError(f"atoms-v3 {name} signs must be fp16[hidden_size]")
        if not bool(torch.all(torch.abs(signs) == 1)):
            raise ValueError(f"atoms-v3 {name} signs must be exactly +/-1")

    intermediate_signs = {
        "gate_svh": gate_svh_signs,
        "up_svh": up_svh_signs,
        "down_suh": down_suh_signs,
    }
    for name, signs in intermediate_signs.items():
        if signs.dtype != torch.float16 or tuple(signs.shape) != (
            global_intermediate_size,
        ):
            raise ValueError(
                f"atoms-v3 {name} signs must be fp16[global_intermediate_size]"
            )
        if not bool(torch.all(torch.abs(signs) == 1)):
            raise ValueError(f"atoms-v3 {name} signs must be exactly +/-1")

    if (
        layer_scale_magnitudes.dtype != torch.float16
        or tuple(layer_scale_magnitudes.shape) != (3,)
        or not bool(torch.all(torch.isfinite(layer_scale_magnitudes)))
        or bool(torch.any(layer_scale_magnitudes <= 0))
    ):
        raise ValueError("atoms-v3 layer magnitudes must be positive finite fp16[3]")
    if (
        expert_scale_magnitudes.dtype != torch.float16
        or tuple(expert_scale_magnitudes.shape) != (3, num_experts)
        or not bool(torch.all(torch.isfinite(expert_scale_magnitudes)))
        or bool(torch.any(expert_scale_magnitudes <= 0))
    ):
        raise ValueError(
            "atoms-v3 expert magnitudes must be positive finite "
            "fp16[3, num_experts]"
        )
    if (
        rotation_draws.dtype != torch.uint8
        or tuple(rotation_draws.shape) != (num_experts,)
        or bool(torch.any(rotation_draws != 0))
    ):
        raise ValueError("production atoms-v3 requires uint8 zero draws")

    manifest = _lift_manifest(
        num_experts=num_experts,
        hidden_size=hidden_size,
        global_intermediate_size=global_intermediate_size,
        layer_index=layer_index,
        rates={"structure": "uniform", "bits": 3},
        coupled=True,
    )
    manifest.validate_extent(first_atom_slot, atom_count)

    last_atom_slot = first_atom_slot + atom_count
    if last_atom_slot * manifest.geometry.atom_channels > global_intermediate_size:
        raise ValueError("atoms-v3 intermediate extent exceeds model geometry")

    side_tables = [
        hidden_signs["gate_suh"] * layer_scale_magnitudes[0],
        hidden_signs["up_suh"] * layer_scale_magnitudes[1],
        hidden_signs["down_svh"] * layer_scale_magnitudes[2],
    ]
    gate_atoms = _uniform_k3_scale_atoms(intermediate_signs["gate_svh"])
    up_atoms = _uniform_k3_scale_atoms(intermediate_signs["up_svh"])
    down_atoms = _uniform_k3_scale_atoms(intermediate_signs["down_suh"])
    upstream_atoms = torch.cat((gate_atoms, up_atoms), dim=0)
    atom_indices = torch.arange(
        first_atom_slot,
        last_atom_slot,
        dtype=torch.long,
        device=atom_payload.device,
    )
    rotation_signs = torch.stack(
        (
            upstream_atoms.index_select(0, 2 * atom_indices),
            upstream_atoms.index_select(0, 2 * atom_indices + 1),
            down_atoms.index_select(0, atom_indices),
        ),
        dim=1,
    )
    upstream_magnitude_indices = (
        atom_indices >= manifest.geometry.atom_slots // 2
    ).to(torch.long)
    upstream_magnitudes = expert_scale_magnitudes.index_select(
        0, upstream_magnitude_indices
    )
    down_magnitudes = expert_scale_magnitudes[2].expand(atom_count, -1)
    rotation_magnitudes = torch.stack(
        (upstream_magnitudes, upstream_magnitudes, down_magnitudes),
        dim=2,
    )
    rotations = (
        rotation_magnitudes[..., None] * rotation_signs[:, None, :, :]
    ).contiguous()
    bundles = atom_payload[:, :payload].reshape(atom_count, num_experts, bundle)
    return BtxLayer(
        manifest=manifest,
        layer_index=layer_index,
        first_slot=first_atom_slot,
        slot_count=atom_count,
        atoms=bundles.reshape(atom_count, -1),
        rotations=rotations,
        gate_suh=_side_table(side_tables[0]),
        up_suh=_side_table(side_tables[1]),
        down_svh=_side_table(side_tables[2]),
        rates_fc1=None,
        rates_fc2=None,
        rotation_draws=rotation_draws.contiguous(),
    )
