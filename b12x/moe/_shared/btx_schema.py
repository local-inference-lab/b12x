"""BTX (b12x trellis exchange) checkpoint container schema.

BTX is the TP-shard-independent checkpoint container for trellis-coded MoE
expert weights. Storage is organized around 32-channel *atom slots* on the
intermediate axis: every slot row holds all experts' code words for those
channels, so a tensor-parallel rank loads a contiguous slot range and
nothing else. All behavior is declared in a manifest — codebook, rate
structure, coupled-Hadamard transform, and geometry — and the reader
derives byte addressing purely from those declarations.

Storage schema id: ``btx-atoms-v1``. A checkpoint directory contains
``btx-manifest.json`` plus one ``btx-layer-<NNNNN>.safetensors`` file per
MoE layer. Per-layer tensors:

- ``atoms``: u8 ``[atom_slots, row_stride]`` — trellis code words only,
  expert-id-major bundles per row, zero padding to the row stride.
- ``rotations``: fp16 ``[atom_slots, num_experts, 3, atom_channels]`` —
  per-channel intermediate-boundary values for gate/up/down in physical
  channel order.
- ``rates_fc1``/``rates_fc2``: u8 ``[atom_slots/8, num_experts]`` — one
  rate byte per (256-channel pair, expert); present iff the rate structure
  is ``per_expert_pair``.
- ``gate_suh``/``up_suh``/``down_svh``: fp16 ``[hidden_size]`` or
  ``[num_experts, hidden_size]`` — hidden-axis incoherence values.
- ``rotation_draws``: u8 ``[num_experts]`` in ``0..7`` — present iff the
  coupled-Hadamard transform is declared.

A rate byte is ``(low_bits << 4) | high_bits`` and is exactly the fused
kernel's pair-kind vocabulary expressed as data. Uniform checkpoints carry
no rate tables; their single bitrate is declared in the manifest.

Within one ``atoms`` row, expert bundles are concatenated in expert-id
order; each bundle is gate ‖ up ‖ down and each matrix section stores its
low-record plane followed by its high-record plane (``[H/16][16*low]i16``
‖ ``[H/16][16*high]i16``). Under a uniform rate structure the two planes
are the atom's two consecutive N16 (FC1) or K16 (FC2) tiles.

This module is torch-free: manifest parsing, fail-closed validation, extent
legality, and byte arithmetic. Tensor I/O and preparation live with the
W4A16 kernel host code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trellis_codebooks import (
    CODEBOOKS,
    MCG,
    MCG_MULTIPLIER,
    validate_codebook_bits,
)

BTX_SCHEMA = "btx-atoms-v1"
BTX_MANIFEST_KIND = "btx-manifest"
BTX_MANIFEST_FILENAME = "btx-manifest.json"

ATOM_CHANNELS = 32
ATOMS_PER_PAIR = 8

RATE_STRUCTURE_UNIFORM = "uniform"
RATE_STRUCTURE_PER_EXPERT_PAIR = "per_expert_pair"

# The fused kernel's pair-kind vocabulary as rate bytes.
RATE_CODE_PAIR_KINDS: dict[int, str] = {
    0x22: "P22",
    0x33: "P33",
    0x24: "P24",
    0x43: "P43",
    0x44: "P44",
}
PAIR_KIND_RATE_CODES: dict[str, int] = {
    kind: code for code, kind in RATE_CODE_PAIR_KINDS.items()
}


def layer_filename(layer_index: int) -> str:
    return f"btx-layer-{int(layer_index):05d}.safetensors"


def rate_code(low_bits: int, high_bits: int) -> int:
    return (int(low_bits) << 4) | int(high_bits)


def rate_code_bits(code: int) -> tuple[int, int]:
    return (int(code) >> 4) & 0xF, int(code) & 0xF


def matrix_atom_bytes(hidden_size: int, low_bits: int, high_bits: int) -> int:
    """Trellis bytes one atom contributes to one expert matrix.

    An atom holds two 16-channel record planes; each plane stores
    ``hidden_size/16`` tiles of ``16*bits`` int16 words.
    """

    return (int(hidden_size) // 16) * 32 * (int(low_bits) + int(high_bits))


def bundle_bytes(
    hidden_size: int, fc1_code: int, fc2_code: int
) -> int:
    """Per-(expert, atom) bundle size: gate ‖ up ‖ down trellis words."""

    fc1_low, fc1_high = rate_code_bits(fc1_code)
    fc2_low, fc2_high = rate_code_bits(fc2_code)
    return 2 * matrix_atom_bytes(hidden_size, fc1_low, fc1_high) + (
        matrix_atom_bytes(hidden_size, fc2_low, fc2_high)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_keys(
    mapping: dict, *, required: set[str], optional: set[str], where: str
) -> None:
    _require(isinstance(mapping, dict), f"{where} must be a JSON object")
    keys = set(mapping.keys())
    unknown = keys - required - optional
    _require(not unknown, f"{where} has unknown keys {sorted(unknown)}")
    missing = required - keys
    _require(not missing, f"{where} is missing keys {sorted(missing)}")


@dataclass(frozen=True)
class BtxGeometry:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    atom_channels: int
    atom_slots: int
    moe_layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class BtxRates:
    structure: str
    bits: int | None
    pair_kinds: frozenset[str] | None

    def uniform_code(self) -> int | None:
        if self.structure != RATE_STRUCTURE_UNIFORM:
            return None
        assert self.bits is not None
        return rate_code(self.bits, self.bits)


@dataclass(frozen=True)
class BtxHadamard:
    coupled: bool
    pre_block: int | None
    post_block: int | None
    per_expert_input_rotations: bool


@dataclass(frozen=True)
class BtxLayout:
    atom_row_alignment: int
    extent_alignment_slots: int
    extent_barriers: tuple[int, ...]


@dataclass(frozen=True)
class BtxLayerRef:
    file: str
    sha256: str


@dataclass(frozen=True)
class BtxManifest:
    codebook: str
    codebook_seed: int | None
    geometry: BtxGeometry
    rates: BtxRates
    hadamard: BtxHadamard
    layout: BtxLayout
    layers: dict[int, BtxLayerRef]

    @staticmethod
    def from_dict(data: dict) -> "BtxManifest":
        _require_keys(
            data,
            required={
                "kind",
                "schema",
                "codebook",
                "geometry",
                "rates",
                "hadamard",
                "layout",
                "layers",
            },
            optional={"codebook_seed"},
            where="BTX manifest",
        )
        _require(
            data["kind"] == BTX_MANIFEST_KIND,
            f"BTX manifest kind must be {BTX_MANIFEST_KIND!r}, "
            f"got {data['kind']!r}",
        )
        _require(
            data["schema"] == BTX_SCHEMA,
            f"BTX manifest schema must be {BTX_SCHEMA!r}, got {data['schema']!r}",
        )

        codebook = data["codebook"]
        _require(
            codebook in CODEBOOKS,
            f"BTX codebook must be one of {sorted(CODEBOOKS)}, got {codebook!r}",
        )
        seed = data.get("codebook_seed")
        if codebook == MCG:
            _require(
                isinstance(seed, int) and seed == MCG_MULTIPLIER,
                "BTX mcg checkpoints must declare codebook_seed "
                f"{MCG_MULTIPLIER:#010x}",
            )
        else:
            _require(
                seed is None,
                f"BTX codebook_seed is valid only for mcg, not {codebook!r}",
            )

        geometry = _parse_geometry(data["geometry"])
        rates = _parse_rates(data["rates"], codebook=codebook)
        hadamard = _parse_hadamard(data["hadamard"], geometry=geometry)
        layout = _parse_layout(data["layout"], geometry=geometry)
        layers = _parse_layers(data["layers"], geometry=geometry)
        return BtxManifest(
            codebook=codebook,
            codebook_seed=seed,
            geometry=geometry,
            rates=rates,
            hadamard=hadamard,
            layout=layout,
            layers=layers,
        )

    def partition_extents(self, world_size: int) -> tuple[tuple[int, int], ...]:
        """Partition all slots into nonempty, aligned rank extents.

        Barriers separate rank groups. Within each group, complete alignment
        units are balanced across ranks; local intermediate widths may differ.
        Kernel eligibility remains the execution planner's responsibility.
        """
        _require(
            type(world_size) is int and world_size > 0,
            "BTX world_size must be a positive integer",
        )
        alignment = self.layout.extent_alignment_slots
        boundaries = (0, *self.layout.extent_barriers, self.geometry.atom_slots)
        units = tuple(
            (end - start) // alignment
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        )
        _require(
            all(boundary % alignment == 0 for boundary in boundaries),
            "BTX barriers must align to complete rank extent units",
        )
        _require(
            len(units) <= world_size <= sum(units),
            "BTX world_size cannot cover all barriers with nonempty aligned extents",
        )
        ranks = [1] * len(units)
        for _ in range(world_size - len(units)):
            segment = max(
                (index for index, count in enumerate(units) if ranks[index] < count),
                key=lambda index: (
                    (units[index] + ranks[index] - 1) // ranks[index],
                    units[index],
                    -index,
                ),
            )
            ranks[segment] += 1
        extents = []
        for start, count, rank_count in zip(boundaries[:-1], units, ranks, strict=True):
            width, extra = divmod(count, rank_count)
            for rank in range(rank_count):
                size = (width + (rank < extra)) * alignment
                self.validate_extent(start, size)
                extents.append((start, size))
                start += size
        return tuple(extents)

    def validate_extent(self, first_slot: int, slot_count: int) -> None:
        """Reject rank extents the layout declarations make illegal."""

        alignment = self.layout.extent_alignment_slots
        slots = self.geometry.atom_slots
        _require(
            slot_count > 0 and first_slot >= 0,
            f"BTX extent [{first_slot}, {first_slot + slot_count}) is empty "
            "or negative",
        )
        _require(
            first_slot + slot_count <= slots,
            f"BTX extent [{first_slot}, {first_slot + slot_count}) exceeds "
            f"{slots} atom slots",
        )
        _require(
            first_slot % alignment == 0 and slot_count % alignment == 0,
            f"BTX extent [{first_slot}, {first_slot + slot_count}) must "
            f"align to {alignment} slots",
        )
        for barrier in self.layout.extent_barriers:
            _require(
                not (first_slot < barrier < first_slot + slot_count),
                f"BTX extent [{first_slot}, {first_slot + slot_count}) "
                f"crosses the declared barrier at slot {barrier}",
            )


def _parse_geometry(data: dict) -> BtxGeometry:
    _require_keys(
        data,
        required={
            "num_experts",
            "hidden_size",
            "intermediate_size",
            "atom_channels",
            "atom_slots",
            "moe_layer_indices",
        },
        optional=set(),
        where="BTX geometry",
    )
    for name in (
        "num_experts",
        "hidden_size",
        "intermediate_size",
        "atom_channels",
        "atom_slots",
    ):
        _require(
            isinstance(data[name], int) and data[name] > 0,
            f"BTX geometry {name} must be a positive integer",
        )
    _require(
        data["atom_channels"] == ATOM_CHANNELS,
        f"BTX atoms hold {ATOM_CHANNELS} channels; got {data['atom_channels']}",
    )
    _require(
        data["intermediate_size"] % data["atom_channels"] == 0,
        "BTX intermediate_size must be a multiple of atom_channels",
    )
    _require(
        data["atom_slots"] * data["atom_channels"] == data["intermediate_size"],
        "BTX atom_slots must equal intermediate_size / atom_channels",
    )
    _require(
        data["hidden_size"] % 16 == 0,
        "BTX hidden_size must be a multiple of 16",
    )
    indices = data["moe_layer_indices"]
    _require(
        isinstance(indices, list)
        and len(indices) > 0
        and all(isinstance(i, int) and i >= 0 for i in indices)
        and len(set(indices)) == len(indices),
        "BTX moe_layer_indices must be distinct non-negative integers",
    )
    return BtxGeometry(
        num_experts=data["num_experts"],
        hidden_size=data["hidden_size"],
        intermediate_size=data["intermediate_size"],
        atom_channels=data["atom_channels"],
        atom_slots=data["atom_slots"],
        moe_layer_indices=tuple(sorted(indices)),
    )


def _parse_rates(data: dict, *, codebook: str) -> BtxRates:
    _require_keys(
        data,
        required={"structure"},
        optional={"bits", "pair_kinds"},
        where="BTX rates",
    )
    structure = data["structure"]
    if structure == RATE_STRUCTURE_UNIFORM:
        _require(
            "bits" in data and "pair_kinds" not in data,
            "uniform BTX rates declare bits and no pair_kinds",
        )
        bits = data["bits"]
        _require(
            isinstance(bits, int) and bits in (2, 3, 4, 5, 6),
            f"BTX uniform bits must be one of 2..6, got {bits!r}",
        )
        validate_codebook_bits(codebook, bits)
        return BtxRates(structure=structure, bits=bits, pair_kinds=None)
    if structure == RATE_STRUCTURE_PER_EXPERT_PAIR:
        _require(
            "pair_kinds" in data and "bits" not in data,
            "per_expert_pair BTX rates declare pair_kinds and no bits",
        )
        kinds = data["pair_kinds"]
        _require(
            isinstance(kinds, list)
            and len(kinds) > 0
            and all(kind in PAIR_KIND_RATE_CODES for kind in kinds)
            and len(set(kinds)) == len(kinds),
            "BTX pair_kinds must be distinct members of "
            f"{sorted(PAIR_KIND_RATE_CODES)}",
        )
        for kind in kinds:
            for bits in rate_code_bits(PAIR_KIND_RATE_CODES[kind]):
                validate_codebook_bits(codebook, bits)
        return BtxRates(
            structure=structure, bits=None, pair_kinds=frozenset(kinds)
        )
    raise ValueError(
        "BTX rates structure must be 'uniform' or 'per_expert_pair', "
        f"got {structure!r}"
    )


def _parse_hadamard(data: dict, *, geometry: BtxGeometry) -> BtxHadamard:
    _require_keys(
        data,
        required={"coupled", "per_expert_input_rotations"},
        optional={"pre_block", "post_block"},
        where="BTX hadamard",
    )
    coupled = data["coupled"]
    _require(
        isinstance(coupled, bool), "BTX hadamard coupled must be a boolean"
    )
    per_expert = data["per_expert_input_rotations"]
    _require(
        isinstance(per_expert, bool),
        "BTX per_expert_input_rotations must be a boolean",
    )
    if not coupled:
        _require(
            "pre_block" not in data and "post_block" not in data,
            "BTX hadamard blocks are valid only for coupled checkpoints",
        )
        return BtxHadamard(
            coupled=False,
            pre_block=None,
            post_block=None,
            per_expert_input_rotations=per_expert,
        )
    _require(
        "pre_block" in data and "post_block" in data,
        "coupled BTX checkpoints must declare pre_block and post_block",
    )
    pre_block, post_block = data["pre_block"], data["post_block"]
    for name, value in (("pre_block", pre_block), ("post_block", post_block)):
        _require(
            isinstance(value, int) and value > 0 and value % ATOM_CHANNELS == 0,
            f"BTX hadamard {name} must be a positive multiple of "
            f"{ATOM_CHANNELS}",
        )
    _require(
        geometry.intermediate_size % post_block == 0,
        "BTX intermediate_size must be a multiple of post_block",
    )
    _require(
        geometry.hidden_size % pre_block == 0,
        "BTX hidden_size must be a multiple of pre_block",
    )
    return BtxHadamard(
        coupled=True,
        pre_block=pre_block,
        post_block=post_block,
        per_expert_input_rotations=per_expert,
    )


def _parse_layout(data: dict, *, geometry: BtxGeometry) -> BtxLayout:
    _require_keys(
        data,
        required={"atom_row_alignment", "extent_alignment_slots"},
        optional={"extent_barriers"},
        where="BTX layout",
    )
    alignment = data["atom_row_alignment"]
    _require(
        isinstance(alignment, int) and alignment > 0,
        "BTX atom_row_alignment must be a positive integer",
    )
    extent_alignment = data["extent_alignment_slots"]
    _require(
        isinstance(extent_alignment, int)
        and extent_alignment > 0
        and geometry.atom_slots % extent_alignment == 0,
        "BTX extent_alignment_slots must be a positive divisor of atom_slots",
    )
    barriers = data.get("extent_barriers", [])
    _require(
        isinstance(barriers, list)
        and all(
            isinstance(b, int) and 0 < b < geometry.atom_slots
            for b in barriers
        )
        and len(set(barriers)) == len(barriers),
        "BTX extent_barriers must be distinct interior slot indices",
    )
    return BtxLayout(
        atom_row_alignment=alignment,
        extent_alignment_slots=extent_alignment,
        extent_barriers=tuple(sorted(barriers)),
    )


def _parse_layers(data: dict, *, geometry: BtxGeometry) -> dict[int, BtxLayerRef]:
    _require(isinstance(data, dict) and data, "BTX layers must be non-empty")
    layers: dict[int, BtxLayerRef] = {}
    for key, value in data.items():
        _require(
            isinstance(key, str) and key.isdigit(),
            f"BTX layer keys must be decimal strings, got {key!r}",
        )
        index = int(key)
        _require_keys(
            value,
            required={"file", "sha256"},
            optional=set(),
            where=f"BTX layer {index}",
        )
        _require(
            isinstance(value["file"], str)
            and isinstance(value["sha256"], str)
            and len(value["sha256"]) == 64,
            f"BTX layer {index} must declare file and hex sha256",
        )
        layers[index] = BtxLayerRef(file=value["file"], sha256=value["sha256"])
    _require(
        set(layers.keys()) == set(geometry.moe_layer_indices),
        "BTX layers must cover exactly geometry.moe_layer_indices",
    )
    return layers
