"""BTX container reading and W4A16 MoE weight preparation.

One metadata-driven load path serves every declared configuration of the
`btx-atoms-v1` container: the manifest and rate tables locate every byte,
the extent rules come from the manifest's layout section, and preparation
reuses the shared trellis machinery (`prepare_trellis256_moe_weights`
for uniform rate structures, the pair finalizer for per-expert ones).
Nothing in this module depends on a specific model geometry, profile
name, or rate-placement convention.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, replace

import torch

from b12x.moe._shared.btx_schema import (
    ATOMS_PER_PAIR,
    BTX_MANIFEST_FILENAME,
    BTX_SCHEMA,
    BtxManifest,
    RATE_CODE_PAIR_KINDS,
    RATE_STRUCTURE_PER_EXPERT_PAIR,
    RATE_STRUCTURE_UNIFORM,
    matrix_atom_bytes,
    rate_code,
    rate_code_bits,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    PreparedW4A16MoeWeights,
    _finalize_prepared_trellis_weights,
    _coupled_rotation_signs,
    _restore_plane_words,
    prepare_trellis256_moe_weights,
)

_EXPERT_CHUNK = 64


@dataclass(frozen=True)
class BtxLayer:
    """One layer's extent-sliced content, validated against the manifest."""

    manifest: BtxManifest
    layer_index: int
    first_slot: int
    slot_count: int
    atoms: torch.Tensor
    rotations: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    rates_fc1: torch.Tensor | None
    rates_fc2: torch.Tensor | None
    rotation_draws: torch.Tensor | None

    @property
    def local_intermediate_size(self) -> int:
        return self.slot_count * self.manifest.geometry.atom_channels


def read_btx_manifest(root: str | pathlib.Path) -> BtxManifest:
    root = pathlib.Path(root)
    data = json.loads((root / BTX_MANIFEST_FILENAME).read_text())
    return BtxManifest.from_dict(data)


def read_btx_layer(
    root: str | pathlib.Path,
    manifest: BtxManifest,
    layer_index: int,
    *,
    first_slot: int,
    slot_count: int,
    verify_sha: bool = False,
) -> BtxLayer:
    """Load one rank extent of one layer as CPU tensors.

    The extent is validated against the manifest's layout declarations and
    the safetensors metadata is cross-checked against the manifest before
    any tensor is interpreted.
    """

    from safetensors import safe_open

    root = pathlib.Path(root)
    manifest.validate_extent(first_slot, slot_count)
    if layer_index not in manifest.layers:
        raise ValueError(f"BTX manifest does not declare layer {layer_index}")
    ref = manifest.layers[layer_index]
    path = root / ref.file
    if verify_sha:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ref.sha256:
            raise ValueError(
                f"BTX layer {layer_index} sha256 mismatch: manifest "
                f"{ref.sha256}, file {digest}"
            )

    geometry = manifest.geometry
    per_expert = manifest.rates.structure == RATE_STRUCTURE_PER_EXPERT_PAIR
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        expected = {
            "schema": BTX_SCHEMA,
            "codebook": manifest.codebook,
            "layer": str(int(layer_index)),
            "num_experts": str(geometry.num_experts),
            "hidden_size": str(geometry.hidden_size),
            "intermediate_size": str(geometry.intermediate_size),
            "atom_channels": str(geometry.atom_channels),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"BTX layer {layer_index} metadata {key!r} is "
                    f"{metadata.get(key)!r}; the manifest declares {value!r}"
                )
        names = set(handle.keys())
        required = {"atoms", "rotations", "gate_suh", "up_suh", "down_svh"}
        if per_expert:
            required |= {"rates_fc1", "rates_fc2"}
        if manifest.hadamard.coupled:
            required |= {"rotation_draws"}
        if names != required:
            raise ValueError(
                f"BTX layer {layer_index} tensors {sorted(names)} do not "
                f"match the declared set {sorted(required)}"
            )

        atoms_slice = handle.get_slice("atoms")
        atoms_shape = atoms_slice.get_shape()
        if (
            len(atoms_shape) != 2
            or atoms_shape[0] != geometry.atom_slots
            or atoms_shape[1] % manifest.layout.atom_row_alignment
        ):
            raise ValueError(
                f"BTX layer {layer_index} atoms shape {atoms_shape} violates "
                "the declared geometry or row alignment"
            )
        atoms = atoms_slice[first_slot : first_slot + slot_count]

        rotations_slice = handle.get_slice("rotations")
        if tuple(rotations_slice.get_shape()) != (
            geometry.atom_slots,
            geometry.num_experts,
            3,
            geometry.atom_channels,
        ):
            raise ValueError(
                f"BTX layer {layer_index} rotations shape is not "
                "[atom_slots, num_experts, 3, atom_channels]"
            )
        rotations = rotations_slice[first_slot : first_slot + slot_count]

        h_shapes = (
            (geometry.num_experts, geometry.hidden_size)
            if manifest.hadamard.per_expert_input_rotations
            else (geometry.hidden_size,)
        )
        sides = {}
        for name in ("gate_suh", "up_suh", "down_svh"):
            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != h_shapes or tensor.dtype != torch.float16:
                raise ValueError(
                    f"BTX layer {layer_index} {name} must be fp16 {h_shapes}"
                )
            sides[name] = tensor

        rates_fc1 = rates_fc2 = None
        if per_expert:
            pairs = geometry.atom_slots // ATOMS_PER_PAIR
            first_pair = first_slot // ATOMS_PER_PAIR
            pair_count = slot_count // ATOMS_PER_PAIR
            declared = manifest.rates.pair_kinds or frozenset()
            observed: set[str] = set()
            tables = {}
            for name in ("rates_fc1", "rates_fc2"):
                table_slice = handle.get_slice(name)
                if tuple(table_slice.get_shape()) != (
                    pairs,
                    geometry.num_experts,
                ):
                    raise ValueError(
                        f"BTX layer {layer_index} {name} must be "
                        "[atom_slots/8, num_experts]"
                    )
                table = table_slice[first_pair : first_pair + pair_count]
                for code in table.unique().tolist():
                    kind = RATE_CODE_PAIR_KINDS.get(int(code))
                    if kind is None:
                        raise ValueError(
                            f"BTX layer {layer_index} {name} contains "
                            f"unknown rate code {int(code):#x}"
                        )
                    observed.add(kind)
                tables[name] = table
            if not observed <= set(declared):
                raise ValueError(
                    f"BTX layer {layer_index} rate tables use kinds "
                    f"{sorted(observed)} outside the declared "
                    f"{sorted(declared)}"
                )
            rates_fc1, rates_fc2 = tables["rates_fc1"], tables["rates_fc2"]

        rotation_draws = None
        if manifest.hadamard.coupled:
            rotation_draws = handle.get_tensor("rotation_draws")
            if (
                tuple(rotation_draws.shape) != (geometry.num_experts,)
                or rotation_draws.dtype != torch.uint8
                or bool(torch.any(rotation_draws > 7))
            ):
                raise ValueError(
                    f"BTX layer {layer_index} rotation_draws must be "
                    "uint8[num_experts] in 0..7"
                )

    return BtxLayer(
        manifest=manifest,
        layer_index=layer_index,
        first_slot=first_slot,
        slot_count=slot_count,
        atoms=atoms,
        rotations=rotations,
        gate_suh=sides["gate_suh"],
        up_suh=sides["up_suh"],
        down_svh=sides["down_svh"],
        rates_fc1=rates_fc1,
        rates_fc2=rates_fc2,
        rotation_draws=rotation_draws,
    )


def _extent_rotation_tables(
    layer: BtxLayer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """suh/svh device tensors plus [E, 3*I_local] boundary values."""

    experts = layer.manifest.geometry.num_experts
    local = layer.local_intermediate_size
    values = layer.rotations.to(device=device)
    columns = [
        values[:, :, matrix, :].permute(1, 0, 2).reshape(experts, local)
        for matrix in range(3)
    ]
    intermediate = torch.cat(columns, dim=1).contiguous()

    def _side(tensor: torch.Tensor) -> torch.Tensor:
        moved = tensor.to(device=device)
        if moved.dim() == 1:
            moved = moved.reshape(1, -1)
        return moved.contiguous()

    return (
        _side(layer.gate_suh),
        _side(layer.up_suh),
        _side(layer.down_svh),
        intermediate,
    )


def _coupled_rotation_rows(
    layer: BtxLayer, intermediate: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Append the frozen coupled sign rows: [values 3I | pre 2I | post I]."""

    manifest = layer.manifest
    if (manifest.hadamard.pre_block, manifest.hadamard.post_block) != (
        512,
        128,
    ):
        raise ValueError(
            "coupled BTX preparation currently implements pre/post Hadamard "
            "blocks (512, 128); the manifest declares "
            f"({manifest.hadamard.pre_block}, {manifest.hadamard.post_block})"
        )
    assert layer.rotation_draws is not None
    experts = manifest.geometry.num_experts
    global_i = manifest.geometry.intermediate_size
    local = layer.local_intermediate_size
    pre_begin = 2 * layer.first_slot * manifest.geometry.atom_channels
    post_begin = layer.first_slot * manifest.geometry.atom_channels
    signs = torch.empty((experts, 3 * local), dtype=torch.float16)
    draws = layer.rotation_draws
    for draw in sorted(set(int(value) for value in draws.tolist())):
        rows = torch.nonzero(draws == draw, as_tuple=False).flatten()
        pre = _coupled_rotation_signs(2 * global_i, draw=draw, axis=1)[
            pre_begin : pre_begin + 2 * local
        ]
        post = _coupled_rotation_signs(global_i, draw=draw, axis=2)[
            post_begin : post_begin + local
        ]
        signs.index_copy_(
            0,
            rows,
            torch.cat((pre, post)).to(torch.float16).expand(rows.numel(), -1),
        )
    return torch.cat(
        (intermediate, signs.to(device=device)), dim=1
    ).contiguous()


def _uniform_native_tensors(
    layer: BtxLayer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble native trellis_t256 tensors from a uniform-rate extent.

    Returns projection-major FC1 ``[2, E, H/16, I_local/16, 16*bits]`` and
    FC2 ``[E, I_local/16, H/16, 16*bits]`` int16 tensors.
    """

    manifest = layer.manifest
    geometry = manifest.geometry
    bits = manifest.rates.bits
    assert bits is not None
    experts = geometry.num_experts
    hidden_tiles = geometry.hidden_size // 16
    slots = layer.slot_count
    section = matrix_atom_bytes(geometry.hidden_size, bits, bits)
    payload = experts * 3 * section
    if layer.atoms.shape[1] < payload:
        raise ValueError("BTX atoms rows are shorter than their expert bundles")
    if bool(torch.any(layer.atoms[:, payload:] != 0)):
        raise ValueError("BTX atoms row padding must be zero")

    w13 = torch.empty(
        (2, experts, hidden_tiles, 2 * slots, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.empty(
        (experts, 2 * slots, hidden_tiles, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    bundles = layer.atoms[:, :payload].reshape(slots, experts, 3 * section)
    for first in range(0, experts, _EXPERT_CHUNK):
        count = min(_EXPERT_CHUNK, experts - first)
        chunk = (
            bundles[:, first : first + count]
            .contiguous()
            .to(device=device)
            .view(torch.int16)
            .reshape(slots, count, 3, 2, hidden_tiles, 16 * bits)
        )
        for matrix in range(2):
            # FC1 planes are the atom's two consecutive N16 columns.
            w13[matrix, first : first + count].copy_(
                chunk[:, :, matrix].permute(1, 3, 0, 2, 4).reshape(
                    count, hidden_tiles, 2 * slots, 16 * bits
                )
            )
        # FC2 planes are the atom's two consecutive K16 rows.
        w2[first : first + count].copy_(
            chunk[:, :, 2].permute(1, 0, 2, 3, 4).reshape(
                count, 2 * slots, hidden_tiles, 16 * bits
            )
        )
    return w13, w2


def prepare_btx_moe_weights(
    layer: BtxLayer,
    *,
    activation: str,
    device: torch.device | str,
    params_dtype: torch.dtype = torch.float16,
    tile_config: tuple[int, int, int, int] | None = None,
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    """Prepare one BTX rank extent for the fused W4A16 serving path."""

    manifest = layer.manifest
    device = torch.device(device)
    gate_suh, up_suh, down_svh, intermediate = _extent_rotation_tables(
        layer, device
    )
    rotations = intermediate
    input_scale_split = None
    if manifest.hadamard.coupled:
        rotations = _coupled_rotation_rows(layer, intermediate, device)
        # The coupled residual transform interleaves gate/up into one
        # length-2I axis whose two stored halves each carry one input-side
        # table. Both physical FC1 slots use the same column-wise split.
        pre_half_slots = manifest.geometry.atom_slots // 2
        if layer.first_slot < pre_half_slots < layer.first_slot + layer.slot_count:
            input_scale_split = (
                pre_half_slots - layer.first_slot
            ) * manifest.geometry.atom_channels
        else:
            source_suh = gate_suh if layer.first_slot < pre_half_slots else up_suh
            gate_suh = source_suh
            up_suh = source_suh

    if manifest.rates.structure == RATE_STRUCTURE_UNIFORM:
        assert manifest.rates.bits is not None
        w13, w2 = _uniform_native_tensors(layer, device)
        if tile_config is None:
            tile_config = (
                (128, 128, 128, 128)
                if manifest.hadamard.coupled and (
                    manifest.rates.bits == 2
                    or (
                        input_scale_split is not None
                        and layer.local_intermediate_size % 256
                    )
                )
                else (64, 256, 64, 256)
            )
        prepared = prepare_trellis256_moe_weights(
            w13=w13,
            w2=w2,
            hidden_size=manifest.geometry.hidden_size,
            intermediate_size=layer.local_intermediate_size,
            num_experts=manifest.geometry.num_experts,
            activation=activation,
            fc1_tile_n=tile_config[1],
            fc2_tile_n=tile_config[3],
            device=device,
            params_dtype=params_dtype,
            w13_layout="trellis_t256_proj",
            trellis_bits=manifest.rates.bits,
            codebook=manifest.codebook,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate,
            down_svh=down_svh,
            tile_config=tile_config,
            dummy_scale=dummy_scale,
            workspace=workspace,
        )
        if not manifest.hadamard.coupled:
            return prepared
        assert prepared.trellis is not None
        return replace(
            prepared,
            trellis=replace(
                prepared.trellis,
                coupled_hadamard=True,
                intermediate_rotations=rotations,
                input_scale_split=input_scale_split,
            ),
        )

    return _prepare_btx_pair_extent(
        layer,
        device=device,
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
        rotations=rotations,
        params_dtype=params_dtype,
        tile_config=tile_config or (64, 256, 64, 256),
        dummy_scale=dummy_scale,
        workspace=workspace,
    )


def _prepare_btx_pair_extent(
    layer: BtxLayer,
    *,
    device: torch.device,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    rotations: torch.Tensor,
    params_dtype: torch.dtype,
    tile_config: tuple[int, int, int, int],
    dummy_scale: torch.Tensor | None,
    workspace: torch.Tensor | None,
) -> PreparedW4A16MoeWeights:
    """Prepare a per-expert-pair extent through the pair kernel machinery.

    The fused kernel's pair decode operates on one 256-channel pair per
    rank (FC2 pairs lie on the local K axis), so a per-expert-rate extent
    is exactly one pair of atom slots.
    """

    manifest = layer.manifest
    geometry = manifest.geometry
    if layer.slot_count != ATOMS_PER_PAIR:
        raise ValueError(
            "per-expert-pair BTX extents must cover exactly one "
            f"256-channel pair ({ATOMS_PER_PAIR} atom slots); got "
            f"{layer.slot_count}"
        )
    if manifest.hadamard.coupled:
        raise ValueError(
            "coupled-Hadamard execution of per-expert-pair BTX extents has "
            "no qualified kernel path"
        )
    # The pair runtime orders each 256-channel pair record-major: every
    # atom contributes its first 16 channels to the low record and its
    # last 16 to the high record. Rotation rows must match that order.
    experts_count = geometry.num_experts
    per_matrix = []
    values = layer.rotations.to(rotations.device)
    for matrix in range(3):
        planes = values[:, :, matrix, :].reshape(
            ATOMS_PER_PAIR, experts_count, 2, 16
        )
        low = planes[:, :, 0, :].permute(1, 0, 2).reshape(experts_count, -1)
        high = planes[:, :, 1, :].permute(1, 0, 2).reshape(experts_count, -1)
        per_matrix.append(torch.cat((low, high), dim=1))
    rotations = torch.cat(per_matrix, dim=1).contiguous()
    assert layer.rates_fc1 is not None and layer.rates_fc2 is not None
    fc1_codes = layer.rates_fc1[0].to(torch.int64)
    fc2_codes = layer.rates_fc2[0].to(torch.int64)
    experts = geometry.num_experts
    hidden_tiles = geometry.hidden_size // 16

    kinds = {
        RATE_CODE_PAIR_KINDS[int(code)]
        for code in torch.cat((fc1_codes, fc2_codes)).unique().tolist()
    }
    if kinds == {"P33"} or kinds == {"P33", "P24"}:
        pair_kind = "PDYNAMIC"
        high_code = rate_code(2, 4)
    elif kinds == {"P33", "P43"}:
        pair_kind = "P33_P43"
        high_code = rate_code(4, 3)
    else:
        raise ValueError(
            f"BTX per-expert extents with pair kinds {sorted(kinds)} have "
            "no fused execution arm; whole-expert K4 tiers run through "
            "mixed-tier or multi-launch execution"
        )

    def _restore(codes: torch.Tensor, matrix: int, *, fc1: bool):
        sections = []
        for expert in range(experts):
            low_bits, high_bits = rate_code_bits(int(codes[expert]))
            begin = 0
            for m in range(matrix):
                m_codes = fc1_codes if m < 2 else fc2_codes
                lo, hi = rate_code_bits(int(m_codes[expert]))
                begin += matrix_atom_bytes(geometry.hidden_size, lo, hi)
            section = matrix_atom_bytes(geometry.hidden_size, low_bits, high_bits)
            raw = layer.atoms[:, _bundle_offset(layer, expert) + begin :][
                :, :section
            ]
            words = (
                raw.contiguous()
                .to(device=device)
                .view(torch.int16)
                .reshape(ATOMS_PER_PAIR, 1, -1)
                .permute(1, 0, 2)
            )
            low_words = hidden_tiles * 16 * low_bits
            low = words[..., :low_words].reshape(
                1, ATOMS_PER_PAIR, hidden_tiles, 16 * low_bits
            )
            high = words[..., low_words:].reshape(
                1, ATOMS_PER_PAIR, hidden_tiles, 16 * high_bits
            )
            sections.append(_restore_plane_words(low, high, fc1=fc1))
        return sections

    if pair_kind == "PDYNAMIC":
        fc1_modes = (fc1_codes == high_code).to(torch.int32).to(device)
        fc2_modes = (fc2_codes == high_code).to(torch.int32).to(device)
        gate = _restore(fc1_codes, 0, fc1=True)
        up = _restore(fc1_codes, 1, fc1=True)
        down = _restore(fc2_codes, 2, fc1=False)
        w13 = torch.cat(
            [torch.cat(gate, dim=0), torch.cat(up, dim=0)]
        ).reshape(-1)
        w2 = torch.cat(down, dim=0).reshape(-1)
        fc1_pair_modes: torch.Tensor = fc1_modes.contiguous()
        fc2_pair_modes: torch.Tensor = fc2_modes.contiguous()
    else:
        # Compact gap-free pools with per-expert descriptors, matching the
        # fused kernel's P33_P43 addressing.
        def _compact(codes, sections):
            lengths = torch.tensor(
                [section.numel() // 2 for section in sections],
                dtype=torch.int64,
            )
            offsets = torch.zeros_like(lengths)
            offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
            modes = (codes == high_code).to(torch.int64)
            descriptors = ((offsets << 1) | modes).to(device)
            return offsets, descriptors

        gate = _restore(fc1_codes, 0, fc1=True)
        up = _restore(fc1_codes, 1, fc1=True)
        down = _restore(fc2_codes, 2, fc1=False)
        _, fc1_descriptors = _compact(fc1_codes, gate)
        _, fc2_descriptors = _compact(fc2_codes, down)
        w13 = torch.cat(
            [torch.cat(gate, dim=1), torch.cat(up, dim=1)]
        ).reshape(-1)
        w2 = torch.cat(down, dim=1).reshape(-1)
        fc1_pair_modes = fc1_descriptors.contiguous()
        fc2_pair_modes = fc2_descriptors.contiguous()

    return _finalize_prepared_trellis_weights(
        context="BTX per-expert-pair preparation",
        device=device,
        hidden_size=geometry.hidden_size,
        intermediate_size=layer.local_intermediate_size,
        num_experts=experts,
        params_dtype=params_dtype,
        w13=w13,
        w2=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=rotations,
        down_svh=down_svh,
        rotation_columns=rotations.shape[1],
        tile_config=tile_config,
        required_fc1_tile_n=256,
        dummy_scale=dummy_scale,
        workspace=workspace,
        codebook=manifest.codebook,
        trellis_bits=3,
        fc1_pair_kind=pair_kind,
        fc2_pair_kind=pair_kind,
        fc1_pair_modes=fc1_pair_modes,
        fc2_pair_modes=fc2_pair_modes,
        coupled_hadamard=manifest.hadamard.coupled,
    )


def _bundle_offset(layer: BtxLayer, expert: int) -> int:
    """Byte offset of one expert's bundle within this extent's rows."""

    manifest = layer.manifest
    if manifest.rates.structure == RATE_STRUCTURE_UNIFORM:
        assert manifest.rates.bits is not None
        section = matrix_atom_bytes(
            manifest.geometry.hidden_size,
            manifest.rates.bits,
            manifest.rates.bits,
        )
        return expert * 3 * section
    assert layer.rates_fc1 is not None and layer.rates_fc2 is not None
    offset = 0
    for e in range(expert):
        fc1_lo, fc1_hi = rate_code_bits(int(layer.rates_fc1[0, e]))
        fc2_lo, fc2_hi = rate_code_bits(int(layer.rates_fc2[0, e]))
        offset += 2 * matrix_atom_bytes(
            manifest.geometry.hidden_size, fc1_lo, fc1_hi
        ) + matrix_atom_bytes(manifest.geometry.hidden_size, fc2_lo, fc2_hi)
    return offset
