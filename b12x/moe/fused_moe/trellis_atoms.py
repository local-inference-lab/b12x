"""Compressed atom planes with independent rates per intermediate group."""

from dataclasses import dataclass

import torch

from .._shared.kernels.w4a16.prepare import TrellisWeightState
from .config import RateGranularity


@dataclass(frozen=True)
class PreparedAtomTrellisWeights:
    """Compressed atom rows and group/expert/projection word offsets.

    Canonical FC1/FC2 planes span adjacent N16/K16 tiles. EXL3 paired planes
    contribute one tile to each of two 128-channel records. Each plane contains
    H/16 native t256 tiles; rates use the low nibble for the first plane.
    Offsets describe the compressed payload without a decoded weight copy.
    """

    w13: torch.Tensor
    rates: torch.Tensor
    offsets: torch.Tensor
    row_stride_words: int
    group_size: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    trellis: TrellisWeightState
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    workspace: torch.Tensor
    params_dtype: torch.dtype = torch.float16
    source_format: str = "b12x_trellis"
    weight_layout: str = "trellis_atoms"
    w13_layout: str = "trellis_atoms"
    scale_format: str = "e4m3_k32"
    paired_records: bool = False

    @property
    def w2(self):
        return self.w13

    @property
    def w2_scale(self):
        return self.w13_scale

    @property
    def w2_global_scale(self):
        return self.w13_global_scale


def normalize_rates(config, rate, *, experts, intermediate_size, device):
    """Validate checkpoint nibbles and return contiguous [group,expert,3] rates."""
    from .trellis import _require_cuda_tensor

    _require_cuda_tensor(rate, name="rate", dtype=torch.uint8, device=device)
    group_size = config.rate.group_size or intermediate_size
    if intermediate_size % group_size:
        raise ValueError("Trellis group size must divide the local intermediate width")
    groups = intermediate_size // group_size
    grouped = config.rate.group_size is not None
    granularity = config.rate.granularity
    if granularity in (RateGranularity.UNIFORM, RateGranularity.PER_LAYER):
        shapes = {(groups,), (1, groups)} if grouped else {(), (1,)}
        if tuple(rate.shape) not in shapes:
            raise ValueError(
                f"selected uniform/per-layer rates must have shape {shapes}"
            )
        rates = rate.reshape(groups, 1, 1).expand(groups, experts, 3)
    elif granularity is RateGranularity.PER_EXPERT:
        shape = (experts, groups) if grouped else (experts,)
        if tuple(rate.shape) != shape:
            raise ValueError(f"selected per-expert rates must have shape {shape}")
        rates = rate.reshape(experts, groups).T[:, :, None].expand(groups, experts, 3)
    else:
        shape = (experts, 3, groups) if grouped else (experts, 3)
        if tuple(rate.shape) != shape:
            raise ValueError(f"selected per-projection rates must have shape {shape}")
        rates = rate.reshape(experts, 3, groups).permute(2, 0, 1)
    host = rates.detach().cpu().to(torch.int64)
    low, high = host & 15, host >> 4
    allowed = {
        "mcg": (2, 3, 4, 5, 6),
        "lut_e4m3": (2, 3, 4),
        "lut_fp16": (5, 6),
    }[config.codebook.value]
    observed = set(low.flatten().tolist()) | set(high.flatten().tolist())
    if not observed.issubset(allowed):
        raise ValueError(
            f"{config.codebook.value} atom planes require K{allowed}; observed {sorted(observed)}"
        )
    atom_layout = (
        grouped
        or not torch.equal(low, high)
        or (config.codebook.value == "mcg" and not observed.issubset((3, 4, 5)))
        or (config.codebook.value != "mcg" and len(observed) != 1)
    )
    return rates.contiguous(), atom_layout


def _atom_offsets(atoms, rates, group_size, hidden_size):
    if atoms.dtype != torch.uint8 or atoms.ndim != 2 or not atoms.is_contiguous():
        raise ValueError("Trellis atom storage must be contiguous uint8 rows")
    if atoms.data_ptr() % 16 or atoms.shape[1] % 16:
        raise ValueError(
            "Trellis atom storage and row stride must be aligned to 16 bytes"
        )
    # Native planes store eight Uint32 words per bit and hidden tile.
    host = rates.detach().cpu().to(torch.int64)
    words = ((host & 15) + (host >> 4)) * (hidden_size // 16) * 8
    sections = words.reshape(words.shape[0], -1)
    offsets = (sections.cumsum(1) - sections).reshape_as(words)
    used = sections.sum(1).tolist()
    row_stride_words = atoms.shape[1] // 4
    if row_stride_words < max(used):
        raise ValueError("Trellis atom rows are shorter than their group payload")
    slots = group_size // 32
    for group, size in enumerate(used):
        padding = atoms[group * slots : (group + 1) * slots, size * 4 :]
        if torch.any(padding != 0).item():
            raise ValueError("Trellis atom row padding must be zero")
    return offsets.to(device=atoms.device).contiguous(), row_stride_words


def prepare_atom_weights(
    config,
    weights,
    rates,
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    gate_suh,
    up_suh,
    intermediate,
    down_svh,
    params_dtype=torch.float16,
):
    from .trellis import _expert_transform_rows, _input_scale_split

    group_size = config.rate.group_size or intermediate_size
    if config.rate.group_size is not None and weights.intermediate_offset % group_size:
        raise ValueError("Trellis rank extent must start at a rate-group boundary")
    atoms = weights.codes
    offsets, row_stride_words = _atom_offsets(atoms, rates, group_size, hidden_size)
    intermediate_hadamard, rotations = _expert_transform_rows(
        config,
        weights,
        intermediate,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
    )
    state = TrellisWeightState(
        codebook=config.codebook.value,
        bits=5 if config.codebook.value == "lut_fp16" else 3,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=rotations,
        down_svh=down_svh,
        intermediate_hadamard=intermediate_hadamard,
        input_scale_split=_input_scale_split(weights, gate_suh, up_suh)
        if intermediate_hadamard
        else None,
    )
    return PreparedAtomTrellisWeights(
        w13=atoms.view(torch.int32).reshape(-1),
        rates=rates,
        offsets=offsets,
        row_stride_words=row_stride_words,
        group_size=group_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        trellis=state,
        w13_scale=torch.zeros(4, dtype=torch.uint8, device=atoms.device),
        w13_global_scale=torch.ones(
            num_experts, dtype=torch.float32, device=atoms.device
        ),
        workspace=torch.empty(0, dtype=torch.int32, device=atoms.device),
        params_dtype=params_dtype,
    )


def prepare_exl3_atom_weights(
    layer, *, activation, device, params_dtype=torch.float16,
    dummy_scale=None, workspace=None,
):
    """Retain EXL3 rows while restoring the pair's 128-channel record order."""
    from .._shared.exl3_schema import RATE_CODE_PAIR_KINDS
    from .._shared.trellis_codebooks import validate_codebook_bits
    from .._shared.kernels.w4a16.exl3 import (
        _extent_rotation_tables, _intermediate_hadamard_rotation_rows, _intermediate_hadamard_input_tables,
    )

    manifest = layer.manifest
    manifest.validate_extent(layer.first_slot, layer.slot_count)
    if manifest.rates.structure != "per_expert_pair":
        raise ValueError("EXL3 atom preparation requires per-expert-pair records")
    hidden, width, experts = (
        manifest.geometry.hidden_size,
        layer.local_intermediate_size,
        manifest.geometry.num_experts,
    )
    if hidden % 128 or width % 256:
        raise ValueError("EXL3 paired execution requires H divisible by 128 and whole 256-channel pairs")
    if activation not in {"silu", "situ"} or (
        manifest.hadamard.intermediate_hadamard and (activation != "situ" or hidden % 512)
    ):
        raise ValueError("EXL3 paired execution requires SiLU or SiTU; intermediate-Hadamard execution requires SiTU and H divisible by 512")
    if layer.rotations.dtype != torch.float16 or layer.rotations.shape != (
        layer.slot_count, experts, 3, 32
    ):
        raise ValueError("EXL3 paired rotations must be fp16 [slot,expert,3,32]")
    side_shape = (experts, hidden) if manifest.hadamard.per_expert_input_rotations else (hidden,)
    for side in (layer.gate_suh, layer.up_suh, layer.down_svh):
        if side.dtype != torch.float16 or side.shape != side_shape:
            raise ValueError("EXL3 side tables differ from the manifest's dtype or geometry")
    if manifest.hadamard.intermediate_hadamard:
        sign_patterns = layer.sign_pattern
        if (
            sign_patterns is None or sign_patterns.dtype != torch.uint8 or sign_patterns.shape != (experts,)
            or torch.any(sign_patterns > 7).item()
        ):
            raise ValueError("EXL3 sign patterns must be uint8 [expert] in 0..7")
    groups = width // 256
    tables = []
    for table in (layer.rates_fc1, layer.rates_fc2):
        if table is None or table.dtype != torch.uint8 or table.shape != (groups, experts):
            raise ValueError("EXL3 paired rates must be uint8 [pair,expert]")
        for code in table.detach().cpu().unique().tolist():
            kind = RATE_CODE_PAIR_KINDS.get(code)
            if kind is None or kind not in manifest.rates.pair_kinds:
                raise ValueError("EXL3 paired rates differ from their declared pair kinds")
            validate_codebook_bits(manifest.codebook, code >> 4)
            validate_codebook_bits(manifest.codebook, code & 15)
        # The internal atom table stores its low-record rate in the low nibble.
        tables.append(((table >> 4) | (table << 4)).to(device=device))
    rates = torch.stack((tables[0], tables[0], tables[1]), dim=-1).contiguous()
    atoms = layer.codes.to(device=device)
    if atoms.ndim != 2 or atoms.shape[0] != layer.slot_count:
        raise ValueError("EXL3 paired atom rows differ from the declared extent")
    offsets, stride = _atom_offsets(atoms, rates, 256, hidden)
    gate, up, down, _ = _extent_rotation_tables(layer, torch.device(device))
    # Every atom contributes N16/K16 to each 128-channel record in its pair.
    rotations = (
        layer.rotations.to(device=device)
        .reshape(groups, 8, experts, 3, 2, 16)
        .permute(2, 3, 0, 4, 1, 5)
        .reshape(experts, 3 * width)
        .contiguous()
    )
    split = None
    if manifest.hadamard.intermediate_hadamard:
        rotations = _intermediate_hadamard_rotation_rows(layer, rotations, torch.device(device))
        gate, up, split = _intermediate_hadamard_input_tables(layer, gate, up)
    state = TrellisWeightState(
        codebook=manifest.codebook, bits=3,
        gate_suh=gate, up_suh=up, down_svh=down,
        intermediate_rotations=rotations,
        intermediate_hadamard=manifest.hadamard.intermediate_hadamard,
        input_scale_split=split,
    )
    return PreparedAtomTrellisWeights(
        w13=atoms.view(torch.int32).reshape(-1), rates=rates, offsets=offsets,
        row_stride_words=stride, group_size=256, hidden_size=hidden,
        intermediate_size=width, num_experts=experts, trellis=state,
        w13_scale=dummy_scale if dummy_scale is not None else torch.zeros(4, dtype=torch.uint8, device=device),
        w13_global_scale=torch.ones(experts, dtype=torch.float32, device=device),
        workspace=workspace if workspace is not None else torch.empty(0, dtype=torch.int32, device=device),
        params_dtype=params_dtype, source_format="exl3", paired_records=True,
    )
