"""Native canonical atom planes with independent rates per intermediate group."""

from dataclasses import dataclass

import torch

from .._shared.kernels.w4a16.prepare import TrellisWeightState
from .config import RateGranularity


@dataclass(frozen=True)
class PreparedAtomTrellisWeights:
    """Compressed atom rows and group/expert/projection word offsets.

    FC1 planes span adjacent N16 columns; FC2 planes span adjacent K16 rows.
    Each plane contains H/16 native t256 tiles. Rates and offsets describe
    the payload directly, without a decoded or uniformly repacked weight copy.
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
        "sqg_e4m3": (2, 3, 4),
        "sqg_fp16": (5, 6),
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
    atoms = weights.atoms
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
    coupled, rotations = _expert_transform_rows(
        config,
        weights,
        intermediate,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
    )
    state = TrellisWeightState(
        codebook=config.codebook.value,
        bits=5 if config.codebook.value == "sqg_fp16" else 3,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=rotations,
        down_svh=down_svh,
        coupled_hadamard=coupled,
        input_scale_split=_input_scale_split(weights, gate_suh, up_suh)
        if coupled
        else None,
    )
    return PreparedAtomTrellisWeights(
        w13=atoms.view(torch.int32).reshape(-1),
        rates=rates,
        offsets=offsets.to(device=atoms.device).contiguous(),
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
