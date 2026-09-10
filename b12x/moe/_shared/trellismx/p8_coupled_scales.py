"""Validated P8 scale-sandwich metadata and exact CPU reference.

This module deliberately has no CUTLASS/CUDA imports.  It validates the five
FP16 scale roles before the native runtime moves them to a device and records
the storage casts used by Luke's coupled GLM reference.  The scale component
is not the H512/H128/sign transform itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import torch


SCHEMA = "glm53-p8-coupled-scale-component-tp4-rank.v1"
COUPLED_SCHEMA = "glm53-p8-coupled-h512-h128-tp4-rank.v1"
COMPONENT = "p8-suh-svh-scale-sandwich-v1"
COUPLED_COMPONENT = "p8-coupled-h512-h128-sign-v1"
COMPOSITION_TARGET = "coupled-h512-h128-suh-svh-v1"
CAST_ORDER = "mul-f32-cvt-rn-f16-h128"
COUPLED_CAST_ORDER = "luke-qsrt-coupled-reference-v1"
SIGN_GENERATOR = "qsrt-coupled-signs-v1"
SIGN_DRAW = 0
TRANSFORM_ID = "normalized-h512-outer-h128-inner-sign-draw0-v1"

SCALE_NAMES = (
    "gate_up_suh_fp16",
    "intermediate_scales_fp16",
    "down_svh_fp16",
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a CPU-contiguous tensor's physical bytes."""

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _hadamard_128(*, device: torch.device | str = "cpu") -> torch.Tensor:
    cols = torch.arange(128, dtype=torch.int64)
    rows = []
    for row in range(128):
        parity = torch.tensor(
            [(row & int(col)).bit_count() & 1 for col in cols],
            dtype=torch.bool,
        )
        rows.append(torch.where(parity, -1.0, 1.0))
    return (torch.stack(rows) / (128.0**0.5)).to(device=device)


def had128_scaled(
    value: torch.Tensor,
    *,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    store_fp16: bool,
) -> torch.Tensor:
    """Optional suh multiply, FP16 store, H128, then optional svh multiply."""

    if value.ndim != 2 or value.shape[1] % 128:
        raise ValueError("H128 input must be rank 2 with width divisible by 128")
    work = value.float()
    if suh is not None:
        work = (work * suh.float()).to(torch.float16).float()
    rows, width = work.shape
    had = _hadamard_128(device=work.device)
    work = (work.view(rows, width // 128, 128) @ had).reshape(rows, width)
    if svh is not None:
        work = work * svh.float()
    return work.to(torch.float16) if store_fp16 else work


@dataclass(frozen=True)
class P8ScaleSandwich:
    """TP-local five-role scale component.

    ``intermediate_scales`` stores ``gate_svh | up_svh | down_suh`` along its
    last dimension.  The shared input/output vectors are replicated per rank.
    """

    gate_up_suh: torch.Tensor
    intermediate_scales: torch.Tensor
    down_svh: torch.Tensor
    coupled_signs: torch.Tensor | None = None
    full_coupled: bool = False
    transform_sha256: str | None = None

    @property
    def packed(self) -> torch.Tensor:
        return torch.cat(
            (
                self.gate_up_suh.reshape(-1),
                self.intermediate_scales.reshape(-1),
                self.down_svh.reshape(-1),
                *(
                    (self.coupled_signs.reshape(-1),)
                    if self.coupled_signs is not None
                    else ()
                ),
            )
        ).contiguous()

    def split_intermediate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intermediate = self.intermediate_scales.shape[1] // 3
        return self.intermediate_scales.split(intermediate, dim=1)

    def split_signs(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.coupled_signs is None:
            raise RuntimeError("scale-only P8 component has no coupled signs")
        intermediate = self.coupled_signs.numel() // 3
        return self.coupled_signs[: 2 * intermediate], self.coupled_signs[2 * intermediate :]


def qsrt_coupled_signs_reference(
    length: int,
    *,
    draw: int,
    axis: int,
) -> torch.Tensor:
    """Luke/QSRT fixed sign generator, byte-identical on CPU."""

    if draw == 0:
        return torch.ones(length, dtype=torch.float16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (0x6A09E667F3BCC909 * draw + 0xBB67AE8584CAA73B * axis)
        & ((1 << 63) - 1)
    )
    return (
        torch.randint(0, 2, (length,), generator=generator)
        .mul_(2)
        .sub_(1)
        .to(torch.float16)
    )


def rank_local_coupled_signs(
    *,
    intermediate: int,
    rank: int,
    world_size: int = 4,
    draw: int = SIGN_DRAW,
) -> torch.Tensor:
    """Generate the contiguous atom32 TP slice used by the encoder/runtime."""

    if intermediate % 32 or rank not in range(world_size):
        raise ValueError("coupled signs require an atom32-aligned valid TP slice")
    global_intermediate = intermediate * world_size
    first_atom = rank * (intermediate // 32)
    pre_global = qsrt_coupled_signs_reference(
        2 * global_intermediate, draw=draw, axis=1
    )
    post_global = qsrt_coupled_signs_reference(
        global_intermediate, draw=draw, axis=2
    )
    pre_begin = 2 * first_atom * 32
    post_begin = first_atom * 32
    return torch.cat(
        (
            pre_global[pre_begin : pre_begin + 2 * intermediate],
            post_global[post_begin : post_begin + intermediate],
        )
    ).contiguous()


def validate_scale_component(
    metadata: Mapping[str, str],
    tensors: Mapping[str, torch.Tensor],
    *,
    layer: int,
    rank: int,
    experts: int,
    hidden: int,
    intermediate: int,
) -> P8ScaleSandwich:
    """Validate scale metadata and byte hashes, rejecting partial coupling."""

    required = {
        "schema": SCHEMA,
        "layer": str(layer),
        "rank": str(rank),
        "world_size": "4",
        "component": COMPONENT,
        "composition_target": COMPOSITION_TARGET,
        "boundary": "h128-suh-svh-scale-component",
        "cast_order": CAST_ORDER,
        "gate_up_suh_shared": "true",
        "down_svh_shared": "true",
        # This sidecar implements only the scales.  A caller must separately
        # own the H512/H128/sign stages before it can advertise full coupling.
        "full_coupled": "false",
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"invalid P8 scale component metadata: {mismatches}")

    expected_shapes = {
        "gate_up_suh_fp16": (hidden,),
        "intermediate_scales_fp16": (experts, 3 * intermediate),
        "down_svh_fp16": (hidden,),
    }
    checked: dict[str, torch.Tensor] = {}
    for name, shape in expected_shapes.items():
        tensor = tensors.get(name)
        if tensor is None:
            raise RuntimeError(f"missing P8 scale tensor {name}")
        if tensor.dtype != torch.float16 or tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"invalid {name}: expected FP16 {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite P8 scale tensor {name}")
        if torch.count_nonzero(tensor) != tensor.numel():
            raise RuntimeError(f"zero-valued P8 scale tensor {name}")
        expected_hash = metadata.get(f"sha256_{name}")
        actual_hash = tensor_sha256(tensor)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"P8 scale hash mismatch for {name}: "
                f"expected={expected_hash!r} actual={actual_hash}"
            )
        checked[name] = tensor.contiguous()

    # Signed scales are intentional format evidence.  Do not require that
    # every checkpoint contains a negative value, but reject metadata which
    # claims a positive-only ABI.
    if metadata.get("signed_scales") != "true":
        raise RuntimeError("P8 scale component must declare signed_scales=true")

    return P8ScaleSandwich(
        gate_up_suh=checked["gate_up_suh_fp16"],
        intermediate_scales=checked["intermediate_scales_fp16"],
        down_svh=checked["down_svh_fp16"],
    )


def validate_coupled_component(
    metadata: Mapping[str, str],
    tensors: Mapping[str, torch.Tensor],
    *,
    layer: int,
    rank: int,
    experts: int,
    hidden: int,
    intermediate: int,
    expected_transform_sha256: str | None = None,
) -> P8ScaleSandwich:
    """Fail-closed validation for the complete H512/H128/sign candidate."""

    transform_sha256 = metadata.get("encoder_transform_sha256")
    if (
        not isinstance(transform_sha256, str)
        or len(transform_sha256) != 64
        or any(char not in "0123456789abcdef" for char in transform_sha256)
    ):
        raise RuntimeError("coupled P8 sidecar lacks encoder transform identity")
    if (
        expected_transform_sha256 is not None
        and transform_sha256 != expected_transform_sha256
    ):
        raise RuntimeError("coupled P8 encoder/runtime transform identity mismatch")

    local_atoms = intermediate // 32
    required = {
        "schema": COUPLED_SCHEMA,
        "layer": str(layer),
        "rank": str(rank),
        "world_size": "4",
        "component": COUPLED_COMPONENT,
        "composition_target": COMPOSITION_TARGET,
        "boundary": COMPOSITION_TARGET,
        "cast_order": COUPLED_CAST_ORDER,
        "full_coupled": "true",
        "h512": "normalized-sylvester-512-v1",
        "h128": "normalized-sylvester-128-v1",
        "transform_id": TRANSFORM_ID,
        "fc1_interleave": "slot0-atom32-slot1-atom32-v1",
        "sign_generator": SIGN_GENERATOR,
        "sign_draw": str(SIGN_DRAW),
        "sign_pre_axis": "1",
        "sign_post_axis": "2",
        "activation": "silu-cap10",
        "global_intermediate": str(intermediate * 4),
        "local_atom_begin": str(rank * local_atoms),
        "tp_slice": "contiguous-atom32-v1",
        "gate_up_suh_shared": "true",
        "down_svh_shared": "true",
        "coupled_signs_shared": "true",
        "signed_scales": "true",
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"invalid full-coupled P8 metadata: {mismatches}")

    # Reuse the tensor checks without weakening the scale-only schema.
    scale_metadata = dict(metadata)
    scale_metadata.update(
        {
            "schema": SCHEMA,
            "component": COMPONENT,
            "boundary": "h128-suh-svh-scale-component",
            "cast_order": CAST_ORDER,
            "gate_up_suh_shared": "true",
            "down_svh_shared": "true",
            "full_coupled": "false",
        }
    )
    scales = validate_scale_component(
        scale_metadata,
        tensors,
        layer=layer,
        rank=rank,
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    signs = rank_local_coupled_signs(intermediate=intermediate, rank=rank)
    if metadata.get("sha256_coupled_signs_fp16") != tensor_sha256(signs):
        raise RuntimeError("coupled P8 fixed sign seed/hash mismatch")
    return P8ScaleSandwich(
        gate_up_suh=scales.gate_up_suh,
        intermediate_scales=scales.intermediate_scales,
        down_svh=scales.down_svh,
        coupled_signs=signs,
        full_coupled=True,
        transform_sha256=transform_sha256,
    )


def hadamard_blocks(value: torch.Tensor, size: int) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] % size or size & (size - 1):
        raise ValueError("Hadamard input must be rank 2 and power-of-two aligned")
    work = value.float().reshape(-1, size).clone()
    stride = 1
    while stride < size:
        work = work.view(-1, size // (2 * stride), 2, stride)
        left = work[:, :, 0, :].clone()
        right = work[:, :, 1, :].clone()
        work[:, :, 0, :] = left + right
        work[:, :, 1, :] = left - right
        work = work.view(-1, size)
        stride *= 2
    return (work / (size**0.5)).view_as(value.float())


def coupled_reference(
    x: torch.Tensor,
    gate_physical: torch.Tensor,
    up_physical: torch.Tensor,
    down_physical: torch.Tensor,
    scales: P8ScaleSandwich,
) -> torch.Tensor:
    """Exact CPU topology for the complete Luke/QSRT coupled P8 boundary."""

    if not scales.full_coupled:
        raise RuntimeError("full coupled reference rejects a scale-only component")
    gate_svh, up_svh, down_suh = scales.split_intermediate()
    pre_signs, post_signs = scales.split_signs()
    source = hadamard_blocks(x.to(torch.float16).float(), 512)
    # The quantized-activation Luke branch casts the original input to FP16,
    # then keeps H512 -> suh -> H128 in FP32 until E4M3 quantization.
    source = hadamard_blocks(
        source * scales.gate_up_suh.float(), 128
    )
    gate = (source.float() @ gate_physical.float().T).to(torch.float16)
    up = (source.float() @ up_physical.float().T).to(torch.float16)
    rows, width = gate.shape
    raw = torch.stack(
        (gate.view(rows, width // 32, 32), up.view(rows, width // 32, 32)),
        dim=2,
    ).reshape(rows, 2 * width)
    scale_raw = torch.stack(
        (
            gate_svh[0].view(width // 32, 32),
            up_svh[0].view(width // 32, 32),
        ),
        dim=1,
    ).reshape(2 * width)
    pre = hadamard_blocks(raw.float(), 128)
    pre = hadamard_blocks(pre * scale_raw.float(), 128)
    pre = pre * pre_signs.float()
    gate_joint, up_joint = pre[:, 0::2], pre[:, 1::2]
    gate_work = gate_joint.clamp(max=10.0)
    up_work = up_joint.clamp(min=-10.0, max=10.0)
    activated = gate_work * torch.sigmoid(gate_work) * up_work
    activated = hadamard_blocks(activated * post_signs.float(), 128)
    down_input = hadamard_blocks(activated * down_suh[0].float(), 128)
    down = (down_input @ down_physical.float().T).to(torch.float16)
    route = hadamard_blocks(down.float(), 128) * scales.down_svh.float()
    return hadamard_blocks(route, 512)


def scale_sandwich_reference(
    x: torch.Tensor,
    gate_physical: torch.Tensor,
    up_physical: torch.Tensor,
    down_physical: torch.Tensor,
    scales: P8ScaleSandwich,
) -> torch.Tensor:
    """Exact ordinary H128 scale sequence, without H512 or coupled signs.

    This reference is a closure tool for this component only.  Physical
    weights must already be encoded in the matching transformed bases.
    """

    gate_svh, up_svh, down_suh = scales.split_intermediate()
    source = had128_scaled(x, suh=scales.gate_up_suh, store_fp16=True)
    gate = (source.float() @ gate_physical.float().T).to(torch.float16)
    up = (source.float() @ up_physical.float().T).to(torch.float16)
    gate = had128_scaled(gate, svh=gate_svh[0], store_fp16=True)
    up = had128_scaled(up, svh=up_svh[0], store_fp16=True)
    gate_work = gate.float().clamp(max=10.0)
    up_work = up.float().clamp(min=-10.0, max=10.0)
    activated = (gate_work * torch.sigmoid(gate_work) * up_work).to(torch.float16)
    down_input = had128_scaled(activated, suh=down_suh[0], store_fp16=True)
    down = (down_input.float() @ down_physical.float().T).to(torch.float16)
    return had128_scaled(down, svh=scales.down_svh, store_fp16=False)


def quantize_e4m3_ue8m0_per32(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference native activation payload, UE8M0 bytes, and reconstruction."""

    if value.ndim != 2 or value.shape[1] % 32:
        raise ValueError("MXFP8 input must be rank 2 with width divisible by 32")
    blocks = value.float().reshape(value.shape[0], value.shape[1] // 32, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0, device=value.device), exponent)
    scale = torch.where(maximum > 0, scale, torch.ones_like(scale))
    payload = (blocks / scale).to(torch.float8_e4m3fn)
    scale_code = (exponent.squeeze(-1).to(torch.int16) + 127).to(torch.uint8)
    reconstruction = payload.float().mul(scale).reshape_as(value.float())
    return payload.view(torch.uint8).contiguous(), scale_code.contiguous(), reconstruction


__all__ = [
    "CAST_ORDER",
    "COMPONENT",
    "COMPOSITION_TARGET",
    "COUPLED_CAST_ORDER",
    "COUPLED_COMPONENT",
    "COUPLED_SCHEMA",
    "P8ScaleSandwich",
    "SCALE_NAMES",
    "SCHEMA",
    "SIGN_DRAW",
    "SIGN_GENERATOR",
    "TRANSFORM_ID",
    "coupled_reference",
    "hadamard_blocks",
    "had128_scaled",
    "quantize_e4m3_ue8m0_per32",
    "scale_sandwich_reference",
    "qsrt_coupled_signs_reference",
    "rank_local_coupled_signs",
    "tensor_sha256",
    "validate_scale_component",
    "validate_coupled_component",
]

# Compatibility for callers using the original exported reference name.
had128_luke = had128_scaled
