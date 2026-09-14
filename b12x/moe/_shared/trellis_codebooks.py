"""Trellis codebook registry shared by preparation, planning, and kernels.

A ``trellis_t256`` tile stores 256 tail-biting codes whose 16-bit decode
windows are interpreted by one codebook. The codebook is a model-level
setting; three are defined:

- ``mcg``: multiplicative congruential decode (multiplier ``0xCBAC1FED``)
  with a lop3 mask/or into two added fp16 halves.
- ``sqg_e4m3``: XOR-Cheb-T12 bijection over the retained L16 history with a
  frozen E4M3 reconstruction staircase; defined for K2/K3/K4.
- ``sqg_fp16``: D3L descriptor decode to fp16; defined for K5/K6 tiles.

This module is torch-free. Kernel modules embed the ids as compile-time
constants, so the ids participate in kernel cache keys.
"""

from __future__ import annotations

MCG = "mcg"
SQG_E4M3 = "sqg_e4m3"
SQG_FP16 = "sqg_fp16"

CODEBOOKS: tuple[str, ...] = (MCG, SQG_E4M3, SQG_FP16)

MCG_MULTIPLIER = 0xCBAC1FED
CODEBOOK_SENTINELS: dict[int, str] = {MCG_MULTIPLIER: MCG}

def normalize_codebook(codebook: str | int) -> str:
    """Return the canonical codebook id for ``codebook``.

    Integers are checkpoint sentinels (the MCG multiplier); strings are
    matched case-insensitively against the canonical ids.
    """

    if isinstance(codebook, int):
        normalized = CODEBOOK_SENTINELS.get(int(codebook) & 0xFFFFFFFF)
        if normalized is None:
            raise ValueError(
                "unsupported trellis codebook sentinel "
                f"{int(codebook) & 0xFFFFFFFF:#010x}; expected MCG 0xcbac1fed"
            )
        return normalized
    text = str(codebook).strip().lower()
    if text in CODEBOOKS:
        return text
    raise ValueError(
        f"unsupported trellis codebook {codebook!r}; expected "
        "'mcg', 'sqg_e4m3', or 'sqg_fp16'"
    )


def validate_codebook_bits(codebook: str, bits: int) -> None:
    """Reject (codebook, bitrate) pairs the decoders do not define.

    MCG decodes any supported tile bitrate; the SQG codebooks are defined
    only on their construction ranges.
    """

    if codebook == SQG_E4M3 and bits not in (2, 3, 4):
        raise ValueError("sqg_e4m3 is defined only for K2/K3/K4")
    if codebook == SQG_FP16 and bits not in (5, 6):
        raise ValueError("sqg_fp16 is defined only for K5/K6")
