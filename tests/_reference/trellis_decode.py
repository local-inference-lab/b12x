"""CPU oracle for circular t256 bit extraction and FP16 codebook values."""

import torch


from tests._reference.trellis_reference import (
    native_weight as native_weight,
)


def codebook_tensor(codebook: str, device) -> torch.Tensor:
    if codebook == "lut_e4m3":
        from b12x._lib.quant.lut_e4m3 import lut_e4m3_direct_table_cpu

        return lut_e4m3_direct_table_cpu().to(device)
    if codebook == "lut_fp16":
        from b12x._lib.quant.lut_fp16 import lut_fp16_segment_table_cpu

        return lut_fp16_segment_table_cpu().to(device)
    return torch.zeros(16, dtype=torch.uint8, device=device)


CODEBOOK_RATES = (
    [("mcg", b) for b in (2, 3, 4, 5, 6)]
    + [("lut_e4m3", b) for b in (2, 3, 4)]
    + [("lut_fp16", b) for b in (5, 6)]
)
