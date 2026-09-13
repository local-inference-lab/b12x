"""CPU oracle for circular t256 bit extraction and FP16 codebook values."""

import torch


from b12x.policy.generation.providers.trellis_reference import (
    native_weight as native_weight,
)


def codebook_tensor(codebook: str, device) -> torch.Tensor:
    if codebook == "sqg_e4m3":
        from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_direct_lut_cpu

        return sqg_xor_cheb_t12_direct_lut_cpu().to(device)
    if codebook == "sqg_fp16":
        from b12x._lib.quant.sqg_fp16_d3l import sqg_fp16_d3l_descriptors_cpu

        return sqg_fp16_d3l_descriptors_cpu().to(device)
    return torch.zeros(16, dtype=torch.uint8, device=device)


CODEBOOK_RATES = (
    [("mcg", b) for b in (2, 3, 4, 5, 6)]
    + [("sqg_e4m3", b) for b in (2, 3, 4)]
    + [("sqg_fp16", b) for b in (5, 6)]
)
