"""CPU oracle for circular t256 bit extraction and FP16 codebook values."""

import numpy as np
import torch


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
