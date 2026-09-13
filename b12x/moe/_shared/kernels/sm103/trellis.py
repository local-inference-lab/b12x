"""Trellis reconstruction primitive and deferred SM103 projection contract.

The tile decoder consumes the existing t256 SQG E4M3 bitstream and codebook.
Its BF16 output is in the quantizer basis, before scales and rotations. It
does not execute a complete expert or substitute for a fused MoE backend.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.architecture import UnsupportedArchitectureError
from b12x._lib.intrinsics import (
    cvt_e4m3_to_f32_via_f16,
    packed_decode_trellis_sqg_direct_lut_to_e4m3x8,
)
from ..trellis_ring import trellis256_lane_geom_bits


@dataclass(frozen=True)
class TrellisPipeline:
    hidden: int
    intermediate: int
    experts: int
    codebook: str
    coupled_hadamard: bool

    @classmethod
    def from_weight_plan(cls, plan):
        if plan.source_format not in {"b12x_trellis", "btx"}:
            raise ValueError("Trellis pipeline requires a Trellis weight plan")
        return cls(
            plan.hidden_size,
            plan.intermediate_size,
            plan.num_experts,
            plan.trellis_codebook or "mcg",
            plan.coupled_hadamard,
        )

    def reconstruction(self, *, projection: str, bits: int):
        if self.codebook != "sqg_e4m3":
            raise UnsupportedArchitectureError(
                "SM103 tile reconstruction implements SQG E4M3 only"
            )
        if projection not in {"w13", "w2"}:
            raise ValueError("projection must be w13 or w2")
        tiles = self.experts * (self.hidden // 16) * (self.intermediate // 16)
        return ReconstructTrellisTiles(bits, tiles * (2 if projection == "w13" else 1))

    def require_execution(self):
        raise UnsupportedArchitectureError(
            "SM103 Trellis has a t256 reconstruction primitive; expert scale/rotation "
            "staging, mixed-rate dispatch, and tcgen05 projection are not implemented"
        )


class ReconstructTrellisTiles:
    """Decode native [tiles,16*bits] int16 payloads to [tiles,16,16] BF16.

    Output tile axes are (N,K), matching the logical GEMM weight. Capacity
    sizes pointer layouts; live_tiles changes only the launch grid.
    """

    def __init__(self, bits: int, capacity: int):
        if bits not in (2, 3, 4) or capacity <= 0:
            raise ValueError(
                "SQG tile reconstruction requires K2/K3/K4 and positive capacity"
            )
        self.bits, self.capacity = bits, capacity

    @cute.jit
    def __call__(
        self,
        packed: cute.Pointer,
        lut: cute.Pointer,
        out: cute.Pointer,
        live_tiles: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(packed, lut, out).launch(
            grid=(live_tiles, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, packed: cute.Pointer, lut: cute.Pointer, out: cute.Pointer):
        lane, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        source = cute.make_tensor(
            packed, cute.make_layout(self.capacity * 8 * self.bits)
        )
        output = cute.make_tensor(out, cute.make_layout(self.capacity * 256))
        ia, ib, shift, _ = trellis256_lane_geom_bits(Int32(lane), 0, 8, self.bits)
        base = Int64(tile) * Int64(8 * self.bits)
        merged = (cutlass.Uint64(source[base + Int64(ia)]) << 32) | cutlass.Uint64(
            source[base + Int64(ib)]
        )
        lo, hi = packed_decode_trellis_sqg_direct_lut_to_e4m3x8(
            Uint32(merged >> cutlass.Uint64(shift)),
            Uint32(merged >> cutlass.Uint64(shift + 4 * self.bits)),
            lut.toint(),
            self.bits,
            rate_indexed=True,
        )
        for j in cutlass.range_constexpr(8):
            word = lo if cutlass.const_expr(j < 4) else hi
            value = cvt_e4m3_to_f32_via_f16((word >> Uint32(8 * (j % 4))) & Uint32(255))
            row = 2 * (lane // 8) + ((lane >> 2) & 1) + (8 if j >= 4 else 0)
            col = 2 * (lane % 4) + (j % 2) + (8 if j % 4 >= 2 else 0)
            output[Int64(tile) * 256 + Int64(row) * 16 + Int64(col)] = cutlass.BFloat16(
                value
            )
