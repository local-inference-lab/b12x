"""Trellis reconstruction and quantizer-basis SM103 projection contracts.

The decoder and inline FP16 projection consume existing t256 weights. Complete
uniform-rate execution is owned by the fused MoE architecture backend.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from .trellis_decode import decode_lane, decoder_contract, tile_coordinates


@dataclass(frozen=True)
class TrellisPipeline:
    hidden: int
    intermediate: int
    experts: int
    codebook: str
    intermediate_hadamard: bool

    @classmethod
    def from_weight_plan(cls, plan):
        if plan.source_format not in {"b12x_trellis", "exl3"}:
            raise ValueError("Trellis pipeline requires a Trellis weight plan")
        return cls(
            plan.hidden_size,
            plan.intermediate_size,
            plan.num_experts,
            plan.trellis_codebook or "mcg",
            plan.intermediate_hadamard,
        )

    def reconstruction(self, *, projection: str, bits: int):
        if projection not in {"w13", "w2"}:
            raise ValueError("projection must be w13 or w2")
        tiles = self.experts * (self.hidden // 16) * (self.intermediate // 16)
        return ReconstructTrellisTiles(
            bits, tiles * (2 if projection == "w13" else 1), codebook=self.codebook
        )

    def projection(self, *, projection: str, bits: int, capacity: int):
        from .trellis_gemm import RoutedTrellisGemm

        if projection in {"gate", "up"}:
            n, k = self.intermediate, self.hidden
        elif projection == "down":
            n, k = self.hidden, self.intermediate
        else:
            raise ValueError("Trellis projection must be gate, up, or down")
        return RoutedTrellisGemm(
            n, k, self.experts, capacity, bits=bits, codebook=self.codebook
        )


class ReconstructTrellisTiles:
    """Decode native [tiles,16*bits] int16 payloads to [tiles,16,16] FP16/BF16.

    Output tile axes are (N,K), matching the logical GEMM weight. Capacity
    sizes pointer layouts; live_tiles changes only the launch grid.
    """

    def __init__(self, bits: int, capacity: int, *, codebook: str = "lut_e4m3"):
        self.codebook = decoder_contract(bits, codebook)
        if capacity <= 0:
            raise ValueError("tile reconstruction requires positive capacity")
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
        decoded = decode_lane(
            source, Int64(tile), Int32(lane), lut, self.bits, self.codebook
        )
        for j in cutlass.range_constexpr(8):
            row, col = tile_coordinates(Int32(lane), j)
            output[Int64(tile) * 256 + Int64(row) * 16 + Int64(col)] = decoded[j].to(
                output.element_type
            )
