"""Byte-preserving replica of owner-striped, paged DS4.1 compressed KV."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.program_cache import program_cache
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._dcp_cute_common import block_pair_barrier
from ._dcp_topk_cute import _copy_16b


class _ReplicaBarrier:
    def __init__(self, world: int, rank: int):
        self.world, self.rank = world, rank

    @cute.jit
    def __call__(self, signals: tuple, stream: cuda.CUstream):
        self.kernel(signals).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, signals: tuple):
        block_pair_barrier(
            signals,
            self_signal=signals[self.rank],
            rank=self.rank,
            world_size=self.world,
            max_blocks=1,
        )


class _ReplicaCopy:
    def __init__(self, world, rank, page_size, stripe, ratio, stage):
        self.world, self.rank = world, rank
        self.page_size, self.stripe, self.ratio = page_size, stripe, ratio
        self.stage = stage

    @cute.jit
    def __call__(
        self,
        cache: cute.Pointer,
        table: cute.Pointer,
        positions: cute.Pointer,
        starts: cute.Pointer,
        out: cute.Pointer,
        staging: tuple,
        requests: Int32,
        max_tokens: Int32,
        table_stride: Int64,
        cache_stride: Int64,
        cache_pages: Int64,
        local_capacity: Int64,
        output_pages_per_request: Int64,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cache,
            table,
            positions,
            starts,
            out,
            staging,
            requests,
            max_tokens,
            table_stride,
            cache_stride,
            cache_pages,
            local_capacity,
            output_pages_per_request,
        ).launch(grid=(blocks, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        cache: cute.Pointer,
        table: cute.Pointer,
        positions: cute.Pointer,
        starts: cute.Pointer,
        out: cute.Pointer,
        staging: tuple,
        requests: Int32,
        max_tokens: Int32,
        table_stride: Int64,
        cache_stride: Int64,
        cache_pages: Int64,
        local_capacity: Int64,
        output_pages_per_request: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = Int32(tid) % Int32(32)
        row = Int64(bid) * Int64(8) + Int64(tid) // Int64(32)
        step = Int64(grid) * Int64(8)
        while row < Int64(requests) * Int64(max_tokens):
            req = row // Int64(max_tokens)
            token = row % Int64(max_tokens)
            end = (
                cute.arch.load(positions + req, Int64)
                + Int64(cute.arch.load(starts + req + Int64(1), Int32))
                - Int64(cute.arch.load(starts + req, Int32))
            ) // Int64(self.ratio)
            owner = (token // Int64(self.stripe)) % Int64(self.world)
            local = token // Int64(self.stripe * self.world) * Int64(
                self.stripe
            ) + token % Int64(self.stripe)
            # Both record/page products must remain Int64 for recycled pool IDs.
            staged_record = req * local_capacity + local
            if token < end:
                if cutlass.const_expr(self.stage):
                    if owner == Int64(self.rank):
                        page = Int64(
                            cute.arch.load(
                                table
                                + req * table_stride
                                + local // Int64(self.page_size),
                                Int32,
                            )
                        )
                        source = page * cache_stride + local % Int64(
                            self.page_size
                        ) * Int64(72)
                        if lane < Int32(18):
                            target = staging[self.rank] + staged_record * Int64(72)
                            offset = Int64(lane) * Int64(4)
                            if page > Int64(0) and page < cache_pages:
                                _copy_16b(cache + source + offset, target + offset)
                            else:
                                for word in cutlass.range_constexpr(4):
                                    cute.arch.store(
                                        target + offset + Int64(word), Uint32(0)
                                    )
                else:
                    address = Int64(staging[0].toint())
                    for peer in cutlass.range_constexpr(1, self.world):
                        if owner == Int64(peer):
                            address = Int64(staging[peer].toint())
                    source = cute.make_ptr(
                        Uint32,
                        address,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ) + staged_record * Int64(72)
                    target_record = (req * output_pages_per_request + Int64(1)) * Int64(
                        self.page_size
                    ) + token
                    if lane < Int32(18):
                        offset = Int64(lane) * Int64(4)
                        _copy_16b(
                            source + offset, out + target_record * Int64(72) + offset
                        )
            row += step


@program_cache
def get_kv_replica_launchers(world, rank, page_size, stripe, ratio):
    """Compile fixed geometry only; every request quantity is a runtime scalar."""
    word = make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)
    integer = make_ptr(Int32, 16, cute.AddressSpace.gmem, assumed_align=4)
    position = make_ptr(Int64, 16, cute.AddressSpace.gmem, assumed_align=8)
    pointers = tuple(word for _ in range(world))
    identity = (world, rank, page_size, stripe, ratio)
    barrier = compile_cute(
        _ReplicaBarrier(world, rank),
        pointers,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.kv_replica.barrier",
            1,
            (world, rank),
            labels=("world", "rank"),
        ),
    )
    copies = {}
    for stage in (True, False):
        copies[stage] = compile_cute(
            _ReplicaCopy(*identity, stage),
            word,
            integer,
            position,
            integer,
            word,
            pointers,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "comm.pcie.kv_replica.copy",
                2,
                (*identity, stage),
                labels=("world", "rank", "page_size", "stripe", "ratio", "stage"),
            ),
        )
    return {"barrier": barrier, "stage": copies[True], "replicate": copies[False]}


def run_kv_replica(
    launchers,
    runtime,
    cache,
    table,
    positions,
    starts,
    out,
    *,
    requests,
    max_tokens,
    output_pages_per_request,
):
    """Release old consumers, stage, publish, then replicate on one ordered stream."""

    def words(address):
        return make_ptr(Uint32, address, cute.AddressSpace.gmem, assumed_align=16)

    def integers(tensor):
        return make_ptr(
            Int32, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )

    staging = tuple(words(ptr) for ptr in runtime.staging_ptrs)
    signals = tuple(words(ptr) for ptr in runtime.signal_ptrs)
    stream = current_cuda_stream()
    blocks = max(1, min(64, (requests * max_tokens + 7) // 8))
    call = (
        words(cache.data_ptr()),
        integers(table),
        make_ptr(Int64, positions.data_ptr(), cute.AddressSpace.gmem, assumed_align=8),
        integers(starts),
        words(out.data_ptr()),
        staging,
        requests,
        max_tokens,
        table.stride(0),
        cache.stride(0) // 4,
        cache.shape[0],
        runtime.local_capacity,
        output_pages_per_request,
        blocks,
        stream,
    )
    # A one-CTA barrier on each stream follows completion of the entire previous
    # consumer grid. Per-block barriers alone are unsafe when live grids change.
    launchers["barrier"](signals, stream)
    launchers["stage"](*call)
    launchers["barrier"](signals, stream)
    launchers["replicate"](*call)
