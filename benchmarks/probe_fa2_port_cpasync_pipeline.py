from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


class ProbeCpAsyncPipeline:
    dtype = cutlass.BFloat16
    num_threads = 128
    num_stages = 1
    rows = 64
    cols = 256

    @cute.jit
    def _make_copy(self):
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=128,
        )
        async_copy_elems = 128 // self.dtype.width
        t_layout = cute.make_ordered_layout(
            (self.num_threads // (self.cols // async_copy_elems), self.cols // async_copy_elems),
            order=(1, 0),
        )
        v_layout = cute.make_layout((1, async_copy_elems))
        return cute.make_tiled_copy_tv(atom, t_layout, v_layout)

    def _storage(self):
        mbar = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        buf = cute.struct.Align[
            cute.struct.MemRange[self.dtype, self.rows * self.cols * self.num_stages],
            128,
        ]

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar": mbar,
            "s": buf,
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def _copy_tile(self, mSrc, sDst, copy, tidx):
        thr = copy.get_slice(tidx)
        tGs = thr.partition_S(mSrc)
        tSs = thr.partition_D(sDst)
        for m in cutlass.range_constexpr(cute.size(tSs.shape[1])):
            for k in cutlass.range_constexpr(cute.size(tSs.shape[2])):
                cute.copy(thr, tGs[None, m, k], tSs[None, m, k])

    @cute.jit
    def __call__(self, mIn: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mIn, mOut).launch(
            grid=(1, 1, 1),
            block=[32, 4, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOut: cute.Tensor):
        lane, warp_q_idx, warp_kv_idx = cute.arch.thread_idx()
        tidx = lane + 32 * (warp_q_idx + 4 * warp_kv_idx)
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._storage()
        storage = smem.allocate(SharedStorage)
        prod = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads)
        cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads)
        pipe = pipeline.PipelineCpAsync.create(
            storage.mbar.data_ptr(),
            self.num_stages,
            prod,
            cons,
        )
        s = storage.s.get_tensor(
            cute.make_layout(
                (self.rows, self.cols, self.num_stages),
                stride=(self.cols, 1, self.rows * self.cols),
            )
        )
        copy = self._make_copy()
        prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)

        pipe.producer_acquire(prod_state)
        self._copy_tile(mIn, s[None, None, prod_state.index], copy, tidx)
        pipe.producer_commit(prod_state)
        pipe.consumer_wait(cons_state, pipe.consumer_try_wait(cons_state))

        linear = tidx
        total = self.rows * self.cols
        while linear < total:
            row = linear // self.cols
            col = linear - row * self.cols
            mOut[row, col] = s[row, col, cons_state.index]
            linear += self.num_threads

        pipe.consumer_release(cons_state)


def _to_cute(x: torch.Tensor, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


def main():
    x = torch.randn((64, 256), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = ProbeCpAsyncPipeline()
    kernel(
        _to_cute(x, cutlass.BFloat16),
        _to_cute(y, cutlass.BFloat16),
        stream,
    )
    torch.cuda.synchronize()
    print("max_abs", (x - y).abs().max().item())


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
