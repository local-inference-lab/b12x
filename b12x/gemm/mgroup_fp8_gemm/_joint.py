import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32

from b12x._lib import dense_gemm as dense
from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import make_ptr, current_cuda_stream, cuda_stream_from_int_or_current


class _Body(dense.DenseGemmKernel):
    def __init__(self, tile, capacity, sfa_prefetch=False, role_local_scheduler=False, nmajor_mma=False, packed_sf=False):
        super().__init__(sf_vec_size=32, mma_tiler_mn=tile[:2], tile_k=tile[2], cluster_shape_mn=(1, 1),
                         mgroup_labels=True, mgroup_joint=True,
                         mgroup_sfa_prefetch=sfa_prefetch,
                         mgroup_role_local_scheduler=role_local_scheduler,
                         mgroup_nmajor_mma=nmajor_mma,
                         mgroup_packed_sf=packed_sf,
                         mgroup_capacity_tiles=(capacity + 127) // 128,
                         alpha_is_one=True, sched_raster_along_m=False)
        if tile == (128, 64, 128):
            self.atom_shape = (2, 2, 1)
            self.num_mma_warps = 4
            self.tma_load_warp_id = 4
            self.mma_sync_barrier = dense.pipeline.NamedBarrier(barrier_id=1, num_threads=128)
            self.epilog_sync_barrier = dense.pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        self.threads_per_cta = 384

    def _setup_attributes(self):
        self.epi_tile = (64, self.tile_shape_mnk[1])
        super()._setup_attributes()
        if self.mgroup_packed_sf:
            assert self.tile_shape_mnk == (128, 128, 64)
            assert self.mma_tile_shape_mnk == (128, 128, 64)
            assert self.atom_shape == (4, 2, 1) and self.ab_stage == 4
            assert self.num_m_tiles == 2 and self.num_n_tiles == 8

    @staticmethod
    def _compute_stages(*args, **kwargs):
        kwargs['epi_stage_cap'] = 1
        kwargs['ab_stage_cap'] = 4 if args[0][2] == 64 else 3
        return dense.DenseGemmKernel._compute_stages(*args, **kwargs)


class _JointLaunch:
    def __init__(self, n, k, groups, capacity, sms, sfa_prefetch=False, role_local_scheduler=False, nmajor_mma=False, packed_sf=False):
        self.n, self.k, self.groups, self.capacity, self.sms = n, k, groups, capacity, sms
        self.sfa_prefetch = bool(sfa_prefetch)
        self.role_local_scheduler = bool(role_local_scheduler)
        self.nmajor_mma = bool(nmajor_mma)
        self.packed_sf = bool(packed_sf)
        self.full = _Body((128, 128, 64), capacity, self.sfa_prefetch, self.role_local_scheduler, self.nmajor_mma, self.packed_sf)
        self.narrow = _Body((128, 64, 128), capacity)

    def compile_key(self):
        variant = 'mgroup-joint-sfa-prefetch-v1' if self.sfa_prefetch else 'mgroup-joint-v1'
        if self.role_local_scheduler:
            variant = 'mgroup-joint-role-local-v1'
        if self.nmajor_mma:
            variant = 'mgroup-joint-nmajor-mma-v1'
        if self.packed_sf:
            variant = 'mgroup-joint-packed-sf-v1'
        return (variant, 'generic-shared-tma-bidirect-v3', self.n, self.k, self.groups, self.capacity, self.sms)

    @cute.jit
    def __call__(self, a_ptr: cute.Pointer, labels_ptr: cute.Pointer, b_ptr: cute.Pointer,
                 sa_ptr: cute.Pointer, sb_ptr: cute.Pointer, c_ptr: cute.Pointer,
                 alpha_ptr: cute.Pointer, selector_ptr: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        a = cute.make_tensor(a_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, cute.make_ordered_layout((self.n, self.k, self.groups), order=(1, 0, 2)))
        c = cute.make_tensor(c_ptr, cute.make_ordered_layout((rows, self.n, 1), order=(1, 0, 2)))
        labels = cute.make_tensor(labels_ptr, cute.make_layout((rows,)))
        sa = cute.make_tensor(sa_ptr, cute.make_layout((1,)))
        sb = cute.make_tensor(sb_ptr, cute.make_layout((1,)))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout((1,)))
        selector = cute.make_tensor(selector_ptr, cute.make_layout((1,)))
        full, _ = self.full.prepare_body(a, a, alpha, alpha, b, sa, sb, c, alpha,
            alpha, alpha, alpha, self.sms, stream, mgroup_labels_tensor=labels)
        narrow, _ = self.narrow.prepare_body(a, a, alpha, alpha, b, sa, sb, c, alpha,
            alpha, alpha, alpha, self.sms, stream, mgroup_labels_tensor=labels)
        self.smem_bytes = max(self.full.shared_storage.size_in_bytes(), self.narrow.shared_storage.size_in_bytes())
        assert self.smem_bytes <= 101376
        self.kernel(full, narrow, selector).launch(grid=(1, 1, self.sms), block=(384, 1, 1),
            cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(self, full, narrow, selector):
        shared = cutlass.utils.SmemAllocator().allocate(self.smem_bytes, byte_alignment=1024)
        if selector[0] == 0:
            self.full.body(*full, self.full.shared_storage(shared))
        else:
            self.narrow.body(*narrow, self.narrow.shared_storage(shared))


@program_cache
def compile_joint(n, k, groups, capacity, sms, *, sfa_prefetch=False, role_local_scheduler=False, nmajor_mma=False, packed_sf=False):
    launch = _JointLaunch(n, k, groups, capacity, sms, sfa_prefetch, role_local_scheduler, nmajor_mma, packed_sf)
    key = launch.compile_key()
    raise_if_kernel_resolution_frozen('cute.compile', target=launch, cache_key=key)
    types = (cutlass.Float8E4M3FN, cutlass.Int32, cutlass.Float8E4M3FN,
             cutlass.Float8E8M0FNU, cutlass.Float8E8M0FNU, cutlass.BFloat16,
             cutlass.Float32, cutlass.Int32)
    pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if t == cutlass.Int32 else 16) for t in types)
    compiled = dense.b12x_compile(launch, *pointers, Int32(1), current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key('gemm.mgroup_joint', 1, key))

    def run(a, labels, b, sa, sb, c, alpha, selector, stream=None):
        tensors = (a, labels, b, sa, sb, c, alpha, selector)
        ptrs = tuple(make_ptr(t, v.data_ptr(), cute.AddressSpace.gmem,
                             assumed_align=4 if t == cutlass.Int32 else 16)
                     for t, v in zip(types, tensors, strict=True))
        compiled(*ptrs, Int32(a.shape[0]), cuda_stream_from_int_or_current(stream))
        return c
    return attach_programs(run, compiled)
