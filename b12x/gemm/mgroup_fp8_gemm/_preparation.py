"""Prepared grouped FP8 GEMM launchers.

Masked mode compiles the ``mgroup_masked`` variant of ``DenseGemmKernel``
(batched L = groups): a device live-tile prefix omits tiles at/after
the device ``masked_m`` count of the tile's group, so dead rows cost no
mainloop work. Rows ``>= masked_m[g]`` of ``D[g]`` stay contract-undefined
(skipped or straddling tiles leave/compute garbage there); the host never
touches the device-side ``masked_m``.

Contiguous labels mode compiles the ``mgroup_labels`` variant of
``DenseGemmKernel``: each M tile reads its group from the device ``labels``
tensor at the tile's first row; live ``m_total`` is a runtime launch
argument. Whole padding tiles (label -1 on the first row) are skipped like
masked dead tiles; straddling padding rows are computed against a clamped
group and then zero-filled by a small Triton cleanup over the live span.

Both modes own private capacity-sized scale buffers. Masked packing uses the
current tensor row count for group strides within those buffers. Contiguous
packing validates label values and 128-row run starts on the GPU; invalid
labels raise an asynchronous CUDA device error.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan
from ._tuning import (
    MGroupFP8GemmConfig,
    MGroupFP8GemmQuery,
    TUNING,
    validate_query,
)

def _masked_workspace_sizes(query, config):
    return (
        query.num_groups * ((query.m_capacity + 127) // 128) * (query.k // 128) * 512,
        query.num_groups * ((query.n + 127) // 128)
        * ((query.k + 511) // 512 if config.implementation == "masked_compact" else query.k // 128) * 512,
    )


def _mgroup_policy():
    from b12x._lib import dense_gemm as dense

    return dense._DenseGemmPolicy(
        single_work_tile_per_cta=False,
        direct_one_m_tile_scheduler=False,
        use_m1_non_tma=False,
        split_k_slices=1,
        split_k_atomic_bf16=False,
        large_m_unroll=False,
    )


@program_cache(scope="preparation")
def compile_mgroup_fp8(query_payload, config_payload, ordinal, sm_count):
    """Compile grouped GEMM, scale packing, and padding cleanup programs."""
    query = MGroupFP8GemmQuery(**dict(query_payload))
    config = MGroupFP8GemmConfig.from_config(FrozenMapping(config_payload))
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x._lib import dense_gemm as dense
    from b12x._lib.compile_plan import compile_only_launches
    from ._contiguous_packing import normalize_g1, pack_contiguous, workspace_sizes, zero_padding
    from ._packing import pack_grouped_scales_into

    with torch.cuda.device(ordinal):
        if config.implementation == "joint_v1":
            from ._joint import compile_joint
            gemm = compile_joint(query.n, query.k, query.num_groups, query.m_capacity, sm_count, packed_sf=True)
        elif query.mode == "masked":
            gemm = dense._get_compiled_dense_gemm_masked_mgroup(
                query.n, query.k, query.num_groups, _mgroup_policy(),
                (config.tile_m, config.tile_n), config.tile_k, sm_count,
                compact_sfb=config.implementation == "masked_compact",
            )
        else:
            gemm = dense._get_compiled_dense_gemm_mgroup(
                query.n, query.k, query.num_groups, _mgroup_policy(),
                (config.tile_m, config.tile_n), sm_count,
            )
        programs = {"gemm": gemm}
        with FakeTensorMode(), compile_only_launches():
            def empty(shape, dtype=torch.float32):
                return torch.empty(shape, dtype=dtype, device=torch.device("cuda", ordinal))

            g, m, n, k = query.num_groups, query.m_capacity, query.n, query.k
            sfb = empty((g, n, k // 128))
            if query.mode == "masked":
                masked_m = empty((g,), torch.int32)
                sizes = _masked_workspace_sizes(query, config)
                for size, (name, scales, rows, compact) in zip(sizes, (
                    ("pack_a", empty((g, m, k // 128)), m, False),
                    ("pack_b", sfb, n, config.implementation == "masked_compact"),
                ), strict=True):
                    programs[name] = pack_grouped_scales_into(
                        normalize_g1(scales, rows, k // 128, g), empty(size, torch.uint8), rows=rows, k=k,
                        num_groups=g, gran=128, masked_m=masked_m, compact128=compact,
                    )
            else:
                joint = config.implementation == "joint_v1"
                oa, ob = (empty(size, torch.uint8) for size in workspace_sizes(m, n, k, g, compact=joint))
                labels, selector = empty((m,), torch.int32), empty((g + 1 if joint else 1,), torch.int32)
                programs["pack"] = pack_contiguous(
                    empty((m, k // 32)), sfb, oa, ob, labels, selector,
                    rows=m, capacity=m, n=n, k=k, groups=g, compact=joint,
                    use_selector=joint, ctas=sm_count,
                )
                programs["zero"] = zero_padding(empty((m, n), torch.bfloat16), labels, capacity=m)
        return programs


def _check_scale_tensor(name, tensor, shape, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device or tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be an f32 tensor on the prepared device")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} differs from preparation {shape}")


def _check_e4m3_tensor(name, tensor, shape, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device or tensor.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name} must be an e4m3 tensor on the prepared device")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} differs from preparation {shape}")


@dataclass(frozen=True)
class _MGroupFP8ExecutionState:
    """Frozen prepared state: compiled kernel + capacity contract checks."""

    query: MGroupFP8GemmQuery
    config: MGroupFP8GemmConfig
    device: torch.device
    gemm: object
    alpha_one: torch.Tensor
    sfa_workspace: torch.Tensor | None = None
    sfb_workspace: torch.Tensor | None = None
    selector: torch.Tensor | None = None
    selector_ctas: int = 1

    def _check_weights(self, b, sfb):
        q = self.query
        _check_e4m3_tensor("B", b, (q.num_groups, q.n, q.k), self.device)
        _check_scale_tensor("SFB", sfb, (q.num_groups, q.n, q.k // 128), self.device)

    def _check_output(self, d, rows):
        q = self.query
        shape = (q.num_groups, rows, q.n) if q.mode == "masked" else (rows, q.n)
        if not isinstance(d, torch.Tensor):
            raise TypeError("D must be a torch.Tensor")
        if d.device != self.device or d.dtype != torch.bfloat16:
            raise ValueError("D must be a BF16 tensor on the prepared device")
        if tuple(d.shape) != shape:
            raise ValueError(f"D shape {tuple(d.shape)} differs from preparation {shape}")

    def _check_live_capacity(self, live):
        if type(live) is not int or live < 0:
            raise ValueError("live M must be a nonnegative integer")
        if live > self.query.m_capacity:
            raise ValueError(
                f"live M {live} exceeds planned m_capacity {self.query.m_capacity}"
            )

    def run_masked(self, lhs, rhs, d, masked_m, *, expected_m=None, stream=None):
        from b12x._lib.utils import cuda_stream_to_int
        from ._packing import pack_grouped_scales_into

        q = self.query
        if q.mode != "masked":
            raise ValueError("plan is prepared for contiguous mode, not masked")
        a, sfa = lhs
        b, sfb = rhs
        if not isinstance(a, torch.Tensor) or a.ndim != 3:
            raise ValueError("masked A must be (G, m_cap, k)")
        m_cap = a.shape[1]
        self._check_live_capacity(m_cap)
        _check_e4m3_tensor("A", a, (q.num_groups, m_cap, q.k), self.device)
        _check_scale_tensor("SFA", sfa, (q.num_groups, m_cap, q.k // 128), self.device)
        self._check_weights(b, sfb)
        self._check_output(d, m_cap)
        if not isinstance(masked_m, torch.Tensor) or masked_m.dtype != torch.int32:
            raise ValueError("masked_m must be a device int32 (G,) tensor")
        if masked_m.device != self.device or tuple(masked_m.shape) != (q.num_groups,):
            raise ValueError("masked_m must live on the prepared device with shape (G,)")
        if not masked_m.is_contiguous():
            raise ValueError("masked_m must be contiguous")
        if not a.is_contiguous() or not d.is_contiguous():
            raise ValueError("masked A and D must be contiguous")
        if not b.is_contiguous():
            raise ValueError("B must be contiguous")
        if a.device.type != "cuda":
            raise ValueError("masked execution requires CUDA")
        from ._contiguous_packing import normalize_g1
        stream_int = cuda_stream_to_int(stream)
        selected_stream = (torch.cuda.current_stream(self.device) if stream_int is None
                           else torch.cuda.ExternalStream(stream_int, device=self.device))
        with torch.cuda.stream(selected_stream):
            sfa_mma = pack_grouped_scales_into(
                normalize_g1(sfa, m_cap, q.k // 128, q.num_groups), self.sfa_workspace,
                rows=m_cap, k=q.k, num_groups=q.num_groups, gran=128,
                masked_m=masked_m,
            )
            sfb_mma = pack_grouped_scales_into(
                normalize_g1(sfb, q.n, q.k // 128, q.num_groups), self.sfb_workspace,
                rows=q.n, k=q.k, num_groups=q.num_groups, gran=128,
                masked_m=masked_m, compact128=self.config.implementation == "masked_compact",
            )
            self.gemm(a, masked_m, b, sfa_mma, sfb_mma, d, self.alpha_one, stream_int)
        return d

    def run_contiguous(self, lhs, rhs, d, labels, *, stream=None):
        from b12x._lib.utils import cuda_stream_to_int
        from ._contiguous_packing import pack_contiguous, zero_padding

        q = self.query
        if q.mode != "contiguous":
            raise ValueError("plan is prepared for masked mode, not contiguous")
        a, sfa = lhs
        b, sfb = rhs
        if not isinstance(a, torch.Tensor) or a.ndim != 2:
            raise ValueError("contiguous A must be (m_total, k)")
        m_total = a.shape[0]
        self._check_live_capacity(m_total)
        _check_e4m3_tensor("A", a, (m_total, q.k), self.device)
        _check_scale_tensor("SFA", sfa, (m_total, q.k // 32), self.device)
        self._check_weights(b, sfb)
        self._check_output(d, m_total)
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int32:
            raise ValueError("labels must be a device int32 (m_total,) tensor")
        if labels.device != self.device or tuple(labels.shape) != (m_total,):
            raise ValueError("labels must live on the prepared device with shape (m_total,)")
        if not labels.is_contiguous():
            raise ValueError("labels must be contiguous")
        if not a.is_contiguous() or not d.is_contiguous():
            raise ValueError("contiguous A and D must be contiguous")
        if not b.is_contiguous():
            raise ValueError("B must be contiguous")
        if a.device.type != "cuda":
            raise ValueError("contiguous execution requires CUDA")
        stream_int = cuda_stream_to_int(stream)
        selected_stream = (torch.cuda.current_stream(self.device) if stream_int is None
                           else torch.cuda.ExternalStream(stream_int, device=self.device))
        joint = self.config.implementation == "joint_v1"
        if joint and b.data_ptr() % 128:
            raise ValueError("joint B requires 128-byte base alignment")
        with torch.cuda.stream(selected_stream):
            sfa_mma, sfb_mma = pack_contiguous(
                sfa, sfb, self.sfa_workspace, self.sfb_workspace, labels, self.selector,
                rows=m_total, capacity=q.m_capacity, n=q.n, k=q.k,
                groups=q.num_groups, compact=joint, use_selector=joint, ctas=self.selector_ctas,
            )
            if m_total:
                if joint:
                    self.gemm(a, labels, b, sfa_mma, sfb_mma, d, self.alpha_one, self.selector, stream_int)
                else:
                    self.gemm(a, labels, b, sfa_mma, sfb_mma, d, self.alpha_one, stream_int)
                zero_padding(d, labels, capacity=q.m_capacity)
        return d


def plan(query, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(query, MGroupFP8GemmQuery):
        raise TypeError("mgroup_fp8_gemm plan requires MGroupFP8GemmQuery")
    validate_query(query)
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("grouped FP8 invocation semantics belong in MGroupFP8GemmQuery")

    def compile_jobs(config, device):
        return (CompileJob.create(
            "b12x.gemm.mgroup_fp8_gemm._preparation:compile_mgroup_fp8",
            TUNING.encode_query(query), TUNING.encode_config(config),
            device.ordinal, device.identity.sm_count,
        ),)

    def memory(config, device):
        from b12x._lib import dense_gemm as dense
        resident = dense._ALPHA_ONE_CACHE.get(("cuda", device.ordinal))
        persistent = [PersistentMemory(
            ("dense.alpha_one", device.ordinal), 4,
            0 if resident is None else resident.numel() * resident.element_size(),
        )]
        if query.mode == "contiguous":
            from b12x.preparation.types import current_plan, current_prepared_state, _owned_tensor_nbytes
            from ._contiguous_packing import workspace_sizes
            sizes = workspace_sizes(query.m_capacity, query.n, query.k, query.num_groups, compact=config.implementation == "joint_v1")
            existing = current_prepared_state()
            owned = 0 if existing is None else _owned_tensor_nbytes((existing.sfa_workspace, existing.sfb_workspace, existing.selector))
            selector_bytes = 4 * (query.num_groups + 1 if config.implementation == "joint_v1" else 1)
            persistent.append(PersistentMemory(("mgroup.contiguous", current_plan()), sum(sizes) + selector_bytes, owned))
        else:
            from b12x.preparation.types import current_plan, current_prepared_state, _owned_tensor_nbytes
            sizes = _masked_workspace_sizes(query, config)
            existing = current_prepared_state()
            owned = 0 if existing is None else _owned_tensor_nbytes((existing.sfa_workspace, existing.sfb_workspace))
            persistent.append(PersistentMemory(("mgroup.masked", current_plan()), sum(sizes), owned))
        return MemoryRequirements(persistent=tuple(persistent))

    def materialize(selection, device):
        from b12x._lib import dense_gemm as dense
        config = selection.config
        resolved_device = torch.device("cuda", device.ordinal)
        alpha_one = dense._cached_alpha_one(resolved_device)
        programs = compile_mgroup_fp8(
            TUNING.encode_query(query), TUNING.encode_config(config),
            device.ordinal, device.identity.sm_count,
        )
        workspaces = (None, None, None)
        if query.mode == "contiguous":
            from ._contiguous_packing import workspace_sizes
            sizes = workspace_sizes(query.m_capacity, query.n, query.k, query.num_groups, compact=config.implementation == "joint_v1")
            workspaces = (*[torch.empty(size, device=resolved_device, dtype=torch.uint8) for size in sizes],
                          torch.empty(query.num_groups + 1 if config.implementation == "joint_v1" else 1,
                                      device=resolved_device, dtype=torch.int32))
        else:
            workspaces = (*[torch.empty(size, device=resolved_device, dtype=torch.uint8)
                            for size in _masked_workspace_sizes(query, config)], None)
        return _MGroupFP8ExecutionState(
            query, config, resolved_device, programs["gemm"], alpha_one, *workspaces,
            selector_ctas=device.identity.sm_count,
        )

    return Plan(contract=TUNING, query=query, invocation=invocation, override=override,
                shared=False, _compile_jobs=compile_jobs, _memory_requirements=memory,
                _materialize=materialize)


def query_from_call(source, weight, out=None, *, expected_m=None, m_capacity=None, **options):
    """Derive the static query from operand metadata (never reads device scalars)."""
    if options:
        raise ValueError("grouped FP8 declarations reject launch overrides")
    if expected_m is not None and (type(expected_m) is not int or expected_m <= 0):
        raise ValueError("expected_m must be a positive integer or None")
    if m_capacity is not None and (type(m_capacity) is not int or m_capacity <= 0):
        raise ValueError("m_capacity must be a positive integer or None")
    if not isinstance(source, tuple) or not isinstance(weight, tuple):
        raise TypeError("grouped FP8 declarations require (values, scales) operand pairs")
    a, sfa = source
    b, sfb = weight
    for name, tensor in (("A", a), ("SFA", sfa), ("B", b), ("SFB", sfb)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if out is None:
        raise ValueError("grouped FP8 declarations require the caller output buffer")
    if a.ndim not in (2, 3):
        raise ValueError("A must be (G, m_cap, k) masked or (m_total, k) contiguous")
    mode = "masked" if a.ndim == 3 else "contiguous"
    if b.ndim != 3:
        raise ValueError("B must be (G, n, k)")
    num_groups, n, k = b.shape
    if mode == "masked":
        if a.shape[0] != num_groups or a.shape[2] != k:
            raise ValueError("masked A geometry disagrees with B")
        planned_rows = a.shape[1]
        expected_d = (num_groups, planned_rows, n)
        a_sf_gran = 128
        expected_sfa = (num_groups, planned_rows, -(-k // 128))
    else:
        if a.shape[1] != k:
            raise ValueError("contiguous A geometry disagrees with B")
        planned_rows = a.shape[0]
        expected_d = (planned_rows, n)
        a_sf_gran = 32
        expected_sfa = (planned_rows, -(-k // 32))
    if m_capacity is not None:
        if m_capacity < planned_rows:
            raise ValueError("planned m_capacity cannot be below the live operand rows")
        planned_rows = m_capacity
        expected_d = (num_groups, planned_rows, n) if mode == "masked" else (planned_rows, n)
    if tuple(out.shape) != expected_d:
        raise ValueError(f"D shape {tuple(out.shape)} differs from the declared contract {expected_d}")
    if tuple(sfa.shape) != expected_sfa:
        raise ValueError(f"SFA shape {tuple(sfa.shape)} implies the wrong granularity")
    if tuple(sfb.shape) != (num_groups, n, -(-k // 128)):
        raise ValueError("SFB must be gran-128 (G, n, k/128)")
    query = MGroupFP8GemmQuery(
        mode=mode, num_groups=num_groups, n=n, k=k, m_capacity=planned_rows,
        a_sf_gran=a_sf_gran, b_sf_gran=128,
        c_dtype=str(out.dtype).removeprefix("torch."),
        expected_m=expected_m,
    )
    validate_query(query)
    return query


__all__ = ["plan", "query_from_call", "compile_mgroup_fp8"]
