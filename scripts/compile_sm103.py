#!/usr/bin/env python3
"""Cross-compile SM103 kernels and record artifact identities without a GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._sm103_source import package_source_sha256, source_identity


def compile_sequence(out):
    """Compile production recurrent launch factories using pointer prototypes.

    Binding descriptors supply static geometry and dtypes only. No device
    allocation or kernel execution occurs, and these objects are not runtime
    qualification evidence.
    """
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from b12x.sequence._shared.delta_prefill import _cute_kernels as prefill
    from b12x.sequence.gdn_decode import _cute_kernels as qwen, _cute_kda as kda
    from b12x.sequence.kda_prefill import _impl as kp
    from b12x.sequence.gdn_prefill import _impl as gp

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec):
        name = case + "_" + compile_spec.kernel_id.rsplit(".", 1)[-1]
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled
        return compiled

    def descriptor(dtype):
        return SimpleNamespace(
            dtype=dtype,
            device=torch.device("cuda:0"),
            stride=lambda: (12 * 128 * 128, 128 * 128, 128, 1),
        )

    for index_type in (torch.int32, torch.int64):
        suffix = str(index_type).removeprefix("torch.")
        for recipe, impl in (("kda", kp), ("gdn", gp)):
            for checkpoint in (False, True):
                case = f"{recipe}_prefill_{suffix}_checkpoint{int(checkpoint)}"
                geometry = (
                    dict(heads=16)
                    if recipe == "kda"
                    else dict(key_heads=4, value_heads=12)
                )
                caps = impl.Caps(
                    device="cuda:0",
                    max_tokens=4096,
                    max_seqs=16,
                    max_state_slots=129,
                    checkpoint_export=checkpoint,
                    null_state_index=0,
                    **geometry,
                )
                plan = impl._materialize_plan(
                    caps,
                    v_split=64,
                    k_split=1,
                    stages=3,
                    window_tiles=128,
                    policy_resolution=None,
                )
                binding = SimpleNamespace(
                    plan=plan,
                    output=descriptor(torch.bfloat16),
                    initial_state_indices=descriptor(index_type),
                    A_log=descriptor(torch.float32),
                    dt_bias=descriptor(torch.float32),
                )
                prefill.clear_caches()
                with (
                    patch.object(prefill, "b12x_compile", capture),
                    patch.object(
                        prefill, "current_cuda_stream", lambda: cuda.CUstream(0)
                    ),
                ):
                    prefill._compile_prologue(binding)
                    prefill._compile_prepare(binding)
                    prefill._compile_recurrence(binding)
        for state_type in (torch.bfloat16, torch.float32):
            case = f"kda_decode_{suffix}_{str(state_type).removeprefix('torch.')}"
            key = (
                0,
                128,
                32,
                129,
                4,
                16,
                32,
                True,
                0,
                True,
                state_type,
                index_type,
                torch.float32,
                torch.float32,
                torch.float32,
            )
            with (
                patch.object(kda, "b12x_compile", capture),
                patch.object(kda, "current_cuda_stream", lambda: cuda.CUstream(0)),
            ):
                kda.compile_kernels(key)
            case = f"qwen_decode_{suffix}_{str(state_type).removeprefix('torch.')}"
            caps = SimpleNamespace(
                max_seqs=32,
                state_index_columns=4,
                key_heads=4,
                value_heads=12,
                key_head_dim=128,
                value_head_dim=128,
                qk_l2norm=True,
                null_state_index=0,
            )
            binding = qwen.Binding.__new__(qwen.Binding)
            fields = dict(
                plan=SimpleNamespace(caps=caps),
                output=descriptor(torch.bfloat16),
                recurrent_state=descriptor(state_type),
                state_indices=descriptor(index_type),
                A_log=descriptor(torch.float32),
                dt_bias=descriptor(torch.float32),
            )
            for field, value in fields.items():
                object.__setattr__(binding, field, value)
            qwen._KERNEL_CACHE.clear()
            with (
                patch.object(qwen, "b12x_compile", capture),
                patch.object(qwen, "current_cuda_stream", lambda: cuda.CUstream(0)),
            ):
                qwen._compile(binding)
    prefill.clear_caches()
    qwen._KERNEL_CACHE.clear()
    kda._CACHE.clear()
    return launches


def compile_roce(out):
    """Trace the existing TP2 peer protocol without allocating host/NIC regions."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x.comm.roce._oneshot_cute import _RoceOneshotLaunch, _dummy
    from b12x.comm.roce._allgather_cute import _RoceAllGatherLaunch

    launches = {}
    for rank in (0, 1):
        for dtype in ("bfloat16", "float16", "float32", "gather"):
            name = f"roce_{dtype}_rank{rank}"
            directory = out / name
            directory.mkdir()
            args = [_dummy(cutlass.Uint32, 16), _dummy(cutlass.Uint32, 16), 1, 16]
            if dtype == "gather":
                launch = _RoceAllGatherLaunch(2, rank, 512, 2, 128, 1)
                args.append(1)
            else:
                launch = _RoceOneshotLaunch(dtype, 2, rank, 512, 2, 128, 1)
            args += [16, 16, 16, 16, 4096, 16, 16, 16, 16, 1, 1, cuda.CUstream(0)]
            compiled = cute.compile(
                launch,
                *args,
                no_jit_engine=True,
                options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
            )
            (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
            launches[name] = compiled
    return launches


def compile_dense_mla(out):
    """Compile dense MLA math, split reduction, and query quantization."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor as tensor
    from b12x.attention.dense_mla._forward import DenseMlaForwardKernel
    from b12x.attention.dense_mla._merge import DenseMlaMergeKernel
    from b12x.attention.dense_mla._layout import make_smem_layout

    launches = {}

    def compile_case(name, kernel, args):
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled

    for fp8 in (False, True):
        for qk_dim, value_dim in ((576, 512), (1088, 1024)):
            for query_tile in (1, 2, 4) if fp8 else (1, 2):
                for window in (None, 128):
                    if window is not None and query_tile != 1:
                        continue
                    name = f"dense_mla_{'fp8' if fp8 else 'bf16'}_d{qk_dim}_q{query_tile}_w{window}"
                    kernel = DenseMlaForwardKernel(
                        layout=make_smem_layout(
                            query_tile=query_tile, fp8=fp8, qk_dim=qk_dim
                        ),
                        page_size=64,
                        num_heads=16,
                        num_splits=4,
                        chunks_per_split=16,
                        query_tile=query_tile,
                        fp8=fp8,
                        qk_dim=qk_dim,
                        value_dim=value_dim,
                        window_size=window,
                    )
                    args = [
                        tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
                        for _ in range(2)
                    ]
                    args += [
                        tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
                        for _ in range(3)
                    ]
                    args += [
                        tensor(
                            cutlass.BFloat16,
                            (cute.sym_int(), 16, value_dim),
                            assumed_align=16,
                        ),
                        tensor(cutlass.Float32, (128, 16), assumed_align=4),
                        tensor(
                            cutlass.BFloat16, (128, 16, 4, value_dim), assumed_align=16
                        ),
                        tensor(cutlass.Float32, (128, 16, 4), assumed_align=4),
                    ]
                    args += [
                        tensor(cutlass.Float32, (1,), assumed_align=4) for _ in range(2)
                    ]
                    args += [
                        cutlass.Float32(1.0),
                        *([cutlass.Int64(1)] * 5),
                        *([cutlass.Int32(1)] * 3),
                        cuda.CUstream(0),
                    ]
                    compile_case(name, kernel, args)
    for value_dim in (512, 1024):
        compile_case(
            f"dense_mla_merge_v{value_dim}",
            DenseMlaMergeKernel(4, value_dim),
            [
                tensor(cutlass.BFloat16, (128, 16, 4, value_dim), assumed_align=16),
                tensor(cutlass.Float32, (128, 16, 4), assumed_align=4),
                tensor(
                    cutlass.BFloat16, (cute.sym_int(), 16, value_dim), assumed_align=16
                ),
                tensor(cutlass.Float32, (128, 16), assumed_align=4),
                cutlass.Int32(1),
                cuda.CUstream(0),
            ],
        )
    from b12x.attention._shared.static_fp8_quant import _StaticFp8QuantKernel

    compile_case(
        "dense_mla_query_quant",
        _StaticFp8QuantKernel(128 * 16 * 1088),
        [
            tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16),
            tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16),
            tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4),
            cutlass.Int32(4),
            cuda.CUstream(0),
        ],
    )
    return launches


def compile_dsa_indexer(out):
    """Compile FP8 scoring, exact selection, and inline MXFP4 dequantization."""
    import cuda.bindings.driver as cuda
    import cutlass as c
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor
    from b12x.attention.dsa_indexer import (
        kernel as paged,
        contiguous_kernel as contiguous,
    )
    from b12x.attention.dsa_indexer.fused_indexer import DSAFusedIndexerKernel
    from b12x.attention.dsa_indexer.tiled_topk import DSATiledTopkKernel
    from b12x.attention.dsa_indexer.persistent_topk import DSAPersistentTopK2048Kernel
    from b12x.attention.dsa_indexer import mxfp4

    launches = {}

    def tensor(dtype, shape):
        return make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=16,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    def emit(name, kernel, args):
        name = "indexer_" + name
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            cuda.CUstream(0),
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled

    rows, pages = cute.sym_int(), cute.sym_int()
    k = tensor(c.Uint8, (pages, 64, 128))
    scales = tensor(c.Float32, (pages, 64))
    table = tensor(c.Int32, (rows, 64))
    lengths = tensor(c.Int32, (rows,))
    scalar = tensor(c.Int32, (1,))
    desc = tensor(c.Int64, (pages,))
    logits = tensor(c.Float32, (rows, 4096))
    flat = tensor(c.Float32, (cute.sym_int(),))
    for heads in (16, 32, 64):
        q = tensor(c.Uint8, (rows, heads, 128))
        weights = tensor(c.Float32, (rows, heads))
        args = (q, weights, k, desc, scalar, scales, table, lengths, scalar)
        emit(f"paged_h{heads}", paged.DSAPagedLogitsKernel(4, heads), (*args, logits))
        emit(
            f"stream_h{heads}",
            paged.DSAPagedStreamLogitsKernel(
                4, heads, k_quant_page_stride=8448, k_scales_row_stride=2112
            ),
            (*args, c.Int32(0), c.Int32(4096), flat),
        )
        for topk in (512, 2048):
            for ctas in (1, 4, 192):
                output_i = tensor(c.Int32, (rows, topk))
                output_v = tensor(c.Float32, (rows, topk))
                kernel = DSAFusedIndexerKernel(
                    num_heads_static=heads,
                    topk=topk,
                    ctas_per_group=ctas,
                    num_sms=148,
                    paged_output=True,
                    k_quant_page_stride=8448,
                    k_scales_row_stride=2112,
                    max_seq_capacity=65536,
                    vectorized_q_load=True,
                    q_row_stride_bytes=heads * 128,
                )
                emit(
                    f"fused_h{heads}_k{topk}_ctas{ctas}",
                    kernel,
                    (
                        q,
                        weights,
                        k,
                        scales,
                        table,
                        lengths,
                        lengths,
                        lengths,
                        output_i,
                        output_v,
                        flat,
                        tensor(c.Int32, (cute.sym_int(),)),
                        tensor(c.Int32, (cute.sym_int(),)),
                    ),
                )
        q32 = tensor(c.Uint32, (rows, heads, 32))
        kflat = tensor(c.Uint8, (cute.sym_int(), 128))
        args = (
            q32,
            weights,
            kflat,
            desc,
            tensor(c.Float32, (cute.sym_int(),)),
            lengths,
            lengths,
            logits,
            flat,
        )
        runtime = (c.Int32(4), c.Int32(4096), c.Int32(0), c.Int32(8), c.Int32(1))
        emit(
            f"contiguous_h{heads}",
            contiguous.DSAContiguousLogitsKernel(),
            (*args, *runtime),
        )
        emit(
            f"prefill_h{heads}",
            contiguous.DSAContiguousLogitsPrefillKernel(tiled_output=True),
            (
                *args,
                tensor(c.Float32, (heads, rows, 64)),
                c.Int32(128),
                c.Int32(4096),
                c.Int32(64),
                c.Int32(0),
                c.Int32(8),
                c.Int32(1),
            ),
        )
        emit(
            f"prefill512_h{heads}",
            contiguous.DSAContiguousLogitsPrefill512Kernel(tiled_output=True),
            (*args, *runtime),
        )
    for topk in (512, 2048):
        output_i = tensor(c.Int32, (cute.sym_int(),))
        for physical in (False, True):
            emit(
                f"persistent_k{topk}_physical{int(physical)}",
                DSAPersistentTopK2048Kernel(paged_output=physical, topk=topk),
                (
                    flat,
                    lengths,
                    table,
                    output_i,
                    scalar,
                    *[c.Int32(v) for v in (4, 4096, 1024, 4, 4, 64)],
                ),
            )
            for first in (False, True):
                emit(
                    f"tiled_k{topk}_physical{int(physical)}_first{int(first)}",
                    DSATiledTopkKernel(
                        topk=topk, is_first=first, output_physical_slots=physical
                    ),
                    (
                        flat,
                        lengths,
                        lengths,
                        flat,
                        output_i,
                        flat,
                        output_i,
                        table,
                        *[
                            c.Int32(v)
                            for v in (
                                64,
                                4,
                                4096,
                                8,
                                0,
                                1,
                                1,
                                topk,
                                0,
                                4096,
                                0,
                                64,
                                topk,
                                0,
                            )
                        ],
                    ),
                )

    def capture(kernel, *args, compile_spec):
        emit(compile_spec.kernel_id.rsplit(".", 1)[-1] + "_" + case, kernel, args[:-1])
        return launches[next(reversed(launches))]

    for heads in (8, 16, 32):
        for candidates in (False, True):
            case = f"h{heads}_candidates{int(candidates)}"
            with (
                patch.object(mxfp4, "b12x_compile", capture),
                patch.object(mxfp4, "current_cuda_stream", lambda: cuda.CUstream(0)),
            ):
                mxfp4._compile("score", (heads, candidates, 64), 0)
    for kind, recipe in (
        ("quantize", (False, 64)),
        ("quantize", (True, 64)),
        ("prepare", (False, False)),
        ("prepare", (True, False)),
        ("prepare", (False, True)),
        ("sort", (512, False)),
        ("sort", (2048, True)),
    ):
        case = f"{kind}_{str(recipe).replace(' ', '')}"
        with (
            patch.object(mxfp4, "b12x_compile", capture),
            patch.object(mxfp4, "current_cuda_stream", lambda: cuda.CUstream(0)),
        ):
            mxfp4._compile(kind, recipe, 0)
    return launches


def compile_compressed_mla(out):
    """Compile planned native-cache decode, extend, metadata, and sink paths."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from b12x.attention.compressed_sparse_mla import _warp
    from b12x.attention.compressed_sparse_mla._policy import SparseMlaConfig
    from b12x.attention.compressed_sparse_mla._scratch import (
        B12XCompressedSparseMLAScratchCaps as Caps,
        plan_compressed_sparse_mla_scratch,
    )

    launches = {}
    for recipe, precision in (
        ("deepseek_v4", "fp8"), ("deepseek_v41", "bf16"), ("deepseek_v41", "fp8"),
    ):
        for mode in ("decode", "extend"):
            for heads, swa_width, index_width in ((20, 65, 65), (12, 65, 0), (8, 0, 65)):
                caps = Caps(
                    device="cpu", num_q_heads=heads, max_q_rows=19,
                    max_width=swa_width + index_width, swa_width=swa_width,
                    indexed_width=index_width, cache_format=recipe, mode=mode,
                    max_chunks_per_row=4, swa_page_size=64, indexed_page_size=32,
                )
                plan = plan_compressed_sparse_mla_scratch(caps, execution_config=SparseMlaConfig(
                    max_chunks_per_row=4, v41_compute_mode=precision, backend="warp",
                ))
                storage = torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8)
                q = torch.empty(3, heads, 512, dtype=torch.bfloat16)
                selected = torch.empty(3, min(swa_width, 13), dtype=torch.int32)
                index_selected = torch.empty(3, min(index_width, 13), dtype=torch.int32) if index_width else None
                lengths = torch.empty(3, dtype=torch.int32)
                binding = plan.bind(
                    scratch=storage, q=q, swa_indices=selected, swa_lengths=lengths,
                    indexed_indices=index_selected, indexed_lengths=lengths if index_width else None,
                )
                swa = torch.empty(2, 64 * (584 if recipe == "deepseek_v4" else 528), dtype=torch.uint8)
                indexed = torch.empty(4, 32 * (584 if recipe == "deepseek_v4" else 288), dtype=torch.uint8) if index_width else None
                specs = _warp.launch_specs(
                    plan, binding.scratch, q, swa, selected, lengths,
                    indexed, index_selected, lengths if index_width else None,
                    None, None, torch.empty_like(q), 512**-0.5,
                )
                for entry, kernel, args in specs:
                    name = f"compressed_mla_{recipe}_{precision}_{mode}_h{heads}_swa{swa_width}_idx{index_width}_{entry}"
                    directory = out / name
                    directory.mkdir()
                    compiled = cute.compile(
                        kernel, *args, cuda.CUstream(0), no_jit_engine=True,
                        options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
                    )
                    (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
                    launches[name] = compiled
    from b12x.attention._shared.mla import kv_cache

    for page_size in (3, 32, 64):
        for kind in ("swa", "indexed"):
            for slot_type in (torch.int32, torch.int64):
                kv = torch.empty(5, 512, dtype=torch.bfloat16)
                cache = torch.empty(7, page_size * (528 if kind == "swa" else 288), dtype=torch.uint8)
                slots = torch.empty(5, dtype=slot_type)
                with patch.object(kv_cache, "current_cuda_stream", lambda: cuda.CUstream(0)):
                    kernel, args, _ = kv_cache._compressed_cache_writer_launch(kv, cache, slots, page_size, kind)
                name = f"compressed_mla_writer_{kind}_page{page_size}_{str(slot_type).removeprefix('torch.')}"
                directory = out / name
                directory.mkdir()
                compiled = cute.compile(
                    kernel, *args, no_jit_engine=True,
                    options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
                )
                (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
                launches[name] = compiled
    return launches


def compile_sparse_mla(out):
    """Compile GLM cache recipes with ordinary FP8 or BF16 warp MMA."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor as tensor
    from b12x.attention._shared.mla.kernel import UnifiedDecodeKernel
    from b12x.attention._shared.mla.prefill_mg import UnifiedPrefillMGKernel
    from b12x.attention._shared.mla.smem import make_smem_layout
    from b12x.attention._shared.mla.smem_mg import make_smem_layout_mg
    from b12x.attention._shared.mla.traits import (
        ComputeMode,
        ModelType,
        make_unified_traits,
    )

    launches = {}
    for model in (ModelType.GLM_NSA, ModelType.GLM_NEXT):
        for scale_format in (1, 2):
            for per_token_scale in (False, True) if scale_format == 2 else (False,):
                if (
                    model == ModelType.GLM_NEXT
                    and scale_format == 2
                    and not per_token_scale
                ):
                    continue
                for heads in (8, 16, 32):
                    traits = make_unified_traits(
                        model,
                        ComputeMode.FP8,
                        scale_format,
                        fp8_rope=(model == ModelType.GLM_NSA and per_token_scale),
                        latent_scale_per_token=per_token_scale,
                    )
                    width, splits = 256, 4
                    dim = traits.d_nope + traits.d_rope
                    rows = cute.sym_int()
                    q = tensor(cutlass.BFloat16, (rows, heads, dim), assumed_align=16)
                    kv = tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
                    indices = tensor(cutlass.Int32, (rows, width), assumed_align=4)
                    lengths = tensor(cutlass.Int32, (rows,), assumed_align=4)
                    for mode in ("decode", "prefill"):
                        if mode == "decode":
                            kernel = UnifiedDecodeKernel(
                                traits,
                                make_smem_layout(traits),
                                64,
                                1,
                                h_blocks=max(1, heads // 16),
                                num_splits=splits,
                                num_heads=heads,
                                q_head_dim=dim,
                                topk=width,
                                extra_topk=0,
                                q_stride=(heads * dim, dim, 1),
                                swa_indices_stride0=width,
                                extra_indices_stride0=width,
                                mid_out_stride=(
                                    heads * splits * 512,
                                    splits * 512,
                                    512,
                                    1,
                                ),
                                mid_lse_stride=(heads * splits, splits, 1),
                                valid_hpb=min(16, heads),
                                per_token_len=True,
                                vector_q=True,
                                block_scaled_mma=False,
                            )
                            entry = kernel.call_pertok
                            args = [
                                q,
                                kv,
                                indices,
                                tensor(
                                    cutlass.BFloat16,
                                    (rows, heads, splits, 512),
                                    assumed_align=16,
                                ),
                                tensor(
                                    cutlass.Float32,
                                    (rows, heads, splits),
                                    assumed_align=4,
                                ),
                                cutlass.Float32(0.1),
                                cutlass.Float32(1.0),
                                lengths,
                                cutlass.Int64(64 * traits.kv_gmem_stride),
                                cutlass.Int32(1),
                                cuda.CUstream(0),
                            ]
                        else:
                            groups = 2 if heads == 32 else 1
                            kernel = UnifiedPrefillMGKernel(
                                traits,
                                make_smem_layout_mg(traits, groups),
                                64,
                                width // 64,
                                replicate_h=1,
                                num_heads=heads,
                                q_stride=(heads * dim, dim, 1),
                                indices_stride0=width,
                                output_stride=(heads * 512, 512, 1),
                                out_lse_stride=(heads, 1),
                                has_sink=False,
                                topk=width,
                                valid_hpb=min(16, heads),
                                block_scaled_mma=False,
                            )
                            entry = kernel
                            args = [
                                q,
                                kv,
                                indices,
                                lengths,
                                tensor(cutlass.Float32, (heads,), assumed_align=4),
                                tensor(
                                    cutlass.BFloat16,
                                    (rows, heads, 512),
                                    assumed_align=16,
                                ),
                                tensor(cutlass.Float32, (rows, heads), assumed_align=4),
                                cutlass.Float32(0.1),
                                cutlass.Float32(1.0),
                                cutlass.Int64(64 * traits.kv_gmem_stride),
                                cutlass.Int32(1),
                                cuda.CUstream(0),
                            ]
                        name = f"sparse_mla_model{model}_sf{scale_format}_pts{int(per_token_scale)}_h{heads}_{mode}"
                        directory = out / name
                        directory.mkdir()
                        compiled = cute.compile(
                            entry,
                            *args,
                            no_jit_engine=True,
                            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
                        )
                        (directory / (name + ".mlir")).write_text(
                            str(compiled.ir_module)
                        )
                        launches[name] = compiled
    from b12x.attention.sparse_mla._sm103 import SplitLse, ConvertLse
    from b12x.attention._shared.mla.merge import (
        SparseMLASplitDecodeMergeKernel,
        SparseMLASplitDecodeSinkMergeKernel,
    )
    from b12x.attention._shared.mla.kv_cache import (
        ConcatAndCacheGlmNextMlaKernel,
        ConcatAndCacheNvfp4MlaFp8RopeKernel,
    )

    def emit(name, kernel, args):
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            cuda.CUstream(0),
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled

    rows = cute.sym_int()
    partial = tensor(cutlass.BFloat16, (rows, 16, 4, 512), assumed_align=16)
    partial_lse = tensor(cutlass.Float32, (rows, 16, 4), assumed_align=4)
    output = tensor(cutlass.BFloat16, (rows, 16, 512), assumed_align=16)
    lse = tensor(cutlass.Float32, (rows, 16), assumed_align=4)
    control = tensor(cutlass.Int32, (1,), assumed_align=4)
    sink = tensor(cutlass.Float32, (16,), assumed_align=4)
    emit(
        "sparse_mla_merge",
        SparseMLASplitDecodeMergeKernel(4),
        [partial, partial_lse, control, output],
    )
    emit(
        "sparse_mla_sink_merge",
        SparseMLASplitDecodeSinkMergeKernel(4),
        [partial, partial_lse, control, sink, output],
    )
    for natural in (False, True):
        emit(
            f"sparse_mla_lse_natural{int(natural)}",
            SplitLse(4, natural),
            [partial_lse, lse],
        )
    emit("sparse_mla_convert_lse", ConvertLse(), [lse])
    for index_type in (cutlass.Int32, cutlass.Int64):
        slots = tensor(index_type, (rows,), assumed_align=8)
        values = tensor(cutlass.BFloat16, (rows, 512), assumed_align=16)
        cache = tensor(cutlass.Uint8, (cute.sym_int(), 64, 528), assumed_align=16)
        emit(
            f"sparse_mla_fp8_writer_{index_type.__name__}",
            ConcatAndCacheGlmNextMlaKernel(64),
            [values, cache, slots, *([cutlass.Int64(1)] * 4), cutlass.Int32(1)],
        )
        for rope in (False, True):
            for dtype in (cutlass.BFloat16, cutlass.Float16):
                values = tensor(dtype, (rows, 512), assumed_align=16)
                rope_values = tensor(dtype, (rows, 64), assumed_align=16)
                cache = tensor(
                    cutlass.Uint8,
                    (cute.sym_int(), 64, 368 if rope else 304),
                    assumed_align=16,
                )
                emit(
                    f"sparse_mla_nvfp4_writer_rope{int(rope)}_{dtype.__name__}_{index_type.__name__}",
                    ConcatAndCacheNvfp4MlaFp8RopeKernel(
                        64, dtype == cutlass.BFloat16, True, rope
                    ),
                    [
                        values,
                        rope_values,
                        cache,
                        slots,
                        *([cutlass.Int64(1)] * 5),
                        cutlass.Int32(1),
                    ],
                )
    return launches


def compile_trellis(out):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis import ReconstructTrellisTiles
    from b12x.moe._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm
    from b12x.moe._shared.kernels.sm103.trellis_mixed_gemm import RoutedMixedTrellisGemm

    launches = {}

    def compile_case(name, kernel, args):
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel, *args, no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled

    # One V4.1 FC2 family in the checkpoint's K16/N16 tile ordering.
    capacity = 384 * (5120 // 16) * (2304 // 16)
    for codebook, rates in (("mcg", (2, 3, 4, 5, 6)), ("sqg_e4m3", (2, 3, 4)), ("sqg_fp16", (5, 6))):
        for bits in rates:
            for dtype in (cutlass.Float16, cutlass.BFloat16):
                name = f"trellis_{codebook}_k{bits}_{dtype.__name__}"
                if codebook == "sqg_e4m3" and dtype == cutlass.BFloat16:
                    name = f"trellis_k{bits}"
                args = [pointer(t) for t in (cutlass.Uint32, cutlass.Uint8, dtype)]
                args += [cutlass.Int32(1), cuda.CUstream(0)]
                compile_case(name, ReconstructTrellisTiles(bits, capacity, codebook=codebook), args)
            geometries = [("tail", 144, 80)]
            if bits == 3 and codebook in {"mcg", "sqg_e4m3"}:
                geometries += [("gate", 2304, 5120), ("down", 5120, 2304)]
            for stage, n, k in geometries:
                for id_dtype in (cutlass.Int32, cutlass.Int64):
                    name = f"trellis_projection_{codebook}_k{bits}_{stage}_{id_dtype.__name__}"
                    args = [pointer(t) for t in (cutlass.Float16, cutlass.Uint32, cutlass.Uint8, id_dtype, cutlass.Float16)]
                    args += [cutlass.Int32(1), cutlass.Int64(k), cutlass.Int64(n), cuda.CUstream(0)]
                    compile_case(name, RoutedTrellisGemm(n, k, 384, 128, bits=bits, codebook=codebook), args)
    from b12x.moe.fused_moe._sm103_trellis import compile_launches as compile_moe
    import torch
    for label, codebook, bits, coupled, activation, dtype in (
        ("v41", "sqg_e4m3", 3, True, "situ", torch.float16),
        ("mcg", "mcg", 3, False, "silu", torch.bfloat16),
        ("sqg", "sqg_e4m3", 4, False, "situ", torch.float16),
        ("sqg_fp16", "sqg_fp16", 5, False, "silu", torch.bfloat16),
    ):
        caps = SimpleNamespace(
            k=5120, n=2304, weight_E=384, max_tokens=128, num_topk=8,
            route_num_experts=768, dtype=dtype, activation=activation,
            weight_plan=SimpleNamespace(coupled_hadamard=coupled, source_format="b12x_trellis" if codebook == "sqg_e4m3" else "btx",
                                        trellis_bits=bits, trellis_codebook=codebook),
        )
        compiled = compile_moe(caps, offline=True, artifact_dir=out, artifact_prefix="trellis_moe_" + label + "_")
        launches.update({"trellis_moe_" + label + "_" + key: fn for key, fn in compiled.items()})
    for local_bits, experts, dtype, activation, coupled in (
        (8, 5, torch.float16, "situ", False),
        (24, 384, torch.bfloat16, "silu", False),
        (8, 5, torch.float16, "situ", True),
        (24, 384, torch.bfloat16, "situ", True),
    ):
        label = f"trellis_mixed_d{local_bits}_" + ("coupled_" if coupled else "")
        caps = SimpleNamespace(
            k=5120, n=2304, weight_E=experts, max_tokens=128, num_topk=min(experts, 6),
            route_num_experts=2 * experts, dtype=dtype, activation=activation,
            weight_plan=SimpleNamespace(coupled_hadamard=coupled, source_format="b12x_trellis",
                                        trellis_bits=3, trellis_codebook="mcg"),
        )
        compiled = compile_moe(caps, offline=True, artifact_dir=out, artifact_prefix=label)
        launches.update({label + key: fn for key, fn in compiled.items()})
        if coupled:
            continue
        for id_dtype in (cutlass.Int32, cutlass.Int64):
            args = [pointer(t) for t in (cutlass.Float16, cutlass.Uint32, cutlass.Uint8, id_dtype, cutlass.Int32, cutlass.Float16)]
            args += [cutlass.Int32(0), cutlass.Int64(3 * experts), cutlass.Int64(1),
                     (cutlass.Int64(0),) * 3, (cutlass.Int32(0),) * 3,
                     cutlass.Int32(1), cutlass.Int64(80), cutlass.Int64(144), cuda.CUstream(0)]
            compile_case(label + "tail_" + id_dtype.__name__,
                         RoutedMixedTrellisGemm(144, 80, experts, 128, descriptor_local_bits=local_bits), args)
    return launches



def compile_mhc(out):
    """Compile production mHC launch factories with nonexecuting tensor metadata.

    CPU DLPack tensors preserve each factory's exact dynamic layout annotation.
    Fake CUDA tensors exercise shape/dtype validation without a CUDA context.
    """
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.base_dsl.runtime import cuda as cuda_helpers
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily
    from b12x.norm.mhc import _kernels as kernels
    from b12x.norm.mhc import _pre_prefill as prepare
    from b12x.norm.mhc._policy import MHC_POLICY, MhcQuery, native_config_for
    from b12x.policy import DeviceIdentity

    launches = {}
    case = ""
    convert = kernels._to_kernel_tensor

    def target_attribute(attribute, device_id=0):
        # CuTe derives the preferred SMEM carveout from min_blocks_per_mp.
        # CUDA Programming Guide table 32 specifies 228 KiB for CC 10.3.
        if (
            attribute
            != cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        ):
            raise RuntimeError(f"unreviewed offline CUDA attribute {attribute}")
        return 228 * 1024

    def prototype(value, dtype, **kwargs):
        with unset_fake_temporarily():
            cpu = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype)
            return convert(cpu, dtype, **kwargs)

    def capture(kernel, *, compile_spec, compile_args, runtime_args):
        directory = out / case
        directory.mkdir()
        with unset_fake_temporarily():
            compiled = cute.compile(
                kernel,
                *compile_args,
                no_jit_engine=True,
                options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
            )
        (directory / (case + ".mlir")).write_text(str(compiled.ir_module))
        launches[case] = compiled

    device = DeviceIdentity(
        vendor="nvidia", product_name="B300", compute_capability=(10, 3), sm_count=148
    )
    with (
        FakeTensorMode(),
        patch.object(cuda_helpers, "get_device_attribute", target_attribute),
        patch.object(kernels, "_to_kernel_tensor", prototype),
        patch.object(prepare, "_to_kernel_tensor", prototype),
        patch.object(kernels, "b12x_launch", capture),
        patch.object(prepare, "launch", capture),
        patch.object(kernels, "current_cuda_stream", lambda: cuda.CUstream(0)),
        patch.object(prepare, "current_cuda_stream", lambda: cuda.CUstream(0)),
        patch.object(torch.cuda, "is_available", lambda: False),
    ):
        for hidden in (4096, 5120, 7168):

            def tensor(*shape, dtype=torch.float32):
                return torch.empty(shape, dtype=dtype, device="cuda:0")

            native = native_config_for(8, hidden, (10, 3))
            residual = tensor(3, 4, hidden, dtype=torch.bfloat16)
            output = tensor(3, 4, hidden, dtype=torch.bfloat16)
            x = tensor(3, hidden, dtype=torch.bfloat16)
            y = tensor(3, hidden, dtype=torch.bfloat16)
            fn = tensor(24, 4 * hidden)
            partials = tensor(3, hidden // 64, 25)
            post, comb = tensor(3, 4), tensor(3, 4, 4)
            scale, bias = tensor(3), tensor(24)
            incoming, predicted = tensor(3, 4), tensor(3, 4)
            for phase in ("broadcast", "pre", "post_pre"):
                for gram, lagged in ((False, False), (True, False), (False, True)):
                    case = f"mhc_h{hidden}_{phase}_gram{int(gram)}_lagged{int(lagged)}"
                    kwargs = dict(
                        fn=fn,
                        partials=partials,
                        out=output,
                        compute_gram=gram,
                        pre_mix=incoming if lagged else None,
                        y=y if lagged else None,
                        planned_tokens=8,
                        native_config=native,
                    )
                    if phase == "post_pre":
                        kernels._run_mhc_post_pre_partial_launch(
                            x=x,
                            residual=residual,
                            prev_post=post,
                            prev_comb=comb,
                            **kwargs,
                        )
                    else:
                        if phase == "broadcast":
                            kwargs["fn"] = tensor(24, hidden)
                        kernels._run_mhc_pre_partial_launch(
                            residual=x if phase == "broadcast" else residual, **kwargs
                        )
            case = f"mhc_h{hidden}_post"
            kernels._run_mhc_post_launch(
                x=x, residual=residual, prev_post=post, prev_comb=comb, out=output
            )
            for gram in (False, True):
                for block_m in (0, 2):
                    case = f"mhc_h{hidden}_compact_b{block_m}_gram{int(gram)}"
                    entry = (
                        kernels._run_mhc_post_pre_prefill_block_m_partial_launch
                        if block_m
                        else kernels._run_mhc_post_pre_prefill_partial_launch
                    )
                    kwargs = (
                        dict(block_m=block_m, tile_n=12 if hidden == 7168 else 24)
                        if block_m
                        else {}
                    )
                    entry(
                        x=x,
                        residual=residual,
                        prev_post=post,
                        prev_comb=comb,
                        fn=fn,
                        partials=partials,
                        out=output,
                        compute_gram=gram,
                        **kwargs,
                    )
            case = f"mhc_h{hidden}_prefill_gram"
            kernels._run_mhc_post_pre_prefill_gram_launch(
                x=x,
                residual=residual,
                prev_post=post,
                prev_comb=comb,
                partials=partials,
                out=output,
            )
            for tma in (False, True):
                case = f"mhc_h{hidden}_bf16_tma{int(tma)}"
                kernels._run_mhc_prefill_bf16_project_launch(
                    out=output,
                    fn_bf16=tensor(24, 4 * hidden, dtype=torch.bfloat16),
                    partials=partials,
                    use_tma=tma,
                )
            for splits, tile_n in ((4, 6), (8, 6)):
                if hidden // splits % 256:
                    continue
                for gram in (False, True):
                    case = f"mhc_h{hidden}_split{splits}_gram{int(gram)}"
                    kernels._run_mhc_post_pre_partial_launch(
                        x=x,
                        residual=residual,
                        prev_post=post,
                        prev_comb=comb,
                        fn=fn,
                        out=output,
                        partials=partials,
                        planned_tokens=8,
                        compute_gram=gram,
                        decode_source_splits=splits,
                        decode_tile_n=tile_n,
                    )
            for capacity in (
                (384, 2304, 3072, 3584, 8192) if hidden == 4096 else (384, 4096)
            ):
                config = MHC_POLICY.heuristic(
                    MhcQuery(
                        dtype="bfloat16",
                        max_tokens=capacity,
                        hidden_size=hidden,
                        split_k=hidden // 64,
                    ),
                    device,
                )
                for split in (False, True):
                    case = f"mhc_h{hidden}_tf32_capacity{capacity}_split{int(split)}"
                    kernels._run_mhc_prefill_tf32_project_launch(
                        out=output,
                        fn=fn,
                        partials=partials,
                        tile_m=config.projection_tile_m,
                        tile_n=config.projection_tile_n,
                        tile_k=config.projection_tile_k,
                        num_stages=config.projection_num_stages,
                        num_m_warps=config.projection_num_m_warps,
                        num_n_warps=config.projection_num_n_warps,
                        k_splits=config.projection_k_splits,
                        split_fp32_fn=split,
                    )
            for compact, lagged, ready in (
                (False, False, False),
                (False, True, False),
                (False, True, True),
                (True, False, False),
                (True, True, False),
            ):
                for norm in (False, True):
                    for weight_type in (
                        (torch.bfloat16, torch.float32) if norm else (torch.bfloat16,)
                    ):
                        case = f"mhc_h{hidden}_finalize_c{int(compact)}_l{int(lagged)}_r{int(ready)}_norm{int(norm)}_{weight_type}"
                        kernels._run_mhc_finalize_gram_launch(
                            residual=output,
                            partials=partials,
                            scale=scale,
                            bias=bias,
                            y=y,
                            post=post,
                            comb=comb,
                            rms_eps=1e-20 if lagged else 1e-6,
                            hc_eps=1e-6,
                            sinkhorn_iters=20,
                            norm_weight=tensor(hidden, dtype=weight_type),
                            norm_eps=1e-20 if lagged else 1e-6,
                            fuse_norm=norm,
                            compact_partials=compact,
                            compact_projection_splits=1,
                            pre_mix=incoming if lagged else None,
                            pre_out=predicted if lagged else None,
                            lagged_prepared=ready,
                            planned_tokens=384 if compact else 8,
                            native_config=native,
                        )
            # Custom-op bodies are invoked directly so fake dispatch cannot skip compilation.
            case = f"mhc_h{hidden}_lagged_prepare"
            prepare.prepare_lagged_prefill._init_fn(residual, output, partials)
            for weighted in (False, True):
                case = f"mhc_h{hidden}_collapse_weighted{int(weighted)}"
                kernels._mhc_collapse_op._init_fn(
                    residual, incoming if weighted else None, y
                )
    return launches


def compile_bf16_projection(out):
    """Compile SIMT and warp-MMA unquantized projection entry points."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor as tensor
    from b12x.gemm.bf16_gemv._kernel import ProjectionKernel
    from b12x.gemm.bf16_gemv._prefill import Bf16PrefillKernel
    from b12x.moe._shared.kernels.sm103.launch import pointer

    launches = {}

    def compile_case(name, kernel, args):
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled

    for x_type in (cutlass.BFloat16, cutlass.Float32):
        for w_type in (cutlass.BFloat16, cutlass.Float32):
            for out_type in (cutlass.BFloat16, cutlass.Float32):
                name = f"projection_{x_type.__name__}_{w_type.__name__}_{out_type.__name__}"
                kernel = ProjectionKernel(
                    512, 4096, x_type == w_type == cutlass.BFloat16, True
                )
                args = [pointer(t) for t in (x_type, w_type, cutlass.Float32, out_type)]
                args += [
                    cutlass.Int32(1),
                    *([cutlass.Int64(1)] * 5),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cuda.CUstream(0),
                ]
                compile_case(name, kernel, args)
    for out_type in (cutlass.BFloat16, cutlass.Float32):
        compile_case(
            f"projection_prefill_{out_type.__name__}",
            Bf16PrefillKernel(512, 5120),
            [
                tensor(cutlass.BFloat16, (cute.sym_int(), 5120), assumed_align=16),
                tensor(cutlass.BFloat16, (512, 5120), assumed_align=16),
                tensor(out_type, (cute.sym_int(), 512), assumed_align=16),
                cutlass.Int32(1),
                cuda.CUstream(0),
            ],
        )
    return launches


def compile_block_fp8_linear(out):
    """Compile production activation quantizers used by planned block-FP8 GEMM."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from b12x._lib.quant import mxfp8_rows as quant

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec, **kwargs):
        directory = out / case
        directory.mkdir()
        compiled = cute.compile(
            kernel, *args, no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (case + ".mlir")).write_text(str(compiled.ir_module))
        launches[case] = compiled
        return compiled

    quant._get_compiled_mxfp8_rows_quant.cache_clear()
    with (patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
          patch.object(quant, "current_cuda_stream", lambda: cuda.CUstream(0)),
          patch.object(quant, "b12x_compile", capture)):
        for dtype in (torch.bfloat16, torch.float16):
            for k in (128, 256, 6144, 32768):
                for floor in (0.0, 1e-4):
                    case = f"mxfp8_rows_{dtype}_k{k}_floor{floor}"
                    quant._get_compiled_mxfp8_rows_quant(
                        k, dtype, 8, 128, "linear", floor, 0, "sm_103a",
                    )
            for subgroup, threads, order in ((0, 256, "linear"), (4, 256, "linear"),
                                              (8, 256, "trellis_native_mma")):
                for floor in (0.0, 1e-4):
                    case = f"mxfp8_rows_{dtype}_lanes{subgroup}_{order}_floor{floor}"
                    quant._get_compiled_mxfp8_rows_quant(
                        8192, dtype, subgroup, threads, order, floor, 0, "sm_103a",
                    )
    return launches


def compile_wo_projection(out):
    """Compile WO quantization, inverse RoPE, and both native projection stages."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from b12x.gemm.wo_projection import _quant_cute as quant
    from b12x.gemm.blockscaled import _sm103 as gemm

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec, **kwargs):
        directory = out / case
        directory.mkdir()
        compiled = cute.compile(
            kernel, *args, no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (case + ".mlir")).write_text(str(compiled.ir_module))
        launches[case] = compiled
        return compiled

    quant._get_compiled_wo_quant.cache_clear()
    gemm.compile_kernel.cache_clear()
    with (patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
          patch.object(quant, "current_cuda_stream", lambda: cuda.CUstream(0)),
          patch.object(quant, "b12x_compile", capture),
          patch.object(gemm, "b12x_compile", capture)):
        for groups, width, rank, hidden in ((1, 128, 128, 256), (2, 512, 256, 2560),
                                          (3, 512, 512, 2560), (4, 4096, 1024, 4096)):
            shape = f"g{groups}_w{width}_r{rank}"
            for dtype in (torch.bfloat16, torch.float16):
                for mode in ("grouped", "group_major"):
                    span = width if mode == "grouped" else rank
                    case = f"wo_{shape}_{dtype}_{mode}"
                    quant._get_compiled_wo_quant(mode, groups * span, span, dtype,
                                                False, 0, 0, 0, torch.int64,
                                                torch.bfloat16, 0, "sm_103a")
                head = 128 if width == 128 else 512
                rope = 32 if head == 128 else 64
                for positions_dtype in (torch.int32, torch.int64):
                    for cache_dtype in (torch.bfloat16, torch.float32):
                        case = f"wo_{shape}_{dtype}_rope_{positions_dtype}_{cache_dtype}"
                        quant._get_compiled_wo_quant(
                            "grouped", groups * width, width, dtype, True,
                            head, head - rope, rope, positions_dtype, cache_dtype, 0, "sm_103a",
                        )
            for stage, n, k, batch in (("a", rank, width, groups), ("b", hidden, groups * rank, 1)):
                case = f"wo_{shape}_{stage}"
                gemm.compile_kernel(n, k, batch, "mxfp8", "bfloat16", 0, True)
    return launches


def compile_activation_packing(out):
    """Compile the supporting BF16/FP16 quantizer with runtime row counts."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from b12x.gemm.blockscaled._quantize import _quantize

    launches = {}
    cases = [
        (recipe, dtype, input_k, False)
        for recipe in ("mxfp8", "mxfp4")
        for dtype in ("bf16", "fp16") for input_k in (128, 160, 1024)
    ] + [
        ("nvfp4", "bf16", input_k, reciprocal)
        for input_k in (128, 1024) for reciprocal in (False, True)
    ]
    options = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
    for recipe, dtype, input_k, reciprocal in cases:
        fp4 = recipe != "mxfp8"
        name = f"activation_pack_{recipe}_{dtype}_k{input_k}_reciprocal{int(reciprocal)}"
        padded_k = input_k if recipe == "mxfp4" else (input_k + 127) // 128 * 128
        constants = dict(INPUT_K=input_k, K=padded_k,
                         FP4=fp4, RECIPROCAL=reciprocal, GROUP=16 if recipe == "nvfp4" else 32,
                         CHUNKS=16)
        signature = dict(X=f"*{dtype}", Q="*u8" if fp4 else "*fp8e4nv", S="*u8",
                         AG="*fp32", WG="*fp32", ALPHA="*fp32", M="i32")
        if recipe == "mxfp4":
            for pointer in ("AG", "WG", "ALPHA"):
                signature[pointer] = "constexpr"
                constants[pointer] = None
        source = ASTSource(
            _quantize, signature, constexprs=constants,
            attrs={(i,): [["tt.divisibility", 16]] for i in range(3 if recipe == "mxfp4" else 6)},
        )
        compiled = triton.compile(source, target=GPUTarget("cuda", 103, 32), options=options)
        if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
            raise RuntimeError(f"{name}: implicit launch scratch is unsupported")
        directory = out / name
        directory.mkdir()
        (directory / (name + ".ptx")).write_text(compiled.asm["ptx"])
        (directory / (name + ".cubin")).write_bytes(compiled.asm["cubin"])
        (directory / (name + ".mlir")).write_text(compiled.asm["ttgir"])
        (directory / (name + ".metadata.json")).write_text(json.dumps({
            "role": "supporting activation packing; core GEMM remains CuTe DSL",
            "signature": signature, "constants": constants, "options": options,
            "metadata": compiled.metadata._asdict(),
        }, indent=2, default=str) + "\n")
        launches[name] = compiled
    return launches


def compile_fp8(out):
    """Compile the production tensor and compact K128 FP8 launch factory."""
    import cutlass.cute as cute
    import torch
    from b12x.gemm.blockscaled import _fp8_cute as fp8

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec, **kwargs):
        directory = out / case
        directory.mkdir()
        compiled = cute.compile(
            kernel, *args, no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (case + ".mlir")).write_text(str(compiled.ir_module))
        launches[case] = compiled
        return compiled

    fp8.compile_kernel.cache_clear()
    with (patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
          patch.object(fp8, "b12x_compile", capture)):
        for block, n, k, groups in (
            (False, 64, 128, 1), (False, 132, 256, 1),
            (False, 16386, 1024, 1), (False, 136, 384, 2),
            (True, 256, 384, 1),
        ):
            for c_dtype in ("bfloat16", "float16", "float32"):
                case = f"fp8_block{int(block)}_n{n}_k{k}_g{groups}_{c_dtype}"
                fp8.compile_kernel(n, k, groups, c_dtype, block, False, 0, 148, "sm_103a")
        for block in (False, True):
            case = f"fp8_block{int(block)}_alpha_one"
            fp8.compile_kernel(128, 128, 1, "bfloat16", block, True, 0, 148, "sm_103a")
        for block in (False, True):
            case = f"fp8_block{int(block)}_large_output_address"
            fp8.compile_kernel(524288, 128, 1, "bfloat16", block, True, 0, 148, "sm_103a")
    return launches


def compile_fp6(out):
    """Compile packed FP6 GEMM, mixed formats, and runtime-row quantization."""
    import cutlass.cute as cute
    import torch
    from b12x.gemm.blockscaled import _fp6
    from b12x.quantization.mxfp6 import _rows

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec, **kwargs):
        directory = out / case
        directory.mkdir()
        compiled = cute.compile(kernel, *args, no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}")
        (directory / (case + ".mlir")).write_text(str(compiled.ir_module))
        launches[case] = compiled
        return compiled

    formats = (("e3m2", "e2m3", False, False), ("e2m3", "e3m2", False, False),
               ("e4m3", "e2m3", True, False), ("e3m2", "e4m3", False, True),
               ("e3m2", "e2m3", True, False), ("e2m3", "e3m2", False, True),
               ("e3m2", "e3m2", True, True))
    for fn in (_fp6.compile_kernel, _rows.compile_scales, _rows.compile_quantizer):
        fn.cache_clear()
    with (patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
          patch.object(_fp6, "b12x_compile", capture),
          patch.object(_rows, "b12x_compile", capture)):
        for a_fmt, b_fmt, a_bytes, b_bytes in formats:
            for dtype in ("bfloat16", "float16", "float32"):
                case = f"fp6_{a_fmt}_{b_fmt}_bytes{int(a_bytes)}{int(b_bytes)}_{dtype}_n136_k384_g2"
                _fp6.compile_kernel(136, 384, 2, a_fmt, b_fmt, a_bytes, b_bytes, dtype, False, True, 0)
        for a_fmt in ("e3m2", "e4m3"):
            for n, k, one in ((8192, 4096, False), (8, 128, True)):
                case = f"fp6_{a_fmt}_e2m3_n{n}_k{k}_alpha_one{int(one)}"
                _fp6.compile_kernel(n, k, 1, a_fmt, "e2m3", a_fmt == "e4m3", False, "bfloat16", one, False, 0)
        case = "fp6_large_output_address"
        _fp6.compile_kernel(524288, 128, 1, "e3m2", "e2m3", False, False, "bfloat16", True, False, 0)
        for k in (384, 65536):
            for fmt in ("e2m3", "e3m2", "e4m3"):
                for per_row in (False, True):
                    case = f"fp6_row_scales_k{k}_{fmt}_per_row{int(per_row)}"
                    _rows.compile_scales(k, fmt, per_row, 0, "sm_103a")
                    for packed in (False, True) if fmt != "e4m3" else (False,):
                        case = f"fp6_rows_k{k}_{fmt}_per_row{int(per_row)}_packed{int(packed)}"
                        _rows.compile_quantizer(k, fmt, per_row, packed, 0, "sm_103a")
    return launches


def compile_blockscaled(out):
    """Compile production dense tcgen05 and inline A16 launch factories."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import torch
    from b12x._lib import dense_gemm as a16
    from b12x.gemm.blockscaled import _sm103 as native

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec, **kwargs):
        directory = out / case
        directory.mkdir()
        compiled = cute.compile(
            kernel, *args, no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (case + ".mlir")).write_text(str(compiled.ir_module))
        launches[case] = compiled
        return compiled

    with patch.object(torch.cuda, "is_current_stream_capturing", lambda: False):
        native.compile_kernel.cache_clear()
        with patch.object(native, "b12x_compile", capture):
            for recipe in ("nvfp4", "mxfp4", "mxfp8"):
                for n, k, groups in ((136, 384, 2), (512, 1024, 1)):
                    for c_dtype in ("bfloat16", "float16", "float32"):
                        case = f"blockscaled_{recipe}_n{n}_k{k}_g{groups}_{c_dtype}"
                        native.compile_kernel(n, k, groups, recipe, c_dtype, 0)
                case = f"blockscaled_{recipe}_n8_k128_g1_alpha_one"
                native.compile_kernel(8, 128, 1, recipe, "bfloat16", 0, True)
            case = "blockscaled_nvfp4_large_output_address"
            native.compile_kernel(524288, 128, 1, "nvfp4", "bfloat16", 0, True)
        a16._get_compiled_dense_gemm.cache_clear()
        with (patch.object(a16, "b12x_compile", capture),
              patch.object(a16, "current_cuda_stream", lambda: cuda.CUstream(0))):
            for recipe in ("nvfp4", "mxfp8"):
                for bn, bk, split, k, input_k, reciprocal in (
                    (64, 64, 1, 1024, 1024, False),
                    (128, 64, 4, 1024, 1024, False),
                    (64, 128, 4, 1024, 1024, False),
                    (128, 128, 8, 1024, 1024, False),
                    (64, 64, 1, 96, 80, False),
                    *(([(128, 64, 4, 1024, 1024, True)]) if recipe == "nvfp4" else []),
                ):
                    case = f"a16_{recipe}_n{bn}_k{bk}_s{split}_width{k}_input{input_k}_reciprocal{int(reciprocal)}"
                    a16._get_compiled_dense_gemm(
                        136, k, 1, split, "k", "k", "n", cutlass.BFloat16, cutlass.Uint8,
                        cutlass.Float32 if split > 1 else cutlass.BFloat16,
                        cutlass.Float32, 16 if recipe == "nvfp4" else 32, 16, bk,
                        (16, bn), (1, 1), a16._DenseGemmPolicy(True, True, False, split, False, False),
                        148, "sm_103a", "tma", False, False, False,
                        alpha_is_one=recipe == "mxfp8", target_occupancy_override=1,
                        weight_only=recipe, alpha_reciprocal=reciprocal,
                        input_k=input_k, device_ordinal=0,
                    )
            with patch("b12x._lib.gating.get_compute_capability", lambda device=None: (10, 3)):
                a16._get_compiled_dense_split_k_reduce.cache_clear()
                for split in (2, 4, 8):
                    case = f"a16_reduce_s{split}"
                    a16._get_compiled_dense_split_k_reduce(136, split, 0)
                for split in (2, 4):
                    case = f"dense_reduce_float16_s{split}"
                    a16._get_compiled_dense_split_k_reduce(136, split, 0, "float16")
    return launches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--gate-first", action="store_true")
    parser.add_argument("--nvdisasm", type=Path)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument(
        "--component",
        choices=(
            "moe",
            "roce",
            "trellis",
            "sequence",
            "dense_mla",
            "sparse_mla",
            "compressed_mla",
            "mhc",
            "dsa_indexer",
            "projection",
            "blockscaled",
            "fp8",
            "fp6",
            "block_fp8_linear",
            "wo_projection",
            "activation_packing",
            "all",
        ),
        default="moe",
    )
    args = parser.parse_args()
    # CuTe must have an architecture even when the CUDA driver is unavailable.
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_103a")
    from b12x.moe._shared.kernels.sm103.launch import compile_launches
    from b12x.moe.fused_moe._sm103 import heuristic
    from b12x.moe.fused_moe._policy import MoeDecodeQuery

    heuristic(
        MoeDecodeQuery(
            "nvfp4",
            "modelopt_nvfp4",
            "silu",
            args.experts,
            args.hidden,
            args.intermediate,
            args.top_k,
            args.capacity,
            args.top_k * args.capacity,
        )
    )
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(
            "--output-dir must be empty to keep artifact provenance unambiguous"
        )
    if any(c.isspace() for c in str(out)):
        parser.error("the CuTe dump directory must not contain whitespace")
    out.mkdir(parents=True, exist_ok=True)
    caps = SimpleNamespace(
        k=args.hidden,
        n=args.intermediate,
        weight_E=args.experts,
        max_tokens=args.capacity,
        num_topk=args.top_k,
        w13_layout="w31" if args.gate_first else "w13",
        swiglu_limit=None,
    )
    manifest = {
        "status": "compiling",
        "runtime_qualified": False,
        "target": "sm_103a",
        "component": args.component,
        "command": sys.argv,
        "mhc_offline_device_attributes": {
            "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": 228 * 1024,
            "source": "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#shared-memory-capacity",
            "use": "CuTe preferred-SMEM-carveout calculation for minimum CTA residency",
        } if args.component in ("mhc", "all") else None,
        "trellis_geometry": {
            "experts": 384,
            "hidden": 5120,
            "intermediate": 2304,
            "stage": "quantizer-basis tiles, inline FP16 projections, and uniform or MCG projection-tiered expert MoE",
        },
        "roce_geometry": {
            "world_size": 2,
            "threads": 512,
            "slots": 2,
            "flag_stride": 128,
            "hca_count": 1,
        },
        "geometry": vars(caps),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        **source_identity(ROOT),
        "source_sha256": package_source_sha256(ROOT),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "cuda-python",
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "nvidia-cutlass-dsl-libs-core",
                "nvidia-cutlass-dsl-libs-cu13",
                "triton",
            )
        },
    }
    for name in ("nvdisasm", "cuobjdump"):
        tool = getattr(args, name)
        if tool:
            manifest[name] = subprocess.check_output(
                [str(tool.resolve()), "--version"], text=True
            )
    try:
        launches = {}
        if args.component in ("moe", "all"):
            launches.update(compile_launches(caps, offline=True, artifact_dir=out))
        if args.component in ("roce", "all"):
            launches.update(compile_roce(out))
        if args.component in ("trellis", "all"):
            launches.update(compile_trellis(out))
        if args.component in ("sequence", "all"):
            launches.update(compile_sequence(out))
        if args.component in ("dense_mla", "all"):
            launches.update(compile_dense_mla(out))
        if args.component in ("sparse_mla", "all"):
            launches.update(compile_sparse_mla(out))
        if args.component in ("compressed_mla", "all"):
            launches.update(compile_compressed_mla(out))
        if args.component in ("mhc", "all"):
            launches.update(compile_mhc(out))
        if args.component in ("dsa_indexer", "all"):
            launches.update(compile_dsa_indexer(out))
        if args.component in ("projection", "all"):
            launches.update(compile_bf16_projection(out))
        if args.component in ("blockscaled", "all"):
            launches.update(compile_blockscaled(out))
        if args.component in ("fp8", "all"):
            launches.update(compile_fp8(out))
        if args.component in ("fp6", "all"):
            launches.update(compile_fp6(out))
        if args.component in ("block_fp8_linear", "all"):
            launches.update(compile_block_fp8_linear(out))
        if args.component in ("wo_projection", "all"):
            launches.update(compile_wo_projection(out))
        if args.component in ("activation_packing", "all"):
            launches.update(compile_activation_packing(out))
        artifacts = []
        for key in launches:
            name = key if isinstance(key, str) else key[0] + "_" + key[1].__name__
            directory = out / name
            ptxs = list(directory.glob("*.ptx"))
            cubins = list(directory.glob("*.cubin"))
            if len(ptxs) != 1 or len(cubins) != 1:
                raise RuntimeError(
                    f"{name}: expected exactly one PTX and cubin; use an empty output directory"
                )
            text = ptxs[0].read_text()
            if ".target sm_103a" not in text:
                raise RuntimeError(f"{name}: wrong PTX target")
            if "tcgen05.ld." in text and "tcgen05.wait::ld" not in text:
                raise RuntimeError(f"{name}: missing asynchronous TMEM load completion wait")
            if name.startswith("fc") and "tcgen05.mma" not in text:
                raise RuntimeError(f"{name}: missing native tcgen05 MMA")
            inspected = []
            if args.nvdisasm:
                sass = subprocess.check_output(
                    [str(args.nvdisasm.resolve()), str(cubins[0])],
                    text=True,
                    timeout=60,
                )
                if name.startswith("fc") and "UTCOMMA.BLOCK16" not in sass:
                    raise RuntimeError(f"{name}: missing block-scaled UMMA in SASS")
                path = directory / (name + ".sass")
                path.write_text(sass)
                inspected.append(path)
            if args.cuobjdump:
                resources = subprocess.check_output(
                    [str(args.cuobjdump.resolve()), "-res-usage", str(cubins[0])],
                    text=True,
                    timeout=60,
                )
                path = directory / (name + ".resources.txt")
                path.write_text(resources)
                inspected.append(path)
            artifacts.append(
                {
                    "name": name,
                    "native_mma": "tcgen05.mma" in text,
                    "files": {
                        p.name: {
                            "bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        }
                        for p in [*ptxs, *cubins, *directory.glob("*.mlir"),
                                  *directory.glob("*.metadata.json"), *inspected]
                    },
                }
            )
        if package_source_sha256(ROOT) != manifest["source_sha256"]:
            raise RuntimeError("package source changed during compilation")
        if (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != manifest["script_sha256"]
        ):
            raise RuntimeError("compile script changed during compilation")
        manifest.update(
            status="cross-compiled", callables=len(launches), artifacts=artifacts
        )
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
