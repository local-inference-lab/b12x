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


_EXPORT_DIRECTORIES = {}


def _export_compile(kernel, args, compile_spec, directory):
    """Keep production program identities while exporting native compiler artifacts."""
    import shutil
    from b12x._lib import compiler
    from b12x._lib.compile_plan import program_keys
    original = compiler._call_cute_compile

    def export(compile_callable, func, operands, kwargs, *, compile_spec, cache_key):
        compiled = original(compile_callable, func, operands,
            {**kwargs, "options": kwargs.get("options", "") +
             f" --keep-ptx --keep-cubin --dump-dir={directory}"},
            compile_spec=compile_spec, cache_key=cache_key)
        _EXPORT_DIRECTORIES[cache_key] = directory
        return compiled

    with patch.object(compiler, "_call_cute_compile", export):
        compiled = compiler.compile(kernel, *args, compile_spec=compile_spec)
    key, = program_keys(compiled)
    source = _EXPORT_DIRECTORIES[key.key]
    if source != directory:
        for artifact in source.iterdir():
            if artifact.suffix in {".ptx", ".cubin"}:
                shutil.copyfile(artifact, directory / artifact.name)
    return compiled


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
        compiled = _export_compile(kernel, args, compile_spec, directory)
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
                from b12x.sequence.kda_prefill._tuning import KdaPrefillConfig
                from b12x.sequence.gdn_prefill._tuning import GdnPrefillConfig
                config_type = KdaPrefillConfig if recipe == "kda" else GdnPrefillConfig
                plan = impl._materialize_layout(caps, config_type(
                    backend="cutedsl", v_split=64, k_split=1, stages=3, window_tiles=128,
                ))
                binding = SimpleNamespace(
                    _state=plan,
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
                _state=SimpleNamespace(caps=caps),
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
    kda.compile_kernels.cache_clear()
    return launches


def compile_mtp_feedback(out):
    """Compile production GLM, Qwen and DeepSeek feedback launch factories."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from b12x.sequence.mtp_feedback import _concat as concat, _cute_prefill as gemm
    from b12x.sequence.mtp_feedback import _kernels as aux
    from b12x.sequence.mtp_feedback import _fp8 as stream_fp8
    from b12x._lib import fp8_gemm

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec, **kwargs):
        name = case
        if case.startswith("mtp_fp8_aux"):
            name += "_" + compile_spec.kernel_id.rsplit(".", 1)[-1]
        directory = out / name
        directory.mkdir()
        compiled = _export_compile(kernel, args, compile_spec, directory)
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled
        return compiled

    concat.compile_norm.cache_clear()
    gemm._KERNEL_CACHE.clear()
    stream_fp8.compile_aux.cache_clear()
    fp8_gemm.compile_kernel.cache_clear()
    with (patch.object(torch.cuda, "device"),
          patch.object(concat, "current_cuda_stream", lambda: cuda.CUstream(0)),
          patch.object(gemm, "current_cuda_stream", lambda: cuda.CUstream(0)),
          patch.object(stream_fp8, "current_cuda_stream", lambda: cuda.CUstream(0)),
          patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
          patch.object(concat, "compile_cute", capture),
          patch.object(stream_fp8, "compile_cute", capture),
          patch.object(fp8_gemm, "b12x_compile", capture),
          patch.object(gemm, "b12x_compile", capture)):
        for hidden in (256, 320, 4096):
            for warps in (4, 8):
                for dtype in (torch.int32, torch.int64):
                    case = f"mtp_concat_norm_h{hidden}_w{warps}_{dtype}"
                    concat.compile_norm(hidden, 1 << (hidden - 1).bit_length(),
                                        warps, dtype, 0, (10, 3))
        for hidden, streams in ((128, 1), (256, 3), (5120, 4)):
            for warps in (4, 8):
                for dtype in (torch.int32, torch.int64):
                    case = f"mtp_fp8_norm_h{hidden}_s{streams}_w{warps}_{dtype}"
                    concat.compile_norm(hidden, 1 << (hidden - 1).bit_length(),
                                        warps, dtype, 0, (10, 3), streams, 20)
            case = f"mtp_fp8_aux_h{hidden}_s{streams}"
            stream_fp8.compile_aux(hidden, streams, 0, "sm_103a")
            case = f"mtp_fp8_projection_h{hidden}"
            fp8_gemm.compile_kernel(hidden, hidden, 1, "bfloat16", True, True,
                                    0, 148, "sm_103a")
        for rows in (16, 32, 64, 128):
            for contract, hidden, streams, add in (
                ("concat", 4096, 1, False),
                ("qwen_embedding", 2560, 4, False),
                ("qwen_state", 2560, 4, True),
            ):
                case = f"mtp_{contract}_projection_rows{rows}"
                gemm.compile_mtp_prefill_bf16_gemm(
                    rows, hidden, 2 * hidden if contract == "concat" else hidden,
                    device=torch.device("cuda:0"), streams=streams, add_token_path=add,
                )

    cases = (
        ("token_norm", aux._token_norm_kernel,
         dict(token_embedding="*bf16", token_norm_weight="*bf16", token_normalized="*bf16", eps="fp32"),
         dict(HIDDEN_SIZE=2560, BLOCK_H=4096)),
        ("state_partial", aux._state_partial_sum_kernel,
         dict(multi_state="*bf16", state_partial_sums="*fp32"),
         dict(HIDDEN_SIZE=2560, BLOCK_H=4096)),
        ("state_norm", aux._state_norm_kernel,
         dict(multi_state="*bf16", state_partial_sums="*fp32", state_norm_weight="*bf16", state_normalized="*bf16", eps="fp32"),
         dict(HIDDEN_SIZE=2560, BLOCK_H=4096, STREAMS=4, BLOCK_S=4)),
    )
    for name, kernel, signature, constants in cases:
        for warps in (4, 8):
            case = f"mtp_qwen_{name}_w{warps}"
            options = dict(num_warps=warps, num_stages=1)
            compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants),
                                      target=GPUTarget("cuda", 103, 32), options=options)
            if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
                raise RuntimeError(f"{case}: implicit launch scratch is unsupported")
            directory = out / case
            directory.mkdir()
            (directory / (case + ".ptx")).write_text(compiled.asm["ptx"])
            (directory / (case + ".cubin")).write_bytes(compiled.asm["cubin"])
            (directory / (case + ".mlir")).write_text(compiled.asm["ttgir"])
            (directory / (case + ".metadata.json")).write_text(json.dumps({
                "role": "existing Qwen normalization auxiliary; projections use CuTe DSL",
                "signature": signature, "constants": constants, "options": options,
                "metadata": compiled.metadata._asdict(),
            }, indent=2, default=str) + "\n")
            launches[case] = compiled
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
        name = "indexer_" + compile_spec.kernel_id.rsplit(".", 1)[-1] + "_" + case
        directory = out / name
        directory.mkdir()
        compiled = _export_compile(kernel, args, compile_spec, directory)
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled
        return compiled

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
    from b12x.attention.compressed_sparse_mla._tuning import SparseMlaConfig
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
                    max_chunks_per_row=4 if mode == "decode" else 1, swa_page_size=64, indexed_page_size=32,
                )
                plan = plan_compressed_sparse_mla_scratch(caps, execution_config=SparseMlaConfig(
                    max_chunks_per_row=4 if mode == "decode" else 1, split_chunk_size=1,
                    single_pass=mode != "decode", v41_compute_mode=precision, backend="warp",
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
        ("btx_mcg_coupled", "mcg", 3, True, "situ", torch.bfloat16),
        ("sqg", "sqg_e4m3", 4, False, "situ", torch.float16),
        ("sqg_fp16", "sqg_fp16", 5, False, "silu", torch.bfloat16),
        ("canonical_sqg_fp16", "sqg_fp16", 5, False, "silu", torch.bfloat16),
        ("canonical_sqg_fp16_coupled", "sqg_fp16", 5, True, "situ", torch.bfloat16),
    ):
        caps = SimpleNamespace(
            k=5120, n=2304, weight_E=384, max_tokens=128, num_topk=8,
            route_num_experts=768, dtype=dtype, activation=activation,
            weight_plan=SimpleNamespace(coupled_hadamard=coupled, source_format="b12x_trellis" if codebook == "sqg_e4m3" or label.startswith("canonical_") else "btx",
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
    for codebook, group_size, coupled in (
        ("mcg", 32, False), ("mcg", 256, True),
        ("sqg_e4m3", 32, False), ("sqg_e4m3", 256, True),
        ("sqg_fp16", 32, False), ("sqg_fp16", 256, True),
    ):
        label = f"trellis_atoms_{codebook}_g{group_size}_" + ("coupled_" if coupled else "")
        caps = SimpleNamespace(
            k=5120, n=2304, weight_E=384, max_tokens=128, num_topk=8,
            route_num_experts=768, dtype=torch.bfloat16,
            activation="situ" if coupled else "silu",
            weight_plan=SimpleNamespace(
                coupled_hadamard=coupled, source_format="b12x_trellis",
                trellis_bits=5 if codebook == "sqg_fp16" else 3, trellis_codebook=codebook,
                trellis_group_size=group_size,
            ),
        )
        compiled = compile_moe(caps, offline=True, artifact_dir=out, artifact_prefix=label)
        launches.update({label + key: fn for key, fn in compiled.items()})
    for codebook in ("mcg", "sqg_e4m3"):
        for coupled in (False, True):
            label = f"trellis_btx_pairs_{codebook}_" + ("coupled_" if coupled else "")
            caps = SimpleNamespace(
                k=5120, n=2304, weight_E=384, max_tokens=128, num_topk=8,
                route_num_experts=768, dtype=torch.bfloat16,
                activation="situ" if coupled else "silu",
                weight_plan=SimpleNamespace(
                    coupled_hadamard=coupled, source_format="btx",
                    trellis_bits=3, trellis_codebook=codebook,
                    trellis_rate_granularity="per_expert_pair",
                ),
            )
            compiled = compile_moe(caps, offline=True, artifact_dir=out, artifact_prefix=label)
            launches.update({label + key: fn for key, fn in compiled.items()})
    launches.update(compile_trellis_clamped(out))
    return launches


def compile_trellis_clamped(out):
    """Compile V4.1 SiLU-clamped K3 expert geometry for TP1 and TP2."""
    import torch
    from b12x.moe.fused_moe._sm103_trellis import compile_launches

    launches = {}
    for tp in (1, 2):
        label = f"trellis_clamped_v41_tp{tp}_"
        caps = SimpleNamespace(
            k=5120, n=2304 // tp, weight_E=384, max_tokens=128, num_topk=6,
            route_num_experts=384, dtype=torch.bfloat16, activation="silu",
            swiglu_limit=10.0,
            weight_plan=SimpleNamespace(
                coupled_hadamard=False, source_format="btx",
                trellis_bits=3, trellis_codebook="mcg",
            ),
        )
        compiled = compile_launches(
            caps, offline=True, artifact_dir=out, artifact_prefix=label
        )
        launches.update({label + key: fn for key, fn in compiled.items()})
    return launches


def compile_mhc(out):
    """Export native mHC artifacts from the production preparation factories."""
    import shutil
    from scripts._sm103_preparation_corpus import CASES

    evidence = out / "mhc_preparation"
    command = [sys.executable, str(ROOT / "scripts/compile_sm103_prepared.py"),
               "--output-dir", str(evidence), "--workers", "2"]
    for case in CASES:
        if case.startswith("mhc:"):
            command.extend(("--case", case))
    subprocess.run(command, check=True)
    launches = {}
    for source in sorted((evidence / "native").iterdir()):
        name = "mhc_" + source.name
        directory = out / name
        directory.mkdir()
        for artifact in source.iterdir():
            destination = name + ".metadata.json" if artifact.name == "metadata.json" else artifact.name
            shutil.copyfile(artifact, directory / destination)
        launches[name] = None
    if not launches:
        raise RuntimeError("mHC preparation exported no native artifacts")
    return launches


def compile_bf16_projection(out):
    """Export selected SIMT, warp-MMA, prefill and vocabulary programs."""
    from b12x.gemm.bf16_gemv import _kernel as projection, _prefill as prefill
    from b12x.gemm.bf16_vocab_projection import _cute as vocab

    launches = {}
    name = ""

    def capture(kernel, *args, compile_spec=None, **kwargs):
        directory = out / name
        directory.mkdir()
        compiled = _export_compile(kernel, args, compile_spec, directory)
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled
        return compiled

    projection.compile_projection.cache_clear()
    with patch.object(projection, "b12x_compile", capture):
        for source in ("bfloat16", "float32"):
            for weight in ("bfloat16", "float32"):
                for output in ("bfloat16", "float32"):
                    for rows in (1, 2, 4, 8):
                        name = f"projection_simt_{source}_{weight}_{output}_r{rows}"
                        projection.compile_projection(0, "simt", rows, 512, 4096,
                                                      source, weight, output, "float32")
        for output in ("bfloat16", "float32"):
            for bias in (None, "float32"):
                name = f"projection_mma_{output}_bias{bias}"
                projection.compile_projection(0, "mma", 8, 512, 4096,
                                              "bfloat16", "bfloat16", output, bias)
    prefill.compile_prefill.cache_clear()
    with patch.object(prefill, "b12x_compile", capture):
        for output in ("bfloat16", "float32"):
            name = f"projection_prefill_{output}"
            prefill.compile_prefill(0, 257, 512, 5120, output)
    vocab.compile_kernel.cache_clear()
    with patch.object(vocab, "b12x_compile", capture):
        for n, k in ((97, 259), (248320, 2560), (124160, 5120), (129280, 5120),
                     (154880, 4096), (77440, 6144), (524297, 4096)):
            name = f"vocabulary_n{n}_k{k}"
            vocab.compile_kernel(n, k, 0, "sm_103a")
    return launches


def compile_v41_support(out, component):
    """Compile production CSA, embedding and HyperConnection launch factories."""
    from contextlib import nullcontext

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec=None, **kwargs):
        directory = out / case
        directory.mkdir()
        compiled = _export_compile(kernel, args, compile_spec, directory)
        (directory / (case + ".mlir")).write_text(str(compiled.ir_module))
        launches[case] = compiled
        return compiled

    with patch.object(torch.cuda, "device", lambda *args: nullcontext()):
        if component in ("mla_compress", "all"):
            from b12x.attention.mla_compress import _cute as compressor

            compressor.compile_compress.cache_clear()
            with (
                patch.object(compressor, "compile_cute", capture),
                patch.object(compressor, "current_cuda_stream", lambda: cuda.CUstream(0)),
            ):
                for ratio in (1, 2):
                    for tokens, requests, states in ((12, 4, 8), (8192, 128, 2**22 + 2)):
                        case = f"mla_compress_ratio{ratio}_t{tokens}_r{requests}_s{states}"
                        compressor.compile_compress(ratio, tokens, requests, states, 0)
        if component in ("embedding", "all"):
            from b12x.sequence.embedding import _kernel as embedding

            embedding.compile_embedding.cache_clear()
            with (
                patch.object(embedding, "compile_cute", capture),
                patch.object(embedding, "current_cuda_stream", lambda: cuda.CUstream(0)),
            ):
                for width in (129, 4096, 5120):
                    for dtype in (torch.bfloat16, torch.float32):
                        for ids in (torch.int32, torch.int64):
                            case = f"embedding_h{width}_{dtype}_{ids}"
                            embedding.compile_embedding(width, dtype, ids, 0)
        if component in ("hyperconnection", "all"):
            from b12x.norm.hyperconnection import _cute as hc

            def tensor(*shape, dtype=torch.bfloat16):
                return torch.empty(shape, device="meta", dtype=dtype)

            def compile_run(key, entry, tensors, *, eps=None, runtime_ints=(), runtime_int64s=()):
                hc._compile(
                    key, entry, len(tensors), torch.device("cuda:0"),
                    has_eps=eps is not None, runtime_ints=len(runtime_ints),
                    runtime_int64s=len(runtime_int64s),
                    pointer_dtypes=tuple(t.dtype for t in tensors),
                )

            hc.clear_caches()
            with (
                patch.object(hc, "compile_cute", capture),
                patch.object(hc, "current_cuda_stream", lambda: cuda.CUstream(0)),
                patch.object(hc, "_device_index", lambda tensor: 0),
                patch.object(hc, "_run", compile_run),
            ):
                for streams, hidden in ((1, 5120), (4, 5120), (4, 2560), (3, 257)):
                    prefix = f"hyperconnection_s{streams}_h{hidden}"
                    state = tensor(1, streams * hidden)
                    for zero_centered, dtype in ((True, torch.bfloat16), (False, torch.bfloat16), (False, torch.float32)):
                        case = f"{prefix}_norm_centered{int(zero_centered)}_{dtype}"
                        hc.grouped_rmsnorm(state, tensor(streams * hidden, dtype=dtype),
                            tensor(1, streams * hidden), eps=1e-6, streams=streams,
                            hidden_size=hidden, zero_centered=zero_centered)
                    for mask in (False, True):
                        case = f"{prefix}_engram_mask{int(mask)}"
                        hc.engram_mix(state, tensor(1, 2 * hidden), tensor(2, streams, hidden),
                            tensor(1, dtype=torch.bool) if mask else None,
                            tensor(1, streams * hidden), eps=1e-6,
                            streams=streams, hidden_size=hidden)
                    case = f"{prefix}_gate"
                    hc.gate_mean(state, tensor(1, streams * hidden), tensor(1, hidden),
                        streams=streams, hidden_size=hidden)
                    case = f"{prefix}_combine"
                    hc.combine(state, tensor(1, hidden), tensor(1, streams),
                        tensor(1, streams * hidden), streams=streams, hidden_size=hidden)
                    if (streams, hidden) == (4, 2560):
                        case = f"{prefix}_combine_norm"
                        hc.combine_norm(state, tensor(1, hidden), tensor(1, streams),
                            tensor(streams * hidden), tensor(1, streams * hidden),
                            tensor(1, streams * hidden), eps=1e-6,
                            streams=streams, hidden_size=hidden)
                    case = f"{prefix}_scaled_silu"
                    hc.scaled_silu(tensor(1, 320), tensor(1, 320), streams=streams)
                for left in (torch.bfloat16, torch.float32):
                    for right in (torch.bfloat16, torch.float32):
                        for output in (torch.bfloat16, torch.float32):
                            case = f"hyperconnection_add_{left}_{right}_{output}"
                            hc.pointwise("add", tensor(1, 257, dtype=left),
                                tensor(1, 257, dtype=right), tensor(1, 257, dtype=output))
                    for output in (torch.bfloat16, torch.float32):
                        case = f"hyperconnection_sigmoid_{left}_{output}"
                        value = tensor(1, 257, dtype=left)
                        hc.pointwise("sigmoid", value, value, tensor(1, 257, dtype=output))
                for width in (257, 2048, 2304):
                    for limit in (2.0, float("inf")):
                        for rounded in (False, True):
                            case = f"hyperconnection_swiglu_h{width}_limit{limit}_round{int(rounded)}"
                            value = tensor(1, 2 * width)
                            hc.pointwise("swiglu", value, value, tensor(1, width),
                                width=width, limit=limit, round_silu=rounded)
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
        compiled = _export_compile(kernel, args, compile_spec, directory)
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
                        k, dtype, 8, 128, "linear", floor, device_ordinal=0, sm_count=148,
                    )
            for subgroup, threads, order in ((0, 256, "linear"), (4, 256, "linear"),
                                              (8, 256, "trellis_native_mma")):
                for floor in (0.0, 1e-4):
                    case = f"mxfp8_rows_{dtype}_lanes{subgroup}_{order}_floor{floor}"
                    quant._get_compiled_mxfp8_rows_quant(
                        8192, dtype, subgroup, threads, order, floor, device_ordinal=0, sm_count=148,
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
        compiled = _export_compile(kernel, args, compile_spec, directory)
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


def compile_engram(out):
    """Compile Engram hash metadata and packed gathers without allocating tables."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from b12x.sequence.engram import _kernels as kernels
    from b12x.sequence.engram.geometry import build_geometry

    geometry = build_geometry()
    from b12x.sequence.ple_hash import _kernels as hash_kernels
    cases = [
        ("engram_requests", hash_kernels._request_ids_kernel,
         dict(query_start_loc_ptr="*i32", num_seqs_ptr="*i32", num_tokens_ptr="*i32",
              request_ids_ptr="*i32"), dict(MAX_TOKENS=4096), 1),
        ("engram_compress", kernels._compress,
         dict(ids="*i64", token_mask="*i1", token_map="*i64", num_tokens="*i32",
              compressed="*i64"), dict(V=129280), 1),
        ("engram_hash", kernels._hash,
         dict(compressed="*i64", starts="*i32", slots="*i32", history="*i64",
              num_tokens="*i32", request_ids="*i32", multipliers="*i64",
              primes="*i64", offsets="*i64", hashes="*i64"), dict(PAD=2), 1),
    ]
    for layer, rows in zip(geometry.layer_ids, geometry.num_embeddings, strict=True):
        for rank in (0, 1):
            shard = (rows + 1) // 2
            for compact, resident in ((False, False), (True, False), (True, True)):
                name = f"engram_lookup_l{layer}_tp2r{rank}_compact{int(compact)}_resident{int(resident)}"
                cases.append((name, kernels._lookup,
                              dict(weight="*fp8e4nv", scales="*u8", hashes="*i64",
                                   num_tokens="*i32", out="*bf16", prepared_tokens="i32"),
                              dict(T=4096, ROWS=rows, START=rank * shard,
                                   END=(rank + 1) * shard, COMPACT=compact,
                                   RESIDENT_SCALES=resident), 4))
    launches = {}
    for name, kernel, signature, constants, warps in cases:
        options = dict(num_warps=warps, num_stages=1)
        compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants),
                                  target=GPUTarget("cuda", 103, 32), options=options)
        if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
            raise RuntimeError(f"{name}: implicit launch scratch is unsupported")
        directory = out / name
        directory.mkdir()
        (directory / (name + ".ptx")).write_text(compiled.asm["ptx"])
        (directory / (name + ".cubin")).write_bytes(compiled.asm["cubin"])
        (directory / (name + ".mlir")).write_text(compiled.asm["ttgir"])
        (directory / (name + ".metadata.json")).write_text(json.dumps({
            "role": "supporting Engram hash metadata and packed FP8 gather",
            "signature": signature, "constants": constants, "options": options,
            "metadata": compiled.metadata._asdict(),
        }, indent=2, default=str) + "\n")
        launches[name] = compiled
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
    from b12x._lib import fp8_gemm as fp8

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec, **kwargs):
        directory = out / case
        directory.mkdir()
        compiled = _export_compile(kernel, args, compile_spec, directory)
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
        compiled = _export_compile(kernel, args, compile_spec, directory)
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
        compiled = _export_compile(kernel, args, compile_spec, directory)
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
    parser.add_argument("--hot-experts", type=int, help="HBM expert rows for the residency component")
    parser.add_argument("--swiglu-limit", type=float)
    parser.add_argument("--nvdisasm", type=Path)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument(
        "--component",
        choices=(
            "moe",
            "residency",
            "roce",
            "trellis",
            "trellis_clamped",
            "sequence",
            "mtp_feedback",
            "mla_compress",
            "hyperconnection",
            "embedding",
            "engram",
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
    from b12x.moe.fused_moe._impl import plan_b12x_fp4_moe_weights
    from b12x.moe.fused_moe._sm103 import query_for_weight_plan
    import torch
    if args.component != "residency":
        weight_plan = plan_b12x_fp4_moe_weights(
            params_dtype=torch.bfloat16,
            quant_modes="nvfp4", source_format="modelopt_nvfp4", activation="silu",
            num_experts=args.experts, hidden_size=args.hidden, intermediate_size=args.intermediate,
        )
        heuristic(query_for_weight_plan(weight_plan, quant_mode="nvfp4",
                                        num_tokens=args.capacity, num_topk=args.top_k))
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(
            "--output-dir must be empty to keep artifact provenance unambiguous"
        )
    if any(c.isspace() for c in str(out)):
        parser.error("the CuTe dump directory must not contain whitespace")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["B12X_COMPILE_CACHE_DIR"] = str(out / "compile-cache")
    from multiprocessing import get_context
    from b12x._lib.compile_pool import _initialize_worker
    activity = get_context("spawn").Array("q", (0, 0))
    _initialize_worker(0, (10, 3), "synthetic-sm103-resource-corpus", "NVIDIA B300", 148,
                       227 * 1024, 228 * 1024, activity)
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
            "stage": "quantizer-basis tiles, inline FP16 projections, and uniform, MCG projection-tiered or grouped atom expert MoE",
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
        if args.component in ("residency", "all"):
            from b12x.moe.fused_moe._residency_preparation import _compile_programs
            from b12x.moe.fused_moe._residency_tuning import ResidencyQuery, TUNING
            query = ResidencyQuery(hidden=args.hidden, intermediate=args.intermediate,
                experts=args.experts, hot_experts=args.experts//2 if args.hot_experts is None else args.hot_experts,
                max_tokens=args.capacity, max_top_k=args.top_k, profile_hash="0"*64,
                model_fingerprint="synthetic-compile", gate_first=args.gate_first, swiglu_limit=args.swiglu_limit)
            TUNING.validate_query(query, None)
            manifest["residency_query"] = TUNING.encode_query(query)
            launches.update(_compile_programs(query, offline_dir=out))
        if args.component in ("roce", "all"):
            launches.update(compile_roce(out))
        if args.component in ("trellis", "all"):
            launches.update(compile_trellis(out))
        if args.component == "trellis_clamped":
            launches.update(compile_trellis_clamped(out))
        if args.component in ("sequence", "all"):
            launches.update(compile_sequence(out))
        if args.component in ("mtp_feedback", "all"):
            launches.update(compile_mtp_feedback(out))
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
        if args.component in ("mla_compress", "hyperconnection", "embedding", "all"):
            launches.update(compile_v41_support(out, args.component))
        if args.component in ("wo_projection", "all"):
            launches.update(compile_wo_projection(out))
        if args.component in ("activation_packing", "all"):
            launches.update(compile_activation_packing(out))
        if args.component in ("engram", "all"):
            launches.update(compile_engram(out))
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
