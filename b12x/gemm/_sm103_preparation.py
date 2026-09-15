"""SM103 dense lowering and retained native launchers for startup preparation."""
from __future__ import annotations



def config_for(query):
    from ._tuning import DenseGemmConfig, launch_options

    ordinary = query.recipe in {"tensor_fp8", "block_fp8"}
    swap = ordinary and query.out_features * 2 % 128 != 0
    tile = ((64, 16) if swap else (16, 64)) if ordinary else (128, 128)
    config = DenseGemmConfig(
        backend="sm103", tile_m=tile[0], tile_n=tile[1], tile_k=128,
        load_path="tma", swap_ab=swap, split_k_slices=1,
        large_m_unroll=False, target_occupancy=1,
    )
    if query.in_features % 128 or (not ordinary and query.out_features % 8):
        raise ValueError("SM103 dense preparation requires K128 and N8")
    if query.max_rows >= 2**31 or query.batch > 65535:
        raise ValueError("SM103 dense capacity exceeds its launch grid")
    if ordinary and swap and query.batch != 1:
        raise ValueError("unaligned FP8 output rows require a single group")
    options = launch_options(query, config)
    if any(options.get(name) != value for name, value in query.overrides.items()):
        raise ValueError("SM103 dense override differs from the implemented launch")
    return config


def lower(query, device):
    from b12x._lib.dense_gemm import _DenseGemmPolicy, _DenseLowering
    from ._tuning import operand_options

    config = config_for(query)
    options = operand_options(query)
    fp6 = query.recipe.startswith(("mxfp6_", "w6a8_"))
    fp4 = query.recipe in {"nvfp4", "mxfp4"}
    k = query.in_features
    return _DenseLowering(
        m=query.max_rows, n=query.out_features, k=k, l=query.batch,
        a_storage_k=k // 2 if fp4 else k,
        b_storage_k=3 * k // 4 if fp6 and query.weight_storage == "packed" else k // 2 if fp4 else k,
        ab_dtype=options["ab_dtype"], sf_dtype=options["sf_dtype"],
        c_dtype=query.output_dtype, alpha_dtype="float32", sf_vec_size=options["sf_vec_size"],
        sm_count=device.sm_count, mma_k=64 if fp4 else 32, tile_k=128,
        mma_tiler_mn=(config.tile_m, config.tile_n), cluster_shape_mn=(1, 1),
        policy=_DenseGemmPolicy(True, True, False, 1, False, False),
        load_path="tma", swap_ab=config.swap_ab, expected_m=query.expected_m,
        kernel_c_l=query.batch, alpha_is_one=query.alpha_mode == "unit",
        is_mxfp6=fp6, mxfp6_fmt_a=options.get("a_fmt"), mxfp6_fmt_b=options.get("b_fmt"),
        a_preexpanded=options.get("a_preexpanded", False),
        b_preexpanded=options.get("b_preexpanded", False), b_packed=options.get("b_packed", False),
        plain_fp8=query.recipe == "tensor_fp8", block_fp8=query.recipe == "block_fp8",
        sfb_k_reuse=False, b_tile_major=False, quantize_c=False, fused_quant=False,
        row_scale=False, output_provided=query.output_mode == "provided",
        target_occupancy_override=1, direct_sfa_live16=False, direct_m1_wo_a_inputs=False,
        architecture="sm103",
    )


def compile_dense(p, ordinal):
    if p.is_mxfp6:
        from .blockscaled._fp6 import compile_kernel
        return {"gemm": compile_kernel(
            p.n, p.k, p.l, p.mxfp6_fmt_a, p.mxfp6_fmt_b,
            p.a_preexpanded, p.b_preexpanded, p.c_dtype, p.alpha_is_one, False, ordinal,
        )}
    if p.plain_fp8 or p.block_fp8:
        from b12x._lib.fp8_gemm import compile_kernel
        return {"gemm": compile_kernel(
            p.n, p.k, p.l, p.c_dtype, p.block_fp8, p.alpha_is_one,
            ordinal, p.sm_count, "sm_103a",
        )}
    from .blockscaled._sm103 import RECIPES, compile_kernel
    return {"gemm": compile_kernel(
        p.n, p.k, p.l, RECIPES[(p.ab_dtype, p.sf_dtype, p.sf_vec_size)],
        p.c_dtype, ordinal, p.alpha_is_one,
    )}


def run_dense(state, lhs, rhs, out, *, alpha, stream):
    p = state.lowering
    a, b = lhs[0], rhs[0]
    if (a.ndim != 3 or b.ndim != 3 or a.device != state.device or b.device != state.device
            or tuple(a.shape[1:]) != (p.a_storage_k, p.l)
            or tuple(b.shape) != (p.n, p.b_storage_k, p.l)):
        raise ValueError("SM103 dense operands differ from the prepared geometry/device")
    if lhs[0].shape[0] > p.m:
        raise ValueError("SM103 dense execution exceeds its prepared row capacity")
    if (alpha is None) != p.alpha_is_one:
        raise ValueError("SM103 dense alpha presence differs from preparation")
    if p.is_mxfp6:
        from .blockscaled._fp6 import execute
        return execute(lhs, rhs, out, ab_dtype=p.ab_dtype, sf_dtype=p.sf_dtype,
                       sf_vec_size=p.sf_vec_size, c_dtype=p.c_dtype,
                       a_fmt=p.mxfp6_fmt_a, b_fmt=p.mxfp6_fmt_b,
                       a_preexpanded=p.a_preexpanded, b_preexpanded=p.b_preexpanded,
                       b_packed=p.b_packed, alpha=alpha, stream=stream, compiled=state.gemm)
    if p.plain_fp8 or p.block_fp8:
        from .blockscaled._fp8_cute import execute
        return execute(lhs, rhs, out, ab_dtype=p.ab_dtype, sf_dtype=p.sf_dtype,
                       sf_vec_size=p.sf_vec_size, c_dtype=p.c_dtype, block_fp8=p.block_fp8,
                       alpha=alpha, stream=stream, compiled=state.gemm)
    from .blockscaled._sm103 import execute
    return execute(lhs, rhs, out, ab_dtype=p.ab_dtype, sf_dtype=p.sf_dtype,
                   sf_vec_size=p.sf_vec_size, c_dtype=p.c_dtype,
                   alpha=alpha, stream=stream, compiled=state.gemm)
