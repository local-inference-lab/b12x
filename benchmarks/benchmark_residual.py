#!/usr/bin/env python3
"""Benchmark native mHC post_pre, including the V4/V4.1 vLLM adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys
import subprocess
from contextlib import ExitStack
from dataclasses import asdict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.common import (
    bench_cuda_graph,
    bench_gpu_ms,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
    nvidia_smi_gpu_mode_snapshot,
)
from b12x.norm import mhc
from b12x.norm.mhc._impl import _b12x_mhc_post_pre_impl, b12x_mhc_post_pre
from b12x.preparation import PreparationSession, PreparedCall, FrozenMapping
from benchmarks.mhc_profiles import MODEL_PROFILES, load_mhc_profile
from b12x._lib.runtime_control import kernel_resolution_guard


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    y_dtype: torch.dtype | None = None,
    pre_mix: torch.Tensor | None = None,
    pre_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    if pre_out is not None:
        pre_out.copy_(pre)
    collapse_mix = pre if pre_mix is None else pre_mix
    y = (collapse_mix.unsqueeze(-1) * residual.float()).sum(dim=1)
    y = y.to(residual.dtype if y_dtype is None else y_dtype)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _post_pre_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    pre_mix: torch.Tensor | None = None,
    pre_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V4 uses Gram variance before BF16 collapse rounding; V4.1's lagged
    input mix is independent of this projection and RMSNorm follows rounding."""
    residual_out = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_fp32, post, comb = _mhc_pre_reference(
        residual_out,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        y_dtype=torch.float32,
        pre_mix=pre_mix,
        pre_out=pre_out,
    )
    if pre_mix is not None:
        y_raw_fp32 = y_raw_fp32.bfloat16().float()
    if norm_weight is not None:
        rms = torch.rsqrt(y_raw_fp32.square().mean(dim=-1, keepdim=True) + norm_eps)
        y = (
            y_raw_fp32.to(torch.bfloat16).float() * rms * norm_weight.float()
        ).to(torch.bfloat16)
    else:
        y = y_raw_fp32.to(torch.bfloat16)
    return residual_out, y, post, comb


def _make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    seed: int,
    device: torch.device,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    residual = (
        torch.randn((tokens, 4, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 3
    ).to(torch.bfloat16)
    x = (
        torch.randn((tokens, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 4
    ).to(torch.bfloat16)
    if weights is None:
        fn = torch.randn((24, 4 * hidden_size), generator=gen, dtype=torch.float32).to(device) / 64
        scale = torch.randn((3,), generator=gen, dtype=torch.float32).to(device) / 3
        bias = torch.randn((24,), generator=gen, dtype=torch.float32).to(device) / 5
    else:
        fn, scale, bias = weights
    return residual.contiguous(), x.contiguous(), fn.contiguous(), scale.contiguous(), bias.contiguous()


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = actual.float() - expected.float()
    return float(diff.abs().max().item()), float(torch.sqrt(torch.mean(diff * diff)).item())


def _bench_graph(fn, *, warmup: int, iters: int, l2_flush, samples_out=None, check=None, cupti=False) -> tuple[float, float]:
    graph = capture_cuda_graph(fn, warmup=warmup)
    try:
        graph.replay()
        if check is not None:
            check()
        if cupti:
            from importlib.metadata import version
            from cupti import cupti as _cupti  # noqa: F401
            from flashinfer.testing import bench_gpu_time_with_cupti

            if int(version("cupti-python").split(".")[0]) < 13:
                raise RuntimeError("CUPTI timing requires cupti-python >= 13")
            samples = [ms * 1000 for ms in bench_gpu_time_with_cupti(
                graph.replay, dry_run_iters=warmup, repeat_iters=iters,
                use_cuda_graph=False, cold_l2_cache=l2_flush is not None,
            )]
        else:
            stats = bench_cuda_graph(graph, replays=iters, l2_flush=l2_flush)
            samples = stats["replay_us"]
        if check is not None:
            check()
        if samples_out is not None:
            samples_out.extend(samples)
        return statistics.median(samples), min(samples)

    finally:
        graph.reset()


def _bench_eager(fn, *, warmup: int, iters: int, l2_flush, samples_out=None, check=None) -> tuple[float, float]:
    samples = []
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        samples.append(bench_gpu_ms(fn, warmup=0, iters=1, l2_flush=l2_flush) * 1000.0)
    if check is not None:
        check()
    if samples_out is not None:
        samples_out.extend(samples)
    return statistics.median(samples), min(samples)


def _register_vllm_mhc_tilelang(vllm_path: pathlib.Path) -> None:
    vllm_path = vllm_path.expanduser().resolve()
    vllm_root = str(vllm_path)
    if vllm_root not in sys.path:
        sys.path.insert(1, vllm_root)

    try:
        import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401
    except (ImportError, OSError) as exc:
        vllm_python = vllm_path / ".venv" / "bin" / "python"
        raise RuntimeError(
            "Failed to import the production vLLM mHC stack. Run this benchmark "
            f"with {vllm_python} so TileLang and DeepGEMM come from the same "
            "environment as vLLM."
        ) from exc

    if not hasattr(torch.ops.vllm, "mhc_fused_post_pre_tilelang"):
        raise RuntimeError("vLLM mhc_fused_post_pre_tilelang custom op was not registered")

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported():
        raise RuntimeError(
            "The production vLLM mHC comparison requires DeepGEMM, but vLLM "
            "reported that DeepGEMM is disabled or unsupported in this environment."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", choices=["custom", *MODEL_PROFILES], default="custom")
    parser.add_argument("--model-path", type=pathlib.Path)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--split-k", type=int)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--block-h", type=int, default=512)
    parser.add_argument("--sinkhorn-iters", type=int)
    parser.add_argument("--rms-eps", type=float)
    parser.add_argument("--hc-eps", type=float)
    parser.add_argument("--norm-eps", type=float)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--cupti", action="store_true", help="Time graph replay with CUPTI; --l2-flush uses FlashInfer's write eviction instead of the default read sweep.")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--fuse-rmsnorm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--expected-m", type=int, help="Planned capture capacity; defaults to the live row count.")
    parser.add_argument("--prefill-bf16-mma", action="store_true")
    parser.add_argument("--prefill-tf32-mma", action="store_true")
    parser.add_argument("--no-prefill-tf32-mma", action="store_true")
    parser.add_argument("--prefill-block-m", action="store_true")
    parser.add_argument("--no-prefill-block-m", action="store_true")
    parser.add_argument("--prefill-block-m-size", type=int, default=2)
    parser.add_argument("--prefill-tile-n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=91_500)
    parser.add_argument("--compare-vllm", action="store_true", help="Custom mode only: compare the TileLang implementation.")
    parser.add_argument("--vllm-path", type=pathlib.Path, help="Checkout for the production adapters or TileLang comparison.")
    parser.add_argument("--output", type=pathlib.Path, help="Write raw timing, correctness and source provenance JSON.")
    args = parser.parse_args()
    if args.cupti and (args.eager or args.l2_flush_bytes):
        parser.error("--cupti requires graph mode and the default L2 flush size")
    with ExitStack() as stack:
        _run_benchmark(args, stack)


def _run_benchmark(args, stack: ExitStack) -> None:
    bundle = None
    if args.model_profile != "custom":
        bundle = load_mhc_profile(args.model_profile, args.model_path, args.layer_idx)
        if args.compare_vllm:
            raise ValueError("Named profiles already run the production vLLM adapter; --compare-vllm is custom-only")
        if args.fuse_rmsnorm is False:
            raise ValueError("The vLLM mHC profiles include fused RMSNorm")
        if args.prefill_bf16_mma and bundle.profile.lagged:
            raise ValueError("V4.1's adapter retains native FP32 projection weights")
    cfg = bundle.config if bundle is not None else {}
    defaults = {
        "hidden_size": cfg.get("hidden_size", 4096),
        "sinkhorn_iters": cfg.get("hc_sinkhorn_iters", 20),
        "rms_eps": cfg.get("rms_norm_eps", 1e-6),
        "hc_eps": cfg.get("hc_eps", 1e-6),
        "norm_eps": cfg.get("rms_norm_eps", 1e-6),
        "fuse_rmsnorm": bundle is not None,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
        elif bundle is not None and getattr(args, name) != value:
            raise ValueError(f"--{name.replace('_', '-')} contradicts the checkpoint integration contract")
    if args.tokens < 1 or args.iters < 1 or args.warmup < 1:
        raise ValueError("tokens, iters, and warmup must be positive")
    if args.expected_m is not None and args.expected_m < args.tokens:
        raise ValueError("--expected-m must cover --tokens")
    if bundle is not None and not bundle.profile.lagged and args.expected_m not in (None, args.tokens):
        raise ValueError("The V4.0 adapter sets expected_m from its live tensor shape")
    if args.vllm_path is None:
        args.vllm_path = pathlib.Path("~/projects/vllm-hh-rebase" if bundle is not None else "~/projects/vllm")
    args.vllm_path = args.vllm_path.expanduser().resolve()
    if args.no_prefill_tf32_mma:
        os.environ["B12X_MHC_PREFILL_TF32_MMA"] = "0"
    elif args.prefill_tf32_mma:
        os.environ["B12X_MHC_PREFILL_TF32_MMA"] = "1"
    if args.prefill_bf16_mma:
        os.environ["B12X_MHC_PREFILL_BF16_MMA"] = "1"
    if args.no_prefill_block_m:
        os.environ["B12X_MHC_PREFILL_BLOCK_M"] = "0"
    elif args.prefill_block_m:
        os.environ["B12X_MHC_PREFILL_BLOCK_M"] = "1"
    if args.prefill_block_m or (bundle is None and not args.no_prefill_block_m):
        os.environ["B12X_MHC_PREFILL_BLOCK_M_SIZE"] = str(args.prefill_block_m_size)
        os.environ["B12X_MHC_PREFILL_TILE_N"] = str(args.prefill_tile_n)

    device = require_sm120()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    stack.callback(setattr, torch.backends.cuda.matmul, "allow_tf32", previous_tf32)
    tensors = (
        {name: tensor.to(device=device).contiguous() for name, tensor in bundle.tensors.items()}
        if bundle is not None else {}
    )
    residual, x, fn, scale, bias = _make_inputs(
        tokens=args.tokens, hidden_size=args.hidden_size, seed=args.seed, device=device,
        weights=(tensors["fn"], tensors["scale"], tensors["bias"]) if tensors else None,
    )
    lagged = bundle is not None and bundle.profile.lagged
    pre_mix = torch.empty((args.tokens, 4), dtype=torch.float32, device=device) if lagged else None
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual, tensors.get("prev_fn", fn), tensors.get("prev_scale", scale),
        tensors.get("prev_bias", bias), rms_eps=args.rms_eps, hc_eps=args.hc_eps,
        sinkhorn_iters=args.sinkhorn_iters, pre_out=pre_mix,
    )
    prev_post, prev_comb = prev_post.contiguous(), prev_comb.contiguous()
    norm_weight = tensors.get("norm_weight")
    if args.fuse_rmsnorm and norm_weight is None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
        norm_weight = torch.randn(args.hidden_size, generator=generator).to(device=device, dtype=torch.bfloat16)
    fn_bf16 = (
        fn.bfloat16().contiguous()
        if args.prefill_bf16_mma or (bundle is not None and not lagged) else None
    )
    fused_out = fused_post = fused_comb = fused_y = fused_pre = None
    runner = None
    planned_config = None
    session = None
    if bundle is not None:
        from benchmarks.vllm_mhc import vllm_mhc_runner

        capture_sizes = (
            (args.expected_m,) if args.expected_m is not None
            else tuple(sorted({1, 2, 4, 8, args.tokens}))
        )
        runner = stack.enter_context(vllm_mhc_runner(
            args.vllm_path, profile_name=args.model_profile, model_config=cfg,
            capture_sizes=capture_sizes,
        ))
        runner.prepare_weights(tensors)
        # The integration owns the split and capture-capacity decisions.
        required_split = 4 * args.hidden_size // args.block_k
        if args.split_k is not None and args.split_k != required_split:
            raise ValueError("--split-k contradicts the vLLM mHC integration")
        if args.block_k != 256 or args.block_h != 512:
            raise ValueError("Named profiles use the integration's block_k=256/block_h=512")
        args.split_k = required_split
    else:
        if args.compare_vllm:
            _register_vllm_mhc_tilelang(args.vllm_path)
        caps = mhc.Caps(
            device=device, max_tokens=max(args.tokens, args.expected_m or args.tokens),
            hidden_size=args.hidden_size, split_k=args.split_k,
        )
        fused_plan = mhc.plan(caps, invocation=FrozenMapping({
            "operation": "post_pre", "output_mode": "provided",
            "has_norm_weight": norm_weight is not None,
            "norm_weight_dtype": "bfloat16" if norm_weight is None else str(norm_weight.dtype).removeprefix("torch."),
            "has_fn_bf16": fn_bf16 is not None, "rms_eps": args.rms_eps,
            "hc_eps": args.hc_eps, "sinkhorn_iters": args.sinkhorn_iters, "norm_eps": args.norm_eps,
            "block_k": args.block_k, "block_h": args.block_h,
        }))
        args.split_k = caps.split_k
        session = stack.enter_context(PreparationSession(device=device, autotune=False, compile_workers=2))

        def prime(state):
            (spec,) = state.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(
                scratch=scratch, tokens=args.tokens, expected_m=args.expected_m,
                y=torch.empty_like(x), post=torch.empty_like(prev_post),
                comb=torch.empty_like(prev_comb), out=torch.empty_like(residual),
            )
            return PreparedCall(run=lambda: _b12x_mhc_post_pre_impl(
                x, residual, prev_post, prev_comb, fn, scale, bias,
                rms_eps=args.rms_eps, hc_eps=args.hc_eps, sinkhorn_iters=args.sinkhorn_iters,
                norm_weight=norm_weight, norm_eps=args.norm_eps, fn_bf16=fn_bf16,
                binding=binding, _state=state,
            ), owners=(scratch, binding))

        session.prepare((fused_plan.request(name="residual-mhc", prepare_call=prime),))
        planned_config = fused_plan.prepared.state.config
        fused_scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                              for spec in fused_plan.scratch_specs())
        fused_y = torch.empty((args.tokens, args.hidden_size), dtype=torch.bfloat16, device=device)
        fused_post = torch.empty((args.tokens, 4), dtype=torch.float32, device=device)
        fused_comb = torch.empty((args.tokens, 4, 4), dtype=torch.float32, device=device)
        fused_out = torch.empty((args.tokens, 4, args.hidden_size), dtype=torch.bfloat16, device=device)
        fused_binding = mhc.bind(fused_plan,
            scratch=fused_scratch, tokens=args.tokens, expected_m=args.expected_m,
            y=fused_y, post=fused_post, comb=fused_comb, out=fused_out,
        )

    def run_fused():
        nonlocal fused_out, fused_post, fused_comb, fused_y, fused_pre
        if runner is not None:
            fused_out, fused_post, fused_comb, fused_y, fused_pre = runner.run(
                x, residual, prev_post, prev_comb, fn, scale, bias, norm_weight,
                pre_mix=pre_mix, fn_bf16=fn_bf16,
            )
        else:
            b12x_mhc_post_pre(
                x, residual, prev_post, prev_comb, fn, scale, bias,
                rms_eps=args.rms_eps, hc_eps=args.hc_eps, sinkhorn_iters=args.sinkhorn_iters,
                norm_weight=norm_weight, norm_eps=args.norm_eps, fn_bf16=fn_bf16,
                binding=fused_binding,
            )

    vllm_out = vllm_post = vllm_comb = vllm_y = None

    def run_vllm_fused():
        nonlocal vllm_out, vllm_post, vllm_comb, vllm_y
        vllm_out, vllm_post, vllm_comb, vllm_y = torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, residual, prev_post, prev_comb, fn, scale, bias,
            args.rms_eps, args.hc_eps, args.hc_eps, 2.0, args.sinkhorn_iters, 1, 1,
            norm_weight, args.norm_eps if norm_weight is not None else 0.0,
        )

    run_fused()
    if args.compare_vllm:
        run_vllm_fused()
    torch.cuda.synchronize()
    errors = {}
    expected_pre = torch.empty_like(pre_mix) if pre_mix is not None else None
    expected = None
    if not args.skip_check:
        expected = _post_pre_reference(
            x, residual, prev_post, prev_comb, fn, scale, bias,
            rms_eps=args.rms_eps, hc_eps=args.hc_eps, sinkhorn_iters=args.sinkhorn_iters,
            norm_weight=norm_weight, norm_eps=args.norm_eps, pre_mix=pre_mix, pre_out=expected_pre,
        )

    def check():
        if expected is None:
            return
        out_ref, y_ref, post_ref, comb_ref = expected
        pairs = {
            "out": (fused_out, out_ref), "y": (fused_y, y_ref),
            "post": (fused_post, post_ref), "comb": (fused_comb, comb_ref),
        }
        if expected_pre is not None:
            pairs["pre"] = (fused_pre, expected_pre)
        for name, (actual, reference) in pairs.items():
            errors[name] = dict(zip(("max_abs", "rmse"), _error_stats(actual, reference), strict=True))
            if bundle is not None:
                torch.testing.assert_close(
                    actual, reference, rtol=.01 if actual.dtype == torch.bfloat16 else 2e-5,
                    atol=.008 if actual.dtype == torch.bfloat16 else 4e-5,
                )
                if not torch.isfinite(actual).all():
                    raise AssertionError(f"nonfinite mHC {name}")
                if reference.count_nonzero() and not actual.count_nonzero():
                    raise AssertionError(f"zero mHC {name} for nonzero reference")

    check()
    if runner is not None:
        runner.lock_workspace()
    if session is not None:
        session.freeze()
    else:
        stack.enter_context(kernel_resolution_guard("mHC benchmark capture and replay"))
    l2_flush = make_l2_flush_fn(args.l2_flush, args.l2_flush_bytes)
    samples = []
    gpu_before = nvidia_smi_gpu_mode_snapshot() if args.output else None
    bench = _bench_eager if args.eager else _bench_graph
    fused_median, fused_min = bench(
        run_fused, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush,
        samples_out=samples, check=check,
        **({"cupti": args.cupti} if not args.eager else {}),
    )
    vllm_median = vllm_min = None
    if args.compare_vllm:
        vllm_median, vllm_min = bench(
            run_vllm_fused, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush,
            **({"cupti": args.cupti} if not args.eager else {}),
        )
    mode = "eager" if args.eager else "graph"
    print(
        f"residual_mhc profile={args.model_profile} mode={mode} tokens={args.tokens} "
        f"hidden={args.hidden_size} split_k={args.split_k} fused_rmsnorm={args.fuse_rmsnorm} "
        f"lagged={lagged} "
        f"layer={args.layer_idx} boundary=attn_post_ffn_pre "
        f"post_pre_us={fused_median:.2f}/{fused_min:.2f} errors={json.dumps(errors, sort_keys=True)}"
    )
    if vllm_median is not None:
        print(f"vllm_post_pre_us={vllm_median:.2f}/{vllm_min:.2f} b12x_vs_vllm={vllm_median/fused_median:.3f}x")
    if args.output:
        root = pathlib.Path(__file__).resolve().parents[1]
        source_paths = [
            pathlib.Path(__file__), root / "benchmarks/mhc_profiles.py",
            root / "benchmarks/vllm_mhc.py", root / "benchmarks/common.py",
            *(root / "b12x/norm/mhc").glob("*.py"),
        ]
        if runner is not None:
            source_paths.extend(runner.integration_files)
        result = {
            "command": [sys.executable, *sys.argv], "worktree": str(root),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "profile": args.model_profile, "model_path": str(bundle.model_path) if bundle else None,
            "layer_idx": args.layer_idx, "boundary": "attention post + FFN pre + RMSNorm",
            "tokens": args.tokens, "hidden_size": args.hidden_size, "lagged": lagged,
            "input_dtype": str(residual.dtype), "fn_dtype": str(fn.dtype),
            "fused_rmsnorm": args.fuse_rmsnorm,
            "rms_eps": args.rms_eps, "norm_eps": args.norm_eps, "hc_eps": args.hc_eps,
            "sinkhorn_iters": args.sinkhorn_iters, "split_k": args.split_k,
            "timing_mode": mode, "l2_flush": args.l2_flush, "warmup": args.warmup,
            "timing_backend": "cupti" if args.cupti else "cuda_events",
            "l2_eviction": (
                "flashinfer_zero_write" if args.cupti else "read_sweep"
            ) if args.l2_flush else "none",
            "samples_us": samples, "median_us": fused_median, "min_us": fused_min,
            "correctness": (
                "unchecked" if args.skip_check else
                "oracle and replay checked" if bundle is not None else "oracle metrics only"
            ),
            "errors": errors, "gpu_before": gpu_before, "gpu_after": nvidia_smi_gpu_mode_snapshot(),
            "config": asdict(planned_config) if planned_config is not None else None,
            "integration": runner.provenance if runner is not None else None,
            "weight_sha256": {
                name: hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, tensor in bundle.tensors.items()
            } if bundle is not None else None,
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
