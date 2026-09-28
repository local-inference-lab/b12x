"""Model-derived native sparse-MLA profiles; prepared Q/K, not a model forward."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from b12x.preparation import PreparationSession
from b12x.attention.compressed_sparse_mla._tuning import TUNING, _split_config
from benchmarks.attention_preparation import prepare_compressed
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
    pack_deepseek_v41_cache_reference,
)
from benchmarks.benchmark_v41_attention import tensor_hash
from benchmarks.benchmark_v41_serving import repository_state
from benchmarks.common import nvidia_smi_gpu_mode_snapshot
from benchmarks.deepseek_attention_profiles import (
    MODEL_REPOS,
    integration_provenance,
    load_attention_profile,
)


def _csv(value):
    return [int(part) for part in value.split(",")]


def _pack(profile, values, page_size, kind):
    if profile.cache_format == "deepseek_v41":
        return pack_deepseek_v41_cache_reference(
            values, page_size=page_size, cache_kind=kind
        )
    return pack_compressed_sparse_mla_kv_cache_reference(
        values[:, :448], values[:, 448:], page_size=page_size
    )


@torch.inference_mode()
def run_form(args, profile, form):
    device = torch.device("cuda", args.device)
    query_width = args.speculative_tokens if form.draft else args.query_tokens
    if query_width < 1 or query_width > args.context_tokens:
        raise ValueError("query token count must fit the live context")
    row_counts = args.query_rows or [batch * query_width for batch in args.batch_sizes]
    capacities = {
        mode: args.planned_decode_rows if mode == "decode" else args.planned_extend_rows
        for mode in args.modes
    }
    storage_rows = max(capacities.values())
    if max(row_counts) > min(capacities.values()):
        raise ValueError("live rows exceed a requested mode's declared capacity")
    requests = math.ceil(max(row_counts) / query_width)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    swa_page, indexed_page = form.swa_page_size, form.indexed_page_size
    window = profile.config["sliding_window"]
    # Include the page containing the beginning of the retained local window.
    first_visible = args.context_tokens - query_width + (0 if form.draft else 1)
    swa_start = max(0, first_visible - window) // swa_page * swa_page
    swa_pages_per_request = (
        math.ceil(args.context_tokens / swa_page) - swa_start // swa_page
    )
    swa_tokens = swa_pages_per_request * swa_page
    indexed_pages_per_request = (
        math.ceil(args.context_tokens / max(form.ratio, 1) / indexed_page)
        if form.ratio
        else 0
    )
    indexed_tokens = indexed_pages_per_request * indexed_page
    planned_pages = math.ceil(args.max_model_len / profile.block_size)
    indexed_width = form.indexed_width
    if indexed_width is None:
        indexed_width = (
            math.ceil(math.ceil(args.max_model_len / form.ratio) / 128) * 128
        )
    scales = torch.linspace(0.03, 0.7, 32, device=device).repeat_interleave(16)
    swa_values = (
        torch.randn((requests * swa_tokens, 512), generator=generator, device=device)
        * scales
    ).bfloat16()
    swa_reference = _pack(profile, swa_values, swa_page, "swa")
    indexed_reference = None
    if indexed_tokens:
        values = (
            torch.randn(
                (requests * indexed_tokens, 512), generator=generator, device=device
            )
            * scales.flip(0)
            * 3
        ).bfloat16()
        indexed_reference = _pack(profile, values, indexed_page, "indexed")
    swa_bytes = swa_reference.shape[1]
    indexed_bytes = indexed_reference.shape[1] if indexed_reference is not None else 0
    stride = args.page_stride or math.ceil(max(swa_bytes, indexed_bytes) / 256) * 256
    if stride < max(swa_bytes, indexed_bytes) or stride % 16:
        raise ValueError(
            "allocator page stride must fit both native payloads and be 16-byte aligned"
        )
    swa_base = 2**31 // stride + 1 if args.high_pid else 1
    indexed_base = swa_base + swa_reference.shape[0]
    total_pages = indexed_base + (
        indexed_reference.shape[0] if indexed_reference is not None else 0
    )
    if torch.cuda.mem_get_info(device)[0] < total_pages * stride + (2 << 30):
        raise RuntimeError(
            "insufficient VRAM for the high-pid pool and qualification workspace"
        )
    pool = torch.empty(total_pages * stride, dtype=torch.uint8, device=device)
    swa = pool.as_strided((total_pages, swa_bytes), (stride, 1))
    swa[swa_base:indexed_base].copy_(swa_reference)
    indexed = None
    if indexed_reference is not None:
        indexed = pool.as_strided((total_pages, indexed_bytes), (stride, 1))
        indexed[indexed_base:total_pages].copy_(indexed_reference)
    # Each measured request has its own native cache pages. Warmup rows can
    # repeat those requests without allocating capacity-sized context pools.
    row = torch.arange(storage_rows, device=device, dtype=torch.int64)
    req = (row // query_width) % requests
    positions = args.context_tokens - query_width + row % query_width
    q = (
        torch.randn(
            (storage_rows, profile.attention_heads, 512),
            generator=generator,
            device=device,
        )
        * 0.2
    ).bfloat16()
    if args.query_rms is not None:
        if args.query_rms <= 0:
            raise ValueError("query RMS must be positive")
        q.copy_(
            (
                q.float()
                * torch.rsqrt(q.float().square().mean(-1, keepdim=True))
                * args.query_rms
            ).bfloat16()
        )
    q[:, 0, :448].zero_()
    q[:, 0, 448:].mul_(4)
    if form.draft:
        start = torch.full_like(
            positions, max(0, args.context_tokens - query_width - window)
        )
        lengths = torch.full_like(
            positions, min(args.context_tokens, window + query_width)
        )
    else:
        start = (positions + 1 - window).clamp_min(0)
        lengths = positions + 1 - start
    local = (
        start[:, None] - swa_start + torch.arange(form.swa_width, device=device)[None]
    )
    swa_indices = (swa_base * swa_page + req[:, None] * swa_tokens + local).int()
    swa_lengths = lengths.int()
    swa_indices.masked_fill_(
        torch.arange(form.swa_width, device=device)[None] >= swa_lengths[:, None], -1
    )
    visible = ((positions + 1) // max(form.ratio, 1)).int()
    logical = None
    indexed_lengths = None
    table = None
    if indexed_tokens:
        if form.indexed_width is None:
            logical = (
                torch.arange(indexed_width, device=device)
                .expand(storage_rows, -1)
                .clone()
                .int()
            )
        else:
            selected = torch.full(
                (requests * query_width, indexed_width),
                -1,
                device=device,
                dtype=torch.int64,
            )
            for request in range(requests):
                for position in range(query_width):
                    available = (
                        args.context_tokens - query_width + position + 1
                    ) // form.ratio
                    count = min(indexed_width, available)
                    chosen = torch.randperm(
                        available,
                        generator=generator,
                        device=device,
                    )[:count]
                    if profile.cache_format == "deepseek_v41":
                        chosen = chosen.sort().values
                    selected[request * query_width + position, :count] = chosen
            logical = selected[row % (requests * query_width)].int()
        indexed_lengths = torch.minimum(
            visible, torch.full_like(visible, indexed_width)
        )
        logical.masked_fill_(
            torch.arange(indexed_width, device=device)[None]
            >= indexed_lengths[:, None],
            -1,
        )
        page_columns = torch.arange(planned_pages, device=device)[None]
        table = (
            indexed_base + req[:, None] * indexed_pages_per_request + page_columns
        ).int()
        table.masked_fill_(page_columns >= indexed_pages_per_request, -1)
    sink = torch.linspace(-0.3, 0.3, profile.attention_heads, device=device)
    out = torch.empty_like(q)

    def oracle(rows, fp8=False, mode="decode"):
        physical_swa = swa_indices[:rows] - swa_base * swa_page
        physical_main = None
        if logical is not None:
            physical_main = req[:rows, None] * indexed_tokens + logical[:rows]
        if fp8:
            from tests._reference.v41_fp8 import canonical_fp8_rows, split64_fp8_attention

            source = swa_reference.reshape(-1, 528)[physical_swa.long().clamp_min(0)]
            values, scales = canonical_fp8_rows(source, "swa")
            key_values = [values.reshape(rows, form.swa_width, 512)]
            key_scales = [scales.reshape(rows, form.swa_width, 8)]
            columns = torch.arange(form.swa_width, device=device)[None]
            valid = [(columns < swa_lengths[:rows, None]) & (physical_swa >= 0)]
            if physical_main is not None:
                source = indexed_reference.reshape(-1, 288)[physical_main.long().clamp_min(0)]
                values, scales = canonical_fp8_rows(source, "indexed")
                key_values.append(values.reshape(rows, indexed_width, 512))
                key_scales.append(scales.reshape(rows, indexed_width, 8))
                columns = torch.arange(indexed_width, device=device)[None]
                valid.append((columns < indexed_lengths[:rows, None]) & (logical[:rows] >= 0))
            return split64_fp8_attention(
                q[:rows], torch.cat(key_values, 1), torch.cat(key_scales, 1),
                torch.cat(valid, 1), 512**-0.5, attn_sink=sink,
                qk_fp8=mode == "decode",
                round_split_outputs=mode == "decode",
            )[0]
        return compressed_sparse_mla_reference(
            q[:rows],
            swa_reference,
            physical_swa,
            swa_lengths[:rows],
            extra_k_cache=indexed_reference,
            extra_indices=physical_main,
            extra_topk_lengths=indexed_lengths[:rows]
            if indexed_lengths is not None
            else None,
            swa_page_size=swa_page,
            extra_page_size=indexed_page,
            sm_scale=512**-0.5,
            attn_sink=sink,
            cache_format=profile.cache_format,
        )

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    plans, scratch = {}, {}
    v41 = profile.cache_format == "deepseek_v41"
    # V4's integration plans each declared capture shape; V4.1 plans a fixed
    # scheduler capacity. Do not disguise those distinct planner contracts.
    plan_keys = [
        (mode, capacity)
        for mode, limit in capacities.items()
        for capacity in ([limit] if v41 else row_counts)
    ]
    for key in plan_keys:
        mode, capacity = key
        if v41:
            caps = mla.Caps(
                device=device,
                num_q_heads=profile.attention_heads,
                max_q_rows=capacity,
                max_width=form.swa_width + indexed_width,
                swa_width=form.swa_width,
                indexed_width=indexed_width,
                swa_page_size=swa_page,
                indexed_page_size=indexed_page,
                max_page_table_width=planned_pages,
                cache_format=profile.cache_format,
                mode=mode,
                use_cuda_graph=True,
            )
        else:
            decode_capacity = args.planned_decode_rows if mode == "decode" else None
            chunks = mla.split_chunks_for_contract(
                rows=capacity,
                width=form.swa_width + indexed_width,
                decode_row_capacity=decode_capacity,
            )
            caps = mla.Caps(
                device=device,
                num_q_heads=profile.attention_heads,
                max_q_rows=capacity,
                max_width=form.swa_width + indexed_width,
                mode=mode, swa_width=form.swa_width, indexed_width=indexed_width,
                swa_page_size=swa_page, indexed_page_size=indexed_page,
                max_page_table_width=planned_pages, cache_format=profile.cache_format,
                max_chunks_per_row=chunks,
                decode_row_capacity=decode_capacity,
            )
        bind_args = dict(q=q[:capacity], swa_indices=swa_indices[:capacity], swa_lengths=swa_lengths[:capacity])
        if logical is not None:
            if v41:
                indices = logical[:capacity]
                bind_args["indexed_page_table"] = table[:capacity]
            else:
                indices = (indexed_base * indexed_page + req[:capacity, None] * indexed_tokens + logical[:capacity]).int()
            bind_args.update(indexed_indices=indices, indexed_lengths=indexed_lengths[:capacity])
        run_args = dict(swa_k_cache=swa, indexed_k_cache=indexed, sm_scale=512**-0.5,
                        attn_sink=sink, out=out[:capacity])
        invocation = mla.invocation_from_tensors(q=q[:capacity], swa_k_cache=swa,
                                                indexed_k_cache=indexed, attn_sink=sink, out=out[:capacity])
        base_plan = mla.plan(caps, invocation=invocation)
        default = TUNING.configure(base_plan.query, device=session.device.identity).default
        if caps.max_chunks_per_row is not None and not default.single_pass:
            split = _split_config(base_plan.query, caps.max_chunks_per_row)
            default = replace(default, max_chunks_per_row=split.num_chunks, split_chunk_size=split.chunk_size)
        for compute in args.compute_modes:
            config = default
            if compute != "auto":
                config = replace(default,
                    v41_compute_mode="bf16" if compute == "bf16" else "fp8",
                    v41_heads_per_block=(8 if compute == "fp8-h8" and mode == "decode" else 16))
            arm_key = (mode, capacity, compute)
            selected_plan, binding = prepare_compressed(session, caps,
                bind_args=bind_args, run_args=run_args, config=config)
            plans[arm_key] = selected_plan
            scratch[arm_key] = (binding.scratch.shared_scratch,)

    def bind(mode, rows, compute="auto"):
        key = (mode, capacities[mode] if v41 else rows, compute)
        kwargs = {}
        if logical is not None:
            if profile.cache_format == "deepseek_v41":
                indices = logical[:rows]
                kwargs["indexed_page_table"] = table[:rows]
            else:
                indices = (
                    indexed_base * indexed_page
                    + req[:rows, None] * indexed_tokens
                    + logical[:rows]
                ).int()
            kwargs.update(
                indexed_indices=indices, indexed_lengths=indexed_lengths[:rows]
            )
        binding = mla.bind(
            plans[key],
            scratch=scratch[key],
            q=q[:rows],
            swa_indices=swa_indices[:rows],
            swa_lengths=swa_lengths[:rows],
            **kwargs,
        )
        return binding

    def execute(binding, rows):
        return mla.run(
            binding=binding,
            swa_k_cache=swa,
            swa_page_size=swa_page,
            indexed_k_cache=indexed,
            indexed_page_size=indexed_page,
            sm_scale=512**-0.5,
            attn_sink=sink,
            out=out[:rows],
            cache_format=profile.cache_format,
        )

    for mode, capacity in plan_keys:
        for compute in args.compute_modes:
            execute(bind(mode, capacity, compute), capacity)
    torch.cuda.synchronize(device)
    session.freeze()

    def validate_reference(actual, expected, fp8):
        assert bool(torch.isfinite(actual).all())
        tolerance = 0.008 if fp8 else 0.035
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        reference_norm = expected.float().norm()
        if reference_norm.item() == 0:
            assert bool(actual.eq(0).all())
        else:
            error = (actual.float() - expected.float()).norm() / reference_norm
            limit = tolerance
            assert error.item() < limit, (error.item(), limit)

    records = []
    graphs, timing_graphs = {}, {}
    try:
        for rows in row_counts:
            for graph in (*graphs.values(), *timing_graphs.values()):
                graph.reset()
            timing_graphs = {}
            expected = oracle(rows)
            graphs, bindings, cosine, relative_l2, active_splits = {}, {}, {}, {}, {}
            arm_modes, arm_keys = {}, {}
            eager_outputs = {}
            arm_fp8 = {}
            for mode, compute in [
                (mode, compute) for mode in capacities for compute in args.compute_modes
            ]:
                arm = mode if compute == "auto" else f"{mode}-{compute}"
                binding = bind(mode, rows, compute)
                bindings[arm] = binding
                arm_modes[arm] = mode
                arm_keys[arm] = (mode, capacities[mode] if v41 else rows, compute)
                for buffer in scratch[arm_keys[arm]]:
                    buffer.fill_(255)
                out[:rows].fill_(float("nan"))
                actual = execute(binding, rows)
                arm_fp8[arm] = v41 and (
                    plans[arm_keys[arm]].prepared.state.config.v41_compute_mode == "fp8"
                )
                active_splits[arm] = int(binding.scratch.num_chunks_ptr.item())
                eager_outputs[arm] = actual.clone()
                validate_reference(actual, oracle(rows, fp8=True, mode=mode) if arm_fp8[arm] else expected, arm_fp8[arm])
                assert bool(torch.isfinite(actual).all()) and bool(
                    torch.count_nonzero(actual)
                )
                cos = (
                    torch.nn.functional.cosine_similarity(
                        actual.float().reshape(rows, -1),
                        expected.float().reshape(rows, -1),
                    )
                    .min()
                    .item()
                )
                if not arm_fp8[arm]:
                    assert cos > 0.999, cos
                cosine[arm] = cos
                error = (
                    actual.float() - expected.float()
                ).norm() / expected.float().norm()
                if args.relative_error_limit is not None:
                    assert error.item() < args.relative_error_limit, (error.item(), args.relative_error_limit)
                relative_l2[arm] = error.item()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    execute(binding, rows)
                graphs[arm] = graph
            for arm, graph in graphs.items():
                graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(
                    out[:rows], eager_outputs[arm], rtol=0, atol=0
                )
            original_swa = swa_lengths[0].clone()
            original_main = (
                indexed_lengths[0].clone() if indexed_lengths is not None else None
            )
            original_q = q[:rows].clone()
            swa_lengths[0] = 0
            if indexed_lengths is not None:
                indexed_lengths[0] = 0
            q[:rows].mul_(0.7)
            mutated = oracle(rows)
            for arm, graph in graphs.items():
                out[:rows].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                validate_reference(out[:rows], oracle(rows, fp8=True, mode=arm_modes[arm]) if arm_fp8[arm] else mutated, arm_fp8[arm])
                assert bool(out[0].eq(0).all())
            swa_lengths[0].copy_(original_swa)
            if indexed_lengths is not None:
                indexed_lengths[0].copy_(original_main)
            q[:rows].copy_(original_q)
            expected = oracle(rows)
            for arm, graph in graphs.items():
                graph.replay()
                torch.cuda.synchronize(device)
                validate_reference(out[:rows], oracle(rows, fp8=True, mode=arm_modes[arm]) if arm_fp8[arm] else expected, arm_fp8[arm])
                for _ in range(args.warmup):
                    graph.replay()
            # Capture repeated native invocations in one graph. A Python loop
            # over tiny graph launches can measure host submission gaps instead
            # of the GPU work we are comparing.
            timing_graphs = {}
            for arm, binding in bindings.items():
                batch_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(batch_graph):
                    for _ in range(args.replays):
                        execute(binding, rows)
                batch_graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(
                    out[:rows], eager_outputs[arm], rtol=0, atol=0
                )
                timing_graphs[arm] = batch_graph
            torch.cuda.synchronize(device)
            mode_before = nvidia_smi_gpu_mode_snapshot()
            allocation_count = torch.cuda.memory_stats(device)[
                "allocation.all.allocated"
            ]
            addresses = (
                pool.data_ptr(),
                q.data_ptr(),
                out.data_ptr(),
                *(t.data_ptr() for tensors in scratch.values() for t in tensors),
            )
            samples = {mode: [] for mode in graphs}
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for sample in range(args.samples):
                for mode in list(graphs) if sample % 2 == 0 else reversed(graphs):
                    begin.record()
                    timing_graphs[mode].replay()
                    end.record()
                    end.synchronize()
                    samples[mode].append(begin.elapsed_time(end) * 1000 / args.replays)
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"]
                == allocation_count
            )
            assert addresses == (
                pool.data_ptr(),
                q.data_ptr(),
                out.data_ptr(),
                *(t.data_ptr() for tensors in scratch.values() for t in tensors),
            )
            records.append(
                {
                    "profile": profile.name,
                    "form": form.name,
                    "layers": form.layers,
                    "request_batch_size": None
                    if args.query_rows
                    else rows // query_width,
                    "query_width": query_width,
                    "live_query_rows": rows,
                    "context_tokens": args.context_tokens,
                    "planned_max_model_len": args.max_model_len,
                    "planned_rows": {
                        mode: capacities[mode] if v41 else rows for mode in capacities
                    },
                    "heads": profile.attention_heads,
                    "planning_contract": "fixed scheduler capacity"
                    if v41
                    else "one prepared plan per declared capture shape",
                    "swa_width": form.swa_width,
                    "indexed_width": indexed_width,
                    "swa_page_size": swa_page,
                    "indexed_page_size": indexed_page,
                    "cache_format": profile.cache_format,
                    "native_page_bytes": {"swa": swa_bytes, "indexed": indexed_bytes},
                    "page_stride": stride,
                    "high_pid_byte_offsets": {
                        "swa": swa_base * stride,
                        "indexed": indexed_base * stride
                        if indexed is not None
                        else None,
                    },
                    "split_capacity": {
                        mode: binding.scratch.max_chunks_per_row
                        for mode, binding in bindings.items()
                    },
                    "active_splits": active_splits,
                    "policy": {
                        arm: repr(plans[key].prepared.state.config)
                        for arm, key in arm_keys.items()
                    },
                    "raw_samples_us": samples,
                    "median_us": {
                        mode: median(values) for mode, values in samples.items()
                    },
                    "minimum_cosine": cosine,
                    "relative_l2_error": relative_l2,
                    "relative_error_limit": args.relative_error_limit,
                    "timing_scope": "Repeated native invocations in one captured graph; GPU elapsed time divided by invocation count.",
                    "high_pid_qualified": args.high_pid,
                    "output_sha256": tensor_hash(out[:rows]),
                    "gpu_before_timing": mode_before,
                    "gpu_after_timing": nvidia_smi_gpu_mode_snapshot(),
                    "correctness": "Native-cache oracle for every live row; finite/nonzero output; poisoned scratch; query and empty-visibility replay mutation; stable addresses/no replay allocations; physical page offsets; frozen resolution for the declared planning contract.",
                }
            )
            print(
                json.dumps(
                    {
                        key: records[-1][key]
                        for key in ("form", "live_query_rows", "median_us")
                    }
                ),
                flush=True,
            )
    finally:
        for graph in (*graphs.values(), *timing_graphs.values()):
            graph.reset()
        session.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-profile", dest="profile", choices=tuple(MODEL_REPOS), required=True
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--vllm-path", help="integration checkout for source provenance"
    )
    parser.add_argument(
        "--forms", help="comma-separated forms; default all model forms"
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-sizes", type=_csv, default=[1, 2, 4, 8])
    parser.add_argument(
        "--query-rows",
        type=_csv,
        help="controlled live query rows instead of request batches",
    )
    parser.add_argument("--speculative-tokens", type=int, default=7)
    parser.add_argument(
        "--query-tokens",
        type=int,
        default=1,
        help="target query tokens/request; use e.g. 128 with --modes extend",
    )
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--planned-decode-rows", type=int, default=64)
    parser.add_argument("--planned-extend-rows", type=int, default=4096)
    parser.add_argument("--modes", default="decode", help="decode,extend or both")
    parser.add_argument(
        "--compute-modes",
        default="auto",
        help="V4.1 decode/extend A/B: bf16,fp8; auto uses production selection",
    )
    parser.add_argument(
        "--page-stride",
        type=int,
        default=0,
        help="allocator stride; default native payload aligned to 256 bytes",
    )
    parser.add_argument(
        "--high-pid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41071)
    parser.add_argument(
        "--query-rms",
        type=float,
        help="normalize synthetic queries to this RMS before the RoPE-only probe",
    )
    parser.add_argument(
        "--relative-error-limit",
        type=float,
        default=None,
        help="optional BF16-comparison L2 budget; native arithmetic correctness is always checked against its own oracle",
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    # Explicit numerical arms are diagnostic choices, not cache-format adapters.
    args = parser.parse_args()
    args.modes = args.modes.split(",")
    args.compute_modes = args.compute_modes.split(",")
    if set(args.compute_modes) - {"auto", "bf16", "fp8", "fp8-h8", "fp8-h16"}:
        parser.error("compute modes must be auto, bf16, fp8, fp8-h8, or fp8-h16")
    if args.compute_modes != ["auto"] and args.profile != "deepseek-v4.1-flash":
        parser.error("explicit BF16/FP8 comparison requires V4.1")
    if any(mode in ("fp8-h8", "fp8-h16") for mode in args.compute_modes) and args.modes != ["decode"]:
        parser.error("head-group comparisons require decode")
    if (
        min(
            args.context_tokens,
            args.max_model_len,
            args.planned_decode_rows,
            args.planned_extend_rows,
            args.samples,
            args.replays,
            *(args.query_rows or args.batch_sizes),
        )
        <= 0
        or args.context_tokens > args.max_model_len
        or set(args.modes) - {"decode", "extend"}
    ):
        parser.error(
            "positive counts, bounded live context, and decode/extend modes required"
        )
    profile = load_attention_profile(
        args.profile,
        args.model_path,
        tp_size=args.tp_size,
        block_size=args.block_size,
        speculative_tokens=args.speculative_tokens,
    )
    forms = profile.mla_forms
    if args.forms:
        selected = set(args.forms.split(","))
        forms = [form for form in forms if form.name in selected]
        if {form.name for form in forms} != selected:
            parser.error("unknown profile form")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = {
        "command": [sys.executable, *sys.argv],
        "repository": repository_state(ROOT),
        "arguments": {
            **vars(args),
            "model_path": str(args.model_path) if args.model_path else None,
            "output": str(args.output),
        },
        "profile_config": profile.config,
        "toolchain": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl")
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "benchmarks/deepseek_attention_profiles.py",
                *sorted((ROOT / "b12x/attention/_shared/mla").glob("*.py")),
                *sorted((ROOT / "b12x/attention/compressed_sparse_mla").glob("*.py")),
            )
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "CUTE_", "B12X_"))
        },
        "scope": "Native production plan/bind/run contracts over synthetic prepared Q/K, with distinct request cache pages. Excludes projections, cache writes, metadata construction and whole-model forward. Warm graph replay; V4/V4.1 cache recipes are not numerically interchangeable.",
        "results": [],
    }
    payload["integration"] = integration_provenance(args.vllm_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for form in forms:
        payload["results"].extend(run_form(args, profile, form))
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
