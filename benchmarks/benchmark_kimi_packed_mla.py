"""Compare packed Kimi MLA decode at TP9 head counts and one-million-token capacity.

Record CUDA-graph timings, output digests, and error against fp32 dense attention.
The cache keeps every visible token. Query-head padding changes only zero heads
that are removed before the DCP reduction. Run with an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    unpack_mla_kv_cache_reference,
)
from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(104, 112), required=True)
    parser.add_argument("--policy", choices=("static", "balanced"), required=True)
    parser.add_argument("--partial", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--lengths", default="128,2048,8192,16384")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--amplitudes", default="0.25,1,4")
    parser.add_argument("--capacity", type=int, default=116736)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--flush-l2", action="store_true")
    args = parser.parse_args()
    if torch.cuda.get_device_capability()[0] != 12:
        raise SystemExit("Requires SM12x")
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, valid_heads, page = 4, 99, 1536
    sm_scale = 192 ** -0.5
    result = dict(
        settings=vars(args) | {"output": str(args.output)},
        environment={key: os.environ.get(key) for key in (
            "B12X_MLA_SM120_GLM_FASTPATH", "B12X_MLA_SM120_GLM_W_HW_DEQUANT",
            "B12X_MLA_SM120_BALANCED_WAVES",
        )},
        gpu_before=nvidia_smi_gpu_mode_snapshot(), records=[],
    )
    flush = make_l2_flush_fn(args.flush_l2)
    for seed in map(int, args.seeds.split(",")):
        for length in map(int, args.lengths.split(",")):
            capacity = max(args.capacity, length)
            physical = ((length + page - 1) // page) * page
            gen = torch.Generator(device="cpu").manual_seed(seed + length)
            # Generate only semantic heads before padding, so both geometries
            # receive the same bytes under the same seed.
            q_raw = torch.randn(rows, valid_heads, 576, generator=gen).to("cuda")
            k_raw = torch.randn(physical, 512, generator=gen).to("cuda")
            rope_raw = torch.randn(physical, 64, generator=gen).to("cuda")
            lengths = torch.tensor([max(1, length - i) for i in range(rows)], device="cuda", dtype=torch.int32)
            idx = torch.arange(capacity, device="cuda", dtype=torch.int32).repeat(rows, 1)
            idx.masked_fill_(idx >= lengths[:, None], -1)
            extra = {}
            if "partial_dtype" in sparse_mla.Caps.__dataclass_fields__:
                extra["partial_dtype"] = torch.float32 if args.partial == "fp32" else torch.bfloat16
            elif args.partial != "bf16":
                raise RuntimeError("Reader does not support fp32 partials")
            plan = sparse_mla.plan(sparse_mla.Caps(
                device="cuda", num_q_heads=args.heads, max_q_rows=rows,
                max_batch=rows, max_width=capacity, head_dim=576, v_head_dim=512,
                dtype=torch.bfloat16, kv_dtype=torch.uint8,
                max_chunks_per_row=64, page_size=page, head_major_output=False,
                **extra,
            ))
            spec = plan.scratch_specs()[0]
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device="cuda")
            q = torch.zeros(rows, args.heads, 576, dtype=torch.bfloat16, device="cuda")
            binding = plan.bind(
                scratch=scratch, q=q, selected_indices=idx,
                cache_seqlens_int32=lengths, nsa_cache_seqlens_int32=lengths,
            )
            kwargs = {}
            if "split_policy" in inspect.signature(sparse_mla.run_decode).parameters:
                kwargs["split_policy"] = args.policy
            elif args.policy != "static":
                raise RuntimeError("Reader does not support balanced splits")
            for amplitude in map(float, args.amplitudes.split(",")):
                q[:, :valid_heads].copy_((q_raw * amplitude).to(torch.bfloat16))
                packed = pack_mla_kv_cache_reference(
                    (k_raw * 0.25).to(torch.bfloat16),
                    (rope_raw * 0.25).to(torch.bfloat16),
                )
                cache = packed.view(-1, page, 656)
                last_outputs = {}
                def run(bound=binding, kv=cache):
                    outputs = sparse_mla.run_decode(
                        binding=bound, kv_cache=kv, sm_scale=sm_scale,
                        v_head_dim=512, forced_num_splits=64, return_lse=True,
                        lse_scale="natural", **kwargs,
                    )
                    last_outputs["value"] = outputs
                    return outputs
                eager, eager_lse = run()
                eager = eager[:, :valid_heads].clone()
                eager_lse = eager_lse[:, :valid_heads].clone()
                torch.cuda.synchronize()
                graph = capture_cuda_graph(run, warmup=3)
                samples = bench_cuda_graph(graph, replays=args.iters, l2_flush=flush)["replay_us"]
                graph_out, graph_lse = last_outputs["value"]
                assert torch.equal(eager, graph_out[:, :valid_heads])
                assert torch.equal(eager_lse, graph_lse[:, :valid_heads])
                again, again_lse = run()
                assert torch.equal(eager, again[:, :valid_heads])
                assert torch.equal(eager_lse, again_lse[:, :valid_heads])
                assert torch.isfinite(eager).all() and torch.isfinite(eager_lse).all()
                keys = unpack_mla_kv_cache_reference(packed).float().view(-1, 576)
                expected = []
                expected_lse = []
                for row in range(rows):
                    visible = max(1, length - row)
                    scores = q[row, :valid_heads].float() @ keys[:visible].T * sm_scale
                    expected.append(torch.softmax(scores, dim=-1) @ keys[:visible, :512])
                    expected_lse.append(torch.logsumexp(scores, dim=-1))
                expected = torch.stack(expected)
                expected_lse = torch.stack(expected_lse)
                error = eager.float() - expected
                record = dict(
                    seed=seed, local_tokens=length, amplitude=amplitude,
                    output_sha256=digest(eager), lse_sha256=digest(eager_lse),
                    relative_l2=float(error.norm() / expected.norm()),
                    max_abs_error=float(error.abs().max()),
                    lse_max_abs_error=float((eager_lse - expected_lse).abs().max()),
                    graph_replay_us=samples, median_us=statistics.median(samples),
                    finite=True, eager_repeat_identical=True,
                )
                result["records"].append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "graph_replay_us"}), flush=True)
                del graph, eager, eager_lse, keys, expected, expected_lse, error, cache, packed
            del scratch, binding, plan, q_raw, k_raw, rope_raw, q, idx
            torch.cuda.empty_cache()
    result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
