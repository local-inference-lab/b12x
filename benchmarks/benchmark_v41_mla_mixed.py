"""Graph timings for native full-post-RoPE FP8/FP4 sparse MLA.

Use identical arguments, device clocks and this file across source revisions.
The KV inputs retain the mixed cache ABI; this is a kernel microbenchmark,
not a serving-throughput measurement.
"""

import json
import statistics

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention._shared.mla.compressed_reference import (
    pack_deepseek_v41_cache_reference,
)

torch.manual_seed(41512)
device = torch.device("cuda", torch.cuda.current_device())
page, pool = 64, 32768
cache_values = torch.randn((pool, 512), device=device, dtype=torch.bfloat16) / 3
swa = pack_deepseek_v41_cache_reference(cache_values, page_size=page, cache_kind="swa")
indexed = pack_deepseek_v41_cache_reference(cache_values, page_size=page, cache_kind="indexed")
for mode, rows in (("decode", 1), ("decode", 8), ("extend", 64), ("extend", 1024)):
    q = torch.randn((rows, 16, 512), device=device, dtype=torch.bfloat16) / 3
    swa_indices = torch.randint(pool, (rows, 128), device=device, dtype=torch.int32)
    indices = torch.randint(pool, (rows, 512), device=device, dtype=torch.int32)
    swa_lengths = torch.full((rows,), 128, device=device, dtype=torch.int32)
    lengths = torch.full((rows,), 512, device=device, dtype=torch.int32)
    table = torch.arange(pool // page, device=device, dtype=torch.int32)[None].expand(rows, -1).contiguous()
    plan = mla.plan(mla.Caps(
        device=device, num_q_heads=16, max_q_rows=1024, max_width=640,
        swa_width=128, indexed_width=512, max_page_table_width=pool // page,
        swa_page_size=page, indexed_page_size=page,
        cache_format="deepseek_v41", mode=mode, max_chunks_per_row=4,
        use_cuda_graph=True,
    ))
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = plan.bind(scratch=scratch, q=q, swa_indices=swa_indices,
                       swa_lengths=swa_lengths, indexed_indices=indices,
                       indexed_lengths=lengths, indexed_page_table=table)
    out = torch.empty_like(q)

    def run():
        return mla.run(binding=binding, swa_k_cache=swa, swa_page_size=page,
                       indexed_k_cache=indexed, indexed_page_size=page,
                       sm_scale=512 ** -0.5, out=out)

    run()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(out).all())
    freeze_kernel_resolution("native mixed MLA graph benchmark")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(5):
                run()
        samples = []
        for _ in range(7):
            graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 20)
        print(json.dumps({"mode": mode, "rows": rows, "heads": 16,
                          "samples_us": samples, "median_us": statistics.median(samples),
                          "output_sum": float(out.float().sum()),
                          "output_square_sum": float(out.float().square().sum())}), flush=True)
    finally:
        unfreeze_kernel_resolution()
