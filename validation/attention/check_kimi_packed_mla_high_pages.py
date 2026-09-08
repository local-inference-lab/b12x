"""Compare packed MLA low and high physical pages across a signed-32-bit byte boundary."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from b12x.attention import sparse_mla
from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference


def main():
    torch.manual_seed(20260908)
    page, width, rows, heads = 1536, 2048, 4, 112
    base_page = (2**31 // (page * 656)) + 2
    base_slot = base_page * page
    packed = pack_mla_kv_cache_reference(
        torch.randn(3072, 512, device="cuda", dtype=torch.bfloat16) / 4,
        torch.randn(3072, 64, device="cuda", dtype=torch.bfloat16) / 4,
    )
    low = packed.view(2, page, 656)
    high = torch.empty((base_page + 2, page, 656), device="cuda", dtype=torch.uint8)
    high[0].zero_()
    high[base_page:].copy_(low)
    q = torch.randn(rows, heads, 576, device="cuda", dtype=torch.bfloat16)
    q[:, 99:].zero_()
    lens = torch.tensor([2048, 2047, 1920, 65], device="cuda", dtype=torch.int32)
    indices = torch.arange(width, device="cuda", dtype=torch.int32).repeat(rows, 1)
    indices.masked_fill_(indices >= lens[:, None], -1)
    high_indices = torch.where(indices >= 0, indices + base_slot, -1)
    plan = sparse_mla.plan(sparse_mla.Caps(
        device="cuda", num_q_heads=heads, max_q_rows=rows, max_batch=rows,
        max_width=116736, dtype=torch.bfloat16, kv_dtype=torch.uint8,
        head_dim=576, v_head_dim=512, max_chunks_per_row=64, page_size=page,
        partial_dtype=torch.bfloat16,
    ))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device="cuda")

    def run(cache, slots):
        binding = plan.bind(
            scratch=scratch, q=q, selected_indices=slots,
            cache_seqlens_int32=lens, nsa_cache_seqlens_int32=lens,
        )
        out, lse = sparse_mla.run_decode(
            binding=binding, kv_cache=cache, sm_scale=192**-0.5,
            v_head_dim=512, forced_num_splits=64, split_policy="static",
            return_lse=True, lse_scale="natural",
        )
        torch.cuda.synchronize()
        return out.clone(), lse.clone()

    expected, expected_lse = run(low, indices)
    got, got_lse = run(high, high_indices)
    assert torch.isfinite(got).all() and torch.isfinite(got_lse).all()
    assert torch.equal(got, expected)
    assert torch.equal(got_lse, expected_lse)
    print({"status": "passed", "page_id": base_page,
           "byte_offset": base_slot * 656, "output_and_lse_bit_identical": True})


if __name__ == "__main__":
    main()
