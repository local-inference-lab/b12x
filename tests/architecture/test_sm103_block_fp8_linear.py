"""Planned block-FP8 policy, compiler identity, and storage contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.quant import mxfp8_rows as quant
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import block_fp8_linear as linear
from b12x.gemm.block_fp8_linear._tuning import TUNING
from b12x.gemm.block_fp8_linear._tuning import BlockFp8LinearQuery
from tests.architecture.test_sm103_blockscaled import B300


def test_quantizer_rejects_layout_capacity_and_alias_errors():
    from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
    source = torch.empty(129, 128, dtype=torch.bfloat16)
    storage = empty_mxfp8_rows_for_dense_gemm(129, 128, device="cpu")
    args = [source, storage.values, storage.scale_rows, storage.scale_mma, 128]
    quant._validate_storage(*args)
    for index, replacement in (
        (1, storage.values[:1]),
        (1, storage.values.T),
        (1, source.view(torch.uint8)[:, :128].contiguous().to(torch.int32)),
        (1, source.view(torch.uint8).view(-1)[:129 * 128].view(129, 128)),
        (2, storage.values.view(torch.uint8).view(-1)[:129 * 4].view(1, 129, 4)),
        (3, storage.scale_mma.contiguous()),
    ):
        bad = args.copy()
        bad[index] = replacement
        with pytest.raises(ValueError):
            quant._validate_storage(*bad)

@pytest.mark.parametrize("capacity", [1, 8, 129])
@pytest.mark.parametrize("block_size", [32, 128])
def test_sm103_preparation_preserves_capacity_and_weight_scale_provenance(capacity, block_size):
    query = BlockFp8LinearQuery(max_tokens=capacity, in_features=160, out_features=136,
                               source_dtype="bfloat16", output_dtype="bfloat16",
                               output_mode="provided", weight_block_size=block_size)
    selected = TUNING.configure(query, device=B300)
    assert selected.default.backend == "sm103"
    assert (selected.default.tile_m, selected.default.tile_n) == (128, 128)
    assert TUNING.encode_query(query)["max_tokens"] == capacity
    assert len(list(TUNING.iterate(selected))) == 1
    for config in (replace(selected.default, backend="sm120"), replace(selected.default, tile_m=64)):
        with pytest.raises(ValueError):
            TUNING.configure(query, device=B300, override=config)
    with pytest.raises(ValueError):
        TUNING.configure(replace(query, out_features=132), device=B300)
