import cutlass
import cutlass.cute as cute
import pytest

from b12x._lib.dense_gemm import DenseGemmKernel
from types import SimpleNamespace
import math


@pytest.fixture(autouse=True)
def static_layout_arithmetic(monkeypatch):
    monkeypatch.setattr(cute, "make_layout", lambda shape: SimpleNamespace(shape=shape))
    monkeypatch.setattr(cute, "filter_zeros", lambda layout: layout)
    monkeypatch.setattr(cute, "size", math.prod)
    monkeypatch.setattr(cute, "slice_", lambda shape, coord: tuple(v for v, c in zip(shape, coord) if c is None))


@pytest.mark.parametrize("minimum,cap,expected", [(0, 0, 0), (1, 0, 1), (0, 3, 0), (1, 3, 1)])
def test_stage_minimum_survives_insufficient_smem_and_cap(minimum, cap, expected):
    layout = cute.make_layout((128,))
    stages, epi = DenseGemmKernel._compute_stages(
        (64, 128, 256), cutlass.BFloat16, cutlass.BFloat16, cutlass.Uint8,
        layout, layout, (64, 128), cutlass.BFloat16, 20_000, 1,
        epi_stage_cap=1, b_storage_bits=8, minimum_ab_stage=minimum, ab_stage_cap=cap,
    )
    assert stages == expected and epi == 1


def test_stage_storage_width_and_joint_cap_coexist():
    layout = cute.make_layout((128,))
    args = ((64, 128, 256), cutlass.BFloat16, cutlass.BFloat16, cutlass.Uint8,
            layout, layout, (64, 128), cutlass.BFloat16, 120_000, 1)
    wide, _ = DenseGemmKernel._compute_stages(*args, epi_stage_cap=1, b_storage_bits=16)
    compact, _ = DenseGemmKernel._compute_stages(*args, epi_stage_cap=1, b_storage_bits=2)
    capped, _ = DenseGemmKernel._compute_stages(*args, epi_stage_cap=1, b_storage_bits=2, ab_stage_cap=2)
    assert wide == 1 and compact == 2 and capped == 2


@pytest.mark.parametrize("mode", ["masked", "contiguous"])
def test_grouped_candidates_respect_dense_mxfp8_tile_support(mode):
    from b12x.gemm.mgroup_fp8_gemm._tuning import MGroupFP8GemmQuery, TUNING
    query = MGroupFP8GemmQuery(mode=mode, num_groups=2, n=128, k=128,
                              m_capacity=128, a_sf_gran=128 if mode == "masked" else 32)
    choices = list(TUNING.choices(query, None))
    assert len(choices) > 1
    for _, config in choices:
        assert DenseGemmKernel.can_implement(
            cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
            (config.tile_m, config.tile_n), (1, 1), 128, 128, 2, "k", "k", "n",
        )
