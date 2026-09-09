"""CPU proofs of the granule mapping and fp32 hops on the served ring code.

``tests/comm/test_pcie_ring_granule_cpu.py`` proves the arithmetic contract
from the step table alone. These tests run ``PCIeDmaAllReduce`` itself over
the host-memory emulation of ``tests/comm/pcie_dma_emulation.py`` (the same
Python loop, kernel calls, flag slots, streams and events the device issues)
with the row-granule mapping and the fp32 reduce-scatter hops enabled, and
check:

* the executed all-reduce equals ``pcie_dma_reference.ring_all_reduce_reference``
  bit for bit for the granule mapping, for fp32 hops, and for both together;
* a tensor reduced whole and as two row halves gives identical bits through
  the ring, which is the property the mapping exists for;
* the in-place replay entry (captured on the second sighting) reproduces the
  eager result and is keyed by the granule size, so equal-sized tensors of
  different widths do not share an entry;
* the nine-rank wire-padding path (element counts that are not a multiple of
  72) keeps working with fp32 hops;
* the schedule model finds no unordered overlapping accesses: the fp32 stage,
  the widened receive areas and the in-place static buffer are ordered by the
  events and flags the loop records;
* ``B12X_PCIE_RING_CHECK_BOUNDS=1`` accepts every plan the ring issues.
"""

from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie.pcie_dma_reference import ring_all_reduce_reference
from tests.comm.pcie_dma_emulation import EmulatedRing, install_cuda_fakes

WORLD = 9
WIDTH = 64
MAX_BYTES = 1 << 20


@pytest.fixture(autouse=True)
def _cuda_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("B12X_PCIE_DMA_PIECES", raising=False)
    monkeypatch.delenv("B12X_PCIE_RING_CHECK_BOUNDS", raising=False)
    install_cuda_fakes(monkeypatch)


def _inputs(rows: int, width: int, seed: int) -> list[torch.Tensor]:
    """Heavy-tailed bf16 activations with a per-rank magnitude spread so the
    summation order is visible in the bits."""
    out = []
    for rank in range(WORLD):
        generator = torch.Generator().manual_seed(seed * 1000 + rank)
        values = torch.randn(rows, width, generator=generator) * 2.0 ** (rank - 4)
        out.append(values.to(torch.bfloat16))
    return out


def _reduce_all(emu: EmulatedRing, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    return emu.run(lambda ring, rank: ring.all_reduce(inputs[rank]).clone())


def _expected(inputs: list[torch.Tensor], granule_rows: int, hops: int) -> torch.Tensor:
    granule_elems = granule_rows * inputs[0].shape[-1] if granule_rows else 0
    return ring_all_reduce_reference(
        inputs, WORLD, granule_elems=granule_elems, fp32_hops=hops
    )


@pytest.mark.parametrize("granule_rows", [1, 2, 4])
@pytest.mark.parametrize("hops", [0, 1, 3, 7])
def test_ring_matches_reference(granule_rows: int, hops: int) -> None:
    rows = 4 * WORLD * granule_rows
    inputs = _inputs(rows, WIDTH, seed=1)
    emu = EmulatedRing(
        WORLD, MAX_BYTES, graph_replay=False, granule_rows=granule_rows, fp32_hops=hops
    )
    outs = _reduce_all(emu, inputs)
    expected = _expected(inputs, granule_rows, hops)
    for out in outs:
        assert torch.equal(out, expected)


@pytest.mark.parametrize("hops", [0, 1, 3, 7])
def test_served_mapping_with_fp32_hops_matches_reference(hops: int) -> None:
    inputs = _inputs(2 * WORLD, WIDTH, seed=2)
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, fp32_hops=hops)
    outs = _reduce_all(emu, inputs)
    expected = _expected(inputs, 0, hops)
    for out in outs:
        assert torch.equal(out, expected)


@pytest.mark.parametrize("granule_rows", [1, 2])
@pytest.mark.parametrize("hops", [0, 3])
def test_row_halves_reduce_bit_identically_to_the_whole(
    granule_rows: int, hops: int
) -> None:
    rows = 4 * WORLD * granule_rows
    half = rows // 2
    inputs = _inputs(rows, WIDTH, seed=3)
    emu = EmulatedRing(
        WORLD, MAX_BYTES, graph_replay=False, granule_rows=granule_rows, fp32_hops=hops
    )
    whole = _reduce_all(emu, inputs)[0]
    top = _reduce_all(emu, [x[:half].contiguous() for x in inputs])[0]
    bottom = _reduce_all(emu, [x[half:].contiguous() for x in inputs])[0]
    assert torch.equal(torch.cat([top, bottom]), whole)


def test_served_mapping_is_not_split_invariant() -> None:
    """Calibration: without granules the halves differ from the whole, so the
    equality above is a property of the mapping and not of the inputs."""
    rows = 4 * WORLD
    half = rows // 2
    inputs = _inputs(rows, WIDTH, seed=3)
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False)
    whole = _reduce_all(emu, inputs)[0]
    top = _reduce_all(emu, [x[:half].contiguous() for x in inputs])[0]
    bottom = _reduce_all(emu, [x[half:].contiguous() for x in inputs])[0]
    assert not torch.equal(torch.cat([top, bottom]), whole)


@pytest.mark.parametrize("hops", [0, 7])
def test_replay_entry_is_in_place_and_keyed_by_granule(hops: int) -> None:
    granule_rows = 2
    rows = 4 * WORLD * granule_rows
    inputs = _inputs(rows, WIDTH, seed=4)
    # Same element count, different width: a different granule mapping.
    narrow = [x.reshape(rows * 2, WIDTH // 2).contiguous() for x in inputs]
    emu = EmulatedRing(WORLD, MAX_BYTES, granule_rows=granule_rows, fp32_hops=hops)

    def sequence(ring, rank):
        eager = ring.all_reduce(inputs[rank]).clone()
        assert not ring._replay_entries
        replayed = ring.all_reduce(inputs[rank]).clone()
        (key,) = ring._replay_entries
        entry = ring._replay_entries[key]
        assert entry.inp.data_ptr() == entry.out.data_ptr()
        other = ring.all_reduce(narrow[rank]).clone()
        other = ring.all_reduce(narrow[rank]).clone()
        keys = list(ring._replay_entries)
        return eager, replayed, key, other, keys

    results = emu.run(sequence)
    expected = _expected(inputs, granule_rows, hops)
    expected_narrow = _expected(narrow, granule_rows, hops)
    for eager, replayed, key, other, keys in results:
        assert torch.equal(eager, expected)
        assert torch.equal(replayed, expected)
        assert key == ("ar", inputs[0].numel(), torch.bfloat16, granule_rows * WIDTH)
        assert torch.equal(other, expected_narrow)
        assert len(keys) == 2
        assert keys[1] == ("ar", inputs[0].numel(), torch.bfloat16, granule_rows * WIDTH // 2)


def test_wire_padding_path_keeps_fp32_hops() -> None:
    """80 elements are not a multiple of 72: the ring pads them to 144 on its
    wire buffers, and the padded 1-D wire takes the served mapping with the
    channel's fp32 hops."""
    hops = 3
    inputs = _inputs(10, 8, seed=5)
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, fp32_hops=hops)
    outs = _reduce_all(emu, inputs)
    padded = [torch.cat([x.reshape(-1), torch.zeros(64, dtype=x.dtype)]) for x in inputs]
    expected = _expected(padded, 0, hops)[:80].view(10, 8)
    for out in outs:
        assert torch.equal(out, expected)


def test_granule_rows_fall_back_when_rows_do_not_tile() -> None:
    """A row count that is not a multiple of world * granule keeps the served
    mapping (the key carries granule size 0) and the served result."""
    # 45 rows tile neither 9 * 2 granules nor 9 * 2 halves; 45 * 64 elements
    # are still a multiple of 72, so the unpadded path is taken.
    inputs = _inputs(45, WIDTH, seed=6)
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, granule_rows=2)
    outs = _reduce_all(emu, inputs)
    expected = _expected(inputs, 0, 0)
    for out in outs:
        assert torch.equal(out, expected)
    assert emu.rings[0]._all_reduce_key(inputs[0])[3] == 0


@pytest.mark.parametrize("hops", [0, 1, 7])
def test_lossless_schedule_reuses_nothing_unordered(hops: int) -> None:
    """The fp32 stage, the widened receive areas and the in-place static buffer
    are reused across pieces, steps, calls and replays; every overlapping pair
    of accesses must be ordered by the events and flags the loop records."""
    granule_rows = 2
    rows = 4 * WORLD * granule_rows
    inputs = _inputs(rows, WIDTH, seed=7)
    emu = EmulatedRing(
        WORLD, MAX_BYTES, model=True, granule_rows=granule_rows, fp32_hops=hops
    )

    def three_calls(ring, rank):
        return [ring.all_reduce(inputs[rank]).clone() for _ in range(3)]

    results = emu.run(three_calls)
    expected = _expected(inputs, granule_rows, hops)
    for outs in results:
        for out in outs:
            assert torch.equal(out, expected)
    assert not emu.conflicts, "\n".join(str(c) for c in emu.conflicts)


def test_bounds_check_accepts_issued_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B12X_PCIE_RING_CHECK_BOUNDS", "1")
    inputs = _inputs(4 * WORLD * 2, WIDTH, seed=8)
    emu = EmulatedRing(WORLD, MAX_BYTES, granule_rows=2, fp32_hops=7)
    outs = emu.run(
        lambda ring, rank: [ring.all_reduce(inputs[rank]).clone() for _ in range(2)]
    )
    expected = _expected(inputs, 2, 7)
    for pair in outs:
        for out in pair:
            assert torch.equal(out, expected)
