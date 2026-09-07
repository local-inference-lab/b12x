"""GPU equivalence of the two batched Kimi-K3 top-16 selection algorithms.

The batched expert selection (`kimi_topk16`, one CTA per router row) can
run its sixteen rounds either as block-wide shared-memory scans (`scan`, the
selection served before this test existed) or over register-held packed
keys (`register`). Both orders are the same total order on packed
(score, expert) keys, so they must return identical expert ids and
bit-identical weights for every input, including ties, non-finite logits and
non-finite correction biases. One CUDA device suffices; set
``B12X_RUN_PCIE_KIMI_TOPK_TEST=1`` to run.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_KIMI_TOPK_TEST") != "1",
    reason="set B12X_RUN_PCIE_KIMI_TOPK_TEST=1 to run the Kimi top-16 GPU test",
)

ROUTER_WIDTH = 896
TOPK = 16


def _select(
    select: str,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from b12x.comm.pcie._dcp_a2a_cute import _get_compiled_kimi_topk16

    rows = int(router_logits.shape[0])
    weights = torch.empty((rows, TOPK), dtype=torch.float32, device=router_logits.device)
    ids = torch.empty((rows, TOPK), dtype=torch.int32, device=router_logits.device)
    _get_compiled_kimi_topk16(threads, select)(
        router_logits.data_ptr(),
        correction_bias.data_ptr(),
        weights.data_ptr(),
        ids.data_ptr(),
        rows,
    )
    torch.cuda.synchronize(router_logits.device)
    return weights, ids


def _cases(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(2026_09_08)

    def randn(rows: int, scale: float = 4.0) -> torch.Tensor:
        return (torch.randn(rows, ROUTER_WIDTH, generator=generator) * scale).to(device)

    bias = (torch.randn(ROUTER_WIDTH, generator=generator) * 0.2).to(device)
    for rows in (1, 2, 3, 4, 5, 8):
        for _ in range(12):
            yield f"normal rows={rows}", randn(rows), bias
    # Heavy ties: three distinct logit levels, so most of the sixteen picks
    # resolve on the expert id.
    for rows in (1, 4, 8):
        levels = torch.tensor([-2.0, 0.5, 3.0])
        ties = levels[torch.randint(0, 3, (rows, ROUTER_WIDTH), generator=generator)]
        yield f"ties rows={rows}", ties.to(device), bias
        yield f"ties zero-bias rows={rows}", ties.to(device), torch.zeros_like(bias)
    # Non-finite logits and biases (the kernel canonicalizes them).
    for rows in (1, 4, 8):
        logits = randn(rows)
        logits[:, 3] = float("inf")
        logits[:, 5] = float("-inf")
        logits[:, 7] = float("nan")
        logits[0, 11:40] = float("inf")
        yield f"nonfinite logits rows={rows}", logits, bias
        wild_bias = bias.clone()
        wild_bias[2] = float("inf")
        wild_bias[4] = float("-inf")
        wild_bias[6] = float("nan")
        yield f"nonfinite bias rows={rows}", logits, wild_bias
    # Every candidate identical: the selection is purely id order.
    for rows in (1, 8):
        yield f"constant rows={rows}", torch.full((rows, ROUTER_WIDTH), 1.5, device=device), torch.zeros_like(bias)
        yield f"all -inf rows={rows}", torch.full((rows, ROUTER_WIDTH), float("-inf"), device=device), bias
    # Signed zeros tie on the canonicalized key.
    zeros = randn(4, 0.0)
    zeros[:, ::2] = -0.0
    yield "signed zeros", zeros, torch.zeros_like(bias)


@pytest.mark.parametrize("threads", (128, 256, 512))
def test_register_selection_matches_the_scan_selection(threads: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    checked = 0
    for name, logits, bias in _cases(device):
        logits = logits.contiguous()
        bias = bias.contiguous()
        scan_weights, scan_ids = _select("scan", logits, bias, threads)
        reg_weights, reg_ids = _select("register", logits, bias, threads)
        assert torch.equal(reg_ids, scan_ids), f"{name}: expert ids differ"
        assert torch.equal(
            reg_weights.view(torch.int32), scan_weights.view(torch.int32)
        ), f"{name}: weights differ bitwise"
        # The sixteen ids of a row are distinct real experts.
        for row in range(int(logits.shape[0])):
            ids = reg_ids[row].tolist()
            assert len(set(ids)) == TOPK, f"{name}: duplicate expert in row {row}"
            assert all(0 <= value < ROUTER_WIDTH for value in ids), name
        checked += 1
    assert checked >= 60


def test_register_selection_is_not_slower(capsys) -> None:
    """Report both kernels' latency at the served decode shape (4 rows)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device="cpu").manual_seed(7)
    logits = (torch.randn(4, ROUTER_WIDTH, generator=generator) * 4).to(device)
    bias = (torch.randn(ROUTER_WIDTH, generator=generator) * 0.2).to(device)
    from b12x.comm.pcie._dcp_a2a_cute import _get_compiled_kimi_topk16

    results = {}
    for select in ("scan", "register"):
        launch = _get_compiled_kimi_topk16(256, select)
        weights = torch.empty((4, TOPK), dtype=torch.float32, device=device)
        ids = torch.empty((4, TOPK), dtype=torch.int32, device=device)

        def run() -> None:
            launch(logits.data_ptr(), bias.data_ptr(), weights.data_ptr(), ids.data_ptr(), 4)

        for _ in range(50):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                run()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end) * 1000 / (50 * 20))
        results[select] = min(samples)
    print(f"kimi_topk16 rows=4 graph replay us: {results}", flush=True)
    assert results["register"] <= results["scan"] * 1.05
