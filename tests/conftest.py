"""Shared fixtures/helpers for flashinfer.experimental tests.

Keep this module importable without torch: the isolation/arch lint tests run
in a stdlib-only CI job (with --confcutdir to skip the repo-root conftest).
"""

from __future__ import annotations

import pytest


def require_b12x():
    """Skip unless running on a consumer-Blackwell (SM120/SM121) GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for experimental b12x tests")
    major, minor = torch.cuda.get_device_capability(torch.device("cuda"))
    if major != 12 or minor not in (0, 1):
        pytest.skip(f"SM12x (SM120/SM121) GPU required, found sm_{major}{minor}")
    return torch.device("cuda")


def require_sm103_or_sm12x():
    """Require a supported Blackwell device for portable CuTe qualification."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Blackwell kernel qualification")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 3), (12, 0), (12, 1)):
        pytest.skip(f"Blackwell target required, found {capability}")
    return torch.device("cuda")


@pytest.fixture(autouse=True)
def _deterministic_seed():
    try:
        import torch
    except ImportError:
        yield
        return
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield
