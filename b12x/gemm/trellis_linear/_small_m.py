"""JIT binding for the B12X-owned K6/MCG small-M CUDA kernel."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_SOURCE_DIR = Path(__file__).resolve().parent / "csrc"
_SOURCE = _SOURCE_DIR / "trellis_k6_small.cu"
_VENDORED_FILES = tuple(sorted((_SOURCE_DIR / "vendor").rglob("*.[ch]*")))


_GLM_K6_DECODE_SMS = {
    # Q/indexer projection on the target stream.
    (2048, 4096): 128,
    # TP4 shared-expert FC1/FC2 run beside the target stream. These are the
    # rank-local dimensions after column/row parallel slicing, not the full
    # 4096/2048-wide shared MLP dimensions. The budgets match the E2E-optimal
    # ExLlama autotuner result; using all 188 SMs serializes the graph branches.
    (6144, 1024): 64,
    (512, 6144): 96,
}


def _default_num_sms(size_k: int, size_n: int, available_sms: int) -> int:
    """Select the measured GLM K6 decode overlap budget when applicable."""
    target = _GLM_K6_DECODE_SMS.get((size_k, size_n))
    return available_sms if target is None else min(available_sms, target)


@lru_cache(maxsize=None)
def _available_sms(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _extension_name() -> str:
    digest = hashlib.sha256()
    for path in (_SOURCE, *_VENDORED_FILES):
        if path.is_file():
            digest.update(path.relative_to(_SOURCE_DIR).as_posix().encode())
            digest.update(path.read_bytes())
    return f"b12x_trellis_k6_{digest.hexdigest()[:12]}"


@lru_cache(maxsize=1)
def _extension():
    build_directory = os.environ.get("B12X_TRELLIS_BUILD_DIR")
    if build_directory:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    return load(
        name=_extension_name(),
        sources=[str(_SOURCE)],
        extra_include_paths=[str(_SOURCE_DIR)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        extra_cflags=["-O3"],
        build_directory=build_directory,
        verbose=os.environ.get("B12X_JIT_VERBOSE", "0") == "1",
    )


def run_k6_mcg(
    x: torch.Tensor,
    trellis: torch.Tensor,
    output: torch.Tensor,
    suh: torch.Tensor,
    rotated_input: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    *,
    num_sms: int = 0,
) -> None:
    """Launch the capture-safe K6/MCG kernel on Torch's current stream."""
    capability = torch.cuda.get_device_capability(x.device)
    if capability != (12, 0):
        raise NotImplementedError(
            "Trellis K6 small-M kernel is built for sm_120 only; "
            f"device reports sm_{capability[0]}{capability[1]}"
        )
    if num_sms <= 0:
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        num_sms = _default_num_sms(
            int(x.shape[1]),
            int(output.shape[1]),
            _available_sms(int(device_index)),
        )
    _extension().launch_k6_mcg(
        x,
        trellis,
        output,
        suh,
        rotated_input,
        svh,
        locks,
        int(num_sms),
    )


__all__ = ["run_k6_mcg"]
