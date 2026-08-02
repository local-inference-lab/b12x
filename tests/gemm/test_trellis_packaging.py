from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]
TRELLIS_SOURCE_DIR = ROOT / "sparkinfer" / "gemm" / "trellis_linear" / "csrc"


def test_trellis_runtime_sources_and_license_are_in_source_manifest() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert (
        "recursive-include sparkinfer/gemm/trellis_linear/csrc "
        "*.cu *.cuh *.h LICENSE*"
    ) in manifest.splitlines()

    required = (
        TRELLIS_SOURCE_DIR / "trellis_k6_small.cu",
        TRELLIS_SOURCE_DIR / "vendor" / "LICENSE.exllamav3",
        TRELLIS_SOURCE_DIR / "vendor" / "quant" / "exl3_gemm_kernel.cuh",
    )
    assert all(path.is_file() for path in required)
