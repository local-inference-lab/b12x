"""Every PCIe collective kernel triggers programmatic dependents first.

The one-shot, fused one-shot, FP8-wire two-shot and the three BF16 two-shot
kernels execute ``griddepcontrol.launch_dependents`` as their first statement,
before any peer barrier, so a kernel launched behind them with the
programmatic-stream-serialization attribute can stage its own inputs while
the collective waits on the fabric. This test reads the kernel sources; the
nine-GPU ordering check is ``test_pcie_pdl_dependent_tp9_gpu.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import b12x.comm.pcie as pcie

KERNEL_COUNTS = {
    "_oneshot_cute.py": 2,
    "_twoshot_cute.py": 1,
    "_twoshot_bf16_cute.py": 3,
}
TRIGGER = "cute.arch.griddepcontrol_launch_dependents()"
_KERNEL = re.compile(
    r"@cute\.kernel\n    def kernel\((?:.|\n)*?\) -> None:\n"
    r"((?:        #[^\n]*\n)*)(        [^\n]*\n)"
)


def test_every_collective_kernel_triggers_dependents_first() -> None:
    root = Path(pcie.__file__).parent
    for name, count in KERNEL_COUNTS.items():
        text = (root / name).read_text()
        bodies = _KERNEL.findall(text)
        assert len(bodies) == count, (name, len(bodies))
        for _comment, first in bodies:
            assert first.strip() == TRIGGER, (name, first.strip())
