"""Per-CTA global-timer diagnostics for an isolated W4A16 MoE launch.

Status: research-only. The caller preallocates a workspace tail with ten uint64
values per CTA. Thread zero owns each row; no recording atomics are needed.
Counters overwrite the preceding launch, including CUDA graph replays.
Only actual grid rows are valid. Body start excludes LUT staging. Barrier
timestamps surround the arrival atomic and polling, inside the two CTA syncs.
FC1 polling includes the first successful acquire load; its count is calls,
not stalls. FC2 end follows an extra diagnostic CTA sync. Per-CTA durations
overlap and must not be added to obtain step latency.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int64, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def globaltimer(*, loc=None, ip=None):
    value = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint64(value)


@cute.jit
def record_row(locks: cute.Tensor, offset: cutlass.Constexpr[int]):
    cta, _, _ = cute.arch.block_idx()
    ptr = cute.recast_ptr(
        locks.iterator + Int64(offset) + Int64(cta) * Int64(20),
        dtype=Uint64,
    )
    return cute.make_tensor(ptr, cute.make_layout((10,), stride=(1,)))


@cute.jit
def stamp(
    locks: cute.Tensor, offset: cutlass.Constexpr[int], slot: cutlass.Constexpr[int]
):
    record_row(locks, offset)[slot] = globaltimer()
