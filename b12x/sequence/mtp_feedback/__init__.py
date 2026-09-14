"""MTP token and hidden-state feedback through fixed-capacity plans.

``qwen_multistream`` normalizes the target's pre-final state over flattened
``S*H`` with Gemma weights, then adds separate projections into BF16 ``[T,S,H]``.
``rms_concat`` masks zero-position embeddings, independently RMS-normalizes
embedding and hidden state with ordinary learned weights, and projects their
concatenation into BF16 ``[T,H]`` for GLM feedback.
``rms_streams_fp8`` applies ordinary RMS independently to each hidden stream,
then adds separate K128 block-FP8 projections into BF16 ``[T,S,H]`` for DeepSeek.

``plan(Caps(...), invocation=invocation_from_tensors(...))`` declares capacity
and input alignment. ``PreparationSession`` compiles and primes the complete
projection/normalization route. ``bind`` consumes its prepared ``Plan``;
``run`` uses only the stored launchers, with live rows remaining dynamic.
The explicitly named ``reference`` module is a PyTorch correctness oracle and
is never a runtime fallback for the public GPU entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mtp_feedback",
    group="sequence",
    api_style="planned",
    archs=("sm103a", "sm120a", "sm121a"),
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "MtpFeedbackConfig",
        "MtpFeedbackQuery",
        "plan",
        "invocation_from_tensors",
        "bind",
        "run",
        "reference",
        "is_supported",
    ),
    dtypes=("bf16",),
    requires=(),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="fa097786643f49d9e9591fd8b2eb0cb3398d8f79",
        paths=("b12x/sequence/mtp_feedback/",),
    ),
    test_path="tests/sequence/test_mtp_feedback.py",
    since="1.3.0",
    notes=(
        "All contracts use CuTeDSL projections with runtime live-row grids. "
        "Qwen S=4,H=2560 requires Triton normalization auxiliaries; RMS-concat "
        "uses CuTe normalization, S=1 and H divisible by 64 through 16384. "
        "FP8 stream feedback uses CuTe normalization and activation quantization, "
        "FP32 K128 scales, S<=16 and H divisible by 128 through 16384. "
        "SM103 runtime qualification requires physical B300 hardware."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        MtpFeedbackConfig,
        MtpFeedbackQuery,
        Plan,
        bind,
        is_supported,
        plan,
        invocation_from_tensors,
        reference,
        run,
    )

install_lazy_api(globals(), META)
