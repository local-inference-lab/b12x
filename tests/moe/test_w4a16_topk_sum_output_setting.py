"""Process setting of the full-rotation top-k sum's output element (CPU)."""

from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.w4a16 import host as w4a16_host


def test_setting_defaults_to_fp32(monkeypatch) -> None:
    monkeypatch.delenv("B12X_W4A16_TOPK_SUM_OUTPUT", raising=False)
    assert w4a16_host.w4a16_topk_sum_rotation_output_dtype() == "fp32"
    assert w4a16_host.w4a16_topk_sum_rotation_output_torch_dtype() == torch.float32


@pytest.mark.parametrize(
    ("raw", "name", "dtype"),
    [
        ("bf16", "bf16", torch.bfloat16),
        (" BF16 ", "bf16", torch.bfloat16),
        ("fp16", "fp16", torch.float16),
        ("fp32", "fp32", torch.float32),
    ],
)
def test_setting_names_the_store_element(monkeypatch, raw, name, dtype) -> None:
    monkeypatch.setenv("B12X_W4A16_TOPK_SUM_OUTPUT", raw)
    assert w4a16_host.w4a16_topk_sum_rotation_output_dtype() == name
    assert w4a16_host.w4a16_topk_sum_rotation_output_torch_dtype() == dtype


def test_setting_rejects_other_values(monkeypatch) -> None:
    monkeypatch.setenv("B12X_W4A16_TOPK_SUM_OUTPUT", "float64")
    with pytest.raises(ValueError, match="fp32, bf16 or fp16"):
        w4a16_host.w4a16_topk_sum_rotation_output_dtype()


def test_kernel_object_carries_the_output_element() -> None:
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16TopKSumKernel

    kernel = W4A16TopKSumKernel(
        topk=16, hidden_size=1024, element_dtype="fp16", full_rotation=True,
        num_experts=4, output_dtype="bf16",
    )
    assert kernel.output_dtype == "bf16"
    with pytest.raises(ValueError, match="output_dtype"):
        W4A16TopKSumKernel(
            topk=16, hidden_size=1024, element_dtype="fp16", full_rotation=True,
            num_experts=4, output_dtype="fp64",
        )


def test_compile_key_distinguishes_the_output_element(monkeypatch) -> None:
    """The compile cache key carries the store element, and the plain sum
    rejects an output element other than its element dtype."""
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    seen: list[tuple] = []

    def fake_compile(kernel, *args, compile_spec=None, **kwargs):
        seen.append(compile_spec.cache_key if hasattr(compile_spec, "cache_key") else None)
        return object()

    monkeypatch.setattr(w4a16_kernel, "b12x_compile", fake_compile)
    monkeypatch.setattr(w4a16_kernel, "raise_if_kernel_resolution_frozen", lambda *a, **k: None)
    monkeypatch.setattr(w4a16_kernel, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(w4a16_kernel, "_SUM_CACHE", {})
    common = dict(
        m=4, topk=16, hidden_size=1024, element_dtype="fp16", full_rotation=True,
        coupled_hadamard=False, num_experts=4, route_num_experts=0,
        route_ids_dtype=torch.int32, use_expert_map=False, broadcast_svh=False,
    )
    monkeypatch.setenv("B12X_W4A16_TOPK_SUM_OUTPUT", "fp32")
    default = w4a16_kernel.compile_w4a16_topk_sum(**common)
    monkeypatch.setenv("B12X_W4A16_TOPK_SUM_OUTPUT", "bf16")
    from_setting = w4a16_kernel.compile_w4a16_topk_sum(**common)
    explicit = w4a16_kernel.compile_w4a16_topk_sum(**common, output_dtype="fp16")
    assert (default.output_dtype, from_setting.output_dtype, explicit.output_dtype) == (
        "fp32", "bf16", "fp16"
    )
    assert default.compiled is not from_setting.compiled is not explicit.compiled
    keys = list(w4a16_kernel._SUM_CACHE)
    assert len(keys) == 3 and {k[-1] for k in keys} == {"fp32", "bf16", "fp16"}
    with pytest.raises(ValueError, match="plain top-k sum"):
        w4a16_kernel.compile_w4a16_topk_sum(
            m=4, topk=16, hidden_size=1024, element_dtype="bf16", output_dtype="fp16"
        )
