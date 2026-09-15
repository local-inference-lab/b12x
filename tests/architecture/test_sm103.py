"""Architecture gates and canonical SM103 preparation contracts."""
from dataclasses import replace
import pytest
import torch
import b12x
from b12x._lib import gating
from b12x._lib.architecture import architecture_for, UnsupportedArchitectureError
from b12x.moe import fused_moe
from b12x.moe.fused_moe._sm103 import BACKEND, scratch_layout, capacity_regime
from b12x.moe.fused_moe._tuning import TUNING, MoeDecodeConfig
from tests.preparation.test_sm103_contracts import IDENTITY as B300


def test_architecture_features_and_unknown_device():
    arch = architecture_for((10, 3))
    assert arch.has_tmem and arch.tmem_columns == 512
    assert arch.compilation_target == "sm_103a"
    assert arch.max_smem_per_block == 227 * 1024
    assert arch.mma_family == "tcgen05"
    assert not architecture_for((12, 0)).has_tmem
    assert not architecture_for((12, 1)).has_tmem
    assert not architecture_for((10, 0)).implemented
    assert architecture_for((10, 9)) is None


@pytest.mark.parametrize(
    "capability,recognized,default",
    [
        ((10, 3), True, False),
        ((12, 0), True, True),
        ((12, 1), True, True),
        ((10, 0), False, False),
        (None, False, False),
    ],
)
def test_op_gates_are_distinct_from_architecture_recognition(
    monkeypatch, capability, recognized, default
):
    monkeypatch.setattr(
        gating, "get_compute_capability", lambda device=None: capability
    )
    monkeypatch.setattr(gating, "has_cutlass_dsl", lambda: True)
    monkeypatch.setattr(gating, "has_triton", lambda: True)
    assert gating.is_b12x() is recognized
    assert gating.default_is_supported() is default
    assert fused_moe.is_supported() is recognized


def test_scratch_layout_disjoint_and_model_size_independent():
    buffers, nbytes = scratch_layout(8, 8, 4096, 2048)
    assert all(b.offset % 1024 == 0 for b in buffers)
    for a, b in zip(buffers[:-1], buffers[1:], strict=True):
        assert a.offset + a.nbytes <= b.offset
    assert buffers[-1].offset + buffers[-1].nbytes <= nbytes
    assert next(b for b in buffers if b.name == "fc1").shape == (64, 4096)
    assert nbytes < 16 * 1024 * 1024


def make_experts(*, device="cpu", e=2, k=256, n=256, w13_layout="w13", mode="a4"):
    wp = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout=w13_layout),
        activation=fused_moe.ActivationSpec(
            mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=e, hidden_size=k, intermediate_size=n
        ),
    )
    weights = fused_moe.PackedWeights(
        w13=torch.zeros((e, 2 * n, k // 2), dtype=torch.uint8, device=device),
        w2=torch.zeros((e, k, n // 2), dtype=torch.uint8, device=device),
        w13_block_scales=torch.ones((e, 2 * n, k // 16), device=device).to(
            torch.float8_e4m3fn
        ),
        w2_block_scales=torch.ones((e, k, n // 16), device=device).to(
            torch.float8_e4m3fn
        ),
        w13_global_scales=torch.ones(e, device=device),
        w2_global_scales=torch.ones(e, device=device),
        input_scale=torch.ones(e, device=device),
        intermediate_scale=torch.ones(e, device=device),
    )
    return fused_moe.prepare_weights(plan=wp, weights=weights)


def test_compile_gate_rejects_sm120_kernel_before_compiler_resolution(monkeypatch):
    from b12x._lib import compiler

    class WarpKernel:
        __module__ = "b12x.moe._shared.kernels.dynamic"

    monkeypatch.setattr(gating, "get_compute_capability", lambda: (10, 3))
    with pytest.raises(UnsupportedArchitectureError, match="no admitted SM103"):
        compiler.compile(WarpKernel())


def test_v41_trellis_pipeline_consumes_canonical_weight_plan():
    from b12x.moe._shared.kernels.sm103.trellis import TrellisPipeline
    from tests.moe.test_trellis_config import _k3_config

    source = fused_moe.TrellisConfig.from_dict(_k3_config())
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=384, hidden_size=5120, intermediate_size=2304
        ),
    )
    pipeline = TrellisPipeline.from_weight_plan(plan._impl)
    assert pipeline.coupled_hadamard
    assert pipeline.reconstruction(projection="w2", bits=3).capacity == 384 * 320 * 144
    assert (
        pipeline.reconstruction(projection="w13", bits=4).capacity
        == 2 * 384 * 320 * 144
    )
    from b12x.moe.fused_moe._sm103_trellis import validate_weight_plan
    with pytest.raises(UnsupportedArchitectureError, match="SiTU"):
        validate_weight_plan(plan._impl)


def test_public_capability_queries_keep_operation_coverage(monkeypatch):
    import b12x
    from b12x._lib import gating
    from b12x.gemm import mxfp8_linear

    assert b12x.supports_architecture((10, 3), mxfp8_linear.META.archs)
    assert not b12x.supports_architecture((10, 3), b12x.find_op('attention.paged').archs)
    monkeypatch.setattr(gating, 'get_compute_capability', lambda *args: (10, 3))
    monkeypatch.setattr(gating, 'has_cutlass_dsl', lambda: True)
    monkeypatch.setattr(gating, 'has_triton', lambda: True)
    assert mxfp8_linear.is_supported()

@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 256, 8192, 65536])
def test_sm103_preparation_selects_native_capacity(tokens):
    plan = fused_moe.plan_execution(
        experts=make_experts(),
        capacity=fused_moe.ExecutionCapacity(max_tokens=tokens, top_k=2),
    )
    assert plan.prepared is None
    selected = TUNING.configure(plan.query, device=B300)
    assert selected.default.backend == BACKEND
    assert len(list(TUNING.iterate(selected))) == 1
    assert capacity_regime(tokens) in {"m1", "m2_m4", "m5_m8", "m9_m32", "prefill"}
    for foreign in ("micro", "dynamic", "w4a16"):
        with pytest.raises(UnsupportedArchitectureError):
            TUNING.configure(plan.query, device=B300,
                             override=MoeDecodeConfig(foreign, "internal", None))


def test_sm103_tuning_rejects_unsupported_routing_before_allocation():
    plan = fused_moe.plan_execution(experts=make_experts(),
                                  capacity=fused_moe.ExecutionCapacity(max_tokens=8, top_k=2))
    for changes in ({"io_dtype": "float16"}, {"apply_router_weight_on_input": True},
                    {"collect_activation_amax": True}, {"route_logits_dtype": "float32"},
                    {"hidden_size": 4128}, {"num_tokens": 2**30, "routed_rows": 2**31}):
        with pytest.raises(UnsupportedArchitectureError):
            TUNING.configure(replace(plan.query, **changes), device=B300)
