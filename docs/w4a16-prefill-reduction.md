# Bounded W4A16 prefill route reduction

Status: implemented and component-qualified. Full-model serving qualification
is separate from the kernel checks described here.

Set `B12X_W4A16_PREFILL_FUSED_SUM=1` before declaring a fused-MoE plan to use
one FP32 output accumulator per token instead of a BF16 output per routed
expert. This applies to BF16 activations, packed or checkpoint-native MXFP4 /
NVFP4 weights, and planned token capacities above eight. Trellis rotation,
FP16 activations, activation-max calibration and decode-capacity plans retain
their existing reduction paths. The default remains disabled.

The FC2 epilogue retains the BF16-rounded, router-weighted contribution of
each expert, converts that contribution to FP32, and reduces it into the
caller-owned accumulator. The fused kernel zeroes that accumulator before
FC1; its mandatory grid barrier orders initialization before FC2 reduction.
The output is cast to BF16 once after FC2. Relaxed FP32 atomic ordering is
not a bitwise-deterministic summation contract.

Set `B12X_W4A16_STABLE_ROUTE_PACK=1` before plan declaration to retain
ascending token-major route order within each expert for capacities of at
least 4096 routed rows. This stabilizes expert packing, not floating-point
atomic accumulation order. Smaller capacity plans retain atomic packing.

Both options are captured in the immutable MoE query. Changing the process
environment after declaration does not alter scratch geometry or dispatch.
Preparation compiles and retains capacity-specialized programs. Binding
maps caller-owned storage and accepts live token counts within that capacity;
it does not allocate an arena or select another kernel specialization.

Validation covers prepared eager execution, CUDA Graph replay, changing
inputs, mapped/disabled experts, finite nonzero outputs, allocation-free
replay, and fixed-capacity dispatch at several live row counts. Run:

```bash
.venv/bin/python -m pytest \
  tests/moe/test_w4a16_e2e.py::test_w4a16_prefill_reduction_prepared_capacity_and_graph \
  tests/moe/test_tp_moe_scratch_bindings.py::test_w4a16_prefill_reduction_freezes_caller_scratch_contract \
  tests/moe/test_w4a16_route_pack.py
```

For 4096 tokens, 896 experts, top-16 routing, hidden width 7168 and local
intermediate width 192, the planned scratch size decreases from
1,071,913,568 to 308,550,240 bytes. This is a scratch-memory reduction, not
a claim about end-to-end model throughput.
