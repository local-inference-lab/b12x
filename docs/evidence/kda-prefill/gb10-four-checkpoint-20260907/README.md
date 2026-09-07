# GB10 four-checkpoint KDA evidence

[The serving report](report.md) and [sanitized measurements](evidence.json) contain the four-configuration GLM-5.3-Flash experiment: 36 cold prefill samples, 16 decode cells and 20 exact-answer/cache smoke checks. They compare neither feature, continuation coalescing, token-sharded mHC and both features. The report discloses sequential arm order, both recovery reboots, diagnostic overhead and the limits of these observations.

The B12X component exports up to four recurrent states during one traversal. The serving integration owns checkpoint planning, physical-page retention, convolution history and pass coalescing. Serving throughput changes therefore do not establish a standalone B12X kernel speedup. mHC is a separate optimization.

The tested component revision is `70fe41974ef4b18f61caaa2579c81cdc05d1265f`, based on `06b4de7c723e6f166d65abf5909c5b7d0f8acc68`. The tested serving composition is vLLM `abb715f132bdccb592a34b2596a3d3a8d757ffbc`. Runtime source IDs and the immutable image digest are included in the JSON. Documentation packaging does not change the tested kernel source.

[Component-check evidence](component-checks.json) records **12 GPU tests passed** and **41 CPU tests passed**, their source-log digests and coverage. The committed [GPU suite](../../../../tests/sequence/test_kda_prefill_two_checkpoints_gpu.py) checks independent FP32 oracles, exact H16 8K checkpoint positions, graph replay, invalid metadata and high pool addresses. The [CPU suite](../../../../tests/sequence/test_kda_prefill_two_checkpoints_cpu.py) covers capacity, policy, metadata and reference contracts. The GPU selection enables `B12X_RUN_LARGE_POOL_TESTS=1`; no selected cases were skipped.

Eligibility remains NVIDIA GB10. These execution/correctness checks are not a measured component-policy profile. KDA catalog/offline-provider integration and embedded measured profiles remain unresolved. The serving smoke tests do not establish full numerical or model-quality equivalence.

Both native vLLM split-page settings were held at 512 in every serving arm: `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE` and `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE`. They preserve physical/lookup/scheduler alignment `(512, 512, 2048)` and do not require SparkCache. Enabled feature flags were accompanied by request-associated checkpoint/mHC dispatch evidence and completed API requests.

The companion [serving reproduction instructions](https://github.com/FujitsuPolycom/vllm/blob/feat/gb10-continuation-prefill/docs/benchmarking/glm-kda-checkpoints-20260907/reproduction.md) identify the model snapshot, client inputs and public runtime composition.
