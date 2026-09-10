# Reference-stack KV-cache KLD — September 9, 2026

| Cache | Mean KL(teacher || student) | Window BCa 95% interval |
| --- | ---: | --- |
| Current r27 DCP4 / NVFP4 MLA KV | 0.0354562238 | [0.0295558848, 0.0434620368] |
| Current r27 DCP4 / FP8 MLA KV | 0.0319451732 | [0.0268267482, 0.0387948613] |

FP8 minus NVFP4: -0.0035110506; paired-window BCa 95% interval
[-0.0081718772, -0.0013675941].
FP8 has lower observed KLD in 22/32 windows.

Exact same 32 previously opened conditional-fit windows and BF16 teacher as the
September 8 measurement: 2048 input tokens, 2047 captured predictions, exclude
row zero, leaving 2046 true-decode rows/window. CPU FP64 KL over vocabulary154880;
equal mean of window means; BCa20000 resamples with seed20260902. Teacher stored F32.
TP4/DCP4, MTP off, maxseq1, batch4096, GMU0.97, maxlen1M; prefix caching enabled,
all prefix-hit counters zero. FP8 first then NVFP4, one server preparation each.
Window uncertainty does not estimate server-run variability. These are development
measurements, not untouched-final qualification or independent reproduction.

Selected image: verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf.
Capture-only derivative image, seal, source identities and all64 window scores are
in comparison.json. Capture seam preserves pre-mask logits and forces exact histories.
NCCL8, plain one-shot cutoff131072, fused cutoff86016, shared expert threshold4096.
Checkpoint and FP8 weight/activation math held fixed; KV dtype and native cache
specialization vary. Capture timing is not a throughput measurement.

Raw logits retired only after hashes, durable numerical scores and receipts.
Audit recomputes aggregates from retained FP64 scores and verifies metadata,
zero prefix hits, row masks and receipt hashes; it is not independent recapture.
No measured windows excluded or rerun. Prior image/cache values remain historical.
The separate production recipe uses MTP3 and24slots; its KV capacity must not be
confused with capacity from this MTP-disabled correctness profile.
