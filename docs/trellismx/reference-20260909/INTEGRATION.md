# Selected September9 reference integration

This adds the measured reference's native FP8 route-hoist dispatch, tile policy,
K5 funnel and grouped FC2 modules. The generic dynamic.py is intentionally retained
from the reviewed PR head, including its direct-input block-offset helper and
M16 rejection. The native wrapper selects separate route_hoist_dynamic.py.
No checkpoint reencoding or BF16 expert-MMA substitution is included.

Selected runtime image: verdictai/trellismx:glm53-flash-p8-r27-reference-20260909@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf.
All21 integrated files are byte-identical to that image. Full Python runtime recipe:
https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8/tree/6dbc20306b4cd5e57053a8b35bd4a17a221d88e5/runtime-reference-20260909

CPU checks: syntax and selected-source parity21files; native dispatch import;
pytest --confcutdir=tests/moe -p no:cacheprovider tests/moe/test_trellismx_contract.py -q,
24passed in the selected image with this checkout mounted, no GPUs exposed.
Full-model speed/KLD evidence belongs to the immutable selected image, not a fresh
GPU run of this PR checkout. Generic path retained from prior reviewed head differs
from the selected image; no claim of complete PR/image source identity is made.

Measured reference: C1 decode0K204.611,8K222.100,16K216.636 tokens/s;
32K/64K prefill8407/8407 tokens/s in the finalist screen. Expanded run and all146
benchmark JSONs, including negative and profiled diagnostics, are available at:
https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/tree/da46f3dfdb9eb40c971d00ba7ed5e6c1f1796a20/releases/reference-20260909

Audited same-window development KLD FP8 KV0.0319451732, NVFP4 MLA KV0.0354562238;
TP4/DCP4,MTPoff,maxseq1,32previously-opened windows,2046true-decode rows/window.
One server per arm, FP8 then NVFP4. Not independent replication or final qualification.
Speed serving uses MTP3,24slots,NVFP4 cache; correctness timings are not speed evidence.
AI assistance was used to integrate measured sources and prepare these records.
