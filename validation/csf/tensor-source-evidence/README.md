# Historical tensor-source loader evidence

These receipts validate the B12X tensor-source checkpoint API in revision
`157cfd2900804e5bd5c81356dd35485715acac5d` with vLLM
`4d11fdcff1198a4f19b1c87d3cb1647754bf7f6f`. That API was removed when CSF
file loading and TP slicing moved entirely into vLLM. The receipts are retained
for provenance and do not qualify the CPU-scale `prepare_weights` interface.
See the [preparation qualification](../README.md) for that interface.
