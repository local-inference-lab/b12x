# GLM-5.2 DCP selected-record transport research

> **Research archive, not a merge proposal.** These APIs are preserved so the
> experiment can be reviewed or continued. Matched vLLM E2E tests regressed,
> so none of the paths should become a default.

This branch is the SparkInfer half of the GLM-5.2 DCP follow-up performed
after [rtx6kpro issue #35](https://github.com/local-inference-lab/rtx6kpro/issues/35).
The companion vLLM branch is
[`research/glm52-dcp-post35-poc-20260726`](https://github.com/local-inference-lab/vllm/tree/research/glm52-dcp-post35-poc-20260726).

## Scope

The branch tests whether sparse MLA can avoid a dense selected-record copy by
consuming an address table. It contains three transport generations:

1. `PCIeSelectedRecordExchange`: direct peer writes into a dense destination
   slab.
2. `PCIeSelectedRecordCopyExchange`: copy-engine staging and exchange, with
   exact native record sizes including 368 and 656 bytes.
3. `PCIeSelectedStoragePointerExchange`: exchange compact source ordinals and
   create pointers directly into persistent model-owned peer KV storage.

The final sparse MLA interface accepts `record_ptrs`. When present, each
selected index resolves through an `int64` address table instead of indexing a
contiguous local CKV tensor. The implementation supports native FP8-RoPE
records and preserves the existing contiguous path as the default.

## Lifecycle rules

The transport was made CUDA-graph safe rather than treated as a one-shot
microbenchmark:

- packet-pointer modes use a graph-stable ring;
- producer writes are ordered before consumer barriers;
- pointer slots have explicit prepare, consume, and release phases;
- model-storage mappings remain stable across eager and alternating graph
  replay;
- teardown clears persistent IPC mappings;
- the vLLM integration gives graph lanes disjoint exchange slots and a shared
  side stream with deterministic barrier ordering.

`PCIeSelectedStoragePointerExchange.record_ptrs_require_release` is false
because pointers refer to persistent model storage rather than a recyclable
packet slot. Packet-backed pointer modes require release before reuse.

## Correctness evidence

The following gates passed on the measured branch:

- sparse MLA contiguous versus pointer input: bit-identical output for the
  tested case;
- 368-byte and 656-byte selected-record transport coverage;
- source tensors with non-zero storage offsets;
- exact pointer-consumer checksums;
- world-size-2 persistent storage pointers in eager execution and alternating
  CUDA graph replay;
- reset and recreation of storage mappings.

Representative commands inside the test image:

```bash
pytest -q tests/attention/test_sparse_mla.py \
  -k 'record_pointers or run_decode_matches_reference'

CUDA_VISIBLE_DEVICES=0,1 \
SPARKINFER_RUN_PCIE_SELECTED_RECORDS_CE_TEST=1 \
SPARKINFER_SELECTED_STORAGE_POINTER_WORLD_SIZE=2 \
pytest -q tests/comm/test_pcie_selected_records_ce_gpu.py \
  -k 'selected_storage_pointer and record368'
```

The complete vLLM-side validation reported 134 passed/1 skipped for selected
record policy and 80 passed/17 skipped for topology/indexer tests.

## Performance result

Correctness did not translate into E2E performance. On TP4/DCP4 NF3 MTP0,
stock local-CKV decode measured 57.72/56.34/55.61 tok/s at ctx0/64k/256k.
Direct model-storage pointers measured 29.5/18.2/17.1 tok/s. Packet direct
pointers measured 28.9/17.7/16.6 tok/s.

The reason is architectural. Stock DCP decode leaves CKV local and exchanges
compact query, LSE, and output state. The selected-record variants route up to
2,048 complete records per destination. Materializing a slab costs transport
and a copy; consuming pointers removes the copy but makes the attention kernel
perform irregular PCIe reads. Neither beats contiguous local CKV.

Observed PCIe traffic was about 31-33 GB/s for selected-record transport versus
13-14 GB/s for stock decode. This is why reducing the number of logical token
positions did not reduce the real communication workload.

See the companion vLLM branch for the complete result table, overlay build,
launcher, and the separate sharding-through-attention experiment.

## Source identity and disposition

- Base: `local-inference-lab/sparkinfer` `master` at
  `c39b8062ba450c030e669d898a026d10980c9470`.
- Branch: `research/glm52-dcp-post35-poc-20260726`.
- Measured with the companion vLLM research branch in
  `local/vllm:gilded-gnosis-selected-record-consistent-bulk-poc-20260726`.

Do not submit this branch as one production PR. If a future design changes the
communication geometry enough to make remote consumption useful, extract the
minimal transport/kernel pieces into independently reviewed changes and rerun
the stock oracle, E2E throughput, PCIe-byte, KV-capacity, and graph-replay
gates.
