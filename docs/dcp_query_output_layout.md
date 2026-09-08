# DCP query gather output layout

Status: implemented; GPU correctness and performance qualification pending.

`PCIeDCPA2APool.all_gather_heads(local_input, out=out)` accepts a
caller-owned `[batch, total_heads, head_dim]` view inside a buffer with padded
batch rows. For TP9 Kimi MLA, the logical shape can be `[4, 99, 656]` inside
`[4, 112, 656]` storage. A 656-byte query record uses a float8 view for raw
transport; it is not a vector of numeric FP8 values.

Heads and their elements are contiguous inside each batch row. The output
pointer, head size in bytes, and batch stride must be 16-byte aligned. Batch
rows must not overlap. A singleton batch does not use its batch stride.
The method `supports_all_gather_heads_output(out)` checks this layout using
tensor metadata. The runtime separately checks shape, device and dtype against
the channel and input. Input and output must use separate storage.

Both pull and push transports retain compact staging and the same pool
capacity. Only the final output address uses the caller's stride, in Int64
16-byte packs. Padding is left untouched; an attention consumer remains
responsible for initializing inactive heads. Compile specification
`comm.pcie.dcp_a2a.all_gather_heads` version 14 distinguishes the stride ABI.
Graph replay uses the caller's preallocated output and warmed channel.

A vLLM integration can query the capability before passing a padded view.
When the method is absent or declines the layout, gathering into compact
storage and then copying into the view preserves compatibility.

CPU coverage is in `tests/comm/test_pcie_dcp_a2a.py`: padding preservation,
invalid layout rejection, and a launch stride exceeding 2^31 sixteen-byte
packs. The GPU suite adds FP16/BF16 padded output and opaque 656-byte records,
input mutation during graph replay, and a second batch row beyond 2 GiB.

Run the GPU gate only on an available nine-GPU host, once for each transport:

```sh
B12X_RUN_PCIE_DCP_A2A_TEST=1 \
B12X_PCIE_DCP_A2A_WORLD_SIZE=9 \
B12X_PCIE_DCP_A2A_TEST_TOTAL_HEADS=99 \
B12X_PCIE_DCP_A2A_TRANSPORT=pull \
.venv/bin/python -m pytest -q tests/comm/test_pcie_dcp_a2a_gpu.py
```

Repeat with `B12X_PCIE_DCP_A2A_TRANSPORT=push`. The large-stride case reserves
slightly over 2 GiB per rank and touches only the logical rows. Host policy
tests and offline compilation do not qualify collective correctness or speed.
