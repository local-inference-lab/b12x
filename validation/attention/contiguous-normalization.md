# Contiguous attention normalization compatibility

The original-scale softmax finalizer uses FlashAttention 2's four-lane
reduction order and natural-log normalization. The base-two-only finalizer
is unchanged. Implementation status: implemented. Numerical compatibility
is qualified for the cases below; universal bitwise parity is unsupported.

## Reproducible component gate

Tested source: `c3f763af7255e65b5fc22d19d8594d9f97b8f6b5`, compared with
unmodified `master` at `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`.
Subsequent prose changes do not alter the executable test or kernel AST.
The worktrees are `/root/vllm/kimi/b12x-k3-prefill-normalization-pr` and
`/root/vllm/kimi/b12x-k3-prefill-normalization-reference` respectively.

Run in an environment with B12X dependencies and vLLM's FlashAttention 2
extension installed, using one SM120 GPU:

```bash
python -m pytest -q tests/attention/test_varlen_fa2_normalization.py tests/attention/test_varlen.py
```

Hardware: RTX PRO 6000 Blackwell Workstation, 96 GiB, 188 SMs, physical UUID
`GPU-cd323562-fdc3-78c3-012e-86e281433050`. Clocks were not locked.
Toolchain: CUDA 13.4.1, PyTorch `2.14.0a0+4fdf77b`, CUTLASS DSL 4.6.2.

Result: 14 tests pass, including 13 GPU cases and one CPU reference case.
Eight normalization cases cover causal/noncausal attention, ragged sequences,
BF16 Q/K width 192, V width 128, a 128-by-64 tile, poisoned outputs, input
mutation, exact output/LSE parity and an independent FP64 oracle. Graph
replay must retain the prepared compiled program. Broader contiguous tests
cover grouped-query attention, unequal value dimensions, windows and sinks.

Negative control: run the added test with the unmodified implementation.
The 2,049-query/4,097-key causal and noncausal cases fail with respectively
3 and 2 differing BF16 output elements out of 786,816. These tests detect
the arithmetic difference; the FP64 check does not establish that one
evaluation order is universally more accurate.

## Supplemental installed-model control

This is a same-process **compatibility check**, not an isolated speedup claim
for this patch. Both arms share the same QSRT, workspace and communication
implementation. Only the eager prefill backend changes between FlashAttention
2 and B12X; target decode and speculative execution are unchanged. The
integration's separate overlap improvement is not attributed to normalization.

- Target: `lukealonso/Kimi-K3-QSRT-K2@3b98114115f1d41ce7963ba346c3fca19918b0bd`.
- Draft: BF16 `lightseekorg/kimi-k3-dflash2@e77935fb4804e17eb55085bffd045eae1d779769`.
- B12X: `4741d3ffab512268ef54f4501d30e2de2a0c8ed7`, worktree
  `/root/vllm/kimi/b12x-k3-prefill-numerics`. Its two normalization files
  match the tested PR implementation before comment-only edits.
- vLLM: `9cc106d97644918655cf2190fba916ca43a588b4`, worktree
  `/root/vllm/kimi/vllm-k3-prefill-overlap`.
- Image: `local/kimi-k3:kk-cu134-prefill-normalization-9cc106d976-4741d3ff`,
  ID `sha256:1e07ddd08dff3fe32f9accc037e87423bc421bff2202ddf2b7fe11a560164f0e`.
  Installed wheels, not serving-source overlays. This is a local integration
  image, not a published registry artifact or a dependency of this PR.
- Ten 96 GiB RTX PRO 6000 Blackwell Workstation GPUs, TP10/DCP10, one request,
  five proposals, six-row verification graph, p4096, A16 expert activations,
  aligned BF16 projections, FP8 KV, FP32 recurrent state. Clocks unlocked,
  600 W configured limit; no hardware polling within the measured intervals.
  Physical PCIe addresses are `03:00.0`, `04:00.0`, `23:00.0`, `24:00.0`,
  `43:00.0`, `44:00.0`, `63:00.0`, `64:00.0`, `83:00.0`, `84:00.0`.

The operator-side capture command against the resident diagnostic service was:

```bash
cd /root/vllm/kimi
kk-integration/.venv/bin/python kk-integration/compare-prefill-coding-controls.py \
  --url http://127.0.0.1:8012 \
  --output /mnt/luke/kimi-k3-runs/kk-integration-20260919/qsrt-prefill-numerics/installed/coding-controls
```

That integration harness and image require the operator's artifacts; they
are not part of the self-contained component reproducer above. The harness
SHA256 is `321a5976d601971f295b1c74b7b7d0aee767f53390553cb1a191f36a37d86b3a`.
It submits fixed Python JSON-lines, TypeScript undo/redo and Rust LRU prompts,
greedy sampling with seed 1 and 2,048 output tokens, alternating backend order.
Tokenized input receipt SHA256:
`1858bbd6b89166a7eeeb124d981dd7c72fda11fdd939d84451c7b084b6fa90b5`.

Every timed sample is listed below. Decode rate is `(2048 - 1) / seconds`,
where seconds span the first through last streamed output event. The ratio
is B12X rate divided by FlashAttention 2 rate; greater than one favors B12X.

| Prompt / input tokens | Backend | Decode seconds | Decode tok/s | Rate ratio |
| --- | --- | ---: | ---: | ---: |
| Python / 153 | FlashAttention 2 | 27.728256096001132 | 73.823611298 | 1 |
| Python / 153 | B12X | 27.717659664005623 | 73.851833986 | 1.000382299 |
| TypeScript / 157 | B12X | 33.23043460099143 | 61.600157343 | 0.999836784 |
| TypeScript / 157 | FlashAttention 2 | 33.22501087500132 | 61.610213092 | 1 |
| Rust / 158 | FlashAttention 2 | 33.51567737699952 | 61.075895229 | 1 |
| Rust / 158 | B12X | 33.51026274799369 | 61.085763946 | 1.000161581 |

Each pair has identical output and speculative counters. Output SHA256s:

- Python: `b8ae4142337c2d842cd2dd2c2f7d6b6948d65fdf7fcb767abdef5db201aa2cde`.
- TypeScript: `b7b6345f65aec921045de6278c05f788bd8a0c900ea8a926940a0c362f000a85`.
- Rust: `275700da74d6deecba5f3bcce5890a1e412a8019b02aacc9a8abbb208b15715d`.

Separately, 128 preselected 2,048-token analysis contexts have bit-identical
final BF16 pre-LM-head states: 262,016 scored positions. Conclusion: no
observed numerical or decode-throughput regression against the same-process
FlashAttention 2 control for these inputs. One timing pair per prompt is not
a statistically powered performance comparison. These checks do not establish
capability accuracy, long-context determinism or multi-request qualification.

Operator receipts are sealed under
`/mnt/luke/kimi-k3-runs/kk-integration-20260919/qsrt-prefill-numerics/`.
The 474-file `evidence-index.json` SHA256 is
`12375f7103f83bf27c6ae2b7a00716c1346fa362bb7b887b55b596395b2fc839`.
It includes JUnit results, failing negative controls, hidden-state identities
and the six raw request receipts transcribed above. The digest identifies
the retained evidence; it does not make those local files publicly available.
