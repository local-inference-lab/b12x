# EXL3 checkpoint format

EXL3 is the checkpoint container for trellis-coded
MoE expert weights. Its defining property is tensor-parallel shard
independence: one stored artifact serves every TP degree, and a rank loads
exactly its slice without decoding or re-encoding any trellis symbol. Its
second property is that the container is declarative: codebook, rate
structure, intermediate Hadamard transform, and geometry are metadata, and the
reader derives all byte addressing from those declarations. There are no
named profiles and no arithmetic conventions baked into reader code.

Storage schema id: `exl3-v1`. The in-repo schema implementation is
`b12x/moe/_shared/exl3_schema.py`.

## Concepts

**Slot.** Weights are quantized as `trellis_t256` tiles over the
model's global intermediate axis. That axis is divided into 32-channel
*slots*; a slot is the resharding grain. Every per-channel datum a
rank needs is stored slot-major, so a rank extent — a contiguous, aligned
range of slots — is a row-range read of each tensor.

**Codebook.** The trellis decode law is a model-level setting, one of
`mcg`, `lut_e4m3`, or `lut_fp16` (registry:
`b12x/moe/_shared/trellis_codebooks.py`). The container's payload bytes are
codebook-agnostic; the manifest names the interpretation.

**Rate structure.** Either `uniform` — one bitrate K2..K6 for the whole
model — or `per_expert_pair` — an explicit rate byte per (256-channel
pair, expert), stored as tensors. A rate byte is
`(low_bits << 4) | high_bits` and maps one-to-one onto the fused kernel's
pair-kind vocabulary (`0x33` = P33, `0x24` = P24, `0x43` = P43, `0x44` =
P44, `0x22` = P22). There is no finer granularity: the fused decoder's
intra-pair low/high plane split is exactly one rate byte.

**Intermediate Hadamard.** An optional exact activation-boundary
reparameterization: a residual Hadamard of declared `pre_block` width and
pre/post-activation Hadamards of declared `post_block` width, with one
frozen random sign pattern per expert stored as data.

## Directory layout

```
<root>/exl3-manifest.json
<root>/exl3-layer-<NNNNN>.safetensors      one per MoE layer
```

## Manifest

`exl3-manifest.json` is a single JSON object. Validation is fail-closed:
unknown keys are rejected, and every cross-field consistency rule below is
enforced before any tensor is read.

| field | contents |
|---|---|
| `kind` | `"exl3-manifest"` |
| `schema` | `"exl3-v1"` |
| `codebook` | `"mcg"` \| `"lut_e4m3"` \| `"lut_fp16"` |
| `codebook_seed` | required iff `mcg`: the MCG multiplier `0xCBAC1FED` as an integer; forbidden otherwise |
| `geometry.num_experts` | experts per MoE layer |
| `geometry.hidden_size` | hidden width, multiple of 16 |
| `geometry.intermediate_size` | global (pre-TP) intermediate width |
| `geometry.slot_channels` | `32` |
| `geometry.num_slots` | `intermediate_size / slot_channels` (validated) |
| `geometry.moe_layer_indices` | explicit list of the MoE layer indices |
| `rates.structure` | `"uniform"` \| `"per_expert_pair"` |
| `rates.bits` | uniform only: 2..6, within the codebook's range |
| `rates.pair_kinds` | per_expert_pair only: the exact set of pair kinds the tables use |
| `hadamard.intermediate_hadamard` | boolean |
| `hadamard.pre_block` / `post_block` | intermediate Hadamard only: block widths, multiples of 32; `intermediate_size % post_block == 0` |
| `hadamard.per_expert_input_rotations` | `[E,H]` suh/svh tables vs `[H]` broadcast |
| `layout.row_alignment` | row stride alignment of `codes` (bytes) |
| `layout.extent_alignment_slots` | legal slice granularity in slots |
| `layout.extent_barriers` | slot indices no rank extent may cross |
| `layers` | map of layer index → `{file, sha256}` |

Codebook bit ranges: `lut_e4m3` K2–K4, `lut_fp16` K5–K6, `mcg` K3–K6.

## Per-layer tensors

Each `exl3-layer-<NNNNN>.safetensors` contains exactly the tensors its
declarations call for. The safetensors metadata echoes the manifest
identity (`schema`, `codebook`, `layer`, geometry) and the reader rejects
any disagreement.

| tensor | dtype / shape | presence |
|---|---|---|
| `codes` | u8 `[num_slots, row_stride]` | always |
| `rotations` | fp16 `[num_slots, num_experts, 3, slot_channels]` | always |
| `rates_fc1`, `rates_fc2` | u8 `[num_slots/8, num_experts]` | iff `per_expert_pair` |
| `gate_suh`, `up_suh`, `down_svh` | fp16 `[hidden_size]` or `[num_experts, hidden_size]` | always |
| `sign_pattern` | u8 `[num_experts]`, values 0..7 | iff intermediate Hadamard |

`rotations` holds the intermediate-boundary incoherence values (gate svh,
up svh, down suh) for each slot's 32 channels in physical channel order,
for every rate structure. `gate_suh`/`up_suh`/`down_svh` hold the
hidden-axis values; their shape matches
`hadamard.per_expert_input_rotations`.

## `codes` payload layout

Row `r` of `codes` covers slot `r`. Within a row, expert bundles are
concatenated in expert-id order, followed by zero padding to the row
stride; the stride is a multiple of `layout.row_alignment`, and any
non-zero byte in the padding is rejected.

One bundle holds the trellis code words slot `r` contributes to one
expert: gate ‖ up ‖ down. Each matrix section stores its low-record plane
followed by its high-record plane, each plane K-major:
`[hidden_size/16][16*low_bits]i16 ‖ [hidden_size/16][16*high_bits]i16`.

- Under `per_expert_pair`, the planes are the slot's tile in the pair's
  low-rate 128-channel record and its tile in the high-rate record, and
  `(low_bits, high_bits)` come from the expert's rate byte for the pair
  containing slot `r`.
- Under `uniform`, records do not exist; the two planes are the slot's two
  consecutive N16 (FC1) or K16 (FC2) tiles at the declared bitrate.

Section sizes are pure functions of the declarations
(`exl3_schema.matrix_slot_bytes` / `exl3_schema.bundle_bytes`):

```
matrix_slot_bytes = (hidden_size / 16) * 32 * (low_bits + high_bits)
bundle_bytes      = 3 * matrix_slot_bytes            # equal-rate matrices
```

Bundle offsets within a row follow from prefix sums over the rate tables
(uniform rows have constant bundles). Nothing else about the layout is
implicit: there is no expert reordering by rate class, no rotation of
logical pairs across physical slots, and no embedded scale or format
sections.

## Rank extents

A rank owns the slot range `[first_slot, first_slot + slot_count)`. The
range must align to `layout.extent_alignment_slots` and must not cross any
`layout.extent_barriers` entry. The writer declares those values from what
the payload's transforms require — pair-decoded rates need whole
256-channel pairs (8 slots); blockwise intermediate rotations need whole
rotation blocks; an intermediate-Hadamard pre-activation transform whose halves must not
split contributes a barrier. The reader enforces the declarations without
knowing the reasons.

A rank shorter than the serving kernel's minimum local width is padded at
preparation time, exactly as any other legal extent.

`BtxManifest.partition_extents(world_size)` plans complete storage ownership
from alignment and barrier metadata. It returns ordered `(first_slot, slot_count)`
pairs covering every slot exactly once. Local intermediate widths may differ:
V4.1's 2,304 channels contain nine 256-channel records, so TP2 assigns five
records (1,280 channels) to one rank and four (1,024 channels) to the other.
Integrations must use those local widths rather than split a record at 1,152
channels. The execution planner still validates each local kernel geometry.

## Support status

- **Production**: uniform rate structures across the codebook bit ranges
  (K2–K6), with and without the intermediate Hadamard, on the fused W4A16
  serving path. The qualified production deployment is uniform K2 with the
  intermediate Hadamard and `lut_e4m3`.
- **Implemented on SM103**: `per_expert_pair` with MCG or `lut_e4m3` and P22,
  P33, P24, P43 or P44. Execution with and without the intermediate Hadamard
  retains compressed slot rows, and local intermediate widths may contain
  multiple complete 256-channel pairs. The intermediate Hadamard requires SiTU
  and H512/H128 transforms. See the [SM103 qualification runbook](sm103-qualification.md).
- **Supported on SM120/SM121, non-production**: one ordinary 256-channel
  `per_expert_pair` extent with `{P33}`, `{P33, P43}` or `{P33, P24}` and the
  bits-3 base specialization. The fused kernel uses expert-static dispatch.
  P22, P44, intermediate-Hadamard pairs and multiple local pairs fail closed on
  this backend.
- **Declared, unfused on SM12x**: pair-kind sets containing `P44` (whole-expert
  K4 tiers mixed with K3 experts). Single-launch fused execution has no dynamic
  P44 arm, so SM12x planning fails closed with a status message. Two serial
  launches or the mixed-tier benchmark kernel
  (`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py`) are the execution
  vehicles there.
- Checkpoint extent alignment and barriers remain authoritative for every
  backend. The SM12x FC2 implementation additionally requires a 256-channel
  local intermediate width, so some legal checkpoints are unservable at a
  given TP degree on SM12x; planning reports this fail-closed.
