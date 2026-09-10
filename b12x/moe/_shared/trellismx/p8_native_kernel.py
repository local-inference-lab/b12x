"""Experimental TP-local P8 procedural-MCG MoE runtime for GLM-5.3.

This is a direct device path: a K3, K4 or K5 trellis stream is decoded to E4M3 inside
the MMA kernel and physical UE8M0/32 scales are consumed by the tensor core.
It implements the frozen TP4 identity-boundary P8 contract and an opt-in M1
H128 suh/svh scale component for a GLM routed layer whose sidecar carries the
matching immutable layer identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cute.runtime import make_ptr
from safetensors import safe_open
from .p8_coupled_scales import (
    COUPLED_SCHEMA as P8_COUPLED_SCHEMA,
    SCALE_NAMES,
    SCHEMA as P8_SCALE_COMPONENT_SCHEMA,
    validate_coupled_component,
    validate_scale_component,
)
from .policy_smallm_schedule import P8SmallMGeometry, p8_small_m_scratch_layout, use_small_m
from .tile_policy import select_tile

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.utils import get_max_active_clusters
from b12x.moe._shared.kernels.route_hoist_dynamic import MoEDynamicKernelBackend
from b12x.moe.fused_moe._impl import (
    _DynamicMoEW4A8Launch,
    _e8m0_scale_to_w4a8_sfb_inplace,
    _launch_dynamic_topk_sum,
    current_cuda_stream,
)


def _gptr(dtype, tensor: torch.Tensor, align: int = 16):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake_i32(shape: tuple[int, ...]):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, shape, assumed_align=4
    )


def _fake_f32(shape: tuple[int, ...]):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, shape, assumed_align=16
    )


@dataclass
class _CompiledArm:
    compiled: object
    tile_m: int
    materialized: bool
    mac: int


class P8NativeTPMoE:
    """Own one TP rank's physical P8 payload and launch compiled MoE kernels."""

    def __init__(
        self,
        sidecar: Path | tuple[Path, Path],
        *,
        device: torch.device,
        tp_rank: int,
        layer: int = 3,
        expected_design_sha256: str | None = None,
        expected_transform_sha256: str | None = None,
        topk: int = 8,
        hidden: int = 4096,
        intermediate: int = 512,
        swiglu_limit: float = 10.0,
        force_materialized: bool | None = None,
        mac_override: int | None = None,
        deterministic_output: bool = True,
        small_m_scheduler: bool = False,
        fc1_tile_n: int = 128,
        debug_capture: bool = False,
        diagnostic_raw_fc1: bool = False,
        fuse_scratch_zero: bool = False,
        compact_scale_storage: bool = False,
        compact_input_storage: bool = False,
        shared_workspace: bool = False,
        world_size: int = 4,
        tp4_parent_sha256: tuple[str, str] | None = None,
        prefill_chunk_tokens: int = 0,
        grid_policy: bool | None = None,
        grouped_m16: bool = False,
        fuse_grouped_scratch: bool = False,
        tile_major_tasks: bool = False,
        fc1_pipeline_stages: int = 2,
        fc1_warps: int = 4,
        fc1_a_swizzle_rotate: bool = False,
        fc1_warp_quant: bool | None = None,
        fc1_exact_staging: bool = False,
        fc1_broadcast_a: bool | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.grouped_m16 = bool(grouped_m16)
        self.fuse_grouped_scratch = bool(fuse_grouped_scratch)
        self.tile_major_tasks = bool(tile_major_tasks)
        if fc1_pipeline_stages not in (2, 3):
            raise ValueError('FC1 pipeline supports two or three stages')
        self.fc1_pipeline_stages = fc1_pipeline_stages
        if fc1_warps not in (4, 8):
            raise ValueError('FC1 supports four or eight warps')
        self.fc1_warps = fc1_warps
        self.fc1_exact_staging = bool(fc1_exact_staging)
        self.fc1_broadcast_a = (os.environ.get('GLM53_P8_FC1_BROADCAST_A') == '1'
                                if fc1_broadcast_a is None else bool(fc1_broadcast_a))
        self.fc1_a_swizzle_rotate = bool(fc1_a_swizzle_rotate)
        self.fc1_warp_quant = (os.environ.get('GLM53_P8_FC1_WARP_QUANT') == '1'
                              if fc1_warp_quant is None else bool(fc1_warp_quant))
        self.grid_policy = (os.environ.get('GLM53_P8_GRID_POLICY') == '1'
                            if grid_policy is None else bool(grid_policy))
        self.prefill_chunk_tokens = int(prefill_chunk_tokens)
        if self.prefill_chunk_tokens < 0 or self.prefill_chunk_tokens % 64:
            raise ValueError("P8 prefill chunk must be zero or a positive multiple of 64")
        self.compact_scale_storage = bool(compact_scale_storage)
        self.compact_input_storage = bool(compact_input_storage)
        self.shared_workspace = bool(shared_workspace)
        self.tp_rank = int(tp_rank)
        self.world_size = int(world_size)
        if self.world_size != 4 or self.tp_rank not in range(4):
            raise ValueError("P8 native supports TP4 only; TP2 validators are unsupported")
        self.layer = int(layer)
        if not 3 <= self.layer <= 44:
            raise ValueError("P8 native layer must be in GLM routed layers 3..44")
        self.topk = int(topk)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.swiglu_limit = float(swiglu_limit)
        self.force_materialized = force_materialized
        self.mac_override = None if mac_override is None else int(mac_override)
        self.deterministic_output = bool(deterministic_output)
        # Explicit developmental opt-in. M2/M3 and prefill retain baseline
        # selection; this is not enabled through a serving environment flag.
        self.small_m_scheduler = bool(small_m_scheduler)
        self.fc1_tile_n = int(fc1_tile_n)
        self.debug_capture = bool(debug_capture)
        self.diagnostic_raw_fc1 = bool(diagnostic_raw_fc1)
        self.fuse_scratch_zero = bool(fuse_scratch_zero)
        self._scratch_layout = (
            p8_small_m_scratch_layout(intermediate=self.intermediate) if self.fuse_scratch_zero else None
        )
        self.debug_tensors = {}
        if self.fc1_tile_n not in (32, 64, 128):
            raise ValueError("FC1 tile N must be 32, 64, or 128")
        if self.fc1_tile_n != 128 and (
            not self.small_m_scheduler or self.swiglu_limit != 10.0
        ):
            raise ValueError("Narrow FC1 requires small-M and SwiGLU limit 10")
        if self.small_m_scheduler and (
            not self.deterministic_output or force_materialized is not None
            or (topk, hidden, intermediate) != (8, 4096, 2048 // self.world_size)
        ):
            raise ValueError("P8 small-M requires deterministic GLM TP4 and automatic fallback")
        if self.diagnostic_raw_fc1 and (
            not self.debug_capture
            or not self.small_m_scheduler
            or self.fc1_tile_n != 128
            or self.fuse_scratch_zero
        ):
            raise ValueError(
                "raw FC1 diagnostic requires debug M1 small-M N128 capture"
            )
        if self.mac_override is not None and self.mac_override <= 0:
            raise ValueError("mac_override must be positive")
        if tp4_parent_sha256 is not None:
            if self.world_size != 2:
                raise ValueError("Parent-pair adapter requires TP2")
            from glm53_nvfp4.p8_tp2_repack import open_tp2_pair
            source = open_tp2_pair(sidecar, tp4_parent_sha256, layer=self.layer, rank=self.tp_rank)
        else:
            source = safe_open(sidecar, framework="pt", device="cpu")
        with source as src:
            metadata = src.metadata() or {}
            schema = metadata.get("schema")
            source_design_sha256 = metadata.get("source_design_sha256")
            bits_text = metadata.get("bits", "")
            if bits_text not in {"3", "4", "5"}:
                raise RuntimeError(f"invalid P8 trellis rate: {bits_text!r}")
            self.trellis_bits = int(bits_text)
            base_required = {
                "layer": str(self.layer),
                "rank": str(self.tp_rank),
                "world_size": str(self.world_size),
                "bits": bits_text,
                "alphabet": "e4m3",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "ldlq": "false",
            }
            identity_schema = schema in {
                    "glm53-p8-identity-mcg-tp4-rank.v1",
                    "glm53-p8-mcg-tp4-rank.v2",
            }
            scale_component_schema = schema == P8_SCALE_COMPONENT_SCHEMA
            full_coupled_schema = schema == P8_COUPLED_SCHEMA.replace("tp4", f"tp{self.world_size}")
            if self.world_size == 2 and not full_coupled_schema:
                raise RuntimeError("TP2 port only supports the full-coupled schema")
            if (
                not (identity_schema or scale_component_schema or full_coupled_schema)
                or any(metadata.get(key) != value for key, value in base_required.items())
                or (identity_schema and metadata.get("boundary") != "identity")
            ):
                raise RuntimeError(f"invalid P8 native sidecar metadata: {metadata}")
            if full_coupled_schema or schema in {
                "glm53-p8-mcg-tp4-rank.v2",
                P8_SCALE_COMPONENT_SCHEMA,
                P8_COUPLED_SCHEMA,
            }:
                if (
                    not isinstance(source_design_sha256, str)
                    or len(source_design_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in source_design_sha256)
                ):
                    raise RuntimeError("v2 P8 sidecar lacks a valid source design hash")
                if (
                    expected_design_sha256 is not None
                    and source_design_sha256 != expected_design_sha256
                ):
                    raise RuntimeError("P8 sidecar does not match the expected design")
            elif expected_design_sha256 is not None:
                raise RuntimeError("historical P8 sidecars cannot satisfy a v2 design pin")
            w13 = src.get_tensor("w13_trellis")
            w2 = src.get_tensor("w2_trellis")
            w13_scale = src.get_tensor("w13_scale_ue8m0")
            w2_scale = src.get_tensor("w2_scale_ue8m0")
            scale_tensors = (
                {name: src.get_tensor(name) for name in SCALE_NAMES}
                if scale_component_schema or full_coupled_schema
                else None
            )
        self.source_design_sha256 = source_design_sha256
        experts = int(w13.shape[1])
        stream_words = 16 * self.trellis_bits
        if tuple(w13.shape) != (
            2, experts, hidden // 16, intermediate // 16, stream_words
        ):
            raise RuntimeError(f"unexpected W13 trellis shape {tuple(w13.shape)}")
        if tuple(w2.shape) != (
            experts, intermediate // 16, hidden // 16, stream_words
        ):
            raise RuntimeError(f"unexpected W2 trellis shape {tuple(w2.shape)}")
        if tuple(w13_scale.shape) != (experts, 2 * intermediate, hidden // 32):
            raise RuntimeError(f"unexpected W13 scale shape {tuple(w13_scale.shape)}")
        if tuple(w2_scale.shape) != (experts, hidden, intermediate // 32):
            raise RuntimeError(f"unexpected W2 scale shape {tuple(w2_scale.shape)}")
        self.experts = experts
        self.scale_component = None
        self.full_coupled = bool(full_coupled_schema)
        if (self.grouped_m16 or self.fuse_grouped_scratch) and not self.full_coupled:
            raise ValueError("grouped scratch requires the full-coupled kernel owner")
        if self.fuse_grouped_scratch and not self.grouped_m16:
            raise ValueError("fused grouped scratch requires grouped_m16")
        if self.shared_workspace and (not self.full_coupled or self.debug_capture or self.fuse_scratch_zero):
            raise ValueError('shared workspace requires full coupling without retained debug tensors or fused arena')
        if self.compact_scale_storage and not self.full_coupled:
            raise ValueError("compact scales require the external full-coupled owners")
        if self.full_coupled and expected_transform_sha256 is None:
            raise RuntimeError(
                "full-coupled P8 requires an externally pinned encoder transform"
            )
        if scale_tensors is not None:
            validator = (
                validate_coupled_component
                if self.full_coupled
                else validate_scale_component
            )
            # Preserve compatibility with the pinned TP4 validator during
            # isolated storage diagnostics. TP2 requires the ported validator.
            validator_kwargs = {} if self.world_size == 4 else {"world_size": self.world_size}
            if self.full_coupled:
                validator_kwargs["expected_transform_sha256"] = (
                    expected_transform_sha256
                )
            self.scale_component = validator(
                metadata, scale_tensors, layer=self.layer, rank=self.tp_rank,
                experts=experts, hidden=hidden, intermediate=intermediate,
                **validator_kwargs,
            )
            if not self.small_m_scheduler or self.fc1_tile_n != 128:
                raise RuntimeError(
                    "P8 scale component requires the M1 N128 owner path"
                )
        # The trellis storage is byte-for-byte the same size as the packed
        # E2M1 descriptor carrier expected by the inherited W4A8 launch ABI.
        # Alias it for the descriptor-only arguments instead of allocating a
        # second ~0.9 GiB of unread dummy weights per layer and TP rank.  The
        # kernel reads the procedural stream through the uint32 pointers below;
        # it never dereferences the descriptor carrier values.
        w13_stream_storage = w13.to(device=self.device).contiguous()
        w2_stream_storage = w2.to(device=self.device).contiguous()
        self.w13_stream = w13_stream_storage.view(torch.int32).reshape(-1)
        if self.fc1_exact_staging:
            if ((self.world_size, self.experts, self.hidden, self.intermediate, self.fc1_tile_n)
                    != (4, 288, 4096, 512, 128) or not self.full_coupled):
                raise ValueError('exact staging requires full-coupled TP4 E288/H4096/I512 N128')
            expected_words = 2 * 288 * 4096 * 512 * self.trellis_bits // 32
            if self.w13_stream.numel() != expected_words:
                raise ValueError('exact staging FC1 stream extent mismatch')
        self.w2_stream = w2_stream_storage.view(torch.int32).reshape(-1)
        w13_scale = w13_scale.to(device=self.device).contiguous()
        w2_scale = w2_scale.to(device=self.device).contiguous()
        # The monolithic kernel consumes the logical [E, N, K/32] UE8M0
        # plane through the sfb_*_mx ABI slots.  The split materialized
        # kernels consume a separately repacked copy through *_sfb_rp.
        # Keep both representations: a one-byte sentinel in the logical slots
        # is an out-of-bounds scale read, not an identity scale.
        self.w13_scale_mx = w13_scale.reshape(-1)
        self.w2_scale_mx = w2_scale.reshape(-1)
        self.w13_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w13_scale.clone(),
            weight_E=experts,
            rows=2 * intermediate,
            k_dim=hidden,
            gated_half_rows=intermediate,
        ).reshape(-1)
        self.w2_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w2_scale.clone(),
            weight_E=experts,
            rows=hidden,
            k_dim=intermediate,
        ).reshape(-1)
        if self.compact_scale_storage:
            # Full coupling forces external materialized FC1/FC2 for every M.
            # Logical-scale ABI arguments remain non-null aliases, but are
            # not read by those owners. Never enable for a monolithic arm.
            # This is opt-in pending device closure against separate storage.
            self.w13_scale_mx = self.w13_sfb
            self.w2_scale_mx = self.w2_sfb
        # These are descriptor carriers only; they alias the trellis storage
        # above and therefore add zero payload bytes.  Their trailing extent is
        # the PACKED row length, which is bits/8 bytes per weight: hidden // 2
        # only at K4.  Deriving it from the stored rate keeps K4 byte-identical
        # while letting K3 and K5 describe their own shorter or longer rows.
        w13_row_bytes = hidden * self.trellis_bits // 8
        w2_row_bytes = intermediate * self.trellis_bits // 8
        w13_dummy_bytes = experts * 2 * intermediate * w13_row_bytes
        w2_dummy_bytes = experts * hidden * w2_row_bytes
        self.w13_dummy = w13_stream_storage.view(torch.uint8).reshape(-1)[
            :w13_dummy_bytes
        ].reshape(
            experts, 2 * intermediate, w13_row_bytes
        )
        self.w2_dummy = w2_stream_storage.view(torch.uint8).reshape(-1)[
            :w2_dummy_bytes
        ].reshape(
            experts, hidden, w2_row_bytes
        )
        self.sentinel = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self.zero_lut = torch.zeros(1, dtype=torch.uint8, device=self.device)
        # MCG never dereferences the LUT pointer. The diagnostic-only arm
        # reuses that dead ABI slot for exactly 128 FP32 trace values (512 B),
        # initialized to an all-ones NaN sentinel so partial writes fail closed.
        self.input_prequant_trace = (
            torch.full((512,), 0xFF, dtype=torch.uint8, device=self.device)
            if self.diagnostic_raw_fc1
            else self.zero_lut
        )
        self.zero_rotation = torch.zeros(1, dtype=torch.float16, device=self.device)
        self.scale_component_packed = (
            self.scale_component.packed.to(device=self.device)
            if self.scale_component is not None
            else self.zero_rotation
        )
        self.ones = torch.ones(experts, dtype=torch.float32, device=self.device)
        if self.small_m_scheduler and self.experts != 288:
            raise ValueError("P8 small-M requires 288 experts")
        # v11: the small-M owner path is compiled per stored rate (K3/K4/K5);
        # M>1 on a non-K4 layer is served row by row through that same exact
        # kernel because the grouped M64 prefill kernels remain K4-only.
        self._compiled: dict[tuple[bool, bool], _CompiledArm] = {}
        self._coupled_reducer = None

    def _compile(self, materialized: bool, small_m: bool = False, expected_m: int | None = None) -> _CompiledArm:
        if self.compact_scale_storage and not (self.full_coupled and materialized):
            raise RuntimeError("compact scales cannot enter a monolithic path")
        selected_m = select_tile(expected_m if expected_m is not None else (1 if small_m else 4096))[0]
        cache_key = (materialized, small_m, selected_m)
        cached = self._compiled.get(cache_key)
        if cached is not None:
            return cached
        tile_m = selected_m
        if self.grouped_m16:
            raise RuntimeError("fixed M16 override conflicts with requested tile policy")
        mac = (
            self.mac_override
            if self.mac_override is not None
            else (64 if materialized else int(get_max_active_clusters(1)))
        )
        kernel = MoEDynamicKernelBackend(
            16,
            (tile_m, 128),
            activation="silu",
            quant_recipe="w4a8_trellis",
            w4a8_repacked=True,
            num_topk=self.topk,
            trellis_bits=self.trellis_bits,
            trellis_codebook="mcg",
            trellis_scaled=True,
            trellis_identity_boundary=not self.full_coupled,
            direct_routing=small_m,
            materialize_intermediate=materialized,
            p8_small_m=small_m,
            p8_fc1_tile_n=self.fc1_tile_n if small_m else 128,
            p8_scale_sandwich=self.scale_component is not None,
            p8_full_coupled=self.full_coupled,
            share_input_across_experts=materialized,
            deterministic_output=self.deterministic_output,
            swiglu_limit=self.swiglu_limit,
        )
        if self.diagnostic_raw_fc1:
            if not (small_m and self.full_coupled):
                raise RuntimeError(
                    "raw FC1 diagnostic dispatched outside full-coupled M1"
                )
            from b12x.moe._shared.kernels.p8_h128_fc1 import (
                P8H128FC1RawCaptureKernel,
            )

            kernel.materialized_phase1_kernel = P8H128FC1RawCaptureKernel()
            kernel.p8_input_prequant_diagnostic = True
        if self.full_coupled:
            # Both M1 and grouped owners share the scale/sign plane geometry.
            kernel.materialized_phase1_kernel.p8_intermediate = self.intermediate
            kernel.materialized_phase2_kernel.p8_intermediate = self.intermediate
            if small_m:
                kernel.materialized_phase1_kernel.p8_tile_major = self.tile_major_tasks
                kernel.materialized_phase2_kernel.p8_tile_major = self.tile_major_tasks
                fc1 = kernel.materialized_phase1_kernel
                fc1.p8_a_swizzle_rotate = self.fc1_a_swizzle_rotate
                fc1.p8_warp_quant = self.fc1_warp_quant
                fc1.p8_exact_staging = self.fc1_exact_staging
                # Direct owner invokes _run_task with valid_rows=1. Never
                # apply this specialization to grouped multi-row owners.
                fc1.p8_broadcast_a = self.fc1_broadcast_a
                fc1.num_warps = self.fc1_warps
                fc1.threads_per_cta = 32 * self.fc1_warps
                fc1.p8_n8_per_warp = 16 // self.fc1_warps
                fc1.owned_row_groups = fc1.tile_m // self.fc1_warps
                fc1.p8_pipeline_stages = self.fc1_pipeline_stages
                fc1.shared_bytes = max(fc1.shared_bytes, self.fc1_pipeline_stages * fc1.stage_bytes)
                fc1.shared_words = (fc1.shared_bytes + 3) // 4
                fc1.trellis_lut_offset = fc1.shared_bytes
        launch = _DynamicMoEW4A8Launch(
            kernel,
            k=self.hidden,
            n=self.intermediate,
            w1_n=2 * self.intermediate,
            num_topk=self.topk,
        )

        def ptr(dtype, address: int, align: int = 16):
            return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)

        def fake_ptr_u8():
            return ptr(cutlass.Uint8, 16)

        def fake_ptr_i32():
            return ptr(cutlass.Int32, 4, 4)

        def fake_ptr_u32():
            return ptr(cutlass.Uint32, 16)

        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float4E2M1FN,
            (2 * self.intermediate, self.hidden, self.experts),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )
        b_w2_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float4E2M1FN,
            (self.hidden, self.intermediate, self.experts),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )
        compiled = b12x_compile(
            launch,
            ptr(cutlass.BFloat16, 16),
            fake_ptr_i32(),
            ptr(cutlass.Float32, 4, 4),
            ptr(cutlass.Float4E2M1FN, 16),
            ptr(cutlass.Float8E4M3FN, 16),
            fake_ptr_u8(),
            fake_ptr_u8(),
            fake_ptr_u32(),
            _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
            _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
            fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
            fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
            b_w13_fake,
            ptr(cutlass.Float8E4M3FN, 16),
            b_w2_fake,
            ptr(cutlass.Float8E4M3FN, 16),
            fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(),
            fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(),
            _fake_i32((self.experts,)),
            _fake_i32((self.experts,)),
            _fake_i32((self.experts + 1,)),
            _fake_f32((self.experts,)), _fake_f32((self.experts,)),
            _fake_f32((self.experts,)), _fake_f32((self.experts,)),
            ptr(
                cutlass.Float32 if self.full_coupled else cutlass.BFloat16,
                16,
            ),
            fake_ptr_i32(),
            ptr(cutlass.Float32, 16),
            1, 1, 1, 1, 1, 1, 1,
            current_cuda_stream(),
            fake_ptr_u8(),
            ptr(cutlass.Float16, 16),
            # The compile spec is the JIT cache key and includes every specialized
            # field, especially stored rate, topology and epilogue dimensions.
            compile_spec=KernelCompileSpec.from_fields(
                "glm53.p8.native.tp",
                4,
                ("fc1_row_alias376", 1),
                ("requested_m_regime_direct_policy", 1),
                ("fc1_route_hoist", 1),
                ("tile_m", tile_m),
                ("trellis_bits", self.trellis_bits),
                ("mcg_k5_funnel", int(small_m and self.trellis_bits == 5)),
                ("fc2_carveout100_grid564", int(small_m)),
                ("fc2_k5_funnel", int(small_m and self.trellis_bits == 5)),
                ("grouped_fc2_grid376", int(not small_m)),
                ("tile_major_tasks", int(self.tile_major_tasks and small_m)),
                ("fc1_pipeline_stages", self.fc1_pipeline_stages if small_m else 2),
                ("fc1_warps", self.fc1_warps if small_m else 4),
                ("fc1_a_swizzle_rotate", int(self.fc1_a_swizzle_rotate and small_m)),
                ("fc1_warp_quant", int(self.fc1_warp_quant and small_m)),
                ("fc1_exact_staging", int(self.fc1_exact_staging and small_m)),
                ("fc1_broadcast_a", int(self.fc1_broadcast_a and small_m)),
                ("materialized", int(materialized)),
                ("small_m_scheduler", int(small_m)),
                ("fc1_tile_n", self.fc1_tile_n if small_m else 128),
                ("experts", self.experts),
                ("hidden", self.hidden),
                ("intermediate", self.intermediate),
                ("topk", self.topk),
                ("rank", self.tp_rank),
                ("scaled", 1),
                ("identity", int(not self.full_coupled)),
                ("scale_sandwich", int(self.scale_component is not None)),
                ("full_coupled", int(self.full_coupled)),
                ("raw_fc1_diagnostic", int(self.diagnostic_raw_fc1)),
                ("input_prequant_diagnostic", int(self.diagnostic_raw_fc1)),
                ("codebook", "mcg"),
                ("deterministic_output", int(self.deterministic_output)),
            ),
            dsl_compile_options=OptLevel(2),
        )
        arm = _CompiledArm(compiled=compiled, tile_m=tile_m, materialized=materialized, mac=mac)
        self._compiled[cache_key] = arm
        return arm

    def _compile_full_coupled_reducer(self):
        if not self.full_coupled:
            raise RuntimeError("coupled reducer requested for non-coupled P8")
        if self._coupled_reducer is not None:
            return self._coupled_reducer
        from b12x.moe._shared.kernels.p8_coupled_topk import (
            P8CoupledTopKSumKernel,
        )

        reducer = P8CoupledTopKSumKernel(topk=self.topk, hidden=self.hidden)
        self._coupled_reducer = b12x_compile(
            reducer,
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_fields(
                "glm53.p8.coupled_topk_h512",
                2,
                ("topk", self.topk),
                ("hidden", self.hidden),
                ("rank", self.tp_rank),
                ("route_dtype", "fp32"),
                ("output_dtype", "bf16"),
            ),
            dsl_compile_options=OptLevel(2),
        )
        return self._coupled_reducer

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != self.hidden:
            raise RuntimeError(f"P8 native input contract mismatch: {x.dtype} {tuple(x.shape)}")
        m = int(x.shape[0])
        if tuple(topk_ids.shape) != (m, self.topk) or tuple(topk_weights.shape) != (m, self.topk):
            raise RuntimeError("P8 native routing shape mismatch")
        if self.prefill_chunk_tokens and m > self.prefill_chunk_tokens:
            if not self.full_coupled or self.debug_capture:
                raise RuntimeError("bounded prefill requires full coupling without debug capture")
            # MoE is token-local. Bound route-output and materialized carrier
            # storage without changing the scheduler batch or attention work.
            # Reuse allocations on the same stream; no host readback or sync.
            output = torch.empty_like(x)
            for start in range(0, m, self.prefill_chunk_tokens):
                stop = min(start + self.prefill_chunk_tokens, m)
                output[start:stop].copy_(self(x[start:stop], topk_weights[start:stop], topk_ids[start:stop]))
            return output
        # Every stored rate now has a fused grouped M64/N128 owner, so a non-K4 layer at M>1
        # runs the same native path as K4 rather than looping the M1 kernel row by row. The
        # row-by-row fallback is deliberately gone: a rate without a grouped specialization
        # must fail closed instead of silently serving at a fraction of the speed.
        if self.scale_component is not None and not self.full_coupled and m != 1:
            raise RuntimeError("P8 scale sandwich currently supports M=1 only")
        # Match the W4A8 planner's measured M16-to-M64 transition: sparse
        # decode and ordinary prefill stay monolithic; only dense routed
        # batches pay for the split materialized phase kernels.
        materialized = (
            m * self.topk >= 36 * self.experts
            if self.force_materialized is None
            else self.force_materialized
        )
        small_m = use_small_m(self.small_m_scheduler, m)
        if self.full_coupled:
            # Experimental direct-route batches through M16; requires matching
            # multirow FC1 and input-prologue patches. Not serving-qualified.
            small_m = m <= 16 and not self.grouped_m16
            materialized = True
        materialized = materialized or small_m
        arm = self._compile(materialized, small_m=small_m, expected_m=m)
        tile_m = arm.tile_m
        x = x.contiguous()
        flat_ids = topk_ids.to(dtype=torch.int32).contiguous().reshape(-1)
        flat_weights = topk_weights.to(dtype=torch.float32).contiguous().reshape(-1)
        physical_tiles = (
            m * self.topk if small_m
            else self.experts + (m * self.topk + tile_m - 1) // tile_m
        )
        rows_padded = physical_tiles * tile_m
        gate_tile_count = ((2 * self.intermediate) // 128) // 2
        max_tasks = physical_tiles * max(gate_tile_count, 1)
        fused_scratch_zero = ((self.fuse_scratch_zero and small_m) or
                              (self.fuse_grouped_scratch and self.grouped_m16))
        shared_kernel_output = None
        if fused_scratch_zero or self.shared_workspace:
            if fused_scratch_zero:
                from .direct_policy_scratch import direct_scratch_layout
                self._scratch_layout = (direct_scratch_layout(m, self.intermediate, tile_m=tile_m) if small_m
                    else p8_small_m_scratch_layout(intermediate=self.intermediate, tokens=m,
                        shared=True, grouped=True, tile_m=tile_m, direct=False))
            layout = (p8_small_m_scratch_layout(intermediate=self.intermediate, tokens=m, shared=True,
                          grouped=self.full_coupled and materialized and not small_m, tile_m=tile_m, direct=small_m)
                      if self.shared_workspace else self._scratch_layout)
            assert layout is not None
            # A single GPU fill initializes all original bytes plus alignment
            # padding. The views add no casts, copies, or device kernels.
            if self.shared_workspace:
                from vllm.v1.worker.workspace import current_workspace_manager
                arena, shared_kernel_output = current_workspace_manager().get_simultaneous(
                    ((layout.nbytes,), torch.uint8),
                    ((m * self.topk, self.hidden), torch.float32),
                )
                arena.zero_()
            else:
                arena = torch.zeros(layout.nbytes, dtype=torch.uint8, device=self.device)
            buffers = {
                region.name: arena.narrow(0, region.offset, region.nbytes)
                .view(getattr(torch, region.dtype)).reshape(region.shape)
                for region in layout.regions
            }
            packed_a = buffers["packed_a"]
            scale_flat = buffers["scale_flat"]
            intermediate_u32 = buffers["intermediate_u32"]
            barrier_count = buffers["barrier_count"]
            barrier_epoch = buffers["barrier_epoch"]
            pair_head = buffers["pair_head"]
            producers_done = buffers["producers_done"]
            all_published = buffers["all_published"]
            task_head = buffers["task_head"]
            task_tail = buffers["task_tail"]
            task_ready = buffers["task_ready"]
            task_expert = buffers["task_expert"]
            task_m_tile = buffers["task_m_tile"]
            task_slice_begin = buffers["task_slice_begin"]
            task_slice_count = buffers["task_slice_count"]
            task_valid_rows = buffers["task_valid_rows"]
            tile_write_count = buffers["tile_write_count"]
            row_counts = buffers["row_counts"]
            expert_write_rows = buffers["expert_write_rows"]
            expert_tile_base = buffers["expert_tile_base"]
            token_map = buffers["token_map"]
            token_weights = buffers["token_weights"]
            output = (torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=self.device)
                      if self.shared_workspace else buffers["output"])
        else:
            # Coupled grouped FC1 addresses the shared carrier by token index,
            # not padded expert row (p8_h128_fc1 src_word uses tok). Keep the
            # original extent for every other owner, including M1.
            input_rows = (m if self.compact_input_storage and self.full_coupled
                          and materialized and not small_m else rows_padded)
            packed_a = torch.zeros(input_rows * self.hidden, dtype=torch.uint8, device=self.device)
            scale_elements = (
                m * (self.hidden // 32)
                if self.full_coupled and materialized and not small_m
                else (self.experts + m * self.topk + 1)
                * tile_m
                * (self.hidden // 8)
            )
            scale_flat = torch.zeros(
                scale_elements, dtype=torch.uint8, device=self.device
            )
            intermediate_count = rows_padded * (self.intermediate + self.intermediate // 32) // 4
            intermediate_u32 = torch.zeros(intermediate_count, dtype=torch.int32, device=self.device)

            def z1():
                return torch.zeros(1, dtype=torch.int32, device=self.device)

            def ztask():
                return torch.zeros(max_tasks, dtype=torch.int32, device=self.device)

            barrier_count, barrier_epoch = z1(), z1()
            pair_head, producers_done, all_published = z1(), z1(), z1()
            task_head, task_tail = z1(), z1()
            task_ready, task_expert, task_m_tile = ztask(), ztask(), ztask()
            task_slice_begin, task_slice_count, task_valid_rows = ztask(), ztask(), ztask()
            tile_write_count = torch.zeros(physical_tiles, dtype=torch.int32, device=self.device)
            row_counts = torch.zeros(self.experts, dtype=torch.int32, device=self.device)
            expert_write_rows = torch.zeros(self.experts, dtype=torch.int32, device=self.device)
            expert_tile_base = torch.zeros(self.experts + 1, dtype=torch.int32, device=self.device)
            token_map = torch.zeros(rows_padded, dtype=torch.int32, device=self.device)
            token_weights = torch.zeros(rows_padded, dtype=torch.float32, device=self.device)
            output = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=self.device)
        if self.diagnostic_raw_fc1:
            # Keep the ordinary allocation block exactly unchanged. Only the
            # diagnostic arm fills NaN payload sentinels and zeroes counters.
            intermediate_u32.fill_(-1)
            trace_base = rows_padded * (self.intermediate // 4)
            intermediate_u32[trace_base + 32 : trace_base + 64].zero_()
        kernel_output = (
            shared_kernel_output if shared_kernel_output is not None else torch.empty(
                m * self.topk,
                self.hidden,
                dtype=torch.float32 if self.full_coupled else torch.bfloat16,
                device=self.device,
            )
            if self.deterministic_output
            else output
        )
        launch_mac = arm.mac
        if self.grid_policy and self.world_size == 4 and self.mac_override is None:
            from .p8_multirow_scratch import direct_grid_capacity
            launch_mac = direct_grid_capacity(m, arm.mac)
        arm.compiled(
            _gptr(cutlass.BFloat16, x),
            _gptr(cutlass.Int32, flat_ids, 4),
            _gptr(cutlass.Float32, flat_weights, 4),
            _gptr(cutlass.Float4E2M1FN, packed_a),
            _gptr(cutlass.Float8E4M3FN, scale_flat),
            _gptr(cutlass.Uint8, packed_a),
            _gptr(cutlass.Uint8, scale_flat),
            _gptr(cutlass.Uint32, intermediate_u32),
            barrier_count, barrier_epoch, pair_head, producers_done, all_published,
            task_head, task_tail,
            _gptr(cutlass.Int32, task_ready, 4),
            _gptr(cutlass.Int32, task_expert, 4),
            _gptr(cutlass.Int32, task_m_tile, 4),
            _gptr(cutlass.Int32, task_slice_begin, 4),
            _gptr(cutlass.Int32, task_slice_count, 4),
            _gptr(cutlass.Int32, task_valid_rows, 4),
            _gptr(cutlass.Int32, tile_write_count, 4),
            self.w13_dummy,
            _gptr(cutlass.Float8E4M3FN, self.sentinel),
            self.w2_dummy,
            _gptr(cutlass.Float8E4M3FN, self.sentinel),
            _gptr(cutlass.Uint8, self.w13_scale_mx),
            _gptr(cutlass.Uint8, self.w2_scale_mx),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint32, self.w13_stream),
            _gptr(cutlass.Uint32, self.w13_sfb),
            _gptr(cutlass.Uint32, self.w2_stream),
            _gptr(cutlass.Uint32, self.w2_sfb),
            row_counts, expert_write_rows, expert_tile_base,
            self.ones, self.ones, self.ones, self.ones,
            _gptr(
                cutlass.Float32 if self.full_coupled else cutlass.BFloat16,
                kernel_output,
            ),
            _gptr(cutlass.Int32, token_map, 4),
            _gptr(cutlass.Float32, token_weights, 4),
            m,
            m * self.topk,
            m * self.topk if self.deterministic_output else m,
            rows_padded,
            max_tasks,
            physical_tiles,
            launch_mac,
            current_cuda_stream(),
            _gptr(cutlass.Uint8, self.input_prequant_trace),
            _gptr(cutlass.Float16, self.scale_component_packed),
        )
        if self.deterministic_output and not self.diagnostic_raw_fc1:
            if self.full_coupled:
                reducer = self._compile_full_coupled_reducer()
                reducer(
                    _gptr(cutlass.Float32, kernel_output),
                    _gptr(cutlass.Float32, flat_weights, 4),
                    _gptr(cutlass.BFloat16, output),
                    m,
                    current_cuda_stream(),
                )
            else:
                _launch_dynamic_topk_sum(
                    route_output=kernel_output,
                    output=output,
                    m=m,
                    num_topk=self.topk,
                    k=self.hidden,
                    stream=current_cuda_stream(),
                )
        if self.debug_capture:
            self.debug_tensors = {
                "packed_a": packed_a, "scale_flat": scale_flat,
                "intermediate_u32": intermediate_u32,
                "route_output": kernel_output,
                "token_map": token_map, "row_counts": row_counts,
                "expert_tile_base": expert_tile_base,
            }
            self.debug_dispatch = {"small_m": small_m, "materialized": materialized,
                                   "fused_scratch_zero": fused_scratch_zero,
                                   "fc1_tile_n": self.fc1_tile_n if small_m else 128,
                                   "tile_m": tile_m}
            if self.diagnostic_raw_fc1:
                self.debug_dispatch["diagnostic_raw_fc1"] = True
                self.debug_tensors["input_prequant_trace"] = (
                    self.input_prequant_trace
                )
        return output
