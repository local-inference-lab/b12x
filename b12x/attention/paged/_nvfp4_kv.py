"""NVFP4 records for paged GQA K/V caches.

Each (token, KV head) row of ``head_dim`` values is stored per cache side as
one self-describing record of ``record_nbytes(head_dim)`` bytes (148 for
``head_dim=256``; FP8 E4M3 uses 256 and BF16 512):

    [0, D/2)              E2M1 values, two per byte, low nibble = even element
    [D/2, D/2 + D/16)     one E4M3 scale byte per 16 consecutive values
    [D/2 + D/16, +4)      fp32 per-row outer scale ``s = amax / (6 * 448)``

and decodes as ``e2m1 * e4m3_decode(group_scale) * s``. This is the two-level
per-token recipe of the NVFP4 MLA K/V records in ``_shared/mla/kv_cache.py``:
the outer scale places the row's largest group scale at the top of the E4M3
range, so group scales stay out of E4M3 subnormals at any activation
magnitude. Every field starts on a four-byte boundary when the page, token and
head strides are multiples of four bytes, so no record padding is needed.

The writer is one CTA per token and one warp per KV head; lanes 0-15 quantize
the key row's sixteen 16-value groups and lanes 16-31 the value row's. Slots
outside ``[0, pages * page_size)`` are skipped (padded CUDA-graph rows). All
page, slot and record offsets are Int64.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    cvt_e4m3_to_f32_via_f16,
    cvt_f32_to_e4m3,
    fmax_f32,
    ld_global_nc_u32,
    max_abs_16,
    quantize_and_pack_16_fast,
    rcp_approx_ftz,
    st_global_f32,
    st_global_u32,
    st_global_u8,
)
from b12x._lib.program_cache import register_program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

# Declaration marker for the record format. Cache tensors are raw ``uint8``
# records; this dtype only names the format in plans and queries.
NVFP4_KV_DTYPE = torch.float4_e2m1fn_x2
GROUP_SIZE = 16
OUTER_SCALE_NBYTES = 4
# Exact-constant contract shared with the MLA NVFP4 writer (``_TWO_LEVEL_RCP``
# in ``_shared/mla/kv_cache.py``).
TWO_LEVEL_RCP = 1.0 / (6.0 * 448.0)
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
_WARP = 32


def record_nbytes(head_dim: int) -> int:
    """Bytes of one NVFP4 record for a ``head_dim``-wide row."""
    head_dim = int(head_dim)
    if head_dim <= 0 or head_dim % (2 * GROUP_SIZE):
        raise ValueError("NVFP4 K/V head_dim must be a positive multiple of 32")
    return head_dim // 2 + head_dim // GROUP_SIZE + OUTER_SCALE_NBYTES


def scale_offset(head_dim: int) -> int:
    return int(head_dim) // 2


def outer_scale_offset(head_dim: int) -> int:
    return int(head_dim) // 2 + int(head_dim) // GROUP_SIZE


def is_nvfp4_kv_dtype(dtype: torch.dtype) -> bool:
    return dtype == NVFP4_KV_DTYPE


def kv_storage(kv_dtype: torch.dtype, head_dim: int) -> tuple[torch.dtype, int]:
    """Storage dtype and innermost width of one K or V cache row."""
    if kv_dtype == NVFP4_KV_DTYPE:
        return torch.uint8, record_nbytes(head_dim)
    if kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        return kv_dtype, int(head_dim)
    raise TypeError(
        "paged K/V cache dtype must be BF16, FP8 E4M3FN or NVFP4 "
        f"({NVFP4_KV_DTYPE}), got {kv_dtype}"
    )


def is_nvfp4_cache(cache: torch.Tensor, head_dim: int) -> bool:
    """Whether a ``[..., record]`` cache view holds NVFP4 records."""
    return cache.dtype == torch.uint8 and int(cache.shape[-1]) == record_nbytes(
        head_dim
    )


# ---------------------------------------------------------------------------
# Host references (tests and diagnostics).
# ---------------------------------------------------------------------------

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_nvfp4_kv_torch(records: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Decode ``[..., record_nbytes(head_dim)]`` uint8 records to float32."""
    if records.dtype != torch.uint8 or int(records.shape[-1]) != record_nbytes(
        head_dim
    ):
        raise ValueError("records must be uint8 NVFP4 K/V records")
    lead = records.shape[:-1]
    # Explicit row counts keep empty inputs (a cache shard that holds no pages) well defined.
    rows = math.prod(lead)
    flat = records.reshape(rows, records.shape[-1])
    packed = flat[:, : head_dim // 2].to(torch.int32)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, head_dim)
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=records.device)
    magnitude = table[codes & 0x7]
    values = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    group_scales = (
        flat[:, scale_offset(head_dim) : outer_scale_offset(head_dim)]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    outer = (
        flat[:, outer_scale_offset(head_dim) :]
        .contiguous()
        .view(torch.float32)
        .reshape(rows, 1)
    )
    factor = (group_scales * outer).repeat_interleave(GROUP_SIZE, dim=-1)
    return (values * factor).reshape(*lead, head_dim)


def quantize_nvfp4_kv_torch(rows: torch.Tensor) -> torch.Tensor:
    """Encode ``[..., D]`` rows with the writer's recipe (exact reciprocals).

    The CUDA writer uses ``rcp.approx.ftz``; byte identity with this reference
    is therefore expected for the outer scale and nearly all group scales, and
    tests compare decoded values rather than every payload byte.
    """
    head_dim = int(rows.shape[-1])
    lead = rows.shape[:-1]
    count = math.prod(lead)
    x = rows.reshape(count, head_dim).float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    outer = amax * torch.tensor(TWO_LEVEL_RCP, dtype=torch.float32)
    groups = x.view(count, head_dim // GROUP_SIZE, GROUP_SIZE)
    group_amax = groups.abs().amax(dim=-1)
    safe_outer = torch.where(outer == 0, torch.ones_like(outer), outer)
    scale = (group_amax / safe_outer / _E2M1_MAX).clamp(max=_E4M3_MAX)
    scale = torch.where(outer == 0, torch.zeros_like(scale), scale)
    scale_bytes = scale.to(torch.float8_e4m3fn)
    step = scale_bytes.float() * outer
    q = groups / torch.where(step == 0, torch.ones_like(step), step).unsqueeze(-1)
    q = torch.where((step == 0).unsqueeze(-1), torch.zeros_like(q), q)
    # Round to nearest, ties to the even code (the cvt.rn.satfinite rule);
    # the E2M1 codes 0..7 are ordered, so ties sit between neighbours.
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=rows.device)
    magnitude = q.abs().clamp(max=_E2M1_MAX)
    upper = torch.searchsorted(table, magnitude.contiguous()).clamp(max=7)
    lower = (upper - 1).clamp(min=0)
    to_upper = (table[upper] - magnitude) < (magnitude - table[lower])
    tie = (table[upper] - magnitude) == (magnitude - table[lower])
    code = torch.where(to_upper | (tie & (upper % 2 == 0)), upper, lower)
    code = torch.where(magnitude == table[upper], upper, code)
    code = code | torch.where(q < 0, 8, 0)
    code = code.reshape(x.shape[0], head_dim).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    out = torch.empty(
        (x.shape[0], record_nbytes(head_dim)), dtype=torch.uint8, device=rows.device
    )
    out[:, : head_dim // 2] = packed
    out[:, scale_offset(head_dim) : outer_scale_offset(head_dim)] = scale_bytes.view(
        torch.uint8
    )
    out[:, outer_scale_offset(head_dim) :] = (
        outer.reshape(count).contiguous().view(torch.uint8).view(count, 4)
    )
    return out.reshape(*lead, record_nbytes(head_dim))


# ---------------------------------------------------------------------------
# Device helpers.
# ---------------------------------------------------------------------------


@dsl_user_op
def nvfp4x8_scaled_to_bfloat2x4(
    packed: Uint32, factor: Float32, *, loc=None, ip=None
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Decode eight E2M1 values times ``factor`` to four packed bf16x2 words.

    Each E2M1 value converts exactly to f16 and f32; the fp32 product with
    ``factor`` (group scale times outer scale) is rounded to BF16 once.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(packed).ir_value(loc=loc, ip=ip),
            Float32(factor).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 q;
            .reg .b16 lo, hi;
            .reg .b32 byte, pair;
            .reg .f32 a, b;
            cvt.u8.u32 q, $4;
            cvt.rn.f16x2.e2m1x2 pair, q;
            mov.b32 {lo, hi}, pair;
            cvt.f32.f16 a, lo;
            cvt.f32.f16 b, hi;
            mul.rn.f32 a, a, $5;
            mul.rn.f32 b, b, $5;
            cvt.rn.bf16x2.f32 $0, b, a;
            shr.b32 byte, $4, 8;
            cvt.u8.u32 q, byte;
            cvt.rn.f16x2.e2m1x2 pair, q;
            mov.b32 {lo, hi}, pair;
            cvt.f32.f16 a, lo;
            cvt.f32.f16 b, hi;
            mul.rn.f32 a, a, $5;
            mul.rn.f32 b, b, $5;
            cvt.rn.bf16x2.f32 $1, b, a;
            shr.b32 byte, $4, 16;
            cvt.u8.u32 q, byte;
            cvt.rn.f16x2.e2m1x2 pair, q;
            mov.b32 {lo, hi}, pair;
            cvt.f32.f16 a, lo;
            cvt.f32.f16 b, hi;
            mul.rn.f32 a, a, $5;
            mul.rn.f32 b, b, $5;
            cvt.rn.bf16x2.f32 $2, b, a;
            shr.b32 byte, $4, 24;
            cvt.u8.u32 q, byte;
            cvt.rn.f16x2.e2m1x2 pair, q;
            mov.b32 {lo, hi}, pair;
            cvt.f32.f16 a, lo;
            cvt.f32.f16 b, hi;
            mul.rn.f32 a, a, $5;
            mul.rn.f32 b, b, $5;
            cvt.rn.bf16x2.f32 $3, b, a;
        }
        """,
        "=r,=r,=r,=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def _address(pointer, offset, *, loc=None, ip=None) -> Int64:
    element_pointer = pointer + offset
    return Int64(llvm.ptrtoint(T.i64(), element_pointer.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _bf16x2_to_f32x2(word: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """Exact promotion of packed bf16x2 (low half first) to two float32."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            shl.b32 lo, $2, 16;
            and.b32 hi, $2, 0xFFFF0000;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


class Nvfp4KVWriteKernel:
    """Quantize BF16 K/V rows into paged NVFP4 records (one CTA per token)."""

    def __init__(
        self,
        *,
        kv_heads: int,
        head_dim: int,
        page_size: int,
        key_strides: tuple[int, int, int],
        value_strides: tuple[int, int, int],
    ) -> None:
        self.kv_heads = int(kv_heads)
        self.key_strides = tuple(map(int, key_strides))
        self.value_strides = tuple(map(int, value_strides))
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.groups = self.head_dim // GROUP_SIZE
        if self.head_dim != 256:
            raise ValueError("the NVFP4 K/V writer requires head_dim=256")
        if not 1 <= self.kv_heads <= 32:
            raise ValueError("the NVFP4 K/V writer supports 1 to 32 KV heads")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        self.threads = _WARP * self.kv_heads
        self.scale_offset = scale_offset(self.head_dim)
        self.outer_offset = outer_scale_offset(self.head_dim)

    @cute.jit
    def __call__(
        self,
        key: cute.Pointer,
        value: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        slot_mapping: cute.Pointer,
        key_row_stride: Int64,
        value_row_stride: Int64,
        slot_capacity: Int64,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            key_row_stride,
            value_row_stride,
            slot_capacity,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        key: cute.Pointer,
        value: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        slot_mapping: cute.Pointer,
        key_row_stride: Int64,
        value_row_stride: Int64,
        slot_capacity: Int64,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        thread = Int32(thread_idx)
        token = Int64(token_idx)
        head = thread // Int32(_WARP)
        lane = thread % Int32(_WARP)
        is_value = lane >= Int32(self.groups)
        group = lane % Int32(self.groups)

        slot = Int64(slot_mapping[token])
        # Every lane of the CTA shares one slot, so the branch is uniform and
        # the butterfly reductions below always see the full warp.
        if (slot >= Int64(0)) & (slot < slot_capacity):
            element = head.to(Int64) * Int64(self.head_dim) + (
                group * Int32(GROUP_SIZE)
            ).to(Int64)
            source = cutlass.select_(
                is_value,
                _address(value, token * value_row_stride + element),
                _address(key, token * key_row_stride + element),
            )
            values = cute.make_rmem_tensor((GROUP_SIZE,), Float32)
            for pair in cutlass.range_constexpr(GROUP_SIZE // 2):
                word = ld_global_nc_u32(source + Int64(4 * pair))
                low, high = _bf16x2_to_f32x2(word)
                values[2 * pair] = low
                values[2 * pair + 1] = high

            group_amax = max_abs_16(values)
            row_amax = group_amax
            # Offsets below 16 keep the key half and the value half separate.
            for offset in cutlass.range_constexpr(4):
                other = cute.arch.shuffle_sync_bfly(row_amax, offset=1 << offset)
                row_amax = fmax_f32(row_amax, other)
            outer = row_amax * Float32(TWO_LEVEL_RCP)

            page = slot // Int64(self.page_size)
            offset_in_page = slot - page * Int64(self.page_size)
            key_record = (
                page * Int64(self.key_strides[0])
                + offset_in_page * Int64(self.key_strides[1])
                + head.to(Int64) * Int64(self.key_strides[2])
            )
            value_record = (
                page * Int64(self.value_strides[0])
                + offset_in_page * Int64(self.value_strides[1])
                + head.to(Int64) * Int64(self.value_strides[2])
            )
            record = cutlass.select_(
                is_value,
                _address(value_cache, value_record),
                _address(key_cache, key_record),
            )

            scale_u32 = Uint32(0)
            packed = Uint64(0)
            if outer != Float32(0.0):
                inverse_outer = rcp_approx_ftz(outer)
                scale_u32 = cvt_f32_to_e4m3(
                    (group_amax * inverse_outer) * rcp_approx_ftz(Float32(_E2M1_MAX))
                )
                decoded_scale = cvt_e4m3_to_f32_via_f16(scale_u32)
                if decoded_scale != Float32(0.0):
                    packed = quantize_and_pack_16_fast(
                        values, rcp_approx_ftz(decoded_scale) * inverse_outer
                    )
            data = record + (group * Int32(GROUP_SIZE // 2)).to(Int64)
            st_global_u32(data, Uint32(packed & Uint64(0xFFFFFFFF)))
            st_global_u32(data + Int64(4), Uint32(packed >> Uint64(32)))
            st_global_u8(
                record + Int64(self.scale_offset) + group.to(Int64),
                cutlass.Uint8(scale_u32 & Uint32(0xFF)),
            )
            if group == Int32(0):
                st_global_f32(record + Int64(self.outer_offset), outer)


_LOCK = RLock()
_WRITER_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
register_program_cache(_WRITER_CACHE, lock=_LOCK)


def _pointer(tensor: torch.Tensor, dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype, 16, cute.AddressSpace.gmem, assumed_align=max(1, dtype.width // 8)
    )


def validate_record_caches(
    key_cache: torch.Tensor, value_cache: torch.Tensor, *, kv_heads: int, head_dim: int
) -> None:
    """Check that K/V record caches can be written and read with four-byte accesses."""
    width = record_nbytes(head_dim)
    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        if cache.dtype != torch.uint8 or cache.ndim != 4:
            raise ValueError(
                f"{name} must be uint8 [pages, page_size, kv_heads, record]"
            )
        if tuple(map(int, cache.shape[2:])) != (int(kv_heads), width):
            raise ValueError(f"{name} must hold {kv_heads} heads of {width}-byte records")
        if int(cache.stride(3)) != 1:
            raise ValueError(f"{name} records must be contiguous")
        if cache.data_ptr() % 4 or any(
            int(stride) % 4 for stride in cache.stride()[:3]
        ):
            raise ValueError(f"{name} record offsets must be four-byte aligned")
    if key_cache.shape[:2] != value_cache.shape[:2]:
        raise ValueError("key_cache and value_cache page geometry must match")
    if key_cache.device != value_cache.device or not key_cache.is_cuda:
        raise ValueError("NVFP4 K/V caches must share one CUDA device")


def _validate_write(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    for name, rows in (("key", key), ("value", value)):
        if rows.dtype != torch.bfloat16 or rows.ndim != 3:
            raise ValueError(f"{name} must be BF16 [tokens, kv_heads, head_dim]")
        if int(rows.stride(2)) != 1 or int(rows.stride(1)) != int(rows.shape[2]):
            raise ValueError(f"{name} heads must be contiguous within a token row")
        if rows.data_ptr() % 4 or int(rows.stride(0)) % 2:
            raise ValueError(f"{name} rows must be four-byte aligned")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    tokens, heads, head_dim = map(int, key.shape)
    validate_record_caches(key_cache, value_cache, kv_heads=heads, head_dim=head_dim)
    if slot_mapping.dtype != torch.int64 or slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be int64 [tokens]")
    if not slot_mapping.is_contiguous() or int(slot_mapping.shape[0]) > tokens:
        raise ValueError("slot_mapping must be contiguous and cover at most the rows")
    devices = {t.device for t in (key, value, key_cache, value_cache, slot_mapping)}
    if len(devices) != 1 or not key.is_cuda:
        raise ValueError("NVFP4 K/V writer tensors must share one CUDA device")


def _writer_key(
    key: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor
) -> tuple[object, ...]:
    device_index = key.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        int(device_index),
        int(key.shape[1]),
        int(key.shape[2]),
        int(key_cache.shape[1]),
        tuple(map(int, key_cache.stride()[:3])),
        tuple(map(int, value_cache.stride()[:3])),
    )


def compile_nvfp4_kv_writer(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> Callable[..., None]:
    """Compile (or return) the writer for this head count and cache layout."""
    cache_key = _writer_key(key, key_cache, value_cache)
    with _LOCK:
        cached = _WRITER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        _, heads, head_dim, page_size, key_strides, value_strides = cache_key
        kernel = Nvfp4KVWriteKernel(
            kv_heads=heads,
            head_dim=head_dim,
            page_size=page_size,
            key_strides=key_strides,
            value_strides=value_strides,
        )
        with torch.cuda.device(cache_key[0]):
            raise_if_kernel_resolution_frozen(
                "cute.compile", target=kernel, cache_key=cache_key
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(cutlass.BFloat16),
                _fake_pointer(cutlass.BFloat16),
                _fake_pointer(cutlass.Uint8),
                _fake_pointer(cutlass.Uint8),
                _fake_pointer(Int64),
                Int64(1),
                Int64(1),
                Int64(1),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.paged.nvfp4_kv_writer",
                    1,
                    (heads, head_dim, page_size, key_strides, value_strides),
                    labels=(
                        "kv_heads",
                        "head_dim",
                        "page_size",
                        "key_cache_strides",
                        "value_cache_strides",
                    ),
                ),
            )
        _WRITER_CACHE[cache_key] = raw
        return raw


def launch_nvfp4_kv_writer(
    compiled: Callable[..., None],
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Run a compiled writer; capture-safe (no allocation or synchronization)."""
    from b12x._lib.compiler import run_compiled

    num_tokens = int(slot_mapping.shape[0])
    if num_tokens == 0:
        return
    run_compiled(
        compiled,
        (
            _pointer(key, cutlass.BFloat16),
            _pointer(value, cutlass.BFloat16),
            _pointer(key_cache, cutlass.Uint8),
            _pointer(value_cache, cutlass.Uint8),
            _pointer(slot_mapping, Int64),
            int(key.stride(0)),
            int(value.stride(0)),
            int(key_cache.shape[0]) * int(key_cache.shape[1]),
            num_tokens,
            current_cuda_stream(),
        ),
    )


def write_nvfp4_kv(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Validate, compile on first use outside capture, and write records."""
    _validate_write(key, value, key_cache, value_cache, slot_mapping)
    if int(slot_mapping.shape[0]) == 0:
        return
    cache_key = _writer_key(key, key_cache, value_cache)
    with _LOCK:
        compiled = _WRITER_CACHE.get(cache_key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "NVFP4 K/V writer compile miss during CUDA graph capture; "
                "prepare the exact specialization before capture"
            )
        compiled = compile_nvfp4_kv_writer(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache
        )
    launch_nvfp4_kv_writer(
        compiled,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
    )


def clear_caches() -> None:
    with _LOCK:
        _WRITER_CACHE.clear()


__all__ = [
    "GROUP_SIZE",
    "NVFP4_KV_DTYPE",
    "Nvfp4KVWriteKernel",
    "clear_caches",
    "compile_nvfp4_kv_writer",
    "dequantize_nvfp4_kv_torch",
    "is_nvfp4_cache",
    "is_nvfp4_kv_dtype",
    "kv_storage",
    "launch_nvfp4_kv_writer",
    "nvfp4x8_scaled_to_bfloat2x4",
    "outer_scale_offset",
    "quantize_nvfp4_kv_torch",
    "record_nbytes",
    "scale_offset",
    "write_nvfp4_kv",
]
