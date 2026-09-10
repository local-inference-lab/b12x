"""In-place ascending sort of sparse-attention top-k selections with
conversion of logical KV positions to physical cache slots.

The DSA indexer (``attention.dsa_indexer``) packs the selected KV rows of a
query in the order its CTAs reserve output slots, which is an atomic arrival
order and therefore follows kernel timing. The sparse MLA decode kernel then
sums the selected rows in that order in fp32, so a change in the timing of
the kernels around the indexer (a concurrent weight prefetch, a different
launch sequence) moves the summation order and, through the BF16 rounding of
the layer output, the deep-tail log probabilities of long contexts. Sorting
each row makes the order a function of the selected set alone: the same set
gives bit-identical attention whatever the surrounding timing.

``sort_convert(indices, seq_lens, block_table, block_size, max_positions)``:
each row of ``indices`` (int32 ``[rows, topk]``) holds logical KV positions
below ``seq_lens[row]`` (``-1`` unused). One CTA of 256 threads per row marks
them in a shared-memory bitmap, emits the set bits ascending by warp prefix
sums over rounds of 1024 words and converts each position to its physical
slot ``block_table[row, pos >> log2(block_size)] << log2(block_size) | pos &
(block_size - 1)`` in place, ``-1`` filling the tail. Slots are int32: a
position whose block lies beyond the table, whose page is negative, or whose
page exceeds ``INT32_MAX >> log2(block_size)`` (its slots would not fit the
signed 32-bit range) is written as ``-1`` instead of a wrapped slot. The
sorted order of a row depends on the selected set alone; the slots also
depend on the row's block table and the block size. ``max_positions`` (the
model length) sizes the bitmap and is the only compile key besides the
device and toolchain identity; the row count, ``topk`` and the strides are
runtime launch arguments. The kernel allocates nothing, so it can be launched
on a side stream inside a CUDA-graph capture; ``precompile`` compiles and
warm-runs it before capture. Compilation and launches select the stream of
the tensors' device, whatever the current device.

Example:
    from b12x.attention import topk_sort

    topk_sort.precompile(max_model_len, device)
    topk_sort.sort_convert(indices, seq_lens, block_table, 64, max_model_len)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="topk_sort",
    group="attention",
    api_style="oneshot",
    entry_points=(
        "sort_convert",
        "sort_convert_reference",
        "precompile",
        "supports",
        "is_supported",
        "is_disabled",
        "bitmap_words",
        "MAX_BITMAP_WORDS",
    ),
    dtypes=("int32",),
    # First revision of the op (CuTe DSL kernel, opaque custom op and
    # reference), on the branch that contributed it.
    provenance=Provenance(
        repo="https://github.com/joninco/b12x",
        commit="4f5ef5b2",
        paths=("b12x/attention/topk_sort/",),
    ),
    test_path="tests/attention/test_topk_sort.py",
    since="1.3.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        MAX_BITMAP_WORDS,
        bitmap_words,
        is_disabled,
        is_supported,
        precompile,
        sort_convert,
        sort_convert_reference,
        supports,
    )

install_lazy_api(globals(), META)
