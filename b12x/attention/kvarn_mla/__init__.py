"""KVarN packed-latent staging and native sparse-MLA decode.

Reads KVarN G64 tiles (2/4/5-bit packed latent, exact BF16 RoPE) directly:

* ``stage_*`` functions rewrite packed or exact rows into the canonical FP8
  record format the promoted SM120 sparse-MLA runtime already consumes;
* ``stage_compact_kvarn_native_history`` /
  ``materialize_compact_kvarn_native_records`` move one rank's live pages
  through the compact DCP wire and back;
* ``native_packed_k5_decode`` decodes straight from packed pages plus the
  live exact-slot pool, backed by the CuTeDSL grids in
  ``b12x.attention._shared.mla.kernel``.
"""

from __future__ import annotations

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="kvarn_mla",
    group="attention",
    api_style="oneshot",
    entry_points=(
        "compact_kvarn_native_rank_nbytes",
        "stage_compact_kvarn_native_history",
        "materialize_compact_kvarn_native_records",
        "stage_k5_as_fp8_records",
        "stage_bf16_as_exact_pool_fp8_records",
        "stage_bf16_sylvester_as_exact_pool_fp8_records",
        "native_packed_k5_decode",
        "is_kvarn_mla_supported",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("glm_nsa",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/JMPSequeira/glm52-kvarn-k4v2-runtime",
        commit="a84759313d2e5525aef0a9189edbbdd16a3f18c9",
        paths=(
            "active-r634-b12x-compact/kvarn_api_k4.py",
            "active-r634-kvarn-k4-native/kvarn_mla/io.py",
            "active-r634-kvarn-k4-native/_shared/mla/kernel.py",
        ),
    ),
    test_path="tests/attention/test_attention_kvarn_mla.py",
    since="1.3.0",
)

install_lazy_api(globals(), META)
