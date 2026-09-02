"""Reviewed DSA indexer corpus.

The raced part is the compressed (C4) paged decode ladder DeepSeek-V4-Flash
and GLM-5.3-Flash serve: TP-sliced head counts, decode row counts through the
speculative bucket, and the page-table widths of the served context limits,
each scored at several live context lengths (scenarios). The remaining
production shapes (GLM-5.2's top-k-2048 decode/extend, packed-contiguous
prefill, MiniMax MSA) stay single-candidate qualification cases so the
profile keeps covering them with the runtime's own route selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .attention_corpus import COMMON_PREFILL_TOKEN_CAPACITIES
from .sweep import SweepCase

INDEX_PAGE_SIZE = 64
C4_COMPRESS_RATIO = 4


@dataclass(frozen=True)
class DsaIndexerGeometry:
    model_id: str
    num_q_heads: int
    top_k: int
    compress_ratio: int
    source: str


DSA_INDEXER_RACED_GEOMETRIES = tuple(
    DsaIndexerGeometry(
        model_id=f"deepseek-v4-flash-h{heads}",
        num_q_heads=heads,
        top_k=512,
        compress_ratio=C4_COMPRESS_RATIO,
        source="vllm deepseek_v4 b12x_indexer decode plan (TP-sliced heads)",
    )
    for heads in (64, 32, 16, 8)
)

# Decode query rows: plain decode through the DSpark speculative bucket.
DSA_INDEXER_DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64)
# Served max_model_len values; the plan's page-table width derives from them.
DSA_INDEXER_MODEL_LEN_TOKENS = (131_072, 524_288)
# Live context lengths scored inside one width (scenario-robust reduction).
DSA_INDEXER_CONTEXT_TOKENS = (16_384, 65_536, 131_072)


def page_table_width(model_len_tokens: int, *, compress_ratio: int) -> int:
    rows = -(-int(model_len_tokens) // int(compress_ratio))
    return max(1, -(-rows // INDEX_PAGE_SIZE))


def _paged_decode_query(
    geometry: DsaIndexerGeometry, *, rows: int, width: int
) -> dict[str, object]:
    return {
        "source_layout": "paged",
        "mode": "decode",
        "dtype": "bfloat16",
        "kv_dtype": "uint8",
        "num_q_heads": geometry.num_q_heads,
        "num_idx_heads": 1,
        "max_q_rows": int(rows),
        "max_k_rows": 0,
        "max_page_table_width": int(width),
        "top_k": geometry.top_k,
        "page_size": INDEX_PAGE_SIZE,
        "score_mode": "dsa",
        "shared_page_table": False,
    }


def raced_dsa_indexer_cases() -> tuple[SweepCase, ...]:
    cases = []
    for geometry in DSA_INDEXER_RACED_GEOMETRIES:
        for model_len in DSA_INDEXER_MODEL_LEN_TOKENS:
            width = page_table_width(model_len, compress_ratio=geometry.compress_ratio)
            for rows in DSA_INDEXER_DECODE_ROWS:
                for context in DSA_INDEXER_CONTEXT_TOKENS:
                    if context > model_len:
                        continue
                    cases.append(
                        SweepCase.create(
                            group_id=geometry.model_id,
                            query=_paged_decode_query(geometry, rows=rows, width=width),
                            scenario=f"ctx{context}",
                            metadata={
                                "model_id": geometry.model_id,
                                "source": geometry.source,
                                "raced": True,
                                "compress_ratio": geometry.compress_ratio,
                                "context_tokens": int(context),
                                "model_len_tokens": int(model_len),
                            },
                            label=f"{geometry.model_id}-m{rows}-w{width}",
                        )
                    )
    return tuple(cases)


def _qualification_query(**fields: object) -> dict[str, object]:
    query = {
        "source_layout": "paged",
        "mode": "decode",
        "dtype": "bfloat16",
        "kv_dtype": "uint8",
        "num_q_heads": 32,
        "num_idx_heads": 1,
        "max_q_rows": 4,
        "max_k_rows": 0,
        "max_page_table_width": 64,
        "top_k": 2_048,
        "page_size": INDEX_PAGE_SIZE,
        "score_mode": "dsa",
        "shared_page_table": False,
    }
    query.update(fields)
    return query


def _dsa_probe(mode: str, rows: int, cache_len: int, heads: int, top_k: int):
    return {
        "kind": "dsa",
        "mode": mode,
        "rows": int(rows),
        "cache_len": int(cache_len),
        "heads": int(heads),
        "top_k": int(top_k),
    }


def _msa_probe(mode: str, rows: int, heads: int, width: int):
    return {
        "kind": "msa",
        "mode": mode,
        "rows": int(rows),
        "heads": int(heads),
        "width": int(width),
    }


def _msa_query(**fields: object) -> dict[str, object]:
    return _qualification_query(
        dtype="float8_e4m3fn",
        num_q_heads=1,
        num_idx_heads=4,
        top_k=16,
        score_mode="msa",
        **fields,
    )


def qualification_dsa_indexer_cases() -> tuple[SweepCase, ...]:
    """Production shapes kept on the runtime's own route selection.

    Each carries the benchmark-harness probe (from the former qualification
    generator) that times it; the profile records the timing as evidence.
    """
    entries: list[tuple[str, str, dict[str, object], dict[str, object]]] = [
        (
            "glm52",
            "glm52-decode-spec4",
            _qualification_query(max_q_rows=4),
            _dsa_probe("decode", 4, 4_096, 32, 2_048),
        ),
        (
            "glm53-flash",
            "glm53-pooled-spec6",
            _qualification_query(max_q_rows=6, top_k=512),
            _dsa_probe("decode", 6, 4_096, 32, 512),
        ),
        (
            "glm52",
            "glm52-decode-bucket16",
            _qualification_query(max_q_rows=16),
            _dsa_probe("decode", 16, 4_096, 32, 2_048),
        ),
        (
            "glm52",
            "glm52-extend",
            _qualification_query(mode="extend", max_q_rows=128),
            _dsa_probe("extend", 1, 4_096, 32, 2_048),
        ),
        (
            "glm52-prefill",
            "glm52-prefill-16384",
            _qualification_query(
                source_layout="contiguous",
                mode="prefill",
                max_q_rows=16_384,
                max_k_rows=131_072,
                max_page_table_width=0,
            ),
            _dsa_probe("extend", 16_384 // 128, 16_384, 32, 2_048),
        ),
        (
            "minimax-m3-msa",
            "minimax-m3-msa-decode",
            _msa_query(max_q_rows=4, max_page_table_width=128),
            _msa_probe("decode", 4, 4, 8_192),
        ),
        (
            "minimax-m3-msa",
            "minimax-m3-msa-prefill",
            _msa_query(
                source_layout="contiguous",
                mode="prefill",
                max_q_rows=16,
                max_k_rows=8_192,
                max_page_table_width=0,
            ),
            _msa_probe("prefill", 16, 4, 8_192),
        ),
    ]
    for tokens in COMMON_PREFILL_TOKEN_CAPACITIES:
        entries.append(
            (
                "glm52-prefill",
                f"glm52-extend-m{tokens}",
                _qualification_query(
                    source_layout="contiguous",
                    mode="prefill",
                    max_q_rows=tokens,
                    max_k_rows=16_384,
                    max_page_table_width=0,
                ),
                _dsa_probe("extend", tokens // 128, 16_384, 32, 2_048),
            )
        )
        entries.append(
            (
                "minimax-m3-msa",
                f"minimax-m3-msa-prefill-m{tokens}",
                _msa_query(
                    source_layout="contiguous",
                    mode="prefill",
                    max_q_rows=tokens,
                    max_k_rows=8_192,
                    max_page_table_width=0,
                ),
                _msa_probe("prefill", tokens, 4, 8_192),
            )
        )
    cases = []
    seen: set[tuple[object, ...]] = set()
    for group_id, label, query, probe in entries:
        key = tuple(sorted(query.items()))
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            SweepCase.create(
                group_id=group_id,
                query=query,
                metadata={
                    "model_id": group_id,
                    "source": "production DSA and MSA indexer qualification",
                    "raced": False,
                    "probe": {**probe, "label": label},
                },
                label=label,
            )
        )
    return tuple(cases)


def dsa_indexer_cases() -> tuple[SweepCase, ...]:
    return (*raced_dsa_indexer_cases(), *qualification_dsa_indexer_cases())


def dsa_indexer_corpus_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "raced_geometries": [asdict(item) for item in DSA_INDEXER_RACED_GEOMETRIES],
        "decode_rows": list(DSA_INDEXER_DECODE_ROWS),
        "model_len_tokens": list(DSA_INDEXER_MODEL_LEN_TOKENS),
        "context_tokens": list(DSA_INDEXER_CONTEXT_TOKENS),
    }


__all__ = [
    "C4_COMPRESS_RATIO",
    "DSA_INDEXER_CONTEXT_TOKENS",
    "DSA_INDEXER_DECODE_ROWS",
    "DSA_INDEXER_MODEL_LEN_TOKENS",
    "DSA_INDEXER_RACED_GEOMETRIES",
    "INDEX_PAGE_SIZE",
    "DsaIndexerGeometry",
    "dsa_indexer_cases",
    "dsa_indexer_corpus_manifest",
    "page_table_width",
    "qualification_dsa_indexer_cases",
    "raced_dsa_indexer_cases",
]
