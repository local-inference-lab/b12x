"""Fixed-backend policy for Engram metadata hashing and row-sharded gather."""
from dataclasses import dataclass

from b12x.policy import BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class EngramQuery:
    max_tokens: int
    max_seqs: int
    max_requests: int
    vocab_size: int
    compressed_vocab_size: int
    layer_id: int
    table_rows: int
    tp_size: int


ENGRAM_POLICY = make_fixed_backend_policy(
    component_id="sequence.engram", query_type=EngramQuery, backend="triton",
)
EngramConfig = BackendConfig
