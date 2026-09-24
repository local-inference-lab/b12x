"""Host execution contract for MLA query projection and RoPE assembly."""

from dataclasses import asdict, dataclass, fields

from b12x.preparation import (
    BackendConfig,
    Knob,
    ParameterBinding,
    TuningContract,
)


@dataclass(frozen=True, kw_only=True)
class ProjectionQuery:
    """One prepared MLA query projection.

    ``max_rows`` is the token count M the plan is declared for.  A BF16 plan
    treats it as a capacity: its Triton launcher masks rows at runtime, so the
    plan serves every live ``1 <= M <= max_rows``.  An MXFP8 plan compiles M
    into its CuTe launcher and serves exactly ``M == max_rows``.
    """

    heads: int
    max_rows: int
    weight_format: str
    output_dtype: str
    b_major: str
    sf_axis: str

    @property
    def served_rows(self) -> range:
        """Live token counts M this plan executes."""
        first = 1 if self.weight_format == "bf16" else self.max_rows
        return range(first, self.max_rows + 1)


def validate_query(query):
    if query.b_major != "n" or query.sf_axis != "n":
        raise ValueError("MLA query projection requires N-major MXFP8 scales")
    if (
        query.heads not in ((8, 16) if query.weight_format == "mxfp8" else (8, 11, 16))
        or not 1 <= query.max_rows <= 32
        or query.weight_format not in ("bf16", "mxfp8")
        or query.output_dtype not in ("bfloat16", "float8_e4m3fn")
    ):
        raise ValueError(
            "MLA query plan requires a supported 192-to-512 projection and 64-wide RoPE suffix"
        )


def execution_backend(query):
    return "triton" if query.weight_format == "bf16" else "cutedsl"


def _validate_query(query, device) -> None:
    validate_query(query)


def _validate_config(query, config, device):
    if not isinstance(config, BackendConfig) or config.backend != execution_backend(
        query
    ):
        raise ValueError(
            "MLA query backend must match the production weight-format dispatch"
        )


TUNING = TuningContract(
    component_id="gemm.mla_query_projection",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(field.name for field in fields(ProjectionQuery)),
    config_fields=frozenset({"backend"}),
    encode_query=asdict,
    encode_config=BackendConfig.to_dict,
    decode_config=BackendConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda query, device: BackendConfig(
        backend=execution_backend(query)
    ),
    candidate_contract_version=2,
    knobs=(Knob(name="backend", values=("triton", "cutedsl"), binding=ParameterBinding.COMPILE),),
    parameters=lambda query, device: {"backend": (execution_backend(query),)},
)
