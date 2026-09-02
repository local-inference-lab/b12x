"""Generic transformer layers around the DeepSeek-V4 sparse attention chain.

Per layer: an MXFP8 neighbour stands in for the fused q-norm/RoPE and
indexer-Q projections and writes ``q`` and ``index_q`` right before use; the
DSA indexer selects top-k physical slots over that layer's own cold index-K
cache; the compressed sparse MLA reads the cold SWA and indexed caches with
those slots (a real producer/consumer dependency); the WO projection
consumes the attention output on cold weights; a closing neighbour plus
residual/RMSNorm feed the next layer. ``slot`` selects which stage the
timing events bracket.
"""

from __future__ import annotations

import math

from .layer_stack import CONTEXT_LAYERS, GenericLayerStack

DSV4_HIDDEN = 4096
DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_INDEX_HEAD_DIM = 128
DSV4_INDEX_TOPK = 512
DSV4_HEADS_PER_GROUP = 8
DSV4_WO_RANK = 1024
DSV4_COMPRESS_RATIO = 4
DSV4_PAGE = 64
DSV4_CONTEXT_TOKENS = 65536


class _DsvLayerContext(GenericLayerStack):
    def __init__(
        self,
        *,
        heads: int,
        rows: int,
        swa_width: int,
        indexed_width: int,
        context_tokens: int = DSV4_CONTEXT_TOKENS,
        page_table_width: int | None = None,
        device,
        generator,
        seed: int,
        layers: int = CONTEXT_LAYERS,
        slot: str = "attention",
    ) -> None:
        import torch

        from benchmarks.benchmark_paged_indexer import _make_page_table
        from benchmarks.benchmark_wo_projection import make_case
        from b12x.attention.dsa_indexer.paged import (
            pack_paged_index_k_cache_reference,
        )
        from b12x.gemm import wo_projection
        from b12x.policy import PolicyContext, PolicyMode

        from .gpu_workers import _compressed_cache, _sparse_indices

        if indexed_width != DSV4_INDEX_TOPK:
            raise ValueError(
                "the DSV4 layer stack models the C4 contract (indexed_width 512)"
            )
        if heads % DSV4_HEADS_PER_GROUP:
            raise ValueError("DSV4 local heads must be a multiple of 8")
        self.heads = int(heads)
        self.rows = int(rows)
        self.swa_width = int(swa_width)
        self.indexed_width = int(indexed_width)
        self.slot = slot
        in_width = self.heads * (DSV4_HEAD_DIM + DSV4_INDEX_HEAD_DIM)
        super().__init__(
            hidden=DSV4_HIDDEN,
            in_width=in_width,
            out_width=DSV4_HIDDEN,
            tokens=self.rows,
            layers=int(layers),
            device=device,
            generator=generator,
        )
        heuristic = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        self._heuristic_policy = heuristic
        compressed_tokens = max(DSV4_PAGE, int(context_tokens) // DSV4_COMPRESS_RATIO)
        pages_per_row = -(-compressed_tokens // DSV4_PAGE)
        # The plan is sized for the served page-table width (max_model_len);
        # the live context fills ``pages_per_row`` of it, as in serving.
        self.page_table_width = max(pages_per_row, int(page_table_width or 0))
        self.pages_per_row = pages_per_row
        self.compressed_tokens = compressed_tokens
        cache_pages = self.rows * pages_per_row
        cache_tokens = cache_pages * DSV4_PAGE
        self.q = []
        self.index_q = []
        self.index_weights = []
        self.topk = []
        self.o = []
        self.index_k_cache = []
        self.indexer_bindings = []
        self.indexer_plans = []
        self.swa_cache = []
        self.swa_indices = []
        self.swa_lengths = []
        self.indexed_cache = []
        self.indexed_lengths = []
        self.wo_bindings = []
        self.attention_bindings: dict[int, list] = {}
        self._attention_owners: dict[int, list] = {}
        page_table = _make_page_table(
            rows=self.rows,
            page_table_width=self.page_table_width,
            seq_len=compressed_tokens,
            page_stride=pages_per_row,
            device=device,
        )
        seqlens = torch.full(
            (self.rows,), compressed_tokens, dtype=torch.int32, device=device
        )
        self.page_table = page_table
        self.seqlens = seqlens
        self.indexer_candidates: dict[str, list] = {}
        self._default_chunk_cap: int | None = None
        for layer in range(self.layers):
            self.q.append(
                torch.empty(
                    (self.rows, self.heads, DSV4_HEAD_DIM),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )
            self.index_q.append(
                torch.empty(
                    (self.rows, self.heads, DSV4_INDEX_HEAD_DIM),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
            )
            self.index_weights.append(
                torch.rand(
                    (self.rows, self.heads), device=device, generator=generator
                ).add_(0.5)
            )
            self.topk.append(
                torch.full(
                    (self.rows, DSV4_INDEX_TOPK), -1, dtype=torch.int32, device=device
                )
            )
            self.o.append(
                torch.empty(
                    (self.rows, self.heads, DSV4_HEAD_DIM),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )
            k_source = torch.randn(
                (cache_tokens, DSV4_INDEX_HEAD_DIM), device=device, generator=generator
            ).div_(3.0)
            cache = pack_paged_index_k_cache_reference(k_source)
            self.index_k_cache.append(cache.reshape(cache_pages, -1).contiguous())
            plan, scratch, metadata, binding = self._plan_indexer(heuristic)
            self.indexer_plans.append((plan, scratch, metadata))
            self.indexer_bindings.append(binding)
            swa_tokens = max(self.swa_width * self.rows, 1)
            self.swa_cache.append(
                _compressed_cache(
                    tokens=swa_tokens,
                    page_size=DSV4_PAGE,
                    device=device,
                    generator=generator,
                )
            )
            self.swa_indices.append(
                _sparse_indices(
                    rows=self.rows,
                    width=self.swa_width,
                    tokens=swa_tokens,
                    device=device,
                )
            )
            self.swa_lengths.append(
                torch.full(
                    (self.rows,), self.swa_width, dtype=torch.int32, device=device
                )
            )
            # The indexed cache shares the index-K page numbering, so the
            # indexer's physical slots address it directly.
            self.indexed_cache.append(
                _compressed_cache(
                    tokens=cache_tokens,
                    page_size=DSV4_PAGE,
                    device=device,
                    generator=generator,
                )
            )
            self.indexed_lengths.append(
                torch.full(
                    (self.rows,), DSV4_INDEX_TOPK, dtype=torch.int32, device=device
                )
            )
            groups = self.heads // DSV4_HEADS_PER_GROUP
            data = make_case(
                tokens=self.rows,
                groups=groups,
                group_width=DSV4_HEADS_PER_GROUP * DSV4_HEAD_DIM,
                rank=DSV4_WO_RANK,
                hidden=DSV4_HIDDEN,
                seed=int(seed) + 1 + layer,
                inv_rope=True,
                context_length=max(4096, self.rows),
                nope_dim=DSV4_NOPE_DIM,
                rope_dim=DSV4_ROPE_DIM,
            )
            caps = wo_projection.Caps(
                device=device,
                max_tokens=self.rows,
                groups=groups,
                group_width=DSV4_HEADS_PER_GROUP * DSV4_HEAD_DIM,
                rank=DSV4_WO_RANK,
                hidden=DSV4_HIDDEN,
            )
            wo_plan = wo_projection.plan(caps, policy=heuristic)
            spec = wo_plan.scratch_specs()[0]
            wo_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            self.wo_bindings.append(
                (
                    wo_projection.bind_inv_rope(
                        wo_plan,
                        scratch=wo_scratch,
                        o=self.o[layer],
                        positions=data["positions"],
                        cos_sin_cache=data["cos_sin_cache"],
                        weights=data["weights"],
                        heads_per_group=DSV4_HEADS_PER_GROUP,
                        nope_dim=DSV4_NOPE_DIM,
                        rope_dim=DSV4_ROPE_DIM,
                        expected_m=self.rows,
                    ),
                    data,
                    wo_scratch,
                )
            )
        self.sm_scale = 1.0 / math.sqrt(DSV4_HEAD_DIM)
        self._indexer_supertile = int(self.indexer_plans[0][0].layout.supertile_tokens)

    def _plan_indexer(self, policy):
        """Plan, allocate and bind one layer's indexer under ``policy``."""
        import torch

        from b12x.attention.dsa_indexer._impl import uses_paged_mqa_schedule
        from b12x.attention.dsa_indexer.paged import prepare_paged_indexer_metadata
        from b12x.attention.dsa_indexer.scratch import (
            INDEXER_SOURCE_LAYOUT_PAGED,
            B12XIndexerScratchCaps,
            plan_indexer_scratch,
        )

        plan = plan_indexer_scratch(
            B12XIndexerScratchCaps(
                device=self.device,
                source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
                num_q_heads=self.heads,
                max_q_rows=self.rows,
                max_page_table_width=self.page_table_width,
                topk=DSV4_INDEX_TOPK,
                page_size=DSV4_PAGE,
                reserve_paged_logits=False,
                mode="decode",
                shared_page_table=False,
            ),
            policy=policy,
        )
        metadata = prepare_paged_indexer_metadata(
            real_page_table=self.page_table,
            cache_seqlens_int32=self.seqlens,
            expected_num_q_heads=self.heads,
            schedule_out=None,
            build_schedule=uses_paged_mqa_schedule(
                q_rows=self.rows, max_pages=self.page_table_width
            ),
            shared_page_table=False,
        )
        scratch = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for shape, dtype in plan.shapes_and_dtypes()
        ]
        binding = plan.bind(
            scratch=scratch,
            real_page_table=metadata.real_page_table,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            schedule_metadata=metadata.schedule_metadata,
            expected_num_q_heads=self.heads,
            shared_page_table=False,
            output_physical_slots=True,
        )
        return plan, scratch, metadata, binding

    def prepare_indexer(self, candidate_id: str, config) -> str:
        """Plan/bind the indexer with a candidate config on every layer.

        Returns the route the plan resolved. Raises ``ValueError`` when the
        config is not available for this shape, exactly as serving would.
        """
        from b12x.attention.dsa_indexer._policy import DsaIndexerConfig
        from b12x.policy import DSA_INDEXER

        if candidate_id in self.indexer_candidates:
            return str(self.indexer_candidates[candidate_id][0][0].layout.route)
        policy = self._heuristic_policy.with_override(
            DSA_INDEXER, DsaIndexerConfig(**dict(config))
        )
        planned = [self._plan_indexer(policy) for _ in range(self.layers)]
        self.indexer_candidates[candidate_id] = planned
        return str(planned[0][0].layout.route)

    def release_indexer(self, candidate_id: str) -> None:
        self.indexer_candidates.pop(candidate_id, None)

    def indexer_reference_values(self, layer: int):
        """Sorted reference top-k logits per row for ``layer``'s inputs."""
        import torch

        from b12x.attention.dsa_indexer.reference import paged_decode_logits_reference

        logits = paged_decode_logits_reference(
            q_fp8=self.index_q[layer],
            weights=self.index_weights[layer],
            index_k_cache=self.index_k_cache[layer],
            real_page_table=self.page_table,
            query_row_to_batch=torch.arange(
                self.rows, dtype=torch.int32, device=self.device
            ),
            seqlens_per_query=self.seqlens,
            page_size=DSV4_PAGE,
        )
        values = torch.topk(logits, DSV4_INDEX_TOPK, dim=1).values
        return logits, values

    def indexer_selected_values(self, layer: int, logits):
        """Reference logits at the physical slots the indexer selected."""
        import torch

        slots = self.topk[layer].to(torch.int64)
        valid = slots >= 0
        page = torch.div(slots, DSV4_PAGE, rounding_mode="floor")
        row_base = (
            torch.arange(self.rows, device=self.device, dtype=torch.int64)
            * self.pages_per_row
        ).unsqueeze(1)
        logical = (page - row_base) * DSV4_PAGE + slots % DSV4_PAGE
        in_range = valid & (logical >= 0) & (logical < logits.shape[1])
        gathered = torch.gather(logits, 1, logical.clamp_(0, logits.shape[1] - 1))
        gathered = torch.where(in_range, gathered, torch.full_like(gathered, -math.inf))
        return torch.sort(gathered, dim=1, descending=True).values, bool(
            in_range.all().item()
        )

    def prepare_attention(self, chunk_cap: int) -> None:
        """Plan/bind the sparse MLA with ``chunk_cap`` on every layer."""
        import torch

        from b12x.attention import compressed_sparse_mla

        if self._default_chunk_cap is None:
            self._default_chunk_cap = int(chunk_cap)
        if chunk_cap in self.attention_bindings:
            return
        bindings, owners = [], []
        width = self.swa_width + self.indexed_width
        for layer in range(self.layers):
            plan = compressed_sparse_mla.plan(
                compressed_sparse_mla.Caps(
                    device=self.device,
                    num_q_heads=self.heads,
                    max_q_rows=self.rows,
                    max_width=width,
                    head_dim=DSV4_HEAD_DIM,
                    v_head_dim=DSV4_HEAD_DIM,
                    max_batch=self.rows,
                    page_size=DSV4_PAGE,
                    layout="compressed_dsv4",
                    mode="decode",
                    swa_width=self.swa_width,
                    indexed_width=self.indexed_width,
                    swa_page_size=DSV4_PAGE,
                    indexed_page_size=DSV4_PAGE,
                    use_cuda_graph=True,
                    max_chunks_per_row=int(chunk_cap),
                )
            )
            (spec,) = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
            binding = compressed_sparse_mla.bind(
                plan,
                scratch=scratch,
                q=self.q[layer],
                swa_indices=self.swa_indices[layer],
                swa_lengths=self.swa_lengths[layer],
                indexed_indices=self.topk[layer],
                indexed_lengths=self.indexed_lengths[layer],
            )
            bindings.append(binding)
            owners.append((plan, scratch))
        self.attention_bindings[int(chunk_cap)] = bindings
        self._attention_owners[int(chunk_cap)] = owners

    def _run_indexer(self, layer: int, candidate_id: str | None = None) -> None:
        from b12x.attention.dsa_indexer.paged import index_topk_fp8

        if candidate_id is None:
            binding = self.indexer_bindings[layer]
            supertile = self._indexer_supertile
        else:
            plan, _scratch, _metadata, binding = self.indexer_candidates[candidate_id][
                layer
            ]
            supertile = int(plan.layout.supertile_tokens)
        index_topk_fp8(
            q_fp8=self.index_q[layer],
            weights=self.index_weights[layer],
            index_k_cache=self.index_k_cache[layer],
            binding=binding,
            topk=DSV4_INDEX_TOPK,
            expected_num_q_heads=self.heads,
            out_indices=self.topk[layer],
            supertile_k=supertile,
        )

    def _run_attention(self, layer: int, chunk_cap: int) -> None:
        from b12x.attention import compressed_sparse_mla

        compressed_sparse_mla.run(
            binding=self.attention_bindings[int(chunk_cap)][layer],
            swa_k_cache=self.swa_cache[layer],
            swa_page_size=DSV4_PAGE,
            indexed_k_cache=self.indexed_cache[layer],
            indexed_page_size=DSV4_PAGE,
            sm_scale=self.sm_scale,
            expected_num_q_heads=self.heads,
            out=self.o[layer],
        )

    def _run_wo(self, layer: int):
        return self.wo_bindings[layer][0].run()

    # GenericLayerStack hooks. ``slot`` is (stage, config): only the stage
    # under test is bracketed; the other stages run as its neighbours. The
    # indexer slot's config is a prepared candidate id, the attention slot's
    # config is a chunk cap.
    def produce(self, layer: int, activation) -> None:
        rows, heads = self.rows, self.heads
        q_width = heads * DSV4_HEAD_DIM
        self.q[layer].copy_(activation[:, :q_width].view(rows, heads, DSV4_HEAD_DIM))
        self.index_q[layer].copy_(
            activation[:, q_width:].view(rows, heads, DSV4_INDEX_HEAD_DIM)
        )
        if self.slot != "indexer":
            self._run_indexer(layer)

    def tested(self, layer: int, slot):
        stage, config = slot
        if stage == "indexer":
            self._run_indexer(layer, str(config))
            return None
        self._run_attention(layer, config)
        return None

    def consume(self, layer: int, output):
        if self.slot == "indexer":
            if self._default_chunk_cap is None:
                raise RuntimeError(
                    "prepare_attention() must run before the indexer slot"
                )
            self._run_attention(layer, self._default_chunk_cap)
        return self._run_wo(layer).reshape(self.rows, DSV4_HIDDEN)


__all__ = ["_DsvLayerContext", "DSV4_CONTEXT_TOKENS"]
