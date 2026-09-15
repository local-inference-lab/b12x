"""SM103 declarations retain production programs before any device execution."""
from __future__ import annotations

import multiprocessing

import pytest


from scripts._sm103_preparation_corpus import CASES, IDENTITY, DEVICE, declare

def probe(case):
    import torch
    from b12x._lib.compile_pool import describe_compilation
    from b12x.preparation.types import _plan_scope
    declaration = declare(case)
    assert declaration.prepared is None
    tuning = declaration.contract.configure(declaration.query, device=IDENTITY, search=False)
    with _plan_scope(declaration):
        memory = declaration._memory_requirements(tuning.default, DEVICE)
    jobs = declaration._compile_jobs(tuning.default, DEVICE)
    programs = {key for job in jobs for key in describe_compilation(job).programs}
    assert programs, f"{case} retained no production programs"
    assert not torch.cuda.is_initialized()
    return len(programs), memory.scratch_nbytes, declaration.contract.config_payload(tuning.default)


@pytest.fixture(scope="module")
def compiler_worker():
    from b12x._lib import compile_pool
    context = multiprocessing.get_context("spawn")
    activity = context.Array("q", (0, 0))
    with compile_pool._offline_compiler_spawn_environment():
        pool = context.Pool(1, initializer=compile_pool._initialize_worker,
                            initargs=(0, (10, 3), DEVICE.uuid, IDENTITY.product_name, 148,
                                      DEVICE.max_shared_memory_per_block,
                                      DEVICE.max_shared_memory_per_multiprocessor, activity))
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()


@pytest.mark.parametrize("case", CASES)
def test_preparation_declaration_retains_sm103_programs_without_cuda(compiler_worker, case):
    count, _, config = compiler_worker.apply_async(probe, (case,)).get(timeout=120)
    assert count > 0
    if case == "gdn:kda":
        assert config["backend"] == "cutedsl"
    if case.startswith("dense:"):
        assert config["backend"] == "sm103"


def test_packed_fp16_preserves_quantized_recipe_and_rejects_a16():
    from dataclasses import replace
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, BlockscaledConfig, TUNING
    query = BlockscaledQuery(recipe="mxfp8", num_tokens=8, in_features=256,
                              padded_in_features=256, out_features=256, input_dtype="float16")
    TUNING.validate_query(query, IDENTITY)
    selected = TUNING.default_config(query, IDENTITY)
    assert selected.mode == "quantized"
    TUNING.validate_config(query, selected, IDENTITY)
    assert TUNING.encode_query(query)["input_dtype"] == "float16"
    with pytest.raises(ValueError, match="BF16"):
        TUNING.validate_config(query, BlockscaledConfig(mode="a16", tile_n=64, tile_k=64, split_k=1), IDENTITY)
    with pytest.raises(ValueError, match="MXFP8"):
        TUNING.validate_query(replace(query, recipe="nvfp4"), IDENTITY)


def probe_dsa_cache_eviction():
    from b12x._lib.compile_pool import describe_compilation
    from b12x._lib.compile_plan import evict_planning_artifacts, program_keys
    from b12x.attention.dsa_indexer.mxfp4 import _compile

    _compile.cache_clear()
    declaration = declare("dsa:decode")
    config = declaration.contract.configure(declaration.query, device=IDENTITY, search=False).default
    descriptions = [describe_compilation(job) for job in declaration._compile_jobs(config, DEVICE)]
    before = _compile.cache_info().currsize
    assert len(program_keys(_compile("quantize", (False, 64), 0))) == 1
    evict_planning_artifacts(program for item in descriptions for program in item.programs)
    return before, _compile.cache_info().currsize


def test_dsa_deferred_programs_leave_the_cache_before_compilation(compiler_worker):
    before, after = compiler_worker.apply_async(probe_dsa_cache_eviction).get(timeout=120)
    assert before == 5
    assert after == 0
