"""Two-shot host launchers expose every eager and graph-slot program.

Metadata discovery runs in an offline compiler child with CUDA execution
disabled. The test exercises the production compile factory, including every
supported transport/world/rank combination and all three slot variants.
"""

from __future__ import annotations

import multiprocessing
import traceback
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("metadata_only", (False, True))
def test_twoshot_rejects_partial_rank_rows(metadata_only):
    import torch
    from b12x.comm.pcie._twoshot_preparation import (
        query_from_metadata,
        query_from_runtime,
    )

    runtime = SimpleNamespace(device=torch.device("cpu"), row_elems=4096, world_size=4)
    # Twelve 5120-element token rows are fifteen native rows, not four equal shards.
    with pytest.raises(ValueError, match="rows must be divisible by world size"):
        if metadata_only:
            query_from_metadata(
                runtime,
                surface="PCIeTwoShotBF16.all_reduce",
                shape=(12, 5120),
                dtype=torch.bfloat16,
            )
        else:
            query_from_runtime(
                runtime,
                surface="PCIeTwoShotBF16.all_reduce",
                call={"inp": torch.empty((12, 5120), dtype=torch.bfloat16)},
            )


def _discover_programs(connection):
    try:
        from b12x._lib import compile_pool
        from b12x._lib.compile_plan import DeferredCuTeKernel, program_keys
        from b12x._lib.compile_pool import CompileJob, describe_compilation
        from b12x.comm.pcie._twoshot_preparation import compile_twoshot_surface

        activity = multiprocessing.get_context("spawn").Array("q", (0, 0))
        compile_pool._initialize_worker(
            0,
            (12, 0),
            "synthetic-twoshot-device",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            188,
            101376,
            102400,
            activity,
        )
        surfaces = {
            "PCIeTwoShotBF16.all_reduce": (4,),
            "PCIeTwoShotBF16.reduce_scatter": (4,),
            "PCIeTwoShotBF16.all_gather": (4,),
            "TwoShotReduceScatter.reduce_scatter_fp8": (2, 4, 8),
            "TwoShotReduceScatter.all_gather_fp8": (2, 4, 8),
        }
        variants = ((False, 0), (True, 0), (True, 1))
        checked = 0
        for surface, worlds in surfaces.items():
            for world in worlds:
                for rank in range(world):
                    payload = {
                        "surface": surface,
                        "world_size": world,
                        "rank": rank,
                        "topology": "pcie_ipc",
                        "call": {
                            "operation": surface.split(".")[1].removesuffix("_fp8"),
                            "threads": 512,
                            "row_elems": 4096,
                            "device_slot_variants": variants,
                        },
                        "setup": {},
                    }
                    job = CompileJob.create(
                        "b12x.comm.pcie._twoshot_preparation:compile_twoshot_surface",
                        payload,
                        0,
                    )
                    plan = describe_compilation(job)
                    assert len(plan.programs) == len(variants), (surface, world, rank)
                    wrappers = compile_twoshot_surface(payload, 0)
                    assert set(wrappers) == set(variants)
                    assert set(program_keys(wrappers)) == set(plan.programs)
                    for wrapper in wrappers.values():
                        dependencies = wrapper.__b12x_dependencies__
                        assert len(dependencies) == 1
                        assert isinstance(dependencies[0], DeferredCuTeKernel)
                        assert program_keys(wrapper) == program_keys(dependencies[0])
                    assert describe_compilation(job).programs == plan.programs
                    checked += 1
        assert activity[0] == activity[1] == 0, "discovery must not compile kernels"
        connection.send((True, checked))
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def test_twoshot_compile_factory_retains_all_program_dependencies(
    monkeypatch, tmp_path
):
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(tmp_path / "compile"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_discover_programs, args=(child,))
    process.start()
    child.close()
    try:
        assert parent.poll(120), "two-shot metadata discovery timed out"
        success, result = parent.recv()
        assert success, result
        assert result == 40
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
