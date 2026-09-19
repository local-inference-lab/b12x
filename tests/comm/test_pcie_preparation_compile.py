"""PCIe owner plans retain compiler identities without connecting peer GPUs."""

import pytest
import torch

from b12x._lib.compile_plan import compiled_program_available, load_programs, program_keys
from b12x._lib.compile_pool import CompileJob, compile_in_process, describe_compilation
from b12x.comm.pcie._owner_preparation import compile_owner_surface
from b12x.comm.pcie._tuning import PcieQuery, TUNING
from b12x.preparation import FrozenMapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA compilation")
@pytest.mark.parametrize("surface,vectorized", [
    ("PCIeHierarchicalAllReduce.all_reduce", False),
    ("PCIeHierarchicalAllReduce.all_reduce", True),
    ("PCIeIslandRSAllReduce.all_reduce", False),
    ("VocabParallelArgmax.fused_add_argmax", False),
])
def test_owner_compile_factory_retains_native_program(surface, vectorized):
    query = PcieQuery(
        surface=surface, world_size=16, rank=0, topology="pcie_ipc",
        call=FrozenMapping({
            "threads": 128 if "IslandRS" in surface else 256,
            "wait_nanosleep_cycles": 24, "double_buffered": False,
            "deferred_consumption": True, "vectorized": vectorized,
        }),
        setup=FrozenMapping(),
    )
    payload = TUNING.encode_query(query)
    ordinal = torch.cuda.current_device()
    job = CompileJob.create(
        "b12x.comm.pcie._owner_preparation:compile_owner_surface", payload, ordinal
    )
    description = describe_compilation(job)
    assert len(description.programs) == 1
    compile_in_process((description,))
    assert all(compiled_program_available(key) for key in description.programs)
    launchers = compile_owner_surface(payload, ordinal)
    assert set(program_keys(launchers)) == set(description.programs)
    load_programs(launchers)
