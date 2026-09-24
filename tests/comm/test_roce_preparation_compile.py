"""RoCE preparation compiles the declared dtypes without connecting peers."""

import pytest
import torch

from b12x.comm.roce._preparation import compile_roce
from b12x.comm.roce._oneshot_cute import get_launcher
from b12x.comm.roce._tuning import RoceQuery, TUNING
from b12x.preparation import FrozenMapping
from b12x._lib.compile_pool import CompileJob, describe_compilation, compile_in_process
from b12x._lib.compile_plan import compiled_program_available, program_keys, load_programs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA compilation")
def test_compile_declared_dtypes():
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    query = RoceQuery(
        surface="AllReduce.all_reduce",
        world_size=4,
        rank=0,
        topology="roce_rdma",
        peer_hosts=("rank-0", "rank-1", "rank-2", "rank-3"),
        hca_names=("hca-0", "hca-1"),
        call=FrozenMapping({
            "dtypes": ("float16", "bfloat16", "float32"),
        }),
        setup=FrozenMapping({"threads": 512, "slots": 2, "flag_stride": 16, "hca_count": 2}),
    )
    payload = TUNING.encode_query(query)
    ordinal = torch.cuda.current_device()
    job = CompileJob.create("b12x.comm.roce._preparation:compile_roce", payload, ordinal)
    description = describe_compilation(job)
    assert len(description.programs) == 4
    compile_in_process((description,))
    assert all(compiled_program_available(program) for program in description.programs)
    launchers = compile_roce(payload, ordinal)
    assert set(program_keys(launchers)) == set(description.programs)
    load_programs(launchers)
    assert set(launchers) == {*dtypes, "gather"}
    assert all(callable(launcher) for launcher in launchers.values())


def test_declaration_rejects_nonserializable_dtypes():
    with pytest.raises(TypeError, match="JSON-compatible"):
        FrozenMapping({"dtypes": (torch.float16,)})


def test_unsupported_dtype_fails_alike_whatever_the_spelling():
    """The launcher boundary raises one ValueError for a torch.dtype or a name."""
    with pytest.raises(ValueError, match="unsupported RoCE one-shot dtype"):
        get_launcher(torch.float8_e4m3fn, 2, 0, 32, 4, 4, 1, 0)
    with pytest.raises(ValueError, match="unsupported RoCE one-shot dtype"):
        get_launcher("float8_e4m3fn", 2, 0, 32, 4, 4, 1, 0)
