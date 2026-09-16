"""RoCE preparation compiles the declared dtypes without connecting peers."""

import pytest
import torch

from b12x.comm.roce._preparation import compile_roce
from b12x.comm.roce._tuning import RoceQuery, TUNING
from b12x.preparation import FrozenMapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA compilation")
@pytest.mark.parametrize("dtype_objects", [False, True])
def test_compile_declared_dtypes(dtype_objects):
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    query = RoceQuery(
        surface="AllReduce.all_reduce",
        world_size=4,
        rank=0,
        topology="roce_rdma",
        peer_hosts=("rank-0", "rank-1", "rank-2", "rank-3"),
        hca_names=("hca-0", "hca-1"),
        call=FrozenMapping({
            "dtypes": dtypes if dtype_objects else ("float16", "bfloat16", "float32"),
        }),
        setup=FrozenMapping({"threads": 512, "slots": 2, "flag_stride": 16, "hca_count": 2}),
    )
    launchers = compile_roce(TUNING.encode_query(query), torch.cuda.current_device())
    assert set(launchers) == {*dtypes, "gather"}
    assert all(callable(launcher) for launcher in launchers.values())
