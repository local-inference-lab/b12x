import pytest
import torch

from b12x._lib.compile_plan import program_keys, retained_program_keys
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import mgroup_fp8_gemm as mgg
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from tests._reference.helpers import require_b12x


@pytest.mark.parametrize("mode", ["masked", "contiguous"])
def test_grouped_autotune_producer_stream_gate_retention_and_cache(mode, tmp_path):
    device = require_b12x()
    groups, rows, n, k = 2, 128, 128, 128
    shape = (groups, rows, k) if mode == "masked" else (groups * rows, k)
    gran = 128 if mode == "masked" else 32
    a = torch.ones(shape, device=device, dtype=torch.float8_e4m3fn)
    sfa = torch.ones((*shape[:-1], k // gran), device=device)
    b = torch.ones((groups, n, k), device=device, dtype=torch.float8_e4m3fn)
    sfb = torch.ones((groups, n, k // 128), device=device)
    output = torch.empty((*shape[:-1], n), device=device, dtype=torch.bfloat16)
    activity = (torch.full((groups,), rows, device=device, dtype=torch.int32)
                if mode == "masked" else torch.arange(groups, device=device, dtype=torch.int32).repeat_interleave(rows))
    lhs, rhs = (a, sfa), (b, sfb)
    query = mgg.query_from_call(lhs, rhs, output)
    produced, created, closed = [], [], []
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    for cached in (False, True):
        plan = mgg.plan(query)
        def call(state):
            token = len(created)
            created.append(token)
            run = state.run_masked if mode == "masked" else state.run_contiguous

            def produce():
                produced.append(torch.cuda.current_stream(device).cuda_stream)
                sfa.fill_(2.0 if len(produced) % 2 else 4.0)

            return PreparedCall(run=lambda: run(lhs, rhs, output, activity),
                                produce=produce, close=lambda: closed.append(token),
                                owners=(a, sfa, b, sfb, activity, output))

        with torch.cuda.stream(stream), PreparationSession(
            device=device, autotune=True, cache_dir=tmp_path, compile_workers=2,
            samples=3, rounds=1, race_batch=4,
        ) as session:
            result = session.prepare((plan.request(name="grouped", prepare_call=call, benchmark_call=call),))
            if cached:
                assert result.benchmarked_candidates == 0
            else:
                assert result.benchmarked_candidates > 1
            assert (plan.selection.source == "cached") is cached
            state = require_prepared(plan, "gemm.mgroup_fp8_gemm")
            payload = plan._prepared
            keys = set(program_keys(state.gemm))
            assert keys and keys <= payload.programs and payload.retained is not None
            assert plan in payload.users and keys <= set(retained_program_keys())
            triton_names = {key.name for key in payload.programs if key.dialect == "triton"}
            expected_names = ({"_pack_ue8m0_mma_kernel"} if mode == "masked"
                              else {"_pack_contiguous", "_zero_flat"})
            if state.config.implementation == "joint_v1":
                expected_names.add("_mark_active_groups")
            assert expected_names <= triton_names
            assert payload.programs <= set(retained_program_keys())
            session.freeze()
            probe = call(state)
            try:
                with kernel_resolution_guard():
                    for _ in range(2):
                        probe.produce()
                        probe.invoke()
                        stream.synchronize()
                        expected_scale = 2 if len(produced) % 2 else 4
                        torch.testing.assert_close(output, torch.full_like(output, k * expected_scale), rtol=0, atol=0)
            finally:
                probe.close()
            session.release(plan)
            assert plan.prepared is None and payload.closed and not payload.users
            assert payload.retained is None and payload.state is None
        assert sorted(closed) == created
    assert produced and set(produced) == {stream.cuda_stream}
