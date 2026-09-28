"""Compare complete QSA transactions with exact stable-selection fusion."""

import ast
from contextlib import contextmanager
from dataclasses import replace
from b12x._lib.compile_plan import compile_only_launches
from b12x.attention.qsa import _kernels
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
from b12x.attention.qsa import _contract
from b12x._lib.runtime_control import kernel_resolution_guard

OUT = Path('/evidence')
source = Path('/opt/spark-vllm/b12x/b12x/attention/qsa/_contract.py')
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_qsa_decode_impl')
namespace = dict(vars(_contract))
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
baseline = namespace['_qsa_decode_impl']
candidate = _contract._qsa_decode_impl
candidate_stabilize = _kernels.launch_stabilize_topk
legacy_spec = importlib.util.spec_from_file_location('b12x.attention.qsa._baseline_kernels', '/opt/spark-vllm/b12x/b12x/attention/qsa/_kernels.py')
legacy = importlib.util.module_from_spec(legacy_spec)
sys.modules[legacy_spec.name] = legacy
legacy_spec.loader.exec_module(legacy)
spec = importlib.util.spec_from_file_location('qsa_harness', '/candidate/benchmarks/benchmark_qsa.py')
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)


@contextmanager
def implementation(fn):
    saved = _contract._qsa_decode_impl
    saved_stabilize = _kernels.launch_stabilize_topk
    _contract._qsa_decode_impl = fn
    _kernels.launch_stabilize_topk = legacy.launch_stabilize_topk if fn is baseline else candidate_stabilize
    try:
        yield
    finally:
        _contract._qsa_decode_impl = saved
        _kernels.launch_stabilize_topk = saved_stabilize


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv,noheader'], text=True).strip()


def samples(graph):
    result = []
    for _ in range(15):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        result.append(start.elapsed_time(end) * 1000)
    return result


def main():
    report = {'status': 'research-only', 'scope': 'Isolated FP8 complete-QSA comparison, including state restoration in both graphs; exact same inputs and prepared kernels.', 'cases': []}
    for rows, context, kind in [(4, 8192, 'speculative'), (8, 8192, 'throughput'), (7520, 8192, 'prefill'), (7520, 65536, 'prefill'), (7520, 131072, 'prefill')]:
        print('prepare', rows, context, kind, flush=True)
        case = harness.BenchmarkCase(harness.PROFILES['tp4'], rows, context, kind=kind,
                                     main_page_size=1424, planned_max_batch=16,
                                     planned_max_q_rows=8192, planned_max_speculative_tokens=3)
        prepared = harness._prepare_case(case, device=torch.device('cuda'), seed=20260917,
                                         main_cache_layout='interleaved', kv_cache_dtype='fp8_e4m3')
        binding = prepared.binding
        support = dict(binding.state.programs.support)
        meta_ids = torch.empty((binding.state.workspace_q_rows, binding.state.caps.group_budget), device='meta', dtype=torch.int32)
        meta_counts = torch.empty((binding.state.workspace_q_rows,), device='meta', dtype=torch.int32)
        with compile_only_launches(), _kernels._support_context(support, compiling=True):
            for chunk in range(binding.state.num_score_chunks):
                _kernels.launch_remap_topk_group_ids(
                    local_ids=meta_ids, prior_ids=meta_ids, eligible_counts=meta_counts,
                    merge_lengths=meta_counts, group_offset=chunk * binding.state.score_chunk_groups,
                    group_budget=binding.state.caps.group_budget)
        meta_scores = torch.empty((binding.state.workspace_q_rows, binding.state.score_workspace_width), device='meta', dtype=torch.float32)
        meta_values = torch.empty_like(meta_ids, dtype=torch.float32)
        meta_hist = torch.empty((binding.state.workspace_q_rows, (binding.state.score_workspace_width+511)//512), device='meta', dtype=torch.int32)
        with compile_only_launches(), legacy._support_context(support, compiling=True):
            for chunk in range(binding.state.num_score_chunks):
                legacy.launch_stabilize_topk(
                    scores=meta_scores, merge_lengths=meta_counts, prior_ids=meta_ids,
                    eligible_counts=meta_counts, topk_values=meta_values, topk_group_ids=meta_ids,
                    tie_counts=meta_hist, greater_counts=meta_hist, stable_values=meta_values,
                    stable_ids=meta_ids, thresholds=meta_counts.to(torch.float32), greater_totals=meta_counts,
                    group_offset=chunk * binding.state.score_chunk_groups,
                    group_budget=binding.state.caps.group_budget)
        programs = replace(binding.state.programs, support=support)
        with implementation(baseline):
            prepared.state_restore.restore()
            expected = _contract._run(binding, programs=programs, **prepared.dynamic).clone()
            selected = binding.selected_positions[:rows].clone()
            state = {name: getattr(binding, name).clone() for name in ['compressed_k_cache', 'raw_k_ring', 'raw_logical_positions', 'raw_rope_positions', 'raw_interval_start_positions']}
        assert torch.isfinite(expected).all() and torch.count_nonzero(expected) > 0
        graphs = {}
        with kernel_resolution_guard('QSA selector comparison replays prepared kernels only'):
            for name, fn in [('baseline', baseline), ('fused_cleanup', candidate)]:
                with implementation(fn):
                    prepared.state_restore.restore()
                    _contract._run(binding, programs=programs, **prepared.dynamic)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        prepared.state_restore.restore()
                        _contract._run(binding, programs=programs, **prepared.dynamic)
                    binding.output.fill_(float('nan'))
                    binding.selected_positions.fill_(-98765)
                    allocation = torch.cuda.memory_allocated()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated() == allocation
                    torch.testing.assert_close(binding.output[:rows], expected, rtol=0, atol=0)
                    assert torch.equal(binding.selected_positions[:rows], selected)
                    for key, value in state.items():
                        assert torch.equal(getattr(binding, key), value), key
                    graphs[name] = graph
        times = {name: [] for name in graphs}
        before = gpu()
        for name in ['baseline', 'fused_cleanup', 'fused_cleanup', 'baseline']:
            times[name].extend(samples(graphs[name]))
        medians = {name: statistics.median(values) for name, values in times.items()}
        record = {'rows': rows, 'context': context, 'kind': kind, 'correctness': 'exact nonzero output, selected positions and persistent state; no graph replay allocation', 'median_us': medians, 'candidate_over_baseline': medians['fused_cleanup']/medians['baseline'], 'samples_us': times, 'gpu_before': before, 'gpu_after': gpu()}
        report['cases'].append(record)
        (OUT/'fusion-results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ['samples_us']}), flush=True)
        prepared.close()
        del graphs, graph, prepared, binding, expected, selected, state
        gc.collect()
        torch.cuda.empty_cache()
    report['complete'] = True
    (OUT/'fusion-results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
