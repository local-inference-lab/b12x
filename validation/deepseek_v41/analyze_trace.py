"""Summarize complete worker CUPTI traces; device sums are not wall latency."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def summarize(path: Path):
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
    events = data['traceEvents']
    boundaries = {}
    for event in events:
        name = event.get('name', '')
        if event.get('cat') == 'user_annotation' and name.startswith('dsv41/repeat='):
            label, side = name.rsplit('/', 1)
            if side in ('begin', 'end'):
                boundaries.setdefault(label, {})[side] = event['ts']
    intervals = [(label, edge['begin'], edge['end']) for label, edge in boundaries.items()
                 if 'begin' in edge and 'end' in edge]
    kernels = defaultdict(lambda: {'instances': 0, 'device_us': 0.0, 'graph_instances': 0})
    batches = defaultdict(lambda: defaultdict(lambda: {'instances': 0, 'device_us': 0.0}))
    graph_launches = 0
    for event in events:
        name = event.get('name', '')
        if event.get('cat') in ('cuda_runtime', 'cuda_driver') and 'GraphLaunch' in name:
            graph_launches += 1
        if event.get('cat') != 'kernel':
            continue
        duration = float(event.get('dur', 0.0))
        entry = kernels[name]
        entry['instances'] += 1
        entry['device_us'] += duration
        metadata = event.get('args', {})
        entry['graph_instances'] += int(bool(metadata.get('graph id', 0) or metadata.get('graph node id', 0)))
        label = next((label for label, start, end in intervals if start <= event['ts'] < end), 'between_batches')
        batch_entry = batches[label][name]
        batch_entry['instances'] += 1
        batch_entry['device_us'] += duration
    total = sum(row['device_us'] for row in kernels.values())
    return {
        'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
        'timing_kind': 'profiled_device_sum_not_wall_latency',
        'cuda_graph_launches': graph_launches,
        'kernel_instances': sum(row['instances'] for row in kernels.values()),
        'device_us': total, 'batch_boundaries': boundaries,
        'kernels': [{'name': name, **row, 'fraction_device_time': row['device_us'] / total if total else 0}
                    for name, row in sorted(kernels.items(), key=lambda item: item[1]['device_us'], reverse=True)],
        'batches': {label: [{'name': name, **row} for name, row in
                            sorted(items.items(), key=lambda item: item[1]['device_us'], reverse=True)]
                    for label, items in batches.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    workers = [summarize(path) for path in args.traces]
    result = {'schema_version': 1, 'workers': workers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for worker in workers:
        print(json.dumps({'trace': worker['path'], 'kernel_instances': worker['kernel_instances'],
                          'graph_launches': worker['cuda_graph_launches'],
                          'top_kernels': worker['kernels'][:12]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
