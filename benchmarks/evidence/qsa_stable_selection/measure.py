"""Measure matched cold prefill and synchronized fixed-batch Qwen decoding."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import os
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import urllib.request

ROOT = Path(os.environ.get('QSA_BENCH_ROOT', '/home/jasonc/spark_vllm'))
OUT = Path(os.environ.get('QSA_BENCH_OUT', '.')).resolve()
BASE = os.environ.get('QSA_BENCH_URL', 'http://maxwell:8000')
MODEL = 'Qwen3.8-Flash-Next'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--depth', type=int, choices=(2, 3, 4), required=True)
parser.add_argument('--profile', type=Path)
parser.add_argument('--label')
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--skip-shared', action='store_true')
args = parser.parse_args()
PROFILE = json.loads((args.profile or ROOT / f'deployments/qwen38-qsa-mtp{args.depth}-20260916/profile.json').read_text())
TARGET = OUT / (args.label or f'performance-mtp{args.depth}')
TARGET.mkdir(exist_ok=False)
PROMPTS = OUT / 'matched-prompts'
PROMPTS.mkdir(exist_ok=True)
import prompt_helpers as bench


def request(route, body=None):
    req = urllib.request.Request(BASE + route, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        return response.read()


def counters():
    raw = request('/metrics').decode()
    metrics = {}
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        key, value = line.rsplit(' ', 1)
        metrics[key] = float(value)
    return raw, metrics


def total(metrics, name):
    return sum(value for key, value in metrics.items() if key == name or key.startswith(name + '{'))


def health():
    def one(item):
        host, config = item
        command = shlex.join(['docker', 'inspect', config['container']])
        info = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', host, command], text=True, timeout=30))[0]
        assert info['Image'] == PROFILE['image_id'] and info['State']['Running'], host
        assert not info['State']['OOMKilled'] and info['RestartCount'] == 0, host
        return host, {'image': info['Image'], 'state': info['State'], 'restarts': info['RestartCount']}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return dict(pool.map(one, PROFILE['hosts'].items()))


def prompt(name, nominal):
    path = PROMPTS / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    prefix = f'[QWEN_MATCHED_{name}_20260916] '
    padding = bench.generate_padding_text(max(nominal * 2, 1))

    def at(chars):
        messages = bench.build_messages(nominal, prefix + padding[:chars] if nominal else '')
        if not nominal:
            messages[-1]['content'] += '\nRequest identifier: ' + prefix
        data = json.loads(request('/tokenize', {'model': MODEL, 'messages': messages,
                          'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}))
        return data, messages

    if nominal:
        lo, hi = 0, len(padding)
        while lo <= hi:
            middle = (lo + hi) // 2
            tokenized, messages = at(middle)
            if tokenized['count'] == nominal + 2:
                break
            if tokenized['count'] < nominal + 2:
                lo = middle + 1
            else:
                hi = middle - 1
        assert tokenized['count'] == nominal + 2, (name, tokenized['count'])
    else:
        tokenized, messages = at(0)
    result = {'name': name, 'nominal_tokens': nominal, 'actual_tokens': tokenized['count'],
              'messages': messages, 'prompt_ids_sha256': hashlib.sha256(json.dumps(tokenized['tokens']).encode()).hexdigest()}
    path.write_text(json.dumps(result) + '\n')
    return result


def generate(case, tokens, salt, barrier):
    body = {'model': MODEL, 'messages': case['messages'], 'max_tokens': tokens, 'temperature': 0,
            'ignore_eos': tokens > 1, 'stream': True, 'stream_options': {'include_usage': True},
            'return_token_ids': True, 'cache_salt': salt, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    barrier.wait(timeout=30)
    start = time.perf_counter()
    events, ids, usage, finish, text = [], [], None, None, ''
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            item = json.loads(line[6:])
            assert not item.get('error'), item
            if item.get('prompt_token_ids'):
                digest = hashlib.sha256(json.dumps(item['prompt_token_ids']).encode()).hexdigest()
                assert digest == case['prompt_ids_sha256'], (case['name'], digest)
            usage = item.get('usage') or usage
            for choice in item.get('choices', []):
                delta_ids = choice.get('token_ids') or []
                if delta_ids:
                    events.append([time.perf_counter(), len(delta_ids)])
                    ids.extend(delta_ids)
                text += choice.get('delta', {}).get('content') or ''
                finish = choice.get('finish_reason') or finish
    assert usage and len(ids) == tokens == usage['completion_tokens'], (len(ids), usage)
    assert usage['prompt_tokens'] == case['actual_tokens'], usage
    assert finish == 'length' or tokens == 1, finish
    return {'name': case['name'], 'start': start, 'end': time.perf_counter(),
            'events': events, 'usage': usage, 'finish_reason': finish,
            'token_ids': ids, 'output_preview': text[:400],
            'ttft_seconds': events[0][0] - start,
            'decode_tps': (tokens - events[0][1]) / (events[-1][0] - events[0][0]) if tokens > 1 else None}


def run(name, cases, tokens):
    target = TARGET / name
    target.mkdir()
    raw_before, before = counters()
    (target / 'before.prom').write_text(raw_before)
    barrier = threading.Barrier(len(cases))
    with concurrent.futures.ThreadPoolExecutor(len(cases)) as pool:
        futures = [pool.submit(generate, case, tokens, f'checkpoint-spacing-{TARGET.name}-{name}-{index}', barrier)
                   for index, case in enumerate(cases)]
        results = [future.result() for future in futures]
    deadline = time.monotonic() + 30
    while True:
        raw_after, after = counters()
        completed = total(after, 'vllm:request_success_total') - total(before, 'vllm:request_success_total')
        if completed == len(cases) or time.monotonic() > deadline:
            break
        time.sleep(0.25)
    delta = {key: value - before.get(key, 0) for key, value in after.items()}
    (target / 'after.prom').write_text(raw_after)
    assert completed == len(cases), completed
    assert total(delta, 'vllm:prompt_tokens_cached_total') == 0, 'Unexpected prefix cache hit'
    assert total(delta, 'vllm:prompt_tokens_total') == sum(c['actual_tokens'] for c in cases)
    stats = {'completed': completed, 'cached_tokens': 0}
    if tokens == 1:
        elapsed = total(delta, 'vllm:request_prefill_time_seconds_sum')
        assert total(delta, 'vllm:request_prefill_time_seconds_count') == 1 and elapsed > 0
        stats.update(prefill_tps=cases[0]['actual_tokens'] / elapsed, prefill_seconds=elapsed)
    else:
        begin = max(r['events'][0][0] for r in results)
        end = min(r['events'][-1][0] for r in results)
        emitted = sum(n for result in results for t, n in result['events'] if begin < t <= end)
        assert end - begin > (0.25 if name.startswith('warmup-') else 2)
        wall = max(r['end'] for r in results) - min(r['start'] for r in results)
        decode_span = max(r['events'][-1][0] for r in results) - min(r['events'][0][0] for r in results)
        drafts = total(delta, 'vllm:spec_decode_num_drafts_total')
        accepted = total(delta, 'vllm:spec_decode_num_accepted_tokens_total')
        proposed = total(delta, 'vllm:spec_decode_num_draft_tokens_total')
        assert drafts > 0 and proposed > 0
        positions = {re.search(r'position="(\d+)"', key)[1]: value for key, value in delta.items()
                     if key.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{')}
        stats.update(common_decode_tps=emitted / (end - begin), common_seconds=end - begin,
                     common_tokens=emitted, batch_generation_tps=tokens * len(cases) / wall,
                     mean_request_decode_tps=sum(r['decode_tps'] for r in results) / len(results),
                     drafts=drafts, accepted_tokens=accepted, proposed_tokens=proposed,
                     acceptance_fraction=accepted / proposed, accepted_per_draft=accepted / drafts,
                     emitted_per_draft=1 + accepted / drafts, per_position_accepted=positions,
                     request_drafts_per_decode_second=drafts / decode_span,
                     approximate_batch_steps_per_second=drafts / len(cases) / decode_span,
                     step_metric_scope='Whole-batch request drafts divided by decode span, including batch ramp-up and drain; approximate engine step rate.')
    data = {'name': name, 'mtp_tokens': args.depth, 'statistics': stats, 'requests': results, 'health': health()}
    (target / 'result.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'name': name, **stats}), flush=True)


with (ROOT / 'benchmark-results/.cluster-window.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    health()
    # Prepare every token sequence before measurements, then preserve the same
    # serialized prompts for every depth. Cache salt changes no input token.
    decode = {(context, concurrency): [prompt(f'decode-ctx{context}-c{concurrency}-r{i}', context) for i in range(concurrency)] for context in (0, 8192) for concurrency in (1, 8)}
    prefill = {size: prompt(f'prefill-{size}', size) for size in (8192, 65536, 131072)}
    for size, case in prefill.items():
        run(f'warmup-prefill-{size}', [case], 1)
    for (context, concurrency), cases in decode.items():
        run(f'warmup-decode-ctx{context}-c{concurrency}', cases, 96)
    for repeat in range(args.repeats):
        for size, case in prefill.items():
            run(f'r{repeat}-prefill-{size}', [case], 1)
        for (context, concurrency), cases in decode.items():
            run(f'r{repeat}-decode-ctx{context}-c{concurrency}', cases, 512)
    (TARGET / 'health.json').write_text(json.dumps(health(), indent=2) + '\n')
    print('Matched performance matrix complete.', flush=True)
