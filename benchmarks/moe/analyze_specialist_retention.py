"""Reconstruct recovery victims and test retention without changing serving policy.

The simulation keeps recorded observation times and anchor candidate transactions.
It may substitute only re-centering victims. Future-demand ordering is an oracle,
not a deployable signal. Counterfactual route coverage is not predicted latency.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from benchmarks.moe.analyze_residency_anchor import load_anchor
from benchmarks.moe.summarize_expert_health import records
from benchmarks.moe.analyze_anchor_recovery import classify_layer


def boundary_demand(rows):
    """Return exact decode counts for each controlled request group and phase."""
    requests = {r['index']: r for r in rows if r['kind'] == 'request'}
    cuts = [r for r in rows if r['kind'] == 'routing_boundary']
    if not cuts or cuts[0]['next_request'] != 0 or cuts[-1]['next_request'] != len(requests):
        raise ValueError('complete routing boundaries are required')
    phases, groups = {}, []
    previous = None
    for cut in cuts:
        if len(cut['receipt']) != 1:
            raise ValueError('diagnostic requires one authoritative worker')
        snapshot = cut['receipt'][0]['snapshot']
        values = {v['layer']: tuple(v['counts']) for v in snapshot['layers'] if v['phase'] == 'decode'}
        if any(type(c) is not int or c < 0 for v in values.values() for c in v):
            raise ValueError('routing counts must be nonnegative integers')
        if previous:
            start, old, epoch = previous
            if start >= cut['next_request'] or snapshot['epoch'] != epoch or values.keys() != old.keys():
                raise ValueError('routing boundary order, reset epoch or layer set changed')
            delta = {n: tuple(b-a for a, b in zip(old[n], v, strict=True)) for n, v in values.items()}
            if any(c < 0 for v in delta.values() for c in v):
                raise ValueError('routing counters decreased')
            labels = {requests[i]['workload'] for i in range(start, cut['next_request'])}
            if len(labels) != 1:
                raise ValueError('routing group crosses a labeled phase')
            phase, = labels
            target = phases.setdefault(phase, {n: [0]*len(v) for n, v in delta.items()})
            for n, v in delta.items():
                target[n] = [a+b for a, b in zip(target[n], v, strict=True)]
            groups.append(dict(phase=phase, start_request=start, end_request=cut['next_request'],
                               counts=delta, time_ns=cut['time_ns']))
        previous = cut['next_request'], values, snapshot['epoch']
    return phases, groups


def choose_victim(eligible, scores, value, protected=(), *, oracle=False):
    """Choose from already-eligible victims; never weaken the score margin."""
    choices = set(eligible) - set(protected)
    return min(choices, key=lambda e: (value[e], scores[e], e) if oracle else (scores[e], e)) if choices else None


def analyze(path, profile, *, return_phase='return_general', future_phase='second_specialist'):
    rows = records(path)
    if rows[-1]['kind'] != 'complete':
        raise ValueError('retention analysis requires a complete serving receipt')
    prepared = next(r['status'] for r in rows if r['kind'] == 'prepared')
    configs = next(r['receipt']['worker']['policy_configs'] for r in rows
                   if r['kind'] == 'maintenance' and r['receipt']['worker']['baseline'])
    plans = load_anchor(profile, prepared)
    anchor = {n: set(p.hbm_expert_ids) for n, p in plans.items()}
    demand, groups = boundary_demand(rows)
    if any(set(ds) != set(plans) or any(len(ds[n]) != plan.total_experts for n, plan in plans.items())
           for ds in demand.values()):
        raise ValueError('routing boundaries differ from learned profile geometry')
    if return_phase not in demand or future_phase not in demand:
        raise ValueError('return and subsequent specialist phases are required')
    requests = [r for r in rows if r['kind'] == 'request']
    starts = sorted((min(r['start_wall_ns'] for r in requests if r['workload'] == p), p) for p in demand)
    future = demand[future_phase]
    future_total = sum(map(sum, future.values()))
    hot = {n: set(v) for n, v in anchor.items()}
    events, windows, entered = [], [], {n: {} for n in hot}
    start_sets, victims = None, []
    for record in rows:
        if record['kind'] != 'maintenance' or record['receipt']['worker']['baseline']:
            continue
        worker = record['receipt']['worker']
        observations = worker.get('policy_observations')
        if not observations or len(observations) != 1:
            raise ValueError('retention analysis requires policy diagnostics without history')
        phase = max(v for v in starts if v[0] <= record['time_ns'])[1]
        values = observations[0]['layers']
        reconstructed_cold = sum(sum(c for e, c in enumerate(v['counts']) if e not in hot[n])
                                 for n, v in values.items())
        if reconstructed_cold != worker['cold_selections']:
            raise ValueError('recorded cold selections differ from reconstructed map')
        if phase == return_phase and start_sets is None:
            start_sets = {n: sorted(v-anchor[n]) for n, v in hot.items()}
        windows.append(dict(phase=phase, time_ns=record['time_ns'], worker=worker))
        for name, value in values.items():
            if worker['movement_mode'] == 'recenter':
                config = configs[name]
                classify_layer(counts=value['counts'], scores=value['scores'], hot=hot[name],
                    anchor=anchor[name], protected=(*value['protected'], *value.get('recenter_protected', ())),
                    minimum_count=config['minimum_cold_selections'], margin=config['minimum_score_gain'],
                    layer_pairs=config['max_pairs'], proposed=value['candidates'], selected=value['selected'],
                    allow=worker['anchor']['anchor_better'])
            for candidate, victim in value['selected']:
                if candidate in hot[name] or victim not in hot[name]:
                    raise ValueError('recorded transaction differs from reconstructed map')
                event = dict(phase=phase, mode=worker['movement_mode'], layer=name,
                             candidate=candidate, victim=victim, time_ns=record['time_ns'],
                             window=value['window'])
                events.append(event)
                if phase == return_phase and worker['movement_mode'] == 'recenter':
                    victims.append({**event, 'score': value['scores'][victim],
                        'hits_since_promotion': dict(value['hits']).get(victim, 0),
                        'promotion_window': entered[name].get(victim, 0),
                        'demand': {p: ds[name][victim] for p, ds in demand.items()}})
                hot[name].remove(victim)
                hot[name].add(candidate)
                entered[name].pop(victim, None)
                entered[name][candidate] = value['window']
    for victim in victims:
        matches = [e for e in events if e['phase'] == future_phase and e['layer'] == victim['layer']
                   and e['candidate'] == victim['victim']]
        victim['subsequent_promotions'] = len(matches)
        victim['first_repromotion_ns'] = matches[0]['time_ns'] if matches else None
        uses = [g for g in groups if g['phase'] == future_phase and g['counts'][victim['layer']][victim['victim']]]
        victim['first_future_group'] = uses[0]['start_request'] if uses else None
    # Hold anchor candidates and transaction budgets fixed. Substitute victims
    # only during recovery; other recorded transactions may become infeasible.
    # Such skips are reported, never silently converted into extra adaptation.
    results = {}
    variants = [('unchanged', 0, 0), ('oracle_victims', 0, 0)] + [
        (signal, count, lifetime) for signal in ('hits', 'score', 'promotion_age', 'oracle_protection')
        for count in (1, 2) for lifetime in (4, 8, 16)]
    for signal, count, lifetime in variants:
        current = {n: set(v) for n, v in anchor.items()}
        protected, first_window = None, None
        totals, trajectory = Counter(), []
        before_second = None
        for window in windows:
            phase, worker = window['phase'], window['worker']
            values = worker['policy_observations'][0]['layers']
            if phase == future_phase:
                before_second = {n: sorted(v) for n, v in current.items()}
                break
            recovery = phase == return_phase and worker['movement_mode'] == 'recenter'
            if recovery and protected is None and worker['anchor']['anchor_better']:
                first_window = {n: v['window'] for n, v in values.items()}
                def weight(n, e):
                    v = values[n]
                    return (dict(v['hits']).get(e, 0) if signal == 'hits' else
                            v['scores'][e] if signal == 'score' else
                            max((ev['window'] for ev in events if ev['layer'] == n and ev['candidate'] == e
                                 and ev['time_ns'] < window['time_ns']), default=0) if signal == 'promotion_age' else
                            future[n][e])
                protected = {n: set(sorted(current[n]-anchor[n], key=lambda e: (-weight(n, e), e))[:count])
                             for n in current}
            total = cold = anchor_total = anchor_missing = 0
            for name, v in values.items():
                counts = v['counts']
                total += sum(counts)
                cold += sum(c for e, c in enumerate(counts) if e not in current[name])
                anchor_total += sum(counts[e] for e in anchor[name])
                anchor_missing += sum(counts[e] for e in anchor[name]-current[name])
                for candidate, original in v['selected']:
                    victim = original
                    if recovery and signal != 'unchanged':
                        eligible = [e for e in current[name]-anchor[name]
                                    if e not in v['protected'] and v['scores'][candidate]-v['scores'][e]
                                    >= configs[name]['minimum_score_gain']]
                        guard = protected[name] if protected is not None and v['window']-first_window[name] < lifetime else ()
                        victim = choose_victim(eligible, v['scores'], future[name], guard,
                                               oracle=signal == 'oracle_victims')
                    if candidate in current[name] or victim not in current[name]:
                        totals['skipped_recovery' if recovery else 'skipped_other'] += 1
                        continue
                    totals['changed_victims'] += victim != original
                    totals['return_moves'] += phase == return_phase
                    current[name].remove(victim)
                    current[name].add(candidate)
            if phase == return_phase:
                totals['observed_return_selections'] += total
                totals['observed_return_cold'] += cold
                totals['observed_anchor_selections'] += anchor_total
                totals['observed_missing_anchor_selections'] += anchor_missing
                trajectory.append(dict(time_ns=window['time_ns'], cold=cold, total=total,
                    overlap=sum(len(v & anchor[n]) for n, v in current.items())/sum(map(len, current.values()))))
        if before_second is None:
            raise ValueError('second specialist observation is missing')
        covered = sum(sum(future[n][e] for e in ids) for n, ids in before_second.items())
        results[f'{signal}:{count}:{lifetime}'] = dict(totals, second_specialist_precoverage=covered/future_total,
            resident_before_second=before_second, return_trajectory=trajectory)
    for key in ('skipped_recovery', 'skipped_other', 'changed_victims'):
        if results['unchanged:0:0'].get(key, 0):
            raise ValueError('unchanged replay failed to reproduce recorded transactions')
    rank_results = {}
    for name, score in [('future_oracle', lambda v: v['demand'][future_phase]),
                        ('preceding_specialist', lambda v: v['demand']['specialist']),
                        ('hits_since_promotion', lambda v: v['hits_since_promotion']),
                        ('score_at_eviction', lambda v: v['score']),
                        ('promotion_age', lambda v: v['promotion_window'])]:
        ordered = sorted(victims, key=lambda v: (-score(v), v['layer'], v['victim']))
        rank_results[name] = {str(k): sum(v['demand'][future_phase] for v in ordered[:k]) for k in (16, 32, 48, 96)}
    return dict(receipt=str(path), phase_counts=demand, groups=groups, recovery_victims=victims,
                non_anchor_at_return=start_sets, events=events, ranking=rank_results, variants=results,
                scope='Fixed recorded windows and anchor candidates, no latency model. Boundary reads give exact decode phase demand; maintenance windows may cross phase boundaries. Infeasible recorded non-recovery transactions are explicitly skipped.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(analyze(args.receipt, args.profile), stream, indent=2)


if __name__ == '__main__':
    main()
