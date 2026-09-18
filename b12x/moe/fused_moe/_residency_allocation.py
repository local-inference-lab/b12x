"""Host-only byte admission for static two-tier expert placement."""
from collections import defaultdict
from fractions import Fraction
from math import gcd
from functools import reduce


def allocate_rows(layers, capacity, grace, counts=None):
    """Prefer selection density, or equal resident fractions without observations.

    Rows within a layer have equal byte cost. Minimum rows are mandatory. If
    greedy packing cannot fit the cold complement, bounded subset-sum repairs
    feasibility without claiming to optimize the varying-size knapsack score.
    """
    hot, candidates = {}, []
    total = sum(s.experts*s.expert_bytes for s in layers)
    minimum = sum(s.minimum_hot*s.expert_bytes for s in layers)
    maximum = sum(s.maximum_hot*s.expert_bytes for s in layers)
    if minimum > capacity:
        raise ValueError("minimum hot placement exceeds HBM budget")
    required = max(0, total-grace)
    if required > min(capacity, maximum):
        raise ValueError("HBM and Grace budgets cannot fit experts within hot bounds")
    for s in layers:
        ranking = (sorted(range(s.experts), key=lambda e: (-counts[s.layer][e], e))
                   if counts is not None else list(range(s.experts)))
        hot[s.layer] = set(ranking[:s.minimum_hot])
        for position in range(s.minimum_hot, s.maximum_hot):
            expert = ranking[position]
            # Equal fractional coverage is a prior, never synthetic route data.
            score = (-Fraction(counts[s.layer][expert], s.expert_bytes)
                     if counts is not None else Fraction(position, s.experts))
            candidates.append((score, s.layer, expert, s.expert_bytes))
    candidates.sort()
    remaining = capacity-minimum
    for _, layer, expert, size in candidates:
        if size <= remaining:
            hot[layer].add(expert)
            remaining -= size
    if capacity-remaining >= required:
        return hot

    # Feasibility depends only on counts of each byte size. Grouping equal sizes
    # keeps the search independent of model expert identities and row scores.
    groups = defaultdict(list)
    for row in candidates:
        groups[row[3]].append(row)
        hot[row[1]].discard(row[2])
    sizes = sorted(groups)
    unit = reduce(gcd, sizes)
    upper = min(capacity-minimum, sum(size*len(groups[size]) for size in sizes))//unit
    lower = max(0, (required-minimum+unit-1)//unit)
    # Bound CPU time and retained bitsets before allocating. This is an explicit
    # unsupported-search result, not a claim that the memory budget is infeasible.
    if upper > 8_000_000 or (upper+1)*(len(sizes)+1) > 512_000_000:
        raise ValueError("joint placement feasibility search exceeds supported size; "
                         "feasibility is unknown, use explicit hot bounds or a static placement")
    mask = (1 << (upper+1))-1
    reachable, history = 1, [1]
    for size in sizes:
        left, chunk = len(groups[size]), 1
        while left:
            take = min(chunk, left)
            shift = take*(size//unit)
            if shift <= upper:
                reachable |= (reachable << shift) & mask
            left -= take
            chunk *= 2
        history.append(reachable)
    target = reachable.bit_length()-1
    if target < lower:
        raise ValueError("no placement satisfies joint HBM/Grace budgets and hot bounds")
    # Prefer the fullest feasible packing. Within every cost class retain the
    # highest-priority rows (density or bootstrap coverage), with stable ties.
    for index in range(len(sizes)-1, -1, -1):
        size = sizes[index]
        step = size//unit
        for n in range(min(len(groups[size]), target//step), -1, -1):
            if (history[index] >> (target-n*step)) & 1:
                for _, layer, expert, _ in groups[size][:n]:
                    hot[layer].add(expert)
                target -= n*step
                break
        else:
            raise RuntimeError("joint placement reconstruction failed")
    return hot
