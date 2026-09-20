"""Observational routing pressure, independent of scoring and replacement policy."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, kw_only=True)
class RoutingHealthThresholds:
    cold_fraction: float
    layer_cold_fraction: float = 0.15
    minimum_layer_fraction: float = 0.0
    minimum_selections: int = 1

    def __post_init__(self):
        for name in ("cold_fraction", "layer_cold_fraction", "minimum_layer_fraction"):
            v = getattr(self, name)
            if type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("health fractions must be finite and in [0, 1]")
        if type(self.minimum_selections) is not int or self.minimum_selections < 1:
            raise ValueError("health observation minimum must be positive")

    def assess(self, summary):
        layers = summary["layers"]
        if not layers:
            raise ValueError("health summary requires declared layers")
        total, cold, elevated, observed = 0, 0, 0, 0
        for row in layers.values():
            n, c = row["selections"], row["cold_selections"]
            if type(n) is not int or type(c) is not int or not 0 <= c <= n:
                raise ValueError("invalid health selection counts")
            total += n
            cold += c
            observed += n > 0
            elevated += n > 0 and c / n >= self.layer_cold_fraction
        if total != summary["selections"] or cold != summary["cold_selections"]:
            raise ValueError("global health summary differs from layer totals")
        fraction = cold / total if total else None
        enough = total >= self.minimum_selections and observed == len(layers)
        pressure = (
            enough
            and fraction >= self.cold_fraction
            and elevated / len(layers) >= self.minimum_layer_fraction
        )
        return dict(
            health="unobserved"
            if not enough
            else "pressure"
            if pressure
            else "healthy",
            cold_fraction=fraction,
            elevated_layers=elevated,
            observed_layers=observed,
            layer_fraction=elevated / len(layers),
        )


@dataclass(frozen=True, kw_only=True)
class RoutingAnchorThresholds:
    """Explicit routing regret and breadth gates, not a throughput prediction."""

    advantage_fraction: float
    minimum_layer_fraction: float
    minimum_selections: int = 1

    def __post_init__(self):
        RoutingHealthThresholds(
            cold_fraction=self.advantage_fraction,
            minimum_layer_fraction=self.minimum_layer_fraction,
            minimum_selections=self.minimum_selections,
        )
        if not self.advantage_fraction:
            raise ValueError("anchor advantage threshold must be positive")

    def assess(self, summary):
        validated = RoutingHealthThresholds(cold_fraction=0).assess(summary)
        total = summary["selections"]
        cold, favoring = 0, 0
        for row in summary["layers"].values():
            value = row["anchor_cold_selections"]
            if type(value) is not int or not 0 <= value <= row["selections"]:
                raise ValueError("invalid anchor selection counts")
            cold += value
            favoring += row["cold_selections"] > value
        if cold != summary["anchor_cold_selections"]:
            raise ValueError("global anchor summary differs from layer totals")
        fraction = (summary["cold_selections"] - cold) / total if total else None
        breadth = favoring / len(summary["layers"])
        enough = total >= self.minimum_selections and validated["health"] != "unobserved"
        return dict(
            anchor_better=bool(enough and fraction >= self.advantage_fraction
                               and breadth >= self.minimum_layer_fraction),
            anchor_advantage_fraction=fraction,
            anchor_cold_fraction=cold / total if total else None,
            anchor_favoring_layers=favoring,
            anchor_layer_fraction=breadth,
        )
