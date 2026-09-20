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
