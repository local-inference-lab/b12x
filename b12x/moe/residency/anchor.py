"""Immutable learned-placement identity and observational routing comparisons."""

from dataclasses import dataclass

from .contracts import ExpertPlacement, _text


@dataclass(frozen=True, kw_only=True)
class ResidencyAnchor:
    """A backend-validated learned profile, never a runtime placement snapshot.

    The integration must validate checkpoint, recipe, geometry and artifact hash
    before constructing this descriptor. It is not an artifact validator itself.
    """

    profile_id: str
    checkpoint: str
    recipe: str
    workload: str
    placements: tuple[tuple[str, ExpertPlacement], ...]

    def __post_init__(self):
        for name in ("profile_id", "checkpoint", "recipe", "workload"):
            _text(name, getattr(self, name))
        if len(self.profile_id) != 64 or any(
            c not in "0123456789abcdef" for c in self.profile_id
        ):
            raise ValueError("anchor requires a verified profile SHA256")
        object.__setattr__(
            self, "placements", tuple((n, p) for n, p in self.placements)
        )
        names = [n for n, _ in self.placements]
        if not names or len(set(names)) != len(names):
            raise ValueError("anchor requires uniquely named layers")
        for name, placement in self.placements:
            _text("anchor layer", name)
            if not isinstance(placement, ExpertPlacement):
                raise TypeError("anchor requires typed placements")

    @property
    def resident_ids(self):
        return {n: p.resident_expert_ids for n, p in self.placements}


def compare_anchor(counts, current, anchor):
    """Score identical canonical counts; positive advantage favors the anchor."""
    if set(counts) != set(current) or set(counts) != set(anchor):
        raise ValueError("placement comparison requires identical layer sets")
    layers = {}
    for name, values in counts.items():
        size = len(values)
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("placement comparison requires nonnegative counts")
        for ids in (current[name], anchor[name]):
            if len(set(ids)) != len(ids) or any(
                type(e) is not int or not 0 <= e < size for e in ids
            ):
                raise ValueError("placement comparison has invalid expert IDs")
        if len(current[name]) != len(anchor[name]):
            raise ValueError("anchor and runtime resident capacities differ")
        hot, prior = set(current[name]), set(anchor[name])
        total = sum(values)
        cold = sum(n for e, n in enumerate(values) if e not in hot)
        anchor_cold = sum(n for e, n in enumerate(values) if e not in prior)
        layers[name] = dict(
            selections=total,
            cold_selections=cold,
            anchor_cold_selections=anchor_cold,
            missing_anchor=len(prior - hot),
            resident=len(hot),
        )
    return dict(
        layers=layers,
        selections=sum(v["selections"] for v in layers.values()),
        cold_selections=sum(v["cold_selections"] for v in layers.values()),
        anchor_cold_selections=sum(
            v["anchor_cold_selections"] for v in layers.values()
        ),
    )
