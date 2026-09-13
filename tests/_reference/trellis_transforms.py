"""Independent dense-matrix oracles shared with profile qualification."""

from b12x.policy.generation.providers.trellis_reference import (
    activation,
    had128,
    had512,
    input_rotation,
    intermediate_rotation,
    output_rotation,
)

__all__ = [
    "activation",
    "had128",
    "had512",
    "input_rotation",
    "intermediate_rotation",
    "output_rotation",
]
