"""Air defaults depend on simple geometry and resolution scalars."""

import pytest

from steercurrent import ExteriorOptions


def test_air_defaults_and_overrides():
    assert ExteriorOptions().resolved(span=12.0, conductor_size=0.3) == {
        "outer_scale": 4.0,
        "near_size": 0.3,
        "far_size": 3.0,
        "transition_distance": 12.0,
    }
    assert ExteriorOptions(
        near_size=0.1, far_size=2.0, transition_distance=5.0
    ).resolved(span=12.0, conductor_size=0.3) == {
        "outer_scale": 4.0,
        "near_size": 0.1,
        "far_size": 2.0,
        "transition_distance": 5.0,
    }


@pytest.mark.parametrize(
    "span, size", [(0, 0.3), (12, 0), (float("nan"), 0.3), (12, float("inf"))]
)
def test_invalid_resolution_inputs(span, size):
    with pytest.raises(ValueError, match="finite and positive"):
        ExteriorOptions().resolved(span=span, conductor_size=size)
