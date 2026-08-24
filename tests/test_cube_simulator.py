"""Regression tests for seeded scramble generation."""

# pylint: disable=missing-function-docstring

import pytest

from cube_bench.sim.cube_simulator import VirtualCube


def test_exact_scramble_retries_and_keeps_only_the_accepted_formula(monkeypatch):
    cube = VirtualCube()
    distances = iter((2, 3))
    distance_calls = 0

    def fake_distance():
        nonlocal distance_calls
        distance_calls += 1
        return next(distances)

    monkeypatch.setattr(cube, "get_distance", fake_distance)

    formula = cube.scramble(random_seed=7, n_moves=3, max_tries=2, exact_depth=True)

    expected = VirtualCube()
    expected.apply(str(formula))
    assert distance_calls == 2
    assert str(cube) == str(expected)
    assert str(cube.formula) == str(formula)


def test_exact_scramble_restores_the_original_state_after_failure(monkeypatch):
    cube = VirtualCube()
    original_state = str(cube)
    monkeypatch.setattr(cube, "get_distance", lambda: 2)

    with pytest.raises(RuntimeError, match="exact depth 3"):
        cube.scramble(random_seed=11, n_moves=3, max_tries=2, exact_depth=True)

    assert str(cube) == original_state
    assert cube.formula is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_moves": -1}, "non-negative"),
        ({"max_tries": 0}, "positive"),
        ({"n_moves": 21, "exact_depth": True}, "cannot exceed 20"),
    ],
)
def test_scramble_rejects_invalid_generation_requests(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VirtualCube().scramble(**kwargs)
