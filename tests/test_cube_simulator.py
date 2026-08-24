"""Regression tests for seeded scramble generation."""

# pylint: disable=missing-function-docstring

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import pytest

import cube_bench.sim.cube_simulator as simulator
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


def test_oracle_calls_are_serialized_across_cube_instances(monkeypatch):
    state_lock = Lock()
    active_calls = 0
    max_active_calls = 0

    def fake_solve(_facelets):
        nonlocal active_calls, max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        sleep(0.01)
        with state_lock:
            active_calls -= 1
        return "R1 (1f)"

    monkeypatch.setattr(simulator.sv, "solve", fake_solve)
    cubes = [VirtualCube() for _ in range(10)]
    for cube in cubes:
        cube.apply("R")

    def query_oracle(item):
        index, cube = item
        return cube.get_distance() if index % 2 == 0 else cube.solve()

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(query_oracle, enumerate(cubes)))

    assert max_active_calls == 1
    assert results == [1, "R", 1, "R", 1, "R", 1, "R", 1, "R"]
