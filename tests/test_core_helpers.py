"""Test shared helpers with a simulator stub that avoids heavy optional imports.
The fidelity test guards the stub's move list against the real simulator."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from __future__ import annotations

import json
import math
import random
import sys
import types

import pytest

from cube_bench.core import BaseTest
from cube_bench.io import save_results

MOVES = (
    "R", "L", "U", "D", "F", "B",
    "R'", "L'", "U'", "D'", "F'", "B'",
    "R2", "L2", "U2", "D2", "F2", "B2",
)


@pytest.fixture(name="stub_simulator")
def fixture_stub_simulator(monkeypatch):
    """Stand in for the simulator module, which the helpers use only for moves."""
    module = types.ModuleType("cube_bench.sim.cube_simulator")

    class VirtualCube:  # pylint: disable=too-few-public-methods
        AVAILABLE_MOVES = MOVES

    module.VirtualCube = VirtualCube
    monkeypatch.setitem(sys.modules, "cube_bench.sim.cube_simulator", module)
    return module


class FakeCube:
    """Cube stand-in where only the moves in *good* reduce distance-to-solved."""

    def __init__(self, good, distance: int = 5, solved: bool = False):
        self.good = set(good)
        self.distance = distance
        self.solved = solved

    def is_solved(self) -> bool:
        return self.solved

    def get_distance(self) -> int:
        return self.distance

    def clone(self) -> "FakeCube":
        return FakeCube(self.good, self.distance, self.solved)

    def apply(self, move: str) -> None:
        self.distance += -1 if move in self.good else 1


class FakeFormula:
    """Mimics pycuber's Formula.reverse(), which mutates in place and returns self."""

    def __init__(self, moves):
        self.moves = list(moves)

    def reverse(self) -> "FakeFormula":
        self.moves.reverse()
        return self

    def __str__(self) -> str:
        return " ".join(self.moves)


class HelperTask(BaseTest):
    """Concrete BaseTest so the instance-method helpers can be exercised."""

    test_type = "helper-test"

    def run(self, num_samples):
        raise NotImplementedError


# MCQ construction

@pytest.mark.usefixtures("stub_simulator")
def test_gen_mcq_places_correct_move_among_unique_distractors():
    options, gold = BaseTest.gen_mcq("R", random.Random(0))
    assert set(options) == set("ABCD")
    assert options[gold] == "R"
    assert len(set(options.values())) == 4


@pytest.mark.parametrize("letter", list("ABCD"))
@pytest.mark.usefixtures("stub_simulator")
def test_gen_mcq_honours_force_letter(letter):
    options, gold = BaseTest.gen_mcq("U2", random.Random(1), force_letter=letter)
    assert gold == letter
    assert options[letter] == "U2"
    assert len(set(options.values())) == 4


@pytest.mark.usefixtures("stub_simulator")
def test_gen_mcq_draws_distractors_from_the_supplied_pool():
    pool = ["R", "L", "U", "D", "F"]
    options, gold = BaseTest.gen_mcq("R", random.Random(2), pool=pool)
    assert options[gold] == "R"
    assert set(options.values()) <= set(pool)


@pytest.mark.usefixtures("stub_simulator")
def test_gen_mcq_is_reproducible_for_a_seed():
    assert BaseTest.gen_mcq("F", random.Random(9)) == BaseTest.gen_mcq("F", random.Random(9))


@pytest.mark.usefixtures("stub_simulator")
def test_gen_mcq_from_good_keeps_distractors_out_of_the_good_set():
    good = {"R", "U"}
    options, gold = BaseTest.gen_mcq_from_good(good, random.Random(3))
    assert options[gold] in good
    distractors = {v for k, v in options.items() if k != gold}
    assert not distractors & good
    assert len(set(options.values())) == 4


@pytest.mark.usefixtures("stub_simulator")
def test_gen_mcq_from_good_handles_an_empty_good_set():
    options, gold = BaseTest.gen_mcq_from_good(set(), random.Random(4))
    assert set(options) == set("ABCD")
    assert gold in options
    assert len(set(options.values())) == 4


@pytest.mark.usefixtures("stub_simulator")
def test_gen_mcq_balanced_includes_exactly_one_progress_distractor():
    task = HelperTask(assistant=None, config=None, n_moves=1)
    options, gold = task.gen_mcq_balanced(FakeCube({"R", "U", "F"}), "R", random.Random(5))
    assert options[gold] == "R"
    distractors = [v for k, v in options.items() if k != gold]
    assert len(set(options.values())) == 4
    assert sum(move in {"U", "F"} for move in distractors) == 1


# Progress detection

@pytest.mark.usefixtures("stub_simulator")
def test_optimal_first_moves_returns_the_distance_reducing_moves():
    assert BaseTest.optimal_first_moves(FakeCube({"R", "U2"})) == {"R", "U2"}


@pytest.mark.usefixtures("stub_simulator")
def test_optimal_first_moves_is_empty_for_a_solved_cube():
    assert BaseTest.optimal_first_moves(FakeCube(set(), solved=True)) == set()


def test_move_makes_progress_reports_distance_before_and_after():
    cube = FakeCube({"R"}, distance=5)
    assert BaseTest.move_makes_progress(cube, "R") == (True, 5, 4)
    assert BaseTest.move_makes_progress(cube, "L") == (False, 5, 6)


# Teacher path

def test_teacher_path_returns_the_inverse_scramble():
    assert BaseTest.teacher_path(FakeFormula(["R", "U", "F"])) == ["F", "U", "R"]


def test_teacher_path_leaves_the_callers_scramble_untouched():
    scramble = FakeFormula(["R", "U", "F"])
    BaseTest.teacher_path(scramble)
    assert str(scramble) == "R U F"


def test_teacher_first_move_is_the_head_of_the_path():
    assert BaseTest.teacher_first_move(FakeFormula(["R", "U", "F"])) == "F"


def test_teacher_path_degrades_gracefully():
    class Unreversible:  # pylint: disable=too-few-public-methods
        def reverse(self):
            raise RuntimeError("boom")

    assert BaseTest.teacher_path(Unreversible()) == []
    assert BaseTest.teacher_first_move(Unreversible()) is None


def test_inverse_move_handles_blank_input():
    assert BaseTest.inverse_move("   ") == ""


# Deterministic item RNG

def test_item_rng_is_reproducible_for_the_same_parts():
    draws = [BaseTest.item_rng("task", 3, i).random() for i in range(5)]
    assert draws == [BaseTest.item_rng("task", 3, i).random() for i in range(5)]


def test_item_rng_separates_namespaces_and_depths():
    base = BaseTest.item_rng("task", 3, 0).random()
    assert BaseTest.item_rng("other", 3, 0).random() != base
    assert BaseTest.item_rng("task", 5, 0).random() != base


# Statistics

def test_wilson_ci_brackets_the_estimate():
    low, high = BaseTest.wilson_ci(0.5, 100)
    assert 0.0 <= low < 0.5 < high <= 1.0


def test_wilson_ci_is_clamped_to_the_unit_interval():
    assert BaseTest.wilson_ci(1.0, 10)[1] <= 1.0
    assert BaseTest.wilson_ci(0.0, 10)[0] >= 0.0


def test_wilson_ci_narrows_as_the_sample_grows():
    def width(n):
        low, high = BaseTest.wilson_ci(0.5, n)
        return high - low

    assert width(1000) < width(100) < width(10)


@pytest.mark.parametrize("proportion,n", [
    (0.5, 0), (0.5, -1), (1.5, 10), (-0.1, 10), (float("nan"), 10),
])
def test_wilson_ci_returns_nan_for_invalid_input(proportion, n):
    low, high = BaseTest.wilson_ci(proportion, n)
    assert math.isnan(low) and math.isnan(high)


# Results files

def test_save_results_creates_parents_and_appends(tmp_path):
    path = tmp_path / "nested" / "out.json"
    save_results(path, {"a": 1})
    save_results(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}, {"a": 2}]


def test_save_results_overwrites_a_file_that_is_not_a_list(tmp_path):
    path = tmp_path / "out.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    save_results(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}]


def test_save_results_overwrites_undecodable_json(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("{not json", encoding="utf-8")
    save_results(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}]


# Stub fidelity

def test_stub_move_list_matches_the_real_one():
    sim = pytest.importorskip("cube_bench.sim.cube_simulator")
    assert tuple(sim.VirtualCube.AVAILABLE_MOVES) == MOVES
