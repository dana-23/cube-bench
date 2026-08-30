"""Test the cross-modal verification item generator's fairness controls."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from __future__ import annotations

from collections import Counter

import pytest

verification = pytest.importorskip("cube_bench.evaluations.verification")

VerificationTest = verification.VerificationTest


def front_affecting_moves():
    """The move pool mismatched items are drawn from."""
    return VerificationTest._FRONT_AFFECTING  # pylint: disable=protected-access


class StubCube:
    """Cube stand-in: tracks the applied move so mismatches stay observable."""

    def __init__(self, seed=None, applied=None):
        self.seed = seed
        self.applied = list(applied or [])

    def scramble(self, random_seed, n_moves, exact_depth):
        del n_moves, exact_depth
        self.seed = random_seed
        return f"scramble-{random_seed}"

    def front_face(self):
        return f"front-{self.seed}"

    def clone(self):
        return StubCube(self.seed, self.applied)

    def apply(self, move):
        self.applied.append(move)

    def to_image(self):
        return f"image-{self.seed}-{'.'.join(self.applied)}"


@pytest.fixture(name="task")
def fixture_task(monkeypatch):
    monkeypatch.setattr(verification, "VirtualCube", StubCube)
    return VerificationTest(assistant=None, config=None, n_moves=5)


def _samples(task, n):
    return [task.build_item(i) for i in range(n)]


# Label balance

def test_labels_are_balanced_overall(task):
    counts = Counter(s["expected"] for s in _samples(task, 120))
    assert counts["Yes"] == counts["No"] == 60


def test_matched_items_answer_yes_and_mismatched_answer_no(task):
    for sample in _samples(task, 40):
        matched = sample["mismatch_move"] is None
        assert sample["expected"] == ("Yes" if matched else "No")


def test_mismatched_items_apply_a_front_affecting_move(task):
    for sample in _samples(task, 40):
        if sample["index"] % 2 == 0:
            assert sample["mismatch_move"] is None
        else:
            assert sample["mismatch_move"] in front_affecting_moves()


# Determinism
#
# The mismatch move was drawn from random.SystemRandom before the sampler was
# seeded, so mismatched items were redrawn on every run and no two runs shared
# an item set. These guard that regression.

def test_items_are_identical_across_instances(monkeypatch):
    monkeypatch.setattr(verification, "VirtualCube", StubCube)

    def build(idx):
        task = VerificationTest(assistant=None, config=None, n_moves=5)
        return task.build_item(idx)

    for idx in range(12):
        first, second = build(idx), build(idx)
        assert first["mismatch_move"] == second["mismatch_move"]
        assert first["image"] == second["image"]
        assert first["front_text"] == second["front_text"]


def test_depth_changes_the_mismatch_draw(monkeypatch):
    monkeypatch.setattr(verification, "VirtualCube", StubCube)

    def moves_at_depth(depth):
        task = VerificationTest(assistant=None, config=None, n_moves=depth)
        return [
            task.build_item(i)["mismatch_move"]
            for i in range(1, 40, 2)
        ]

    assert moves_at_depth(3) != moves_at_depth(5)
