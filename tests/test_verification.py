"""Test the cross-modal verification item generator's fairness controls."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from __future__ import annotations

from collections import Counter

import pytest

verification = pytest.importorskip("cube_bench.evaluations.verification")

VerificationTest = verification.VerificationTest


class StubCube:
    """Cube stand-in: tracks the applied move so mismatches stay observable."""

    def __init__(self, seed=None, applied=None):
        self.seed = seed
        self.applied = list(applied or [])

    def scramble(self, random_seed, n_moves, exact_depth):  # noqa: ARG002
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
    task = VerificationTest.__new__(VerificationTest)
    VerificationTest.__init__(task, assistant=None, config=None, n_moves=5)
    return task


def _samples(task, n):
    return [task._build_sample(i) for i in range(n)]  # pylint: disable=protected-access


# Label balance

def test_labels_are_balanced_overall(task):
    counts = Counter(s["expected"] for s in _samples(task, 120))
    assert counts["Yes"] == counts["No"] == 60


def test_labels_are_balanced_within_each_polarity(task):
    per_polarity = Counter(
        (s["polarity"], s["expected"]) for s in _samples(task, 120)
    )
    for polarity in ("affirmative", "negated"):
        assert per_polarity[(polarity, "Yes")] == per_polarity[(polarity, "No")] == 30


def test_labels_are_balanced_within_each_template(task):
    per_template = Counter(
        (s["template_id"], s["expected"]) for s in _samples(task, 120)
    )
    templates = {t for t, _ in per_template}
    assert len(templates) == 6  # 3 surface forms x 2 polarities
    for template in templates:
        assert per_template[(template, "Yes")] == per_template[(template, "No")]


def test_both_polarities_are_used_equally(task):
    counts = Counter(s["polarity"] for s in _samples(task, 120))
    assert counts["affirmative"] == counts["negated"] == 60


# Polarity semantics

def test_negation_inverts_the_expected_answer(task):
    for sample in _samples(task, 120):
        claim_is_true = sample["states_match"] == (sample["polarity"] == "affirmative")
        assert sample["expected"] == ("Yes" if claim_is_true else "No")


def test_mismatched_items_apply_a_front_affecting_move(task):
    for sample in _samples(task, 40):
        if sample["states_match"]:
            assert sample["mismatch_move"] is None
        else:
            assert sample["mismatch_move"] in VerificationTest._FRONT_AFFECTING


def test_claim_text_carries_the_front_face_and_matches_its_polarity(task):
    for sample in _samples(task, 8):
        assert sample["front_text"] in sample["claim"]
        negated_wording = ("NOT" in sample["claim"]
                           or "does not match" in sample["claim"]
                           or "inconsistent" in sample["claim"])
        assert negated_wording == (sample["polarity"] == "negated")


# Determinism

def test_items_are_identical_across_instances(monkeypatch):
    monkeypatch.setattr(verification, "VirtualCube", StubCube)

    def build(idx):
        task = VerificationTest.__new__(VerificationTest)
        VerificationTest.__init__(task, assistant=None, config=None, n_moves=5)
        return task._build_sample(idx)  # pylint: disable=protected-access

    for idx in range(12):
        first, second = build(idx), build(idx)
        assert first["mismatch_move"] == second["mismatch_move"]
        assert first["image"] == second["image"]
        assert first["claim"] == second["claim"]


def test_depth_changes_the_mismatch_draw(monkeypatch):
    monkeypatch.setattr(verification, "VirtualCube", StubCube)

    def moves_at_depth(depth):
        task = VerificationTest.__new__(VerificationTest)
        VerificationTest.__init__(task, assistant=None, config=None, n_moves=depth)
        return [
            task._build_sample(i)["mismatch_move"]  # pylint: disable=protected-access
            for i in range(1, 40, 2)
        ]

    assert moves_at_depth(3) != moves_at_depth(5)
