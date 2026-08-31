"""Check that ``MockAssistant`` replies round-trip through evaluation parsers."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import pathlib

import pytest
import yaml

from mock_assistant import (GRID_COLORS, LETTERS, MODE_GRID, MODE_MCQ,
                            MODE_MOVE_EFFECT, MODE_YES_NO, MOVE_EFFECT_LABELS,
                            MockAssistant, PromptAnswerKey, detect_mode)

from cube_bench.core import BaseTest

PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "src/cube_bench/prompts/prompts.yaml"


def _key(pairs):
    key = PromptAnswerKey()
    for prompt, gold in pairs:
        key.record(prompt, gold)
    return key


def test_mcq_replies_parse_back_to_gold():
    key = _key((f"p{i}", LETTERS[i % 4]) for i in range(40))
    mock = MockAssistant(accuracy=1.0, seed=1, answer_key=key, mode=MODE_MCQ)
    for i in range(40):
        assert BaseTest.parse_letter(mock.generate(f"p{i}")) == LETTERS[i % 4]


def test_yes_no_replies_parse_back_to_gold():
    key = _key((f"v{i}", "Yes" if i % 2 else "No") for i in range(20))
    mock = MockAssistant(accuracy=1.0, seed=1, answer_key=key, mode=MODE_YES_NO)
    for i in range(20):
        assert BaseTest.parse_yes_no(mock.generate(f"v{i}")) == ("Yes" if i % 2 else "No")


def test_idk_and_garbage_branches():
    key = _key([("p", "A")])
    assert BaseTest.parse_idk(
        MockAssistant(seed=3, answer_key=key, mode=MODE_MCQ, idk_rate=1.0).generate("p")
    )
    assert BaseTest.parse_letter(
        MockAssistant(seed=3, answer_key=key, mode=MODE_MCQ, garbage_rate=1.0).generate("p")
    ) is None


def test_move_effect_reply_parses_with_the_task_regex():
    task = pytest.importorskip("cube_bench.evaluations.move_effect")
    gold = {letter: MOVE_EFFECT_LABELS[i % 3] for i, letter in enumerate(LETTERS)}
    mock = MockAssistant(accuracy=1.0, seed=1, answer_key=_key([("me", gold)]),
                         mode=MODE_MOVE_EFFECT)
    reply = mock.generate("me")
    parsed = {m.group(1).upper(): m.group(2).upper().replace(" ", "_")
              for m in task.MoveEffectTest.TAG_RE.finditer(reply)}
    assert parsed == gold


def test_grid_reply_parses_with_the_task_regex():
    task = pytest.importorskip("cube_bench.evaluations.reconstruction")
    grid = [["White", "Red", "Blue"], ["Green", "Yellow", "Orange"], ["Red", "White", "Green"]]
    mock = MockAssistant(accuracy=1.0, seed=1, answer_key=_key([("g", grid)]), mode=MODE_GRID)
    found = task.ReconstructionTest.GRID_RE.search(mock.generate("g"))
    assert found is not None
    cells = list(found.groups())
    assert [cells[0:3], cells[3:6], cells[6:9]] == grid


@pytest.mark.parametrize("accuracy", [0.0, 0.25, 0.5, 0.7, 1.0])
def test_observed_accuracy_matches_expected_accuracy(accuracy):
    n = 2000
    key = _key((f"s{i}", LETTERS[i % 4]) for i in range(n))
    mock = MockAssistant(accuracy=accuracy, seed=7, answer_key=key, mode=MODE_MCQ)
    hits = sum(BaseTest.parse_letter(mock.generate(f"s{i}")) == LETTERS[i % 4] for i in range(n))
    assert abs(hits / n - mock.expected_accuracy()) < 0.03


def test_guessing_can_exclude_gold_for_an_exact_rate():
    n = 2000
    key = _key((f"s{i}", LETTERS[i % 4]) for i in range(n))
    mock = MockAssistant(accuracy=0.7, seed=7, answer_key=key, mode=MODE_MCQ,
                         guess_includes_gold=False)
    hits = sum(BaseTest.parse_letter(mock.generate(f"s{i}")) == LETTERS[i % 4] for i in range(n))
    assert mock.expected_accuracy() == 0.7
    assert abs(hits / n - 0.7) < 0.03


def test_replies_are_deterministic_per_seed():
    key = _key((f"p{i}", LETTERS[i % 4]) for i in range(40))
    def build(seed):
        return MockAssistant(accuracy=0.6, seed=seed, answer_key=key, mode=MODE_MCQ)

    first = [build(5).generate(f"p{i}") for i in range(40)]
    assert first == [build(5).generate(f"p{i}") for i in range(40)]
    assert first != [build(6).generate(f"p{i}") for i in range(40)]


def test_without_an_answer_key_the_mock_still_answers():
    mock = MockAssistant(accuracy=1.0, seed=1, mode=MODE_MCQ)
    assert BaseTest.parse_letter(mock.generate("anything")) in set(LETTERS)


def test_generate_records_call_metadata():
    mock = MockAssistant(seed=1, mode=MODE_MCQ)
    mock.generate("p", "sys", image=object(), history=[{"role": "user"}])
    assert len(mock.calls) == 1
    assert mock.calls[0]["had_image"] is True
    assert mock.calls[0]["history_turns"] == 1


@pytest.mark.parametrize("name,expected", [
    ("verification", MODE_YES_NO),
    ("reconstruction", MODE_GRID),
    ("step_by_step", MODE_MCQ),
    ("learning_curve", MODE_MCQ),
])
def test_detect_mode_on_real_templates(name, expected):
    templates = yaml.safe_load(PROMPTS.read_text())
    assert detect_mode(templates[name]["sys"], templates[name]["user"]) == expected


def test_detect_mode_on_prediction_variants():
    templates = yaml.safe_load(PROMPTS.read_text())["prediction"]
    for variant in templates.values():
        assert detect_mode(variant["sys"], variant["user"]) == MODE_MCQ


def test_detect_mode_prefers_move_effect_over_the_mcq_cue():
    assert detect_mode("", "<A> DECREASE|NO_CHANGE|INCREASE </A>") == MODE_MOVE_EFFECT


@pytest.mark.parametrize("kwargs", [
    {"accuracy": 1.5}, {"accuracy": -0.1}, {"garbage_rate": 2.0}, {"idk_rate": -1.0},
])
def test_rates_are_validated(kwargs):
    with pytest.raises(ValueError):
        MockAssistant(**kwargs)


def test_grid_colors_are_understood_by_the_reconstruction_normaliser():
    task = pytest.importorskip("cube_bench.evaluations.reconstruction")
    for color in GRID_COLORS:
        assert color.lower() in task.ReconstructionTest.COLOR_MAP
