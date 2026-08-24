"""Smoke tests for the evaluation package and its shared parsing helpers."""

# pylint: disable=missing-function-docstring

import pytest

from cube_bench.core import BaseTest as EvaluationBase


def test_public_evaluations_extend_base_class():
    # Importing the tasks pulls the runtime stack (tqdm, torch); the parser
    # tests below need none of it, so only this one is skipped without it.
    evaluations = pytest.importorskip("cube_bench.evaluations")
    for name in evaluations.__all__:
        evaluation_class = getattr(evaluations, name)
        assert issubclass(evaluation_class, EvaluationBase)


def test_parse_letter_accepts_supported_answer_formats():
    options = {"A": "R", "B": "U'", "C": "F2", "D": "L"}

    assert EvaluationBase.parse_letter("<ANSWER> B </ANSWER>") == "B"
    assert EvaluationBase.parse_letter("ANSWER: c") == "C"
    assert EvaluationBase.parse_letter("<d>") == "D"
    assert EvaluationBase.parse_letter("<ANSWER> U' </ANSWER>", options) == "B"


def test_parse_letter_rejects_unstructured_text():
    assert EvaluationBase.parse_letter("I would choose option A") is None
    assert EvaluationBase.parse_letter(None) is None


def test_parse_idk_accepts_supported_answer_formats():
    assert EvaluationBase.parse_idk("<ANSWER> IDK </ANSWER>")
    assert EvaluationBase.parse_idk("ANSWER: E")
    assert EvaluationBase.parse_idk("I don't know")
    assert not EvaluationBase.parse_idk("ANSWER: A")


def test_inverse_move():
    assert EvaluationBase.inverse_move("R") == "R'"
    assert EvaluationBase.inverse_move("R'") == "R"
    assert EvaluationBase.inverse_move("R2") == "R2"
