"""Guard the generation parameters the mock ignores, so the golden cannot see them."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path

import pytest

from golden_cases import CASES
from golden_runner import generation_contract, make_assistant, run_case

CONTRACT_DIR = Path(__file__).resolve().parent / "contracts"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_generation_parameters_are_unchanged(case, tmp_path):
    path = CONTRACT_DIR / f"{case.name}.json"
    if not path.is_file():
        pytest.fail(f"no contract for {case.name!r}; run tests/regen_golden.py")

    assistant = make_assistant()
    run_case(case, tmp_path / "results", assistant)

    assert generation_contract(assistant) == json.loads(path.read_text(encoding="utf-8"))
