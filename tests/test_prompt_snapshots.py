"""Fail at the prompt, not two hundred lines downstream, when prompt text drifts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest

from golden_cases import CASES
from golden_runner import make_assistant, prompt_snapshot, run_case

SNAPSHOT_DIR = Path(__file__).resolve().parent / "prompt_snapshots"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_prompt_text_is_unchanged(case, tmp_path):
    path = SNAPSHOT_DIR / f"{case.name}.txt"
    if not path.is_file():
        pytest.fail(f"no snapshot for {case.name!r}; run tests/regen_golden.py")

    assistant = make_assistant()
    run_case(case, tmp_path / "results", assistant)

    assert prompt_snapshot(assistant) == path.read_text(encoding="utf-8")
