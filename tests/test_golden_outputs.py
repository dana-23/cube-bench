"""Byte-level guard: each evaluation's saved output must match its recorded golden."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest

from golden_cases import CASES
from golden_runner import run_case

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def read_golden(case_name: str) -> dict[str, str]:
    base = GOLDEN_DIR / case_name
    return {
        path.relative_to(base).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_saved_output_matches_golden(case, tmp_path):
    if not (GOLDEN_DIR / case.name).is_dir():
        pytest.fail(f"no golden for {case.name!r}; run tests/regen_golden.py")

    produced = run_case(case, tmp_path / "results")
    expected = read_golden(case.name)

    assert sorted(produced) == sorted(expected)
    for name in sorted(expected):
        assert produced[name] == expected[name], f"{case.name}/{name} drifted"


def test_step_by_step_aggregate_is_independent_of_concurrency(tmp_path):
    from golden_cases import evaluations, make_config
    from golden_runner import make_assistant, scrub

    payloads = []
    for index, concurrency in enumerate((1, 4)):
        root = tmp_path / f"run{index}"
        root.mkdir()
        task = evaluations.StepByStepTest(
            make_assistant(), make_config(root), n_moves=3,
            concurrency=concurrency, checkpoint=False,
        )
        task.run(8)
        payloads.append(scrub((root / "step_by_step.json").read_text(encoding="utf-8"), root))

    assert payloads[0] == payloads[1]
