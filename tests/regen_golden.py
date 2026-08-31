"""Rewrite tests/golden from the current code. Review the diff before committing."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# pylint: disable=wrong-import-position
from golden_cases import BY_NAME, CASES
from golden_runner import (
    generation_contract, make_assistant, prompt_snapshot, run_case,
)
from test_generation_contract import CONTRACT_DIR
from test_golden_outputs import GOLDEN_DIR
from test_prompt_snapshots import SNAPSHOT_DIR


def regenerate(names: list[str]) -> None:
    """Rewrite fixtures for *names*, or for every case when empty."""
    selected = [BY_NAME[name] for name in names] if names else list(CASES)
    for case in selected:
        target = GOLDEN_DIR / case.name
        shutil.rmtree(target, ignore_errors=True)
        assistant = make_assistant()
        produced = run_case(case, Path(tempfile.mkdtemp()) / "results", assistant)
        for name, text in produced.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
        (CONTRACT_DIR / f"{case.name}.json").write_text(
            json.dumps(generation_contract(assistant), indent=2) + "\n", encoding="utf-8"
        )
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        (SNAPSHOT_DIR / f"{case.name}.txt").write_text(
            prompt_snapshot(assistant), encoding="utf-8"
        )
        print(f"{case.name}: {len(produced)} file(s), {len(assistant.calls)} call(s)")


if __name__ == "__main__":
    regenerate(sys.argv[1:])
