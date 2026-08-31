"""Every shipped test config must construct and run through the orchestrator."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from golden_cases import make_config
from golden_runner import make_assistant

orchestrator = pytest.importorskip("cube_bench.orchestrator")
cli = pytest.importorskip("cube_bench.cli")

CONFIG_DIR = Path(orchestrator.__file__).resolve().parent / "configs" / "test"


def test_all_tests_matches_the_shipped_configs():
    stems = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != "all")
    assert sorted(cli.ALL_TESTS) == stems


@pytest.mark.parametrize("stem", sorted(cli.ALL_TESTS))
def test_config_dispatches_and_runs(stem, tmp_path, monkeypatch):
    assistant = make_assistant()
    monkeypatch.setattr(
        orchestrator, "ModelAssistant", lambda *args, **kwargs: assistant
    )

    results_dir = tmp_path / stem
    results_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    orch = orchestrator.TestOrchestrator(
        model_name="mock-model", config=make_config(results_dir)
    )
    orch.run_test(test_cfg=OmegaConf.load(CONFIG_DIR / f"{stem}.yaml"), num_samples=2)

    assert assistant.calls, f"{stem} made no model calls"
    assert list(results_dir.rglob("*.json")), f"{stem} wrote no results"


def test_unknown_test_name_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orchestrator, "ModelAssistant", lambda *args, **kwargs: make_assistant()
    )
    orch = orchestrator.TestOrchestrator(
        model_name="mock-model", config=make_config(tmp_path)
    )
    with pytest.raises(ValueError, match="Unknown test name"):
        orch.run_test(test_cfg=OmegaConf.create({"name": "nope"}), num_samples=1)
