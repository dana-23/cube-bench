"""The registry-driven scorer, and its agreement with what evaluations saved."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path

import pytest

scoring = pytest.importorskip("cube_bench.core.scoring")
ItemRecord = pytest.importorskip("cube_bench.core.records").ItemRecord

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def yes_no_records():
    rows = [
        ("Yes", "Yes", "affirmative", "affirmative:0"),
        ("Yes", "No", "affirmative", "affirmative:0"),
        ("No", "No", "negated", "negated:0"),
        ("No", None, "negated", "negated:0"),
    ]
    return [
        ItemRecord(
            index=i, gold=gold, pred=pred, correct=gold == pred, parsed=pred is not None,
            extra={"polarity": polarity, "template_id": template},
        )
        for i, (gold, pred, polarity, template) in enumerate(rows)
    ]


def test_registry_lists_only_registered_metrics():
    for test, spec in scoring.load_registry().items():
        for entry in spec.get("metrics", []):
            name = entry if isinstance(entry, str) else next(iter(entry))
            assert name in scoring.METRICS, f"{test} references unregistered metric {name}"


def test_every_shipped_evaluation_has_metrics():
    evaluations = pytest.importorskip("cube_bench.evaluations")
    registry = scoring.load_registry()
    for attr in vars(evaluations).values():
        test_type = getattr(attr, "test_type", None)
        if isinstance(test_type, str) and test_type != "reflection":
            assert test_type in registry, f"{test_type} has no entry in metrics.yaml"


def test_verification_metrics_match_hand_computation():
    scored = scoring.score("verification", yes_no_records())
    assert scored["accuracy"] == 0.5
    assert scored["confusion"] == {"tp": 1, "tn": 1, "fp": 0, "fn": 1}
    assert scored["parse_rate"] == 0.75
    assert scored["unparsed"] == 1
    assert scored["support"] == {"pos": 2, "neg": 2}
    assert scored["by_polarity"]["affirmative"]["n"] == 2
    assert scored["max_template_label_skew"] == 0.5


def test_score_uses_the_runs_own_denominator():
    scored = scoring.score("verification", yes_no_records(), total=8)
    assert scored["accuracy"] == 0.25
    assert scored["parse_rate"] == 0.375


def test_unknown_test_and_metric_are_rejected():
    with pytest.raises(KeyError):
        scoring.score("not-a-test", [])
    with pytest.raises(KeyError):
        scoring.score("verification", [], registry={"verification": {"metrics": ["nope"]}})


def test_observations_follow_a_dotted_path():
    record = ItemRecord(index=0, extra={"result": {"sample_log": {"steps_data": [{"is_correct": True}]}}})
    assert scoring.observations([record], "result.sample_log.steps_data") == [{"is_correct": True}]


def test_observations_tolerate_a_missing_path():
    assert scoring.observations([ItemRecord(index=0)], "result.sample_log.steps_data") == []


@pytest.mark.parametrize("case,filename", [
    ("verification_d3", "verification.json"),
    ("move_effect_d1", "move_effect.json"),
    ("move_effect_d3", "move_effect.json"),
    ("reconstruction_d2", "reconstruction.json"),
    ("learning_curve_d3", "learning_curve.json"),
    ("prediction_mixed_d1", "solve_moves_mixed.json"),
])
def test_saved_metrics_recompute_from_persisted_records(case, filename):
    rescore = pytest.importorskip("cube_bench.analysis.rescore")
    runs = pytest.importorskip("cube_bench.analysis.runs")
    payload = json.loads((GOLDEN_DIR / case / filename).read_text(encoding="utf-8"))
    run = runs.Run(path=GOLDEN_DIR / case / filename, payload=payload[0])
    assert rescore.rescore(run)["differences"] == {}
