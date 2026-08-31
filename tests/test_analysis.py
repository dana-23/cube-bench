"""Statistics and run loading behind the reported numbers."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json

import pytest

metrics = pytest.importorskip("cube_bench.core.metrics")
runs = pytest.importorskip("cube_bench.analysis.runs")


@pytest.mark.parametrize("b,c,expected", [
    (1, 25, 8.05e-07), (1, 23, 2.98e-06), (0, 20, 1.91e-06), (0, 18, 7.63e-06),
    (1, 4, 0.375), (1, 6, 0.125), (4, 3, 1.0), (3, 4, 1.0),
    (5, 5, 1.0), (6, 2, 0.289), (5, 7, 0.774), (1, 18, 7.63e-05),
])
def test_exact_mcnemar_matches_reported_contrasts(b, c, expected):
    assert metrics.mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-2)


def test_mcnemar_is_symmetric_and_bounded():
    assert metrics.mcnemar_exact(0, 0) == 1.0
    assert metrics.mcnemar_exact(7, 3) == metrics.mcnemar_exact(3, 7)
    with pytest.raises(ValueError):
        metrics.mcnemar_exact(-1, 2)


def test_paired_discordant_counts_each_direction():
    assert metrics.paired_discordant([True, False, True], [False, True, True]) == (1, 1)
    with pytest.raises(ValueError):
        metrics.paired_discordant([True], [True, False])


def test_correlation_and_interval_match_the_reported_figure():
    kappa = [-0.025, 0.0014, -0.084, 0.0024, -0.074, -0.029, 0.680, 0.376]
    adherence = [14.0, 18.0, 8.0, 18.0, 10.0, 14.0, 88.0, 66.0]
    r = metrics.pearson_r(kappa, adherence)
    assert r == pytest.approx(0.995, abs=5e-4)
    lo, hi = metrics.pearson_ci(r, len(kappa))
    assert (round(lo, 3), round(hi, 3)) == (0.969, 0.999)


@pytest.mark.parametrize("stat,df,expected", [(3.841, 1, 0.05), (5.991, 2, 0.05), (11.345, 3, 0.01)])
def test_chi_square_tail_matches_published_critical_values(stat, df, expected):
    assert metrics.chi2_sf(stat, df) == pytest.approx(expected, abs=5e-4)


def test_choice_exclusion_counts_offsets_cyclically():
    result = metrics.choice_exclusion(["A", "A", "B", None], ["B", "A", "A", "C"])
    assert result["n_considered"] == 3
    assert result["n_avoided"] == 2
    assert result["avoid_rate"] == pytest.approx(2 / 3)
    assert result["offset_counts"] == {"+1": 1, "+2": 0, "+3": 1}


def test_least_squares_recovers_a_known_line():
    slope, intercept = metrics.least_squares([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert (slope, intercept) == pytest.approx((2.0, 1.0))


def test_load_run_refuses_a_file_holding_several_runs(tmp_path):
    path = tmp_path / "solve_moves_mixed.json"
    path.write_text(json.dumps([
        {"model_name": "m", "test_type": "prediction", "num_samples": 1, "timestamp": "a"},
        {"model_name": "m", "test_type": "prediction", "num_samples": 50, "timestamp": "b"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="holds 2 runs"):
        runs.load_run(path)
    assert [r.record_index for r in runs.load_runs(path)] == [0, 1]
    assert runs.load_runs(path)[1].num_samples == 50


def test_run_reports_where_each_number_came_from(tmp_path):
    path = tmp_path / "verification.json"
    path.write_text(json.dumps({"model_name": "m", "test_type": "verification",
                                "num_samples": 4, "meta": {"scramble_depth": 5}}), encoding="utf-8")
    run = runs.load_run(path)
    assert run.depth == 5
    assert run.origin.endswith("verification.json#0")


def test_run_without_a_sample_count_is_an_error(tmp_path):
    path = tmp_path / "verification.json"
    path.write_text(json.dumps({"model_name": "m", "test_type": "verification"}), encoding="utf-8")
    with pytest.raises(KeyError):
        _ = runs.load_run(path).num_samples
