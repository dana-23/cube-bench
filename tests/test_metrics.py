"""Value parity between core.metrics and the inline expressions it was extracted from."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import math

import pytest

metrics = pytest.importorskip("cube_bench.core.metrics")

CLASSES = ("DECREASE", "NO_CHANGE", "INCREASE")

CONFUSION = {
    "DECREASE": {"DECREASE": 7, "NO_CHANGE": 2, "INCREASE": 1},
    "NO_CHANGE": {"DECREASE": 3, "NO_CHANGE": 5, "INCREASE": 4},
    "INCREASE": {"DECREASE": 1, "NO_CHANGE": 0, "INCREASE": 9},
}


def inline_balanced_accuracy(tp, tn, fp, fn):
    pos = tp + fn
    neg = tn + fp
    tpr = (tp / pos) if pos else 0.0
    tnr = (tn / neg) if neg else 0.0
    return 0.5 * (tpr + tnr) if (pos or neg) else 0.0


def inline_per_class(confusion, tri):
    per_class_recall, per_class_precision, per_class_f1 = {}, {}, {}
    for cls in tri:
        tp = confusion[cls].get(cls, 0)
        fn = sum(v for k, v in confusion[cls].items() if k != cls)
        fp = sum(confusion[g].get(cls, 0) for g in tri if g != cls)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class_recall[cls] = rec
        per_class_precision[cls] = prec
        per_class_f1[cls] = f1
    return per_class_precision, per_class_recall, per_class_f1


@pytest.mark.parametrize("count,total", [(0, 0), (0, 5), (3, 5), (5, 5), (7, 3)])
def test_safe_prop_matches_move_effect(count, total):
    move_effect = pytest.importorskip("cube_bench.evaluations.move_effect")
    original = move_effect.MoveEffectTest._safe_prop  # pylint: disable=protected-access
    assert metrics.safe_prop(count, total) == original(count, total)


@pytest.mark.parametrize("p,q", [
    ({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
    ({"a": 0.5, "b": 0.5, "c": 0.0}, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
    ({"a": 1.0, "b": 0.0, "c": 0.0}, {"a": 0.0, "b": 1.0, "c": 0.0}),
    ({"a": 0.7, "b": 0.2, "c": 0.1}, {"a": 0.25, "b": 0.25, "c": 0.5}),
])
def test_jensen_shannon_matches_move_effect(p, q):
    move_effect = pytest.importorskip("cube_bench.evaluations.move_effect")
    original = move_effect.MoveEffectTest._jsd  # pylint: disable=protected-access
    assert metrics.jensen_shannon(p, q) == original(p, q)


@pytest.mark.parametrize("tp,tn,fp,fn", [
    (0, 0, 0, 0), (5, 5, 0, 0), (5, 0, 5, 0), (3, 7, 2, 8), (0, 9, 1, 0),
])
def test_balanced_accuracy_matches_verification(tp, tn, fp, fn):
    assert metrics.balanced_accuracy(tp, tn, fp, fn) == inline_balanced_accuracy(tp, tn, fp, fn)


def test_per_class_prf_matches_move_effect():
    assert metrics.per_class_prf(CONFUSION, CLASSES) == inline_per_class(CONFUSION, CLASSES)


def test_macro_f1_matches_move_effect():
    _, _, f1_scores = metrics.per_class_prf(CONFUSION, CLASSES)
    assert metrics.macro_f1(f1_scores, CLASSES) == sum(
        f1_scores[c] for c in CLASSES
    ) / 3.0


@pytest.mark.parametrize("observed,expected", [(0.9, 0.33), (0.33, 0.33), (0.5, 1.0), (0.0, 0.0)])
def test_cohens_kappa_matches_move_effect(observed, expected):
    inline = (observed - expected) / (1.0 - expected) if (1.0 - expected) > 0 else 0.0
    assert metrics.cohens_kappa(observed, expected) == inline


def test_dot_matches_move_effect():
    priors = {"DECREASE": 0.5, "NO_CHANGE": 0.2, "INCREASE": 0.3}
    q = {"DECREASE": 0.4, "NO_CHANGE": 0.4, "INCREASE": 0.2}
    inline = sum(priors.get(k, 0.0) * q.get(k, 0.0) for k in CLASSES)
    assert metrics.dot(priors, q, CLASSES) == inline


def test_wilson_ci_matches_base_test():
    base = pytest.importorskip("cube_bench.core.base")
    for p, n in ((0.5, 100), (0.0, 10), (1.0, 10), (0.62, 50), (0.973, 37)):
        assert metrics.wilson_ci(p, n) == base.BaseTest.wilson_ci(p, n)


@pytest.mark.parametrize("p,n", [(0.5, 0), (0.5, -1), (1.5, 10), (float("nan"), 10)])
def test_wilson_ci_rejects_undefined_input(p, n):
    assert all(math.isnan(bound) for bound in metrics.wilson_ci(p, n))
