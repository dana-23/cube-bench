"""Proportion, agreement and divergence statistics shared by the evaluations."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a Bernoulli proportion ``p`` over ``n``."""
    if n <= 0 or not (0.0 <= p <= 1.0) or math.isnan(p):
        return (float("nan"), float("nan"))
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def safe_prop(count: int, total: int) -> float:
    """``count / total``, or 0.0 when the denominator is empty."""
    return (count / total) if total else 0.0


def balanced_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Mean of the true-positive and true-negative rates."""
    pos = tp + fn
    neg = tn + fp
    tpr = (tp / pos) if pos else 0.0
    tnr = (tn / neg) if neg else 0.0
    return 0.5 * (tpr + tnr) if (pos or neg) else 0.0


def dot(a: Mapping[str, float], b: Mapping[str, float], keys: Iterable[str]) -> float:
    """Inner product of two distributions restricted to ``keys``."""
    return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)


def cohens_kappa(observed: float, expected: float) -> float:
    """Chance-corrected agreement given observed and chance-expected accuracy."""
    return (observed - expected) / (1.0 - expected) if (1.0 - expected) > 0 else 0.0


def per_class_prf(
    confusion: Mapping[str, Mapping[str, int]], classes: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Per-class precision, recall and F1 from a gold-to-predicted confusion table."""
    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    f1_scores: Dict[str, float] = {}
    for cls in classes:
        tp = confusion[cls].get(cls, 0)
        fn = sum(v for k, v in confusion[cls].items() if k != cls)
        fp = sum(confusion[g].get(cls, 0) for g in classes if g != cls)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        recall[cls] = rec
        precision[cls] = prec
        f1_scores[cls] = f1
    return precision, recall, f1_scores


def macro_f1(f1_scores: Mapping[str, float], classes: Sequence[str]) -> float:
    """Unweighted mean F1 across ``classes``."""
    return sum(f1_scores[c] for c in classes) / len(classes) if classes else 0.0


def jensen_shannon(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in bits between two distributions over the same keys."""
    keys = list(p.keys())
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}

    def _kl(a, b):
        s = 0.0
        for k in keys:
            if a[k] > 0 and b[k] > 0:
                s += a[k] * math.log(a[k] / b[k], 2)
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
