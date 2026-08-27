"""Proportion, agreement and divergence statistics shared by the evaluations."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


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


def paired_discordant(before: Sequence[bool], after: Sequence[bool]) -> Tuple[int, int]:
    """Discordant pair counts ``(b, c)``: b flipped wrong-to-right, c right-to-wrong."""
    if len(before) != len(after):
        raise ValueError(f"paired vectors differ in length: {len(before)} vs {len(after)}")
    b = sum(1 for x, y in zip(before, after) if not x and y)
    c = sum(1 for x, y in zip(before, after) if x and not y)
    return b, c


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts ``b`` and ``c``."""
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative: b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson product-moment correlation between two equal-length samples."""
    if len(xs) != len(ys):
        raise ValueError(f"samples differ in length: {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(vx * vy)
    return (cov / denom) if denom > 0 else float("nan")


def _gamma_upper_regularized(s: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(s, x), by series below x=s+1 and continued fraction above."""
    if x < 0 or s <= 0:
        raise ValueError(f"domain error: s={s}, x={x}")
    if x == 0:
        return 1.0
    log_prefix = -x + s * math.log(x) - math.lgamma(s)
    if x < s + 1.0:
        term = 1.0 / s
        total = term
        for i in range(1, 1000):
            term *= x / (s + i)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(log_prefix)
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(log_prefix) * h


def chi2_sf(stat: float, df: int) -> float:
    """Upper-tail probability of the chi-square distribution with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError(f"df must be positive: {df}")
    if stat <= 0:
        return 1.0
    return _gamma_upper_regularized(df / 2.0, stat / 2.0)


def chi_square_gof(
    counts: Sequence[int], expected: Optional[Sequence[float]] = None
) -> Tuple[float, int, float]:
    """Goodness-of-fit statistic, degrees of freedom and p-value against ``expected`` (uniform by default)."""
    observed = list(counts)
    total = sum(observed)
    if not observed:
        raise ValueError("chi_square_gof needs at least one cell")
    exp = list(expected) if expected is not None else [total / len(observed)] * len(observed)
    if len(exp) != len(observed):
        raise ValueError(f"cell counts differ: {len(observed)} observed vs {len(exp)} expected")
    if any(e <= 0 for e in exp):
        raise ValueError("expected cell counts must be positive")
    stat = sum((o - e) ** 2 / e for o, e in zip(observed, exp))
    df = len(observed) - 1
    return stat, df, chi2_sf(stat, df)


def choice_exclusion(
    prior: Sequence[Optional[str]],
    chosen: Sequence[Optional[str]],
    letters: Sequence[str] = ("A", "B", "C", "D"),
) -> Dict[str, Any]:
    """Rate at which a re-answer avoids the revealed prior choice, with the offset spread it lands on.

    Offsets are cyclic distances from the prior letter, so a model that merely
    excludes its own choice at random spreads uniformly over ``+1..+len-1``.
    """
    if len(prior) != len(chosen):
        raise ValueError(f"paired vectors differ in length: {len(prior)} vs {len(chosen)}")
    index = {letter: i for i, letter in enumerate(letters)}
    size = len(letters)
    offsets = {k: 0 for k in range(1, size)}
    considered = 0
    avoided = 0
    for before, after in zip(prior, chosen):
        if before not in index or after not in index:
            continue
        considered += 1
        offset = (index[after] - index[before]) % size
        if offset:
            avoided += 1
            offsets[offset] += 1
    counts = [offsets[k] for k in range(1, size)]
    stat, df, p = chi_square_gof(counts) if avoided else (float("nan"), size - 2, float("nan"))
    return {
        "n_considered": considered,
        "n_avoided": avoided,
        "avoid_rate": safe_prop(avoided, considered),
        "offset_counts": {f"+{k}": offsets[k] for k in range(1, size)},
        "offset_chi2": stat,
        "offset_df": df,
        "offset_p": p,
    }


def pearson_ci(r: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Fisher z-transformed confidence interval for a correlation ``r`` over ``n`` pairs."""
    if n < 4 or abs(r) >= 1.0 or math.isnan(r):
        return (float("nan"), float("nan"))
    zr = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    return (math.tanh(zr - z * se), math.tanh(zr + z * se))


def least_squares(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Slope and intercept of the least-squares line through the sample."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError(f"need at least two paired points, got {len(xs)} and {len(ys)}")
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        raise ValueError("cannot fit a line through points with no spread in x")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    return slope, my - slope * mx
