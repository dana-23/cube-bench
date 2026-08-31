"""Metrics computed from the records an evaluation produced, driven by a registry.

Evaluations describe *what* they measured by listing metric names in
``configs/metrics.yaml``; the functions here do the measuring. The same
functions score a finished run offline, so a number reported live and the same
number recomputed later come from one implementation.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from cube_bench.core import metrics as m
from cube_bench.core.records import ItemRecord

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "metrics.yaml"

Observation = Dict[str, Any]
MetricFn = Callable[..., Dict[str, Any]]
METRICS: Dict[str, MetricFn] = {}


def metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Register *name* as a metric the registry may reference."""

    def register(fn: MetricFn) -> MetricFn:
        if name in METRICS:
            raise ValueError(f"metric {name!r} is already registered")
        METRICS[name] = fn
        return fn

    return register


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, Any]:
    """The metric registry, read once."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"metric registry not found at {REGISTRY_PATH}")
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}


def observations(records: Sequence[ItemRecord], unit: str = "item") -> List[Observation]:
    """Records flattened into the units a metric scores.

    ``unit: item`` scores one observation per record; a dotted path into
    ``extra`` instead scores the list found there, which is how tests that ask
    several questions per item, or one per step, are handled.
    """
    out: List[Observation] = []
    for record in records:
        if unit == "item":
            out.append({
                "gold": record.gold,
                "pred": record.pred,
                "correct": record.correct,
                "parsed": record.parsed,
                **record.extra,
            })
            continue
        node: Any = record.extra
        for step in unit.split("."):
            node = (node or {}).get(step, [])
        for entry in node or []:
            observation = dict(entry)
            if "pred" in entry:
                observation.setdefault("parsed", entry["pred"] is not None)
            if "gold" in entry and "pred" in entry:
                observation.setdefault("correct", entry["gold"] == entry["pred"])
            out.append(observation)
    return out


def _matches(observation: Observation, label: Any) -> bool:
    """Case-insensitive equality against a gold or predicted label."""
    if observation is None or label is None:
        return False
    if isinstance(observation, str) and isinstance(label, str):
        return observation.lower() == label.lower()
    return observation == label


def _binary_counts(obs: Sequence[Observation], positive: Any) -> Tuple[int, int, int, int]:
    """True/false positive and negative counts against *positive*, over parsed predictions."""
    tp = tn = fp = fn = 0
    for observation in obs:
        if observation["pred"] is None:
            continue
        gold_pos = _matches(observation["gold"], positive)
        pred_pos = _matches(observation["pred"], positive)
        if gold_pos and pred_pos:
            tp += 1
        elif gold_pos:
            fn += 1
        elif pred_pos:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


@metric("accuracy")
def _accuracy(obs: Sequence[Observation], total: int, key: str = "accuracy") -> Dict[str, Any]:
    """Share of observations scored correct, over the run's own denominator."""
    return {key: m.safe_prop(sum(1 for o in obs if o["correct"]), total)}


@metric("balanced_accuracy")
def _balanced_accuracy(obs: Sequence[Observation], total: int, positive: Any = "Yes") -> Dict[str, Any]:
    """Mean of the true-positive and true-negative rates."""
    del total
    tp, tn, fp, fn = _binary_counts(obs, positive)
    return {"balanced_accuracy": m.balanced_accuracy(tp, tn, fp, fn)}


@metric("parse_rate")
def _parse_rate(obs: Sequence[Observation], total: int) -> Dict[str, Any]:
    """Share of responses the strict parser accepted, and its complement."""
    rate = m.safe_prop(sum(1 for o in obs if o["pred"] is not None), total)
    return {"parse_rate": rate, "parse_violation": 1.0 - rate}


@metric("unparsed")
def _unparsed(obs: Sequence[Observation], total: int) -> Dict[str, Any]:
    """Count of responses no answer could be read from."""
    return {"unparsed": total - sum(1 for o in obs if o["pred"] is not None)}


@metric("label_rate")
def _label_rate(obs: Sequence[Observation], total: int, label: Any, key: str) -> Dict[str, Any]:
    """Share of parsed predictions equal to *label*, which exposes a standing answer bias."""
    del total
    parsed = [o for o in obs if o["pred"] is not None]
    return {key: m.safe_prop(sum(1 for o in parsed if _matches(o["pred"], label)), len(parsed))}


@metric("binary_confusion")
def _binary_confusion(obs: Sequence[Observation], total: int, positive: Any = "Yes") -> Dict[str, Any]:
    """The 2x2 confusion counts."""
    del total
    tp, tn, fp, fn = _binary_counts(obs, positive)
    return {"confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


@metric("binary_support")
def _binary_support(obs: Sequence[Observation], total: int, positive: Any = "Yes") -> Dict[str, Any]:
    """How many items carried each gold label."""
    del total
    pos = sum(1 for o in obs if _matches(o["gold"], positive))
    return {"support": {"pos": pos, "neg": len(obs) - pos}}


def _group_stat(name: str, group: Sequence[Observation], positive: Any) -> Any:
    """One statistic over a subgroup of observations."""
    if name == "n":
        return len(group)
    if name == "accuracy":
        return m.safe_prop(sum(1 for o in group if o["correct"]), len(group))
    if name == "balanced_accuracy":
        tp, tn, fp, fn = _binary_counts(group, positive)
        return m.balanced_accuracy(tp, tn, fp, fn)
    if name == "positive_label_share":
        return m.safe_prop(sum(1 for o in group if _matches(o["gold"], positive)), len(group))
    raise KeyError(f"unknown group statistic {name!r}")


def _buckets(obs: Sequence[Observation], field: str) -> Dict[Any, List[Observation]]:
    """Observations grouped by their value of *field*."""
    buckets: Dict[Any, List[Observation]] = defaultdict(list)
    for observation in obs:
        buckets[observation.get(field)].append(observation)
    return buckets


@metric("group")
def _group(
    obs: Sequence[Observation],
    total: int,
    *,
    field: str,
    key: str,
    stats: Sequence[str],
    names: Optional[Mapping[str, str]] = None,
    positive: Any = "Yes",
) -> Dict[str, Any]:
    """The same statistics computed within each level of *field*, which is how a fairness audit reads."""
    del total
    buckets = _buckets(obs, field)
    labels = dict(names or {})
    return {
        key: {
            str(level): {labels.get(s, s): _group_stat(s, group, positive) for s in stats}
            for level, group in buckets.items()
        }
    }


@metric("max_group_label_skew")
def _max_group_label_skew(
    obs: Sequence[Observation], total: int, field: str, key: str, positive: Any = "Yes"
) -> Dict[str, Any]:
    """Largest deviation from a balanced gold split within any level of *field*."""
    del total
    shares = [
        abs(_group_stat("positive_label_share", group, positive) - 0.5)
        for group in _buckets(obs, field).values()
    ]
    return {key: max(shares, default=0.0)}


@metric("micro_accuracy")
def _micro_accuracy(obs: Sequence[Observation], total: int) -> Dict[str, Any]:
    """Accuracy over every label asked, not every item."""
    del total
    return {"micro_acc": m.safe_prop(sum(1 for o in obs if o["correct"]), len(obs)),
            "labels_total": len(obs)}


@metric("confusion_matrix")
def _confusion_matrix(obs: Sequence[Observation], total: int, classes: Sequence[str]) -> Dict[str, Any]:
    """Gold-to-predicted counts and per-class support."""
    del total
    seen: List[str] = list(classes)
    for observation in obs:
        if observation["pred"] not in seen and observation["pred"] is not None:
            seen.append(observation["pred"])
    table = {gold: {pred: 0 for pred in seen} for gold in classes}
    support = {cls: 0 for cls in classes}
    for observation in obs:
        gold, pred = observation["gold"], observation["pred"]
        if gold in support:
            support[gold] += 1
        if gold in table and pred in table[gold]:
            table[gold][pred] += 1
    return {"confusion": table, "support": support}


@metric("class_report")
def _class_report(obs: Sequence[Observation], total: int, classes: Sequence[str]) -> Dict[str, Any]:
    """Per-class precision, recall and F1, plus their unweighted mean."""
    del total
    table = _confusion_matrix(obs, 0, classes)["confusion"]
    precision, recall, f1 = m.per_class_prf(table, classes)
    return {
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "macro_f1": m.macro_f1(f1, classes),
    }


@metric("chance_corrected")
def _chance_corrected(obs: Sequence[Observation], total: int, classes: Sequence[str]) -> Dict[str, Any]:
    """Cohen's kappa against the model's own predicted mix, with the baselines it corrects for."""
    del total
    golds = [o["gold"] for o in obs]
    preds = [o["pred"] for o in obs]
    priors = {c: m.safe_prop(sum(1 for g in golds if g == c), len(golds)) for c in classes}
    mix = {c: m.safe_prop(sum(1 for p in preds if p == c), sum(1 for p in preds if p in classes)) for c in classes}
    observed = m.safe_prop(sum(1 for o in obs if o["correct"]), len(obs))
    expected = m.dot(priors, mix, classes)
    return {
        "kappa": m.cohens_kappa(observed, expected),
        "gold_priors": priors,
        "pred_mix": mix,
        "expected_dot": expected,
        "maj_baseline": max(priors.values()) if priors else 0.0,
        "prior_sample_baseline": m.dot(priors, priors, classes),
    }


def score(
    test: str,
    records: Sequence[ItemRecord],
    total: Optional[int] = None,
    context: Optional[Mapping[str, Any]] = None,
    registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Every metric the registry lists for *test*, computed from *records*.

    ``total`` is the denominator the run reports against, which is the requested
    sample count rather than the number of records when a run was cut short.
    ``context`` supplies run-time values a metric needs but the registry cannot
    know, such as the attempt budget a recovery run was given.
    """
    spec = (registry or load_registry()).get(test)
    if spec is None:
        raise KeyError(f"no metrics registered for {test!r}")
    obs = observations(records, spec.get("unit", "item"))
    shared = {k: v for k, v in spec.items() if k not in ("unit", "metrics")}
    shared.update(context or {})
    out: Dict[str, Any] = {}
    for entry in spec.get("metrics", []):
        name, params = (entry, {}) if isinstance(entry, str) else next(iter(entry.items()))
        if name not in METRICS:
            raise KeyError(f"{test}: unknown metric {name!r}; registered: {', '.join(sorted(METRICS))}")
        code = METRICS[name].__code__
        accepted = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        merged = {**{k: v for k, v in shared.items() if k in accepted}, **(params or {})}
        out.update(METRICS[name](obs, total if total is not None else len(records), **merged))
    return out


@metric("group_accuracy")
def _group_accuracy(obs: Sequence[Observation], total: int, field: str, key: str) -> Dict[str, Any]:
    """Accuracy within each level of *field*, flattened to one number per level."""
    del total
    buckets = _buckets(obs, field)
    return {key: {level: _group_stat("accuracy", g, None) for level, g in buckets.items()}}


@metric("mean")
def _mean(obs: Sequence[Observation], total: int, field: str, key: str) -> Dict[str, Any]:
    """Mean of a per-item score, for tests that grade an item on a scale rather than pass/fail."""
    del total
    values = [float(o[field]) for o in obs if o.get(field) is not None]
    return {key: (sum(values) / len(values)) if values else 0.0}


@metric("parsed_count")
def _parsed_count(obs: Sequence[Observation], total: int, key: str = "correct_parse") -> Dict[str, Any]:
    """How many responses the parser accepted."""
    del total
    return {key: sum(1 for o in obs if o["parsed"])}


@metric("recovery")
def _recovery(obs: Sequence[Observation], total: int, max_attempts: int) -> Dict[str, Any]:
    """Attempt-budget statistics for episodes that reached the post-error phase.

    Reported over the episodes that actually failed first, since an episode
    that never erred offers no evidence about recovery.
    """
    del total
    attempts = [int(o["attempts"]) for o in obs]
    solved = [bool(o["solved"]) for o in obs]
    n = len(attempts)
    solved_attempts = [a for a, ok in zip(attempts, solved) if ok]
    counts: Dict[int, int] = {}
    for value in solved_attempts:
        counts[value] = counts.get(value, 0) + 1
    if n == 0:
        return {
            "pre_fail_reasons": [], "attempts_needed": [], "solved_flags": [], "n": 0, "solved_n": 0,
            "success_rate": 0.0, "sr_ci95": [0.0, 0.0], "p1": 0.0, "p_le_3": 0.0,
            "med_at_solved": None, "avg_attempts_all_maxed": 0.0, "avg_attempts_all": 0.0,
            "hist_counts": {},
        }
    success_rate = len(solved_attempts) / n
    kmax = min(3, max_attempts)
    return {
        "pre_fail_reasons": [str(o["failure_reason"]) for o in obs],
        "attempts_needed": attempts,
        "solved_flags": solved,
        "n": n,
        "solved_n": len(solved_attempts),
        "success_rate": success_rate,
        "sr_ci95": list(m.wilson_ci(success_rate, n)),
        "p1": counts.get(1, 0) / n,
        "p_le_3": sum(counts.get(k, 0) for k in range(1, kmax + 1)) / n,
        "med_at_solved": statistics.median(solved_attempts) if solved_attempts else None,
        "avg_attempts_all_maxed": sum(a if ok else max_attempts for a, ok in zip(attempts, solved)) / n,
        "avg_attempts_all": sum(attempts) / n,
        "hist_counts": {int(k): int(v) for k, v in counts.items()},
    }


def _decided(observation: Observation) -> bool:
    """True when the model actually chose an option, rather than abstaining or failing to parse."""
    return not observation.get("abstained", False) and not observation.get("parse_fail", False)


@metric("step_positions")
def _step_positions(obs: Sequence[Observation], total: int, steps: int) -> Dict[str, Any]:
    """Accuracy at each step index, and the first-step versus later-step split.

    At the first step the history and markov arms are identical by
    construction, so any history effect can only appear from step two on.
    """
    del total
    correct = [0] * steps
    seen = [0] * steps
    for observation in obs:
        index = int(observation.get("step", 0))
        if 0 <= index < steps:
            seen[index] += 1
            correct[index] += int(bool(observation.get("is_correct")))
    later_correct = sum(correct[1:])
    later_seen = sum(seen[1:])
    return {
        "step_accuracy": [m.safe_prop(c, t) for c, t in zip(correct, seen)],
        "per_step_totals": seen,
        "per_step_correct": correct,
        "step_position_split": {
            "t1_accuracy": m.safe_prop(correct[0] if correct else 0, seen[0] if seen else 0),
            "t1_n": seen[0] if seen else 0,
            "t2plus_accuracy": m.safe_prop(later_correct, later_seen),
            "t2plus_n": later_seen,
        },
    }


@metric("selective_decisions")
def _selective_decisions(
    obs: Sequence[Observation], total: int, steps: int, idk_weight: float = 0.25
) -> Dict[str, Any]:
    """Coverage, selective accuracy and abstention-adjusted accuracy over per-step decisions."""
    del total
    n_correct = sum(1 for o in obs if o.get("is_correct"))
    n_idk = sum(1 for o in obs if o.get("abstained"))
    parse_failures = sum(1 for o in obs if o.get("parse_fail"))
    n_wrong = sum(1 for o in obs if _decided(o) and not o.get("is_correct"))
    decisions = len(obs)
    answered = n_correct + n_wrong
    by_step_total = [0] * steps
    by_step_idk = [0] * steps
    for observation in obs:
        index = int(observation.get("step", 0))
        if 0 <= index < steps:
            by_step_total[index] += 1
            by_step_idk[index] += int(bool(observation.get("abstained")))
    return {
        "selective": {
            "coverage_overall": m.safe_prop(answered, decisions),
            "coverage_by_step": [m.safe_prop(t - z, t) for t, z in zip(by_step_total, by_step_idk)],
            "selective_accuracy": m.safe_prop(n_correct, answered),
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_idk": n_idk,
            "total_decisions": decisions,
        },
        "apa": ((n_correct + idk_weight * n_idk) / decisions) if decisions else 0.0,
        "parse_failures": parse_failures,
        "parse_fail_rate": m.safe_prop(parse_failures, decisions),
    }
