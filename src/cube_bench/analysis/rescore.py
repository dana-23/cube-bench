"""Recompute a finished run's metrics from the per-item records it persisted.

The same registry that scores a run live scores it again here, so a number in a
saved summary can be checked rather than trusted, and a parser or metric change
can be applied to finished runs instead of forcing a re-run.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from cube_bench.analysis.runs import Run
from cube_bench.core import scoring
from cube_bench.core.records import ItemRecord


def records_from(run: Run) -> List[ItemRecord]:
    """The persisted per-item records of *run*, rebuilt as ``ItemRecord``s."""
    samples = run.payload.get("samples")
    if samples and not run.payload.get("per_item"):
        return [
            ItemRecord(index=int(s.get("sample_id", i)), extra={"result": {"sample_log": s}}, saved=s)
            for i, s in enumerate(samples)
        ]
    saved = run.payload.get("per_item")
    if not saved:
        raise LookupError(
            f"{run.origin}: no per_item records persisted; this run predates per-item persistence "
            "and can only be re-scored by re-running it"
        )
    spec = scoring.load_registry().get(run.test_type, {})
    correct_from = spec.get("correct_from")
    records = []
    for entry in saved:
        correct = bool(entry.get(correct_from)) if correct_from else bool(entry.get("correct"))
        records.append(ItemRecord(
            index=int(entry.get("index", len(records))),
            gold=entry.get("gold"),
            pred=entry.get("pred"),
            correct=correct,
            parsed=entry.get("pred") is not None,
            response=entry.get("response"),
            saved=entry,
            extra=dict(entry),
        ))
    return records


def context_for(run: Run) -> Dict[str, Any]:
    """Run-time metric inputs recovered from the saved payload, as the registry maps them."""
    spec = scoring.load_registry().get(run.test_type, {})
    context: Dict[str, Any] = {}
    for param, path in (spec.get("context_from") or {}).items():
        value = run.get(*path)
        if value is not None:
            context[param] = value
    return context


def rescore(run: Run, parse: Optional[Callable[[Optional[str]], Any]] = None) -> Dict[str, Any]:
    """Re-score *run* from its records, optionally re-parsing the stored responses first.

    Passing *parse* applies today's parser to the responses as they were
    received, which is how a parser tightening is carried onto runs that were
    scored under the old one.
    """
    records = records_from(run)
    reparsed = 0
    if parse is not None:
        for record in records:
            fresh = parse(record.response)
            if fresh != record.pred:
                reparsed += 1
                record.pred = fresh
                record.parsed = fresh is not None
                record.correct = fresh is not None and fresh == record.gold
                record.extra["pred"] = fresh
    scored = scoring.score(run.test_type, records, total=run.num_samples, context=context_for(run))
    return {
        "origin": run.origin,
        "test": run.test_type,
        "model": run.model,
        "n_records": len(records),
        "n_reparsed": reparsed,
        "metrics": scored,
        "differences": compare(run, scored),
    }


def compare(run: Run, scored: Dict[str, Any]) -> Dict[str, Any]:
    """Where a freshly computed metric disagrees with the one the run saved."""
    stored = run.payload.get("metrics") if isinstance(run.payload.get("metrics"), dict) else run.payload
    # One round-trip through JSON puts both sides in the encoding the run was saved
    # in, so tuples and integer dict keys compare equal to their stored form.
    stored = json.loads(json.dumps(stored))
    fresh = json.loads(json.dumps(scored))
    out: Dict[str, Any] = {}
    for key, now in fresh.items():
        if key not in stored:
            continue
        was = stored[key]
        if isinstance(now, float) and isinstance(was, (int, float)):
            if abs(now - float(was)) > 1e-9:
                out[key] = {"stored": was, "recomputed": now}
        elif was != now:
            out[key] = {"stored": was, "recomputed": now}
    return out
