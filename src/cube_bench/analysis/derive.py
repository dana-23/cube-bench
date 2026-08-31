"""Quantities the paper reports that are computed from a run rather than stored in it."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from cube_bench.analysis.runs import Run, iter_steps
from cube_bench.core import metrics


def teacher_adherence(run: Run) -> float:
    """Share of steps matching an optimal move from the state the model actually reached.

    Counted from the per-step verdicts rather than the episode tally, which
    older runs left at zero, over the unconditional denominator ``episodes x d``.
    """
    samples = run.payload.get("samples", [])
    depth = run.depth
    if not samples or not depth:
        raise ValueError(f"{run.origin}: no per-episode steps to score")
    correct = sum(
        1 for sample in samples for step in sample.get("steps_data", []) if step.get("is_correct")
    )
    return 100.0 * correct / (len(samples) * depth)


def perfect_solve_rate(run: Run) -> float:
    """Share of episodes that never left an optimal trajectory."""
    ratio = run.payload.get("perfect_solves_ratio")
    if ratio is None:
        raise KeyError(f"{run.origin}: no perfect-solve ratio recorded")
    return 100.0 * float(ratio)


def attempt_budget_curve(run: Run, budgets: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Success rate as a function of attempt budget, recovered from the per-episode attempt counts.

    Answers what the reported budget would have bought at every smaller budget,
    without re-running: an episode solved on attempt ``k`` is solved under any
    budget of at least ``k``.
    """
    attempts = run.payload.get("attempts_needed")
    solved = run.payload.get("solved_flags")
    if attempts is None or solved is None:
        raise KeyError(f"{run.origin}: needs attempts_needed and solved_flags")
    if len(attempts) != len(solved):
        raise ValueError(f"{run.origin}: {len(attempts)} attempt counts against {len(solved)} solved flags")
    total = len(attempts)
    ceiling = int(run.payload.get("max_attempts", max(attempts) if attempts else 0))
    grid = list(budgets) if budgets is not None else list(range(1, ceiling + 1))
    curve = []
    for budget in grid:
        hits = sum(1 for a, ok in zip(attempts, solved) if ok and int(a) <= budget)
        rate = metrics.safe_prop(hits, total)
        curve.append({
            "budget": budget,
            "solved": hits,
            "n": total,
            "success_rate": rate,
            "ci95": metrics.wilson_ci(rate, total),
        })
    return {
        "model": run.model,
        "depth": run.depth,
        "max_attempts": ceiling,
        "origin": run.origin,
        "curve": curve,
    }


def rescore_steps(run: Run, parse) -> Dict[str, Any]:
    """Re-parse every stored step response with *parse*, reporting where the verdict moved.

    The stored ``predicted_letter`` was produced by whichever parser ran at
    collection time, so this is how a parser change is applied to finished runs
    without re-running them.
    """
    changed: List[Dict[str, Any]] = []
    steps = 0
    lost = 0
    for step in iter_steps(run):
        steps += 1
        stored = step.get("predicted_letter")
        reparsed = parse(step.get("full_response"))
        if reparsed == stored:
            continue
        if stored is not None and reparsed is None:
            lost += 1
        changed.append({
            "step": step.get("step"),
            "stored": stored,
            "reparsed": reparsed,
            "correct_letter": step.get("correct_letter"),
            "response": (step.get("full_response") or "")[-160:],
        })
    return {
        "model": run.model,
        "depth": run.depth,
        "origin": run.origin,
        "n_steps": steps,
        "n_changed": len(changed),
        "n_newly_unparsed": lost,
        "changed": changed,
    }
