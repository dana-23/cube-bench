"""The reveal x assert factorial on reflection, and the choice-exclusion check that accompanies it."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Mapping, Sequence

from cube_bench.analysis.runs import ReflectionRun, paired_items, vectors
from cube_bench.analysis.tables import reflection_stats
from cube_bench.core import metrics

REVEAL_LABELS = {True: "reveal=T", False: "reveal=F"}

TREATMENT_RANK = {"reveal=F": 0, "reveal=T": 1, "assert=never": 0, "assert=always": 1}


def arm_key(run: ReflectionRun) -> str:
    """The factorial cell a run occupies."""
    reveal = REVEAL_LABELS[bool(run.summary.get("reveal_choice"))]
    return f"{reveal},assert={run.summary.get('assert_incorrect', 'unknown')}"


def group_arms(runs: Sequence[ReflectionRun]) -> Dict[str, List[ReflectionRun]]:
    """Runs bucketed by factorial cell, replicates preserved in path order."""
    arms: Dict[str, List[ReflectionRun]] = defaultdict(list)
    for run in sorted(runs, key=lambda r: str(r.path)):
        arms[arm_key(run)].append(run)
    return dict(arms)


def cell_stats(arms: Mapping[str, Sequence[ReflectionRun]]) -> Dict[str, Dict[str, Any]]:
    """Pooled statistics per cell, with the per-replicate overthink rates that make up each pool."""
    out: Dict[str, Dict[str, Any]] = {}
    for arm, runs in arms.items():
        pooled = reflection_stats(runs)
        pooled["replicates"] = [reflection_stats([run])["otr"] for run in runs]
        pooled["paths"] = [str(run.path) for run in runs]
        out[arm] = pooled
    return out


def _level_rank(factor: str):
    """Order two arms so the milder condition is the baseline and the stronger the treatment."""
    def rank(arm: str) -> int:
        return TREATMENT_RANK[dict(zip(("reveal", "assert"), arm.split(",")))[factor]]

    return rank


def mcnemar_contrasts(
    arms: Mapping[str, Sequence[ReflectionRun]], population: str = "initially_correct"
) -> List[Dict[str, Any]]:
    """Every replicate-by-replicate paired contrast along each factor, held item-paired.

    Each contrast varies one factor with the other fixed, so a significant
    reveal contrast beside a null assert contrast localises the effect. The
    default population is the initially-correct items, which is the set the
    overthink rate is defined over; pass ``"all"`` to contrast every item.
    """
    if population not in ("initially_correct", "all"):
        raise ValueError(f"unknown population {population!r}; expected 'initially_correct' or 'all'")
    out: List[Dict[str, Any]] = []
    keys = sorted(arms)
    for first, second in ((a, b) for a, b in product(keys, keys) if a < b):
        levels = [dict(zip(("reveal", "assert"), arm.split(","))) for arm in (first, second)]
        varies = [f for f in ("reveal", "assert") if levels[0][f] != levels[1][f]]
        if len(varies) != 1:
            continue
        factor = varies[0]
        left, right = sorted((first, second), key=_level_rank(factor))
        held = dict(zip(("reveal", "assert"), left.split(",")))["assert" if factor == "reveal" else "reveal"]
        for i, run_a in enumerate(arms[left]):
            keep_a = {k for k, was_right in run_a.initial_correct().items() if was_right} \
                if population == "initially_correct" else None
            for j, run_b in enumerate(arms[right]):
                shared = paired_items(run_a, run_b)
                if keep_a is not None:
                    shared = [k for k in shared if k in keep_a]
                before, after = vectors(run_a, shared), vectors(run_b, shared)
                b, c = metrics.paired_discordant(before, after)
                out.append({
                    "factor": factor,
                    "held_fixed": held,
                    "baseline": left,
                    "treatment": right,
                    "replicate_a": i + 1,
                    "replicate_b": j + 1,
                    "population": population,
                    "n_paired": len(shared),
                    "b": b,
                    "c": c,
                    "p": metrics.mcnemar_exact(b, c),
                })
    return out


def exclusion_rows(arms: Mapping[str, Sequence[ReflectionRun]]) -> List[Dict[str, Any]]:
    """Avoid-own-choice rate and offset spread per replicate, for the arms that reveal the choice."""
    rows: List[Dict[str, Any]] = []
    for arm in sorted(arms):
        for i, run in enumerate(arms[arm]):
            prior = run.prior_choices()
            final = run.final_choices()
            indices = run.indices
            result = metrics.choice_exclusion([prior.get(k) for k in indices], [final.get(k) for k in indices])
            if not result["n_considered"]:
                continue
            rows.append({"arm": arm, "replicate": i + 1, "model": run.model, "path": str(run.path), **result})
    return rows


def summarize(runs: Sequence[ReflectionRun], model: str) -> Dict[str, Any]:
    """Everything the factorial subsection reports, from the runs alone."""
    arms = group_arms(runs)
    return {
        "model": model,
        "arms": {arm: [str(r.path) for r in group] for arm, group in arms.items()},
        "cells": cell_stats(arms),
        "contrasts": mcnemar_contrasts(arms),
        "exclusion": exclusion_rows(arms),
    }
