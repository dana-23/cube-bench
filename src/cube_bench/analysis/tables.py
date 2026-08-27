"""The numbers behind each table the paper prints, read off the runs that produced them."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from cube_bench.analysis.runs import ReflectionRun, Run
from cube_bench.core import metrics

MODEL_ORDER = (
    "Gemma 3-27B",
    "GLM-4.5V",
    "Llama 4-Scout 17B",
    "Qwen 2.5-VL 7B",
    "Qwen 2.5-VL 32B",
    "Qwen3-VL-Thinking 30B",
    "Gemini 2.5 Pro",
    "Claude Sonnet 4.5",
)


def _ordered(models: Sequence[str]) -> List[str]:
    """Model labels in the paper's row order, with unknown labels appended alphabetically."""
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def _pct(value: Optional[float]) -> Optional[float]:
    """A recorded proportion as percentage points."""
    return None if value is None else 100.0 * float(value)


def perception(
    reconstruction: Mapping[str, Mapping[int, Run]],
    verification: Mapping[str, Run],
    depths: Sequence[int] = (1, 2, 3),
) -> List[Dict[str, Any]]:
    """Table 1: reconstruction accuracy at each depth beside cross-modal verification."""
    rows: List[Dict[str, Any]] = []
    for model in _ordered(list({*reconstruction, *verification})):
        row: Dict[str, Any] = {"model": model}
        for depth in depths:
            run = reconstruction.get(model, {}).get(depth)
            row[f"element_wise_d{depth}"] = _pct(run.payload.get("average_accuracy_element_wise")) if run else None
            row[f"matrix_d{depth}"] = _pct(run.payload.get("average_accuracy_overall")) if run else None
            row[f"n_d{depth}"] = run.num_samples if run else None
            row[f"max_prior_deviation_d{depth}"] = run.payload.get("max_prior_deviation") if run else None
            row[f"origin_d{depth}"] = run.origin if run else None
        ver = verification.get(model)
        row["verification_depth"] = ver.depth if ver else None
        row["balanced_accuracy"] = _pct(ver.get("metrics", "balanced_accuracy")) if ver else None
        row["parse_rate"] = _pct(ver.get("metrics", "parse_rate")) if ver else None
        row["yes_rate"] = _pct(ver.get("metrics", "yes_rate")) if ver else None
        row["verification_n"] = ver.num_samples if ver else None
        row["verification_origin"] = ver.origin if ver else None
        rows.append(row)
    return rows


def movepred(runs: Mapping[str, Mapping[str, Run]]) -> List[Dict[str, Any]]:
    """Table 2: optimal move prediction across the three input modalities, with the image delta."""
    rows: List[Dict[str, Any]] = []
    for model in _ordered(list(runs)):
        row: Dict[str, Any] = {"model": model}
        for mode in ("mixed", "image", "text"):
            run = runs[model].get(mode)
            row[f"{mode}_accuracy"] = _pct(run.payload.get("average_accuracy")) if run else None
            row[f"{mode}_parse_rate"] = _pct(run.payload.get("parse_rate", 1.0)) if run else None
            row[f"{mode}_n"] = run.num_samples if run else None
            row[f"{mode}_origin"] = run.origin if run else None
        mixed, text = row["mixed_accuracy"], row["text_accuracy"]
        row["delta_img"] = None if mixed is None or text is None else mixed - text
        rows.append(row)
    return rows


def move_effect_by_depth(
    runs: Mapping[str, Mapping[int, Run]], depths: Sequence[int] = (1, 2, 3)
) -> List[Dict[str, Any]]:
    """Table 4: micro-accuracy, macro-F1 and kappa for the causal move-effect probe."""
    rows: List[Dict[str, Any]] = []
    for model in _ordered(list(runs)):
        row: Dict[str, Any] = {"model": model}
        for depth in depths:
            run = runs[model].get(depth)
            row[f"micro_acc_d{depth}"] = _pct(run.payload.get("micro_acc")) if run else None
            row[f"macro_f1_d{depth}"] = run.payload.get("macro_f1") if run else None
            row[f"kappa_d{depth}"] = run.payload.get("kappa") if run else None
            row[f"n_d{depth}"] = run.num_samples if run else None
            row[f"origin_d{depth}"] = run.origin if run else None
        rows.append(row)
    return rows


def reflection_stats(runs: Sequence[ReflectionRun]) -> Dict[str, Any]:
    """EFR, OTR and net change for one arm, pooled over replicates on the same items."""
    wrong = fixed = right = flipped = 0
    initial = final = total = 0
    for run in runs:
        before, after = run.initial_correct(), run.final_correct()
        for index in run.indices:
            was, now = before.get(index, False), after.get(index, False)
            total += 1
            initial += int(was)
            final += int(now)
            if was:
                right += 1
                flipped += int(not now)
            else:
                wrong += 1
                fixed += int(now)
    efr = metrics.safe_prop(fixed, wrong)
    otr = metrics.safe_prop(flipped, right)
    initial_acc = metrics.safe_prop(initial, total)
    final_acc = metrics.safe_prop(final, total)
    return {
        "n_items": total,
        "n_replicates": len(runs),
        "initial_accuracy": 100.0 * initial_acc,
        "final_accuracy": 100.0 * final_acc,
        "delta_points": 100.0 * (final_acc - initial_acc),
        "efr": 100.0 * efr,
        "efr_n": wrong,
        "efr_ci95": [100.0 * b for b in metrics.wilson_ci(efr, wrong)],
        "otr": 100.0 * otr,
        "otr_n": right,
        "otr_ci95": [100.0 * b for b in metrics.wilson_ci(otr, right)],
    }


def reflection_guided(runs: Mapping[str, Sequence[ReflectionRun]]) -> List[Dict[str, Any]]:
    """Table 5: net change, error-fix rate and overthink rate per model."""
    rows: List[Dict[str, Any]] = []
    for model in _ordered(list(runs)):
        arms = runs[model]
        row: Dict[str, Any] = {"model": model, **reflection_stats(arms)}
        row["origins"] = [str(run.path) for run in arms]
        if row["efr_n"] < 10:
            row["caveat"] = f"EFR rests on {row['efr_n']} initially-wrong items and is indicative only"
        rows.append(row)
    return rows


def learning_curve(runs: Mapping[str, Run]) -> List[Dict[str, Any]]:
    """Table 7: recovery success rate and attempt distribution under a fixed budget."""
    rows: List[Dict[str, Any]] = []
    for model in sorted(runs, key=lambda m: -float(runs[m].payload.get("success_rate", 0.0))):
        run = runs[model]
        bounds = run.payload.get("sr_ci95") or []
        rows.append({
            "model": model,
            "depth": run.depth,
            "max_attempts": run.payload.get("max_attempts"),
            "success_rate": _pct(run.payload.get("success_rate")),
            "sr_ci95": [100.0 * b for b in bounds] if bounds else None,
            "p_solved_first": run.payload.get("p1"),
            "p_solved_by_3": run.payload.get("p_le_3"),
            "median_attempts_when_solved": run.payload.get("med_at_solved"),
            "mean_attempts_all": run.payload.get("avg_attempts_all"),
            "n": run.num_samples,
            "origin": run.origin,
        })
    return rows
