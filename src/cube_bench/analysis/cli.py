"""Log the numbers the paper reports: inventory the runs, then report the ones you name."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cube_bench.analysis import derive, factorial, rescore, tables
from cube_bench.analysis.runs import (
    RESULT_FILES,
    ReflectionRun,
    Run,
    find_reflections,
    find_runs,
    load_reflection,
    load_runs,
)

# Keyed on the name the runner records, i.e. ``ModelSpec.name`` in
# runtime.model_assistant.MODEL_REGISTRY. The hyphenated spellings are older
# hand-written run directories that predate the registry.
MODEL_LABELS = {
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini2.5-pro": "Gemini 2.5 Pro",
    "gemma3": "Gemma 3-27B",
    "gemma-3-27b": "Gemma 3-27B",
    "glm4.5v": "GLM-4.5V",
    "glm-4.5v": "GLM-4.5V",
    "llama4": "Llama 4-Scout 17B",
    "llama-4-scout-17b": "Llama 4-Scout 17B",
    "qwen2.5-7b": "Qwen 2.5-VL 7B",
    "qwen2.5-vl-7b": "Qwen 2.5-VL 7B",
    "qwen2.5-32b": "Qwen 2.5-VL 32B",
    "qwen2.5-vl-32b": "Qwen 2.5-VL 32B",
    "qwen3-vl-thinking": "Qwen3-VL-Thinking 30B",
    "qwen3-vl-30b": "Qwen3-VL-Thinking 30B",
}


def label(model: str) -> str:
    """The paper's row label for a recorded model name."""
    return MODEL_LABELS.get(model, model)


def _num(value: Any, digits: int = 2) -> str:
    """A number for the log, or a dash when the run did not record it."""
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, list):
        return "[" + ", ".join(_num(v, digits) for v in value) + "]"
    return str(value)


def scan(root: Path) -> List[Dict[str, Any]]:
    """Every run under *root*: which test, model and depth it covers, and how big it was.

    Appended records are listed separately, because a result file can hold
    several runs and only one of them is the one a table cell quotes.
    """
    rows: List[Dict[str, Any]] = []
    for test in sorted(RESULT_FILES):
        for run in find_runs(root, test):
            rows.append({
                "test": test,
                "model": label(run.model),
                "depth": run.depth,
                "n": run.payload.get("num_samples"),
                "prompt_type": run.payload.get("prompt_type"),
                "timestamp": run.timestamp,
                "origin": run.origin,
            })
    for run in find_reflections(root):
        rows.append({
            "test": "reflection",
            "model": label(run.model),
            "depth": None,
            "n": run.summary.get("n_items"),
            "prompt_type": factorial.arm_key(run),
            "timestamp": "",
            "origin": str(run.path),
        })
    return rows


def scan_report(rows: Sequence[Dict[str, Any]]) -> str:
    """The inventory grouped by test and model, so duplicate candidates are visible."""
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["test"]][row["model"]].append(row)
    lines: List[str] = []
    for test in sorted(grouped):
        lines.append(test)
        for model in sorted(grouped[test]):
            for row in grouped[test][model]:
                facets = [f"d={row['depth']}" if row["depth"] is not None else "",
                          f"n={row['n']}", row["prompt_type"] or "", row["timestamp"][:19]]
                lines.append(f"  {model:22s} {' '.join(f for f in facets if f):38s} {row['origin']}")
        lines.append("")
    return "\n".join(lines)


def _perception_lines(by_test: Dict[str, List[Run]]) -> List[str]:
    """Reconstruction and verification numbers, with the colour-prior deviation each run measured."""
    rec: Dict[str, Dict[int, Run]] = defaultdict(dict)
    for run in by_test.get("reconstruction", []):
        rec[label(run.model)][run.depth] = run
    ver = {label(run.model): run for run in by_test.get("verification", [])}
    if not rec and not ver:
        return []
    depths = sorted({d for by_depth in rec.values() for d in by_depth}) or [1, 2, 3]
    lines = ["tab:perception"]
    for row in tables.perception(rec, ver, depths):
        for depth in depths:
            if row.get(f"element_wise_d{depth}") is None:
                continue
            lines.append(
                f"  {row['model']} d={depth}: element-wise {_num(row[f'element_wise_d{depth}'])} "
                f"matrix {_num(row[f'matrix_d{depth}'])} n={row[f'n_d{depth}']} "
                f"[max_prior_deviation {_num(row[f'max_prior_deviation_d{depth}'], 3)}] "
                f"{row[f'origin_d{depth}']}"
            )
        if row.get("balanced_accuracy") is not None:
            lines.append(
                f"  {row['model']} verification d={row['verification_depth']}: "
                f"balanced {_num(row['balanced_accuracy'])} parse {_num(row['parse_rate'])} "
                f"yes-rate {_num(row['yes_rate'])} n={row['verification_n']} {row['verification_origin']}"
            )
    return lines + [""]


def _movepred_lines(by_test: Dict[str, List[Run]]) -> List[str]:
    """Prediction accuracy per modality, with the image-versus-text delta."""
    if not by_test.get("prediction"):
        return []
    pred: Dict[str, Dict[str, Run]] = defaultdict(dict)
    for run in by_test["prediction"]:
        pred[label(run.model)][str(run.payload.get("prompt_type", "unknown"))] = run
    lines = ["tab:movepred"]
    for row in tables.movepred(pred):
        cells = " ".join(
            f"{mode}={_num(row[f'{mode}_accuracy'], 1)}(n={row[f'{mode}_n']})"
            for mode in ("mixed", "image", "text") if row.get(f"{mode}_accuracy") is not None
        )
        lines.append(f"  {row['model']}: {cells} delta_img={_num(row['delta_img'], 1)}")
        lines.extend(f"      {mode}: {row[f'{mode}_origin']}"
                     for mode in ("mixed", "image", "text") if row.get(f"{mode}_origin"))
    return lines + [""]


def _move_effect_lines(by_test: Dict[str, List[Run]]) -> List[str]:
    """Micro-accuracy, macro-F1 and kappa per depth."""
    if not by_test.get("move_effect"):
        return []
    eff: Dict[str, Dict[int, Run]] = defaultdict(dict)
    for run in by_test["move_effect"]:
        eff[label(run.model)][run.depth] = run
    depths = sorted({r.depth for r in by_test["move_effect"] if r.depth is not None})
    lines = ["tab:move_effect_by_depth"]
    for row in tables.move_effect_by_depth(eff, depths):
        for depth in depths:
            if row.get(f"micro_acc_d{depth}") is None:
                continue
            lines.append(
                f"  {row['model']} d={depth}: acc {_num(row[f'micro_acc_d{depth}'])} "
                f"macro-F1 {_num(row[f'macro_f1_d{depth}'], 3)} kappa {_num(row[f'kappa_d{depth}'], 3)} "
                f"n={row[f'n_d{depth}']} {row[f'origin_d{depth}']}"
            )
    return lines + [""]


def _recovery_lines(by_test: Dict[str, List[Run]]) -> List[str]:
    """Recovery rates, and what every smaller attempt budget would have bought."""
    if not by_test.get("learning_curve"):
        return []
    curves = {label(run.model): run for run in by_test["learning_curve"]}
    lines = ["tab:learning_curve"]
    for row in tables.learning_curve(curves):
        lines.append(
            f"  {row['model']} d={row['depth']}: SR {_num(row['success_rate'])} CI {_num(row['sr_ci95'])} "
            f"P(1) {_num(row['p_solved_first'])} P(<=3) {_num(row['p_solved_by_3'])} "
            f"med@solved {_num(row['median_attempts_when_solved'])} "
            f"avg@all {_num(row['mean_attempts_all'])} n={row['n']} {row['origin']}"
        )
    for run in by_test["learning_curve"]:
        curve = derive.attempt_budget_curve(run)
        lines.append(f"  attempt budget, {label(run.model)} d={curve['depth']} "
                     f"(reported max {curve['max_attempts']})")
        for point in curve["curve"]:
            lines.append(f"      <={point['budget']}: {100 * point['success_rate']:.1f}% "
                         f"({point['solved']}/{point['n']}) CI {_num([100 * b for b in point['ci95']], 1)}")
    return lines + [""]


def _closed_loop_lines(by_test: Dict[str, List[Run]]) -> List[str]:
    """Teacher adherence and perfect-solve rate per run, labelled by history arm."""
    steps = [r for r in by_test.get("step_by_step", []) if r.payload.get("samples")]
    if not steps:
        return []
    lines = ["closed-loop control"]
    for run in steps:
        history = run.get("history_config", "enabled", default=None)
        arm = "" if history is None else f" history={history}"
        lines.append(f"  {label(run.model)} d={run.depth}{arm}: TA {derive.teacher_adherence(run):.1f} "
                     f"perfect {derive.perfect_solve_rate(run):.1f} n={run.num_samples} {run.origin}")
    return lines + [""]


def _report_runs(runs: Sequence[Run]) -> List[str]:
    """Numbers for each named evaluation run, grouped by the table it feeds."""
    by_test: Dict[str, List[Run]] = defaultdict(list)
    for run in runs:
        by_test[run.test_type].append(run)
    lines: List[str] = []
    for section in (_perception_lines, _movepred_lines, _move_effect_lines,
                    _recovery_lines, _closed_loop_lines):
        lines.extend(section(by_test))
    return lines


def _report_reflection(runs: Sequence[ReflectionRun]) -> List[str]:
    """Table 5 rows for each model, plus the factorial when a model spans several cells."""
    lines: List[str] = []
    by_model: Dict[str, List[ReflectionRun]] = defaultdict(list)
    for run in runs:
        by_model[label(run.model)].append(run)

    revealed = {m: [r for r in group if r.summary.get("reveal_choice")] for m, group in by_model.items()}
    revealed = {m: group for m, group in revealed.items() if group}
    if revealed:
        lines.append("tab:reflection-guided (revealed-choice arms only)")
        for row in tables.reflection_guided(revealed):
            lines.append(f"  {row['model']}: N={row['n_replicates']}x{row['n_items'] // row['n_replicates']} "
                         f"delta {row['delta_points']:+.1f} EFR {_num(row['efr'])} (n={row['efr_n']}) "
                         f"OTR {_num(row['otr'])} CI {_num(row['otr_ci95'])} (n={row['otr_n']})")
            if row.get("caveat"):
                lines.append(f"      caveat: {row['caveat']}")
            for origin in row["origins"]:
                lines.append(f"      {origin}")
        lines.append("")

    for model, group in by_model.items():
        if len({factorial.arm_key(r) for r in group}) < 2:
            continue
        summary = factorial.summarize(group, model)
        lines.append(f"reflection factorial, {model}")
        for arm, cell in sorted(summary["cells"].items()):
            reps = ", ".join(f"{r:.1f}" for r in cell["replicates"])
            lines.append(f"  {arm}: OTR {cell['otr']:.1f} CI {_num(cell['otr_ci95'], 1)} "
                         f"n={cell['otr_n']} replicates ({reps})")
        for row in summary["contrasts"]:
            lines.append(f"  {row['factor']:6s} ({row['held_fixed']}) rep{row['replicate_a']}x{row['replicate_b']}: "
                         f"b={row['b']} c={row['c']} p={row['p']:.3g} n={row['n_paired']} [{row['population']}]")
        for row in summary["exclusion"]:
            offsets = "/".join(str(v) for v in row["offset_counts"].values())
            lines.append(f"  exclusion {row['arm']} rep{row['replicate']}: "
                         f"avoid {row['avoid_rate']:.3f} ({row['n_avoided']}/{row['n_considered']}) "
                         f"offsets {offsets} chi2={row['offset_chi2']:.2f} p={row['offset_p']:.2f}")
        lines.append("")
    return lines


def report(paths: Sequence[Path]) -> str:
    """Log the numbers for exactly the runs named, with each one's provenance."""
    runs: List[Run] = []
    reflections: List[ReflectionRun] = []
    for path in paths:
        if (Path(path) / "summary.json").exists():
            reflections.append(load_reflection(path))
        else:
            runs.extend(load_runs(path))
    lines = _report_runs(runs) + _report_reflection(reflections)
    return "\n".join(lines) if lines else "no runs given"


def rescore_report(paths: Sequence[Path], reparse: bool = False) -> str:
    """Recompute each named run's metrics and report where they disagree with what was saved."""
    from cube_bench.core.base import BaseTest  # pylint: disable=import-outside-toplevel

    parse = BaseTest.parse_letter if reparse else None
    lines: List[str] = []
    for path in paths:
        for run in load_runs(path):
            try:
                result = rescore.rescore(run, parse=parse)
            except LookupError as exc:
                lines.append(f"  skipped {run.origin}: {exc}")
                continue
            verdict = "matches saved metrics" if not result["differences"] else "DIFFERS from saved metrics"
            lines.append(f"{run.test_type} {label(run.model)} n={result['n_records']}: {verdict}")
            if reparse:
                lines.append(f"  responses re-parsed differently: {result['n_reparsed']}")
            for key, change in result["differences"].items():
                lines.append(f"  {key}: saved {change['stored']} -> recomputed {change['recomputed']}")
            lines.append(f"  {run.origin}")
    return "\n".join(lines) if lines else "no runs given"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Inventory a runs directory, or log the numbers for the runs named."""
    parser = argparse.ArgumentParser(description="Log the numbers the paper reports.")
    sub = parser.add_subparsers(dest="command", required=True)

    scanner = sub.add_parser("scan", help="list every run under a directory, with n and timestamp")
    scanner.add_argument("--runs", type=Path, default=Path("outputs"))
    scanner.add_argument("--json", type=Path, default=None, help="also write the inventory here")

    reporter = sub.add_parser("report", help="log the numbers for the runs named")
    reporter.add_argument("paths", type=Path, nargs="+", help="result files or run directories")

    checker = sub.add_parser("rescore", help="recompute saved metrics from the persisted per-item records")
    checker.add_argument("paths", type=Path, nargs="+", help="result files or run directories")
    checker.add_argument("--reparse", action="store_true",
                         help="re-parse stored responses with the current strict parser first")

    args = parser.parse_args(argv)
    if args.command == "scan":
        rows = scan(args.runs)
        print(scan_report(rows))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
            print(f"wrote {args.json}")
        return 0
    if args.command == "rescore":
        print(rescore_report(args.paths, args.reparse))
        return 0
    print(report(args.paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
