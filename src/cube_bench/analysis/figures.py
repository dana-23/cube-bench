"""Figure builders, each returning the statistics it draws so text and plot cannot drift."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from cube_bench.core import metrics

TA_BANDS = ((0, 20, "#f8d7da"), (20, 60, "#ffe5cc"), (60, 100, "#dff0d8"))


def _pyplot():
    """Pyplot on a headless backend, imported only when a figure is actually drawn."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    return plt


def kappa_vs_ta(
    kappa: Mapping[str, float],
    adherence: Mapping[str, float],
    out_path: Optional[Path] = None,
    depth: int = 1,
) -> Dict[str, Any]:
    """Move-effect $\\kappa$ against closed-loop teacher adherence, with the least-squares fit.

    Returns the correlation and fit alongside the drawn points, so the reported
    $r$ and the plotted line come from one computation.
    """
    models = [m for m in kappa if m in adherence]
    if len(models) < 3:
        raise ValueError(f"need at least three models present in both inputs, got {len(models)}")
    xs = [float(kappa[m]) for m in models]
    ys = [float(adherence[m]) for m in models]
    r = metrics.pearson_r(xs, ys)
    lo, hi = metrics.pearson_ci(r, len(models))
    slope, intercept = metrics.least_squares(xs, ys)
    if out_path is not None:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.scatter(xs, ys, s=34, color="#2b6cb0", zorder=3)
        for model, x, y in zip(models, xs, ys):
            ax.annotate(model, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        span = [min(xs), max(xs)]
        ax.plot(span, [slope * x + intercept for x in span], color="#c05621", lw=1.2, zorder=2)
        ax.set_xlabel(r"Move-Effect $\kappa$")
        ax.set_ylabel("Closed-Loop TA (%)")
        ax.set_title(rf"$d={depth}$:  $r={r:.3f}$  95% CI [{lo:.3f}, {hi:.3f}]", fontsize=9)
        ax.grid(alpha=0.25, zorder=1)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return {
        "depth": depth,
        "models": models,
        "kappa": xs,
        "teacher_adherence": ys,
        "pearson_r": r,
        "pearson_ci95": [lo, hi],
        "slope": slope,
        "intercept": intercept,
        "n": len(models),
        "path": str(out_path) if out_path else None,
    }


def step_by_step(
    adherence: Mapping[str, Mapping[int, float]],
    perfect: Optional[Mapping[str, Mapping[int, float]]] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Teacher adherence against scramble depth per model, over the reported TA bands."""
    if not adherence:
        raise ValueError("no models to plot")
    depths = sorted({d for by_depth in adherence.values() for d in by_depth})
    if out_path is not None:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        for lo, hi, color in TA_BANDS:
            ax.axhspan(lo, hi, color=color, zorder=0)
        for model, by_depth in adherence.items():
            xs = [d for d in depths if d in by_depth]
            ax.plot(xs, [by_depth[d] for d in xs], marker="o", ms=4, lw=1.3, label=model, zorder=3)
        ax.set_xticks(depths)
        ax.set_xlabel("Scramble depth $d$")
        ax.set_ylabel("TA (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.25, zorder=1)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return {
        "depths": depths,
        "teacher_adherence": {m: dict(v) for m, v in adherence.items()},
        "perfect_solve": {m: dict(v) for m, v in (perfect or {}).items()},
        "path": str(out_path) if out_path else None,
    }


def kappa_vs_ta_panels(
    kappa: Mapping[int, Mapping[str, float]],
    adherence: Mapping[int, Mapping[str, float]],
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """One $\\kappa$-versus-adherence panel per depth, as the appendix prints them."""
    depths = sorted(set(kappa) & set(adherence))
    if not depths:
        raise ValueError("no depth present in both inputs")
    panels = {d: kappa_vs_ta(kappa[d], adherence[d], None, d) for d in depths}
    if out_path is not None:
        plt = _pyplot()
        fig, axes = plt.subplots(1, len(depths), figsize=(3.0 * len(depths), 2.9), squeeze=False)
        for ax, depth in zip(axes[0], depths):
            panel = panels[depth]
            ax.scatter(panel["kappa"], panel["teacher_adherence"], s=26, color="#2b6cb0", zorder=3)
            span = [min(panel["kappa"]), max(panel["kappa"])]
            ax.plot(span, [panel["slope"] * x + panel["intercept"] for x in span], color="#c05621", lw=1.1)
            ax.set_title(rf"$d={depth}$: $r={panel['pearson_r']:.3f}$", fontsize=9)
            ax.set_xlabel(r"$\kappa$")
            ax.grid(alpha=0.25, zorder=1)
        axes[0][0].set_ylabel("TA (%)")
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return {"panels": panels, "path": str(out_path) if out_path else None}
