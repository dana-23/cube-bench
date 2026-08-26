"""Evaluation cases shared by the golden, prompt-snapshot and generation-contract suites."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, NamedTuple

import pytest

evaluations = pytest.importorskip("cube_bench.evaluations")
Config = pytest.importorskip("cube_bench.config").Config

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "cube_bench" / "prompts"
SEED = 20260826


class Case(NamedTuple):
    """One runnable evaluation configuration and its sample count."""

    name: str
    build: Callable[[Any, Config, Path], Any]
    samples: int


def make_config(results_dir: Path) -> Config:
    """Run config pointing every artifact at *results_dir*."""
    return Config(
        dataset_path=results_dir / "unused.json",
        prompts_path=PROMPTS_DIR / "prompts.yaml",
        results_dir=results_dir,
    )


def _prediction(prompt_type: str, n_moves: int):
    def build(assistant, config, _tmp):
        return evaluations.SolveMovesTest(
            assistant, config, prompt_type=prompt_type, n_moves=n_moves
        )
    return build


def _verification(assistant, config, _tmp):
    return evaluations.VerificationTest(assistant, config, n_moves=3)


def _reconstruction(n_moves: int):
    def build(assistant, config, _tmp):
        return evaluations.ReconstructionTest(assistant, config, n_moves=n_moves)
    return build


def _move_effect(n_moves: int):
    def build(assistant, config, _tmp):
        return evaluations.MoveEffectTest(assistant, config, n_moves=n_moves)
    return build


def _invariance_sweep(assistant, config, _tmp):
    return evaluations.InvarianceSweepTest(assistant, config, n_moves=3)


def _learning_curve(assistant, config, _tmp):
    task = evaluations.LearningCurveTest(assistant, config, n_moves=3, max_attempts=6)
    task._sys_rng = random.Random(SEED)  # pylint: disable=protected-access
    return task


def _step_by_step(**kwargs):
    def build(assistant, config, tmp):
        return evaluations.StepByStepTest(
            assistant, config, n_moves=3, concurrency=1,
            checkpoint=True, checkpoint_dir=str(tmp / "ckpt"), **kwargs,
        )
    return build


def _reflection(**kwargs):
    def build(assistant, config, _tmp):
        return evaluations.ReflectionTest(
            assistant=assistant, config=config,
            reflection_prompts=PROMPTS_DIR / "reflection.yaml",
            n_moves=1, **kwargs,
        )
    return build


CASES: tuple[Case, ...] = (
    Case("prediction_mixed_d1", _prediction("mixed", 1), 8),
    Case("prediction_text_d1", _prediction("text", 1), 8),
    Case("prediction_image_d3", _prediction("image", 3), 8),
    Case("prediction_no_authority_d1", _prediction("mixed_no_authority", 1), 8),
    Case("verification_d3", _verification, 24),
    Case("reconstruction_d2", _reconstruction(2), 8),
    Case("reconstruction_d3", _reconstruction(3), 8),
    Case("move_effect_d1", _move_effect(1), 8),
    Case("move_effect_d2", _move_effect(2), 8),
    Case("move_effect_d3", _move_effect(3), 8),
    Case("invariance_sweep_d3", _invariance_sweep, 6),
    Case("learning_curve_d3", _learning_curve, 6),
    Case("step_by_step_markov_d3", _step_by_step(), 8),
    Case("step_by_step_history_d3", _step_by_step(history_enabled=True), 8),
    Case("step_by_step_idk_d3", _step_by_step(idk_enabled=True), 8),
    Case("reflection_redacted_reveal", _reflection(reflection_type="Redacted", reveal_choice=True), 6),
    Case("reflection_redacted_hidden", _reflection(reflection_type="Redacted", reveal_choice=False), 6),
    Case("reflection_unredacted_never", _reflection(
        reflection_type="Unredacted", assert_incorrect="never", reanswer_mode="neutral"), 6),
    Case("reflection_wrong_only", _reflection(reflection_type="Redacted", reflect_all=False), 6),
    Case("degenerate_n1", _prediction("mixed", 1), 1),
)

BY_NAME = {case.name: case for case in CASES}
