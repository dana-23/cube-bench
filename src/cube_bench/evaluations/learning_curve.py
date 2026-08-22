from __future__ import annotations

import logging
import random
import statistics
from collections import Counter, deque
from typing import Deque, List, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..core import BaseTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class LearningCurveTest(BaseTest):
    """Multiple-choice 'learning curve' test, starting from the first Closed-Loop failure.

    For each scramble:
      1) Closed-Loop until first non-progress / parse failure.
      2) Up to ``max_attempts`` additional decisions from the post-error state.
    Metrics are computed only over episodes that entered the LC regime.
    """

    test_type = "learning_curve"
    HIST_FIG_NAME = "learning_curve_hist.png"

    def __init__(
        self,
        assistant,
        config,
        n_moves: int,
        max_attempts: int = 6,
        accept_progress: bool = True,
        verbose: bool = False,
    ):
        super().__init__(assistant, config, n_moves, verbose)
        self.max_attempts = int(max_attempts)
        self.accept_progress = bool(accept_progress)
        self._sys_rng: random.Random = random.SystemRandom()

    def _vlog(self, *a: object) -> None:
        if self.verbose:
            logger.info(" ".join(str(x) for x in a))

    def _replan(self, vc: VirtualCube) -> Deque[str]:
        if vc.is_solved():
            return deque()
        try:
            seq = vc.solve()
            return deque(seq.split()) if seq else deque()
        except Exception as e:
            logger.warning(f"[replan] VirtualCube.solve() failed: {e!r}")
            return deque()

    def _ask_step(self, cube: VirtualCube, options, gold_letter):
        state_text = self.state_text(cube)
        d_cur = cube.get_distance()
        kwargs = {
            "n_moves": d_cur,
            "n_moves_optimal": max(d_cur - 1, 0),
            "textual_representation": state_text,
            "move_A": options["A"],
            "move_B": options["B"],
            "move_C": options["C"],
            "move_D": options["D"],
        }
        sys_prompt, user_prompt = PromptFactory.get("learning_curve", **kwargs)
        resp = self.ask(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            image=cube.to_image(),
            track_latency=False,
        )
        return self.parse_letter(resp), resp

    def run(self, num_samples: int) -> None:
        attempts_needed: List[int] = []
        solved_flags: List[bool] = []
        pre_fail_reasons: List[str] = []

        total_scrambles = 0
        episodes_with_failure = 0

        self._vlog(
            f"[start] test={self.test_type} model={self.assistant.get_name()} "
            f"samples={num_samples} depth={self.n_moves} max_attempts={self.max_attempts} "
            f"accept_progress={self.accept_progress}"
        )

        for idx in tqdm(range(num_samples), desc=f"Learning-curve ({self.n_moves} moves)"):
            total_scrambles += 1
            cube = VirtualCube()
            scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
            plan: Deque[str] = deque(str(scramble.reverse()).split())

            self._vlog(f"[sample {idx}] pre-phase start d={cube.get_distance()} scramble={list(scramble)}")

            # --- Phase 1: Closed-Loop prelude until first failure ---
            failure_happened = False
            failure_reason: Optional[str] = None

            while not cube.is_solved():
                rng = self._sys_rng

                if not plan:
                    plan = self._replan(cube)
                    if not plan:
                        self._vlog(f"[sample {idx}] pre-phase replan: empty; abort scramble")
                        failure_reason = "pre_replan_empty"
                        break

                good = self.optimal_first_moves(cube)
                correct_move = plan[0] if plan else None
                if not correct_move:
                    self._vlog(f"[sample {idx}] pre-phase plan head missing; abort scramble")
                    failure_reason = "pre_plan_missing"
                    break

                options, gold_letter = self.gen_mcq_from_good(good, rng)
                pred_letter, resp = self._ask_step(cube, options, gold_letter)
                predicted_move = options.get(pred_letter) if pred_letter else None

                if not predicted_move:
                    self._vlog(f"[sample {idx}] pre-phase parse failure; pred_letter={pred_letter!r}, resp={resp!r}")
                    failure_happened = True
                    failure_reason = "parse_error"
                    break

                made_progress, d0, d1 = self.move_makes_progress(cube, predicted_move)

                if predicted_move in good:
                    if predicted_move == correct_move:
                        cube.apply(predicted_move)
                        plan.popleft()
                        decision = "APPLY_MATCH(pre)"
                    else:
                        cube.apply(predicted_move)
                        plan = self._replan(cube)
                        decision = "APPLY_DECREASE(pre_replan)"
                    self._vlog(
                        f"[sample {idx}] pre-phase d:{d0}->{d1} pred={predicted_move} gold={gold_letter} "
                        f"correct={correct_move} good=True progress={made_progress} action={decision}"
                    )
                    continue

                cube.apply(predicted_move)
                plan.appendleft(self.inverse_move(predicted_move))
                self._vlog(
                    f"[sample {idx}] pre-phase FAILURE d:{d0}->{d1} pred={predicted_move} "
                    f"gold={gold_letter} correct={correct_move} good=False progress={made_progress} "
                    f"action=APPLY_WITH_INVERSE(pre_error)"
                )
                failure_happened = True
                failure_reason = "non_progress"
                break

            if not failure_happened or cube.is_solved():
                self._vlog(
                    f"[sample {idx}] no eligible failure for learning-curve "
                    f"(failure={failure_happened}, reason={failure_reason}, "
                    f"solved={cube.is_solved()}); skipping LC episode"
                )
                continue

            episodes_with_failure += 1
            pre_fail_reasons.append(failure_reason or "unknown")
            self._vlog(f"[sample {idx}] LC-phase start from failure={failure_reason} d_post={cube.get_distance()}")

            # --- Phase 2: Learning-curve attempts from post-error state ---
            attempts = 0
            while not cube.is_solved() and attempts < self.max_attempts:
                rng = self._sys_rng

                if not plan:
                    plan = self._replan(cube)
                    if not plan:
                        self._vlog(f"[sample {idx}] LC-phase replan: empty; abort LC episode")
                        break

                good = self.optimal_first_moves(cube)
                correct_move = plan[0] if plan else None
                if not correct_move:
                    self._vlog(f"[sample {idx}] LC-phase plan head missing; abort LC episode")
                    break

                options, gold_letter = self.gen_mcq_from_good(good, rng)
                pred_letter, resp = self._ask_step(cube, options, gold_letter)
                predicted_move = options.get(pred_letter) if pred_letter else None

                if not predicted_move:
                    attempts += 1
                    logger.warning(
                        f"[sample {idx}] LC-phase parse failure; pred_letter={pred_letter!r}, resp={resp!r}"
                    )
                    continue

                attempts += 1
                made_progress, d0, d1 = self.move_makes_progress(cube, predicted_move)

                if predicted_move == correct_move:
                    cube.apply(predicted_move)
                    plan.popleft()
                    decision = "APPLY_MATCH(LC)"
                elif predicted_move in good:
                    cube.apply(predicted_move)
                    plan = self._replan(cube)
                    decision = "APPLY_DECREASE(LC_replan)"
                else:
                    cube.apply(predicted_move)
                    plan.appendleft(self.inverse_move(predicted_move))
                    decision = "APPLY_WITH_INVERSE(LC)"

                self._vlog(
                    f"[sample {idx}] LC-phase attempt={attempts} d:{d0}->{d1} "
                    f"pred={predicted_move} gold={gold_letter} correct={correct_move} "
                    f"good={predicted_move in good} progress={made_progress} action={decision}"
                )

            attempts_needed.append(attempts)
            solved = cube.is_solved()
            solved_flags.append(solved)
            self._vlog(f"[sample {idx}] LC-phase done solved={solved} attempts={attempts}")

        # --- Aggregates (over post-error episodes only) ---
        n = len(attempts_needed)
        if n == 0:
            success_rate = 0.0
            ci_lo = ci_hi = 0.0
            solved_n = 0
            counts: Counter[int] = Counter()
            p1 = 0.0
            p_le_3 = 0.0
            med_at_solved: Optional[float] = None
            avg_attempts_all_maxed = 0.0
            avg_attempts_all = 0.0
            xs = list(range(1, self.max_attempts + 1))
            ys = [0 for _ in xs]
        else:
            solved_n = int(sum(solved_flags))
            success_rate = solved_n / n
            counts = Counter([a for a, s in zip(attempts_needed, solved_flags) if s])
            xs = list(range(1, self.max_attempts + 1))
            ys = [counts.get(k, 0) for k in xs]
            ci_lo, ci_hi = self.wilson_ci(success_rate, n)
            p1 = counts.get(1, 0) / n
            kmax = min(3, self.max_attempts)
            p_le_3 = sum(counts.get(k, 0) for k in range(1, kmax + 1)) / n
            solved_attempts = [a for a, s in zip(attempts_needed, solved_flags) if s]
            med_at_solved = statistics.median(solved_attempts) if solved_attempts else None
            avg_attempts_all_maxed = sum(
                (a if s else self.max_attempts) for a, s in zip(attempts_needed, solved_flags)
            ) / n
            avg_attempts_all = sum(attempts_needed) / n

        # Plot histogram
        fig_path = self.config.results_dir / self.HIST_FIG_NAME
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.figure(figsize=(8, 5))
            plt.bar(xs, ys)
            plt.xticks(xs)
            plt.xlabel("Attempts (when solved)")
            plt.ylabel("Number of post-error episodes")
            unsolved_n = int(n - solved_n)
            plt.title(
                f"Solve Attempts Distribution (post-error; N={n}, "
                f"Solved={solved_n}, Unsolved={unsolved_n}, SR={success_rate:.2%})"
            )
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
        finally:
            plt.close()

        self.save({
            "n_moves": self.n_moves,
            "max_attempts": self.max_attempts,
            "accept_progress": self.accept_progress,
            "total_scrambles": total_scrambles,
            "episodes_with_failure": episodes_with_failure,
            "pre_fail_reasons": pre_fail_reasons,
            "attempts_needed": attempts_needed,
            "solved_flags": solved_flags,
            "n": n,
            "solved_n": solved_n,
            "success_rate": success_rate,
            "sr_ci95": [ci_lo, ci_hi],
            "p1": p1,
            "p_le_3": p_le_3,
            "med_at_solved": med_at_solved,
            "avg_attempts_all_maxed": avg_attempts_all_maxed,
            "avg_attempts_all": avg_attempts_all,
            "hist_counts": {int(k): int(v) for k, v in counts.items()},
            "plot_path": str(fig_path),
        })

        self._vlog(
            f"[end] total_scrambles={total_scrambles} episodes_with_failure={episodes_with_failure} "
            f"SR={success_rate:.2%} CI95=({ci_lo:.3f},{ci_hi:.3f}) P(1)={p1:.3f} P(≤3)={p_le_3:.3f} "
            f"Med@Solved={med_at_solved if med_at_solved is not None else 'NA'} "
            f"Avg@All={avg_attempts_all_maxed:.2f} "
            f"plot={fig_path}"
        )
