from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..core import BaseTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class StepByStepTest(BaseTest):
    """Closed-loop: ask for next move at each step, accept optimal or teacher."""

    test_type = "step_by_step"

    def __init__(
        self,
        assistant,
        config,
        n_moves: int,
        verbose: bool = False,
        idk_enabled: bool = False,
        idk_weight: float = 0.25,
        idk_policy: str = "teacher_on_abstain",
        idk_conf_threshold: Optional[float] = None,
    ):
        super().__init__(assistant, config, n_moves, verbose)
        self.per_step_totals: List[int] = [0] * n_moves
        self.per_step_correct: List[int] = [0] * n_moves
        self.first_error_step: List[int] = []
        self.confusion = defaultdict(Counter)

        self.idk_enabled = bool(idk_enabled)
        self.idk_weight = float(idk_weight)
        self.idk_policy = str(idk_policy)
        self.idk_conf_threshold = 50  # kept from prior behavior
        self.per_step_idk: List[int] = [0] * n_moves

        if self.idk_enabled:
            logger.info(f"IDK Enabled -- Policy: {self.idk_policy} -- Weight: {self.idk_weight}")

    def _eval(self, text: str, gold_letter: str) -> Tuple[bool, Optional[str]]:
        """Parse model output; return (is_correct, predicted_letter or 'IDK'/None)."""
        if self.idk_enabled and self.parse_idk(text):
            return False, "IDK"
        pred = self.parse_letter(text)
        if pred is None:
            logger.warning("Could not parse model's output")
            logger.info(f"Model's full response:\n\n{text}")
            return False, None
        return (pred == gold_letter), pred

    def _build_prompts(self, kwargs: Dict[str, Any]) -> Tuple[str, str]:
        """(sys, user). If IDK is enabled, use the abstention-aware template; otherwise PromptFactory."""
        if not self.idk_enabled:
            return PromptFactory.get("step_by_step", **kwargs)

        n_moves = kwargs.get("n_moves", self.n_moves)
        text_state = kwargs.get("textual_representation", "")
        A, B, C, D = kwargs["move_A"], kwargs["move_B"], kwargs["move_C"], kwargs["move_D"]
        conf_thr = self.idk_conf_threshold

        sys_prompt = (
            "You are an expert Rubik's-Cube solver. Pick exactly ONE move (A, B, C, D or IDK) "
            "that most reduces the cube's distance to solved.\n\n"
            "Context\n"
            f"- The cube is {n_moves} moves from solved under God's distance (HTM).\n"
            "- You will receive a textual cube state (ground truth) and an image (reference only). "
            "Use the TEXT ONLY to decide.\n\n"
            "Decision rule (deterministic)\n"
            "1) For each candidate, internally simulate that move on the textual state and estimate the resulting distance d1 (HTM).\n"
            "2) If any candidate solves the cube (d1=0), choose that candidate.\n"
            "3) Otherwise choose the candidate with the lowest d1.\n"
            "4) If there is a tie on d1, break ties by letter: A ≺ B ≺ C ≺ D.\n"
        )
        if conf_thr is not None:
            sys_prompt += f"5) If you are <{int(conf_thr)}% confident, abstain with IDK.\n"

        sys_prompt += (
            "\nOutput format (STRICT)\n"
            "Return exactly one of the following on a single line:\n"
            "<ANSWER> A </ANSWER>\n"
            "<ANSWER> B </ANSWER>\n"
            "<ANSWER> C </ANSWER>\n"
            "<ANSWER> D </ANSWER>\n"
            "<ANSWER> IDK </ANSWER>\n\n"
            "No explanations, no extra text.\n"
        )

        user_prompt = (
            f"The cube is {n_moves} moves away from being solved.\n\n"
            "Textual Cube State (ground truth):\n"
            f"{text_state}\n\n"
            "Candidate moves:\n"
            f"A: {A}\nB: {B}\nC: {C}\nD: {D}\n"
            "E: I don't know (abstain)\n\n"
            "Respond with exactly one line (A/B/C/D or IDK):\n"
            "<ANSWER> X </ANSWER>"
        )
        return sys_prompt, user_prompt

    def run(self, num_samples: int):
        solve_depths: List[int] = []
        all_sample_logs: List[Dict[str, Any]] = []

        n_correct = n_wrong = n_idk = 0
        total_decisions = 0

        for idx in tqdm(range(num_samples), desc=f"Step-by-step ({self.n_moves} moves)"):
            cube = VirtualCube()
            scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
            solution_path = str(deepcopy(scramble).reverse()).split()
            correct_steps = 0
            teacher_help = 0

            if self.verbose:
                logger.info(f"[sample {idx}] Scramble: {scramble}")
                logger.info(f"[sample {idx}] Teacher path: {solution_path}")

            sample_log = {
                "sample_id": idx,
                "scramble": str(scramble),
                "solution_path": solution_path,
                "steps_data": [],
            }

            for step_i, teacher_move in enumerate(solution_path):
                if cube.is_solved():
                    break

                state_text = self.state_text(cube)

                seed_bytes = f"{self.n_moves}:{idx}:{step_i}".encode()
                seed_int = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
                step_rng = random.Random(seed_int)

                options, gold_letter = self.gen_mcq_balanced(cube, teacher_move, step_rng)

                kwargs = {
                    "n_moves": self.n_moves - correct_steps - teacher_help,
                    "n_moves_optimal": (self.n_moves - 1) - correct_steps - teacher_help,
                    "textual_representation": state_text,
                    "move_A": options["A"],
                    "move_B": options["B"],
                    "move_C": options["C"],
                    "move_D": options["D"],
                    "metric": "HTM (Half-Turn Metric)",
                }
                sys_prompt, user_prompt = self._build_prompts(kwargs)

                resp = self.ask(
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                    image=cube.to_image(),
                )

                is_correct, pred_letter = self._eval(resp, gold_letter)
                options_move = options.get(pred_letter) if pred_letter and pred_letter != "IDK" else None

                total_decisions += 1
                self.per_step_totals[step_i] += 1

                if pred_letter is None:
                    sample_log["steps_data"].append({
                        "step": step_i,
                        "cube_state": state_text,
                        "options": options,
                        "correct_letter": gold_letter,
                        "full_response": resp,
                        "predicted_letter": None,
                        "is_correct": False,
                        "parse_fail": True,
                    })
                    self.first_error_step.append(step_i + 1)
                    if self.verbose:
                        logger.info(f"[sample {idx}] Parse failure at step {step_i + 1}; ending episode.")
                    break

                if self.idk_enabled and pred_letter == "IDK":
                    logger.info("Model responded with IDK.")
                    n_idk += 1
                    self.per_step_idk[step_i] += 1

                    sample_log["steps_data"].append({
                        "step": step_i,
                        "cube_state": state_text,
                        "options": options,
                        "correct_letter": gold_letter,
                        "full_response": resp,
                        "predicted_letter": "IDK",
                        "is_correct": False,
                        "abstained": True,
                        "idk_policy": self.idk_policy,
                    })

                    if self.idk_policy == "teacher_on_abstain":
                        cube.apply(teacher_move)
                        teacher_help += 1
                        continue
                    else:
                        self.first_error_step.append(step_i + 1)
                        break

                if self.verbose:
                    logger.info(f"Model's chosen option: {pred_letter} -> {options_move}")

                self.per_step_correct[step_i] += int(is_correct)

                if pred_letter and pred_letter != "IDK":
                    if is_correct:
                        n_correct += 1
                    else:
                        n_wrong += 1

                if options_move is not None:
                    self.confusion[teacher_move][options_move] += 1

                sample_log["steps_data"].append({
                    "step": step_i,
                    "cube_state": state_text,
                    "options": options,
                    "correct_letter": gold_letter,
                    "full_response": resp,
                    "predicted_letter": pred_letter,
                    "is_correct": is_correct,
                })

                good_moves = self.optimal_first_moves(cube)
                if self.verbose:
                    logger.info(f"[Sample: {idx} Step: {step_i}] Oracle-good moves: {sorted(good_moves)}")

                made_progress = False
                if options_move is not None:
                    made_progress, _, _ = self.move_makes_progress(cube, options_move)

                if options_move and (is_correct or options_move in good_moves):
                    cube.apply(options_move)
                    if is_correct:
                        correct_steps += 1
                else:
                    if options_move and made_progress:
                        cube.apply(options_move)
                    else:
                        self.first_error_step.append(step_i + 1)
                        if self.verbose:
                            logger.info(f"[sample {idx}] First error at step {step_i + 1}")
                        break

            all_sample_logs.append(sample_log)
            solve_depths.append(correct_steps)

        # -------- Aggregate --------
        avg_depth = (sum(solve_depths) / len(solve_depths)) if solve_depths else 0.0
        perfect = sum(1 for d in solve_depths if d == self.n_moves)
        step_acc = [c / t if t else 0.0 for c, t in zip(self.per_step_correct, self.per_step_totals)]
        first_err_hist = Counter(self.first_error_step)
        avg_latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0

        answered = n_correct + n_wrong
        coverage_overall = (answered / total_decisions) if total_decisions else 0.0
        selective_acc = (n_correct / answered) if answered else 0.0
        apa = ((n_correct + self.idk_weight * n_idk) / total_decisions) if total_decisions else 0.0
        coverage_by_step = [((t - z) / t) if t else 0.0 for t, z in zip(self.per_step_totals, self.per_step_idk)]

        logger.info(f"Average Correct Steps (teacher-adherence): {avg_depth:.2f} / {self.n_moves}")
        logger.info(f"Perfect Solves: {perfect}/{len(solve_depths)} "
                    f"({(perfect/len(solve_depths))*100:.2f}%)" if solve_depths else "Perfect Solves: 0/0")
        logger.info("Per-step accuracy: %s | Per-step Ns: %s",
                    [round(x, 3) for x in step_acc], [int(t) for t in self.per_step_totals])
        logger.info(f"Avg latency: {avg_latency*1000:.1f} ms")
        logger.info(
            "Selective metrics — coverage=%.3f, selective_acc=%.3f, IDK=%d, APA(%.2f)=%.3f",
            coverage_overall, selective_acc, n_idk, self.idk_weight, apa,
        )

        self.save({
            "n_moves_scrambled": self.n_moves,
            "average_solve_depth": avg_depth,
            "perfect_solves_ratio": perfect / max(1, len(solve_depths)),
            "step_accuracy": step_acc,
            "first_error_hist": dict(first_err_hist),
            "confusion_matrix": {k: dict(v) for k, v in self.confusion.items()},
            "avg_latency_ms": avg_latency * 1000,
            "num_samples": len(solve_depths),
            "selective": {
                "coverage_overall": coverage_overall,
                "coverage_by_step": coverage_by_step,
                "selective_accuracy": selective_acc,
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "n_idk": n_idk,
                "total_decisions": total_decisions,
            },
            "abstention": {
                "enabled": self.idk_enabled,
                "policy": self.idk_policy,
                "idk_weight": self.idk_weight,
                "apa": apa,
                "conf_threshold": self.idk_conf_threshold,
            },
        })
