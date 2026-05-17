from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from ..core import BaseTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class SolveMovesTest(BaseTest):
    """Dynamic MCQ using VirtualCube: generate states on-the-fly (1 move from solved)."""

    test_type = "prediction"

    def __init__(self, assistant, config, prompt_type: str = "mixed", n_moves: int = 1, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self.prompt_type = prompt_type

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        if self.n_moves != 1:
            raise ValueError(
                "SolveMovesTest currently supports n_moves==1. "
                "For >1, integrate a teacher solver (e.g., Kociemba) to get the gold first move."
            )

        cube = VirtualCube()
        scramble = cube.scramble(random_seed=idx, n_moves=1)
        teacher_move = self.teacher_first_move(scramble)

        rng = random.Random(idx)
        forced = "ABCD"[idx % 4]
        options, gold_letter = self.gen_mcq(teacher_move, rng, force_letter=forced)

        return {
            "id": idx,
            "image": cube.to_image() if self.prompt_type in ("image", "mixed") else None,
            "text_state": self.state_text(cube),
            "options": options,
            "correct_letter": gold_letter,
            "correct_move": teacher_move,
            "scramble": str(scramble),
        }

    def _build_prompts(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        kwargs = {
            "move_A": sample["options"]["A"],
            "move_B": sample["options"]["B"],
            "move_C": sample["options"]["C"],
            "move_D": sample["options"]["D"],
            "textual_representation": sample["text_state"] if self.prompt_type != "image" else "",
            "metric": "HTM (Half-Turn Metric)",
        }
        return PromptFactory.get("prediction", prompt_type=self.prompt_type, **kwargs)

    def run(self, num_samples: int) -> Tuple[List[Tuple[str, int]], List[int]]:
        wrong_pairs: List[Tuple[str, int]] = []
        acc_bits: List[int] = []
        parsed = 0

        for i in tqdm(range(num_samples), desc="Solve move test"):
            sample = self._build_sample(i)
            sys_prompt, user_prompt = self._build_prompts(sample)

            resp = self.ask(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                image=sample["image"],
                temperature=0.1,
            )

            pred_letter = self.parse_letter(resp, sample["options"])
            ok = int(pred_letter == sample["correct_letter"])
            acc_bits.append(ok)
            if pred_letter:
                parsed += 1

            logger.info(f"\nModel Predicted: {pred_letter}\nOk : {ok}")

            if not ok:
                wrong_pairs.append((pred_letter, sample["id"]))
                if self.verbose:
                    logger.info(
                        "Wrong #%d: pred=%s gold=%s options=%s scramble=%s",
                        sample["id"], pred_letter, sample["correct_letter"],
                        sample["options"], sample["scramble"],
                    )

            if self.verbose and (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d (acc=%.3f)", i + 1, num_samples, sum(acc_bits) / len(acc_bits))

        avg_acc = (sum(acc_bits) / len(acc_bits)) if acc_bits else 0.0
        logger.info("SolveMoves avg accuracy (%s): %.3f", self.prompt_type, avg_acc)
        logger.info(f"Parsed rate: {(parsed / num_samples) * 100 if num_samples else 0.0}")

        self.save(
            {
                "prompt_type": self.prompt_type,
                "average_accuracy": avg_acc,
                "num_samples": num_samples,
                "meta": {
                    "n_moves": self.n_moves,
                    "generator": "VirtualCube",
                    "prompt_source": "PromptFactory",
                },
            },
            filename=f"solve_moves_{self.prompt_type}.json",
        )

        return wrong_pairs, acc_bits
