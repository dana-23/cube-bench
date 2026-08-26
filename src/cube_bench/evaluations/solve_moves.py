"""Prediction task: single-step multiple-choice move selection from a scramble."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from cube_bench.core import ItemRecord, SingleAskTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube


class SolveMovesTest(SingleAskTest):
    """Dynamic MCQ using VirtualCube: generate states on-the-fly (1 move from solved)."""

    test_type = "prediction"

    # ----- Construction -----

    def __init__(self, assistant, config, prompt_type: str = "mixed", n_moves: int = 1,
                 verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self.prompt_type = prompt_type
        self._num_samples = 0
        self._parsed = 0
        self._acc_bits: List[int] = []
        self._preds: List[Optional[str]] = []
        self._wrong_pairs: List[Tuple[str, int]] = []

    def desc(self) -> str:
        return "Solve move test"

    def setup(self, num_samples: int) -> None:
        self._num_samples = num_samples
        self._parsed = 0
        self._acc_bits = []
        self._preds = []
        self._wrong_pairs = []

    # ----- Item generation -----

    def build_item(self, idx: int) -> Dict[str, Any]:
        cube = VirtualCube()
        scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves, exact_depth=True)

        if self.n_moves == 1:
            teacher_move = self.teacher_first_move(scramble)
        else:
            optimal_path = cube.solve().split()
            if not optimal_path:
                raise RuntimeError(f"Optimal solver returned empty path for sample {idx}")
            teacher_move = optimal_path[0]

        rng = random.Random(idx)
        forced = "ABCD"[idx % 4]
        options, gold_letter = self.gen_mcq(teacher_move, rng, force_letter=forced)

        return {
            "id": idx,
            "image": cube.to_image() if self.prompt_type in (
                "image", "mixed", "mixed_no_authority") else None,
            "text_state": self.state_text(cube),
            "options": options,
            "correct_letter": gold_letter,
            "correct_move": teacher_move,
            "scramble": str(scramble),
        }

    # ----- Prompting -----

    def build_prompts(self, item: Dict[str, Any]) -> Tuple[str, str]:
        kwargs = {
            "move_A": item["options"]["A"],
            "move_B": item["options"]["B"],
            "move_C": item["options"]["C"],
            "move_D": item["options"]["D"],
            "textual_representation": item["text_state"] if self.prompt_type != "image" else "",
            "metric": "HTM (Half-Turn Metric)",
        }
        return PromptFactory.get("prediction", prompt_type=self.prompt_type, **kwargs)

    # ----- Scoring -----

    def score_item(
        self, item: Dict[str, Any], prediction: Optional[str], response: Optional[str]
    ) -> ItemRecord:
        ok = int(prediction == item["correct_letter"])
        self._logger.info("\nModel Predicted: %s\nOk : %s", prediction, ok)
        return ItemRecord(
            index=item["id"],
            gold=item["correct_letter"],
            pred=prediction,
            correct=bool(ok),
            parsed=bool(prediction),
            response=response,
            saved={
                "id": item["id"],
                "pred": prediction,
                "gold": item["correct_letter"],
                "correct_move": item["correct_move"],
                "options": item["options"],
                "scramble": item["scramble"],
                "ok": ok,
            },
            extra={"options": item["options"]},
        )

    def accumulate(self, record: ItemRecord) -> None:
        ok = record.saved["ok"]
        self._acc_bits.append(ok)
        self._preds.append(record.pred)
        if record.pred:
            self._parsed += 1

        if not ok:
            self._wrong_pairs.append((record.pred, record.index))
            self._vlog(
                "Wrong #%d: pred=%s gold=%s options=%s scramble=%s",
                record.index, record.pred, record.gold,
                record.extra["options"], record.saved["scramble"],
            )

        done = len(self._acc_bits)
        if self.verbose and done % 10 == 0:
            self._vlog(
                "Progress: %d/%d (acc=%.3f)",
                done, self._num_samples, sum(self._acc_bits) / done,
            )

    # ----- Aggregation -----

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        avg_acc = (sum(self._acc_bits) / len(self._acc_bits)) if self._acc_bits else 0.0
        return {
            "prompt_type": self.prompt_type,
            "average_accuracy": avg_acc,
            "num_samples": num_samples,
            "per_item": [record.saved for record in records],
            "meta": {
                "n_moves": self.n_moves,
                "generator": "VirtualCube",
                "prompt_source": "PromptFactory",
            },
        }

    # ----- Reporting -----

    def summary(self, payload: Dict[str, Any], records: List[ItemRecord]) -> None:
        num_samples = payload["num_samples"]
        self._logger.info(
            "SolveMoves avg accuracy (%s): %.3f", self.prompt_type, payload["average_accuracy"]
        )
        self._logger.info(
            "Parsed rate: %s", (self._parsed / num_samples) * 100 if num_samples else 0.0
        )

    # ----- Results -----

    def result_filename(self) -> str:
        return f"solve_moves_{self.prompt_type}.json"

    def result(self, payload: Dict[str, Any], records: List[ItemRecord]):
        return self._wrong_pairs, self._acc_bits, self._preds
