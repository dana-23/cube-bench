"""Prediction task: single-step multiple-choice move selection from a scramble."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from cube_bench.core import scoring
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
        self._seen = 0
        self._correct = 0

    def desc(self) -> str:
        return "Solve move test"

    def setup(self, num_samples: int) -> None:
        self._num_samples = num_samples
        self._seen = 0
        self._correct = 0

    # ----- Item generation -----

    def _item_core(self, idx: int) -> Tuple[VirtualCube, Any, Dict[str, str], str, str]:
        """Everything an item is made of except its rendering, which is the costly half."""
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
        return cube, scramble, options, gold_letter, teacher_move

    def correct_letter(self, idx: int) -> str:
        """The gold letter for an item, without paying to render it."""
        return self._item_core(idx)[3]

    def build_item(self, idx: int) -> Dict[str, Any]:
        cube, scramble, options, gold_letter, teacher_move = self._item_core(idx)

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
                "response": response,
            },
            extra={"options": item["options"]},
        )

    def accumulate(self, record: ItemRecord) -> None:
        ok = record.saved["ok"]
        self._seen += 1
        self._correct += ok

        if not ok:
            self._vlog(
                "Wrong #%d: pred=%s gold=%s options=%s scramble=%s",
                record.index, record.pred, record.gold,
                record.extra["options"], record.saved["scramble"],
            )

        if self.verbose and self._seen % 10 == 0:
            self._vlog(
                "Progress: %d/%d (acc=%.3f)",
                self._seen, self._num_samples, self._correct / self._seen,
            )

    # ----- Aggregation -----

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        scored = scoring.score(self.test_type, records)
        return {
            "prompt_type": self.prompt_type,
            "average_accuracy": scored["average_accuracy"],
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
        parsed = sum(1 for record in records if record.pred)
        self._logger.info(
            "Parsed rate: %s", (parsed / num_samples) * 100 if num_samples else 0.0
        )

    # ----- Results -----

    def result_filename(self) -> str:
        return f"solve_moves_{self.prompt_type}.json"

    def result(self, payload: Dict[str, Any], records: List[ItemRecord]):
        wrong_pairs = [(r.pred, r.index) for r in records if not r.saved["ok"]]
        acc_bits = [r.saved["ok"] for r in records]
        return wrong_pairs, acc_bits, [r.pred for r in records]
