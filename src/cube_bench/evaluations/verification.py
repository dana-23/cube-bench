"""Verification task: cross-modal Yes/No agreement between image and text state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from cube_bench.core import ItemRecord, SingleAskTest
from cube_bench.core import scoring
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube


class VerificationTest(SingleAskTest):
    """Cross-modal Yes/No consistency using VirtualCube (no datasets)."""

    test_type = "verification"

    # Moves guaranteed to change the Front face (avoid B/B'/B2 only).
    _FRONT_AFFECTING = (
        "F", "F'", "F2",
        "U", "U'", "U2",
        "D", "D'", "D2",
        "L", "L'", "L2",
        "R", "R'", "R2",
    )

    # ----- Construction -----

    def __init__(self, assistant, config, n_moves: int = 3, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self._sys_prompt = ""
        self._user_template = ""

    def desc(self) -> str:
        return "Verification Test"

    def setup(self, num_samples: int) -> None:
        self._sys_prompt, self._user_template = PromptFactory.get("verification")

    # ----- Item generation -----

    def build_item(self, idx: int) -> Dict[str, Any]:
        text_cube = VirtualCube()
        text_cube.scramble(random_seed=idx, n_moves=self.n_moves, exact_depth=True)
        front_text = self._front_text(text_cube)

        matched = idx % 2 == 0
        if matched:
            img_cube = text_cube
            expected = "Yes"
            mv = None
        else:
            img_cube = text_cube.clone()
            mv = self.item_rng("verification", self.n_moves, idx).choice(self._FRONT_AFFECTING)
            img_cube.apply(mv)
            expected = "No"

        return {
            "index": idx,
            "front_text": front_text,
            "image": img_cube.to_image(),
            "expected": expected,
            "mismatch_move": mv,
        }

    def _front_text(self, cube: VirtualCube) -> str:
        try:
            return cube.front_face()
        except Exception:
            try:
                return cube.observe("text")
            except Exception:
                return "<<<unavailable>>>"

    # ----- Prompting -----

    def build_prompts(self, item: Dict[str, Any]) -> Tuple[str, str]:
        return self._sys_prompt, self._user_template.format(front_face=item["front_text"])

    def ask_kwargs(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"image": item["image"], "max_new_tokens": 2 ** 14}

    # ----- Parsing -----

    def parse(self, response: Optional[str]) -> Optional[str]:
        return self.parse_yes_no(response)

    # ----- Scoring -----

    def score_item(
        self, item: Dict[str, Any], prediction: Optional[str], response: Optional[str]
    ) -> ItemRecord:
        ok = int(prediction is not None and prediction.lower() == item["expected"].lower())
        self._vlog(
            "Sample %s, Expected: %s, Model prediction: %s",
            item["index"], item["expected"], prediction,
        )
        return ItemRecord(
            index=item["index"],
            gold=item["expected"],
            pred=prediction,
            correct=bool(ok),
            parsed=prediction is not None,
            response=response,
            saved={
                "index": item["index"],
                "gold": item["expected"],
                "pred": prediction,
                "ok": ok,
                "response": response,
            },
            extra={"ok": ok, "expected": item["expected"]},
        )

    # ----- Aggregation -----

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        scored = scoring.score(self.test_type, records, total=num_samples or 1)
        return {
            "average_accuracy": scored["accuracy"],
            "num_samples": num_samples,
            "metrics": scored,
            "per_item": [record.saved for record in records],
            "meta": {
                "generator": "VirtualCube",
                "scramble_depth": self.n_moves,
                "front_affecting_mismatch": True,
                "seeded_mismatch": True,
            },
        }

    # ----- Reporting -----

    def summary(self, payload: Dict[str, Any], records: List[ItemRecord]) -> None:
        metrics = payload["metrics"]
        confusion = metrics["confusion"]
        self._logger.info(
            "Verification metrics: acc=%.3f, bal_acc=%.3f, parse_rate=%.3f, yes_rate=%.3f, "
            "TP=%d TN=%d FP=%d FN=%d, unparsed=%d",
            metrics["accuracy"], metrics["balanced_accuracy"], metrics["parse_rate"],
            metrics["yes_rate"], confusion["tp"], confusion["tn"], confusion["fp"],
            confusion["fn"], metrics["unparsed"],
        )

    # ----- Results -----

    def result(
        self, payload: Dict[str, Any], records: List[ItemRecord]
    ) -> Tuple[List[int], float]:
        return [record.extra["ok"] for record in records], payload["average_accuracy"]
