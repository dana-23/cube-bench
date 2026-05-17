from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

from tqdm import tqdm

from ..core import BaseTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class VerificationTest(BaseTest):
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

    def __init__(self, assistant, config, n_moves: int = 3, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self._sys_rng = random.SystemRandom()

    def _front_text(self, cube: VirtualCube) -> str:
        try:
            return cube.front_face()
        except Exception:
            try:
                return cube.observe("text")
            except Exception:
                return "<<<unavailable>>>"

    def _build_sample(self, idx: int) -> Dict:
        text_cube = VirtualCube()
        text_cube.scramble(random_seed=idx, n_moves=self.n_moves)
        front_text = self._front_text(text_cube)

        matched = (idx % 2 == 0)
        if matched:
            img_cube = text_cube
            expected = "Yes"
            mv = None
        else:
            img_cube = text_cube.clone()
            mv = self._sys_rng.choice(self._FRONT_AFFECTING)
            img_cube.apply(mv)
            expected = "No"

        return {
            "index": idx,
            "front_text": front_text,
            "image": img_cube.to_image(),
            "expected": expected,
            "mismatch_move": mv,
        }

    def run(self, num_samples: int) -> Tuple[List[int], float]:
        sys_prompt, user_tpl = PromptFactory.get("verification")

        accuracies: List[int] = []
        parsed = 0
        yes_preds = 0
        tp = tn = fp = fn = 0

        for i in tqdm(range(num_samples), desc="Verification Test"):
            sample = self._build_sample(i)
            user_prompt = user_tpl.format(front_face=sample["front_text"])

            resp = self.ask(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                image=sample["image"],
                max_new_tokens=2**14,
            )

            pred = self.parse_yes_no(resp)
            ok = int(pred is not None and pred.lower() == sample["expected"].lower())
            accuracies.append(ok)

            if pred is not None:
                parsed += 1
                if pred.lower() == "yes":
                    yes_preds += 1

                exp_yes = (sample["expected"].lower() == "yes")
                pred_yes = (pred.lower() == "yes")

                if exp_yes and pred_yes:
                    tp += 1
                elif exp_yes and not pred_yes:
                    fn += 1
                elif (not exp_yes) and (not pred_yes):
                    tn += 1
                else:
                    fp += 1

            if self.verbose:
                logger.info(
                    f"Sample {sample['index']}, Expected: {sample['expected']}, "
                    f"Model prediction: {pred}"
                )

        total = num_samples if num_samples else 1
        avg_acc = (sum(accuracies) / total) if accuracies else 0.0

        parse_rate = parsed / total
        yes_rate = (yes_preds / parsed) if parsed else 0.0

        pos = tp + fn
        neg = tn + fp
        tpr = (tp / pos) if pos else 0.0
        tnr = (tn / neg) if neg else 0.0
        bal_acc = 0.5 * (tpr + tnr) if (pos or neg) else 0.0

        logger.info(
            "Verification metrics: acc=%.3f, bal_acc=%.3f, parse_rate=%.3f, yes_rate=%.3f, "
            "TP=%d TN=%d FP=%d FN=%d, unparsed=%d",
            avg_acc, bal_acc, parse_rate, yes_rate, tp, tn, fp, fn, total - parsed,
        )

        self.save({
            "average_accuracy": avg_acc,
            "num_samples": num_samples,
            "metrics": {
                "accuracy": avg_acc,
                "balanced_accuracy": bal_acc,
                "parse_rate": parse_rate,
                "parse_violation": 1.0 - parse_rate,
                "yes_rate": yes_rate,
                "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                "unparsed": total - parsed,
                "support": {"pos": pos, "neg": neg},
            },
            "meta": {
                "generator": "VirtualCube",
                "scramble_depth": self.n_moves,
                "front_affecting_mismatch": True,
            },
        })

        return accuracies, avg_acc
