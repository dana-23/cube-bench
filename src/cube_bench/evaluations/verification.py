"""Verification task: cross-modal Yes/No agreement between image and text state."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from cube_bench.core import BaseTest
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

    # (polarity, states_match, expected answer). Cycling these by item index
    _CELLS: Tuple[Tuple[str, bool, str], ...] = (
        ("affirmative", True, "Yes"),
        ("affirmative", False, "No"),
        ("negated", True, "No"),
        ("negated", False, "Yes"),
    )

    def __init__(self, assistant, config, n_moves: int = 3, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self._claims: Dict[str, List[str]] = PromptFactory.get_section("verification", "claims")
        missing = [p for p, _, _ in self._CELLS if p not in self._claims]
        if missing:
            raise ValueError(f"prompts.yaml verification.claims is missing polarity {missing}")
        widths = {p: len(forms) for p, forms in self._claims.items()}
        if len(set(widths.values())) != 1:
            raise ValueError(f"Each polarity needs the same number of surface forms, got {widths}")

    def _front_text(self, cube: VirtualCube) -> str:
        try:
            return cube.front_face()
        except Exception:
            try:
                return cube.observe("text")
            except Exception:
                return "<<<unavailable>>>"

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        polarity, states_match, expected = self._CELLS[idx % len(self._CELLS)]
        forms = self._claims[polarity]
        form_idx = (idx // len(self._CELLS)) % len(forms)

        text_cube = VirtualCube()
        text_cube.scramble(random_seed=idx, n_moves=self.n_moves, exact_depth=True)
        front_text = self._front_text(text_cube)

        if states_match:
            img_cube = text_cube
            mv = None
        else:
            img_cube = text_cube.clone()
            mv = self.item_rng("verification", self.n_moves, idx).choice(self._FRONT_AFFECTING)
            img_cube.apply(mv)

        return {
            "index": idx,
            "front_text": front_text,
            "image": img_cube.to_image(),
            "expected": expected,
            "mismatch_move": mv,
            "polarity": polarity,
            "states_match": states_match,
            "template_id": f"{polarity}:{form_idx}",
            "claim": forms[form_idx].format(front_face=front_text),
        }

    def run(self, num_samples: int) -> Tuple[List[int], float]:
        sys_prompt, user_tpl = PromptFactory.get("verification")

        accuracies: List[int] = []
        parsed = 0
        yes_preds = 0
        tp = tn = fp = fn = 0

        by_polarity: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0, "tp": 0, "tn": 0, "pos": 0, "neg": 0}
        )
        by_template: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0, "Yes": 0, "No": 0}
        )

        for i in tqdm(range(num_samples), desc="Verification Test"):
            sample = self._build_sample(i)
            user_prompt = user_tpl.format(claim=sample["claim"])

            resp = self.ask(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                image=sample["image"],
                max_new_tokens=2**14,
            )

            pred = self.parse_yes_no(resp)
            ok = int(pred is not None and pred.lower() == sample["expected"].lower())
            accuracies.append(ok)

            exp_yes = sample["expected"].lower() == "yes"
            pol = by_polarity[sample["polarity"]]
            tpl_stats = by_template[sample["template_id"]]
            pol["total"] += 1
            pol["pos" if exp_yes else "neg"] += 1
            pol["correct"] += ok
            tpl_stats["total"] += 1
            tpl_stats["correct"] += ok
            tpl_stats[sample["expected"]] += 1

            if pred is not None:
                parsed += 1
                pred_yes = pred.lower() == "yes"
                if pred_yes:
                    yes_preds += 1

                if exp_yes and pred_yes:
                    tp += 1
                    pol["tp"] += 1
                elif exp_yes and not pred_yes:
                    fn += 1
                elif (not exp_yes) and (not pred_yes):
                    tn += 1
                    pol["tn"] += 1
                else:
                    fp += 1

            if self.verbose:
                logger.info(
                    "Sample %s [%s, match=%s], Expected: %s, Model prediction: %s",
                    sample['index'], sample['polarity'], sample['states_match'],
                    sample['expected'], pred,
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

        polarity_metrics = {
            name: {
                "n": s["total"],
                "accuracy": (s["correct"] / s["total"]) if s["total"] else 0.0,
                "balanced_accuracy": 0.5 * (
                    ((s["tp"] / s["pos"]) if s["pos"] else 0.0)
                    + ((s["tn"] / s["neg"]) if s["neg"] else 0.0)
                ),
                "yes_label_share": (s["pos"] / s["total"]) if s["total"] else 0.0,
            }
            for name, s in by_polarity.items()
        }
        template_metrics = {
            name: {
                "n": s["total"],
                "accuracy": (s["correct"] / s["total"]) if s["total"] else 0.0,
                "yes_label_share": (s["Yes"] / s["total"]) if s["total"] else 0.0,
            }
            for name, s in by_template.items()
        }
        max_label_skew = max(
            (abs(m["yes_label_share"] - 0.5) for m in template_metrics.values()),
            default=0.0,
        )

        logger.info(
            "Verification metrics: acc=%.3f, bal_acc=%.3f, parse_rate=%.3f, yes_rate=%.3f, "
            "TP=%d TN=%d FP=%d FN=%d, unparsed=%d",
            avg_acc, bal_acc, parse_rate, yes_rate, tp, tn, fp, fn, total - parsed,
        )
        for name, m in sorted(polarity_metrics.items()):
            logger.info(
                "Polarity %-12s n=%-4d acc=%.3f bal_acc=%.3f yes_labels=%.3f",
                name, m["n"], m["accuracy"], m["balanced_accuracy"], m["yes_label_share"],
            )
        logger.info("Max per-template label skew from 50/50: %.3f (target < 0.05)", max_label_skew)

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
                "by_polarity": polarity_metrics,
                "by_template": template_metrics,
                "max_template_label_skew": max_label_skew,
            },
            "meta": {
                "generator": "VirtualCube",
                "scramble_depth": self.n_moves,
                "front_affecting_mismatch": True,
                "polarity_reversals": True,
                "surface_forms_per_polarity": len(next(iter(self._claims.values()))),
            },
        })

        return accuracies, avg_acc
