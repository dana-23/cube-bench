"""Verification task: cross-modal Yes/No agreement between image and text state."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from cube_bench.core import ItemRecord, SingleAskTest
from cube_bench.core.metrics import balanced_accuracy
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

    # (polarity, states_match, expected answer). Cycling these by item index
    _CELLS: Tuple[Tuple[str, bool, str], ...] = (
        ("affirmative", True, "Yes"),
        ("affirmative", False, "No"),
        ("negated", True, "No"),
        ("negated", False, "Yes"),
    )

    # ----- Construction -----

    def __init__(self, assistant, config, n_moves: int = 3, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self._claims: Dict[str, List[str]] = PromptFactory.get_section("verification", "claims")
        missing = [p for p, _, _ in self._CELLS if p not in self._claims]
        if missing:
            raise ValueError(f"prompts.yaml verification.claims is missing polarity {missing}")
        widths = {p: len(forms) for p, forms in self._claims.items()}
        if len(set(widths.values())) != 1:
            raise ValueError(f"Each polarity needs the same number of surface forms, got {widths}")

        self._sys_prompt = ""
        self._user_template = ""
        self._parsed = 0
        self._yes_preds = 0
        self._tp = self._tn = self._fp = self._fn = 0
        self._by_polarity: Dict[str, Dict[str, int]] = {}
        self._by_template: Dict[str, Dict[str, int]] = {}

    def desc(self) -> str:
        return "Verification Test"

    def setup(self, num_samples: int) -> None:
        self._sys_prompt, self._user_template = PromptFactory.get("verification")
        self._parsed = 0
        self._yes_preds = 0
        self._tp = self._tn = self._fp = self._fn = 0
        self._by_polarity = defaultdict(
            lambda: {"correct": 0, "total": 0, "tp": 0, "tn": 0, "pos": 0, "neg": 0}
        )
        self._by_template = defaultdict(
            lambda: {"correct": 0, "total": 0, "Yes": 0, "No": 0}
        )

    # ----- Item generation -----

    def build_item(self, idx: int) -> Dict[str, Any]:
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
        return self._sys_prompt, self._user_template.format(claim=item["claim"])

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
            "Sample %s [%s, match=%s], Expected: %s, Model prediction: %s",
            item["index"], item["polarity"], item["states_match"],
            item["expected"], prediction,
        )
        return ItemRecord(
            index=item["index"],
            gold=item["expected"],
            pred=prediction,
            correct=bool(ok),
            parsed=prediction is not None,
            response=response,
            extra={
                "ok": ok,
                "polarity": item["polarity"],
                "template_id": item["template_id"],
                "expected": item["expected"],
            },
        )

    def accumulate(self, record: ItemRecord) -> None:
        ok = record.extra["ok"]
        exp_yes = record.extra["expected"].lower() == "yes"
        pol = self._by_polarity[record.extra["polarity"]]
        tpl_stats = self._by_template[record.extra["template_id"]]
        pol["total"] += 1
        pol["pos" if exp_yes else "neg"] += 1
        pol["correct"] += ok
        tpl_stats["total"] += 1
        tpl_stats["correct"] += ok
        tpl_stats[record.extra["expected"]] += 1

        if record.pred is not None:
            self._parsed += 1
            pred_yes = record.pred.lower() == "yes"
            if pred_yes:
                self._yes_preds += 1

            if exp_yes and pred_yes:
                self._tp += 1
                pol["tp"] += 1
            elif exp_yes and not pred_yes:
                self._fn += 1
            elif (not exp_yes) and (not pred_yes):
                self._tn += 1
                pol["tn"] += 1
            else:
                self._fp += 1

    # ----- Aggregation -----

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        accuracies = [record.extra["ok"] for record in records]
        total = num_samples if num_samples else 1
        avg_acc = (sum(accuracies) / total) if accuracies else 0.0

        parse_rate = self._parsed / total
        yes_rate = (self._yes_preds / self._parsed) if self._parsed else 0.0

        pos = self._tp + self._fn
        neg = self._tn + self._fp
        bal_acc = balanced_accuracy(self._tp, self._tn, self._fp, self._fn)

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
            for name, s in self._by_polarity.items()
        }
        template_metrics = {
            name: {
                "n": s["total"],
                "accuracy": (s["correct"] / s["total"]) if s["total"] else 0.0,
                "yes_label_share": (s["Yes"] / s["total"]) if s["total"] else 0.0,
            }
            for name, s in self._by_template.items()
        }
        max_label_skew = max(
            (abs(m["yes_label_share"] - 0.5) for m in template_metrics.values()),
            default=0.0,
        )

        return {
            "average_accuracy": avg_acc,
            "num_samples": num_samples,
            "metrics": {
                "accuracy": avg_acc,
                "balanced_accuracy": bal_acc,
                "parse_rate": parse_rate,
                "parse_violation": 1.0 - parse_rate,
                "yes_rate": yes_rate,
                "confusion": {"tp": self._tp, "tn": self._tn, "fp": self._fp, "fn": self._fn},
                "unparsed": total - self._parsed,
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
        for name, m in sorted(metrics["by_polarity"].items()):
            self._logger.info(
                "Polarity %-12s n=%-4d acc=%.3f bal_acc=%.3f yes_labels=%.3f",
                name, m["n"], m["accuracy"], m["balanced_accuracy"], m["yes_label_share"],
            )
        self._logger.info(
            "Max per-template label skew from 50/50: %.3f (target < 0.05)",
            metrics["max_template_label_skew"],
        )

    # ----- Results -----

    def result(
        self, payload: Dict[str, Any], records: List[ItemRecord]
    ) -> Tuple[List[int], float]:
        return [record.extra["ok"] for record in records], payload["average_accuracy"]
