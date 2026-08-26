"""Move-effect task: label each option DECREASE / NO_CHANGE / INCREASE."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from cube_bench.core import ItemRecord, SingleAskTest
from cube_bench.core.metrics import (
    cohens_kappa, dot, jensen_shannon, macro_f1, per_class_prf, safe_prop,
)
from cube_bench.sim.cube_simulator import VirtualCube


class MoveEffectTest(SingleAskTest):
    """Label each move's distance effect with depth- and slot-balanced sampling."""
    TAG_RE = re.compile(r"<([ABCD])>\s*(DECREASE|NO[_ ]?CHANGE|INCREASE)\s*</\1>", re.IGNORECASE)
    # Fallback for replies that use "A: DECREASE" instead of the tagged form.
    LETTER_RES = {
        L: re.compile(rf"{L}\s*:\s*(DECREASE|NO[_ ]?CHANGE|INCREASE)", re.IGNORECASE)
        for L in "ABCD"
    }
    CLASSES = ("DECREASE", "NO_CHANGE", "INCREASE")
    SLOTS = "ABCD"
    test_type = "move_effect"

    # ----- Construction -----

    def __init__(self, assistant, config, n_moves: int = 2, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)

        # Fairness state.
        self.double_cycle = ["INCREASE", "NO_CHANGE", "DECREASE"]
        self.double_idx = 0
        self.double_slot_cycle = deque(self.SLOTS)

        self.presented_counts = Counter({c: 0 for c in self.CLASSES})
        self.per_slot_counts = {s: Counter({c: 0 for c in self.CLASSES}) for s in self.SLOTS}
        self.class_debt = Counter()
        self.target_double_counts = Counter()
        self.target_double_success = Counter()
        self.missing_class_counts = Counter()
        self.composition_counts = Counter()

        self.depth_item_count = Counter()
        self.depth_presented_counts = defaultdict(Counter)
        self.depth_feasible2_counts = defaultdict(lambda: Counter({c: 0 for c in self.CLASSES}))
        self.alpha_smooth = 1.0

        self._micro_correct = 0
        self._total_labels = 0
        self._confusion: Dict[str, Counter] = defaultdict(Counter)
        self._per_class: Counter = Counter()
        self._option_mix_ok: Counter = Counter()
        self._per_distance: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )

    def desc(self) -> str:
        return f"Move-Effect (n_moves={self.n_moves})"

    def setup(self, num_samples: int) -> None:
        self._micro_correct = 0
        self._total_labels = 0
        self._confusion = defaultdict(Counter)
        self._per_class = Counter()
        self._option_mix_ok = Counter()
        self._per_distance = defaultdict(lambda: {"correct": 0, "total": 0})

        self._logger.info("=" * 80)
        self._logger.info("Initializing Move-Effect test on %s", self.assistant.get_name())
        self._logger.info(
            "Number of samples: %s | Scramble depth: %s", num_samples, self.n_moves
        )
        self._logger.info("=" * 80)

    def _label_all_neighbors(self, vc: VirtualCube) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        buckets: Dict[str, List[str]] = {"DECREASE": [], "NO_CHANGE": [], "INCREASE": []}
        labels_by_move: Dict[str, str] = {}
        old_distance = vc.get_distance()

        for m in VirtualCube.AVAILABLE_MOVES:
            c = vc.clone()
            c.apply(m)
            new_distance = c.get_distance()
            delta = new_distance - old_distance
            if new_distance == 0 or delta < 0:
                lbl = "DECREASE"
            elif delta == 0:
                lbl = "NO_CHANGE"
            else:
                lbl = "INCREASE"
            buckets[lbl].append(m)
            labels_by_move[m] = lbl

        return buckets, labels_by_move

    def _feasible_target_for_depth(self, d: int) -> Dict[str, float]:
        """Allocate the fourth option by each class's smoothed feasibility at depth ``d``."""
        items = self.depth_item_count[d]
        p2 = {}
        # Laplace smoothing prevents unobserved depths from dominating the target.
        for c in self.CLASSES:
            succ = self.depth_feasible2_counts[d][c]
            p2[c] = (succ + self.alpha_smooth) / (items + 2 * self.alpha_smooth) if items >= 0 else 1/2
        z = sum(p2.values()) or 1.0
        extra_share = {c: (p2[c] / z) * 0.25 for c in self.CLASSES}
        target = {c: 0.25 + extra_share[c] for c in self.CLASSES}
        s = sum(target.values()) or 1.0
        target = {k: v / s for k, v in target.items()}
        return target

    def _overall_feasible_target(self) -> Dict[str, float]:
        """Weighted average of per-depth feasible targets, weighted by items at each depth."""
        total_items = sum(self.depth_item_count.values()) or 1
        mix = {c: 0.0 for c in self.CLASSES}
        for d, n in self.depth_item_count.items():
            td = self._feasible_target_for_depth(d)
            for c in self.CLASSES:
                mix[c] += td[c] * (n / total_items)
        s = sum(mix.values()) or 1.0
        return {k: v / s for k, v in mix.items()}

    def _pick_target_double(self, d: int, buckets: Dict[str, List[str]]) -> str | None:
        """Double the most underrepresented feasible class at depth ``d``."""
        feasible = [c for c in self.CLASSES if len(buckets.get(c, [])) >= 2]
        if not feasible:
            return None

        target = self._feasible_target_for_depth(d)
        counts = self.depth_presented_counts[d]
        total = sum(counts.values()) or 1
        desired_counts = {c: target[c] * total for c in self.CLASSES}
        deficits = {c: desired_counts[c] - counts.get(c, 0) for c in self.CLASSES}

        best = None
        best_val = -1e9
        for c in feasible:
            val = deficits.get(c, 0.0)
            if val > best_val:
                best = c
                best_val = val
        if best is not None:
            return best

        return feasible[self.double_idx % len(feasible)]

    def _assign_with_double_slot(self, picked: List[Tuple[str, str]], double_cls: str | None) -> Dict[str, str]:
        """Rotate the doubled class across slots, then balance the remaining assignments."""
        slots = list(self.SLOTS)
        assignment: Dict[str, str] = {}

        if double_cls is not None:
            double_slot = self.double_slot_cycle[0]
            self.double_slot_cycle.rotate(-1)
            idx = next((i for i, (_, cls) in enumerate(picked) if cls == double_cls), None)
            if idx is not None:
                move, cls = picked.pop(idx)
                assignment[double_slot] = move
                self.per_slot_counts[double_slot][cls] += 1
                slots.remove(double_slot)

        cls_freq = Counter(cls for _, cls in picked)
        pending = sorted(picked, key=lambda x: cls_freq[x[1]])
        for move, cls in pending:
            slot = min(slots, key=lambda s, cls=cls: self.per_slot_counts[s][cls])
            assignment[slot] = move
            self.per_slot_counts[slot][cls] += 1
            slots.remove(slot)

        return assignment

    def _balanced_sample_ABCD(  # pylint: disable=too-many-branches,too-many-statements
        self, d: int, buckets: Dict[str, List[str]], rng: random.Random
    ) -> Tuple[Dict[str, str], Dict[str, int], str | None]:
        """Build A-D with one move per class plus a feasible doubled class when possible."""
        target_double = self._pick_target_double(d, buckets)
        if target_double is not None:
            self.target_double_counts[target_double] += 1
        else:
            self.class_debt["NO_FEASIBLE_DOUBLE"] += 1

        chosen_moves: set = set()
        picked: List[Tuple[str, str]] = []
        actual_counts = {c: 0 for c in self.CLASSES}

        def take(cls: str, k: int) -> List[str]:
            pool = [m for m in buckets.get(cls, []) if m not in chosen_moves]
            rng.shuffle(pool)
            return pool[:k]

        # Start with one option from each available class.
        for cls in self.CLASSES:
            got = take(cls, 1)
            if got:
                move = got[0]
                chosen_moves.add(move)
                picked.append((move, cls))
                actual_counts[cls] += 1

        # Add the targeted fourth option.
        doubled_class: str | None = None
        if target_double is not None:
            leftover = take(target_double, 1)
            if leftover:
                move = leftover[0]
                chosen_moves.add(move)
                picked.append((move, target_double))
                actual_counts[target_double] += 1
                doubled_class = target_double
                self.target_double_success[target_double] += 1
            else:
                # Recover if the target was not selected during the first pass.
                got2 = take(target_double, 2 - actual_counts[target_double])
                for mv in got2:
                    chosen_moves.add(mv)
                    picked.append((mv, target_double))
                actual_counts[target_double] = sum(1 for _, c in picked if c == target_double)
                if actual_counts[target_double] >= 2:
                    doubled_class = target_double
                    self.target_double_success[target_double] += 1
                else:
                    self.class_debt[target_double] += 1

        # Backfill from the most underrepresented available class.
        while len(picked) < 4:
            target = self._feasible_target_for_depth(d)
            total_here = sum(actual_counts.values()) or 1
            desired_share = target
            def share(cls):
                return actual_counts[cls] / total_here
            order = sorted(self.CLASSES, key=lambda c: desired_share[c] - share(c), reverse=True)
            filled = False
            for cls in order:
                got = take(cls, 1)
                if got:
                    move = got[0]
                    chosen_moves.add(move)
                    picked.append((move, cls))
                    actual_counts[cls] += 1
                    if doubled_class is None and actual_counts[cls] >= 2:
                        doubled_class = cls
                    filled = True
                    break
            if not filled:
                break

        picked = picked[:4]

        for cls in self.CLASSES:
            if actual_counts[cls] == 0:
                self.missing_class_counts[cls] += 1

        comp = (actual_counts["DECREASE"], actual_counts["NO_CHANGE"], actual_counts["INCREASE"])
        self.composition_counts[comp] += 1
        if target_double is not None and doubled_class != target_double:
            self.class_debt[target_double] += 1

        options = self._assign_with_double_slot(picked[:], doubled_class)
        # Preserve four slots even when a bucket unexpectedly exhausts.
        if len(options) < 4:
            remaining_slots = [s for s in self.SLOTS if s not in options]
            leftover_moves = [m for m, _ in picked if m not in options.values()]
            rng.shuffle(leftover_moves)
            for s in remaining_slots:
                if leftover_moves:
                    options[s] = leftover_moves.pop()
                else:
                    options[s] = next(iter(options.values()))

        return options, actual_counts, doubled_class

    def _face_centers(self, cube: VirtualCube) -> Dict[str, str]:
        return {
            "U_color": str(cube.raw.get_face("U")[1][1].colour),
            "R_color": str(cube.raw.get_face("R")[1][1].colour),
            "F_color": str(cube.raw.get_face("F")[1][1].colour),
            "D_color": str(cube.raw.get_face("D")[1][1].colour),
            "L_color": str(cube.raw.get_face("L")[1][1].colour),
            "B_color": str(cube.raw.get_face("B")[1][1].colour),
        }

    @staticmethod
    def _sys_prompt() -> str:
        return (
            "You are an expert Rubik's Cube evaluator. The cube is scrambled.\n"
            "For EACH option (A-D), label how it changes distance-to-solved:\n"
            "DECREASE, NO_CHANGE, INCREASE.\n\n"
            "Rules:\n- Textual cube state is ground truth; ignore image.\n"
            "- Use centers to map colors to faces.\n"
            "- Output exactly four lines <A> ... </A> ... <D> ... </D>\n"
        )

    def _user_prompt(self, centers: Dict[str, str], state_text: str, options: Dict[str, str]) -> str:
        return (
            f"**Face Centers:**\n"
            f"U: {centers['U_color']}\nR: {centers['R_color']}\nF: {centers['F_color']}\n"
            f"D: {centers['D_color']}\nL: {centers['L_color']}\nB: {centers['B_color']}\n\n"
            f"**Textual Cube State:**\n{state_text}\n\n"
            f"**Candidate Moves:**\nA: {options['A']}\nB: {options['B']}\nC: {options['C']}\nD: {options['D']}\n\n"
            "Output exactly:\n"
            "<A> DECREASE|NO_CHANGE|INCREASE </A>\n"
            "<B> DECREASE|NO_CHANGE|INCREASE </B>\n"
            "<C> DECREASE|NO_CHANGE|INCREASE </C>\n"
            "<D> DECREASE|NO_CHANGE|INCREASE </D>\n"
        )

    # ----- Item generation -----

    def build_item(self, idx: int) -> Dict[str, Any]:
        cube = VirtualCube()
        scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves, exact_depth=True)

        d = cube.get_distance()
        self.depth_item_count[d] += 1

        buckets, labels_by_move = self._label_all_neighbors(cube)

        for c in self.CLASSES:
            if len(buckets.get(c, [])) >= 2:
                self.depth_feasible2_counts[d][c] += 1

        options, _actual_counts_item, _doubled_cls = self._balanced_sample_ABCD(
            d, buckets, self.item_rng("move_effect", self.n_moves, idx)
        )

        classes_in_item = {labels_by_move[mv] for mv in options.values()}
        self._option_mix_ok[len(classes_in_item)] += 1

        truth = {k: labels_by_move[mv] for k, mv in options.items()}

        for lbl in truth.values():
            self.presented_counts[lbl] += 1
            self.depth_presented_counts[d][lbl] += 1

        return {
            "index": idx,
            "image": None,
            "distance": d,
            "scramble": scramble,
            "buckets": buckets,
            "options": options,
            "truth": truth,
            "centers": self._face_centers(cube),
            "state_text": self.state_text(cube),
        }

    # ----- Prompting -----

    def build_prompts(self, item: Dict[str, Any]) -> Tuple[str, str]:
        return (
            self._sys_prompt(),
            self._user_prompt(item["centers"], item["state_text"], item["options"]),
        )

    def ask_kwargs(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"image": None}

    # ----- Parsing -----

    def parse(self, response: Optional[str]) -> Dict[str, str]:
        text = response or ""
        preds = {m.group(1).upper(): m.group(2).upper().replace(" ", "_")
                 for m in self.TAG_RE.finditer(text)}
        for k in "ABCD":
            if k not in preds:
                pat = self.LETTER_RES[k].search(text)
                if pat:
                    preds[k] = pat.group(1).replace(" ", "_").upper()
        return preds

    # ----- Scoring -----

    def score_item(
        self, item: Dict[str, Any], prediction: Dict[str, str], response: Optional[str]
    ) -> ItemRecord:
        truth = item["truth"]
        labels = [(k, truth[k], prediction.get(k, "MISSING")) for k in "ABCD"]
        correct_this_item = sum(int(pred == gold) for _, gold, pred in labels)

        if self.verbose:
            predictions = {k: prediction.get(k) for k in 'ABCD'}
            buckets = item["buckets"]
            self._logger.info(
                "[%s] d=%s  scramble=%s", item["index"], item["distance"], item["scramble"])
            self._logger.info(
                "bucket sizes: DEC=%s, NC=%s, INC=%s",
                len(buckets['DECREASE']),
                len(buckets['NO_CHANGE']),
                len(buckets['INCREASE']),
            )
            self._logger.info("options: %s", item["options"])
            self._logger.info("truth:   %s", truth)
            self._logger.info("preds:   %s", predictions)

        return ItemRecord(
            index=item["index"],
            gold=truth,
            pred=prediction,
            correct=correct_this_item == 4,
            parsed=len(prediction) == 4,
            response=response,
            extra={
                "labels": labels,
                "correct_this_item": correct_this_item,
                "distance": item["distance"],
            },
        )

    def accumulate(self, record: ItemRecord) -> None:
        for _, gold, pred in record.extra["labels"]:
            self._per_class[gold] += 1
            self._confusion[gold][pred] += 1
            is_right = int(pred == gold)
            self._micro_correct += is_right
            self._total_labels += 1

        d = record.extra["distance"]
        self._per_distance[d]["correct"] += record.extra["correct_this_item"]
        self._per_distance[d]["total"] += 4

    # ----- Aggregation -----

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        micro_correct = self._micro_correct
        total_labels = self._total_labels
        confusion = self._confusion
        per_class = self._per_class
        option_mix_ok = self._option_mix_ok
        per_distance = self._per_distance

        # Aggregate accuracy and fairness metrics.
        tot = sum(per_class.values())
        tri = ("INCREASE", "NO_CHANGE", "DECREASE")
        priors = {k: safe_prop(per_class[k], tot) for k in tri}

        pred_totals = Counter()
        for _gold, row in confusion.items():
            for pred, c in row.items():
                pred_totals[pred] += c
        pred_tot = sum(pred_totals[k] for k in tri)
        q = {k: (pred_totals[k] / pred_tot) if pred_tot else 0.0 for k in tri}

        maj_baseline = max(priors.values()) if priors else 0.0
        prior_sample_baseline = dot(priors, priors, tri)
        model_expected = dot(priors, q, tri)

        self._logger.info("gold priors: %s", priors)
        self._logger.info("model preds q: %s", q)
        self._logger.info("baseline(always majority): %.3f  baseline(prior-sample): %.3f  exp(acc from q.priors): %.3f",
                    maj_baseline, prior_sample_baseline, model_expected)
        self._logger.info("option class coverage counts (distinct classes per item): %s", dict(option_mix_ok))

        micro_acc = micro_correct / total_labels if total_labels else 0.0

        kappa = cohens_kappa(micro_acc, model_expected)

        per_class_precision, per_class_recall, per_class_f1 = per_class_prf(confusion, tri)
        macro = macro_f1(per_class_f1, tri)

        per_distance_acc = {int(d): (v["correct"] / v["total"]) for d, v in per_distance.items() if v["total"]}

        fairness_metrics = self._fairness_metrics(priors, tri)

        out = {
            "n_moves": self.n_moves,
            "micro_acc": micro_acc,
            "macro_f1": macro,
            "kappa": kappa,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "labels_total": total_labels,
            "confusion": {g: dict(c) for g, c in confusion.items()},
            "support": dict(per_class),
            "gold_priors": priors,
            "pred_mix": q,
            "expected_dot": model_expected,
            "maj_baseline": maj_baseline,
            "prior_sample_baseline": prior_sample_baseline,
            "per_distance_micro_acc": per_distance_acc,
            "option_coverage_counts": dict(option_mix_ok),
            "num_samples": num_samples,
            "fairness_metrics": fairness_metrics,
        }
        return out

    def _fairness_metrics(self, priors: Dict[str, float], tri) -> Dict[str, Any]:
        """Sampling-fairness diagnostics for the option mix actually presented."""
        uniform = {k: 1 / 3 for k in tri}
        jsd_uniform = jensen_shannon(priors, uniform)
        max_abs_dev_uniform = max(abs(priors[k] - 1 / 3) for k in tri) if tri else 0.0

        target_mix = self._overall_feasible_target()
        jsd_target = jensen_shannon(priors, target_mix)
        max_abs_dev_target = max(abs(priors[k] - target_mix[k]) for k in tri)

        slot_priors = {s: {c: safe_prop(self.per_slot_counts[s][c], sum(self.per_slot_counts[s].values()))
                           for c in tri} for s in self.SLOTS}
        slot_jsd_uniform = {s: jensen_shannon(slot_priors[s], uniform) for s in self.SLOTS}
        slot_jsd_target = {s: jensen_shannon(slot_priors[s], target_mix) for s in self.SLOTS}

        double_success_rate = {}
        for c in self.CLASSES:
            attempts = self.target_double_counts.get(c, 0)
            succ = self.target_double_success.get(c, 0)
            double_success_rate[c] = (succ / attempts) if attempts else 0.0

        priors_by_depth = {}
        dev_by_depth = {}
        for d, cnts in self.depth_presented_counts.items():
            tot_d = sum(cnts.values())
            if tot_d:
                pd = {k: safe_prop(cnts.get(k, 0), tot_d) for k in tri}
                priors_by_depth[int(d)] = pd
                target_d = self._feasible_target_for_depth(d)
                dev_by_depth[int(d)] = {
                    "max_abs_dev_uniform": max(abs(pd[k] - 1 / 3) for k in tri),
                    "jsd_from_uniform": jensen_shannon(pd, uniform),
                    "max_abs_dev_target": max(abs(pd[k] - target_d[k]) for k in tri),
                    "jsd_from_target": jensen_shannon(pd, target_d),
                }

        within5_uniform = all(abs(priors[k] - 1 / 3) <= 0.05 for k in tri)
        within5_target = all(abs(priors[k] - target_mix[k]) <= 0.05 for k in tri)
        slots_within7_uniform = all(all(abs(slot_priors[s][k] - 1 / 3) <= 0.07 for k in tri) for s in self.SLOTS)
        slots_within7_target = all(all(abs(slot_priors[s][k] - target_mix[k]) <= 0.07 for k in tri) for s in self.SLOTS)

        self._logger.info("FAIRNESS - overall JSD(uniform)=%.4f  max_abs_dev=%.3f  within 5%%=%s",
                          jsd_uniform, max_abs_dev_uniform, within5_uniform)
        self._logger.info(
            "FAIRNESS - overall JSD(target)=%.4f  max_abs_dev=%.3f  within 5%%=%s  target=%s",
            jsd_target, max_abs_dev_target, within5_target, target_mix)
        self._logger.info("FAIRNESS - per-slot priors: %s", slot_priors)
        self._logger.info("FAIRNESS - per-slot JSD(uniform): %s  within 7%%=%s",
                          slot_jsd_uniform, slots_within7_uniform)
        self._logger.info("FAIRNESS - per-slot JSD(target):  %s  within 7%%=%s",
                          slot_jsd_target, slots_within7_target)
        self._logger.info("FAIRNESS - target-double attempts: %s", dict(self.target_double_counts))
        self._logger.info("FAIRNESS - target-double success:  %s  rates=%s",
                          dict(self.target_double_success), double_success_rate)
        self._logger.info("FAIRNESS - missing-class counts:   %s", dict(self.missing_class_counts))
        self._logger.info("FAIRNESS - composition histogram (#DEC,#NC,#INC): %s",
                          dict(self.composition_counts))

        return {
            "overall_jsd_from_uniform": jsd_uniform,
            "overall_max_abs_dev_uniform": max_abs_dev_uniform,
            "within5_uniform": within5_uniform,
            "overall_jsd_from_target": jsd_target,
            "overall_max_abs_dev_target": max_abs_dev_target,
            "within5_target": within5_target,
            "slot_priors": slot_priors,
            "slot_jsd_from_uniform": slot_jsd_uniform,
            "slots_within7_uniform": slots_within7_uniform,
            "slot_jsd_from_target": slot_jsd_target,
            "slots_within7_target": slots_within7_target,
            "target_double_attempts": dict(self.target_double_counts),
            "target_double_success": dict(self.target_double_success),
            "target_double_success_rate": double_success_rate,
            "missing_class_counts": dict(self.missing_class_counts),
            "composition_histogram": {str(k): v for k, v in self.composition_counts.items()},
            "priors_by_depth": priors_by_depth,
            "dev_by_depth": dev_by_depth,
            "target_mix_overall": target_mix,
        }


    # ----- Reporting -----

    def summary(self, payload: Dict[str, Any], records: List[ItemRecord]) -> None:
        self._logger.info(
            "Move-Effect micro-accuracy: %.3f | macro-F1: %.3f",
            payload["micro_acc"], payload["macro_f1"],
        )
        self._logger.info(
            "round-robin/double debt (could not honor): %s", dict(self.class_debt))
        self._logger.info("presented class totals: %s", dict(self.presented_counts))
