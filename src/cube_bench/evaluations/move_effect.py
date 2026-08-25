"""Move-effect task: label each option DECREASE / NO_CHANGE / INCREASE."""

from __future__ import annotations

import logging
import math
import random
import re
from collections import Counter, defaultdict, deque
from typing import Dict, List, Tuple

from tqdm import tqdm

from cube_bench.core import BaseTest
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class MoveEffectTest(BaseTest):
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

    # Neighbor labels
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

    # Fairness metrics
    @staticmethod
    def _safe_prop(count: int, total: int) -> float:
        return (count / total) if total else 0.0

    @staticmethod
    def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
        """Jensen-Shannon divergence (base 2). p,q over same keys."""
        keys = list(p.keys())
        m = {k: 0.5 * (p[k] + q[k]) for k in keys}
        def _kl(a, b):
            s = 0.0
            for k in keys:
                if a[k] > 0 and b[k] > 0:
                    s += a[k] * math.log(a[k] / b[k], 2)
            return s
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

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

    # Target selection
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

    # Slot assignment
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

    # Option sampling
    def _balanced_sample_ABCD(
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

    # Prompts
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

    def run(self, num_samples: int):
        micro_correct = 0
        total_labels = 0
        confusion = defaultdict(Counter)  # gold -> pred
        per_class = Counter()

        option_mix_ok = Counter()
        per_distance = defaultdict(lambda: {"correct": 0, "total": 0})

        logger.info("=" * 80)
        logger.info("Initializing Move-Effect test on %s", self.assistant.get_name())
        logger.info("Number of samples: %s | Scramble depth: %s", num_samples, self.n_moves)
        logger.info("=" * 80)

        for idx in tqdm(range(num_samples), desc=f"Move-Effect (n_moves={self.n_moves})"):
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
            option_mix_ok[len(classes_in_item)] += 1

            truth = {k: labels_by_move[mv] for k, mv in options.items()}

            for lbl in truth.values():
                self.presented_counts[lbl] += 1
                self.depth_presented_counts[d][lbl] += 1

            centers = self._face_centers(cube)
            state_text = self.state_text(cube)
            sys_prompt = self._sys_prompt()
            user_prompt = self._user_prompt(centers, state_text, options)

            response = self.ask(
                user_prompt=user_prompt, system_prompt=sys_prompt, image=None,
            )

            preds = {m.group(1).upper(): m.group(2).upper().replace(" ", "_")
                     for m in self.TAG_RE.finditer(response)}
            for k in "ABCD":
                if k not in preds:
                    pat = self.LETTER_RES[k].search(response)
                    if pat:
                        preds[k] = pat.group(1).replace(" ", "_").upper()

            correct_this_item = 0
            for k in "ABCD":
                gold = truth[k]
                per_class[gold] += 1
                pred = preds.get(k, "MISSING")
                confusion[gold][pred] += 1
                is_right = int(pred == gold)
                micro_correct += is_right
                total_labels += 1
                correct_this_item += is_right

            per_distance[d]["correct"] += correct_this_item
            per_distance[d]["total"] += 4

            if self.verbose:
                predictions = {k: preds.get(k) for k in 'ABCD'}
                logger.info("[%s] d=%s  scramble=%s", idx, d, scramble)
                logger.info(
                    "bucket sizes: DEC=%s, NC=%s, INC=%s",
                    len(buckets['DECREASE']),
                    len(buckets['NO_CHANGE']),
                    len(buckets['INCREASE']),
                )
                logger.info("options: %s", options)
                logger.info("truth:   %s", truth)
                logger.info("preds:   %s", predictions)

        # Aggregate accuracy and fairness metrics.
        tot = sum(per_class.values())
        tri = ("INCREASE", "NO_CHANGE", "DECREASE")
        priors = {k: self._safe_prop(per_class[k], tot) for k in tri}

        pred_totals = Counter()
        for gold, row in confusion.items():
            for pred, c in row.items():
                pred_totals[pred] += c
        pred_tot = sum(pred_totals[k] for k in tri)
        q = {k: (pred_totals[k] / pred_tot) if pred_tot else 0.0 for k in tri}

        def dot(a, b):
            return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in tri)
        maj_baseline = max(priors.values()) if priors else 0.0
        prior_sample_baseline = dot(priors, priors)
        model_expected = dot(priors, q)

        logger.info("gold priors π: %s", priors)
        logger.info("model preds q: %s", q)
        logger.info("baseline(always majority): %.3f  baseline(prior-sample): %.3f  exp(acc from q·π): %.3f",
                    maj_baseline, prior_sample_baseline, model_expected)
        logger.info("option class coverage counts (distinct classes per item): %s", dict(option_mix_ok))

        micro_acc = micro_correct / total_labels if total_labels else 0.0

        pe = model_expected
        kappa = (micro_acc - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0

        per_class_recall = {}
        per_class_precision = {}
        per_class_f1 = {}
        for cls in tri:
            tp = confusion[cls].get(cls, 0)
            fn = sum(v for k, v in confusion[cls].items() if k != cls)
            fp = sum(confusion[g].get(cls, 0) for g in tri if g != cls)
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_class_recall[cls] = rec
            per_class_precision[cls] = prec
            per_class_f1[cls] = f1

        macro_f1 = sum(per_class_f1[c] for c in tri) / 3.0 if tri else 0.0

        per_distance_acc = {int(d): (v["correct"] / v["total"]) for d, v in per_distance.items() if v["total"]}

        uniform = {k: 1 / 3 for k in tri}
        jsd_uniform = self._jsd(priors, uniform)
        max_abs_dev_uniform = max(abs(priors[k] - 1 / 3) for k in tri) if tri else 0.0

        target_mix = self._overall_feasible_target()
        jsd_target = self._jsd(priors, target_mix)
        max_abs_dev_target = max(abs(priors[k] - target_mix[k]) for k in tri)

        slot_priors = {s: {c: self._safe_prop(self.per_slot_counts[s][c], sum(self.per_slot_counts[s].values()))
                           for c in tri} for s in self.SLOTS}
        slot_jsd_uniform = {s: self._jsd(slot_priors[s], uniform) for s in self.SLOTS}
        slot_jsd_target = {s: self._jsd(slot_priors[s], target_mix) for s in self.SLOTS}

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
                pd = {k: self._safe_prop(cnts.get(k, 0), tot_d) for k in tri}
                priors_by_depth[int(d)] = pd
                target_d = self._feasible_target_for_depth(d)
                dev_by_depth[int(d)] = {
                    "max_abs_dev_uniform": max(abs(pd[k] - 1 / 3) for k in tri),
                    "jsd_from_uniform": self._jsd(pd, uniform),
                    "max_abs_dev_target": max(abs(pd[k] - target_d[k]) for k in tri),
                    "jsd_from_target": self._jsd(pd, target_d),
                }

        within5_uniform = all(abs(priors[k] - 1 / 3) <= 0.05 for k in tri)
        within5_target = all(abs(priors[k] - target_mix[k]) <= 0.05 for k in tri)
        slots_within7_uniform = all(all(abs(slot_priors[s][k] - 1 / 3) <= 0.07 for k in tri) for s in self.SLOTS)
        slots_within7_target = all(all(abs(slot_priors[s][k] - target_mix[k]) <= 0.07 for k in tri) for s in self.SLOTS)

        logger.info("FAIRNESS ─ overall JSD(uniform)=%.4f  max|Δ|=%.3f  within±5%%=%s",
                    jsd_uniform, max_abs_dev_uniform, within5_uniform)
        logger.info("FAIRNESS ─ overall JSD(target)=%.4f  max|Δ|=%.3f  within±5%%=%s  target=%s",
                    jsd_target, max_abs_dev_target, within5_target, target_mix)
        logger.info("FAIRNESS ─ per-slot priors: %s", slot_priors)
        logger.info("FAIRNESS ─ per-slot JSD(uniform): %s  within±7%%=%s", slot_jsd_uniform, slots_within7_uniform)
        logger.info("FAIRNESS ─ per-slot JSD(target):  %s  within±7%%=%s", slot_jsd_target, slots_within7_target)
        logger.info("FAIRNESS ─ target-double attempts: %s", dict(self.target_double_counts))
        logger.info("FAIRNESS ─ target-double success:  %s  rates=%s",
                    dict(self.target_double_success), double_success_rate)
        logger.info("FAIRNESS ─ missing-class counts:   %s", dict(self.missing_class_counts))
        logger.info("FAIRNESS ─ composition histogram (#DEC,#NC,#INC): %s", dict(self.composition_counts))

        fairness_metrics = {
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

        out = {
            "n_moves": self.n_moves,
            "micro_acc": micro_acc,
            "macro_f1": macro_f1,
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
        self.save(out)
        logger.info("Move-Effect micro-accuracy: %.3f | macro-F1: %.3f", micro_acc, macro_f1)
        logger.info("round-robin/double debt (couldn’t honor): %s", dict(self.class_debt))
        logger.info("presented class totals: %s", dict(self.presented_counts))
