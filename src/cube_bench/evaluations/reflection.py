"""Reflection task: does a model abandon a correct answer when prompted to reflect?"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

from cube_bench.core import BaseTest
from cube_bench.core.results import (
    read_jsonl, timestamped_run_dir, write_json, write_jsonl,
)
from cube_bench.evaluations.solve_moves import SolveMovesTest

logger = logging.getLogger(__name__)


ASSERT_MODES = ("always", "when_wrong", "never")
REANSWER_MODES = ("legacy", "neutral")


def _load_reflection_bundle(
    path: Path, reflection_type: str, needs_open: bool = False
) -> Dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if reflection_type not in data:
        raise KeyError(f"'{reflection_type}' not in {list(data.keys())}")
    bundle = data[reflection_type]
    # The *_open pair is only read when some item is allowed to go un-asserted.
    required = {"system", "user"} | ({"system_open", "user_open"} if needs_open else set())
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Reflection bundle must have {sorted(required)}.")
    return bundle


# ``legacy`` changes the directive with assertion status; ``neutral`` fixes one
# non-directive prompt so ``assert_incorrect`` is the only manipulated factor.
_LEGACY_PHASE_ASSERTED = (
    "You are in a RE-ANSWER phase: use the provided reflection to avoid the prior mistake.\n"
    "The reflection may be JSON (e.g., keys: diagnosis, keywords, avoid_rules, eval, recommend, prior_answer)\n"
    "or plain text. If avoid_rules or prior_answer is present, do NOT choose those options.\n"
)
_LEGACY_PHASE_OPEN = (
    "You are in a RE-ANSWER phase: use the provided reflection to answer as accurately as you can.\n"
    "The reflection may endorse your prior answer or propose a different one; follow whichever it\n"
    "concludes is correct. Do not change your answer merely because you were asked again.\n"
)
_NEUTRAL_PHASE = (
    "You are in a RE-ANSWER phase: using the cube state, the options, and the reflection below,\n"
    "give your final answer. The reflection may be JSON or plain text.\n"
)


def _reanswer_bundle(assert_wrong: bool = True, mode: str = "legacy") -> dict:
    if mode not in REANSWER_MODES:
        raise ValueError(f"reanswer_mode must be one of {REANSWER_MODES}")
    if mode == "neutral":
        phase = _NEUTRAL_PHASE
    else:
        phase = _LEGACY_PHASE_ASSERTED if assert_wrong else _LEGACY_PHASE_OPEN
    return {
        "system": (
            "You are an expert Rubik's-Cube assistant.\n"
            "Allowed moves: F B L R U D with optional ' and 2 (e.g., R, U', F2). Centers never move.\n"
            "If an image is attached, TREAT THE TEXT STATE AS AUTHORITATIVE.\n"
            + phase +
            "If the reflection rates options (e.g., DEC/NO_CHANGE/INC), prefer DEC > NO_CHANGE > INC.\n"
            "If the reflection is missing or unclear, choose the option most consistent with its advice; "
            "on ties use this order: A > B > C > D.\n"
            "Return exactly ONE line in the format: ANSWER: A|B|C|D\n"
            "Do NOT include explanations, quotes, JSON, or extra text."
        ),
        "user": (
            "Cube state (text grids):\n{cube_state}\n\n"
            "Candidate moves:\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n\n"
            "Reflection:\n{reflection}\n\n"
            "Respond EXACTLY as: ANSWER: A  or ANSWER: B  or ANSWER: C  or ANSWER: D"
        ),
    }


def _count_tokens(usage: Dict[str, Any]) -> int:
    if not usage:
        return 0
    if "total_tokens" in usage and isinstance(usage["total_tokens"], (int, float)):
        return int(usage["total_tokens"])
    s = 0
    for a, b in (("prompt_tokens", "completion_tokens"), ("input_tokens", "output_tokens")):
        if a in usage or b in usage:
            s = int(usage.get(a, 0)) + int(usage.get(b, 0))
            break
    return s


def _draft_from_reflection_dir(run_dir: Path) -> List[Dict[str, Any]]:
    """Recover legacy draft data by joining reflection and re-answer JSONL files."""
    refl = read_jsonl(run_dir / "reflections.jsonl")
    gold = {r["index"]: r.get("gold") for r in read_jsonl(run_dir / "reanswers.jsonl")}
    return sorted(
        (
            {
                "id": r["index"],
                "pred": r.get("prior_answer"),
                "gold": gold.get(r["index"]),
                "ok": int(bool(r["initially_correct"])),
            }
            for r in refl
        ),
        key=lambda x: x["id"],
    )


def _resolve_draft(draft_from: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Accepts a draft.json, a solve_moves_*.json, or a reflection run directory."""
    p = Path(draft_from)
    if p.is_dir():
        if (p / "draft.json").exists():
            p = p / "draft.json"
        elif (p / "reflections.jsonl").exists() and (p / "reanswers.jsonl").exists():
            return _draft_from_reflection_dir(p), str(p)
        else:
            raise FileNotFoundError(
                f"{p} has no draft.json and no reflections.jsonl/reanswers.jsonl pair."
            )
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):  # save_results() appends; the last entry is the newest run
        data = data[-1]
    per_item = data.get("per_item")
    if not per_item:
        raise ValueError(
            f"{p} has no 'per_item' records (written only since commit 7fb081d). "
            f"Point --draft_from at the reflection run directory instead."
        )
    return sorted(per_item, key=lambda x: x["id"]), str(p)


class ReflectionTest(BaseTest):
    """Run paired all-item or wrong-only reflection and re-answer evaluations."""

    test_type = "reflection"

    def __init__(
        self,
        assistant,
        config,
        *,
        reflection_prompts: Path,
        n_moves: int = 1,
        reflection_type: str = "Unredacted",
        max_reflections: Optional[int] = None,
        results_subdir: str = "reflection",
        verbose: bool = False,
        reflect_all: bool = True,
        reveal_choice: bool = False,
        assert_incorrect: str = "always",
        reanswer_mode: str = "legacy",
        draft_from: Optional[Path] = None,
    ):
        super().__init__(assistant, config, n_moves=n_moves, verbose=verbose)
        if assert_incorrect not in ASSERT_MODES:
            raise ValueError(f"assert_incorrect must be one of {ASSERT_MODES}")
        if reanswer_mode not in REANSWER_MODES:
            raise ValueError(f"reanswer_mode must be one of {REANSWER_MODES}")
        self.reflection_prompts = Path(reflection_prompts)
        self.reflection_type = reflection_type
        self.max_reflections = max_reflections
        self.reflect_all = reflect_all
        self.reveal_choice = reveal_choice
        self.assert_incorrect = assert_incorrect
        self.reanswer_mode = reanswer_mode
        self.draft_from = Path(draft_from) if draft_from else None
        self.results_root = Path(config.results_dir) / results_subdir

    def _make_run_dir(self, model_name: str) -> Path:
        return timestamped_run_dir(self.results_root, model_name, self.reflection_type)

    def _ask_with_usage(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, Dict[str, int], int]:
        """Returns (text, usage_dict, latency_ms) — robust to strategy return shape."""
        t0 = time.perf_counter()
        out = self._generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        text, usage = out, {}
        if isinstance(out, tuple):
            text = out[0]
            if len(out) >= 2 and isinstance(out[1], dict):
                usage = out[1]
        return text, usage, dt_ms

    def run(self, num_samples: int) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("Running %s Reflection", self.reflection_type)
        logger.info("=" * 80)

        model_name = self.assistant.get_name()
        out_dir = self._make_run_dir(model_name)

        # 1) Draft pass via SolveMovesTest (VirtualCube, no dataset dependency)
        solver = SolveMovesTest(
            assistant=self.assistant,
            config=self.config,
            prompt_type="mixed",
            n_moves=self.n_moves,
            verbose=self.verbose,
        )
        if self.draft_from:
            per_item, draft_source = _resolve_draft(self.draft_from)
            if len(per_item) < num_samples or [r["id"] for r in per_item[:num_samples]] != list(range(num_samples)):
                raise ValueError(
                    f"Cached draft {draft_source} does not cover items 0..{num_samples - 1}."
                )
            per_item = per_item[:num_samples]
            # Reject cached drafts when seeded item generation has drifted.
            for r in per_item:
                gold = solver._build_sample(r["id"])["correct_letter"]
                if r["gold"] != gold:
                    raise ValueError(
                        f"Draft item {r['id']} gold={r['gold']} but regenerates as {gold}; "
                        f"the cached draft was built from different items."
                    )
            unparsed = [r["id"] for r in per_item if r["pred"] is None]
            if self.reveal_choice and unparsed:
                if len(unparsed) == len(per_item):
                    raise ValueError(
                        f"reveal_choice=true needs the draft's predicted letters, but every item in "
                        f"{draft_source} has pred=null — that draft came from a reveal_choice=false run."
                    )
                # Correct items without a parsed answer invalidate the OTR denominator.
                if corrupt := [r["id"] for r in per_item if r["pred"] is None and r["ok"]]:
                    raise ValueError(
                        f"Draft items {corrupt} are marked initially-correct but have no parsed "
                        f"answer in {draft_source}; the OTR denominator would be corrupt."
                    )
                # Unparsed wrong answers mirror uncached ``[HIDDEN]`` prompts and do not affect OTR.
                logger.warning(
                    "[Reflection] %d/%d draft answers did not parse (ids %s). They render as "
                    "'[HIDDEN]' in the reveal prompt, exactly as an uncached run would. All are "
                    "initially-wrong, so the OTR denominator (%d items) is unaffected; they sit in "
                    "the EFR denominator.",
                    len(unparsed), len(per_item), unparsed, sum(r["ok"] for r in per_item),
                )
            n_draft_unparsed = len(unparsed)
            acc_bits = [int(r["ok"]) for r in per_item]
            preds = [r["pred"] for r in per_item]
            wrong = [(r["pred"], r["id"]) for r in per_item if not r["ok"]]
            logger.info(
                "[Reflection] draft pass loaded from %s — %d items, InitAcc=%.3f, no draft calls made.",
                draft_source, num_samples, sum(acc_bits) / num_samples,
            )
        else:
            wrong, acc_bits, preds = solver.run(num_samples=num_samples)
            per_item = [
                {"id": i, "pred": preds[i], "gold": solver._build_sample(i)["correct_letter"], "ok": int(acc_bits[i])}
                for i in range(len(acc_bits))
            ]
            draft_source = "fresh"
            n_draft_unparsed = sum(1 for p in preds if p is None)

        # Persist the draft so sibling arms share the same initially-correct set.
        write_json(
            out_dir / "draft.json",
            {"source": draft_source, "num_samples": num_samples, "per_item": per_item},
        )

        n_items = len(acc_bits)
        all_indices = list(range(n_items))

        if self.reflect_all:
            reflect_indices = all_indices
        else:
            reflect_indices = [idx for (_pred, idx) in wrong]
            if self.max_reflections is not None and len(reflect_indices) > self.max_reflections:
                reflect_indices = reflect_indices[: self.max_reflections]
                logger.info("Capped reflections to %d items (wrong-only).", self.max_reflections)

        if not reflect_indices:
            init_acc = round(sum(acc_bits) / n_items, 4) if n_items else 0.0
            summary = {
                "model": model_name,
                "reflection_type": self.reflection_type,
                "reflect_all": self.reflect_all,
                "reveal_choice": self.reveal_choice,
                "assert_incorrect": self.assert_incorrect,
                "reanswer_mode": self.reanswer_mode,
                "draft_source": draft_source,
                "n_draft_unparsed": n_draft_unparsed,
                "n_items": n_items,
                "n_reflected": 0,
                "initial_accuracy": init_acc,
                "final_accuracy_over_all": init_acc if self.reflect_all else None,
                "final_accuracy_over_reflected": init_acc if not self.reflect_all else None,
                "error_fix_rate": None,
                "error_fix_rate_ci95": None,
                "overthink_rate": None,
                "overthink_rate_ci95": None,
                "paired_net_gain_over_all": 0.0,
                "paired_net_gain_over_reflected": 0.0,
                "parse_rate": 1.0,
                "delta_tokens_total": 0,
                "delta_tokens_per_item": 0,
                "delta_latency_ms_total": 0,
                "delta_latency_ms_per_item": 0,
                "run_dir": str(out_dir),
                "notes": "No items to reflect.",
            }
            write_json(out_dir / "summary.json", summary)
            logger.info("[Reflection] nothing to reflect; InitAcc=%.3f", init_acc)
            return summary

        # 2) Reflection pass
        bundle = _load_reflection_bundle(
            self.reflection_prompts,
            self.reflection_type,
            needs_open=self.assert_incorrect != "always",
        )
        reflections: List[Dict[str, Any]] = []
        ref_tokens = ref_latency = 0

        for idx in tqdm(reflect_indices, desc=f"Reflect({self.reflection_type})"):
            sample = solver._build_sample(idx)

            assert_wrong = self.assert_incorrect == "always" or (
                self.assert_incorrect == "when_wrong" and not acc_bits[idx]
            )
            choice = preds[idx] if self.reveal_choice else None
            sys_key, user_key = ("system", "user") if assert_wrong else ("system_open", "user_open")

            user = bundle[user_key].format(
                cube_state=sample["text_state"],
                option_A=sample["options"]["A"],
                option_B=sample["options"]["B"],
                option_C=sample["options"]["C"],
                option_D=sample["options"]["D"],
                model_choice=choice or "[HIDDEN]",
                choice_line=f"You previously chose option: {choice}. " if choice else "",
                correct_answer=sample["correct_letter"],
            )
            text, usage, dt_ms = self._ask_with_usage(
                system_prompt=bundle[sys_key],
                user_prompt=user,
                image=sample["image"],
                max_new_tokens=2**16,
            )
            ref_tokens += _count_tokens(usage)
            ref_latency += dt_ms
            reflections.append({
                "index": idx,
                "prior_answer": choice,
                "asserted_incorrect": assert_wrong,
                "initially_correct": bool(acc_bits[idx]),
                "reflection_text": text,
                "latency_ms": dt_ms,
                "usage": usage,
            })

        write_jsonl(out_dir / "reflections.jsonl", reflections)

        # 3) Re-answer pass
        reanswers: List[Dict[str, Any]] = []
        re_tokens = re_latency = 0

        for r in tqdm(reflections, desc="Reanswer"):
            idx = r["index"]
            reask = _reanswer_bundle(assert_wrong=r["asserted_incorrect"], mode=self.reanswer_mode)
            sample = solver._build_sample(idx)
            reply, usage, dt_ms = self._ask_with_usage(
                system_prompt=reask["system"],
                user_prompt=reask["user"].format(
                    cube_state=sample["text_state"],
                    A=sample["options"]["A"],
                    B=sample["options"]["B"],
                    C=sample["options"]["C"],
                    D=sample["options"]["D"],
                    reflection=r["reflection_text"],
                ),
                image=sample["image"],
                max_new_tokens=2**16,
            )
            pred = self.parse_letter(reply)
            gold = sample["correct_letter"]
            re_tokens += _count_tokens(usage)
            re_latency += dt_ms
            reanswers.append({
                "index": idx,
                "pred": pred,
                "gold": gold,
                "raw": reply,
                "latency_ms": dt_ms,
                "usage": usage,
                "parsed": pred in {"A", "B", "C", "D"},
            })

        write_jsonl(out_dir / "reanswers.jsonl", reanswers)

        # 4) Metrics
        before = {i: int(acc_bits[i]) for i in all_indices}
        after: Dict[int, int] = {}
        parsed: Dict[int, bool] = {}

        for row in reanswers:
            i = row["index"]
            ok = int(row["pred"] == row["gold"]) if row["parsed"] else 0
            after[i] = ok
            parsed[i] = bool(row["parsed"])

        if not self.reflect_all:
            for i in all_indices:
                if i not in after:
                    after[i] = before[i]
                    parsed[i] = True

        n_reflected = len(reflect_indices)
        init_acc = sum(before.values()) / n_items if n_items else 0.0
        final_acc_all = (sum(after[i] for i in all_indices) / n_items) if self.reflect_all else float("nan")
        final_acc_reflected = (
            sum(after[i] for i in reflect_indices) / n_reflected if n_reflected else float("nan")
        )

        png_over_all = mean((after[i] - before[i]) for i in all_indices) if n_items else 0.0
        png_over_reflected = (
            mean((after[i] - before[i]) for i in reflect_indices) if n_reflected else 0.0
        )

        wrong_set = [i for i in reflect_indices if before[i] == 0]
        right_set = [i for i in reflect_indices if before[i] == 1]

        efr = (sum(after[i] for i in wrong_set) / len(wrong_set)) if wrong_set else float("nan")
        otr = (sum(1 - after[i] for i in right_set) / len(right_set)) if right_set else float("nan")

        efr_ci = self.wilson_ci(efr, len(wrong_set)) if wrong_set else (float("nan"), float("nan"))
        otr_ci = self.wilson_ci(otr, len(right_set)) if right_set else (float("nan"), float("nan"))
        fin_ci = self.wilson_ci(final_acc_all, n_items) if self.reflect_all else (float("nan"), float("nan"))

        parse_rate = (
            sum(1 for i in reflect_indices if parsed.get(i, False)) / n_reflected
            if n_reflected else 1.0
        )

        extra_tokens = ref_tokens + re_tokens
        extra_latency_ms = ref_latency + re_latency
        delta_tokens_per_item = int(round(extra_tokens / n_reflected)) if n_reflected else 0
        delta_latency_ms_per_item = int(round(extra_latency_ms / n_reflected)) if n_reflected else 0

        summary = {
            "model": model_name,
            "reflection_type": self.reflection_type,
            "reflect_all": self.reflect_all,
            "reveal_choice": self.reveal_choice,
            "assert_incorrect": self.assert_incorrect,
            "reanswer_mode": self.reanswer_mode,
            "draft_source": draft_source,
            "n_draft_unparsed": n_draft_unparsed,
            "n_items": n_items,
            "n_reflected": n_reflected,
            "initial_accuracy": round(init_acc, 4),
            "final_accuracy_over_all": round(final_acc_all, 4) if self.reflect_all else None,
            "final_accuracy_over_reflected": round(final_acc_reflected, 4),
            "error_fix_rate": round(efr, 4) if not math.isnan(efr) else None,
            "error_fix_rate_ci95": (
                [round(efr_ci[0], 4), round(efr_ci[1], 4)] if not math.isnan(efr) else None
            ),
            "overthink_rate": round(otr, 4) if not math.isnan(otr) else None,
            "overthink_rate_ci95": (
                [round(otr_ci[0], 4), round(otr_ci[1], 4)] if not math.isnan(otr) else None
            ),
            "paired_net_gain_over_all": round(png_over_all, 4),
            "paired_net_gain_over_reflected": round(png_over_reflected, 4),
            "parse_rate": round(parse_rate, 4),
            "delta_tokens_total": int(extra_tokens),
            "delta_tokens_per_item": int(delta_tokens_per_item),
            "delta_latency_ms_total": int(extra_latency_ms),
            "delta_latency_ms_per_item": int(delta_latency_ms_per_item),
            "run_dir": str(out_dir),
        }

        write_json(out_dir / "summary.json", summary)

        if self.reflect_all:
            logger.info(
                "[Reflect-ALL/%s] N=%d | InitAcc=%.3f | FinalAcc=%.3f (CI95 %.3f–%.3f) | "
                "EFR=%.3f (%.3f–%.3f) | OTR=%.3f (%.3f–%.3f) | PNG=%.3f | Parse=%.1f%% | "
                "ΔTok/it=%d | ΔLat/it=%dms",
                self.reflection_type, n_items, init_acc, final_acc_all, fin_ci[0], fin_ci[1],
                efr, efr_ci[0], efr_ci[1], otr, otr_ci[0], otr_ci[1],
                png_over_all, 100 * parse_rate, delta_tokens_per_item, delta_latency_ms_per_item,
            )
        else:
            logger.info(
                "[Reflect-WRONG/%s] N=%d Reflected=%d | InitAcc=%.3f | FinalAcc(reflected)=%.3f | "
                "EFR=%.3f (%.3f–%.3f) | PNG(reflected)=%.3f | Parse=%.1f%% | "
                "ΔTok/it=%d | ΔLat/it=%dms",
                self.reflection_type, n_items, n_reflected, init_acc, final_acc_reflected,
                efr, efr_ci[0], efr_ci[1], png_over_reflected,
                100 * parse_rate, delta_tokens_per_item, delta_latency_ms_per_item,
            )

        return summary
