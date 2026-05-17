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

from ..core import BaseTest
from cube_bench.tests.solve_moves import SolveMovesTest

logger = logging.getLogger(__name__)


def _load_reflection_bundle(path: Path, reflection_type: str) -> Dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if reflection_type not in data:
        raise KeyError(f"'{reflection_type}' not in {list(data.keys())}")
    bundle = data[reflection_type]
    if not isinstance(bundle, dict) or "system" not in bundle or "user" not in bundle:
        raise ValueError("Reflection bundle must have {'system','user'}.")
    return bundle


def _reanswer_bundle() -> dict:
    return {
        "system": (
            "You are an expert Rubik's-Cube assistant.\n"
            "Allowed moves: F B L R U D with optional ' and 2 (e.g., R, U', F2). Centers never move.\n"
            "If an image is attached, TREAT THE TEXT STATE AS AUTHORITATIVE.\n"
            "You are in a RE-ANSWER phase: use the provided reflection to avoid the prior mistake.\n"
            "The reflection may be JSON (e.g., keys: diagnosis, keywords, avoid_rules, eval, recommend, prior_answer)\n"
            "or plain text. If avoid_rules or prior_answer is present, do NOT choose those options.\n"
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


def _save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class ReflectionTest(BaseTest):
    """Maximum-fairness reflection/re-answer evaluation.

    Modes:
      - reflect_all=True  (default): paired, fair comparison on the *same* set.
      - reflect_all=False: wrong-only quick pass for Error-Fix Rate (EFR).
    """

    test_type = "reflection"

    def __init__(
        self,
        assistant,
        config,
        reflection_prompts: Path,
        reflection_type: str = "Unredacted",
        prompt_type: str = "image",
        max_reflections: Optional[int] = None,
        results_subdir: str = "reflection",
        verbose: bool = False,
        reflect_all: bool = True,
    ):
        super().__init__(assistant, config, n_moves=1, verbose=verbose)
        self.reflection_prompts = Path(reflection_prompts)
        self.reflection_type = reflection_type
        self.prompt_type = prompt_type
        self.max_reflections = max_reflections
        self.reflect_all = reflect_all
        self.results_root = Path(config.results_dir) / results_subdir

    def _make_run_dir(self, model_name: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = self.results_root / f"{model_name}_{self.reflection_type}_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        return out

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
        out = self.assistant.generate(
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
        logger.info(f"Running {self.reflection_type} Reflection")
        logger.info("=" * 80)

        model_name = getattr(self.assistant, "model_name", None) or self.assistant.get_name()
        if callable(model_name):
            model_name = model_name()
        out_dir = self._make_run_dir(str(model_name))

        # 1) Draft pass via SolveMovesTest (VirtualCube, no dataset dependency)
        solver = SolveMovesTest(
            assistant=self.assistant,
            config=self.config,
            prompt_type="mixed",
            n_moves=1,
            verbose=self.verbose,
        )
        wrong, acc_bits = solver.run(num_samples=num_samples)
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
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger.info("[Reflection] nothing to reflect; InitAcc=%.3f", init_acc)
            return summary

        # 2) Reflection pass
        bundle = _load_reflection_bundle(self.reflection_prompts, self.reflection_type)
        reflections: List[Dict[str, Any]] = []
        ref_tokens = ref_latency = 0

        for idx in tqdm(reflect_indices, desc=f"Reflect({self.reflection_type})"):
            sample = solver._build_sample(idx)
            user = bundle["user"].format(
                cube_state=sample["text_state"],
                option_A=sample["options"]["A"],
                option_B=sample["options"]["B"],
                option_C=sample["options"]["C"],
                option_D=sample["options"]["D"],
                model_choice="[HIDDEN]",
                correct_answer=sample["correct_letter"],
            )
            text, usage, dt_ms = self._ask_with_usage(
                system_prompt=bundle["system"],
                user_prompt=user,
                image=sample["image"],
                max_new_tokens=2**16,
                temperature=0.4,
            )
            ref_tokens += _count_tokens(usage)
            ref_latency += dt_ms
            reflections.append({
                "index": idx,
                "reflection_text": text,
                "latency_ms": dt_ms,
                "usage": usage,
            })

        _save_jsonl(out_dir / "reflections.jsonl", reflections)

        # 3) Re-answer pass
        reask = _reanswer_bundle()
        reanswers: List[Dict[str, Any]] = []
        re_tokens = re_latency = 0

        for r in tqdm(reflections, desc="Reanswer"):
            idx = r["index"]
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

        _save_jsonl(out_dir / "reanswers.jsonl", reanswers)

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

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
