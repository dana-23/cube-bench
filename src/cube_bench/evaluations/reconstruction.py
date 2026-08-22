from __future__ import annotations

import logging
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..core import BaseTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class ReconstructionTest(BaseTest):
    """Face color 3x3 reconstruction accuracy (element-wise & overall)."""

    test_type = "reconstruction"

    LOG_EVERY = 25
    MAX_NEW_TOKENS = 2**16
    DEFAULT_TEMPERATURE = 0.1
    MAX_SINGLE_COLOR_COUNT = 6

    GRID_RE = re.compile(
        r"answer:\s*"
        r"row\s*1:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]\s*"
        r"row\s*2:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]\s*"
        r"row\s*3:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]",
        flags=re.IGNORECASE | re.DOTALL,
    )
    JSONISH_RE = re.compile(
        r"\[\s*\[\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*\]\s*,\s*"
        r"\[\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*\]\s*,\s*"
        r"\[\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*\]\s*\]",
        flags=re.IGNORECASE | re.DOTALL,
    )
    CODE_FENCE_RE = re.compile(r"^```(?:json|python|txt)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)

    COLOR_MAP = {
        "w": "W", "white": "W",
        "y": "Y", "yellow": "Y",
        "r": "R", "red": "R",
        "o": "O", "orange": "O",
        "b": "B", "blue": "B",
        "g": "G", "green": "G",
    }
    FACE_TO_COLOR = {
        "u": "W", "d": "Y", "f": "G",
        "b": "B", "l": "O", "r": "R",
    }

    def _enable_verbose_logging_if_requested(self) -> None:
        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def _norm_color(self, s: str) -> Optional[str]:
        t = (s or "").strip().lower()
        if t in self.COLOR_MAP:
            return self.COLOR_MAP[t]
        if t and t[0] in self.COLOR_MAP:
            return self.COLOR_MAP[t[0]]
        if t in self.FACE_TO_COLOR:
            return self.FACE_TO_COLOR[t]
        if t and t[0] in self.FACE_TO_COLOR:
            return self.FACE_TO_COLOR[t[0]]
        return None

    def _norm_grid(self, grid: List[List[str]]) -> Optional[List[List[str]]]:
        out: List[List[str]] = []
        for row in grid:
            nr: List[str] = []
            for c in row:
                nc = self._norm_color(c)
                if nc is None:
                    return None
                nr.append(nc)
            out.append(nr)
        return out

    def _score(self, gt: List[List[str]], pred: List[List[str]]) -> Tuple[float, float]:
        ngt = self._norm_grid(gt)
        npred = self._norm_grid(pred)
        if ngt is None or npred is None:
            return 0.0, 0.0
        eq = sum(1 for r1, r2 in zip(ngt, npred) for a, b in zip(r1, r2) if a == b)
        return eq / 9.0, (1.0 if eq == 9 else 0.0)

    def _clean_text(self, resp: str) -> str:
        return self.CODE_FENCE_RE.sub("", resp or "").strip()

    def _parse_grid(self, resp: str) -> Optional[List[List[str]]]:
        if not resp:
            return None
        txt = self._clean_text(resp)
        m = self.GRID_RE.search(txt)
        if m:
            c = m.groups()
            return [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]
        j = self.JSONISH_RE.search(txt)
        if j:
            c = j.groups()
            return [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]
        logger.debug("Reconstruction parse failed; response head: %r", txt[:240])
        return None

    def _eval(self, resp: str, gt: List[List[str]], parse: int) -> Tuple[float, float, int]:
        pred = self._parse_grid(resp)
        if pred is None:
            return 0.0, 0.0, parse
        ew, ov = self._score(gt, pred)
        return ew, ov, parse + 1

    def _prompts(self) -> Tuple[str, str]:
        sys_prompt, user_prompt = PromptFactory.get("reconstruction")
        return sys_prompt, user_prompt

    def run(self, num_samples: int) -> Dict[str, Any]:
        self._enable_verbose_logging_if_requested()
        sys_prompt, user_prompt = self._prompts()

        elem_acc: List[float] = []
        full_acc: List[float] = []

        global_color_counts = Counter()
        total_stickers = 0
        parse_total = 0

        total = max(0, int(num_samples))
        logger.info(
            f"Starting ReconstructionTest: samples={total}, n_moves={self.n_moves}, "
            f"model={self.assistant.get_name()}"
        )

        current_max_count = 9 if self.n_moves < 3 else 6

        for idx in tqdm(range(1, total + 1), desc="Reconstruction Test"):
            cube = VirtualCube()

            valid_scramble = False
            attempt = 0
            scramble: Any = None
            gt: List[List[str]] = []

            while not valid_scramble:
                current_seed = idx if attempt == 0 else (idx * 10000 + attempt)
                try:
                    scramble = cube.scramble(random_seed=current_seed, n_moves=self.n_moves)
                except TypeError:
                    random.seed(current_seed)
                    scramble = cube.scramble(n_moves=self.n_moves)

                gt = cube.front_face()
                flat_face = []
                for row in gt:
                    for c in row:
                        norm = self._norm_color(c)
                        if norm:
                            flat_face.append(norm)

                counts = Counter(flat_face)
                if any(c > current_max_count for c in counts.values()):
                    attempt += 1
                    if attempt > 50:
                        logger.warning(
                            f"Could not satisfy max_count={current_max_count} for idx {idx}, accepting best effort."
                        )
                        valid_scramble = True
                else:
                    valid_scramble = True
                    global_color_counts.update(counts)
                    total_stickers += 9

            image = None
            try:
                image = cube.to_image()
            except Exception as e:
                logger.warning("to_image() failed on sample %d: %s", idx, e)

            try:
                resp = self.ask(
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                    image=image,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    temperature=self.DEFAULT_TEMPERATURE,
                )
                if self.verbose:
                    logger.debug(f"Sample {idx} (attempts={attempt})\ngt={gt}\nscramble={scramble.__str__()}")
                logger.debug(f"Model Response:\n{resp}")
            except Exception as e:
                logger.exception("assistant.generate failed on sample %d: %s", idx, e)
                resp = ""

            ew, ov, parse_total = self._eval(resp, gt, parse_total)
            elem_acc.append(ew)
            full_acc.append(ov)

            if (idx % self.LOG_EVERY == 0) or (idx == total):
                logger.info(
                    "Reconstruction %d/%d — running avg (elem: %.3f, overall: %.3f)",
                    idx, total,
                    (sum(elem_acc) / len(elem_acc)) if elem_acc else 0.0,
                    (sum(full_acc) / len(full_acc)) if full_acc else 0.0,
                )

        # Fairness check
        if total_stickers > 0:
            expected_freq = 1.0 / 6.0
            max_dev = 0.0
            logger.info("--- Fairness Check (Prior Deviation) ---")
            for color in ["W", "Y", "R", "O", "G", "B"]:
                count = global_color_counts[color]
                freq = count / total_stickers
                dev = abs(freq - expected_freq)
                max_dev = max(max_dev, dev)
                logger.info(f"Color {color}: {count} ({freq:.4f}) | Dev: {dev:.4f}")
            logger.info(f"Max Deviation: {max_dev:.4f} (Target < 0.05)")
        else:
            max_dev = 0.0

        avg_ew = (sum(elem_acc) / len(elem_acc)) if elem_acc else 0.0
        avg_ov = (sum(full_acc) / len(full_acc)) if full_acc else 0.0

        result = {
            "average_accuracy_element_wise": avg_ew,
            "average_accuracy_overall": avg_ov,
            "num_samples": total,
            "n_moves": self.n_moves,
            "correct_parse": parse_total,
            "max_prior_deviation": max_dev,
        }
        self.save(result)
        # Return shape preserved for callers (includes the auto-filled fields would require re-reading)
        return result
