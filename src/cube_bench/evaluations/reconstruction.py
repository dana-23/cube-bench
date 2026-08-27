"""Reconstruction task: read the front face off the rendered cube as a 3x3 color grid."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cube_bench.core import ItemRecord, SingleAskTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.core import scoring
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class ReconstructionTest(SingleAskTest):
    """Face color 3x3 reconstruction accuracy (element-wise & overall)."""

    test_type = "reconstruction"

    LOG_EVERY = 25
    MAX_NEW_TOKENS = 2**16
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

    # ----- Construction -----

    def __init__(self, assistant, config, n_moves: int = 3, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self._sys_prompt = ""
        self._user_prompt = ""
        self._total = 0
        self._max_count = self.MAX_SINGLE_COLOR_COUNT
        self._seen = 0
        self._running_elem = 0.0
        self._running_overall = 0.0
        self._color_counts: Counter = Counter()
        self._total_stickers = 0

    def desc(self) -> str:
        return "Reconstruction Test"

    def iter_indices(self, num_samples: int) -> Iterable[int]:
        return range(1, max(0, int(num_samples)) + 1)

    def setup(self, num_samples: int) -> None:
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        self._sys_prompt, self._user_prompt = PromptFactory.get("reconstruction")
        self._total = max(0, int(num_samples))
        self._max_count = 9 if self.n_moves < 3 else 6
        self._seen = 0
        self._running_elem = 0.0
        self._running_overall = 0.0
        self._color_counts = Counter()
        self._total_stickers = 0
        self._logger.info(
            "Starting ReconstructionTest: samples=%s, n_moves=%s, model=%s",
            self._total,
            self.n_moves,
            self.assistant.get_name(),
        )

    # ----- Item generation -----

    def build_item(self, idx: int) -> Dict[str, Any]:
        cube = VirtualCube()

        valid_scramble = False
        attempt = 0
        scramble: Any = None
        gt: List[List[str]] = []

        while not valid_scramble:
            cube.reset()
            current_seed = idx if attempt == 0 else (idx * 10000 + attempt)
            scramble = cube.scramble(
                random_seed=current_seed,
                n_moves=self.n_moves,
                exact_depth=True,
            )

            gt = cube.front_face()
            flat_face = []
            for row in gt:
                for c in row:
                    norm = self._norm_color(c)
                    if norm:
                        flat_face.append(norm)

            counts = Counter(flat_face)
            if any(c > self._max_count for c in counts.values()):
                attempt += 1
                if attempt > 50:
                    self._logger.warning(
                        "Could not satisfy max_count=%s for idx %s, accepting best effort.",
                        self._max_count,
                        idx,
                    )
                    valid_scramble = True
            else:
                valid_scramble = True
                self._color_counts.update(counts)
                self._total_stickers += 9

        image = None
        try:
            image = cube.to_image()
        except Exception as e:
            self._logger.warning("to_image() failed on sample %d: %s", idx, e)

        return {
            "index": idx,
            "image": image,
            "gt": gt,
            "scramble": scramble,
            "attempt": attempt,
        }

    # ----- Prompting -----

    def build_prompts(self, item: Dict[str, Any]) -> Tuple[str, str]:
        return self._sys_prompt, self._user_prompt

    def ask_kwargs(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"image": item["image"], "max_new_tokens": self.MAX_NEW_TOKENS}

    def ask_item(self, item: Dict[str, Any], system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self.ask(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                **self.ask_kwargs(item),
            )
            if self.verbose:
                self._logger.debug(
                    "Sample %s (attempts=%s)\ngt=%s\nscramble=%s",
                    item["index"], item["attempt"], item["gt"], str(item["scramble"]),
                )
            self._logger.debug("Model Response:\n%s", resp)
        except Exception as e:
            self._logger.exception("assistant.generate failed on sample %d: %s", item["index"], e)
            resp = ""
        return resp

    # ----- Parsing -----

    def parse(self, response: Optional[str]) -> Optional[List[List[str]]]:
        return self._parse_grid(response)

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
        self._logger.debug("Reconstruction parse failed; response head: %r", txt[:240])
        return None

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

    # ----- Scoring -----

    def _grid_score(self, gt: List[List[str]], pred: List[List[str]]) -> Tuple[float, float]:
        ngt = self._norm_grid(gt)
        npred = self._norm_grid(pred)
        if ngt is None or npred is None:
            return 0.0, 0.0
        eq = sum(1 for r1, r2 in zip(ngt, npred) for a, b in zip(r1, r2) if a == b)
        return eq / 9.0, (1.0 if eq == 9 else 0.0)

    def score_item(
        self, item: Dict[str, Any], prediction: Optional[List[List[str]]], response: Optional[str]
    ) -> ItemRecord:
        if prediction is None:
            elementwise, overall = 0.0, 0.0
        else:
            elementwise, overall = self._grid_score(item["gt"], prediction)
        return ItemRecord(
            index=item["index"],
            gold=item["gt"],
            pred=prediction,
            correct=overall == 1.0,
            parsed=prediction is not None,
            response=response,
            saved={
                "index": item["index"],
                "gold": item["gt"],
                "pred": prediction,
                "elementwise": elementwise,
                "overall": overall,
                "response": response,
            },
            extra={"elementwise": elementwise, "overall": overall},
        )

    def accumulate(self, record: ItemRecord) -> None:
        self._seen += 1
        self._running_elem += record.extra["elementwise"]
        self._running_overall += record.extra["overall"]

        idx = record.index
        if (idx % self.LOG_EVERY == 0) or (idx == self._total):
            self._logger.info(
                "Reconstruction %d/%d - running avg (elem: %.3f, overall: %.3f)",
                idx, self._total,
                self._running_elem / self._seen,
                self._running_overall / self._seen,
            )

    # ----- Aggregation -----

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        scored = scoring.score(self.test_type, records)
        return {
            "average_accuracy_element_wise": scored["average_accuracy_element_wise"],
            "average_accuracy_overall": scored["average_accuracy_overall"],
            "num_samples": self._total,
            "n_moves": self.n_moves,
            "correct_parse": scored["correct_parse"],
            "max_prior_deviation": self._color_prior_deviation(),
            "per_item": [record.saved for record in records],
        }

    def _color_prior_deviation(self) -> float:
        """Largest deviation of the rendered colour mix from uniform, measured and reported per run."""
        if self._total_stickers <= 0:
            return 0.0
        expected_freq = 1.0 / 6.0
        deviations = []
        self._logger.info("--- Fairness Check (Prior Deviation) ---")
        for color in ["W", "Y", "R", "O", "G", "B"]:
            count = self._color_counts[color]
            freq = count / self._total_stickers
            deviations.append(abs(freq - expected_freq))
            self._logger.info("Color %s: %s (%.4f) | Dev: %.4f", color, count, freq, deviations[-1])
        max_dev = max(deviations)
        self._logger.info("Max Deviation: %.4f (Target < 0.05)", max_dev)
        return max_dev

    # ----- Results -----

    def result(self, payload: Dict[str, Any], records: List[ItemRecord]) -> Dict[str, Any]:
        return payload
