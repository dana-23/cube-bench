from __future__ import annotations

import logging
import math
import random
import re
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..config import Config
from ..io import save_results

logger = logging.getLogger(__name__)


# Unified MCQ answer parsers (accept all formats seen across tests)
_ANSWER_TAG_RE = re.compile(r"<\s*ANSWER\s*>\s*([ABCD])\s*<\s*/\s*ANSWER\s*>", re.IGNORECASE)
_ANSWER_COLON_RE = re.compile(r"\bANSWER\s*[:=]\s*([ABCD])\b", re.IGNORECASE)
_ANSWER_BRACKET_RE = re.compile(r"<\s*([ABCD])\s*>", re.IGNORECASE)
_MOVE_TAG_RE = re.compile(r"<ANSWER>\s*([URFDLB](?:2|')?)\s*</ANSWER>", re.IGNORECASE)
_IDK_RE = re.compile(
    r"(?:<ANSWER>\s*(IDK)\s*</ANSWER>)|(?:\bANSWER\s*[:=]\s*(?:IDK|E)\b)|(?:I\s*DON'?T\s*KNOW)",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(r"Answer:\s*(Yes|No)\b", re.IGNORECASE)


class BaseTest(ABC):
    """Abstract base class for all cube-bench tests.

    Provides shared utilities so each subclass only implements its unique logic
    in ``run(num_samples)``.
    """

    test_type: str = "base"

    def __init__(self, assistant, config: Config, n_moves: int, verbose: bool = False):
        self.assistant = assistant
        self.config = config
        self.n_moves = int(n_moves)
        self.verbose = bool(verbose)
        self.latencies: List[float] = []

    @abstractmethod
    def run(self, num_samples: int):
        ...

    # ----- MCQ answer parsing -----

    @staticmethod
    def parse_letter(text: Optional[str], options: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Return 'A'|'B'|'C'|'D' parsed from any of the answer formats, else None.

        If ``options`` is supplied and the model emitted a bare move (e.g. ``R'``),
        map that move back to its option letter.
        """
        if not text:
            return None
        m = _ANSWER_TAG_RE.search(text) or _ANSWER_COLON_RE.search(text) or _ANSWER_BRACKET_RE.search(text)
        if m:
            return m.group(1).upper()
        if options:
            m2 = _MOVE_TAG_RE.search(text)
            if m2:
                move = m2.group(1).upper()
                for letter, mv in options.items():
                    if mv.upper() == move:
                        return letter
        return None

    @staticmethod
    def parse_idk(text: Optional[str]) -> bool:
        return bool(text and _IDK_RE.search(text))

    @staticmethod
    def parse_yes_no(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = _YES_NO_RE.search(text)
        return m.group(1).capitalize() if m else None

    # ----- MCQ option generators -----

    @staticmethod
    def gen_mcq(
        correct: str,
        rng: random.Random,
        pool: Optional[Iterable[str]] = None,
        force_letter: Optional[str] = None,
    ) -> Tuple[Dict[str, str], str]:
        """Build a 4-option MCQ from a correct move plus 3 distractors.

        ``pool`` defaults to ``VirtualCube.AVAILABLE_MOVES``. If ``force_letter`` is
        set, the correct move is placed at that letter; otherwise placement is
        random.
        """
        from cube_bench.sim.cube_simulator import VirtualCube

        all_moves = list(pool) if pool is not None else list(VirtualCube.AVAILABLE_MOVES)
        candidates = [m for m in all_moves if m != correct]
        distractors = rng.sample(candidates, 3)
        rng.shuffle(distractors)

        if force_letter is None:
            opts = [correct] + distractors
            rng.shuffle(opts)
            d = dict(zip("ABCD", opts))
            gold = next(k for k, v in d.items() if v == correct)
            return d, gold

        letters = [L for L in "ABCD" if L != force_letter]
        d = {force_letter: correct}
        for L, mv in zip(letters, distractors):
            d[L] = mv
        return d, force_letter

    def gen_mcq_balanced(
        self,
        vc,
        teacher_move: str,
        rng: random.Random,
    ) -> Tuple[Dict[str, str], str]:
        """MCQ with exactly one progress-making distractor (when one exists),
        and the rest non-progress. Used by step-by-step."""
        from cube_bench.sim.cube_simulator import VirtualCube

        good = self.optimal_first_moves(vc) - {teacher_move}
        bad = [m for m in VirtualCube.AVAILABLE_MOVES if m != teacher_move and m not in good]

        picks: List[str] = []
        if good:
            picks.append(rng.choice(list(good)))
        need = 3 - len(picks)
        if len(bad) >= need:
            picks += rng.sample(bad, need)
        else:
            pool = [m for m in VirtualCube.AVAILABLE_MOVES if m != teacher_move and m not in picks]
            while len(picks) < 3:
                pick = rng.choice(pool)
                if pick not in picks:
                    picks.append(pick)

        opts = [teacher_move] + picks[:3]
        rng.shuffle(opts)
        d = dict(zip("ABCD", opts))
        gold = next(k for k, v in d.items() if v == teacher_move)
        return d, gold

    @staticmethod
    def gen_mcq_from_good(good_moves: Set[str], rng: random.Random) -> Tuple[Dict[str, str], str]:
        """Pick a correct move from ``good_moves`` (or any move if empty) and 3
        non-good distractors. Used by learning-curve."""
        from cube_bench.sim.cube_simulator import VirtualCube

        all_moves = list(VirtualCube.AVAILABLE_MOVES)
        if good_moves:
            correct = rng.choice(tuple(good_moves))
            pool = [m for m in all_moves if (m != correct and m not in good_moves)]
            if len(pool) >= 3:
                distractors = rng.sample(pool, 3)
            else:
                pool = [m for m in all_moves if m != correct]
                distractors = rng.sample(pool, k=min(3, len(pool)))
                while len(distractors) < 3:
                    pick = rng.choice(pool)
                    if pick not in distractors:
                        distractors.append(pick)
        else:
            correct = rng.choice(all_moves)
            pool = [m for m in all_moves if m != correct]
            distractors = rng.sample(pool, 3)

        opts = [correct] + distractors[:3]
        rng.shuffle(opts)
        d = dict(zip("ABCD", opts))
        gold = next(k for k, v in d.items() if v == correct)
        return d, gold

    # ----- Cube helpers -----

    @staticmethod
    def state_text(cube) -> str:
        """Public-API-first textual cube state, with a fallback to the internal
        pycuber __str__."""
        try:
            return str(cube)
        except Exception:
            try:
                return cube._cube.__str__()  # noqa: SLF001
            except Exception:
                return ""

    @staticmethod
    def optimal_first_moves(vc) -> Set[str]:
        """Neighbor moves that strictly decrease distance-to-solved (or solve)."""
        from cube_bench.sim.cube_simulator import VirtualCube

        if vc.is_solved():
            return set()
        baseline = vc.get_distance()
        good: Set[str] = set()
        for m in VirtualCube.AVAILABLE_MOVES:
            c = vc.clone()
            c.apply(m)
            if c.is_solved() or c.get_distance() < baseline:
                good.add(m)
        return good

    @staticmethod
    def move_makes_progress(vc, move: str) -> Tuple[bool, int, int]:
        """Return (decreases_distance, d_before, d_after)."""
        d0 = vc.get_distance()
        c = vc.clone()
        c.apply(move)
        if c.is_solved():
            return True, d0, 0
        d1 = c.get_distance()
        return (d1 < d0), d0, d1

    @staticmethod
    def teacher_first_move(scramble) -> Optional[str]:
        """First move of the inverse-scramble teacher path."""
        try:
            path = str(deepcopy(scramble).reverse()).split()
            return path[0] if path else None
        except Exception:
            return None

    @staticmethod
    def inverse_move(move: str) -> str:
        move = move.strip()
        if not move:
            return move
        if move.endswith("2"):
            return move
        if move.endswith("'"):
            return move[:-1]
        return move + "'"

    # ----- Assistant call -----

    def ask(
        self,
        user_prompt: str,
        system_prompt: str,
        image=None,
        max_new_tokens: int = 2**16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        track_latency: bool = True,
        **kwargs,
    ) -> str:
        """Single point of entry for ``assistant.generate``. Tracks latency in
        ``self.latencies`` when ``track_latency`` is True."""
        t0 = time.time()
        resp = self.assistant.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        if track_latency:
            self.latencies.append(time.time() - t0)
        return resp if isinstance(resp, str) else (getattr(resp, "text", None) or str(resp))

    # ----- Stats -----

    @staticmethod
    def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
        """95% Wilson score interval for a Bernoulli proportion ``p`` over ``n``."""
        if n <= 0 or not (0.0 <= p <= 1.0) or math.isnan(p):
            return (float("nan"), float("nan"))
        denom = 1.0 + (z * z) / n
        center = (p + (z * z) / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    # ----- Results -----

    def save(self, extra: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Persist a results payload with auto-filled metadata."""
        payload = {
            "model_name": self.assistant.get_name(),
            "test_type": self.test_type,
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        out_path = Path(self.config.results_dir) / (filename or f"{self.test_type}.json")
        save_results(out_path, payload)
        return out_path
