"""BaseTest: prompting, parsing and scoring behaviour common to all evaluations."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from abc import ABC
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..config import Config
from ..io import save_results
from . import metrics
from .records import ItemRecord

logger = logging.getLogger(__name__)


# Accepted MCQ answer formats.
_ANSWER_TAG_RE = re.compile(r"<\s*ANSWER\s*>\s*([ABCD])\s*<\s*/\s*ANSWER\s*>", re.IGNORECASE)
_ANSWER_COLON_RE = re.compile(r"\bANSWER\s*[:=]\s*([ABCD])\b", re.IGNORECASE)
_IDK_RE = re.compile(
    r"(?:<ANSWER>\s*(IDK)\s*</ANSWER>)|(?:\bANSWER\s*[:=]\s*(?:IDK|E)\b)|(?:I\s*DON'?T\s*KNOW)",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(r"Answer:\s*(Yes|No)\b", re.IGNORECASE)


class BaseTest(ABC):
    """Shared prompting, parsing, scoring, and cube helpers for evaluations."""

    test_type: str = "base"

    def __init__(self, assistant, config: Config, n_moves: int = 3, verbose: bool = False):
        self.assistant = assistant
        self.config = config
        self.n_moves = int(n_moves)
        self.verbose = bool(verbose)
        self.latencies: List[float] = []
        self._logger = logging.getLogger(type(self).__module__)

    # ----- Run template -----

    def run(self, num_samples: int):
        """Run the evaluation over *num_samples* generated items."""
        self.setup(num_samples)
        records = self.collect(num_samples)
        payload = self.aggregate(records, num_samples)
        self.summary(payload, records)
        self.persist(payload, records)
        return self.result(payload, records)

    def collect(self, num_samples: int) -> List[ItemRecord]:
        """Run every item in order, keeping the records aggregation needs."""
        records: List[ItemRecord] = []
        for idx in self.progress(self.iter_indices(num_samples)):
            record = self.run_item(idx)
            if record is None:
                continue
            self.accumulate(record)
            records.append(record)
        return records

    # ----- Loop seams -----

    def iter_indices(self, num_samples: int) -> Iterable[int]:
        """The item indices to run, in order."""
        return range(num_samples)

    def progress(self, indices: Iterable[int], total: Optional[int] = None) -> Iterable[int]:
        """Wrap *indices* in this evaluation's progress bar."""
        from tqdm import tqdm
        return tqdm(indices, total=total, desc=self.desc())

    def desc(self) -> str:
        """The progress-bar label."""
        return self.test_type

    # ----- Hooks -----

    def setup(self, num_samples: int) -> None:
        """Prepare state that outlives a single item."""

    def run_item(self, idx: int) -> Optional[ItemRecord]:
        """Produce one scored record, or None to skip the item."""
        raise NotImplementedError

    def accumulate(self, record: ItemRecord) -> None:
        """Fold *record* into running per-test state."""

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        """Derive the payload that gets saved."""
        raise NotImplementedError

    def summary(  # pylint: disable=unused-argument
        self, payload: Dict[str, Any], records: List[ItemRecord]
    ) -> None:
        """Log the run; never compute a saved value here."""

    def persist(  # pylint: disable=unused-argument
        self, payload: Dict[str, Any], records: List[ItemRecord]
    ) -> Any:
        """Write the payload to disk."""
        return self.save(payload, self.result_filename())

    def result_filename(self) -> Optional[str]:
        """Override when the file name is not ``<test_type>.json``."""
        return None

    def result(  # pylint: disable=unused-argument
        self, payload: Dict[str, Any], records: List[ItemRecord]
    ) -> Any:
        """What ``run`` returns to its caller."""
        return None

    def _vlog(self, fmt: str, *args: Any) -> None:
        """Log at info level only while the run is verbose."""
        if self.verbose:
            self._logger.info(fmt, *args)

    # ----- MCQ answer parsing -----

    @staticmethod
    def parse_letter(text: Optional[str]) -> Optional[str]:
        """Parse an A-D answer from ``<ANSWER> X </ANSWER>`` or ``ANSWER: X``."""
        if not text:
            return None
        m = _ANSWER_TAG_RE.search(text) or _ANSWER_COLON_RE.search(text)
        return m.group(1).upper() if m else None

    @staticmethod
    def parse_idk(text: Optional[str]) -> bool:
        """True if *text* is an explicit abstention ("I don't know")."""
        return bool(text and _IDK_RE.search(text))

    @staticmethod
    def parse_yes_no(text: Optional[str]) -> Optional[str]:
        """Return 'Yes' or 'No' parsed from *text*, else None."""
        if not text:
            return None
        m = _YES_NO_RE.search(text)
        return m.group(1).capitalize() if m else None

    # ----- Deterministic item RNG -----

    @staticmethod
    def item_rng(*parts: Any) -> random.Random:
        """Return a per-item RNG seeded only by ``parts``.

        Every model therefore draws the same distractors, mismatches and option
        orders for a given item, independently of execution order or wall clock.
        """
        digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    # ----- MCQ option generators -----

    @staticmethod
    def gen_mcq(
        correct: str,
        rng: random.Random,
        pool: Optional[Iterable[str]] = None,
        force_letter: Optional[str] = None,
    ) -> Tuple[Dict[str, str], str]:
        """Build a four-option MCQ, optionally fixing the correct answer's letter."""
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
        *,
        good_moves: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, str], str]:
        """Build an MCQ with one progress-making distractor when available."""
        from cube_bench.sim.cube_simulator import VirtualCube

        oracle_moves = self.optimal_first_moves(vc) if good_moves is None else good_moves
        good = oracle_moves - {teacher_move}
        bad = [m for m in VirtualCube.AVAILABLE_MOVES if m != teacher_move and m not in good]

        picks: List[str] = []
        if good:
            picks.append(rng.choice(sorted(good)))
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
        """Choose from ``good_moves`` and add three non-good distractors."""
        from cube_bench.sim.cube_simulator import VirtualCube

        all_moves = list(VirtualCube.AVAILABLE_MOVES)
        if good_moves:
            correct = rng.choice(tuple(sorted(good_moves)))
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
        """Return the public cube text, falling back to the wrapped pycuber cube."""
        try:
            return str(cube)
        except Exception:
            try:
                return str(cube.raw)
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
    def teacher_path(scramble) -> List[str]:
        """Return the inverse scramble without mutating the caller's ``Formula``."""
        try:
            return str(deepcopy(scramble).reverse()).split()
        except Exception:
            return []

    @classmethod
    def teacher_first_move(cls, scramble) -> Optional[str]:
        """First move of the inverse-scramble teacher path."""
        path = cls.teacher_path(scramble)
        return path[0] if path else None

    @staticmethod
    def inverse_move(move: str) -> str:
        """Return the move that undoes *move* (half turns are their own inverse)."""
        move = move.strip()
        if not move:
            return move
        if move.endswith("2"):
            return move
        if move.endswith("'"):
            return move[:-1]
        return move + "'"

    # ----- Assistant call -----

    def _generate(self, user_prompt: str, system_prompt: str, **kwargs) -> Any:
        """Call the assistant with run-wide generation settings."""
        kwargs.setdefault("thinking_budget", getattr(self.config, "thinking_budget", None))
        return self.assistant.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    def ask(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        image=None,
        max_new_tokens: int = 2**16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        track_latency: bool = True,
        **kwargs,
    ) -> str:
        """Call ``assistant.generate`` and optionally record its latency."""
        t0 = time.time()
        resp = self._generate(
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
        return metrics.wilson_ci(p, n, z)

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


class SingleAskTest(BaseTest):
    """An evaluation whose every item is one prompt and one reply."""

    def run_item(self, idx: int) -> Optional[ItemRecord]:
        """Build, prompt, ask, parse and score item *idx*."""
        item = self.build_item(idx)
        if item is None:
            return None
        system_prompt, user_prompt = self.build_prompts(item)
        response = self.ask_item(item, system_prompt, user_prompt)
        return self.score_item(item, self.parse(response), response)

    # ----- Item generation -----

    def build_item(self, idx: int) -> Optional[Dict[str, Any]]:
        """Generate item *idx*, or None to skip it."""
        raise NotImplementedError

    # ----- Prompting -----

    def build_prompts(self, item: Dict[str, Any]) -> Tuple[str, str]:
        """The (system, user) prompt pair for *item*."""
        raise NotImplementedError

    def ask_kwargs(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generation arguments for *item*."""
        return {"image": item.get("image")}

    def ask_item(self, item: Dict[str, Any], system_prompt: str, user_prompt: str) -> str:
        """Send one item to the model."""
        return self.ask(
            user_prompt=user_prompt, system_prompt=system_prompt, **self.ask_kwargs(item)
        )

    # ----- Parsing -----

    def parse(self, response: Optional[str]) -> Any:
        """Extract this evaluation's answer from *response*."""
        return self.parse_letter(response)

    # ----- Scoring -----

    def score_item(
        self, item: Dict[str, Any], prediction: Any, response: Optional[str]
    ) -> ItemRecord:
        """Score one parsed answer."""
        raise NotImplementedError
