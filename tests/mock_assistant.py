"""Deterministic, task-aware ``ModelAssistant`` stand-in for evaluation tests."""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

LETTERS = "ABCD"
MOVE_EFFECT_LABELS = ("DECREASE", "NO_CHANGE", "INCREASE")
GRID_COLORS = ("White", "Yellow", "Red", "Orange", "Blue", "Green")

MODE_MCQ = "mcq"
MODE_YES_NO = "yes_no"
MODE_MOVE_EFFECT = "move_effect"
MODE_GRID = "grid"

AnswerKey = Callable[[str, str], Any]


def detect_mode(system_prompt: str, user_prompt: str) -> str:
    """Infer the reply format, checking task-specific cues before generic MCQ prose."""
    text = f"{system_prompt}\n{user_prompt}".lower()
    if "decrease|no_change|increase" in text:
        return MODE_MOVE_EFFECT
    if "row 1:" in text:
        return MODE_GRID
    if "answer: yes" in text:
        return MODE_YES_NO
    return MODE_MCQ


class PromptAnswerKey:
    """Thread-safe prompt-to-gold registry independent of call order."""

    def __init__(self) -> None:
        self._golds: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def record(self, user_prompt: str, gold: Any) -> None:
        """Associate *gold* with the item whose prompt is *user_prompt*."""
        with self._lock:
            self._golds[user_prompt] = gold

    def __call__(self, user_prompt: str, system_prompt: str = "") -> Any:
        with self._lock:
            return self._golds.get(user_prompt)

    def __len__(self) -> int:
        with self._lock:
            return len(self._golds)


class MockAssistant:
    """Tunable, duck-typed ``ModelAssistant`` replacement with seeded decisions."""

    def __init__(
        self,
        *,
        accuracy: float = 1.0,
        seed: int = 0,
        name: str = "mock-model",
        answer_key: Optional[AnswerKey] = None,
        mode: Optional[str] = None,
        guess_includes_gold: bool = True,
        garbage_rate: float = 0.0,
        idk_rate: float = 0.0,
    ) -> None:
        for label, value in (("accuracy", accuracy), ("garbage_rate", garbage_rate),
                             ("idk_rate", idk_rate)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value!r}")
        self.accuracy = accuracy
        self.seed = seed
        self.name = name
        self.answer_key = answer_key
        self.mode = mode
        self.guess_includes_gold = guess_includes_gold
        self.garbage_rate = garbage_rate
        self.idk_rate = idk_rate
        self.calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    # Assistant interface

    def get_name(self) -> str:
        """The name evaluations stamp into results and run directories."""
        return self.name

    def cleanup(self) -> None:
        """No resources to release; present so the orchestrator can call it."""

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        *,
        max_new_tokens: int = 128,
        image: Any = None,
        reference: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        history: Optional[List[Dict[str, Any]]] = None,
        **_: Any,
    ) -> str:
        """Return one simulated reply, formatted for whichever task is asking."""
        mode = self.mode or detect_mode(system_prompt, user_prompt)
        gold = self.answer_key(user_prompt, system_prompt) if self.answer_key else None
        reply = self._reply(mode, gold, user_prompt)
        with self._lock:
            self.calls.append({
                "mode": mode,
                "gold": gold,
                "reply": reply,
                "had_image": image is not None,
                "history_turns": len(history or []),
                "reference": reference,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
            })
        return reply

    # Deterministic decisions

    def _unit(self, *parts: str) -> float:
        """A stable pseudo-random float in [0, 1) derived from *parts*."""
        payload = "\x1f".join((str(self.seed), *parts)).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2 ** 64

    def _pick(self, choices: Sequence[Any], *parts: str) -> Any:
        return choices[min(int(self._unit(*parts) * len(choices)), len(choices) - 1)]

    def _choose(
        self,
        pool: Sequence[Any],
        gold: Any,
        prompt: str,
        tag: str,
        key: Callable[[Any], Any] = lambda value: value,
    ) -> Any:
        """Return *gold* with probability ``accuracy``, else a guess from *pool*."""
        if gold is not None and self._unit("correct", tag, prompt) < self.accuracy:
            return gold
        choices = list(pool)
        if gold is not None and not self.guess_includes_gold:
            choices = [c for c in choices if key(c) != key(gold)] or list(pool)
        return self._pick(choices, "guess", tag, prompt)

    # Reply formatting

    def _reply(self, mode: str, gold: Any, prompt: str) -> str:
        if self.garbage_rate and self._unit("garbage", prompt) < self.garbage_rate:
            return "I considered the cube carefully but cannot commit to an answer."
        if mode == MODE_MOVE_EFFECT:
            return self._move_effect_reply(gold, prompt)
        if mode == MODE_GRID:
            return self._grid_reply(gold, prompt)
        if mode == MODE_YES_NO:
            return f"Answer: {self._choose(('Yes', 'No'), gold, prompt, 'yesno')}"
        if self.idk_rate and self._unit("idk", prompt) < self.idk_rate:
            return "<ANSWER> IDK </ANSWER>"
        return f"<ANSWER> {self._choose(LETTERS, gold, prompt, 'mcq')} </ANSWER>"

    def _move_effect_reply(self, gold: Any, prompt: str) -> str:
        golds = gold if isinstance(gold, dict) else {}
        lines = [
            f"<{L}> {self._choose(MOVE_EFFECT_LABELS, golds.get(L), prompt, f'effect:{L}')} </{L}>"
            for L in LETTERS
        ]
        return "\n".join(lines)

    def _grid_reply(self, gold: Any, prompt: str) -> str:
        rows = []
        for r in range(3):
            cells = [
                self._choose(
                    GRID_COLORS,
                    gold[r][c] if gold else None,
                    prompt,
                    f"grid:{r}:{c}",
                    key=lambda value: str(value)[:1].upper(),
                )
                for c in range(3)
            ]
            rows.append(f"Row {r + 1}: [{', '.join(str(c) for c in cells)}]")
        return "Answer:\n" + "\n".join(rows) + "\nAnswer verified for correctness."

    # Test introspection

    def expected_accuracy(self, n_options: int = len(LETTERS)) -> float:
        """Include chance hits when wrong guesses may still select the gold option."""
        if not self.guess_includes_gold:
            return self.accuracy
        return self.accuracy + (1.0 - self.accuracy) / n_options
