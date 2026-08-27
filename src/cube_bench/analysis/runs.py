"""Discovery and typed access for the artifacts an evaluation leaves on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from cube_bench.core.results import read_jsonl

RESULT_FILES = {
    "reconstruction": "reconstruction.json",
    "verification": "verification.json",
    "prediction": "solve_moves",
    "move_effect": "move_effect.json",
    "step_by_step": "step_by_step",
    "learning_curve": "learning_curve.json",
}

DEPTH_KEYS = ("n_moves", "n_moves_scrambled", "scramble_depth")


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Every result record in *path*.

    Evaluations append to their result file, so one file can hold several runs;
    callers select among them rather than assuming the file is one run.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    raise ValueError(f"{path}: expected an object or a list of objects, found {type(payload).__name__}")


def load_payload(path: Path) -> Dict[str, Any]:
    """The one result record in *path*, rejecting files that hold several."""
    records = load_records(path)
    if len(records) != 1:
        raise ValueError(
            f"{path}: holds {len(records)} records; select one with load_runs() instead of assuming the file is a run"
        )
    return records[0]


@dataclass(frozen=True)
class Run:
    """One completed evaluation: the record it wrote, and where in which file it sits."""

    path: Path
    payload: Dict[str, Any]
    record_index: int = 0

    @property
    def origin(self) -> str:
        """Provenance string naming the exact record, for captions and audit trails."""
        return f"{self.path}#{self.record_index}"

    @property
    def timestamp(self) -> str:
        """When the run finished, as recorded."""
        return str(self.payload.get("timestamp", ""))

    @property
    def model(self) -> str:
        """The model name as the runner recorded it."""
        return str(self.payload.get("model_name") or self.payload.get("model") or "unknown")

    @property
    def test_type(self) -> str:
        """The evaluation that produced this run."""
        return str(self.payload.get("test_type", "unknown"))

    @property
    def num_samples(self) -> int:
        """Episodes actually scored, which is the denominator a caption must quote."""
        for key in ("completed_samples", "num_samples", "n", "total_scrambles"):
            value = self.payload.get(key)
            if isinstance(value, int) and value > 0:
                return value
        raise KeyError(f"{self.path}: no sample count recorded")

    @property
    def depth(self) -> Optional[int]:
        """Scramble depth, read from whichever key this evaluation uses."""
        for source in (self.payload, self.payload.get("meta", {})):
            if not isinstance(source, dict):
                continue
            for key in DEPTH_KEYS:
                if isinstance(source.get(key), int):
                    return int(source[key])
        return None

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup through ``payload``, returning *default* on the first missing key."""
        node: Any = self.payload
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def _result_file(path: Path) -> Path:
    """The result file at *path*, or the single result file inside that directory."""
    path = Path(path)
    if not path.is_dir():
        return path
    candidates = sorted(
        child
        for child in path.glob("*.json")
        if child.name != "summary.json" and not child.name.startswith(".")
    )
    if len(candidates) != 1:
        names = ", ".join(c.name for c in candidates) or "none"
        raise ValueError(f"{path}: expected one result file, found {names}")
    return candidates[0]


def load_runs(path: Path) -> List[Run]:
    """Every run recorded in one result file, in the order it was appended."""
    file = _result_file(path)
    return [Run(path=file, payload=rec, record_index=i) for i, rec in enumerate(load_records(file))]


def load_run(path: Path) -> Run:
    """The one run in a result file, rejecting files that appended several."""
    runs = load_runs(path)
    if len(runs) != 1:
        stamps = ", ".join(f"#{r.record_index} n={r.payload.get('num_samples')} {r.timestamp}" for r in runs)
        raise ValueError(f"{runs[0].path}: holds {len(runs)} runs ({stamps}); pick one with load_runs()")
    return runs[0]


def find_runs(root: Path, test: str) -> List[Run]:
    """Every run of *test* under *root*, flattened across appended records and ordered by path."""
    if test not in RESULT_FILES:
        raise KeyError(f"unknown test {test!r}; expected one of {', '.join(sorted(RESULT_FILES))}")
    stem = RESULT_FILES[test]
    pattern = f"**/{stem}" if stem.endswith(".json") else f"**/{stem}*.json"
    return [run for p in sorted(Path(root).glob(pattern)) for run in load_runs(p)]


def select(
    runs: Sequence[Run],
    *,
    model: Optional[str] = None,
    depth: Optional[int] = None,
    num_samples: Optional[int] = None,
    prompt_type: Optional[str] = None,
) -> List[Run]:
    """The runs matching every constraint given, so a table cell names its source exactly."""
    out = list(runs)
    if model is not None:
        out = [r for r in out if r.model == model]
    if depth is not None:
        out = [r for r in out if r.depth == depth]
    if num_samples is not None:
        out = [r for r in out if r.payload.get("num_samples") == num_samples]
    if prompt_type is not None:
        out = [r for r in out if r.payload.get("prompt_type") == prompt_type]
    return out


def exactly_one(runs: Sequence[Run], what: str) -> Run:
    """The single run in *runs*, with a message naming the candidates when it is not unique."""
    if len(runs) == 1:
        return runs[0]
    if not runs:
        raise LookupError(f"no run found for {what}")
    origins = ", ".join(f"{r.origin} (n={r.payload.get('num_samples')})" for r in runs)
    raise LookupError(f"{len(runs)} runs match {what}; disambiguate: {origins}")


@dataclass(frozen=True)
class ReflectionRun:
    """One reflection arm: its summary plus the per-item reflection and re-answer logs.

    The two JSONL logs are read on first use, so an inventory that only needs the
    summary never pays for them.
    """

    path: Path
    summary: Dict[str, Any]

    @cached_property
    def reanswers(self) -> List[Dict[str, Any]]:
        """Per-item re-answer records."""
        return read_jsonl(self.path / "reanswers.jsonl")

    @cached_property
    def reflections(self) -> List[Dict[str, Any]]:
        """Per-item reflection records."""
        return read_jsonl(self.path / "reflections.jsonl")

    @property
    def model(self) -> str:
        """The model name as the runner recorded it."""
        return str(self.summary.get("model", "unknown"))

    @property
    def arm(self) -> str:
        """The (reveal, assert) cell this run occupies, as a compact label."""
        reveal = "T" if self.summary.get("reveal_choice") else "F"
        return f"reveal={reveal},assert={self.summary.get('assert_incorrect', 'unknown')}"

    @property
    def indices(self) -> List[int]:
        """Item indices in re-answer order."""
        return [int(row["index"]) for row in self.reanswers]

    def final_correct(self) -> Dict[int, bool]:
        """Per item, whether the post-reflection answer matched gold."""
        return {int(r["index"]): r.get("pred") == r.get("gold") for r in self.reanswers}

    def initial_correct(self) -> Dict[int, bool]:
        """Per item, whether the pre-reflection answer matched gold."""
        return {int(r["index"]): bool(r.get("initially_correct")) for r in self.reflections}

    def prior_choices(self) -> Dict[int, Optional[str]]:
        """Per item, the earlier choice shown back to the model, absent when it was hidden."""
        return {int(r["index"]): r.get("prior_answer") for r in self.reflections}

    def final_choices(self) -> Dict[int, Optional[str]]:
        """Per item, the letter the model settled on."""
        return {int(r["index"]): r.get("pred") for r in self.reanswers}


def load_reflection(path: Path) -> ReflectionRun:
    """Load a reflection run directory holding ``summary.json`` and its two JSONL logs."""
    path = Path(path)
    return ReflectionRun(path=path, summary=load_payload(path / "summary.json"))


def find_reflections(root: Path) -> List[ReflectionRun]:
    """Every reflection run directory under *root*, ordered by path."""
    return [load_reflection(p.parent) for p in sorted(Path(root).glob("**/summary.json"))]


def paired_items(*runs: ReflectionRun) -> List[int]:
    """Item indices present in every run, so a contrast stays item-paired."""
    if not runs:
        return []
    shared = set(runs[0].indices)
    for run in runs[1:]:
        shared &= set(run.indices)
    return sorted(shared)


def vectors(run: ReflectionRun, indices: Sequence[int]) -> List[bool]:
    """Final correctness for *indices*, in the order given."""
    correct = run.final_correct()
    return [correct[i] for i in indices]


def iter_steps(run: Run) -> Iterator[Dict[str, Any]]:
    """Every per-step record of a step-by-step run, flattened across episodes."""
    for sample in run.payload.get("samples", []):
        yield from sample.get("steps_data", [])
