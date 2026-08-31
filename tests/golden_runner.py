"""Execute one evaluation case and capture its output files as scrubbed text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from golden_cases import SEED, Case, make_config
from mock_assistant import MockAssistant

TEXT_SUFFIXES = (".json", ".jsonl")

CONTRACT_KEYS = (
    "mode", "had_image", "history_turns", "reference",
    "max_new_tokens", "temperature", "top_p", "do_sample",
)

_LATENCY_KEYS = (
    "avg_latency_s", "avg_latency_ms", "latency", "latency_ms",
    "delta_latency_ms_total", "delta_latency_ms_per_item",
)
_LATENCY_LIST_KEYS = ("latencies", "y_latency_s")
_PATH_KEYS = ("plot_path", "run_dir")

_SCRUBBERS = (
    (re.compile(r'("timestamp":\s*")[^"]*"'), r'\1<TIMESTAMP>"'),
    (re.compile(rf'("(?:{"|".join(_LATENCY_KEYS)})":\s*)-?[0-9][0-9.eE+-]*'), r"\1<LATENCY>"),
    (re.compile(rf'("(?:{"|".join(_LATENCY_LIST_KEYS)})":\s*\[)[^\]]*(\])'), r"\1<LATENCY>\2"),
    (re.compile(rf'("(?:{"|".join(_PATH_KEYS)})":\s*")[^"]*"'), r'\1<PATH>"'),
)

_RUN_DIR = re.compile(r"(reflection/)[^/]+/")


class RecordingAssistant(MockAssistant):
    """MockAssistant that also keeps the exact prompts it was sent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts: List[Tuple[str, str]] = []

    def generate(self, user_prompt, system_prompt="", **kwargs):
        """Record the prompt pair, then reply as MockAssistant would."""
        self.prompts.append((system_prompt, user_prompt))
        return super().generate(user_prompt, system_prompt, **kwargs)


def make_assistant(**kwargs) -> RecordingAssistant:
    """Build the assistant every golden case shares."""
    return RecordingAssistant(accuracy=0.6, seed=SEED, name="mock-model", **kwargs)


def generation_contract(assistant: MockAssistant) -> List[Dict]:
    """The per-call generation parameters, which saved output never records."""
    return [{key: call[key] for key in CONTRACT_KEYS} for call in assistant.calls]


def prompt_snapshot(assistant: RecordingAssistant, limit: int = 3) -> str:
    """The first *limit* prompt pairs, rendered as a comparable fixture."""
    blocks = []
    for index, (system_prompt, user_prompt) in enumerate(assistant.prompts[:limit]):
        blocks.append(
            f"===== call {index} system =====\n{system_prompt}\n"
            f"===== call {index} user =====\n{user_prompt}"
        )
    return "\n".join(blocks) + "\n"


def scrub(text: str, root: Path) -> str:
    """Replace wall-clock and filesystem-dependent values with stable markers."""
    text = text.replace(str(root), "<TMP>")
    for pattern, replacement in _SCRUBBERS:
        text = pattern.sub(replacement, text)
    return text


def relative_name(path: Path, root: Path) -> str:
    """Path relative to *root*, with reflection's timestamped run directory folded away."""
    return _RUN_DIR.sub(r"\1RUN/", path.relative_to(root).as_posix())


def run_case(
    case: Case, root: Path, assistant: Optional[MockAssistant] = None
) -> Dict[str, str]:
    """Run *case* into *root* and return every file it wrote, scrubbed."""
    root.mkdir(parents=True, exist_ok=True)
    assistant = assistant or make_assistant()
    task = case.build(assistant, make_config(root), root)
    task.run(case.samples)

    captured: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = relative_name(path, root)
        if path.suffix in TEXT_SUFFIXES:
            captured[name] = scrub(path.read_text(encoding="utf-8"), root)
        else:
            size = "empty" if path.stat().st_size == 0 else "non-empty"
            captured[name] = f"<binary {size}>"
    return captured
