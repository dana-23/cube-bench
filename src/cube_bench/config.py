"""Run configuration shared by the CLI, the orchestrator and every evaluation."""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Filesystem paths and run-wide limits shared by every evaluation."""

    dataset_path: Path
    prompts_path: Path
    results_dir: Path
    max_scramble_len: int = 10
    batch_size: int = 25
    n_moves: int | None = None
    thinking_budget: int | None = None
