"""Filesystem layout and encodings for the artifacts evaluations persist."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    """Write *payload* to *path* as a standalone JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Overwrite *path* with one JSON object per line, keeping non-ASCII text literal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one ASCII-escaped JSON object to *path*."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Every non-blank line of *path*, parsed."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def timestamped_run_dir(root: Path, *parts: str) -> Path:
    """Create and return ``root/<parts>_<YYYYmmdd_HHMMSS>``."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = root / "_".join((*parts, stamp))
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_checkpoint(path: Path, required_field: str) -> Dict[int, Dict[str, Any]]:
    """Resumable per-sample records, skipping torn lines and superseded schemas."""
    results: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            result = rec["result"]
            if required_field not in result:
                logger.warning("Skipping checkpoint record from an incompatible schema")
                continue
            results[int(rec["sample_id"])] = result
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Skipping malformed checkpoint line: %s", e)
    return results
