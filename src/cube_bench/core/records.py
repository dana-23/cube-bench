"""The per-item result an evaluation produces, independent of how it is saved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ItemRecord:
    """One scored item.

    ``saved`` holds the exact dict this item contributes to a ``per_item`` list,
    built explicitly by the evaluation so that key names and key order stay
    under its control rather than following field order here.
    """

    index: int
    gold: Any = None
    pred: Any = None
    correct: bool = False
    parsed: bool = False
    response: Optional[str] = None
    latency: Optional[float] = None
    saved: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
