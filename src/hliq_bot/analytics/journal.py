from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Mapping


@dataclass(slots=True)
class SignalJournal:
    path: str
    default_context: Mapping[str, Any] | None = None
    _path: Path = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._path = p
        self._lock = threading.Lock()

    def write(self, event_type: str, signal_id: str, payload: dict[str, Any]) -> None:
        row = {
            "event_type": event_type,
            "signal_id": signal_id,
            **dict(self.default_context or {}),
            **payload,
        }
        line = json.dumps(row, separators=(",", ":"), sort_keys=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
