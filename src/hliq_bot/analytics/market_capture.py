from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading

from hliq_bot.models import MarketEvent


@dataclass(slots=True)
class MarketCaptureWriter:
    path: str
    max_bytes: int = 0
    backups: int = 3
    _path: Path = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = Path(self.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: MarketEvent) -> None:
        row: dict[str, object] = {
            "kind": event.kind,
            "ts_ms": event.ts_ms,
        }
        coin = (event.coin or "").strip().upper()
        if coin:
            row["coin"] = coin
        if event.trade is not None:
            row.update(
                {
                    "price": event.trade.price,
                    "size": event.trade.size,
                    "side": event.trade.side,
                }
            )
        if event.book is not None:
            row.update(
                {
                    "best_bid": event.book.best_bid,
                    "best_ask": event.book.best_ask,
                    "bid_size": event.book.bid_size,
                    "ask_size": event.book.ask_size,
                }
            )
        line = json.dumps(row, separators=(",", ":"), sort_keys=False)
        with self._lock:
            self._rotate_if_needed()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _rotate_if_needed(self) -> None:
        max_bytes = max(0, int(self.max_bytes))
        if max_bytes <= 0:
            return
        try:
            if not self._path.exists() or self._path.stat().st_size < max_bytes:
                return
        except OSError:
            return

        backups = max(1, int(self.backups))
        oldest = self._path.with_name(f"{self._path.name}.{backups}")
        try:
            if oldest.exists():
                oldest.unlink()
            for i in range(backups - 1, 0, -1):
                src = self._path.with_name(f"{self._path.name}.{i}")
                dst = self._path.with_name(f"{self._path.name}.{i + 1}")
                if src.exists():
                    src.replace(dst)
            self._path.replace(self._path.with_name(f"{self._path.name}.1"))
        except OSError:
            return
