from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading

from hliq_bot.models import MarketEvent


@dataclass(slots=True)
class MarketCaptureWriter:
    path: str
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
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
