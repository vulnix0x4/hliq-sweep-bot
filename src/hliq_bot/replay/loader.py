from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from hliq_bot.models import BookTopEvent, MarketEvent, TradeEvent


def load_market_events(path: str) -> Iterator[MarketEvent]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            kind = str(row.get("kind", "")).strip().lower()
            ts_ms = _safe_int(row.get("ts_ms"))
            if ts_ms <= 0:
                continue

            if kind == "trade":
                price = _safe_float(row.get("price"))
                size = _safe_float(row.get("size"))
                if price <= 0 or size <= 0:
                    continue
                yield MarketEvent(
                    kind="trade",
                    ts_ms=ts_ms,
                    trade=TradeEvent(
                        ts_ms=ts_ms,
                        price=price,
                        size=size,
                        side=str(row.get("side", "unknown")),
                    ),
                )
                continue

            if kind == "book":
                best_bid = _safe_float(row.get("best_bid"))
                best_ask = _safe_float(row.get("best_ask"))
                if best_bid <= 0 or best_ask <= 0:
                    continue
                yield MarketEvent(
                    kind="book",
                    ts_ms=ts_ms,
                    book=BookTopEvent(
                        ts_ms=ts_ms,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        bid_size=_safe_float(row.get("bid_size")),
                        ask_size=_safe_float(row.get("ask_size")),
                    ),
                )


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
