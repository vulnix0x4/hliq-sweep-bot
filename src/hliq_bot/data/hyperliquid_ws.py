from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

try:
    import websockets
except ImportError:  # pragma: no cover - exercised in runtime environments missing deps
    websockets = None

from hliq_bot.config import FeedConfig
from hliq_bot.models import BookTopEvent, MarketEvent, TradeEvent

log = logging.getLogger(__name__)


class HyperliquidWsClient:
    def __init__(self, config: FeedConfig) -> None:
        self.config = config
        self.last_message_ms: int = 0

    async def stream(self) -> AsyncGenerator[MarketEvent, None]:
        if websockets is None:
            raise RuntimeError("Missing dependency: install 'websockets' to use HyperliquidWsClient")

        while True:
            try:
                async with websockets.connect(
                    self.config.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=2**23,
                ) as ws:
                    await self._subscribe(ws)
                    log.info("Connected to Hyperliquid WS, coin=%s", self.config.coin)

                    async for message in ws:
                        self.last_message_ms = int(time.time() * 1000)
                        for evt in self._parse_message(message):
                            yield evt

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("WS disconnected (%s). Reconnecting in %.1fs", exc, self.config.reconnect_backoff_sec)
                await asyncio.sleep(self.config.reconnect_backoff_sec)

    async def _subscribe(self, ws: Any) -> None:
        subs: list[dict[str, Any]] = [
            {"method": "subscribe", "subscription": {"type": "trades", "coin": self.config.coin}},
            {"method": "subscribe", "subscription": {"type": "l2Book", "coin": self.config.coin}},
        ]
        if self.config.subscribe_user and self.config.user_address:
            subs.append(
                {
                    "method": "subscribe",
                    "subscription": {
                        "type": "userEvents",
                        "user": self.config.user_address,
                    },
                }
            )

        for payload in subs:
            await ws.send(json.dumps(payload))

    def _parse_message(self, raw_message: str) -> list[MarketEvent]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return []

        events: list[MarketEvent] = []
        channel = payload.get("channel", "")
        data = payload.get("data")
        now_ms = int(time.time() * 1000)

        if "trades" in channel.lower():
            for t in self._extract_trades(data):
                price = _to_float(t.get("px", t.get("price")))
                size = _to_float(t.get("sz", t.get("size")))
                if price <= 0 or size <= 0:
                    continue
                ts_ms = int(t.get("time", t.get("ts", now_ms)))
                side_raw = str(t.get("side", "")).strip().lower()
                side = (
                    "buy"
                    if side_raw in {"buy", "b", "bid"}
                    else "sell"
                    if side_raw in {"sell", "s", "ask", "a"}
                    else "unknown"
                )
                trade = TradeEvent(ts_ms=ts_ms, price=price, size=size, side=side)
                events.append(MarketEvent(kind="trade", ts_ms=ts_ms, trade=trade, raw=payload))

        if "l2" in channel.lower():
            book = self._extract_book_top(data, now_ms)
            if book is not None:
                events.append(MarketEvent(kind="book", ts_ms=book.ts_ms, book=book, raw=payload))

        return events

    def _extract_trades(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return data
            return []
        if isinstance(data, dict):
            if isinstance(data.get("trades"), list):
                return [x for x in data["trades"] if isinstance(x, dict)]
            if isinstance(data.get("data"), list):
                return [x for x in data["data"] if isinstance(x, dict)]
        return []

    def _extract_book_top(self, data: Any, now_ms: int) -> BookTopEvent | None:
        levels = None
        if isinstance(data, dict):
            levels = data.get("levels")
        if not isinstance(levels, list) or len(levels) < 2:
            return None
        bids = levels[0] if isinstance(levels[0], list) else []
        asks = levels[1] if isinstance(levels[1], list) else []
        if not bids or not asks:
            return None

        best_bid = _to_float((bids[0] if isinstance(bids[0], dict) else {}).get("px"))
        bid_size = _to_float((bids[0] if isinstance(bids[0], dict) else {}).get("sz"))
        best_ask = _to_float((asks[0] if isinstance(asks[0], dict) else {}).get("px"))
        ask_size = _to_float((asks[0] if isinstance(asks[0], dict) else {}).get("sz"))
        if best_bid <= 0 or best_ask <= 0:
            return None

        ts_ms = int(data.get("time", now_ms)) if isinstance(data, dict) else now_ms
        return BookTopEvent(
            ts_ms=ts_ms,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
        )


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
