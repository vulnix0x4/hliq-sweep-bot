from __future__ import annotations

from dataclasses import dataclass

from hliq_bot.models import Bar, TradeEvent


@dataclass(slots=True)
class _WorkingBar:
    start_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    notional: float
    spread_sum: float
    spread_count: int


class BarBuilder:
    def __init__(self, timeframe_sec: int) -> None:
        self.timeframe_ms = timeframe_sec * 1000
        self._current: _WorkingBar | None = None

    def on_trade(self, trade: TradeEvent, spread_bps: float) -> list[Bar]:
        closed: list[Bar] = []
        bucket_start = (trade.ts_ms // self.timeframe_ms) * self.timeframe_ms

        if self._current is None:
            self._current = self._new_working_bar(bucket_start, trade, spread_bps)
            return closed

        if bucket_start != self._current.start_ms:
            closed.append(self._finalize(self._current))
            self._current = self._new_working_bar(bucket_start, trade, spread_bps)
            return closed

        wb = self._current
        wb.high = max(wb.high, trade.price)
        wb.low = min(wb.low, trade.price)
        wb.close = trade.price
        wb.volume += trade.size
        wb.trade_count += 1
        wb.notional += trade.price * trade.size
        wb.spread_sum += spread_bps
        wb.spread_count += 1
        return closed

    def flush(self) -> Bar | None:
        if self._current is None:
            return None
        bar = self._finalize(self._current)
        self._current = None
        return bar

    def _new_working_bar(self, bucket_start: int, trade: TradeEvent, spread_bps: float) -> _WorkingBar:
        return _WorkingBar(
            start_ms=bucket_start,
            open=trade.price,
            high=trade.price,
            low=trade.price,
            close=trade.price,
            volume=trade.size,
            trade_count=1,
            notional=trade.price * trade.size,
            spread_sum=spread_bps,
            spread_count=1,
        )

    def _finalize(self, wb: _WorkingBar) -> Bar:
        end_ms = wb.start_ms + self.timeframe_ms
        vwap = wb.notional / wb.volume if wb.volume > 0 else wb.close
        avg_spread_bps = wb.spread_sum / wb.spread_count if wb.spread_count > 0 else 0.0
        return Bar(
            start_ms=wb.start_ms,
            end_ms=end_ms,
            open=wb.open,
            high=wb.high,
            low=wb.low,
            close=wb.close,
            volume=wb.volume,
            trade_count=wb.trade_count,
            vwap=vwap,
            avg_spread_bps=avg_spread_bps,
        )

