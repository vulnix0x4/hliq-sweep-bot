"""Build a compact, LLM-friendly view of a coin's market state.

The output is a dict serialized as JSON in the user message. We optimize for:
- Information density (every field has decision-relevant value)
- Stable schema (model learns the shape across many turns)
- Low token count (cost / latency)
- Numeric clarity (bps, ratios, ms, not raw timestamps)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import statistics
from typing import Any

from hliq_bot.models import Bar, Side


@dataclass(slots=True)
class CoinContext:
    """Snapshot of a coin's state at decision time."""
    coin: str
    now_ms: int
    now_utc: str               # human-readable for the LLM
    session: str               # asia/eu/us/late
    last_price: float
    last_spread_bps: float
    recent_bars: list[dict]    # OHLCV summary, oldest -> newest
    flow_bias_5m: float        # signed buyer-vs-seller volume / total, [-1, +1]
    trade_count_5m: int
    realized_vol_5m_bps: float # stdev of 5m returns, in bps
    range_5m_bps: float        # (high-low) over last 5m, in bps
    range_30m_bps: float
    vwap_session: float | None # session VWAP
    vwap_distance_bps: float | None  # signed distance from current price
    open_position: dict | None # current position summary, or None
    account_equity: float
    daily_pnl: float
    daily_r: float
    recent_outcomes: list[dict]   # last N closed AI trades for self-reflection

    def to_prompt_dict(self) -> dict[str, Any]:
        """JSON-serializable dict that goes verbatim into the user message."""
        return {
            "coin": self.coin,
            "now_utc": self.now_utc,
            "session": self.session,
            "last_price": round(self.last_price, 6),
            "last_spread_bps": round(self.last_spread_bps, 2),
            "vwap_session": round(self.vwap_session, 6) if self.vwap_session else None,
            "vwap_distance_bps": round(self.vwap_distance_bps, 2) if self.vwap_distance_bps is not None else None,
            "flow_bias_5m": round(self.flow_bias_5m, 3),
            "trade_count_5m": self.trade_count_5m,
            "realized_vol_5m_bps": round(self.realized_vol_5m_bps, 2),
            "range_5m_bps": round(self.range_5m_bps, 1),
            "range_30m_bps": round(self.range_30m_bps, 1),
            "recent_bars": self.recent_bars,
            "open_position": self.open_position,
            "account": {
                "equity": round(self.account_equity, 4),
                "daily_pnl": round(self.daily_pnl, 4),
                "daily_r": round(self.daily_r, 3),
            },
            "recent_outcomes": self.recent_outcomes,
        }


def _session_from_hour(hour_utc: int) -> str:
    if hour_utc < 7:
        return "asia"
    if hour_utc < 13:
        return "eu"
    if hour_utc < 22:
        return "us"
    return "late"


def _bar_to_dict(bar: Bar) -> dict[str, Any]:
    """Compact OHLCV dict. Times in HH:MM:SS UTC. Prices rounded to 6 sig figs.

    `t` is the bar start time. `o/h/l/c` are OHLC. `v` is volume.
    `range_bps` is (high-low)/close*10000 — saves the LLM from computing it.
    """
    rng_bps = ((bar.high - bar.low) / bar.close) * 10000 if bar.close else 0.0
    body_bps = ((bar.close - bar.open) / bar.open) * 10000 if bar.open else 0.0
    return {
        "t": datetime.fromtimestamp(bar.start_ms / 1000.0, tz=timezone.utc).strftime("%H:%M"),
        "o": round(bar.open, 6),
        "h": round(bar.high, 6),
        "l": round(bar.low, 6),
        "c": round(bar.close, 6),
        "v": round(bar.volume, 2),
        "n": bar.trade_count,
        "body_bps": round(body_bps, 1),
        "range_bps": round(rng_bps, 1),
    }


def _flow_bias(signed_flow: list[tuple[int, float]], window_ms: int = 5 * 60 * 1000, now_ms: int = 0) -> tuple[float, int]:
    """Sum signed flow over the last N ms. Returns (bias, trade_count)."""
    cutoff = now_ms - window_ms
    relevant = [(ts, sz) for ts, sz in signed_flow if ts >= cutoff]
    if not relevant:
        return 0.0, 0
    total_signed = sum(sz for _, sz in relevant)
    total_abs = sum(abs(sz) for _, sz in relevant)
    if total_abs <= 0:
        return 0.0, len(relevant)
    return total_signed / total_abs, len(relevant)


def _realized_vol_bps(prices: list[tuple[int, float]], window_ms: int = 5 * 60 * 1000, now_ms: int = 0) -> float:
    """Stdev of log-returns over the window, expressed as bps."""
    cutoff = now_ms - window_ms
    relevant = [p for ts, p in prices if ts >= cutoff and p > 0]
    if len(relevant) < 3:
        return 0.0
    rets = []
    for i in range(1, len(relevant)):
        if relevant[i - 1] > 0:
            rets.append((relevant[i] - relevant[i - 1]) / relevant[i - 1])
    if len(rets) < 2:
        return 0.0
    return statistics.stdev(rets) * 10000


def _range_bps(prices: list[tuple[int, float]], window_ms: int, now_ms: int) -> float:
    """(max - min) / current_price over the window, in bps."""
    cutoff = now_ms - window_ms
    relevant = [p for ts, p in prices if ts >= cutoff and p > 0]
    if len(relevant) < 2:
        return 0.0
    hi, lo = max(relevant), min(relevant)
    last = relevant[-1]
    if last <= 0:
        return 0.0
    return ((hi - lo) / last) * 10000


def build_coin_context(
    worker: Any,                # bot.CoinWorker (duck-typed to avoid circular import)
    *,
    bars: list[Bar],
    now_ms: int,
    account_equity: float,
    daily_pnl: float,
    daily_r: float,
    recent_outcomes: list[dict],
    context_bars: int = 30,
) -> CoinContext:
    """Pull state from a CoinWorker into a CoinContext."""
    coin = worker.coin
    last_price = (worker.last_best_bid + worker.last_best_ask) / 2.0 if (worker.last_best_bid and worker.last_best_ask) else 0.0
    if last_price <= 0 and bars:
        last_price = bars[-1].close

    recent_bars = [_bar_to_dict(b) for b in bars[-context_bars:]]

    flow_bias, trade_count_5m = _flow_bias(list(worker.recent_signed_flow), window_ms=5 * 60 * 1000, now_ms=now_ms)
    realized_vol_5m = _realized_vol_bps(list(worker.recent_trade_prices), window_ms=5 * 60 * 1000, now_ms=now_ms)
    range_5m = _range_bps(list(worker.recent_trade_prices), window_ms=5 * 60 * 1000, now_ms=now_ms)
    range_30m = _range_bps(list(worker.recent_trade_prices), window_ms=30 * 60 * 1000, now_ms=now_ms)

    vwap_session = None
    vwap_distance = None
    try:
        vwap_session = float(worker.vwap_tracker.session_vwap()) or None
        if vwap_session and last_price > 0:
            vwap_distance = ((last_price - vwap_session) / vwap_session) * 10000
    except Exception:
        vwap_session = None

    position = getattr(worker.executor, "position", None)
    pos_dict: dict | None = None
    if position is not None:
        unrealized = 0.0
        try:
            if position.side == Side.LONG:
                unrealized = (last_price - position.entry_price) * position.qty_remaining
            else:
                unrealized = (position.entry_price - last_price) * position.qty_remaining
        except Exception:
            unrealized = 0.0
        risk_d = max(getattr(position, "risk_dollars", 0) or 0, 1e-9)
        pos_dict = {
            "side": position.side.value,
            "entry_price": round(position.entry_price, 6),
            "stop_price": round(position.stop_price, 6),
            "tp1_price": round(position.tp1_price, 6),
            "tp2_price": round(position.tp2_price, 6),
            "qty_remaining": round(position.qty_remaining, 8),
            "opened_ms_ago": int((now_ms - position.opened_ms) / 1000),
            "tp1_filled": bool(getattr(position, "tp1_filled", False)),
            "unrealized_pnl": round(unrealized, 4),
            "unrealized_r": round(unrealized / risk_d, 3),
            "realized_pnl": round(getattr(position, "realized_pnl", 0.0), 4),
        }

    return CoinContext(
        coin=coin,
        now_ms=now_ms,
        now_utc=datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        session=_session_from_hour(datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc).hour),
        last_price=last_price,
        last_spread_bps=worker.last_spread_bps,
        recent_bars=recent_bars,
        flow_bias_5m=flow_bias,
        trade_count_5m=trade_count_5m,
        realized_vol_5m_bps=realized_vol_5m,
        range_5m_bps=range_5m,
        range_30m_bps=range_30m,
        vwap_session=vwap_session,
        vwap_distance_bps=vwap_distance,
        open_position=pos_dict,
        account_equity=account_equity,
        daily_pnl=daily_pnl,
        daily_r=daily_r,
        recent_outcomes=recent_outcomes,
    )
