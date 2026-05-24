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

from hliq_bot.ai.market_data import CoinMeta, L2Book, MarketDataCache
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
    # Enriched fields (may be None when the market-data cache is offline):
    funding_rate: float | None = None
    open_interest: float | None = None
    mark_price: float | None = None
    day_change_pct: float | None = None
    day_volume_usd: float | None = None
    order_book: dict | None = None             # {bids, asks, spread_bps, depth_imbalance}
    other_coins: list[dict] | None = None      # multi-coin sympathy view
    portfolio: dict | None = None              # all-coin position summary

    def to_prompt_dict(self) -> dict[str, Any]:
        """JSON-serializable dict that goes verbatim into the user message."""
        d: dict[str, Any] = {
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
        # Enriched fields only emitted when present, to keep tokens lean.
        if self.funding_rate is not None:
            d["funding_rate"] = round(self.funding_rate, 8)
        if self.open_interest is not None:
            d["open_interest"] = round(self.open_interest, 2)
        if self.mark_price is not None:
            d["mark_price"] = round(self.mark_price, 6)
        if self.day_change_pct is not None:
            d["day_change_pct"] = round(self.day_change_pct, 2)
        if self.day_volume_usd is not None:
            d["day_volume_usd"] = round(self.day_volume_usd, 0)
        if self.order_book is not None:
            d["order_book"] = self.order_book
        if self.other_coins is not None:
            d["other_coins"] = self.other_coins
        if self.portfolio is not None:
            d["portfolio"] = self.portfolio
        return d


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


def _l2_to_prompt_dict(book: L2Book) -> dict[str, Any]:
    return {
        "bids": [[round(p, 6), round(s, 4)] for p, s in book.bids],
        "asks": [[round(p, 6), round(s, 4)] for p, s in book.asks],
        "spread_bps": round(book.spread_bps() or 0.0, 2),
        "depth_imbalance": round(book.depth_imbalance() or 1.0, 3),
    }


def _other_coins_summary(
    market_data: MarketDataCache,
    *,
    focus_coin: str,
    workers_by_coin: dict[str, Any] | None,
    now_ms: int,
) -> list[dict[str, Any]]:
    """Compact per-other-coin view: change, funding, recent flow bias.

    The LLM uses this to gauge alt-coin sympathy. Capped at 8 coins by
    descending day_ntl_volume to keep tokens in check.
    """
    metas = market_data.all_meta()
    if not metas:
        return []
    others = [m for c, m in metas.items() if c != focus_coin.upper()]
    others.sort(key=lambda m: -(m.day_ntl_volume or 0))
    out: list[dict[str, Any]] = []
    for m in others[:8]:
        row: dict[str, Any] = {
            "coin": m.name,
            "day_change_pct": round(m.day_change_pct, 2) if m.day_change_pct is not None else None,
            "funding": round(m.funding, 8) if m.funding is not None else None,
        }
        # If this coin has a CoinWorker in our system, include its 5m flow bias.
        w = (workers_by_coin or {}).get(m.name)
        if w is not None:
            bias, _ = _flow_bias(list(getattr(w, "recent_signed_flow", [])), 5 * 60 * 1000, now_ms)
            row["flow_bias_5m"] = round(bias, 3)
        out.append(row)
    return out


def _portfolio_summary(
    workers_by_coin: dict[str, Any] | None,
) -> dict[str, Any]:
    """Snapshot of all open positions + pending entries across coins.

    Used by the AI to avoid concentrating exposure or chasing correlated trades.
    """
    summary: dict[str, Any] = {
        "open_positions": [],
        "pending_entries": [],
    }
    if not workers_by_coin:
        return summary
    for coin, w in workers_by_coin.items():
        position = getattr(w.executor, "position", None) if hasattr(w, "executor") else None
        if position is not None:
            summary["open_positions"].append({
                "coin": coin,
                "side": position.side.value,
                "qty_remaining": round(position.qty_remaining, 8),
                "entry_price": round(position.entry_price, 6),
                "stop_price": round(position.stop_price, 6),
            })
        pending = getattr(w.executor, "pending_entry", None) if hasattr(w, "executor") else None
        if pending is not None:
            summary["pending_entries"].append({
                "coin": coin,
                "side": pending.side.value,
                "qty": round(pending.qty, 8),
                "entry_price": round(pending.entry_price, 6),
            })
    summary["concurrent_long"] = sum(1 for p in summary["open_positions"] if p["side"] == "long")
    summary["concurrent_short"] = sum(1 for p in summary["open_positions"] if p["side"] == "short")
    return summary


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
    market_data: MarketDataCache | None = None,
    workers_by_coin: dict[str, Any] | None = None,
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

    # Enrichment from MarketDataCache (best-effort; AI works without it).
    funding = oi = mark = day_change = day_vol = None
    order_book_dict: dict | None = None
    other_coins = None
    if market_data is not None:
        meta = market_data.meta_for(coin)
        if meta is not None:
            funding = meta.funding
            oi = meta.open_interest
            mark = meta.mark_px
            day_change = meta.day_change_pct
            day_vol = meta.day_ntl_volume
        book = market_data.l2_for(coin)
        if book is not None and (book.bids or book.asks):
            order_book_dict = _l2_to_prompt_dict(book)
        other_coins_list = _other_coins_summary(
            market_data, focus_coin=coin,
            workers_by_coin=workers_by_coin, now_ms=now_ms,
        )
        if other_coins_list:
            other_coins = other_coins_list

    portfolio = _portfolio_summary(workers_by_coin) if workers_by_coin else None

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
        funding_rate=funding,
        open_interest=oi,
        mark_price=mark,
        day_change_pct=day_change,
        day_volume_usd=day_vol,
        order_book=order_book_dict,
        other_coins=other_coins,
        portfolio=portfolio,
    )
