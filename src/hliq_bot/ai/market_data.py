"""HL market-data cache for AI context enrichment.

Fetches HL Info data (funding, open interest, mark, 24h, L2 book) and caches
per-key for short TTL so AI calls don't pay round-trip latency on every poll.

All fetches are best-effort — on any failure we return whatever's in cache
(possibly stale) or None. The AI's context falls back gracefully when fields
are missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CoinMeta:
    """Per-coin perp context from HL meta_and_asset_ctxs.

    Field names match the HL Info response semantics so future readers can
    cross-reference the API docs.
    """
    name: str
    mark_px: float | None = None
    mid_px: float | None = None
    funding: float | None = None          # current funding rate (per-hour-ish, see HL docs)
    open_interest: float | None = None    # in base units
    day_volume: float | None = None       # USD 24h
    day_ntl_volume: float | None = None
    prev_day_px: float | None = None      # 24h-ago price
    premium: float | None = None
    impact_pxs: list[float] | None = None

    @property
    def day_change_pct(self) -> float | None:
        if not (self.mark_px and self.prev_day_px):
            return None
        return ((self.mark_px - self.prev_day_px) / self.prev_day_px) * 100.0


@dataclass(slots=True)
class L2Book:
    """Compact top-N order book snapshot."""
    coin: str
    ts_ms: int
    bids: list[tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: list[tuple[float, float]] = field(default_factory=list)

    def spread_bps(self) -> float | None:
        if not (self.bids and self.asks):
            return None
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        if best_bid <= 0 or best_ask <= 0:
            return None
        return ((best_ask - best_bid) / best_bid) * 10000

    def depth_imbalance(self, top_n: int = 5) -> float | None:
        """Sum of bid sizes / sum of ask sizes over top-N levels. >1 = buyers thicker."""
        bs = sum(s for _, s in self.bids[:top_n])
        a_s = sum(s for _, s in self.asks[:top_n])
        if a_s <= 0:
            return None
        return bs / a_s


class MarketDataCache:
    """Cached, thread-safe wrapper over HL Info."""

    def __init__(
        self,
        info: Any,
        *,
        meta_ttl_sec: float = 30.0,
        l2_ttl_sec: float = 5.0,
        l2_depth: int = 5,
    ) -> None:
        self._info = info
        self._meta_ttl = meta_ttl_sec
        self._l2_ttl = l2_ttl_sec
        self._l2_depth = max(1, l2_depth)
        self._lock = threading.Lock()
        # meta_and_asset_ctxs cache (all coins in one call)
        self._meta_ts: float = 0.0
        self._meta_by_coin: dict[str, CoinMeta] = {}
        # L2 cache (per coin)
        self._l2_ts: dict[str, float] = {}
        self._l2_by_coin: dict[str, L2Book] = {}

    def meta_for(self, coin: str) -> CoinMeta | None:
        """Return funding/OI/mark for one coin. Refreshes meta cache when stale."""
        self._refresh_meta_if_stale()
        return self._meta_by_coin.get(coin.upper())

    def all_meta(self) -> dict[str, CoinMeta]:
        self._refresh_meta_if_stale()
        return dict(self._meta_by_coin)

    def l2_for(self, coin: str) -> L2Book | None:
        """Return top-N book for one coin. Refreshes if stale."""
        now = time.monotonic()
        with self._lock:
            ts = self._l2_ts.get(coin, 0.0)
            cached = self._l2_by_coin.get(coin)
        if cached is not None and now - ts < self._l2_ttl:
            return cached
        try:
            raw = self._info.l2_snapshot(coin)
            book = self._parse_l2(coin, raw)
        except Exception as exc:
            log.debug("l2_snapshot failed for %s: %s", coin, exc)
            return cached  # serve stale rather than nothing
        if book is None:
            return cached
        with self._lock:
            self._l2_by_coin[coin] = book
            self._l2_ts[coin] = now
        return book

    # ---- Internals ----

    def _refresh_meta_if_stale(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._meta_ts < self._meta_ttl and self._meta_by_coin:
                return
        try:
            raw = self._info.meta_and_asset_ctxs()
        except Exception as exc:
            log.debug("meta_and_asset_ctxs failed: %s", exc)
            return
        parsed = self._parse_meta(raw)
        if not parsed:
            return
        with self._lock:
            self._meta_by_coin = parsed
            self._meta_ts = now

    def _parse_meta(self, raw: Any) -> dict[str, CoinMeta]:
        """meta_and_asset_ctxs returns [meta_dict, [ctxs...]] aligned by universe order."""
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return {}
        meta, ctxs = raw[0], raw[1]
        universe = meta.get("universe") if isinstance(meta, dict) else None
        if not isinstance(universe, list) or not isinstance(ctxs, list):
            return {}
        out: dict[str, CoinMeta] = {}
        for i, u in enumerate(universe):
            if i >= len(ctxs):
                break
            name = str(u.get("name", "")).upper()
            if not name:
                continue
            c = ctxs[i] if isinstance(ctxs[i], dict) else {}
            out[name] = CoinMeta(
                name=name,
                mark_px=_safe_float(c.get("markPx")),
                mid_px=_safe_float(c.get("midPx")),
                funding=_safe_float(c.get("funding")),
                open_interest=_safe_float(c.get("openInterest")),
                day_volume=_safe_float(c.get("dayBaseVlm")),
                day_ntl_volume=_safe_float(c.get("dayNtlVlm")),
                prev_day_px=_safe_float(c.get("prevDayPx")),
                premium=_safe_float(c.get("premium")),
                impact_pxs=[_safe_float(p) for p in (c.get("impactPxs") or [])] or None,
            )
        return out

    def _parse_l2(self, coin: str, raw: Any) -> L2Book | None:
        if not isinstance(raw, dict):
            return None
        levels = raw.get("levels")
        if not isinstance(levels, list) or len(levels) < 2:
            return None
        bids_raw = levels[0] if isinstance(levels[0], list) else []
        asks_raw = levels[1] if isinstance(levels[1], list) else []
        bids = [(_safe_float(b.get("px")), _safe_float(b.get("sz"))) for b in bids_raw[: self._l2_depth] if isinstance(b, dict)]
        asks = [(_safe_float(a.get("px")), _safe_float(a.get("sz"))) for a in asks_raw[: self._l2_depth] if isinstance(a, dict)]
        # Drop levels with bad data
        bids = [(p, s) for p, s in bids if p and s]
        asks = [(p, s) for p, s in asks if p and s]
        return L2Book(
            coin=coin,
            ts_ms=int(raw.get("time", time.time() * 1000)),
            bids=bids,
            asks=asks,
        )


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN guard
        return None
    return out
