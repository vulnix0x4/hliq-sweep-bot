from __future__ import annotations

import math
from dataclasses import dataclass

from hliq_bot.config import LevelConfig
from hliq_bot.models import Bar
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.vwap_tracker import VWAPTracker


@dataclass(slots=True)
class LevelSet:
    short_levels: list[tuple[str, float]]
    long_levels: list[tuple[str, float]]


# ------------------------------------------------------------------
# Round-number interval table
# ------------------------------------------------------------------
_ROUND_INTERVALS: dict[str, float] = {
    "BTC": 1000.0,
    "ETH": 100.0,
    "SOL": 10.0,
    "AVAX": 10.0,
    "DOGE": 0.01,
    "ARB": 0.1,
    "OP": 0.5,
    "MATIC": 0.1,
    "LINK": 1.0,
    "WIF": 0.1,
}


def _auto_interval(price: float) -> float:
    """Compute a sensible round-number interval for an unknown coin."""
    if price <= 0:
        return 1.0
    return 10 ** math.floor(math.log10(price)) * 0.01


# ------------------------------------------------------------------
# Public: round-number levels
# ------------------------------------------------------------------
def round_number_levels(
    current_price: float,
    coin: str,
    range_pct: float,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (short_levels, long_levels) of round-number price levels.

    Levels above *current_price* are short resistance; levels below are
    long support.  Each is labelled ``round_{price:g}``.
    """
    interval = _ROUND_INTERVALS.get(coin.upper(), _auto_interval(current_price))
    if interval <= 0:
        return [], []

    lower_bound = current_price * (1.0 - range_pct / 100.0)
    upper_bound = current_price * (1.0 + range_pct / 100.0)

    # First round level at or below lower_bound
    start = math.floor(lower_bound / interval) * interval

    short_levels: list[tuple[str, float]] = []
    long_levels: list[tuple[str, float]] = []

    level = start
    while level <= upper_bound:
        if abs(level - current_price) > 1e-12:  # skip if ~equal to price
            label = f"round_{level:g}"
            if level > current_price:
                short_levels.append((label, level))
            else:
                long_levels.append((label, level))
        level += interval
        # safety: avoid infinite loop with tiny intervals
        if level <= start:
            break

    return short_levels, long_levels


# ------------------------------------------------------------------
# Public: derive_levels (expanded)
# ------------------------------------------------------------------
def derive_levels(
    history: list[Bar],
    timeframe_sec: int,
    equal_band_bps: float,
    level_config: LevelConfig | None = None,
    session_tracker: SessionTracker | None = None,
    vwap_tracker: VWAPTracker | None = None,
    current_price: float = 0.0,
    coin: str = "BTC",
) -> LevelSet:
    if not history:
        return LevelSet(short_levels=[], long_levels=[])

    # ------------------------------------------------------------------
    # Existing logic: prior 15m / 1h highs & lows + equal levels
    # ------------------------------------------------------------------
    bars_15m = max(1, int((15 * 60) / timeframe_sec))
    bars_1h = max(1, int((60 * 60) / timeframe_sec))

    look_15m = history[-bars_15m:]
    look_1h = history[-bars_1h:]

    short_levels: list[tuple[str, float]] = []
    long_levels: list[tuple[str, float]] = []

    high_15 = max(b.high for b in look_15m)
    low_15 = min(b.low for b in look_15m)
    high_1h = max(b.high for b in look_1h)
    low_1h = min(b.low for b in look_1h)

    short_levels.append(("prior_15m_high", high_15))
    short_levels.append(("prior_1h_high", high_1h))
    long_levels.append(("prior_15m_low", low_15))
    long_levels.append(("prior_1h_low", low_1h))

    eq_highs, eq_lows = _equal_levels(history, equal_band_bps)
    for i, level in enumerate(eq_highs, start=1):
        short_levels.append((f"equal_high_{i}", level))
    for i, level in enumerate(eq_lows, start=1):
        long_levels.append((f"equal_low_{i}", level))

    short_levels = _dedupe_levels(short_levels, equal_band_bps)
    long_levels = _dedupe_levels(long_levels, equal_band_bps)

    # ------------------------------------------------------------------
    # New level sources (only when level_config is provided)
    # ------------------------------------------------------------------
    cfg = level_config
    if cfg is not None:
        # --- Session tracker levels (PDH/PDL, session opens, prior session H/L) ---
        if session_tracker is not None:
            st_short, st_long = session_tracker.get_levels()
            for label, px in st_short:
                if label in ("pdh",) and not cfg.pdh_pdl:
                    continue
                if label.startswith("session_open") and not cfg.session_open:
                    continue
                if label.startswith("prior_session") and not cfg.prior_session:
                    continue
                short_levels.append((label, px))
            for label, px in st_long:
                if label in ("pdl",) and not cfg.pdh_pdl:
                    continue
                if label.startswith("session_open") and not cfg.session_open:
                    continue
                if label.startswith("prior_session") and not cfg.prior_session:
                    continue
                long_levels.append((label, px))

        # --- VWAP levels ---
        if cfg.vwap and vwap_tracker is not None:
            vwap_short, vwap_long = vwap_tracker.get_levels()
            short_levels.extend(vwap_short)
            long_levels.extend(vwap_long)

        # --- Round-number levels ---
        if cfg.round_numbers and current_price > 0:
            rn_short, rn_long = round_number_levels(
                current_price, coin, cfg.round_number_range_pct
            )
            short_levels.extend(rn_short)
            long_levels.extend(rn_long)

    # ------------------------------------------------------------------
    # Final dedup pass on all levels
    # ------------------------------------------------------------------
    short_levels = _dedupe_levels(short_levels, equal_band_bps)
    long_levels = _dedupe_levels(long_levels, equal_band_bps)
    return LevelSet(short_levels=short_levels, long_levels=long_levels)


# ------------------------------------------------------------------
# Internal helpers (unchanged)
# ------------------------------------------------------------------
def _equal_levels(history: list[Bar], band_bps: float) -> tuple[list[float], list[float]]:
    pivot_highs: list[float] = []
    pivot_lows: list[float] = []
    if len(history) < 3:
        return pivot_highs, pivot_lows

    for i in range(1, len(history) - 1):
        left = history[i - 1]
        mid = history[i]
        right = history[i + 1]
        if mid.high > left.high and mid.high > right.high:
            pivot_highs.append(mid.high)
        if mid.low < left.low and mid.low < right.low:
            pivot_lows.append(mid.low)

    return _cluster_levels(pivot_highs, band_bps), _cluster_levels(pivot_lows, band_bps)


def _cluster_levels(pivots: list[float], band_bps: float) -> list[float]:
    if len(pivots) < 2:
        return []
    clusters: list[list[float]] = []
    for px in pivots:
        matched = False
        for c in clusters:
            ref = sum(c) / len(c)
            diff_bps = abs(px - ref) / ref * 10_000.0
            if diff_bps <= band_bps:
                c.append(px)
                matched = True
                break
        if not matched:
            clusters.append([px])

    return [sum(c) / len(c) for c in clusters if len(c) >= 2]


def _dedupe_levels(levels: list[tuple[str, float]], band_bps: float) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for label, px in levels:
        if px <= 0:
            continue
        duplicate = False
        for _, existing in out:
            diff_bps = abs(px - existing) / existing * 10_000.0
            if diff_bps <= band_bps:
                duplicate = True
                break
        if not duplicate:
            out.append((label, px))
    return out
