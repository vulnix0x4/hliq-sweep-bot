from __future__ import annotations

from dataclasses import dataclass

from hliq_bot.models import Bar


@dataclass(slots=True)
class LevelSet:
    short_levels: list[tuple[str, float]]
    long_levels: list[tuple[str, float]]


def derive_levels(history: list[Bar], timeframe_sec: int, equal_band_bps: float) -> LevelSet:
    if not history:
        return LevelSet(short_levels=[], long_levels=[])

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
    return LevelSet(short_levels=short_levels, long_levels=long_levels)


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

