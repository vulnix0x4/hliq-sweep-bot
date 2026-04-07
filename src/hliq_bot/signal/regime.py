from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    RANGE = "range"
    TREND = "trend"
    HIGH_VOL = "high_vol"
    ILLIQUID = "illiquid"


@dataclass(slots=True)
class RegimeState:
    regime: Regime
    trend_bps: float
    avg_range_pct: float
    spread_bps: float
    move_30s_pct: float
    session: str


def classify_regime(
    closes: list[float],
    ranges_pct: list[float],
    spread_bps: float,
    move_30s_pct: float,
    trend_threshold_bps: float,
    high_vol_threshold_pct: float,
    illiquid_spread_bps: float,
    hour_utc: int,
) -> RegimeState:
    trend_bps = 0.0
    if len(closes) >= 2 and closes[0] > 0:
        trend_bps = abs((closes[-1] - closes[0]) / closes[0]) * 10_000.0
    avg_range = sum(ranges_pct) / len(ranges_pct) if ranges_pct else 0.0

    if spread_bps >= illiquid_spread_bps:
        regime = Regime.ILLIQUID
    elif avg_range >= high_vol_threshold_pct or abs(move_30s_pct) >= high_vol_threshold_pct:
        regime = Regime.HIGH_VOL
    elif trend_bps >= trend_threshold_bps:
        regime = Regime.TREND
    else:
        regime = Regime.RANGE

    session = _session_from_hour(hour_utc)
    return RegimeState(
        regime=regime,
        trend_bps=trend_bps,
        avg_range_pct=avg_range,
        spread_bps=spread_bps,
        move_30s_pct=move_30s_pct,
        session=session,
    )


def _session_from_hour(hour_utc: int) -> str:
    # Coarse UTC buckets for risk shaping.
    if 0 <= hour_utc < 7:
        return "asia"
    if 7 <= hour_utc < 13:
        return "eu"
    if 13 <= hour_utc < 22:
        return "us"
    return "late"

