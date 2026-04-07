from __future__ import annotations

from hliq_bot.models import Bar
from hliq_bot.signal.vwap_tracker import VWAPTracker


def _bar(ts_ms: int, o: float, h: float, l: float, c: float, v: float) -> Bar:
    return Bar(
        start_ms=ts_ms, end_ms=ts_ms + 60_000,
        open=o, high=h, low=l, close=c,
        volume=v, trade_count=1, vwap=(o + c) / 2.0, avg_spread_bps=0.1,
    )


def test_vwap_single_bar():
    vt = VWAPTracker()
    vt.on_bar(_bar(1000, 100.0, 101.0, 99.0, 100.5, 10.0))
    assert vt.vwap > 0


def test_vwap_accumulates_across_bars():
    vt = VWAPTracker()
    vt.on_bar(_bar(1000, 100.0, 101.0, 99.0, 100.0, 10.0))
    vt.on_bar(_bar(61000, 110.0, 111.0, 109.0, 110.0, 20.0))
    expected = (100.0 * 10.0 + 110.0 * 20.0) / 30.0
    assert abs(vt.vwap - expected) < 0.01


def test_vwap_resets_on_new_day():
    vt = VWAPTracker()
    day1_ts = 1775347200000  # 2026-04-05 00:00:00 UTC
    vt.on_bar(_bar(day1_ts, 69000.0, 69100.0, 68900.0, 69050.0, 100.0))
    day2_ts = day1_ts + 86400 * 1000
    vt.on_bar(_bar(day2_ts, 70000.0, 70100.0, 69900.0, 70050.0, 50.0))
    expected = (70000.0 + 70050.0) / 2.0
    assert abs(vt.vwap - expected) < 1.0


def test_vwap_zero_before_any_bars():
    vt = VWAPTracker()
    assert vt.vwap == 0.0


def test_get_levels_returns_vwap_as_dual_sided():
    vt = VWAPTracker()
    vt.on_bar(_bar(1000, 100.0, 101.0, 99.0, 100.0, 10.0))
    short_levels, long_levels = vt.get_levels()
    assert any(label == "vwap_daily" for label, _ in short_levels)
    assert any(label == "vwap_daily" for label, _ in long_levels)
