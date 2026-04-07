from __future__ import annotations

from hliq_bot.models import Bar
from hliq_bot.signal.session_tracker import SessionTracker


def _bar(ts_ms: int, o: float, h: float, l: float, c: float) -> Bar:
    return Bar(
        start_ms=ts_ms, end_ms=ts_ms + 60_000,
        open=o, high=h, low=l, close=c,
        volume=1.0, trade_count=1, vwap=(o + c) / 2.0, avg_spread_bps=0.1,
    )


def test_session_from_hour():
    st = SessionTracker()
    assert st._session_from_hour(0) == "asia"
    assert st._session_from_hour(6) == "asia"
    assert st._session_from_hour(7) == "eu"
    assert st._session_from_hour(12) == "eu"
    assert st._session_from_hour(13) == "us"
    assert st._session_from_hour(21) == "us"
    assert st._session_from_hour(22) == "late"
    assert st._session_from_hour(23) == "late"


def test_session_open_tracked_on_first_bar():
    st = SessionTracker()
    ts = 1775433660000  # 2026-04-06 00:01 UTC -> asia
    st.on_bar(_bar(ts, 70000.0, 70100.0, 69900.0, 70050.0))
    assert st.current_session == "asia"
    assert st.current_session_open == 70000.0


def test_session_rollover_updates_prior():
    st = SessionTracker()
    ts = 1775433600000  # 2026-04-06 00:00:00 UTC
    for i in range(6):
        bar_ts = ts + i * 60_000
        st.on_bar(_bar(bar_ts, 70000.0 + i, 70100.0 + i, 69900.0, 70050.0 + i))
    eu_ts = ts + 7 * 3600 * 1000
    st.on_bar(_bar(eu_ts, 70200.0, 70300.0, 70100.0, 70250.0))
    assert st.current_session == "eu"
    assert st.current_session_open == 70200.0
    assert st.prior_session == "asia"
    assert st.prior_session_open == 70000.0
    assert st.prior_session_high == 70105.0
    assert st.prior_session_low == 69900.0


def test_pdh_pdl_on_day_rollover():
    st = SessionTracker()
    day1_start = 1775347200000  # 2026-04-05 00:00:00 UTC
    for i in range(10):
        bar_ts = day1_start + i * 60_000
        st.on_bar(_bar(bar_ts, 69000.0, 69500.0 + i * 10, 68500.0 - i * 5, 69100.0))
    day2_start = day1_start + 86400 * 1000
    st.on_bar(_bar(day2_start, 69200.0, 69300.0, 69100.0, 69250.0))
    assert st.prior_day_high == 69590.0
    assert st.prior_day_low == 68455.0


def test_levels_returns_empty_before_any_bars():
    st = SessionTracker()
    assert st.get_levels() == ([], [])


def test_levels_include_pdh_pdl_after_rollover():
    st = SessionTracker()
    day1_start = 1775347200000
    for i in range(10):
        st.on_bar(_bar(day1_start + i * 60_000, 69000.0, 69500.0, 68500.0, 69100.0))
    day2_start = day1_start + 86400 * 1000
    st.on_bar(_bar(day2_start, 69200.0, 69300.0, 69100.0, 69250.0))
    short_levels, long_levels = st.get_levels()
    short_labels = [label for label, _ in short_levels]
    long_labels = [label for label, _ in long_levels]
    assert "pdh" in short_labels
    assert "pdl" in long_labels
