from __future__ import annotations

from hliq_bot.config import LevelConfig
from hliq_bot.models import Bar
from hliq_bot.signal.levels import derive_levels, round_number_levels
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.vwap_tracker import VWAPTracker


def _bar(ts_ms: int, o: float, h: float, l: float, c: float, v: float = 1.0) -> Bar:
    return Bar(
        start_ms=ts_ms,
        end_ms=ts_ms + 60_000,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        trade_count=1,
        vwap=(o + c) / 2.0,
        avg_spread_bps=0.1,
    )


def test_round_number_levels_btc():
    short_levels, long_levels = round_number_levels(current_price=68500.0, coin="BTC", range_pct=1.5)
    short_prices = [px for _, px in short_levels]
    long_prices = [px for _, px in long_levels]
    assert 69000.0 in short_prices
    assert 68000.0 in long_prices
    assert all(label.startswith("round_") for label, _ in short_levels)
    assert all(label.startswith("round_") for label, _ in long_levels)


def test_round_number_levels_eth():
    short_levels, long_levels = round_number_levels(current_price=3450.0, coin="ETH", range_pct=1.5)
    assert 3500.0 in [px for _, px in short_levels]
    assert 3400.0 in [px for _, px in long_levels]


def test_round_number_levels_sol():
    short_levels, long_levels = round_number_levels(current_price=135.0, coin="SOL", range_pct=5.0)
    assert 140.0 in [px for _, px in short_levels]
    assert 130.0 in [px for _, px in long_levels]


def test_round_number_levels_unknown_coin():
    short_levels, long_levels = round_number_levels(current_price=5.0, coin="DOGE", range_pct=2.0)
    assert len(short_levels) + len(long_levels) > 0


def test_derive_levels_includes_new_sources():
    # Use bars where VWAP (o+c)/2 = (69000+69500)/2 = 69250 is well-separated
    # from session_open (69000), prior_15m_high (69600), prior_15m_low (68900)
    # to avoid dedup collisions at 6.0 bps band (~42 bps separation needed).
    history = [_bar(i * 60_000, 69000.0, 69600.0, 68900.0, 69500.0) for i in range(20)]
    st = SessionTracker()
    vt = VWAPTracker()
    for bar in history:
        st.on_bar(bar)
        vt.on_bar(bar)
    cfg = LevelConfig(pdh_pdl=True, session_open=True, round_numbers=True, vwap=True, prior_session=True)
    result = derive_levels(
        history=history,
        timeframe_sec=60,
        equal_band_bps=6.0,
        level_config=cfg,
        session_tracker=st,
        vwap_tracker=vt,
        current_price=70050.0,
        coin="BTC",
    )
    all_labels = [l for l, _ in result.short_levels] + [l for l, _ in result.long_levels]
    assert any(l.startswith("round_") for l in all_labels)
    assert any(l == "vwap_daily" for l in all_labels)


def test_derive_levels_respects_disabled_flags():
    history = [_bar(i * 60_000, 70000.0, 70100.0, 69900.0, 70050.0) for i in range(20)]
    st = SessionTracker()
    vt = VWAPTracker()
    for bar in history:
        st.on_bar(bar)
        vt.on_bar(bar)
    cfg = LevelConfig(pdh_pdl=False, session_open=False, round_numbers=False, vwap=False, prior_session=False)
    result = derive_levels(
        history=history,
        timeframe_sec=60,
        equal_band_bps=6.0,
        level_config=cfg,
        session_tracker=st,
        vwap_tracker=vt,
        current_price=70050.0,
        coin="BTC",
    )
    all_labels = [l for l, _ in result.short_levels] + [l for l, _ in result.long_levels]
    assert not any(l.startswith("round_") for l in all_labels)
    assert not any(l == "vwap_daily" for l in all_labels)
    assert not any(l == "pdh" for l in all_labels)


def test_derive_levels_backward_compat_no_new_params():
    """Old call signature (3 positional args) must still work."""
    # Use 60+ bars so 1h lookback (60 bars) sees a different range than 15m (15 bars).
    # First 45 bars have wider range, last 15 bars have narrower range.
    history = [_bar(i * 60_000, 100.0, 101.0, 99.0, 100.1) for i in range(45)]
    history += [_bar((45 + i) * 60_000, 100.0, 100.3, 99.7, 100.1) for i in range(15)]
    result = derive_levels(history, 60, 6.0)
    # Should still produce the basic prior_15m / prior_1h levels
    all_labels = [l for l, _ in result.short_levels] + [l for l, _ in result.long_levels]
    assert "prior_15m_high" in all_labels
    assert "prior_1h_high" in all_labels
    assert "prior_15m_low" in all_labels
    assert "prior_1h_low" in all_labels
    # No new source levels without config
    assert not any(l.startswith("round_") for l in all_labels)
    assert not any(l == "vwap_daily" for l in all_labels)


def test_round_number_levels_labels_contain_price():
    """Labels should encode the price like round_69000."""
    short_levels, long_levels = round_number_levels(current_price=68500.0, coin="BTC", range_pct=1.5)
    for label, px in short_levels + long_levels:
        assert label.startswith("round_")
        # The label should parse to the actual price
        price_str = label[len("round_"):]
        assert float(price_str) == px


def test_round_number_auto_interval_for_unknown_coin():
    """Coins not in _ROUND_INTERVALS should get auto-computed intervals."""
    short_levels, long_levels = round_number_levels(current_price=42.0, coin="OBSCURE", range_pct=5.0)
    assert len(short_levels) + len(long_levels) > 0
    for label, _ in short_levels + long_levels:
        assert label.startswith("round_")
