from __future__ import annotations

from hliq_bot.config import StrategyConfig
from hliq_bot.models import Bar, Side
from hliq_bot.signal.sweep_detector import SweepDetector


def _bar(ts: int, o: float, h: float, l: float, c: float, v: float, spread: float = 2.0) -> Bar:
    return Bar(
        start_ms=ts,
        end_ms=ts + 60_000,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        trade_count=50,
        vwap=(o + c) / 2.0,
        avg_spread_bps=spread,
    )


def test_short_sweep_signal_triggers() -> None:
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=10.0,
        max_sweep_bps=40.0,
        volume_lookback_bars=10,
        volume_spike_mult=1.2,
        wick_body_ratio_min=1.5,
    )
    det = SweepDetector(cfg)

    ts = 0
    for _ in range(70):
        det.on_bar(_bar(ts, 100.0, 100.8, 99.8, 100.1, 100.0))
        ts += 60_000

    # Establish a clear prior high around 101.0
    det.on_bar(_bar(ts, 100.2, 101.0, 99.9, 100.3, 100.0))
    ts += 60_000

    signal = det.on_bar(
        _bar(
            ts,
            o=100.9,
            h=101.25,  # ~24.7 bps sweep over 101.0
            l=100.5,
            c=100.8,  # closes back below swept level
            v=220.0,  # volume spike
            spread=1.5,
        )
    )

    assert signal is not None
    assert signal.side == Side.SHORT
    assert "sweep" in signal.reason


def test_no_signal_without_volume_spike() -> None:
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=10.0,
        max_sweep_bps=40.0,
        volume_lookback_bars=10,
        volume_spike_mult=2.0,  # stricter
        wick_body_ratio_min=1.5,
    )
    det = SweepDetector(cfg)

    ts = 0
    for _ in range(70):
        det.on_bar(_bar(ts, 100.0, 100.9, 99.7, 100.2, 120.0))
        ts += 60_000

    det.on_bar(_bar(ts, 100.3, 101.0, 99.9, 100.4, 120.0))
    ts += 60_000

    signal = det.on_bar(
        _bar(
            ts,
            o=100.95,
            h=101.25,
            l=100.7,
            c=100.85,
            v=140.0,  # not enough for 2x avg
            spread=1.5,
        )
    )
    assert signal is None


def test_no_signal_without_reclaim_strength() -> None:
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=10.0,
        max_sweep_bps=40.0,
        min_reclaim_bps=8.0,
        volume_lookback_bars=10,
        volume_spike_mult=1.2,
        wick_body_ratio_min=1.2,
    )
    det = SweepDetector(cfg)

    ts = 0
    for _ in range(70):
        det.on_bar(_bar(ts, 100.0, 100.8, 99.8, 100.1, 100.0))
        ts += 60_000

    det.on_bar(_bar(ts, 100.2, 101.0, 99.9, 100.3, 100.0))
    ts += 60_000

    signal = det.on_bar(
        _bar(
            ts,
            o=100.95,
            h=101.2,
            l=100.7,
            c=100.95,  # closes only ~5 bps below level (reclaim too weak for 8 bps floor)
            v=230.0,
            spread=1.5,
        )
    )
    assert signal is None


def test_detector_diagnostics_exposed_and_reset() -> None:
    det = SweepDetector(StrategyConfig(timeframe_sec=60))
    det.on_bar(_bar(0, 100.0, 100.2, 99.8, 100.1, 10.0))

    diag = dict(det.consume_diagnostics())
    assert diag.get("skip_history", 0) >= 1
    assert det.consume_diagnostics() == []


def test_detector_history_warmup_is_not_one_hour() -> None:
    det = SweepDetector(StrategyConfig(timeframe_sec=60, volume_lookback_bars=20, trend_lookback_bars=20))
    assert det._min_history_bars == 15


def test_detector_picks_best_ranked_level_not_first_match() -> None:
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=4.0,
        max_sweep_bps=40.0,
        min_reclaim_bps=2.0,
        volume_lookback_bars=5,
        volume_spike_mult=1.1,
        wick_body_ratio_min=1.2,
    )
    det = SweepDetector(cfg)

    signal = det._short_signal(
        _bar(
            0,
            o=100.95,
            h=101.25,
            l=100.70,
            c=100.85,
            v=240.0,
            spread=1.0,
        ),
        short_levels=[("weaker_first", 101.18), ("stronger_second", 101.0)],
        avg_vol=100.0,
    )

    assert signal is not None
    assert signal.signal_score > 0.0
    assert signal.level_label == "stronger_second"


def test_detector_does_not_hard_skip_trending_context() -> None:
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=10.0,
        max_sweep_bps=40.0,
        max_trend_move_bps=20.0,
        volume_lookback_bars=10,
        volume_spike_mult=1.2,
        wick_body_ratio_min=1.5,
    )
    det = SweepDetector(cfg)
    det._is_trending = lambda history: True  # type: ignore[method-assign]

    ts = 0
    for _ in range(70):
        det.on_bar(_bar(ts, 100.0, 100.8, 99.8, 100.1, 100.0))
        ts += 60_000

    det.on_bar(_bar(ts, 100.2, 101.0, 99.9, 100.3, 100.0))
    ts += 60_000
    signal = det.on_bar(
        _bar(
            ts,
            o=100.9,
            h=101.25,
            l=100.5,
            c=100.8,
            v=220.0,
            spread=1.5,
        )
    )

    assert signal is not None


def test_signal_score_includes_level_type_weight() -> None:
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=4.0,
        max_sweep_bps=40.0,
        min_reclaim_bps=2.0,
        volume_lookback_bars=5,
        volume_spike_mult=1.1,
        wick_body_ratio_min=1.2,
    )
    det = SweepDetector(cfg)

    bar = _bar(0, o=100.95, h=101.25, l=100.70, c=100.85, v=240.0, spread=1.0)

    signal_base = det._short_signal(
        bar,
        short_levels=[("prior_15m_high", 101.0)],
        avg_vol=100.0,
    )
    signal_pdh = det._short_signal(
        bar,
        short_levels=[("pdh", 101.0)],
        avg_vol=100.0,
    )

    assert signal_base is not None
    assert signal_pdh is not None
    assert signal_pdh.signal_score > signal_base.signal_score
