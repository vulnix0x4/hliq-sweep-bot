from __future__ import annotations

from hliq_bot.config import StrategyConfig
from hliq_bot.execution.order_manager import PaperOrderManager
from hliq_bot.models import Side, SweepSignal, TradeEvent


def _signal() -> SweepSignal:
    return SweepSignal(
        side=Side.LONG,
        level=100.0,
        level_label="prior_15m_low",
        sweep_extreme=99.4,
        entry_price=100.0,
        stop_price=99.0,
        tp1_price=102.0,
        tp2_price=104.0,
        confidence=0.9,
        reason="test",
        created_ms=1_000,
    )


def test_break_even_promotion_reduces_reversal_loss() -> None:
    mgr = PaperOrderManager(StrategyConfig(break_even_progress_tp1_frac=0.5))
    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=1.0, risk_dollars=1.0)

    fill = mgr.on_trade(TradeEvent(ts_ms=1_500, price=100.0, size=1.0, side="buy"))
    assert any(x.event_type.value == "entry_filled" for x in fill)
    assert mgr.position is not None

    mgr.on_trade(TradeEvent(ts_ms=2_000, price=101.0, size=1.0, side="buy"))
    assert mgr.position is not None
    # Trail from entry: stop = entry + (best - entry) * 0.35 = 100 + 1*0.35 = 100.35
    assert mgr.position.stop_price >= 100.0  # stop promoted above original
    assert mgr.position.stop_price <= 101.0  # but below best price


def _make_signal(side: Side, entry: float, stop: float, tp1: float, tp2: float) -> SweepSignal:
    return SweepSignal(
        side=side, level=entry, level_label="test", sweep_extreme=stop,
        entry_price=entry, stop_price=stop, tp1_price=tp1, tp2_price=tp2,
        confidence=0.9, reason="test", created_ms=1000,
    )


def test_trailing_stop_after_tp1_long():
    cfg = StrategyConfig(
        trail_after_tp1=True, trail_factor=0.5,
        pending_entry_expiry_sec=300, entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=1.0, risk_dollars=2.0)

    # Fill the entry
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=1.0))
    assert mgr.position is not None

    # Hit TP1 at 103 -> best_price=103, trail immediately:
    # entry + (best - entry) * 0.5 = 100 + (103 - 100) * 0.5 = 101.5
    mgr.on_trade(TradeEvent(ts_ms=3000, price=103.0, size=1.0))
    assert mgr.position.tp1_filled
    assert mgr.position.stop_price == 101.5  # trailed above break-even

    # Price runs to 104.5 -> best_price updates
    mgr.on_trade(TradeEvent(ts_ms=4000, price=104.5, size=1.0))
    # Trail: entry + (best - entry) * 0.5 = 100 + (104.5 - 100) * 0.5 = 102.25
    assert mgr.position is not None
    assert mgr.position.stop_price >= 102.0


def test_early_exit_deeply_negative():
    cfg = StrategyConfig(
        early_exit_sec=120, early_exit_r_threshold=-0.3,
        time_stop_sec=300, max_holding_sec=1800,
        pending_entry_expiry_sec=300, entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=1.0, risk_dollars=2.0)

    # Fill
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=1.0))

    # 130 seconds later, price drops to 99.3 -> unrealized = -0.7, r = -0.7/2.0 = -0.35
    updates = mgr.on_trade(TradeEvent(ts_ms=132000, price=99.3, size=1.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    assert closed[0].closed_trade.exit_reason == "early_exit"


def test_no_early_exit_when_not_deeply_negative():
    cfg = StrategyConfig(
        early_exit_sec=120, early_exit_r_threshold=-0.3,
        time_stop_sec=300, max_holding_sec=1800,
        pending_entry_expiry_sec=300, entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=1.0, risk_dollars=2.0)

    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=1.0))

    # 130s later, only slightly negative -> r = -0.1/2.0 = -0.05, above threshold
    updates = mgr.on_trade(TradeEvent(ts_ms=132000, price=99.9, size=1.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 0  # should NOT early exit


def test_runner_trail_tightens_after_threshold():
    cfg = StrategyConfig(
        trail_from_entry=True, trail_from_entry_factor=0.25,
        runner_trail_sec=300, runner_trail_factor=0.45,
        pending_entry_expiry_sec=300, entry_touch_tolerance_bps=10.0,
        max_holding_sec=1800, time_stop_sec=240,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=1.0, risk_dollars=2.0)

    # Fill
    mgr.on_trade(TradeEvent(ts_ms=1000, price=100.0, size=1.0))
    assert mgr.position is not None

    # Price goes to 102 early — trail at 25%: stop = 100 + 2*0.25 = 100.5
    mgr.on_trade(TradeEvent(ts_ms=10000, price=102.0, size=1.0))
    assert mgr.position.stop_price >= 100.4
    assert mgr.position.stop_price <= 100.6
    early_stop = mgr.position.stop_price

    # After 300s, same price 102 — runner trail at 45%: stop = 100 + 2*0.45 = 100.9
    mgr.on_trade(TradeEvent(ts_ms=302000, price=102.0, size=1.0))
    assert mgr.position is not None
    assert mgr.position.stop_price > early_stop  # tightened
    assert mgr.position.stop_price >= 100.8


def test_fee_modeling_on_stop_loss_exit():
    """Maker rebate on entry, taker fee on stop_loss exit.
    HL Tier 0: maker = -0.015% (rebate), taker = +0.045% (fee).
    $3000 notional round-trip -> entry_fee = -$0.45, exit_fee ≈ +$1.35, net ≈ +$0.90.
    """
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    # 30 qty × $100 = $3000 entry notional
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))
    assert mgr.position is not None

    # Price drops to stop -> stop_loss exit at 98.0 (taker)
    updates = mgr.on_trade(TradeEvent(ts_ms=3000, price=98.0, size=30.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    ct = closed[0].closed_trade
    assert ct.exit_reason == "stop_loss"

    # Gross PnL: (98 - 100) * 30 = -60
    assert round(ct.pnl_gross, 2) == -60.00
    # Entry fee (maker): 100 * 30 * -0.00015 = -0.45 (rebate received)
    # Exit fee (taker):   98 * 30 * +0.00045 = +1.323
    # Net fees paid: -0.45 + 1.323 = 0.873
    assert 0.85 <= ct.fees_paid <= 0.90
    # Net PnL: -60 - 0.873 = -60.873
    assert round(ct.pnl, 2) == round(ct.pnl_gross - ct.fees_paid, 2)
    assert ct.pnl < ct.pnl_gross  # fees make PnL worse


def test_fee_modeling_on_winning_trade():
    """Net fees should still be paid on a winning trade."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
        max_holding_sec=100,  # force max_hold quickly
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))

    # Price rises to 101, then time_stop fires (quick max_hold)
    updates = mgr.on_trade(TradeEvent(ts_ms=102_001, price=101.0, size=30.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    ct = closed[0].closed_trade

    # Gross: +30, fees positive (net cost), net PnL = 30 - ~0.88 = ~29.12
    assert round(ct.pnl_gross, 2) == 30.00
    assert ct.fees_paid > 0  # net fees paid even on winner
    assert ct.pnl < ct.pnl_gross


def test_fee_modeling_with_tp1_partial_exit():
    """TP1 partial is maker (limit fill), final close is taker."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        trail_after_tp1=True,
        trail_factor=0.5,
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))

    # Hit TP1 at 103 — partial 15 qty filled as maker (rebate)
    mgr.on_trade(TradeEvent(ts_ms=3000, price=103.0, size=30.0))
    assert mgr.position.tp1_filled
    assert mgr.position.realized_fees < 0  # entry rebate + tp1 rebate = negative (we've been paid)

    # Now price drops to stop at BE (100) -> taker close
    updates = mgr.on_trade(TradeEvent(ts_ms=4000, price=100.0, size=30.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    ct = closed[0].closed_trade

    # Entry maker fee: 100*30*-0.00015 = -0.45
    # TP1 maker fee on 15 qty: 103*15*-0.00015 = -0.232
    # Final taker fee on 15 qty at 100: 100*15*+0.00045 = +0.675
    # Net: -0.45 - 0.232 + 0.675 = -0.007 (close to zero — rebates roughly balance taker)
    assert -0.05 <= ct.fees_paid <= 0.10


def test_fee_default_values_match_hl_tier0():
    """Defaults should be HL Tier 0 rates (can be overridden via env)."""
    cfg = StrategyConfig()
    assert cfg.maker_fee_pct == -0.00015
    assert cfg.taker_fee_pct == 0.00045


def test_long_only_skips_shorts():
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=4.0, max_sweep_bps=40.0,
        min_reclaim_bps=2.0,
        volume_lookback_bars=5, volume_spike_mult=1.1,
        wick_body_ratio_min=1.2,
        long_only=True,
    )
    from hliq_bot.signal.sweep_detector import SweepDetector
    det = SweepDetector(cfg)
    from hliq_bot.models import Bar
    bar = Bar(start_ms=0, end_ms=60000, open=100.95, high=101.25, low=100.70, close=100.85,
              volume=240.0, trade_count=50, vwap=100.9, avg_spread_bps=1.0)
    # This would be a valid short signal but long_only should skip it
    signal = det._find_signal(
        bar,
        short_levels=[("prior_15m_high", 101.0)],
        long_levels=[],
        avg_vol=100.0,
    )
    assert signal is None
