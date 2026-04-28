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
    Slippage explicitly disabled to isolate fee math.
    """
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_exit_slippage_bps=0.0,  # isolate fee math
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
    """Net fees should still be paid on a winning trade. Slippage disabled
    to isolate fee math."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_exit_slippage_bps=0.0,  # isolate fee math
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
    """Legacy TP1-as-maker mode (paper_tp1_is_taker=False).
    Used for back-comparison with pre-slippage replays.
    Live default is now `paper_tp1_is_taker=True`; covered separately by
    test_tp1_partial_with_taker_fee_and_slippage."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_exit_slippage_bps=0.0,
        paper_tp1_is_taker=False,  # legacy maker behavior
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


def test_fee_default_values_match_hl_retail_tier():
    """Defaults match HL retail (Tier 0) rates per fees API verified 2026-04-28:
      userAddRate (maker)  = +0.00015 (paid, NOT a rebate)
      userCrossRate (taker) = +0.00045

    Maker rebate (-0.00015) only applies to the 'mm' tier — accounts doing
    ≥0.5% of exchange-wide maker volume — unreachable for retail traders.
    """
    cfg = StrategyConfig()
    assert cfg.maker_fee_pct == 0.00015   # POSITIVE — paid
    assert cfg.taker_fee_pct == 0.00045


def test_exit_slippage_widens_stop_loss_long() -> None:
    """Long stop_loss exit fills below stop_price by `paper_exit_slippage_bps`."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_entry_slippage_bps=0.0,
        paper_exit_slippage_bps=2.0,  # 2 bps slip on exit
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))
    assert mgr.position is not None

    updates = mgr.on_trade(TradeEvent(ts_ms=3000, price=98.0, size=30.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    ct = closed[0].closed_trade
    assert ct.exit_reason == "stop_loss"

    # Long exit slips DOWN: realized exit = 98.0 * (1 - 0.0002) = 97.9804
    expected_exit = 98.0 * (1 - 2.0 / 10_000.0)
    assert abs(ct.exit_price - expected_exit) < 1e-6
    # Gross PnL: (97.9804 - 100) * 30 = -60.588
    expected_gross = (expected_exit - 100.0) * 30.0
    assert abs(ct.pnl_gross - expected_gross) < 1e-3
    # Strictly worse than the no-slippage case (-60.00)
    assert ct.pnl_gross < -60.00


def test_exit_slippage_widens_stop_loss_short() -> None:
    """Short stop_loss exit fills above stop_price by `paper_exit_slippage_bps`."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_entry_slippage_bps=0.0,
        paper_exit_slippage_bps=2.0,
        long_only=False,
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    # Short: entry 100, stop 102 (above), tp1 97, tp2 94
    sig = _make_signal(Side.SHORT, entry=100.0, stop=102.0, tp1=97.0, tp2=94.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))
    assert mgr.position is not None

    updates = mgr.on_trade(TradeEvent(ts_ms=3000, price=102.0, size=30.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    ct = closed[0].closed_trade
    assert ct.exit_reason == "stop_loss"

    # Short exit slips UP: realized exit = 102.0 * (1 + 0.0002) = 102.0204
    expected_exit = 102.0 * (1 + 2.0 / 10_000.0)
    assert abs(ct.exit_price - expected_exit) < 1e-6
    # Gross PnL for short: (entry - exit) * qty = (100 - 102.0204) * 30 = -60.612
    expected_gross = (100.0 - expected_exit) * 30.0
    assert abs(ct.pnl_gross - expected_gross) < 1e-3
    assert ct.pnl_gross < -60.00


def test_exit_slippage_applies_to_max_hold_and_time_stop() -> None:
    """All taker-style exits eat slippage: max_hold, time_stop, profit_take, early_exit."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_exit_slippage_bps=1.5,
        max_holding_sec=100,
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))

    # Force max_hold at price 101.0
    updates = mgr.on_trade(TradeEvent(ts_ms=102_001, price=101.0, size=30.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    ct = closed[0].closed_trade
    assert ct.exit_reason == "max_hold"

    # Long exit at 101 slips down: 101 * (1 - 0.00015) = 100.98485
    expected_exit = 101.0 * (1 - 1.5 / 10_000.0)
    assert abs(ct.exit_price - expected_exit) < 1e-6
    # Gross PnL strictly less than the no-slippage case (+30.00)
    assert ct.pnl_gross < 30.0


def test_tp2_exit_no_slippage_keeps_maker() -> None:
    """TP2 is a real limit order — no slippage, maker fee preserved."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_exit_slippage_bps=5.0,  # high slippage to prove tp2 ignores it
        trail_after_tp1=False,        # disable trail so tp2 can be hit cleanly
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))

    # Hit tp1 first (partial at 15 qty)
    mgr.on_trade(TradeEvent(ts_ms=3000, price=103.0, size=30.0))
    assert mgr.position is not None and mgr.position.tp1_filled

    # Hit tp2: must fill exactly at 106 (no slippage on real limit)
    updates = mgr.on_trade(TradeEvent(ts_ms=4000, price=106.0, size=30.0))
    closed = [u for u in updates if u.closed_trade is not None]
    assert len(closed) == 1
    ct = closed[0].closed_trade
    assert ct.exit_reason == "tp2"
    assert ct.exit_price == 106.0  # exact, no slippage


def test_tp1_partial_with_taker_fee_and_slippage() -> None:
    """When paper_tp1_is_taker=True, TP1 partial uses taker fee + exit slippage
    (matching live behavior — TP1 is software-triggered market_close)."""
    cfg = StrategyConfig(
        maker_fee_pct=-0.00015,
        taker_fee_pct=0.00045,
        paper_exit_slippage_bps=2.0,
        paper_tp1_is_taker=True,
        trail_after_tp1=False,
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=30.0, risk_dollars=60.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=30.0))

    # Hit TP1 — 15 qty partial at slipped price
    mgr.on_trade(TradeEvent(ts_ms=3000, price=103.0, size=30.0))
    assert mgr.position is not None and mgr.position.tp1_filled

    # TP1 fee should now be POSITIVE (taker, paid) on 15 qty:
    # entry maker = 100*30*-0.00015 = -0.45 (rebate)
    # tp1 taker on slipped 102.9794 * 15 = +0.6951
    # Combined realized_fees ~ 0.245 (positive) — was -0.682 with maker
    assert mgr.position.realized_fees > 0.0  # taker on TP1 makes it net positive


def test_entry_slippage_long_pays_up() -> None:
    """When paper_entry_slippage_bps > 0, long entries fill ABOVE limit."""
    cfg = StrategyConfig(
        paper_entry_slippage_bps=3.0,
        paper_exit_slippage_bps=0.0,
        pending_entry_expiry_sec=300,
        entry_touch_tolerance_bps=10.0,
    )
    mgr = PaperOrderManager(cfg)
    sig = _make_signal(Side.LONG, entry=100.0, stop=98.0, tp1=103.0, tp2=106.0)
    mgr.submit_entry(sig, "s1", qty=10.0, risk_dollars=20.0)
    mgr.on_trade(TradeEvent(ts_ms=2000, price=100.0, size=10.0))
    assert mgr.position is not None

    # Long entry slips UP: 100 * (1 + 0.0003) = 100.03
    expected_entry = 100.0 * (1 + 3.0 / 10_000.0)
    assert abs(mgr.position.entry_price - expected_entry) < 1e-6


def test_default_slippage_config_is_realistic() -> None:
    """Defaults should ship realistic exit slippage and taker TP1 (matches live)."""
    cfg = StrategyConfig()
    assert cfg.paper_entry_slippage_bps == 0.0     # Alo entries clean by design
    assert cfg.paper_exit_slippage_bps > 0.0       # exits eat book
    assert cfg.paper_tp1_is_taker is True          # match live market_close


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
