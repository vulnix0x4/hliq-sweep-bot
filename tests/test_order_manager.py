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
