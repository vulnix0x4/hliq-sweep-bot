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
    assert mgr.position.stop_price == 100.0

    updates = mgr.on_trade(TradeEvent(ts_ms=2_500, price=99.8, size=1.0, side="sell"))
    closed = next(x.closed_trade for x in updates if x.closed_trade is not None)
    assert round(closed.pnl, 6) == 0.0
    assert closed.mfe_pnl > 0.0
    assert closed.mae_pnl <= 0.0
