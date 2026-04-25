from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hliq_bot.config import LiveConfig, StrategyConfig
from hliq_bot.execution.hyperliquid_order_manager import HyperliquidOrderManager
from hliq_bot.models import ExecEventType, Side, SweepSignal, TradeEvent


def _signal() -> SweepSignal:
    return SweepSignal(
        side=Side.LONG, level=100.0, level_label="prior_15m_low",
        sweep_extreme=99.4, entry_price=100.0, stop_price=99.0,
        tp1_price=102.0, tp2_price=104.0, confidence=0.9,
        reason="test", created_ms=1_000,
    )


def test_constructs_with_live_config(tmp_path):
    """Adapter constructs without touching the network when allow_live=False."""
    live_cfg = LiveConfig(allow_live=False)
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    assert mgr.has_exposure() is False
    assert mgr.pending_entry is None
    assert mgr.position is None


def test_refuses_to_submit_when_allow_live_false(tmp_path):
    """submit_entry must reject when allow_live=False (safety guard)."""
    live_cfg = LiveConfig(allow_live=False)
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    with pytest.raises(RuntimeError, match="allow_live"):
        mgr.submit_entry(_signal(), signal_id="abc", qty=1.0, risk_dollars=1.0)


def test_refuses_to_submit_when_no_agent_key():
    """Even with allow_live=true, missing agent key must abort."""
    live_cfg = LiveConfig(allow_live=True, agent_private_key="")
    with pytest.raises(RuntimeError, match="agent_private_key"):
        HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")


def test_refuses_when_notional_exceeds_cap():
    """Hard notional cap must reject oversized orders even live."""
    live_cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=50.0,  # very low cap
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    sig = _signal()  # qty=1.0, entry=100.0 -> notional=$100
    with pytest.raises(RuntimeError, match="max_notional"):
        mgr.submit_entry(sig, signal_id="abc", qty=1.0, risk_dollars=1.0)


def _live_cfg():
    return LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=10000.0,
    )


def test_submit_entry_places_post_only_limit(monkeypatch):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    mgr._exchange = fake_exchange  # inject mock

    sig = _signal()
    update = mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_PLACED
    assert update.signal_id == "abc"
    fake_exchange.order.assert_called_once()
    args, kwargs = fake_exchange.order.call_args
    # Verify the kwargs are correct
    assert (kwargs.get("name") == "BTC") or (len(args) > 0 and args[0] == "BTC")
    # is_buy must reflect side
    is_buy = kwargs.get("is_buy") if "is_buy" in kwargs else (args[1] if len(args) > 1 else None)
    assert is_buy is True
    # Post-only via Alo
    order_type = kwargs.get("order_type")
    if order_type is None and len(args) >= 5:
        order_type = args[4]
    assert order_type == {"limit": {"tif": "Alo"}}
    assert kwargs.get("limit_px") == 100.0
    assert kwargs.get("sz") == 0.001
    assert kwargs.get("reduce_only") is False
    # Pending entry was recorded
    assert mgr.pending_entry is not None
    assert mgr.pending_entry.qty == 0.001
    # pending_entry now also has external_oid
    assert mgr.pending_entry.external_oid == 12345


def test_submit_entry_raises_on_network_error():
    """Network/SDK exceptions must be translated to RuntimeError."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.side_effect = ConnectionError("simulated network failure")
    mgr._exchange = fake_exchange
    sig = _signal()
    with pytest.raises(RuntimeError, match="HL order"):
        mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    # Pending entry should NOT be set after a failed submission
    assert mgr.pending_entry is None


def test_pending_entry_cancels_on_expiry():
    mgr = HyperliquidOrderManager(StrategyConfig(pending_entry_expiry_sec=120), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_exchange.cancel_by_cloid.return_value = {"status": "ok"}
    mgr._exchange = fake_exchange

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    assert mgr.pending_entry is not None

    expired_ms = sig.created_ms + (200 * 1000)
    updates = mgr.on_trade(TradeEvent(ts_ms=expired_ms, price=100.5, size=1.0))

    fake_exchange.cancel_by_cloid.assert_called_once()
    call_args = fake_exchange.cancel_by_cloid.call_args
    assert call_args.args[0] == "BTC" or call_args.kwargs.get("name") == "BTC"
    # cloid argument: just verify it's a Cloid instance
    from hyperliquid.utils.types import Cloid
    cloid_arg = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("cloid")
    assert isinstance(cloid_arg, Cloid)
    assert mgr.pending_entry is None
    assert any(u.event_type == ExecEventType.ORDER_CANCELED for u in updates)


def test_pending_entry_does_not_cancel_before_expiry():
    """Before expiry, on_trade should not cancel or modify pending entry."""
    mgr = HyperliquidOrderManager(StrategyConfig(pending_entry_expiry_sec=120), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    mgr._exchange = fake_exchange

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    updates = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 60_000, price=100.5, size=1.0))
    assert mgr.pending_entry is not None
    fake_exchange.cancel_by_cloid.assert_not_called()


def test_pending_entry_clears_state_even_when_cancel_fails():
    """If HL cancel raises, local state still clears (HL may have already filled/canceled).
    The journal message must surface the failure so operators investigate."""
    mgr = HyperliquidOrderManager(StrategyConfig(pending_entry_expiry_sec=120), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_exchange.cancel_by_cloid.side_effect = ConnectionError("simulated network failure")
    mgr._exchange = fake_exchange

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    expired_ms = sig.created_ms + (200 * 1000)
    updates = mgr.on_trade(TradeEvent(ts_ms=expired_ms, price=100.5, size=1.0))

    # Cancel failed but local state cleared
    assert mgr.pending_entry is None
    # Event still emitted with failure marker
    canceled = [u for u in updates if u.event_type == ExecEventType.ORDER_CANCELED]
    assert len(canceled) == 1
    assert "CANCEL_FAILED" in canceled[0].message


def test_pending_entry_transitions_to_position_on_fill(monkeypatch):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    # First call: empty positions. Second: BTC long open at 100.0 size 0.001.
    fake_info.user_state.side_effect = [
        {"assetPositions": []},
        {"assetPositions": [{"position": {"coin": "BTC", "szi": "0.001", "entryPx": "100.0"}}]},
    ]
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)

    # First tick: not yet filled
    out1 = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 500, price=100.0, size=1.0))
    assert mgr.position is None
    assert mgr.pending_entry is not None

    # Second tick: HL reports the position now exists
    out2 = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 1500, price=100.0, size=1.0))
    assert mgr.position is not None
    assert mgr.position.entry_price == 100.0
    assert mgr.position.qty_initial == 0.001
    assert mgr.pending_entry is None
    assert any(u.event_type == ExecEventType.ENTRY_FILLED for u in out2)


def test_fill_detection_handles_szi_dict_format():
    """HL sometimes returns szi as a dict {base: '...'} rather than a string."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{
            "position": {"coin": "BTC", "szi": {"base": "0.001", "quote": "0.1"}, "entryPx": "100.0"},
        }],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 500, price=100.0, size=1.0))
    assert mgr.position is not None
    assert mgr.position.qty_initial == 0.001


def test_fill_detection_ignores_other_coins():
    """User state may include positions in other coins — must filter to our coin."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [
            {"position": {"coin": "ETH", "szi": "1.0", "entryPx": "2000.0"}},
            {"position": {"coin": "SOL", "szi": "10.0", "entryPx": "100.0"}},
        ],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    out = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 500, price=100.0, size=1.0))
    # No matching BTC position -> no transition
    assert mgr.position is None
    assert mgr.pending_entry is not None


def test_fill_detected_when_expiry_cancel_races_fill():
    """If a fill happens at the same tick as expiry, the fill should win
    (don't cancel a position we already have)."""
    mgr = HyperliquidOrderManager(StrategyConfig(pending_entry_expiry_sec=120), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{"position": {"coin": "BTC", "szi": "0.001", "entryPx": "100.0"}}],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    # Tick well past expiry — but HL says we're filled
    expired_ms = sig.created_ms + (200 * 1000)
    updates = mgr.on_trade(TradeEvent(ts_ms=expired_ms, price=100.5, size=1.0))

    # Fill detected -> position set, no cancel attempted, ENTRY_FILLED emitted (NOT ORDER_CANCELED)
    assert mgr.position is not None
    assert any(u.event_type == ExecEventType.ENTRY_FILLED for u in updates)
    assert all(u.event_type != ExecEventType.ORDER_CANCELED for u in updates)
    fake_exchange.cancel_by_cloid.assert_not_called()


def test_fill_detection_rejects_opposite_side_position():
    """LONG entry should NOT match a SHORT position in the same coin."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{"position": {"coin": "BTC", "szi": "-0.001", "entryPx": "100.0"}}],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()  # LONG signal
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    out = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 1500, price=100.0, size=1.0))
    # Wrong-side szi should NOT trigger fill
    assert mgr.position is None
    assert mgr.pending_entry is not None


def test_fill_polling_rate_limited_to_1_per_second():
    """Multiple trade ticks within 1s should only poll user_state once."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    fake_info.user_state.return_value = {"assetPositions": []}
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    # 5 ticks in 500ms -> only 1 poll
    base = sig.created_ms + 1000
    for offset in (0, 100, 200, 300, 400):
        mgr.on_trade(TradeEvent(ts_ms=base + offset, price=100.0, size=1.0))
    assert fake_info.user_state.call_count == 1
    # Tick at +1100ms (> 1s after first poll) -> second poll allowed
    mgr.on_trade(TradeEvent(ts_ms=base + 1100, price=100.0, size=1.0))
    assert fake_info.user_state.call_count == 2
