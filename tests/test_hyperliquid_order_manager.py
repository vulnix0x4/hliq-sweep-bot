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


def test_clamps_qty_when_notional_exceeds_cap():
    """When desired notional exceeds cap, the executor clamps qty (and emits a
    CLAMPED message) instead of raising — so the signal still trades, just smaller.

    Replaces the old "raises RuntimeError" behavior, which silently dropped every
    signal at small accounts because the worker thread's blanket `except Exception`
    swallowed it without journaling.
    """
    live_cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=50.0,  # cap = $50
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    mgr._sz_decimals = 5  # BTC precision (avoid network meta lookup)
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 999}}]}},
    }
    mgr._exchange = fake_exchange

    sig = _signal()  # entry=100.0
    # qty=1.0 -> desired notional=$100 (over $50 cap). Clamp to qty=0.5
    # -> notional=$50 (matches cap). After lot rounding (5 decimals): 0.5 unchanged.
    update = mgr.submit_entry(sig, signal_id="clamp_me", qty=1.0, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_PLACED
    assert "CLAMPED" in update.message
    # The order actually sent had qty=0.5 (clamped to fit $50 cap)
    sent_qty = fake_exchange.order.call_args.kwargs["sz"]
    assert sent_qty == pytest.approx(0.5, abs=1e-6)
    # Pending entry reflects clamped qty
    assert mgr.pending_entry.qty == pytest.approx(0.5, abs=1e-6)


def test_rejects_when_clamped_qty_below_hl_min_notional():
    """If the cap is so tight that clamping produces a sub-$10 order, emit
    ENTRY_REJECTED instead of placing a doomed order."""
    live_cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=5.0,  # below HL's $10 platform minimum
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    mgr._sz_decimals = 5
    fake_exchange = MagicMock()
    mgr._exchange = fake_exchange

    sig = _signal()
    update = mgr.submit_entry(sig, signal_id="too_small", qty=1.0, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_REJECTED
    assert "below_hl_min_notional" in update.message
    # No order placed
    fake_exchange.order.assert_not_called()
    # No pending entry recorded
    assert mgr.pending_entry is None


def test_rejects_when_qty_rounds_to_zero():
    """If the lot-rounded qty is zero (qty < lot), emit ENTRY_REJECTED."""
    live_cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=10000.0,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="SOL")
    mgr._sz_decimals = 2  # SOL: lot = 0.01
    fake_exchange = MagicMock()
    mgr._exchange = fake_exchange

    sig = _signal()  # entry=100.0
    # qty=0.005 < SOL lot 0.01 -> rounds down to 0
    update = mgr.submit_entry(sig, signal_id="sub_lot", qty=0.005, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_REJECTED
    assert "below_hl_min_lot" in update.message or "below_hl_min_notional" in update.message
    fake_exchange.order.assert_not_called()


def test_round_px_5sig_figs():
    """HL price quantization: max 5 sig figs. Confirmed via SDK error
    'float_to_wire causes rounding' on ETH 2244.5857142857144."""
    live_cfg = LiveConfig(allow_live=True, agent_private_key="0x" + "a" * 64,
                          main_wallet_address="0x" + "b" * 40)
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="ETH")

    # ETH at $2244.5857... → 5 sig figs → 2244.6
    assert mgr._round_px(2244.5857142857144) == pytest.approx(2244.6, abs=1e-9)
    # Already-precise value stays
    assert mgr._round_px(2244.5) == pytest.approx(2244.5, abs=1e-9)
    # BTC large value → 5 sig figs → integer (banker's rounding on .50 → even)
    assert mgr._round_px(76332.51) == pytest.approx(76333.0, abs=1e-9)
    assert mgr._round_px(76332.49) == pytest.approx(76332.0, abs=1e-9)
    # SOL small value → 5 sig figs → 4 decimals OK
    assert mgr._round_px(83.70153) == pytest.approx(83.702, abs=1e-9)
    # Whole-dollar value passes through
    assert mgr._round_px(100.0) == pytest.approx(100.0, abs=1e-9)
    # Very small (e.g. some token under $1)
    assert mgr._round_px(0.012345678) == pytest.approx(0.012346, abs=1e-9)
    # Zero / negative → unchanged (defensive, shouldn't happen in practice)
    assert mgr._round_px(0.0) == 0.0
    assert mgr._round_px(-1.0) == -1.0


def test_submit_entry_rounds_high_precision_price():
    """The price 2244.5857142857144 must NOT reach the SDK — it must be
    rounded to 2244.6 before exchange.order is called. Regression for the
    'float_to_wire causes rounding' bug observed live 2026-04-30 03:16."""
    live_cfg = LiveConfig(
        allow_live=True, agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40, max_notional_per_trade=10000.0,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="ETH")
    mgr._sz_decimals = 4
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 999}}]}},
    }
    mgr._exchange = fake_exchange

    sig = _signal_at_price(
        entry=2244.5857142857144,  # the actual live-bug price
        stop=2238.41, tp1=2252.78, tp2=2263.99,
    )
    update = mgr.submit_entry(sig, signal_id="precision_bug", qty=0.05, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_PLACED
    # Verify the SDK got the rounded price, not the raw one
    sent_px = fake_exchange.order.call_args.kwargs["limit_px"]
    assert sent_px == pytest.approx(2244.6, abs=1e-9)
    # Pending entry stores rounded price too (so fill detection compares correctly)
    assert mgr.pending_entry.entry_price == pytest.approx(2244.6, abs=1e-9)


def _signal_at_price(entry: float, stop: float, tp1: float, tp2: float) -> SweepSignal:
    """Helper for tests that need realistic price levels (BTC/ETH-scale)."""
    return SweepSignal(
        side=Side.LONG, level=entry, level_label="prior_15m_low",
        sweep_extreme=stop, entry_price=entry, stop_price=stop,
        tp1_price=tp1, tp2_price=tp2, confidence=0.9, reason="test", created_ms=1_000,
    )


def test_qty_rounded_down_to_coin_lot_size_btc():
    """BTC szDecimals=5 -> lot 0.00001. Sub-precision qty is rounded DOWN.
    Uses entry=$80,000 so a tiny lot-precision qty still clears $10 min notional."""
    live_cfg = LiveConfig(
        allow_live=True, agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40, max_notional_per_trade=10000.0,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    mgr._sz_decimals = 5
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 1}}]}},
    }
    mgr._exchange = fake_exchange

    sig = _signal_at_price(entry=80000.0, stop=79800.0, tp1=80400.0, tp2=80800.0)
    # qty=0.000234567 -> rounded down to 0.00023 (5 decimals, floored).
    # notional = 80000 * 0.00023 = $18.40 (above $10 min).
    update = mgr.submit_entry(sig, signal_id="round_btc", qty=0.000234567, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_PLACED, update.message
    sent_qty = fake_exchange.order.call_args.kwargs["sz"]
    assert sent_qty == pytest.approx(0.00023, abs=1e-7)


def test_qty_rounded_down_to_coin_lot_size_eth():
    """ETH szDecimals=4 -> lot 0.0001."""
    live_cfg = LiveConfig(
        allow_live=True, agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40, max_notional_per_trade=10000.0,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="ETH")
    mgr._sz_decimals = 4
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 2}}]}},
    }
    mgr._exchange = fake_exchange

    sig = _signal_at_price(entry=2300.0, stop=2293.0, tp1=2310.0, tp2=2320.0)
    # qty=0.012345 -> rounded down to 0.0123 (4 decimals).
    # notional = 2300 * 0.0123 = $28.29 (above $10 min).
    update = mgr.submit_entry(sig, signal_id="round_eth", qty=0.012345, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_PLACED, update.message
    sent_qty = fake_exchange.order.call_args.kwargs["sz"]
    assert sent_qty == pytest.approx(0.0123, abs=1e-6)


def test_clamp_recomputes_risk_dollars_proportionally():
    """When qty is clamped down, effective risk_dollars must shrink to match.

    Otherwise journal r-multiples are deflated (denominator stays at the
    original budget while numerator shrinks with the clamped qty), making
    every clamped trade look 1/N as good as it really is.
    """
    live_cfg = LiveConfig(
        allow_live=True, agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40, max_notional_per_trade=50.0,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    mgr._sz_decimals = 5
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 7}}]}},
    }
    mgr._exchange = fake_exchange

    # entry=100, stop=99 -> stop_distance=1. qty=1.0 (intended) -> intended risk=$1.
    # Clamp halves qty to 0.5 -> actual risk = 0.5 * 1 = $0.50.
    sig = _signal()
    update = mgr.submit_entry(sig, signal_id="r_calc", qty=1.0, risk_dollars=1.0)
    assert update.event_type == ExecEventType.ENTRY_PLACED
    # PendingEntry's risk_dollars must reflect the smaller post-clamp reality
    assert mgr.pending_entry.risk_dollars == pytest.approx(0.5, abs=1e-6)


def test_refuses_when_notional_exceeds_cap_legacy_remains_off():
    """Sanity: the cap clamp behavior is the only mode now. We do NOT raise
    on over-cap qty (that was the old behavior, which silently dropped signals)."""
    live_cfg = LiveConfig(
        allow_live=True, agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40, max_notional_per_trade=50.0,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    mgr._sz_decimals = 5
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 1}}]}},
    }
    mgr._exchange = fake_exchange

    sig = _signal()
    # Must NOT raise — must place a clamped order or emit ENTRY_REJECTED
    update = mgr.submit_entry(sig, signal_id="x", qty=10.0, risk_dollars=1.0)
    assert update.event_type in (ExecEventType.ENTRY_PLACED, ExecEventType.ENTRY_REJECTED)


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
    update = mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)

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
    assert kwargs.get("sz") == pytest.approx(0.2, abs=1e-9)
    assert kwargs.get("reduce_only") is False
    # Pending entry was recorded
    assert mgr.pending_entry is not None
    assert mgr.pending_entry.qty == pytest.approx(0.2, abs=1e-9)
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
        mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
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
    mgr._info = MagicMock()  # prevent real HTTP calls during _maybe_detect_fill

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
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
    mgr._info = MagicMock()  # prevent real HTTP calls during _maybe_detect_fill

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
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
    mgr._info = MagicMock()  # prevent real HTTP calls during _maybe_detect_fill

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
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
        {"assetPositions": [{"position": {"coin": "BTC", "szi": "0.2", "entryPx": "100.0"}}]},
    ]
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)

    # First tick: not yet filled
    out1 = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 500, price=100.0, size=1.0))
    assert mgr.position is None
    assert mgr.pending_entry is not None

    # Second tick: HL reports the position now exists
    out2 = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 1500, price=100.0, size=1.0))
    assert mgr.position is not None
    assert mgr.position.entry_price == 100.0
    assert mgr.position.qty_initial == pytest.approx(0.2, abs=1e-9)
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
            "position": {"coin": "BTC", "szi": {"base": "0.2", "quote": "0.1"}, "entryPx": "100.0"},
        }],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
    mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 500, price=100.0, size=1.0))
    assert mgr.position is not None
    assert mgr.position.qty_initial == pytest.approx(0.2, abs=1e-9)


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
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
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
        "assetPositions": [{"position": {"coin": "BTC", "szi": "0.2", "entryPx": "100.0"}}],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
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
        "assetPositions": [{"position": {"coin": "BTC", "szi": "-0.2", "entryPx": "100.0"}}],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()  # LONG signal
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
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
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
    # 5 ticks in 500ms -> only 1 poll
    base = sig.created_ms + 1000
    for offset in (0, 100, 200, 300, 400):
        mgr.on_trade(TradeEvent(ts_ms=base + offset, price=100.0, size=1.0))
    assert fake_info.user_state.call_count == 1
    # Tick at +1100ms (> 1s after first poll) -> second poll allowed
    mgr.on_trade(TradeEvent(ts_ms=base + 1100, price=100.0, size=1.0))
    assert fake_info.user_state.call_count == 2


def test_stop_loss_triggers_market_close():
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.001", "avgPx": "99.0"}}]}},
    }
    mgr._exchange = fake_exchange
    # Pre-seed an open long position
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=0.001, coin="BTC", best_price=100.0, worst_price=100.0,
    )

    # Price hits stop
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=98.5, size=1.0))
    fake_exchange.market_close.assert_called_once()
    assert mgr.position is None
    assert any(u.event_type == ExecEventType.POSITION_CLOSED for u in updates)
    closed = [u for u in updates if u.closed_trade is not None][0].closed_trade
    assert closed.exit_reason == "stop_loss"
    assert closed.pnl_gross < 0


def test_no_close_when_price_above_stop():
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=0.001, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    # Price above stop -> no close
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=99.5, size=1.0))
    fake_exchange.market_close.assert_not_called()
    assert mgr.position is not None


def test_short_stop_loss_triggers_when_price_above_stop():
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.001", "avgPx": "101.0"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.SHORT, entry_price=100.0,
        stop_price=101.0, tp1_price=98.0, tp2_price=96.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=0.001, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    # SHORT stop is ABOVE entry — price rising hits stop
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=101.5, size=1.0))
    fake_exchange.market_close.assert_called_once()
    assert mgr.position is None
    closed = [u for u in updates if u.closed_trade is not None][0].closed_trade
    assert closed.exit_reason == "stop_loss"


def test_entry_fee_recorded_on_live_fill():
    """Live fills must record the maker entry fee just like paper does. With
    retail-tier defaults (maker = +0.015%), this is a small positive fee paid."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._sz_decimals = 5
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{"position": {"coin": "BTC", "szi": "0.2", "entryPx": "100.0"}}],
    }
    mgr._exchange = fake_exchange
    mgr._info = fake_info
    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.2, risk_dollars=1.0)
    mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 1500, price=100.0, size=1.0))
    # entry_fee = entry_px * qty * maker_fee_pct = 100.0 * 0.2 * +0.00015 = +0.003
    expected = 100.0 * 0.2 * StrategyConfig().maker_fee_pct
    assert mgr.position is not None, "Fill should have been detected with szi >= 50% of qty"
    assert abs(mgr.position.realized_fees - expected) < 1e-9


def test_market_close_none_result_clears_state_with_phantom_close():
    """If HL has no matching position (returns None), clear local state cleanly."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = None  # SDK returns None for no-position
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=98.5, size=1.0))
    # Position cleared, ExecutionUpdate emitted, but closed_trade is None (phantom)
    assert mgr.position is None
    closed_events = [u for u in updates if u.event_type == ExecEventType.POSITION_CLOSED]
    assert len(closed_events) == 1
    assert closed_events[0].closed_trade is None
    assert "phantom_close" in closed_events[0].message


def test_malformed_close_response_keeps_position_open(tmp_path):
    """REGRESSION (bug observed live 2026-05-04): if the SDK returns status=ok
    but no "filled" status, the bot must NOT clear local state. The previous
    behavior fell through with fill_px=trade.price and orphaned the position
    on HL. Now the bot logs CRITICAL and retains state for retry."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{}]}},  # malformed — no "filled"
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=98.5, size=1.0))
    # Position must still be set (no orphan)
    assert mgr.position is not None
    # No POSITION_CLOSED event
    closed_events = [u for u in updates if u.closed_trade is not None]
    assert len(closed_events) == 0


def test_tp1_partial_close_at_target():
    """When price hits tp1, partial-close 50% via market_close + move stop to BE."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.0005", "avgPx": "102.0"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=102.5, size=1.0))
    # market_close called with HALF qty
    fake_exchange.market_close.assert_called_once()
    call_kwargs = fake_exchange.market_close.call_args.kwargs
    assert abs(call_kwargs.get("sz", 0) - 0.0005) < 1e-9
    # Position still open (other half), tp1_filled=True
    # Stop moved to BE then immediately trailed post-TP1 (trail_factor=0.5):
    # new_stop = entry + (best - entry) * 0.5 = 100 + (102.5 - 100) * 0.5 = 101.25
    assert mgr.position is not None
    assert mgr.position.tp1_filled is True
    assert mgr.position.stop_price >= 100.0  # at least BE
    assert abs(mgr.position.stop_price - 101.25) < 1e-9
    assert abs(mgr.position.qty_remaining - 0.0005) < 1e-9
    # PARTIAL_TP event emitted
    assert any(u.event_type == ExecEventType.PARTIAL_TP for u in updates)


def test_tp1_does_not_fire_twice():
    """Once tp1_filled=True, subsequent ticks at tp1 must NOT fire again."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.0005", "avgPx": "102.0"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
        tp1_filled=False,
    )
    mgr.on_trade(TradeEvent(ts_ms=2000, price=102.5, size=1.0))
    assert mgr.position.tp1_filled is True
    # Second tick at tp1 should NOT call market_close again
    fake_exchange.market_close.reset_mock()
    mgr.on_trade(TradeEvent(ts_ms=3000, price=102.7, size=1.0))
    fake_exchange.market_close.assert_not_called()


def test_short_tp1_fires_when_price_below_target():
    """SHORT tp1 should fire when price falls to tp1 (which is below entry)."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.0005", "avgPx": "98.0"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.SHORT, entry_price=100.0,
        stop_price=101.0, tp1_price=98.0, tp2_price=96.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=97.5, size=1.0))
    fake_exchange.market_close.assert_called_once()
    assert mgr.position.tp1_filled is True
    # Stop moved to BE then immediately trailed post-TP1 (SHORT trail_factor=0.5):
    # new_stop = entry - (entry - best) * 0.5 = 100 - (100 - 97.5) * 0.5 = 98.75
    assert mgr.position.stop_price <= 100.0  # at most BE
    assert abs(mgr.position.stop_price - 98.75) < 1e-9
    assert any(u.event_type == ExecEventType.PARTIAL_TP for u in updates)


def test_tp2_triggers_full_close(tmp_path):
    """When price hits tp2 with no remaining TP1 partial to do, close all via
    market_close. Pre-marks tp1_filled=True so only TP2 logic fires this tick
    (otherwise both fire and partial-fill mock semantics get tangled)."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.0005", "avgPx": "104.0"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=100.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.0005,  # half remaining post-TP1
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
        tp1_filled=True,  # TP1 already done
    )
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=104.5, size=1.0))
    fake_exchange.market_close.assert_called()
    assert mgr.position is None
    closed = [u for u in updates if u.closed_trade is not None][0].closed_trade
    assert closed.exit_reason == "tp2"
    assert closed.pnl_gross > 0


def test_max_hold_triggers_after_max_holding_sec():
    cfg = StrategyConfig(max_holding_sec=100)
    mgr = HyperliquidOrderManager(cfg, _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.001", "avgPx": "100.5"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    # 101 sec elapsed -> max_hold triggers
    updates = mgr.on_trade(TradeEvent(ts_ms=1000 + 101 * 1000, price=100.5, size=1.0))
    fake_exchange.market_close.assert_called()
    closed = [u for u in updates if u.closed_trade is not None][0].closed_trade
    assert closed.exit_reason == "max_hold"


def test_early_exit_triggers_when_deeply_negative(tmp_path):
    cfg = StrategyConfig(
        early_exit_sec=120,
        early_exit_r_threshold=-0.5,
        max_holding_sec=1800,
        time_stop_sec=600,
    )
    mgr = HyperliquidOrderManager(cfg, _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"
    fake_exchange = MagicMock()
    # Match the position's qty (1.0) so this is a full-fill close, not partial.
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "1.0", "avgPx": "99.5"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=98.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=1.0, qty_remaining=1.0,
        risk_dollars=2.0, coin="BTC", best_price=100.0, worst_price=98.5,
    )
    # 130 sec elapsed, price=98.8 -> r_unrealized = (98.8-100)*1.0 / 2.0 = -0.6 (< -0.5)
    updates = mgr.on_trade(TradeEvent(ts_ms=1000 + 130 * 1000, price=98.8, size=1.0))
    fake_exchange.market_close.assert_called()
    closed = [u for u in updates if u.closed_trade is not None][0].closed_trade
    assert closed.exit_reason == "early_exit"


def test_time_stop_triggers_when_unprofitable_after_time_limit():
    cfg = StrategyConfig(
        time_stop_sec=240,
        max_holding_sec=1800,
        early_exit_sec=120,
        early_exit_r_threshold=-2.0,  # disabled effectively
        profit_take_sec=0,
    )
    mgr = HyperliquidOrderManager(cfg, _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.001", "avgPx": "99.9"}}]}},
    }
    mgr._exchange = fake_exchange
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=99.9,
    )
    # 250 sec elapsed, price=99.9 (slightly negative) -> time_stop triggers
    updates = mgr.on_trade(TradeEvent(ts_ms=1000 + 250 * 1000, price=99.9, size=1.0))
    fake_exchange.market_close.assert_called()
    closed = [u for u in updates if u.closed_trade is not None][0].closed_trade
    assert closed.exit_reason == "time_stop"


def test_trail_from_entry_promotes_stop_when_price_above_entry():
    cfg = StrategyConfig(
        trail_from_entry=True,
        trail_from_entry_factor=0.5,
        runner_trail_sec=1000,  # disabled in this test
        max_holding_sec=1800,
    )
    mgr = HyperliquidOrderManager(cfg, _live_cfg(), coin="BTC")
    mgr._exchange = MagicMock()
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=110.0, tp2_price=120.0,  # TPs far away
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
    )
    mgr._last_trade_ms = 1000
    # Price moves up to 102 -> best=102, trail = entry + (best-entry)*0.5 = 100 + 1 = 101
    mgr.on_trade(TradeEvent(ts_ms=2000, price=102.0, size=1.0))
    assert mgr.position is not None
    assert abs(mgr.position.stop_price - 101.0) < 1e-9


def test_post_tp1_trail_uses_trail_factor():
    cfg = StrategyConfig(
        trail_after_tp1=True,
        trail_factor=0.5,
        max_holding_sec=1800,
    )
    mgr = HyperliquidOrderManager(cfg, _live_cfg(), coin="BTC")
    mgr._exchange = MagicMock()
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=100.0, tp1_price=102.0, tp2_price=120.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.0005,
        risk_dollars=1.0, coin="BTC", best_price=103.0, worst_price=100.0,
        tp1_filled=True,  # already partial-closed at TP1
    )
    mgr._last_trade_ms = 2000
    # Price at 103 (already best); trail = entry + (103-100)*0.5 = 100 + 1.5 = 101.5
    mgr.on_trade(TradeEvent(ts_ms=3000, price=103.0, size=1.0))
    assert mgr.position is not None
    assert abs(mgr.position.stop_price - 101.5) < 1e-9


def test_reconciles_existing_long_position_on_startup(tmp_path):
    """If HL has an existing LONG position and no persisted state, reconstruct
    OpenPosition with the tightened 0.5% wide-stop fallback (down from 2%)."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"  # no file → wide-stop fallback
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{
            "position": {"coin": "BTC", "szi": "0.005", "entryPx": "100.0"},
        }],
    }
    fake_info.open_orders.return_value = []
    mgr._info = fake_info
    mgr.reconcile_on_startup()
    assert mgr.position is not None
    assert mgr.position.qty_initial == 0.005
    assert mgr.position.entry_price == 100.0
    assert mgr.position.side == Side.LONG
    # Tightened wide stop: -0.5% from entry for LONG (was -2%)
    assert abs(mgr.position.stop_price - 99.5) < 0.01
    # TP1/TP2 disabled (set to entry so they never fire)
    assert mgr.position.tp1_price == 100.0
    assert mgr.position.tp2_price == 100.0


def test_reconciles_existing_short_position_on_startup(tmp_path):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{
            "position": {"coin": "BTC", "szi": "-0.005", "entryPx": "100.0"},
        }],
    }
    fake_info.open_orders.return_value = []
    mgr._info = fake_info
    mgr.reconcile_on_startup()
    assert mgr.position is not None
    assert mgr.position.side == Side.SHORT
    assert mgr.position.qty_initial == 0.005
    # Tightened wide stop: +0.5% from entry for SHORT
    assert abs(mgr.position.stop_price - 100.5) < 0.01


def test_reconcile_no_op_when_no_position():
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_info = MagicMock()
    fake_info.user_state.return_value = {"assetPositions": []}
    fake_info.open_orders.return_value = []
    mgr._info = fake_info
    mgr.reconcile_on_startup()
    assert mgr.position is None


def test_reconcile_skipped_when_allow_live_false():
    """Paper-mode default: reconcile_on_startup must do NOTHING."""
    mgr = HyperliquidOrderManager(StrategyConfig(), LiveConfig(allow_live=False), coin="BTC")
    fake_info = MagicMock()
    mgr._info = fake_info
    mgr.reconcile_on_startup()
    fake_info.user_state.assert_not_called()
    assert mgr.position is None


def test_reconcile_filters_other_coins():
    """Only matching coin triggers reconciliation."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [
            {"position": {"coin": "ETH", "szi": "1.0", "entryPx": "2000.0"}},
            {"position": {"coin": "SOL", "szi": "10.0", "entryPx": "100.0"}},
        ],
    }
    fake_info.open_orders.return_value = []
    mgr._info = fake_info
    mgr.reconcile_on_startup()
    assert mgr.position is None


def test_reconcile_uses_wall_clock_for_opened_ms(monkeypatch):
    """opened_ms must use wall clock, not _last_trade_ms (which is 0 at startup).
    Otherwise the first trade tick would trigger max_hold/time_stop immediately."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{"position": {"coin": "BTC", "szi": "0.2", "entryPx": "100.0"}}],
    }
    fake_info.open_orders.return_value = []
    mgr._info = fake_info

    # Mock time.time() to a known value
    import hliq_bot.execution.hyperliquid_order_manager as adapter_mod
    monkeypatch.setattr(adapter_mod.time, "time", lambda: 1_700_000_000.0)
    mgr.reconcile_on_startup()
    assert mgr.position is not None
    assert mgr.position.opened_ms == 1_700_000_000_000  # ms


def test_reconcile_cancels_stale_resting_orders():
    """Any resting orders for our coin at restart must be cancelled to prevent
    double-entry on the next signal."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.bulk_cancel.return_value = {"status": "ok"}
    fake_info = MagicMock()
    fake_info.user_state.return_value = {"assetPositions": []}
    fake_info.open_orders.return_value = [
        {"coin": "BTC", "oid": 12345},
        {"coin": "BTC", "oid": 12346},
        {"coin": "ETH", "oid": 99999},  # other coin -- ignored
    ]
    mgr._exchange = fake_exchange
    mgr._info = fake_info
    mgr.reconcile_on_startup()
    fake_exchange.bulk_cancel.assert_called_once()
    cancelled = fake_exchange.bulk_cancel.call_args.args[0]
    assert len(cancelled) == 2
    assert all(o["coin"] == "BTC" for o in cancelled)


def test_deadman_refresh_pushes_cancel_timer():
    """refresh_deadman calls schedule_cancel(now + deadman_cancel_sec*1000)."""
    cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=10000.0,
        deadman_cancel_sec=60,
        deadman_refresh_sec=30,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), cfg, coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.schedule_cancel.return_value = {"status": "ok"}
    mgr._exchange = fake_exchange

    mgr.refresh_deadman(now_ms=1_000_000_000_000)
    fake_exchange.schedule_cancel.assert_called_once()
    arg = fake_exchange.schedule_cancel.call_args.args[0]
    assert arg == 1_000_000_000_000 + 60_000


def test_deadman_refresh_paper_no_op():
    """In paper mode (allow_live=False), refresh_deadman must not touch the SDK."""
    mgr = HyperliquidOrderManager(StrategyConfig(), LiveConfig(allow_live=False), coin="BTC")
    fake_exchange = MagicMock()
    mgr._exchange = fake_exchange
    mgr.refresh_deadman(now_ms=1_000_000_000_000)
    fake_exchange.schedule_cancel.assert_not_called()


def test_should_refresh_deadman_returns_false_before_interval():
    """should_refresh_deadman returns True only after deadman_refresh_sec elapsed."""
    cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=10000.0,
        deadman_refresh_sec=30,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), cfg, coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.schedule_cancel.return_value = {"status": "ok"}
    mgr._exchange = fake_exchange

    # First call: never refreshed yet, should return True
    assert mgr.should_refresh_deadman(now_ms=1_000_000_000_000) is True
    # Refresh
    mgr.refresh_deadman(now_ms=1_000_000_000_000)
    # 10s later: too soon
    assert mgr.should_refresh_deadman(now_ms=1_000_000_000_000 + 10_000) is False
    # 31s later: due
    assert mgr.should_refresh_deadman(now_ms=1_000_000_000_000 + 31_000) is True


def test_should_refresh_deadman_paper_returns_false():
    mgr = HyperliquidOrderManager(StrategyConfig(), LiveConfig(allow_live=False), coin="BTC")
    assert mgr.should_refresh_deadman(now_ms=1_000_000_000_000) is False


def test_refresh_deadman_swallows_sdk_failure():
    """If schedule_cancel raises, refresh_deadman logs but doesn't propagate.
    The next refresh attempt will retry."""
    cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=10000.0,
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), cfg, coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.schedule_cancel.side_effect = ConnectionError("network failure")
    mgr._exchange = fake_exchange
    # Should not raise
    mgr.refresh_deadman(now_ms=1_000_000_000_000)


# ============================================================================
# Phantom-close-in-reverse regression tests (bug observed live 2026-05-04)
# ============================================================================
# Failure mode: market_close returns status=ok but with no "filled" key in
# statuses, the bot used to fall through with fill_px=trade_price and clear
# local position state. HL kept the position open as an unmanaged orphan.
# These tests assert the new behavior: do NOT clear local state when the SDK
# response doesn't confirm a fill.


def _open_position(mgr, qty=0.013, entry=2280.0, stop=2275.0):
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="t1", side=Side.LONG, entry_price=entry, stop_price=stop,
        tp1_price=entry + 10, tp2_price=entry + 20, opened_ms=1_000_000,
        qty_initial=qty, qty_remaining=qty, risk_dollars=qty * (entry - stop),
        coin="ETH", best_price=entry, worst_price=entry,
    )


def test_close_via_market_keeps_position_when_response_has_no_filled(tmp_path):
    """Regression: market_close returns status=ok with empty/error statuses → do NOT clear state."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._sz_decimals = 4
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    fake_exchange = MagicMock()
    # SDK returns ok-shaped response but with no "filled" status (e.g. error in statuses)
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"error": "Reduce only failed"}]}},
    }
    mgr._exchange = fake_exchange
    _open_position(mgr)
    # Trigger stop_loss (price below stop)
    updates = mgr.on_trade(TradeEvent(ts_ms=2_000_000, price=2270.0, size=1.0))
    # CRITICAL: position must remain set so the next tick will retry
    assert mgr.position is not None, "position cleared despite no fill confirmation — orphan risk"
    # No POSITION_CLOSED event emitted
    closed_events = [u for u in updates if u.event_type == ExecEventType.POSITION_CLOSED]
    assert len(closed_events) == 0


def test_close_via_market_keeps_position_when_statuses_empty(tmp_path):
    """Same regression with empty statuses list."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._sz_decimals = 4
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": []}},
    }
    mgr._exchange = fake_exchange
    _open_position(mgr)
    mgr.on_trade(TradeEvent(ts_ms=2_000_000, price=2270.0, size=1.0))
    assert mgr.position is not None


def test_close_via_market_clears_position_when_filled_confirmed(tmp_path):
    """Sanity: when filled status IS present, position clears normally."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._sz_decimals = 4
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.013", "avgPx": "2270.5"}}]}},
    }
    mgr._exchange = fake_exchange
    _open_position(mgr)
    updates = mgr.on_trade(TradeEvent(ts_ms=2_000_000, price=2270.0, size=1.0))
    assert mgr.position is None
    closed_events = [u for u in updates if u.event_type == ExecEventType.POSITION_CLOSED]
    assert len(closed_events) == 1
    assert closed_events[0].closed_trade.exit_reason == "stop_loss"


def test_close_via_market_partial_fill_keeps_residual(tmp_path):
    """When HL fills less than requested, the residual qty stays open for the next tick."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._sz_decimals = 4
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    fake_exchange = MagicMock()
    # Requested 0.013, only 0.005 filled
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.005", "avgPx": "2270.5"}}]}},
    }
    mgr._exchange = fake_exchange
    _open_position(mgr, qty=0.013)
    updates = mgr.on_trade(TradeEvent(ts_ms=2_000_000, price=2270.0, size=1.0))
    # Position must remain with residual (~0.008)
    assert mgr.position is not None
    assert mgr.position.qty_remaining == pytest.approx(0.008, abs=1e-6)
    # Should emit a PARTIAL_TP event (not a full close)
    partials = [u for u in updates if u.event_type == ExecEventType.PARTIAL_TP]
    assert len(partials) == 1
    assert "partial close" in partials[0].message.lower() or "qty=0.005" in partials[0].message


def test_extract_fill_returns_none_for_non_filled_responses():
    """The shared parser must return None for any non-filled response."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    # None
    assert mgr._extract_fill(None, default_px=100.0) is None
    # Not a dict
    assert mgr._extract_fill("ok", default_px=100.0) is None
    # status != ok
    assert mgr._extract_fill({"status": "err"}, default_px=100.0) is None
    # Empty statuses
    assert mgr._extract_fill({"status": "ok", "response": {"data": {"statuses": []}}}, default_px=100.0) is None
    # No "filled" in status
    assert mgr._extract_fill(
        {"status": "ok", "response": {"data": {"statuses": [{"error": "x"}]}}},
        default_px=100.0,
    ) is None
    # filled with 0 qty
    assert mgr._extract_fill(
        {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": "0", "avgPx": "100"}}]}}},
        default_px=100.0,
    ) is None
    # Valid filled response
    qty_px = mgr._extract_fill(
        {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": "0.013", "avgPx": "2270.5"}}]}}},
        default_px=100.0,
    )
    assert qty_px == (0.013, 2270.5)


# ============================================================================
# Position state persistence (daily-restart workaround)
# ============================================================================


def test_persist_position_writes_file_on_open(tmp_path):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"
    _open_position(mgr, entry=80000, stop=79800)
    mgr._persist_position()
    assert mgr._state_path.exists()
    import json
    data = json.loads(mgr._state_path.read_text())
    assert data["coin"] == "BTC"
    assert data["entry_price"] == 80000
    assert data["stop_price"] == 79800
    assert data["side"] == "long"
    assert "ts_ms" in data


def test_clear_persisted_state_removes_file(tmp_path):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"
    _open_position(mgr)
    mgr._persist_position()
    assert mgr._state_path.exists()
    mgr.position = None
    mgr._persist_position()  # should clear since position is None
    assert not mgr._state_path.exists()


def test_try_restore_from_persisted_when_fresh_and_matching(tmp_path):
    """Persisted state matching HL position → restore exactly."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    _open_position(mgr, qty=0.013, entry=2280, stop=2275)
    mgr.position.tp1_filled = True  # has been managed
    mgr.position.realized_pnl = 0.05
    mgr._persist_position()
    saved_pos = mgr.position
    mgr.position = None  # simulate restart
    # HL still has matching position
    hl_pos = {"szi": "0.013", "entryPx": "2280"}
    restored = mgr._try_restore_from_persisted(hl_pos)
    assert restored is True
    assert mgr.position is not None
    assert mgr.position.entry_price == 2280
    assert mgr.position.stop_price == 2275
    assert mgr.position.tp1_filled is True
    assert mgr.position.realized_pnl == 0.05


def test_try_restore_falls_through_when_state_stale(tmp_path):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    # Manually write stale state (10 min old)
    import json, time as tm
    stale_ts = int(tm.time() * 1000) - 600_000  # 10 min ago
    stale_data = {
        "ts_ms": stale_ts, "coin": "ETH", "signal_id": "old", "side": "long",
        "entry_price": 2280.0, "stop_price": 2275.0, "tp1_price": 2290.0, "tp2_price": 2300.0,
        "qty_initial": 0.013, "qty_remaining": 0.013, "tp1_filled": False,
        "best_price": 2280.0, "worst_price": 2280.0, "realized_pnl": 0.0,
        "realized_fees": 0.004, "risk_dollars": 0.065, "opened_ms": stale_ts,
    }
    mgr._state_path.write_text(json.dumps(stale_data))
    hl_pos = {"szi": "0.013", "entryPx": "2280"}
    assert mgr._try_restore_from_persisted(hl_pos) is False
    assert mgr.position is None


def test_try_restore_falls_through_when_qty_mismatches_hl(tmp_path):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    _open_position(mgr, qty=0.013)
    mgr._persist_position()
    mgr.position = None
    # HL has different size
    hl_pos = {"szi": "0.020", "entryPx": "2280"}
    assert mgr._try_restore_from_persisted(hl_pos) is False
    assert mgr.position is None


def test_try_restore_falls_through_when_hl_has_no_position(tmp_path):
    """Persisted state but HL has no position → clear stale file, fall through."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="ETH")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "ETH.json"
    _open_position(mgr)
    mgr._persist_position()
    mgr.position = None
    assert mgr._try_restore_from_persisted(None) is False
    # Stale file should be cleaned up
    assert not mgr._state_path.exists()


def test_persist_position_skips_redundant_writes(tmp_path):
    """No filesystem write if state hasn't changed (avoids unnecessary IO every tick)."""
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    mgr._state_dir = tmp_path
    mgr._state_path = tmp_path / "BTC.json"
    _open_position(mgr)
    mgr._persist_position()
    mtime1 = mgr._state_path.stat().st_mtime_ns
    import time as tm
    tm.sleep(0.01)  # ensure mtime resolution
    mgr._persist_position()  # nothing changed
    mtime2 = mgr._state_path.stat().st_mtime_ns
    assert mtime1 == mtime2  # not rewritten

    # Now mutate state and persist again
    mgr.position.stop_price = mgr.position.stop_price + 1.0
    mgr._persist_position()
    mtime3 = mgr._state_path.stat().st_mtime_ns
    assert mtime3 > mtime2  # written this time
