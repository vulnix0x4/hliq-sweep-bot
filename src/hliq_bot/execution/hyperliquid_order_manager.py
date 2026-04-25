from __future__ import annotations

import logging
from typing import Any

from hyperliquid.utils import constants

from hliq_bot.config import LiveConfig, StrategyConfig
from hliq_bot.models import (
    ExecEventType,
    ExecutionUpdate,
    OpenPosition,
    PendingEntry,
    Side,
    SweepSignal,
    TradeEvent,
)

log = logging.getLogger(__name__)


class HyperliquidOrderManager:
    """Live-execution adapter mirroring PaperOrderManager's public surface.

    Safety:
      - Refuses any operation if cfg.allow_live is False.
      - Refuses construction if agent_private_key is empty.
      - Refuses any order whose notional exceeds cfg.max_notional_per_trade.

    The SDK clients (Exchange, Info) are constructed lazily on first use to
    keep import-time and test-time clean (no network calls in __init__).
    """

    def __init__(
        self,
        strategy_cfg: StrategyConfig,
        live_cfg: LiveConfig,
        coin: str,
    ) -> None:
        if live_cfg.allow_live and not live_cfg.agent_private_key:
            raise RuntimeError(
                "LiveConfig.agent_private_key must be set when allow_live=True"
            )
        self.cfg = strategy_cfg
        self.live_cfg = live_cfg
        self.coin = coin
        self.pending_entry: PendingEntry | None = None
        self.position: OpenPosition | None = None
        self._last_trade_ms: int = 0
        self._pending_oid: int | None = None
        # SDK clients — constructed on first network operation.
        self._exchange: Any = None
        self._info: Any = None

    # ---- Public surface (mirrors PaperOrderManager) ----

    def has_exposure(self) -> bool:
        return self.pending_entry is not None or self.position is not None

    def submit_entry(
        self,
        signal: SweepSignal,
        signal_id: str,
        qty: float,
        risk_dollars: float,
    ) -> ExecutionUpdate:
        self._guard_live()
        notional = signal.entry_price * qty
        if notional > self.live_cfg.max_notional_per_trade:
            raise RuntimeError(
                f"order notional ${notional:.2f} exceeds "
                f"max_notional_per_trade=${self.live_cfg.max_notional_per_trade:.2f}"
            )

        exchange = self._ensure_exchange()
        is_buy = signal.side == Side.LONG
        # Alo = "Add Liquidity Only" = post-only. Refused if it would cross the spread.
        result = exchange.order(
            name=self.coin,
            is_buy=is_buy,
            sz=qty,
            limit_px=signal.entry_price,
            order_type={"limit": {"tif": "Alo"}},
            reduce_only=False,
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"HL order failed: {result}")

        statuses = result["response"]["data"]["statuses"]
        status = statuses[0] if statuses else {}
        if "error" in status:
            raise RuntimeError(f"HL order rejected: {status['error']}")

        oid = None
        if "resting" in status:
            oid = status["resting"].get("oid")
        elif "filled" in status:
            # Edge case: post-only filled at limit. Shouldn't happen with Alo but handle it.
            oid = status["filled"].get("oid")

        self.pending_entry = PendingEntry(
            signal_id=signal_id,
            side=signal.side,
            qty=qty,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            created_ms=signal.created_ms,
            expiry_sec=self.cfg.pending_entry_expiry_sec,
            level_label=signal.level_label,
            risk_dollars=risk_dollars,
            coin=self.coin,
        )
        # Stash the HL order id on the pending entry's signal_id mapping.
        self._pending_oid = oid

        return ExecutionUpdate(
            ts_ms=signal.created_ms,
            event_type=ExecEventType.ENTRY_PLACED,
            message=f"hl entry placed: {signal.side.value} qty={qty:.6f} @ {signal.entry_price:.2f} oid={oid}",
            signal_id=signal_id,
        )

    def on_trade(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        # Phase B will fill this in — for now no-op.
        return []

    # ---- Guards ----

    def _guard_live(self) -> None:
        if not self.live_cfg.allow_live:
            raise RuntimeError(
                "HyperliquidOrderManager called with allow_live=False; refusing to send."
            )

    def _ensure_exchange(self):
        if self._exchange is not None:
            return self._exchange
        import eth_account
        from hyperliquid.exchange import Exchange
        wallet = eth_account.Account.from_key(self.live_cfg.agent_private_key)
        self._exchange = Exchange(
            wallet,
            self.api_url,
            account_address=self.live_cfg.main_wallet_address or None,
        )
        return self._exchange

    @property
    def api_url(self) -> str:
        return constants.MAINNET_API_URL if self.live_cfg.network == "mainnet" else constants.TESTNET_API_URL
