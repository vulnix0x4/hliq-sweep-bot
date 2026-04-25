from __future__ import annotations

import logging
from typing import Any

from hliq_bot.config import LiveConfig, StrategyConfig
from hliq_bot.models import (
    ClosedTrade,
    ExecEventType,
    ExecutionUpdate,
    OpenPosition,
    PendingEntry,
    Side,
    SweepSignal,
    TradeEvent,
)

log = logging.getLogger(__name__)

_TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
_MAINNET_URL = "https://api.hyperliquid.xyz"


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
        # Phase B will fill this in — for now we just record a stub.
        raise NotImplementedError("submit_entry implementation pending Phase B")

    def on_trade(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        # Phase B will fill this in — for now no-op.
        return []

    # ---- Guards ----

    def _guard_live(self) -> None:
        if not self.live_cfg.allow_live:
            raise RuntimeError(
                "HyperliquidOrderManager called with allow_live=False; refusing to send."
            )

    @property
    def api_url(self) -> str:
        return _MAINNET_URL if self.live_cfg.network == "mainnet" else _TESTNET_URL
