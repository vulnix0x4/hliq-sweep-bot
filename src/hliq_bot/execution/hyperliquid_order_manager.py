from __future__ import annotations

import hashlib
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


def _cloid_from_signal_id(signal_id: str):
    """Derive a deterministic Cloid from our internal signal_id.

    Hashes the (non-hex) signal_id with sha256 and uses the first 16 bytes
    as a 128-bit int — the format Cloid.from_int expects.
    """
    from hyperliquid.utils.types import Cloid
    digest = hashlib.sha256(signal_id.encode("utf-8")).digest()
    return Cloid.from_int(int.from_bytes(digest[:16], "big"))


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
        cloid = _cloid_from_signal_id(signal_id)
        # Alo = "Add Liquidity Only" = post-only. Refused if it would cross the spread.
        try:
            result = exchange.order(
                name=self.coin,
                is_buy=is_buy,
                sz=qty,
                limit_px=signal.entry_price,
                order_type={"limit": {"tif": "Alo"}},
                reduce_only=False,
                cloid=cloid,
            )
        except Exception as exc:
            # Translate any SDK/network error to RuntimeError with context so the
            # caller (and the journal) sees a structured failure rather than an
            # uncaught exception that leaves pending_entry=None silently.
            raise RuntimeError(
                f"HL order submission failed for signal_id={signal_id} "
                f"side={signal.side.value} qty={qty} px={signal.entry_price}: {exc}"
            ) from exc
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
            external_oid=oid,
        )
        # Backup convenience mirror; source of truth is pending_entry.external_oid.
        self._pending_oid = oid

        return ExecutionUpdate(
            ts_ms=signal.created_ms,
            event_type=ExecEventType.ENTRY_PLACED,
            message=f"hl entry placed [{self.live_cfg.network}]: {signal.side.value} qty={qty:.6f} @ {signal.entry_price:.2f} oid={oid}",
            signal_id=signal_id,
        )

    def on_trade(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        self._last_trade_ms = trade.ts_ms
        updates: list[ExecutionUpdate] = []
        updates.extend(self._maybe_expire_pending(trade.ts_ms))
        # Phase B.3 will add fill detection here
        # Phase C will add position management here
        return updates

    def _maybe_expire_pending(self, now_ms: int) -> list[ExecutionUpdate]:
        if self.pending_entry is None:
            return []
        age_sec = (now_ms - self.pending_entry.created_ms) / 1000.0
        if age_sec < self.pending_entry.expiry_sec:
            return []
        pe = self.pending_entry
        # NOTE: If this cancel fails AND the order subsequently fills, the resulting
        # position will be detected by Phase B.3's user_state polling. Operators
        # should monitor bot.log for "HL cancel_by_cloid failed" and run
        # scripts/flatten_live.py if they see one until B.3 lands.
        cancel_failed = False
        cancel_err: str | None = None
        try:
            cloid = _cloid_from_signal_id(pe.signal_id)
            self._ensure_exchange().cancel_by_cloid(self.coin, cloid)
        except Exception as exc:
            cancel_failed = True
            cancel_err = f"{type(exc).__name__}: {exc}"
            log.warning(
                "HL cancel_by_cloid failed for signal_id=%s oid=%s: %s",
                pe.signal_id, self._pending_oid, exc,
                exc_info=True,
            )
        msg = (
            f"hl pending entry expired after {pe.expiry_sec}s: "
            f"{pe.side.value} @ {pe.entry_price:.2f} signal_id={pe.signal_id} oid={self._pending_oid}"
        )
        if cancel_failed:
            msg += f" CANCEL_FAILED ({cancel_err}) — verify on HL & flatten if needed"
        sid = pe.signal_id
        self.pending_entry = None
        self._pending_oid = None
        return [
            ExecutionUpdate(
                ts_ms=now_ms,
                event_type=ExecEventType.ORDER_CANCELED,
                message=msg,
                signal_id=sid,
            )
        ]

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
        log.info(
            "HL exchange initialized: network=%s api=%s agent=%s account=%s",
            self.live_cfg.network,
            self.api_url,
            wallet.address,
            self.live_cfg.main_wallet_address or "self",
        )
        return self._exchange

    @property
    def api_url(self) -> str:
        return constants.MAINNET_API_URL if self.live_cfg.network == "mainnet" else constants.TESTNET_API_URL
