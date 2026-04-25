from __future__ import annotations

import hashlib
import logging
from typing import Any

from hyperliquid.utils import constants

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
        self._last_fill_poll_ms: int = 0
        self._agent_addr: str | None = None
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
        # Detect fill FIRST: if HL filled our order, that supersedes expiry.
        updates.extend(self._maybe_detect_fill(trade.ts_ms))
        # Only expire if no fill happened (pending_entry still set).
        updates.extend(self._maybe_expire_pending(trade.ts_ms))
        updates.extend(self._maybe_manage_open_position(trade))
        return updates

    def _maybe_detect_fill(self, now_ms: int) -> list[ExecutionUpdate]:
        """Poll info.user_state to see if our pending entry has been filled."""
        pe = self.pending_entry
        if pe is None or self.position is not None:
            return []
        # Rate limit: at most 1 user_state poll per second per worker.
        if now_ms - self._last_fill_poll_ms < 1000:
            return []
        self._last_fill_poll_ms = now_ms
        info = self._ensure_info()
        address = self.live_cfg.main_wallet_address or self._agent_address()
        try:
            state = info.user_state(address)
        except Exception as exc:
            log.warning("HL user_state poll failed: %s", exc, exc_info=True)
            return []
        target_qty = pe.qty
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            if str(pos.get("coin", "")).upper() != self.coin.upper():
                continue
            szi_raw = pos.get("szi")
            # szi may be a string (signed) or a dict {"base": "...", ...}
            if isinstance(szi_raw, dict):
                szi = float(szi_raw.get("base", 0))
            else:
                szi = float(szi_raw or 0)
            # Validate side matches: LONG expects positive szi, SHORT expects negative.
            expected_long = pe.side == Side.LONG
            actual_long = szi > 0
            if expected_long != actual_long:
                log.warning(
                    "HL position side mismatch for %s: expected %s, got szi=%s — skipping fill detection",
                    self.coin, pe.side.value, szi,
                )
                continue
            # Only consider this a fill if at least 50% of target qty is on the book
            if abs(szi) < target_qty * 0.5:
                continue
            entry_px = float(pos.get("entryPx", pe.entry_price))
            # Live entries are post-only Alo (maker), so the fee is the maker
            # rate (negative = rebate received). Mirrors PaperOrderManager parity.
            entry_fee = entry_px * abs(szi) * self.cfg.maker_fee_pct
            self.position = OpenPosition(
                signal_id=pe.signal_id,
                side=pe.side,
                entry_price=entry_px,
                stop_price=pe.stop_price,
                tp1_price=pe.tp1_price,
                tp2_price=pe.tp2_price,
                opened_ms=now_ms,
                qty_initial=abs(szi),
                qty_remaining=abs(szi),
                risk_dollars=pe.risk_dollars,
                realized_fees=entry_fee,
                coin=self.coin,
                best_price=entry_px,
                worst_price=entry_px,
            )
            self.pending_entry = None
            self._pending_oid = None
            return [ExecutionUpdate(
                ts_ms=now_ms,
                event_type=ExecEventType.ENTRY_FILLED,
                message=f"hl entry filled [{self.live_cfg.network}]: {pe.side.value} qty={szi:.6f} @ {entry_px:.2f}",
                signal_id=pe.signal_id,
            )]
        return []

    def _ensure_info(self):
        if self._info is not None:
            return self._info
        from hyperliquid.info import Info
        self._info = Info(self.api_url, skip_ws=True)
        return self._info

    def reconcile_on_startup(self) -> None:
        """If a position already exists on HL (e.g. after a crash), restore local state.

        Does NOT recover stop/TP levels — those are lost. Best practice: manually
        flatten any open positions before restarting the bot. This method exists
        so the bot at least knows it has exposure (instead of placing duplicate entries).

        Sets a conservative wide stop at +/-2% from entry to limit unmanaged drift.
        Disables TP1/TP2 (sets them to entry_price so they can never fire).
        """
        if not self.live_cfg.allow_live:
            return
        info = self._ensure_info()
        address = self.live_cfg.main_wallet_address or self._agent_address()
        try:
            state = info.user_state(address)
        except Exception as exc:
            log.warning("HL user_state poll failed during startup reconcile: %s", exc, exc_info=True)
            return
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            if str(pos.get("coin", "")).upper() != self.coin.upper():
                continue
            szi_raw = pos.get("szi")
            szi = float(szi_raw.get("base", 0)) if isinstance(szi_raw, dict) else float(szi_raw or 0)
            if szi == 0:
                continue
            entry_px = float(pos.get("entryPx", 0))
            if entry_px <= 0:
                continue
            side = Side.LONG if szi > 0 else Side.SHORT
            # Conservative wide stop at -2% (LONG) or +2% (SHORT) to limit unmanaged drift
            wide_stop = entry_px * (0.98 if side == Side.LONG else 1.02)
            # Disable TP1/TP2 by setting them to entry — they can never fire
            self.position = OpenPosition(
                signal_id="reconciled",
                side=side,
                entry_price=entry_px,
                stop_price=wide_stop,
                tp1_price=entry_px,
                tp2_price=entry_px,
                opened_ms=int(self._last_trade_ms or 0),
                qty_initial=abs(szi),
                qty_remaining=abs(szi),
                risk_dollars=abs(szi * entry_px) * 0.02,  # rough — 2% of notional
                coin=self.coin,
                tp1_filled=True,  # mark as filled to prevent retry
                best_price=entry_px,
                worst_price=entry_px,
            )
            log.warning(
                "Reconciled existing %s position [%s]: %s qty=%.6f entry=%.2f stop=%.2f "
                "(manual cleanup recommended via scripts/flatten_live.py)",
                self.coin, self.live_cfg.network, side.value, abs(szi), entry_px, wide_stop,
            )
            break  # one position per coin in HL perp model

    def _agent_address(self) -> str:
        if self._agent_addr is None:
            import eth_account
            self._agent_addr = eth_account.Account.from_key(self.live_cfg.agent_private_key).address
        return self._agent_addr

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

    def _maybe_manage_open_position(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        if self.position is None:
            return []
        p = self.position
        self._update_excursions(trade.price)

        if p.side == Side.LONG:
            stop_hit = trade.price <= p.stop_price
            tp1_hit = (not p.tp1_filled) and trade.price >= p.tp1_price
            tp2_hit = trade.price >= p.tp2_price
        else:
            stop_hit = trade.price >= p.stop_price
            tp1_hit = (not p.tp1_filled) and trade.price <= p.tp1_price
            tp2_hit = trade.price <= p.tp2_price

        if stop_hit:
            return self._close_via_market(trade.ts_ms, "stop_loss", trade.price)

        out: list[ExecutionUpdate] = []
        if tp1_hit:
            out.extend(self._partial_close_tp1(trade.ts_ms, trade.price))

        self._maybe_promote_stop(trade.price)

        if tp2_hit and p.qty_remaining > 0:
            out.extend(self._close_via_market(trade.ts_ms, "tp2", trade.price))
            return out

        elapsed_sec = (trade.ts_ms - p.opened_ms) / 1000.0
        if elapsed_sec >= self.cfg.max_holding_sec:
            out.extend(self._close_via_market(trade.ts_ms, "max_hold", trade.price))
            return out

        # Early exit: if deeply negative after early_exit_sec, cut losses
        if (
            self.cfg.early_exit_sec > 0
            and elapsed_sec >= self.cfg.early_exit_sec
            and p.risk_dollars > 0
        ):
            unrealized_r = self._unrealized_pnl(trade.price) / p.risk_dollars
            if unrealized_r <= self.cfg.early_exit_r_threshold:
                out.extend(self._close_via_market(trade.ts_ms, "early_exit", trade.price))
                return out

        # Profit time stop: take green when you have it
        if (
            self.cfg.profit_take_sec > 0
            and elapsed_sec >= self.cfg.profit_take_sec
            and p.risk_dollars > 0
        ):
            unrealized_r = self._unrealized_pnl(trade.price) / p.risk_dollars
            if unrealized_r >= self.cfg.profit_take_min_r:
                out.extend(self._close_via_market(trade.ts_ms, "profit_take", trade.price))
                return out

        if elapsed_sec >= self.cfg.time_stop_sec and self._unrealized_pnl(trade.price) <= 0:
            out.extend(self._close_via_market(trade.ts_ms, "time_stop", trade.price))
            return out

        return out

    def _unrealized_pnl(self, mark_price: float) -> float:
        if self.position is None:
            return 0.0
        p = self.position
        if p.side == Side.LONG:
            return (mark_price - p.entry_price) * p.qty_remaining
        return (p.entry_price - mark_price) * p.qty_remaining

    def _maybe_promote_stop(self, trade_price: float) -> None:
        if self.position is None:
            return
        p = self.position

        # After TP1: aggressive trail
        if p.tp1_filled and self.cfg.trail_after_tp1:
            trail = max(0.0, min(1.0, self.cfg.trail_factor))
            if trail > 0:
                if p.side == Side.LONG:
                    new_stop = p.entry_price + (p.best_price - p.entry_price) * trail
                    if new_stop > p.stop_price:
                        p.stop_price = new_stop
                else:
                    new_stop = p.entry_price - (p.entry_price - p.best_price) * trail
                    if new_stop < p.stop_price:
                        p.stop_price = new_stop
            return

        # Pre-TP1: trail from entry to lock in any favorable movement
        if self.cfg.trail_from_entry:
            base_trail = max(0.0, min(1.0, self.cfg.trail_from_entry_factor))
            # Tighten trail in "runner" phase (after runner_trail_sec)
            elapsed = (
                (self._last_trade_ms - p.opened_ms) / 1000.0
                if self._last_trade_ms > 0
                else 0.0
            )
            if self.cfg.runner_trail_sec > 0 and elapsed >= self.cfg.runner_trail_sec:
                trail = max(base_trail, min(1.0, self.cfg.runner_trail_factor))
            else:
                trail = base_trail
            if trail > 0:
                if p.side == Side.LONG and p.best_price > p.entry_price:
                    new_stop = p.entry_price + (p.best_price - p.entry_price) * trail
                    if new_stop > p.stop_price:
                        p.stop_price = new_stop
                elif p.side == Side.SHORT and p.best_price < p.entry_price:
                    new_stop = p.entry_price - (p.entry_price - p.best_price) * trail
                    if new_stop < p.stop_price:
                        p.stop_price = new_stop

    def _partial_close_tp1(self, ts_ms: int, trade_price: float) -> list[ExecutionUpdate]:
        """Partial-close 50% of remaining qty at TP1 via market_close (taker fee)."""
        p = self.position
        if p is None or p.tp1_filled:
            return []
        partial_qty = p.qty_remaining * 0.5
        exchange = self._ensure_exchange()
        try:
            result = exchange.market_close(
                coin=self.coin,
                sz=partial_qty,
                slippage=0.005,
            )
        except Exception as exc:
            log.error(
                "HL TP1 partial market_close failed for %s qty=%s: %s",
                self.coin, partial_qty, exc, exc_info=True,
            )
            return []

        if result is None:
            log.warning(
                "HL TP1 partial market_close returned None for %s — HL has no matching position",
                self.coin,
            )
            return []

        fill_px = trade_price
        if isinstance(result, dict) and result.get("status") == "ok":
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses:
                st = statuses[0]
                if "filled" in st:
                    fill_px = float(st["filled"].get("avgPx", trade_price))

        # Realize the partial PnL + taker fee on the closed half
        if p.side == Side.LONG:
            partial_pnl = (fill_px - p.entry_price) * partial_qty
        else:
            partial_pnl = (p.entry_price - fill_px) * partial_qty
        partial_fee = (fill_px * partial_qty) * self.cfg.taker_fee_pct
        p.realized_pnl += partial_pnl
        p.realized_fees += partial_fee
        p.qty_remaining -= partial_qty
        p.tp1_filled = True
        # Reduce tail risk after first scale (move stop to BE).
        p.stop_price = p.entry_price
        return [ExecutionUpdate(
            ts_ms=ts_ms,
            event_type=ExecEventType.PARTIAL_TP,
            message=f"hl tp1 partial [{self.live_cfg.network}]: qty={partial_qty:.6f} @ {fill_px:.2f} (stop->BE)",
            signal_id=p.signal_id,
        )]

    def _close_via_market(self, ts_ms: int, reason: str, trade_price: float) -> list[ExecutionUpdate]:
        p = self.position
        if p is None:
            return []
        exchange = self._ensure_exchange()
        try:
            result = exchange.market_close(
                coin=self.coin,
                sz=p.qty_remaining,
                slippage=0.005,  # 0.5% slippage tolerance
            )
        except Exception as exc:
            log.error(
                "HL market_close failed for %s qty=%s: %s",
                self.coin, p.qty_remaining, exc, exc_info=True,
            )
            # Cannot proceed: leaving position in place. Operator must manually intervene.
            return []

        # The HL SDK returns None when no matching position exists on the
        # exchange — i.e. our local state has drifted out of sync with HL.
        # Clear local state cleanly and emit a phantom-close so the journal
        # records that we tried (but no realized trade exists to book).
        if result is None:
            log.warning(
                "HL market_close returned None for %s — HL has no matching position. "
                "Clearing local state to resync.",
                self.coin,
            )
            sid = p.signal_id
            self.position = None
            return [ExecutionUpdate(
                ts_ms=ts_ms,
                event_type=ExecEventType.POSITION_CLOSED,
                message=f"hl position cleared [{self.live_cfg.network}] ({reason} -> phantom_close): HL had no matching position",
                signal_id=sid,
                closed_trade=None,  # no realized trade — phantom close
            )]

        # Fall back to the trigger trade.price (not entry_price) when the HL
        # response is malformed: entry_price would silently understate losses
        # on a stop-out from a fast move.
        fill_px = trade_price
        if isinstance(result, dict) and result.get("status") == "ok":
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses:
                st = statuses[0]
                if "filled" in st:
                    fill_px = float(st["filled"].get("avgPx", trade_price))

        pnl_gross = (
            (fill_px - p.entry_price) * p.qty_remaining
            if p.side == Side.LONG
            else (p.entry_price - fill_px) * p.qty_remaining
        )
        pnl_gross += p.realized_pnl
        # Fees: taker on the market_close
        final_fee = (fill_px * p.qty_remaining) * self.cfg.taker_fee_pct
        fees_paid = p.realized_fees + final_fee
        pnl_net = pnl_gross - fees_paid
        risk = max(p.risk_dollars, 1e-9)
        hold_sec = max(0.0, (ts_ms - p.opened_ms) / 1000.0)
        closed = ClosedTrade(
            signal_id=p.signal_id,
            side=p.side,
            entry_price=p.entry_price,
            exit_price=fill_px,
            qty=p.qty_initial,
            pnl=pnl_net,
            pnl_gross=pnl_gross,
            fees_paid=fees_paid,
            risk_dollars=risk,
            r_multiple=pnl_net / risk,
            opened_ms=p.opened_ms,
            closed_ms=ts_ms,
            exit_reason=reason,
            coin=self.coin,
            mfe_pnl=self._price_to_pnl(p.best_price),
            mae_pnl=self._price_to_pnl(p.worst_price),
        )
        self.position = None
        return [ExecutionUpdate(
            ts_ms=ts_ms,
            event_type=ExecEventType.POSITION_CLOSED,
            message=f"hl position closed [{self.live_cfg.network}] ({reason}): pnl={pnl_net:.4f} fees={fees_paid:.4f} hold_sec={hold_sec:.1f}",
            signal_id=p.signal_id,
            closed_trade=closed,
        )]

    def _update_excursions(self, price: float) -> None:
        p = self.position
        if p is None:
            return
        if p.best_price <= 0:
            p.best_price = p.entry_price
        if p.worst_price <= 0:
            p.worst_price = p.entry_price
        if p.side == Side.LONG:
            p.best_price = max(p.best_price, price)
            p.worst_price = min(p.worst_price, price)
        else:
            p.best_price = min(p.best_price, price)
            p.worst_price = max(p.worst_price, price)

    def _price_to_pnl(self, price: float) -> float:
        p = self.position
        if p is None:
            return 0.0
        if p.side == Side.LONG:
            return (price - p.entry_price) * p.qty_initial
        return (p.entry_price - price) * p.qty_initial

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
