from __future__ import annotations

from hliq_bot.config import StrategyConfig
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


class PaperOrderManager:
    def __init__(self, strategy_cfg: StrategyConfig) -> None:
        self.cfg = strategy_cfg
        self.pending_entry: PendingEntry | None = None
        self.position: OpenPosition | None = None
        self._last_trade_ms: int = 0

    def has_exposure(self) -> bool:
        return self.pending_entry is not None or self.position is not None

    def submit_entry(
        self,
        signal: SweepSignal,
        signal_id: str,
        qty: float,
        risk_dollars: float,
    ) -> ExecutionUpdate:
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
        )
        return ExecutionUpdate(
            ts_ms=signal.created_ms,
            event_type=ExecEventType.ENTRY_PLACED,
            message=f"paper entry placed: {signal.side.value} qty={qty:.6f} @ {signal.entry_price:.2f}",
            signal_id=signal_id,
        )

    def on_trade(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        self._last_trade_ms = trade.ts_ms
        updates: list[ExecutionUpdate] = []
        updates.extend(self._maybe_expire_pending(trade.ts_ms))
        updates.extend(self._maybe_fill_entry(trade))
        updates.extend(self._maybe_manage_open_position(trade))
        return updates

    def _maybe_expire_pending(self, now_ms: int) -> list[ExecutionUpdate]:
        if self.pending_entry is None:
            return []
        age_sec = (now_ms - self.pending_entry.created_ms) / 1000.0
        if age_sec < self.pending_entry.expiry_sec:
            return []
        msg = (
            f"pending entry expired after {self.pending_entry.expiry_sec}s: "
            f"{self.pending_entry.side.value} @ {self.pending_entry.entry_price:.2f}"
        )
        signal_id = self.pending_entry.signal_id
        self.pending_entry = None
        return [
            ExecutionUpdate(
                ts_ms=now_ms,
                event_type=ExecEventType.ORDER_CANCELED,
                message=msg,
                signal_id=signal_id,
            )
        ]

    def _maybe_fill_entry(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        if self.pending_entry is None:
            return []
        pe = self.pending_entry
        tol = self.cfg.entry_touch_tolerance_bps / 10_000.0
        touched = (pe.side == Side.LONG and trade.price <= pe.entry_price * (1 + tol)) or (
            pe.side == Side.SHORT and trade.price >= pe.entry_price * (1 - tol)
        )
        if not touched:
            return []

        entry_notional = pe.entry_price * pe.qty
        entry_fee = entry_notional * self.cfg.maker_fee_pct  # maker (limit posted at retest)
        self.position = OpenPosition(
            signal_id=pe.signal_id,
            side=pe.side,
            entry_price=pe.entry_price,
            stop_price=pe.stop_price,
            tp1_price=pe.tp1_price,
            tp2_price=pe.tp2_price,
            opened_ms=trade.ts_ms,
            qty_initial=pe.qty,
            qty_remaining=pe.qty,
            risk_dollars=max(pe.risk_dollars, self._position_risk(pe.entry_price, pe.stop_price, pe.qty)),
            realized_fees=entry_fee,
            best_price=pe.entry_price,
            worst_price=pe.entry_price,
        )
        self.pending_entry = None
        return [
            ExecutionUpdate(
                ts_ms=trade.ts_ms,
                event_type=ExecEventType.ENTRY_FILLED,
                message=f"paper entry filled: {pe.side.value} qty={pe.qty:.6f} @ {pe.entry_price:.2f}",
                signal_id=pe.signal_id,
            )
        ]

    def _maybe_manage_open_position(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        if self.position is None:
            return []
        p = self.position
        out: list[ExecutionUpdate] = []
        self._update_excursions(trade.price)

        if p.side == Side.LONG:
            stop_hit = trade.price <= p.stop_price
            tp1_hit = (not p.tp1_filled) and trade.price >= p.tp1_price
            tp2_hit = trade.price >= p.tp2_price
            pnl_fn = lambda exit_px, qty: (exit_px - p.entry_price) * qty
        else:
            stop_hit = trade.price >= p.stop_price
            tp1_hit = (not p.tp1_filled) and trade.price <= p.tp1_price
            tp2_hit = trade.price <= p.tp2_price
            pnl_fn = lambda exit_px, qty: (p.entry_price - exit_px) * qty

        if stop_hit:
            out.extend(self._close_position(trade.ts_ms, p.stop_price, "stop_loss", pnl_fn))
            return out

        if tp1_hit:
            qty = p.qty_remaining * 0.5
            pnl = pnl_fn(p.tp1_price, qty)
            p.realized_pnl += pnl
            p.qty_remaining -= qty
            p.tp1_filled = True
            # TP1 is a limit (maker) fill at the take-profit price.
            p.realized_fees += (p.tp1_price * qty) * self.cfg.maker_fee_pct
            # Reduce tail risk after first scale.
            p.stop_price = p.entry_price
            out.append(
                ExecutionUpdate(
                    ts_ms=trade.ts_ms,
                    event_type=ExecEventType.PARTIAL_TP,
                    message=f"tp1 partial: qty={qty:.6f} @ {p.tp1_price:.2f}",
                    signal_id=p.signal_id,
                )
            )

        self._maybe_promote_stop(trade.price)

        if tp2_hit and p.qty_remaining > 0:
            out.extend(self._close_position(trade.ts_ms, p.tp2_price, "tp2", pnl_fn))
            return out

        elapsed_sec = (trade.ts_ms - p.opened_ms) / 1000.0
        if elapsed_sec >= self.cfg.max_holding_sec:
            out.extend(self._close_position(trade.ts_ms, trade.price, "max_hold", pnl_fn))
            return out

        # Early exit: if deeply negative after early_exit_sec, cut losses
        if (
            self.cfg.early_exit_sec > 0
            and elapsed_sec >= self.cfg.early_exit_sec
            and p.risk_dollars > 0
        ):
            unrealized_r = self._unrealized_pnl(trade.price) / p.risk_dollars
            if unrealized_r <= self.cfg.early_exit_r_threshold:
                out.extend(self._close_position(trade.ts_ms, trade.price, "early_exit", pnl_fn))
                return out

        # Profit time stop: take green when you have it
        if (
            self.cfg.profit_take_sec > 0
            and elapsed_sec >= self.cfg.profit_take_sec
            and p.risk_dollars > 0
        ):
            unrealized_r = self._unrealized_pnl(trade.price) / p.risk_dollars
            if unrealized_r >= self.cfg.profit_take_min_r:
                out.extend(self._close_position(trade.ts_ms, trade.price, "profit_take", pnl_fn))
                return out

        if elapsed_sec >= self.cfg.time_stop_sec and self._unrealized_pnl(trade.price) <= 0:
            out.extend(self._close_position(trade.ts_ms, trade.price, "time_stop", pnl_fn))
            return out

        return out

    def _close_position(
        self,
        ts_ms: int,
        exit_price: float,
        reason: str,
        pnl_fn,
    ) -> list[ExecutionUpdate]:
        if self.position is None:
            return []
        p = self.position
        pnl_gross = p.realized_pnl + pnl_fn(exit_price, p.qty_remaining)
        # Final exit: tp2 fills as maker (limit), all other reasons are taker-style.
        final_exit_is_maker = reason == "tp2"
        final_fee_pct = self.cfg.maker_fee_pct if final_exit_is_maker else self.cfg.taker_fee_pct
        final_exit_fee = (exit_price * p.qty_remaining) * final_fee_pct
        fees_paid = p.realized_fees + final_exit_fee
        pnl_net = pnl_gross - fees_paid
        total_qty = p.qty_initial
        risk = max(p.risk_dollars, 1e-9)
        hold_sec = max(0.0, (ts_ms - p.opened_ms) / 1000.0)
        closed = ClosedTrade(
            signal_id=p.signal_id,
            side=p.side,
            entry_price=p.entry_price,
            exit_price=exit_price,
            qty=total_qty,
            pnl=pnl_net,
            pnl_gross=pnl_gross,
            fees_paid=fees_paid,
            risk_dollars=risk,
            r_multiple=pnl_net / risk,
            opened_ms=p.opened_ms,
            closed_ms=ts_ms,
            exit_reason=reason,
            mfe_pnl=self._price_to_pnl(p.best_price),
            mae_pnl=self._price_to_pnl(p.worst_price),
        )
        self.position = None
        return [
            ExecutionUpdate(
                ts_ms=ts_ms,
                event_type=ExecEventType.POSITION_CLOSED,
                message=f"position closed ({reason}): pnl={pnl_net:.2f} r={closed.r_multiple:.2f} hold_sec={hold_sec:.1f} fees={fees_paid:.2f}",
                signal_id=p.signal_id,
                closed_trade=closed,
            )
        ]

    def _position_risk(self, entry_price: float, stop_price: float, qty: float) -> float:
        return abs(entry_price - stop_price) * qty

    def _unrealized_pnl(self, mark_price: float) -> float:
        if self.position is None:
            return 0.0
        p = self.position
        if p.side == Side.LONG:
            return (mark_price - p.entry_price) * p.qty_remaining
        return (p.entry_price - mark_price) * p.qty_remaining

    def _price_to_pnl(self, price: float) -> float:
        if self.position is None:
            return 0.0
        p = self.position
        if p.side == Side.LONG:
            return (price - p.entry_price) * p.qty_initial
        return (p.entry_price - price) * p.qty_initial

    def _update_excursions(self, trade_price: float) -> None:
        if self.position is None:
            return
        p = self.position
        if p.best_price <= 0:
            p.best_price = p.entry_price
        if p.worst_price <= 0:
            p.worst_price = p.entry_price
        if p.side == Side.LONG:
            p.best_price = max(p.best_price, trade_price)
            p.worst_price = min(p.worst_price, trade_price)
            return
        p.best_price = min(p.best_price, trade_price)
        p.worst_price = max(p.worst_price, trade_price)

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
            elapsed = (self._last_trade_ms - p.opened_ms) / 1000.0 if self._last_trade_ms > 0 else 0.0
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
