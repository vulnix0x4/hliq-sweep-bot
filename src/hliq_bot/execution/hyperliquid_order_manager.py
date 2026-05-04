from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
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

# Hyperliquid platform-wide minimum order notional (verified via fees API + docs 2026-04-28).
HL_MIN_ORDER_NOTIONAL = 10.0

# Maximum age (seconds) for a persisted position-state file to be trusted on restart.
# The host has a daily docker restart pattern around 11:02 UTC; warm-start usually
# completes within 60s. 5 minutes leaves headroom for slower restarts while still
# rejecting state from any restart that genuinely lost track of the position.
PERSISTED_STATE_MAX_AGE_SEC = 300


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
        self._last_deadman_refresh_ms: int = 0
        # SDK clients — constructed on first network operation.
        self._exchange: Any = None
        self._info: Any = None
        # HL meta cache — szDecimals per coin (lot precision). Loaded lazily.
        # Tests can set this directly to skip the network call.
        self._sz_decimals: int | None = None
        # Position state persistence — works around host's daily docker restart
        # pattern (~11:02 UTC). Bot writes full position state to this file on
        # every on_trade tick; on restart, reconcile_on_startup loads from it
        # before falling back to the wide-stop reconcile path. Tests can override
        # the directory by setting state_dir env var or directly mutating
        # _state_path. Defaults under runtime/active_positions/<COIN>.json.
        state_dir_env = os.getenv("BOT_STATE_DIR", "runtime/active_positions")
        self._state_dir = Path(state_dir_env)
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            # Non-fatal: if we can't write state, daily-restart workaround degrades
            # gracefully to the existing wide-stop reconcile.
            log.warning("Could not create state dir %s: %s", self._state_dir, exc)
        self._state_path = self._state_dir / f"{self.coin}.json"
        # Track the last persisted state to avoid redundant writes on every tick.
        self._last_persisted_hash: str | None = None

    # ---- Public surface (mirrors PaperOrderManager) ----

    def has_exposure(self) -> bool:
        return self.pending_entry is not None or self.position is not None

    def _coin_sz_decimals(self) -> int:
        """szDecimals for self.coin (e.g. BTC=5, ETH=4, SOL=2). Cached after first lookup."""
        if self._sz_decimals is not None:
            return self._sz_decimals
        info = self._ensure_info()
        try:
            meta = info.meta()
            for u in meta.get("universe", []):
                if str(u.get("name", "")).upper() == self.coin.upper():
                    self._sz_decimals = int(u.get("szDecimals", 4))
                    return self._sz_decimals
        except Exception as exc:
            log.warning("Failed to fetch HL meta for %s szDecimals: %s", self.coin, exc, exc_info=True)
        # Conservative fallback when meta is unavailable: 4 decimals (matches ETH).
        # Never silently caches the fallback so a later success can recompute.
        return 4

    def _round_qty_down(self, qty: float) -> float:
        """Round qty DOWN to the coin's lot size (10^-szDecimals).

        Always rounds down (not nearest) so we never accidentally exceed the
        intended notional cap or risk budget after rounding.
        """
        decimals = self._coin_sz_decimals()
        multiplier = 10 ** decimals
        # Floor in integer space to avoid float rounding wobble at the lot boundary.
        return int(qty * multiplier) / multiplier

    def _round_px(self, price: float) -> float:
        """Round a price to HL's wire-compatible tick.

        HL price quantization rules (verified empirically — observed
        'float_to_wire causes rounding' SDK error on ETH 2244.5857142857144):
          - Max 5 significant figures
          - Max (6 - szDecimals) decimal places
          - Both rules apply; the more restrictive one wins.

        For our perp universe (BTC szDecimals=5, ETH=4, SOL=2), the 5-sig-fig
        rule is the binding constraint at typical price levels. Python's
        `:.5g` general format implements 5 sig figs cleanly and rounds to
        the nearest representable value, which is what HL expects on the wire.
        """
        if price <= 0:
            return price
        return float(f"{price:.5g}")

    def submit_entry(
        self,
        signal: SweepSignal,
        signal_id: str,
        qty: float,
        risk_dollars: float,
    ) -> ExecutionUpdate:
        self._guard_live()

        # Step 0: Round entry price to HL's wire-compatible tick FIRST. The
        # SDK's float_to_wire rejects high-precision floats; strategy-derived
        # prices (level offsets, VWAP-relative entries) often have many
        # decimals. Round once up-front and use rounded_px everywhere
        # downstream so the cap-clamp / notional / risk math all match what
        # HL will actually fill at.
        rounded_px = self._round_px(signal.entry_price)

        # Step 1: Clamp qty to fit max_notional_per_trade. This replaces the
        # earlier "raise on over-cap" behavior, which silently dropped every
        # signal whose risk-based qty produced a notional above the cap (every
        # signal at small accounts with normal stops). We clamp instead and
        # journal the clamp event so operators can see what's happening.
        cap = self.live_cfg.max_notional_per_trade
        desired_notional = rounded_px * qty
        clamped = False
        if desired_notional > cap:
            qty = cap / rounded_px
            clamped = True

        # Step 2: Round qty DOWN to the coin's lot size. ETH (lot 0.0001),
        # SOL (lot 0.01) etc. require this — sub-lot orders are HL-rejected.
        qty = self._round_qty_down(qty)

        # Step 3: After clamp + round, the order may now be below HL's
        # platform-wide $10 minimum notional, OR rounded to zero (qty < lot).
        # Either way we cannot place: emit ENTRY_REJECTED so bot._maybe_open
        # can journal the reason and increment the per-reason block counter.
        notional = rounded_px * qty
        if qty <= 0 or notional < HL_MIN_ORDER_NOTIONAL:
            reason = (
                "below_hl_min_lot" if qty <= 0
                else "below_hl_min_notional"
            )
            return ExecutionUpdate(
                ts_ms=signal.created_ms,
                event_type=ExecEventType.ENTRY_REJECTED,
                message=(
                    f"hl entry rejected ({reason}): qty={qty:.8f} "
                    f"notional=${notional:.2f} cap=${cap:.2f} "
                    f"min_notional=${HL_MIN_ORDER_NOTIONAL:.2f} "
                    f"side={signal.side.value} px={rounded_px:.4f}"
                    + (" CLAMPED" if clamped else "")
                ),
                signal_id=signal_id,
            )

        # Step 4: After clamp/round, recompute risk_dollars to be proportional
        # to the actual qty. The caller passed the INTENDED risk budget, but
        # if we clamped, the realized risk is smaller. This keeps r-multiples
        # honest in the journal (small qty -> small risk -> realistic R).
        actual_risk = abs(rounded_px - signal.stop_price) * qty
        # Use the smaller of (intended, actual) so we never inflate R.
        effective_risk = min(risk_dollars, actual_risk) if actual_risk > 0 else risk_dollars

        exchange = self._ensure_exchange()
        is_buy = signal.side == Side.LONG
        cloid = _cloid_from_signal_id(signal_id)
        # Alo = "Add Liquidity Only" = post-only. Refused if it would cross the spread.
        try:
            result = exchange.order(
                name=self.coin,
                is_buy=is_buy,
                sz=qty,
                limit_px=rounded_px,
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
                f"side={signal.side.value} qty={qty} px={rounded_px}: {exc}"
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
            entry_price=rounded_px,  # use HL-tick-rounded price so fill detection matches
            stop_price=signal.stop_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            created_ms=signal.created_ms,
            expiry_sec=self.cfg.pending_entry_expiry_sec,
            level_label=signal.level_label,
            risk_dollars=effective_risk,
            coin=self.coin,
            external_oid=oid,
        )
        # Backup convenience mirror; source of truth is pending_entry.external_oid.
        self._pending_oid = oid

        clamp_msg = f" (CLAMPED from notional ${desired_notional:.2f})" if clamped else ""
        return ExecutionUpdate(
            ts_ms=signal.created_ms,
            event_type=ExecEventType.ENTRY_PLACED,
            message=(
                f"hl entry placed [{self.live_cfg.network}]: {signal.side.value} "
                f"qty={qty:.8f} @ {rounded_px:.4f} notional=${notional:.2f} "
                f"oid={oid}{clamp_msg}"
            ),
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
        # Persist position state to disk for the daily-restart workaround.
        # No-op if the serialized state hasn't changed since last write.
        self._persist_position()
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
                # Mirrors PaperOrderManager safeguard: protect against R-multiple
                # inflation when HL fills a different (larger) qty than what the
                # original sizing assumed. Using max() ensures r_multiple stays
                # honest even if pe.risk_dollars was computed on a smaller qty.
                risk_dollars=max(pe.risk_dollars, abs(entry_px - pe.stop_price) * abs(szi)),
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
        """If a position already exists on HL (e.g. after a daily restart),
        restore local state.

        Two-tier strategy:
          1. **Persisted state** (preferred): if runtime/active_positions/<COIN>.json
             exists, was written < PERSISTED_STATE_MAX_AGE_SEC ago, and matches
             the current HL position size, restore the FULL state — original
             stop, TPs, trail history, realized fees. This is the daily-restart
             workaround: 60-90s offline doesn't lose any management context.

          2. **Wide-stop fallback** (existing behavior): if no persisted state
             or it's stale/mismatched, fall back to a conservative wide stop
             (default 0.5% of notional) so unknown positions don't drift far.
             TP1/TP2 set to entry — tp1 gated by tp1_filled=True (never fires),
             tp2 fires on first tick at/above entry → closes at break-even.

        Best practice still: manually flatten any open positions before a
        planned restart if you don't trust the persisted state.
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

        # Find the matching position on HL (if any).
        hl_pos: dict | None = None
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            if str(pos.get("coin", "")).upper() == self.coin.upper():
                szi_raw = pos.get("szi")
                szi = float(szi_raw.get("base", 0)) if isinstance(szi_raw, dict) else float(szi_raw or 0)
                if szi != 0:
                    hl_pos = pos
                break

        # Tier 1: try restoring from persisted state (the daily-restart common case).
        if self._try_restore_from_persisted(hl_pos):
            self._cancel_stale_resting_orders(address)
            return

        # Tier 2: fallback wide-stop reconcile path.
        if hl_pos is None:
            self._cancel_stale_resting_orders(address)
            return

        szi_raw = hl_pos.get("szi")
        szi = float(szi_raw.get("base", 0)) if isinstance(szi_raw, dict) else float(szi_raw or 0)
        entry_px = float(hl_pos.get("entryPx", 0))
        if entry_px <= 0:
            return
        side = Side.LONG if szi > 0 else Side.SHORT
        # Tightened recovery stop: 0.5% (was 2%). For typical $30 notional, that's
        # $0.15 max loss vs the old $0.60 — closer to the bot's normal trade risk.
        # Still wide enough to avoid stop-out on routine market noise.
        recovery_stop_pct = 0.005
        wide_stop = entry_px * (1 - recovery_stop_pct) if side == Side.LONG else entry_px * (1 + recovery_stop_pct)
        self.position = OpenPosition(
            signal_id="reconciled",
            side=side,
            entry_price=entry_px,
            stop_price=wide_stop,
            tp1_price=entry_px,
            tp2_price=entry_px,
            opened_ms=int(time.time() * 1000),
            qty_initial=abs(szi),
            qty_remaining=abs(szi),
            risk_dollars=abs(szi * entry_px) * recovery_stop_pct,
            coin=self.coin,
            tp1_filled=True,
            best_price=entry_px,
            worst_price=entry_px,
        )
        log.warning(
            "Reconciled %s position via WIDE-STOP fallback [%s]: %s qty=%.6f entry=%.4f stop=%.4f "
            "(persisted state missing/stale — original stop/TP lost)",
            self.coin, self.live_cfg.network, side.value, abs(szi), entry_px, wide_stop,
        )
        notional = abs(szi) * entry_px
        if notional > self.live_cfg.max_notional_per_trade:
            log.warning(
                "Reconciled %s position notional $%.2f EXCEEDS max_notional_per_trade=$%.2f — "
                "consider manual flatten via scripts/flatten_live.py",
                self.coin, notional, self.live_cfg.max_notional_per_trade,
            )

        self._cancel_stale_resting_orders(address)

    def _cancel_stale_resting_orders(self, address: str) -> None:
        """Cancel any resting orders for this coin at startup.

        Orders surviving a restart are most likely pre-restart pending entries
        that never filled — the local state for them is gone, so it's safer to
        cancel than to let them potentially fill into a position the bot has
        no management context for.
        """
        try:
            info = self._ensure_info()
            open_orders = info.open_orders(address)
            coin_orders = [
                {"coin": o.get("coin"), "oid": o.get("oid")}
                for o in open_orders
                if str(o.get("coin", "")).upper() == self.coin.upper() and o.get("oid")
            ]
            if coin_orders:
                exchange = self._ensure_exchange()
                exchange.bulk_cancel(coin_orders)
                log.warning(
                    "Reconcile cancelled %d stale resting orders for %s",
                    len(coin_orders), self.coin,
                )
        except Exception as exc:
            log.warning("HL open_orders / bulk_cancel failed during reconcile: %s", exc, exc_info=True)

    def _agent_address(self) -> str:
        if self._agent_addr is None:
            import eth_account
            self._agent_addr = eth_account.Account.from_key(self.live_cfg.agent_private_key).address
        return self._agent_addr

    def refresh_deadman(self, now_ms: int) -> None:
        """Push the schedule_cancel timer forward. Must be called periodically;
        if not called within deadman_cancel_sec, HL auto-cancels all orders."""
        if not self.live_cfg.allow_live:
            return
        cancel_at = now_ms + self.live_cfg.deadman_cancel_sec * 1000
        try:
            self._ensure_exchange().schedule_cancel(cancel_at)
            self._last_deadman_refresh_ms = now_ms
        except Exception as exc:
            log.warning("Deadman refresh failed: %s", exc, exc_info=True)

    def should_refresh_deadman(self, now_ms: int) -> bool:
        """True if deadman_refresh_sec has elapsed since the last refresh (or never)."""
        if not self.live_cfg.allow_live:
            return False
        elapsed_ms = now_ms - self._last_deadman_refresh_ms
        return elapsed_ms >= self.live_cfg.deadman_refresh_sec * 1000

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

    def _extract_fill(self, result, default_px: float) -> tuple[float, float] | None:
        """Parse an exchange.order/market_close response and return (filled_qty, avg_px)
        ONLY if HL confirmed the fill. Returns None when no fill is in the response —
        callers MUST treat None as "do not clear local position state" because HL
        likely still has the position.

        This is the regression guard for the phantom-close-in-reverse bug observed
        2026-05-04: market_close returned status=ok but with empty/error statuses,
        and the bot used to fall through to clearing position state while HL kept
        the position open as an orphan.
        """
        if not isinstance(result, dict) or result.get("status") != "ok":
            return None
        statuses = result.get("response", {}).get("data", {}).get("statuses", [])
        if not statuses:
            return None
        st = statuses[0]
        if not isinstance(st, dict) or "filled" not in st:
            return None
        filled = st["filled"]
        try:
            qty = float(filled.get("totalSz", 0))
            px = float(filled.get("avgPx", default_px))
        except (TypeError, ValueError):
            return None
        if qty <= 0:
            return None
        return qty, px

    def _persist_position(self) -> None:
        """Write current position state to disk so a daily-restart can resume
        management exactly where it left off (correct stop, TP, trail history).

        Called after every state-mutating tick. Writes are cheap (~500 bytes,
        SSD-cached) and we skip writes when the serialized state is unchanged
        from the last write.
        """
        if self.position is None:
            self._clear_persisted_state()
            return
        p = self.position
        data = {
            "ts_ms": int(time.time() * 1000),
            "coin": self.coin,
            "signal_id": p.signal_id,
            "side": p.side.value,
            "entry_price": p.entry_price,
            "stop_price": p.stop_price,
            "tp1_price": p.tp1_price,
            "tp2_price": p.tp2_price,
            "qty_initial": p.qty_initial,
            "qty_remaining": p.qty_remaining,
            "tp1_filled": p.tp1_filled,
            "best_price": p.best_price,
            "worst_price": p.worst_price,
            "realized_pnl": p.realized_pnl,
            "realized_fees": p.realized_fees,
            "risk_dollars": p.risk_dollars,
            "opened_ms": p.opened_ms,
        }
        # Skip the file write if nothing meaningful changed (excluding ts_ms).
        sig = json.dumps({k: v for k, v in data.items() if k != "ts_ms"}, sort_keys=True)
        if sig == self._last_persisted_hash:
            return
        try:
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, self._state_path)  # atomic rename so partial writes can't corrupt
            self._last_persisted_hash = sig
        except Exception as exc:
            log.warning("Failed to persist position state to %s: %s", self._state_path, exc)

    def _clear_persisted_state(self) -> None:
        """Delete the persisted state file. Called on every clean position close."""
        self._last_persisted_hash = None
        try:
            if self._state_path.exists():
                self._state_path.unlink()
        except Exception as exc:
            log.warning("Failed to clear persisted state %s: %s", self._state_path, exc)

    def _try_restore_from_persisted(self, hl_position: dict | None) -> bool:
        """Try to restore self.position from the persisted state file.

        Returns True if restoration succeeded. The caller (reconcile_on_startup)
        falls through to the wide-stop reconcile path on False.

        Safety checks:
          - File must exist and be < PERSISTED_STATE_MAX_AGE_SEC old
          - HL must have a matching position (sign + size within 1%)
          - JSON must parse and contain all required fields
        """
        if not self._state_path.exists():
            return False
        try:
            data = json.loads(self._state_path.read_text())
        except Exception as exc:
            log.warning("Persisted state %s unreadable: %s — falling through to wide-stop reconcile",
                        self._state_path, exc)
            return False

        age_sec = (time.time() * 1000 - data.get("ts_ms", 0)) / 1000.0
        if age_sec > PERSISTED_STATE_MAX_AGE_SEC:
            log.warning("Persisted state for %s is stale (age=%.1fs > %ds) — falling through",
                        self.coin, age_sec, PERSISTED_STATE_MAX_AGE_SEC)
            return False

        if hl_position is None:
            log.warning("Persisted state for %s exists but HL has no matching position — clearing stale file",
                        self.coin)
            self._clear_persisted_state()
            return False

        szi_raw = hl_position.get("szi")
        hl_qty_signed = float(szi_raw.get("base", 0)) if isinstance(szi_raw, dict) else float(szi_raw or 0)
        hl_qty = abs(hl_qty_signed)
        persisted_qty = float(data.get("qty_remaining", 0))
        if persisted_qty <= 0 or abs(hl_qty - persisted_qty) > max(persisted_qty * 0.01, 1e-6):
            log.warning(
                "Persisted state qty=%s doesn't match HL qty=%s for %s — falling through to wide-stop reconcile",
                persisted_qty, hl_qty, self.coin,
            )
            return False

        # Sign check: persisted "long" must match HL positive szi (and vice versa)
        persisted_long = data.get("side") == Side.LONG.value
        hl_long = hl_qty_signed > 0
        if persisted_long != hl_long:
            log.warning("Persisted side doesn't match HL sign for %s — falling through", self.coin)
            return False

        # Restore the position with all original management state
        try:
            self.position = OpenPosition(
                signal_id=str(data["signal_id"]),
                side=Side.LONG if persisted_long else Side.SHORT,
                entry_price=float(data["entry_price"]),
                stop_price=float(data["stop_price"]),
                tp1_price=float(data["tp1_price"]),
                tp2_price=float(data["tp2_price"]),
                opened_ms=int(data["opened_ms"]),
                qty_initial=float(data["qty_initial"]),
                qty_remaining=float(data["qty_remaining"]),
                risk_dollars=float(data["risk_dollars"]),
                coin=self.coin,
                tp1_filled=bool(data.get("tp1_filled", False)),
                realized_pnl=float(data.get("realized_pnl", 0)),
                realized_fees=float(data.get("realized_fees", 0)),
                best_price=float(data.get("best_price", data["entry_price"])),
                worst_price=float(data.get("worst_price", data["entry_price"])),
            )
        except Exception as exc:
            log.warning("Failed to construct OpenPosition from persisted state: %s — falling through", exc)
            return False

        log.info(
            "Restored persisted position state for %s [%s]: %s qty=%.6f entry=%.4f stop=%.4f "
            "tp1=%.4f tp2=%.4f tp1_filled=%s age=%.1fs",
            self.coin, self.live_cfg.network, self.position.side.value,
            self.position.qty_remaining, self.position.entry_price, self.position.stop_price,
            self.position.tp1_price, self.position.tp2_price, self.position.tp1_filled, age_sec,
        )
        return True

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

        # Verify HL actually filled before mutating local state. If no fill in the
        # response, log critically and DO NOT modify position — next tick will retry
        # the trail/exit logic. Phantom-close-in-reverse regression guard.
        extracted = self._extract_fill(result, default_px=trade_price)
        if extracted is None:
            log.critical(
                "HL TP1 market_close for %s returned status=ok but NO filled status in response. "
                "HL likely still has the position. Local state retained; will retry on next tick. "
                "Response: %s",
                self.coin, result,
            )
            return []
        actual_qty, fill_px = extracted

        # Use the actual filled qty (not the requested) so partial-of-partial fills
        # don't desync local from HL.
        if p.side == Side.LONG:
            partial_pnl = (fill_px - p.entry_price) * actual_qty
        else:
            partial_pnl = (p.entry_price - fill_px) * actual_qty
        partial_fee = (fill_px * actual_qty) * self.cfg.taker_fee_pct
        p.realized_pnl += partial_pnl
        p.realized_fees += partial_fee
        p.qty_remaining -= actual_qty
        # Only mark tp1_filled and move stop to BE if we actually closed ~half.
        # If HL filled less than requested (rare on IOC), defer the BE-stop move
        # until a future tick fills the rest.
        if actual_qty >= partial_qty * 0.99:
            p.tp1_filled = True
            p.stop_price = p.entry_price  # reduce tail risk after first scale
        return [ExecutionUpdate(
            ts_ms=ts_ms,
            event_type=ExecEventType.PARTIAL_TP,
            message=f"hl tp1 partial [{self.live_cfg.network}]: qty={actual_qty:.6f} @ {fill_px:.4f} (stop->BE={p.tp1_filled})",
            signal_id=p.signal_id,
        )]

    def _close_via_market(self, ts_ms: int, reason: str, trade_price: float) -> list[ExecutionUpdate]:
        p = self.position
        if p is None:
            return []
        exchange = self._ensure_exchange()
        requested_qty = p.qty_remaining
        try:
            result = exchange.market_close(
                coin=self.coin,
                sz=requested_qty,
                slippage=0.005,  # 0.5% slippage tolerance
            )
        except Exception as exc:
            log.error(
                "HL market_close failed for %s qty=%s: %s",
                self.coin, requested_qty, exc, exc_info=True,
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

        # Verify HL actually filled before mutating local state. The bug observed
        # 2026-05-04: market_close returned status=ok but with no "filled" status,
        # and the bot used to fall through with fill_px=trade_price and clear
        # local state. HL kept the position open as an unmanaged orphan.
        extracted = self._extract_fill(result, default_px=trade_price)
        if extracted is None:
            log.critical(
                "HL market_close for %s qty=%s returned status=ok but NO filled status. "
                "HL likely still has the position. Local state retained; will retry on next tick. "
                "Response: %s",
                self.coin, requested_qty, result,
            )
            return []
        actual_qty, fill_px = extracted

        # Handle partial fills: if HL filled less than requested, only realize PnL
        # on the filled portion and keep the residual position open. Stop/TP logic
        # will continue to fire on subsequent ticks for the remainder.
        if actual_qty < requested_qty * 0.99:
            log.warning(
                "HL market_close for %s only filled %s of requested %s — keeping residual open",
                self.coin, actual_qty, requested_qty,
            )
            if p.side == Side.LONG:
                partial_pnl = (fill_px - p.entry_price) * actual_qty
            else:
                partial_pnl = (p.entry_price - fill_px) * actual_qty
            partial_fee = (fill_px * actual_qty) * self.cfg.taker_fee_pct
            p.realized_pnl += partial_pnl
            p.realized_fees += partial_fee
            p.qty_remaining -= actual_qty
            return [ExecutionUpdate(
                ts_ms=ts_ms,
                event_type=ExecEventType.PARTIAL_TP,
                message=(
                    f"hl partial close [{self.live_cfg.network}] ({reason}): "
                    f"qty={actual_qty:.6f}/{requested_qty:.6f} @ {fill_px:.4f} "
                    f"residual={p.qty_remaining:.6f}"
                ),
                signal_id=p.signal_id,
            )]

        # Full close — proceed with normal close accounting using actual fill data.
        pnl_gross = (
            (fill_px - p.entry_price) * actual_qty
            if p.side == Side.LONG
            else (p.entry_price - fill_px) * actual_qty
        )
        pnl_gross += p.realized_pnl
        # Fees: taker on the market_close
        final_fee = (fill_px * actual_qty) * self.cfg.taker_fee_pct
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
