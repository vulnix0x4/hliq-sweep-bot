from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from hliq_bot.analytics.market_capture import MarketCaptureWriter
from hliq_bot.analytics.journal import SignalJournal
from hliq_bot.config import AppConfig
from hliq_bot.data.bar_builder import BarBuilder
from hliq_bot.data.hyperliquid_ws import HyperliquidWsClient
from hliq_bot.execution.order_manager import PaperOrderManager
from hliq_bot.ml.gate import MLGate
from hliq_bot.models import ClosedTrade, ExecEventType, MarketEvent, MarketState, RiskCheck, Side, SweepSignal
from hliq_bot.risk.governor import RiskGovernor
from hliq_bot.signal.regime import Regime, RegimeState, classify_regime
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.sweep_detector import SweepDetector
from hliq_bot.signal.vwap_tracker import VWAPTracker

log = logging.getLogger(__name__)


def _make_executor(config: AppConfig, coin: str):
    """Pick paper or live executor based on cfg.mode.

    The live import is lazy so paper mode never loads the HL SDK.
    """
    if config.mode == "live":
        if not config.live.allow_live:
            raise RuntimeError(
                "BOT_MODE=live requires BOT_ALLOW_LIVE=true (safety guard)"
            )
        from hliq_bot.execution.hyperliquid_order_manager import HyperliquidOrderManager
        return HyperliquidOrderManager(config.strategy, config.live, coin=coin)
    return PaperOrderManager(config.strategy)


@dataclass
class CoinWorker:
    """Per-coin state: bar builder, detector, executor, trackers, and microstructure data."""
    coin: str
    bar_builder: BarBuilder
    detector: SweepDetector
    executor: Any
    session_tracker: SessionTracker
    vwap_tracker: VWAPTracker
    last_spread_bps: float = 0.0
    last_best_bid: float = 0.0
    last_best_ask: float = 0.0
    last_bid_size: float = 0.0
    last_ask_size: float = 0.0
    recent_bar_ranges: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_closes: deque = field(default_factory=lambda: deque(maxlen=25))
    recent_trade_prices: deque = field(default_factory=deque)
    recent_signed_flow: deque = field(default_factory=deque)
    bars_closed: int = 0
    signals_seen: int = 0
    entries_placed: int = 0
    entries_filled: int = 0
    positions_closed: int = 0


class SweepBot:
    def __init__(self, config: AppConfig) -> None:
        if config.risk.portfolio_max_positions < 1:
            raise ValueError(
                f"risk.portfolio_max_positions must be >= 1 "
                f"(got {config.risk.portfolio_max_positions}); 0 would block all trading"
            )
        self.cfg = config
        self._time_override_ms: int | None = None
        self._suspend_auto_train = False
        self._project_root = self._detect_project_root()
        self._normalize_runtime_paths()
        self._load_runtime_ml_state()
        self.feed = HyperliquidWsClient(config.feed)
        self.risk = RiskGovernor(config.risk, config.strategy)
        self.ml_gate = MLGate(config.runtime)
        self.journal = SignalJournal(config.runtime.journal_path)
        self.capture = (
            MarketCaptureWriter(config.runtime.market_capture_path)
            if config.runtime.market_capture_enabled
            else None
        )
        self._signal_context: dict[str, dict[str, float | str]] = {}

        # Build per-coin workers
        range_maxlen = max(3, config.strategy.circuit_range_bars + 3)
        closes_maxlen = max(5, config.strategy.trend_lookback_bars + 5)
        self._workers: dict[str, CoinWorker] = {}
        for coin in config.feed.coins:
            st = SessionTracker()
            vt = VWAPTracker()
            worker = CoinWorker(
                coin=coin,
                bar_builder=BarBuilder(config.strategy.timeframe_sec),
                detector=SweepDetector(
                    config.strategy,
                    level_config=config.levels,
                    session_tracker=st,
                    vwap_tracker=vt,
                    coin=coin,
                ),
                executor=_make_executor(config, coin=coin),
                session_tracker=st,
                vwap_tracker=vt,
                recent_bar_ranges=deque(maxlen=range_maxlen),
                recent_closes=deque(maxlen=closes_maxlen),
            )
            self._workers[coin] = worker

        # Backward-compat: expose first coin's components as top-level attributes
        first_coin = config.feed.coin
        _first = self._workers[first_coin]
        self.bar_builder = _first.bar_builder
        self.detector = _first.detector
        self.executor = _first.executor
        self._session_tracker = _first.session_tracker
        self._vwap_tracker = _first.vwap_tracker

        # Reconcile any existing live positions on startup (idempotent for paper).
        for w in self._workers.values():
            if hasattr(w.executor, "reconcile_on_startup"):
                try:
                    w.executor.reconcile_on_startup()
                except Exception as exc:
                    log.warning("reconcile_on_startup failed for %s: %s", w.coin, exc, exc_info=True)

        self._resolved_trades = self._restore_risk_from_journal()

        self._last_event_ms = 0
        self._heartbeat_interval_ms = 60_000
        now_ms = self._now_ms()
        self._last_heartbeat_ms = now_ms
        self._trade_events = 0
        self._book_events = 0
        self._bars_closed = 0
        self._signals_seen = 0
        self._signals_blocked = 0
        # Per-reason breakdown of _signals_blocked for ops visibility (heartbeat + summary).
        self._block_reasons: dict[str, int] = {}
        self._entries_placed = 0
        self._entries_filled = 0
        self._positions_closed = 0
        self._signal_seq = 0
        self._event_queue: queue.Queue[MarketEvent] = queue.Queue(maxsize=50_000)
        self._event_drop_count = 0
        self._queue_full_last_log_ms = 0
        now_ms = self._now_ms()
        self._last_auto_train_ms = now_ms
        self._last_auto_train_resolved = self._resolved_trades

    @property
    def _first_worker(self) -> CoinWorker:
        return self._workers[self.cfg.feed.coin]

    # Backward-compat properties: proxy per-coin state to the first worker
    # so existing tests that set bot._last_bid_size etc. continue to work.
    @property
    def _last_spread_bps(self) -> float:
        return self._first_worker.last_spread_bps

    @_last_spread_bps.setter
    def _last_spread_bps(self, v: float) -> None:
        self._first_worker.last_spread_bps = v

    @property
    def _last_best_bid(self) -> float:
        return self._first_worker.last_best_bid

    @_last_best_bid.setter
    def _last_best_bid(self, v: float) -> None:
        self._first_worker.last_best_bid = v

    @property
    def _last_best_ask(self) -> float:
        return self._first_worker.last_best_ask

    @_last_best_ask.setter
    def _last_best_ask(self, v: float) -> None:
        self._first_worker.last_best_ask = v

    @property
    def _last_bid_size(self) -> float:
        return self._first_worker.last_bid_size

    @_last_bid_size.setter
    def _last_bid_size(self, v: float) -> None:
        self._first_worker.last_bid_size = v

    @property
    def _last_ask_size(self) -> float:
        return self._first_worker.last_ask_size

    @_last_ask_size.setter
    def _last_ask_size(self, v: float) -> None:
        self._first_worker.last_ask_size = v

    @property
    def _recent_bar_ranges(self) -> deque:
        return self._first_worker.recent_bar_ranges

    @_recent_bar_ranges.setter
    def _recent_bar_ranges(self, v: deque) -> None:
        self._first_worker.recent_bar_ranges = v

    @property
    def _recent_closes(self) -> deque:
        return self._first_worker.recent_closes

    @_recent_closes.setter
    def _recent_closes(self, v: deque) -> None:
        self._first_worker.recent_closes = v

    @property
    def _recent_trade_prices(self) -> deque:
        return self._first_worker.recent_trade_prices

    @_recent_trade_prices.setter
    def _recent_trade_prices(self, v: deque) -> None:
        self._first_worker.recent_trade_prices = v

    @property
    def _recent_signed_flow(self) -> deque:
        return self._first_worker.recent_signed_flow

    @_recent_signed_flow.setter
    def _recent_signed_flow(self, v: deque) -> None:
        self._first_worker.recent_signed_flow = v

    def _now_ms(self) -> int:
        if self._time_override_ms is not None:
            return self._time_override_ms
        return int(time.time() * 1000)

    def _restore_risk_from_journal(self) -> int:
        path = Path(self.cfg.runtime.journal_path)
        if not path.exists():
            return 0

        candidate_meta: dict[str, dict[str, str]] = {}
        decision_meta: dict[str, dict[str, float]] = {}
        restored = 0
        ml_restored = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sid = str(row.get("signal_id", ""))
                    if not sid:
                        continue
                    et = str(row.get("event_type", ""))

                    if et == "candidate":
                        side = self._parse_side(row.get("side"))
                        meta: dict[str, str] = {}
                        if side is not None:
                            meta["side"] = side.value
                        session = str(row.get("session", "")).strip().lower()
                        if session:
                            meta["session"] = session
                        regime = str(row.get("regime", "")).strip().lower()
                        if regime:
                            meta["regime"] = regime
                        level = str(row.get("level_label", "")).strip().lower()
                        if level:
                            meta["level_label"] = level
                        if meta:
                            candidate_meta[sid] = meta
                        continue

                    if et == "decision":
                        if bool(row.get("allowed", False)):
                            meta: dict[str, float] = {}
                            ml_prob = self._parse_float(row.get("ml_prob"))
                            if ml_prob is not None:
                                meta["ml_prob"] = ml_prob
                            ml_thr = self._parse_float(row.get("ml_threshold"))
                            if ml_thr is not None:
                                meta["ml_threshold"] = ml_thr
                            if meta:
                                decision_meta[sid] = meta
                        continue

                    if et != "outcome":
                        continue

                    ts_ms = int(row.get("ts_ms", 0))
                    pnl = float(row.get("pnl", 0.0))
                    r_mult = float(row.get("r_multiple", 0.0))
                    side = self._parse_side(row.get("side"))
                    meta = candidate_meta.get(sid, {})
                    if side is None:
                        side = self._parse_side(meta.get("side"))
                    if side is None:
                        continue

                    implied_risk = abs(pnl) / max(abs(r_mult), 1e-6) if abs(r_mult) > 1e-6 else 1.0
                    trade = ClosedTrade(
                        signal_id=sid,
                        side=side,
                        entry_price=0.0,
                        exit_price=0.0,
                        qty=0.0,
                        pnl=pnl,
                        risk_dollars=max(implied_risk, 1e-6),
                        r_multiple=r_mult,
                        opened_ms=max(0, ts_ms - 1),
                        closed_ms=ts_ms,
                        exit_reason=str(row.get("exit_reason", "restored")),
                    )
                    self.risk.register_closed_trade(
                        trade,
                        session=meta.get("session", ""),
                        level_label=meta.get("level_label", ""),
                    )
                    ml_prob = self._parse_float(row.get("ml_prob"))
                    if ml_prob is None:
                        ml_prob = decision_meta.get(sid, {}).get("ml_prob")
                    if ml_prob is not None:
                        self.ml_gate.register_outcome(
                            probability=ml_prob,
                            r_multiple=r_mult,
                            regime=meta.get("regime", ""),
                            session=meta.get("session", ""),
                        )
                        ml_restored += 1
                    restored += 1
        except Exception as exc:
            log.warning("Risk warm-start skipped: %s", exc)
            return restored

        if restored > 0:
            log.info(
                "Risk warm-start restored %d trades from %s (equity=%.2f, ml_outcomes=%d)",
                restored,
                path,
                self.risk.equity,
                ml_restored,
            )
        return restored

    def _parse_side(self, raw: object) -> Side | None:
        s = str(raw or "").strip().lower()
        if s == Side.LONG.value:
            return Side.LONG
        if s == Side.SHORT.value:
            return Side.SHORT
        return None

    def _parse_float(self, raw: object) -> float | None:
        try:
            out = float(raw)
        except (TypeError, ValueError):
            return None
        if out != out:  # NaN
            return None
        if out == float("inf") or out == float("-inf"):
            return None
        return out

    def _resolve_project_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return self._project_root / p

    def _detect_project_root(self) -> Path:
        cwd = Path.cwd().resolve()
        if (cwd / "scripts" / "train_gate.py").exists():
            return cwd

        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "scripts" / "train_gate.py").exists():
                return parent
        return cwd

    def _normalize_runtime_paths(self) -> None:
        rt = self.cfg.runtime
        rt.runtime_dir = str(self._resolve_project_path(rt.runtime_dir))
        rt.journal_path = str(self._resolve_project_path(rt.journal_path))
        rt.market_capture_path = str(self._resolve_project_path(rt.market_capture_path))
        rt.ml_model_path = str(self._resolve_project_path(rt.ml_model_path))
        rt.ml_state_path = str(self._resolve_project_path(rt.ml_state_path))
        self.cfg.replay.input_path = str(self._resolve_project_path(self.cfg.replay.input_path))

    def _load_runtime_ml_state(self) -> None:
        path = self._resolve_project_path(self.cfg.runtime.ml_state_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        min_prob = self._parse_float(raw.get("ml_min_prob"))
        if min_prob is None:
            return
        self.cfg.runtime.ml_min_prob = max(0.0, min(1.0, min_prob))

    def _persist_runtime_ml_state(self) -> None:
        path = self._resolve_project_path(self.cfg.runtime.ml_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ml_min_prob": round(max(0.0, min(1.0, self.cfg.runtime.ml_min_prob)), 6),
            "provider": self.cfg.runtime.ml_provider,
            "updated_ms": self._now_ms(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            log.warning("Failed to persist runtime ML state: %s", exc)

    async def run(self) -> None:
        log.info(
            (
                "Starting sweep bot mode=%s coins=%s timeframe=%ss ml_enabled=%s ml_mode=%s ml_provider=%s "
                "ml_auto_train=%s interval=%ss min_resolved=%d min_new=%d warmup=%s target=%d journal=%s capture=%s"
            ),
            self.cfg.mode,
            self.cfg.feed.coins,
            self.cfg.strategy.timeframe_sec,
            self.cfg.runtime.ml_enabled,
            self.cfg.runtime.ml_decision_mode,
            self.cfg.runtime.ml_provider,
            self.cfg.runtime.ml_auto_train,
            self.cfg.runtime.ml_auto_train_interval_sec,
            self.cfg.runtime.ml_auto_train_min_resolved,
            self.cfg.runtime.ml_auto_train_min_new_trades,
            self.cfg.strategy.warmup_enabled,
            self.cfg.strategy.warmup_target_resolved,
            self.cfg.runtime.journal_path,
            self.cfg.runtime.market_capture_path if self.capture is not None else "disabled",
        )
        stop_flag = threading.Event()
        worker = threading.Thread(target=self._event_worker, args=(stop_flag,), name="sweepbot-worker", daemon=True)
        worker.start()
        try:
            async for event in self.feed.stream():
                if stop_flag.is_set():
                    break
                if self.capture is not None:
                    self.capture.write(event)
                self._enqueue_event(event)
        finally:
            stop_flag.set()
            worker.join(timeout=5.0)

    def run_replay(self, events) -> dict[str, float | int]:
        log.info(
            "Starting replay input=%s journal=%s ml_mode=%s",
            self.cfg.replay.input_path,
            self.cfg.runtime.journal_path,
            self.cfg.runtime.ml_decision_mode,
        )
        self._suspend_auto_train = True
        try:
            for event in events:
                self._time_override_ms = event.ts_ms
                self.feed.last_message_ms = event.ts_ms
                self._handle_event(event)
            return self.runtime_summary()
        finally:
            self._time_override_ms = None
            self._suspend_auto_train = False

    def _enqueue_event(self, event: MarketEvent) -> None:
        try:
            self._event_queue.put_nowait(event)
            return
        except queue.Full:
            pass

        # Drop oldest event when saturated to keep the consumer near real-time.
        dropped = False
        try:
            self._event_queue.get_nowait()
            self._event_queue.task_done()
            dropped = True
        except queue.Empty:
            pass

        try:
            self._event_queue.put_nowait(event)
            dropped = True
        except queue.Full:
            pass

        if dropped:
            self._event_drop_count += 1
            now_ms = self._now_ms()
            if now_ms - self._queue_full_last_log_ms >= 10_000:
                self._queue_full_last_log_ms = now_ms
                log.warning(
                    "Event queue saturated; dropped=%d qsize=%d",
                    self._event_drop_count,
                    self._event_queue.qsize(),
                )

    def _event_worker(self, stop_flag: threading.Event) -> None:
        while not stop_flag.is_set():
            try:
                event = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                self._maybe_heartbeat()
                continue

            try:
                self._handle_event(event)
            except Exception:
                log.exception("Unhandled exception while processing market event")
            finally:
                self._event_queue.task_done()
            self._maybe_heartbeat()

    def _resolve_worker(self, event: MarketEvent) -> CoinWorker:
        """Resolve the CoinWorker for an event. Falls back to first worker for untagged events."""
        coin = event.coin
        if coin and coin in self._workers:
            return self._workers[coin]
        # Backward compat: untagged events go to first worker (single-coin mode)
        return self._first_worker

    def _count_block(self, reason: str) -> None:
        """Increment the global blocked counter and the per-reason breakdown.

        `reason` should be a stable bucket name (no embedded floats/IDs). Variable
        details (thresholds, ML probabilities) belong in the journal entry, not
        the bucket key, otherwise the heartbeat breakdown becomes a long tail of
        unique reasons.
        """
        self._signals_blocked += 1
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1

    def _total_exposure_count(self) -> int:
        """Count total exposure (open positions + pending entries) across all coin workers.

        Pending limit orders count as exposure because they can fill into a position
        at any moment. Counting only open positions allows N+1 simultaneous pending
        entries to bypass portfolio_max_positions. Names changed from
        _total_open_positions for clarity.
        """
        count = 0
        for w in self._workers.values():
            if w.executor.position is not None or w.executor.pending_entry is not None:
                count += 1
        return count

    def _handle_event(self, event: MarketEvent) -> None:
        self._last_event_ms = event.ts_ms
        # Deadman switch: refresh per-tick (gated by should_refresh_deadman) so the
        # actual SDK push fires every deadman_refresh_sec with safety margin against
        # heartbeat-cadence jitter. Heartbeat (60s) would only give ~30s effective
        # cadence which equals deadman_cancel_sec — zero margin.
        self._maybe_refresh_deadmans(self._now_ms())
        w = self._resolve_worker(event)

        if event.kind == "book" and event.book is not None:
            self._book_events += 1
            w.last_spread_bps = event.book.spread_bps
            w.last_best_bid = event.book.best_bid
            w.last_best_ask = event.book.best_ask
            w.last_bid_size = event.book.bid_size
            w.last_ask_size = event.book.ask_size
            return

        if event.kind != "trade" or event.trade is None:
            return

        self._trade_events += 1
        trade = event.trade
        w.recent_trade_prices.append((trade.ts_ms, trade.price))
        self._trim_recent_trade_prices_w(w, trade.ts_ms)
        sign = 1.0 if trade.side == "buy" else -1.0 if trade.side == "sell" else 0.0
        if sign != 0.0:
            w.recent_signed_flow.append((trade.ts_ms, sign * trade.size))
        self._trim_recent_signed_flow_w(w, trade.ts_ms)

        updates = w.executor.on_trade(trade)
        for update in updates:
            if update.signal_id:
                self.journal.write(
                    "lifecycle",
                    update.signal_id,
                    {
                        "ts_ms": update.ts_ms,
                        "event": update.event_type.value,
                        "message": update.message,
                        "coin": w.coin,
                    },
                )
            if update.event_type == ExecEventType.ENTRY_FILLED:
                self._entries_filled += 1
                w.entries_filled += 1
            elif update.event_type == ExecEventType.POSITION_CLOSED:
                self._positions_closed += 1
                w.positions_closed += 1
            log.info(update.message)
            if update.closed_trade is not None:
                ctx = self._signal_context.pop(update.closed_trade.signal_id, {})
                ml_prob = self._parse_float(ctx.get("ml_prob"))
                ml_threshold = self._parse_float(ctx.get("ml_threshold"))
                regime = str(ctx.get("regime", "")).strip().lower()
                session = str(ctx.get("session", "")).strip().lower()
                level_label = str(ctx.get("level_label", "")).strip().lower()
                self.journal.write(
                    "outcome",
                    update.closed_trade.signal_id,
                    {
                        "ts_ms": update.closed_trade.closed_ms,
                        "side": update.closed_trade.side.value,
                        "coin": w.coin,
                        "session": session,
                        "regime": regime,
                        "level_label": level_label,
                        "pnl": update.closed_trade.pnl,
                        "pnl_gross": update.closed_trade.pnl_gross,
                        "fees_paid": update.closed_trade.fees_paid,
                        "r_multiple": update.closed_trade.r_multiple,
                        "mfe_pnl": update.closed_trade.mfe_pnl,
                        "mae_pnl": update.closed_trade.mae_pnl,
                        "ml_prob": ml_prob,
                        "ml_threshold": ml_threshold,
                        "exit_reason": update.closed_trade.exit_reason,
                    },
                )
                self.risk.register_closed_trade(
                    update.closed_trade,
                    session=session,
                    level_label=level_label,
                )
                self._resolved_trades += 1
                if ml_prob is not None:
                    self.ml_gate.register_outcome(
                        probability=ml_prob,
                        r_multiple=update.closed_trade.r_multiple,
                        regime=regime,
                        session=session,
                    )
                log.info(
                    "Daily stats: pnl=%.2f r=%.2f equity=%.2f",
                    self.risk.daily_pnl,
                    self.risk.daily_r,
                    self.risk.equity,
                )
                self._maybe_auto_train()

        closed_bars = w.bar_builder.on_trade(trade, w.last_spread_bps)
        for bar in closed_bars:
            self._bars_closed += 1
            w.bars_closed += 1
            w.recent_bar_ranges.append(bar.range_pct)
            w.recent_closes.append(bar.close)
            w.session_tracker.on_bar(bar)
            w.vwap_tracker.on_bar(bar)
            signal = w.detector.on_bar(bar)
            if signal is None:
                continue
            self._signals_seen += 1
            w.signals_seen += 1
            signal_id = self._next_signal_id(bar.end_ms)
            state = self._build_market_state_w(w, bar.end_ms)
            regime = self._regime_state_w(w, bar.end_ms)
            ml_features = self._signal_features_w(w, signal, state, regime, bar)
            self.journal.write(
                "candidate",
                signal_id,
                {
                    "ts_ms": bar.end_ms,
                    "coin": w.coin,
                    "side": signal.side.value,
                    "level_label": signal.level_label,
                    "entry": signal.entry_price,
                    "stop": signal.stop_price,
                    "tp1": signal.tp1_price,
                    "tp2": signal.tp2_price,
                    "confidence": signal.confidence,
                    "signal_score": signal.signal_score,
                    "overshoot_bps": signal.overshoot_bps,
                    "reclaim_bps": signal.reclaim_bps,
                    "volume_ratio": signal.volume_ratio,
                    "wick_ratio": signal.wick_ratio,
                    "regime": regime.regime.value,
                    "session": regime.session,
                    "features": ml_features,
                },
            )
            self._signal_context[signal_id] = {
                "side": signal.side.value,
                "session": regime.session,
                "regime": regime.regime.value,
                "level_label": signal.level_label,
                "signal_score": signal.signal_score,
                "coin": w.coin,
            }
            if w.executor.has_exposure():
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": "existing_exposure",
                        "coin": w.coin,
                    },
                )
                continue

            # Portfolio-level position limit
            if self._total_exposure_count() >= self.cfg.risk.portfolio_max_positions:
                self._count_block("portfolio_position_limit")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": "portfolio_position_limit",
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: portfolio_position_limit coin=%s", w.coin)
                continue

            if self._in_funding_blackout(bar.end_ms):
                self._count_block("funding_blackout")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": "funding_blackout",
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: funding_blackout")
                continue

            micro_check = self._microstructure_check_w(w, signal.side)
            if not micro_check.allowed:
                self._count_block("microstructure")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": micro_check.reason,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s", micro_check.reason)
                continue

            session_check = self.risk.can_trade_session(regime.session)
            if not session_check.allowed:
                self._count_block("session_lockout")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": session_check.reason,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s", session_check.reason)
                continue

            level_check = self.risk.can_trade_level(signal.level_label, ts_ms=bar.end_ms)
            if not level_check.allowed:
                self._count_block("level_cooldown")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": level_check.reason,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s", level_check.reason)
                continue

            side_check = self.risk.can_trade_side(signal.side, ts_ms=bar.end_ms)
            if not side_check.allowed:
                self._count_block("side_cooldown")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": side_check.reason,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s", side_check.reason)
                continue

            if self._regime_blocks_signal(regime):
                self._count_block("regime_block")
                reason = f"regime_block:{regime.regime.value}"
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": reason,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s", reason)
                continue

            conf_floor = self.cfg.strategy.min_confidence_trend if regime.regime == Regime.TREND else self.cfg.strategy.min_confidence_range
            if signal.confidence < conf_floor:
                self._count_block("confidence_floor")
                reason = f"confidence<{conf_floor:.2f}"
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": reason,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s", reason)
                continue

            # Signal-quality floor: block low-score signals entirely (when configured).
            # The data-grounded signal_score (0.0-1.0) is computed in the detector
            # from wick_ratio (45%), reclaim_bps, overshoot_bps, volume_ratio, and
            # a confidence x regime mix. Setting min_signal_score > 0 trades
            # selectivity for fewer trades — useful when fee drag is significant
            # vs typical edge.
            score_floor = self.cfg.strategy.min_signal_score
            if score_floor > 0.0 and signal.signal_score < score_floor:
                self._count_block("signal_score_floor")
                reason = f"signal_score<{score_floor:.2f}"
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": reason,
                        "signal_score": signal.signal_score,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s (score=%.2f)", reason, signal.signal_score)
                continue

            # Regime filter: block longs in clear downtrend, shorts in clear uptrend.
            # Mean-reversion sweep+rejection structurally fails when price is in a
            # sustained directional move — see live data 2026-04-29 to 05-01 (5
            # consecutive long entries during ETH downtrend, all losses). When
            # enabled, computes (current - MA(N)) / MA × 100 and blocks signals
            # whose direction would be fighting the regime.
            if self.cfg.strategy.regime_filter_enabled:
                ma_bars = max(2, self.cfg.strategy.regime_filter_ma_bars)
                threshold = self.cfg.strategy.regime_filter_threshold_pct
                closes = list(w.recent_closes)
                if len(closes) >= ma_bars:
                    ma = sum(closes[-ma_bars:]) / ma_bars
                    pct_from_ma = ((bar.close - ma) / ma) * 100.0
                    blocked_by_regime = (
                        (signal.side == Side.LONG and pct_from_ma < -threshold) or
                        (signal.side == Side.SHORT and pct_from_ma > threshold)
                    )
                    if blocked_by_regime:
                        regime_dir = "downtrend" if pct_from_ma < 0 else "uptrend"
                        self._count_block("regime_filter")
                        reason = f"regime_filter:{regime_dir}_blocks_{signal.side.value}"
                        self.journal.write(
                            "decision",
                            signal_id,
                            {
                                "ts_ms": bar.end_ms,
                                "allowed": False,
                                "reason": reason,
                                "pct_from_ma": pct_from_ma,
                                "ma_bars": ma_bars,
                                "coin": w.coin,
                            },
                        )
                        log.info(
                            "Signal blocked: %s (price %.3f%% from %d-bar MA)",
                            reason, pct_from_ma, ma_bars,
                        )
                        continue

            ml_decision = self.ml_gate.evaluate(ml_features)
            ml_gate_mode = self._ml_decision_mode()
            if ml_gate_mode == "gate" and not ml_decision.allowed:
                self._count_block("ml_gate")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": ml_decision.reason,
                        "ml_prob": ml_decision.probability,
                        "ml_threshold": ml_decision.threshold,
                        "coin": w.coin,
                    },
                )
                log.info(
                    "Signal blocked: %s prob=%.3f threshold=%.3f",
                    ml_decision.reason,
                    ml_decision.probability,
                    ml_decision.threshold,
                )
                continue

            check = self.risk.can_open_new_trade(state)
            if not check.allowed:
                self._count_block("risk_governor")
                log.info("Signal blocked: %s", check.reason)
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": check.reason,
                        "ml_prob": ml_decision.probability,
                        "ml_threshold": ml_decision.threshold,
                        "coin": w.coin,
                    },
                )
                continue

            perf_mult = self.risk.performance_multiplier()
            regime_mult = self.risk.regime_multiplier(regime.regime.value, regime.session)
            ml_mult = self._ml_risk_multiplier(ml_decision.probability, ml_decision.threshold)
            # Signal-quality multiplier: scale size by the (rewritten, data-grounded)
            # signal_score. Conservative range [0.8, 1.2]: low-quality setups take 80% size,
            # top-tier setups take 120%. The governor still clamps the final product to
            # [risk_mult_min, risk_mult_max].
            signal_quality_mult = max(0.8, min(1.2, 0.8 + 0.4 * float(signal.signal_score)))
            risk_mult = perf_mult * regime_mult * ml_mult * signal_quality_mult
            warmup_on = self._in_warmup_mode()
            if warmup_on:
                cap = max(self.cfg.risk.risk_mult_min, self.cfg.strategy.warmup_risk_mult_cap)
                risk_mult = min(risk_mult, cap)
            if risk_mult <= 0:
                self._count_block("risk_multiplier_zero")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": "risk_multiplier_zero",
                        "ml_prob": ml_decision.probability,
                        "ml_threshold": ml_decision.threshold,
                        "warmup": warmup_on,
                        "coin": w.coin,
                    },
                )
                continue

            sizing = self.risk.size_position(signal.entry_price, signal.stop_price, risk_multiplier=risk_mult)
            if sizing.qty <= 0:
                self._count_block("size_zero")
                log.info("Signal skipped: size=0 (entry=%.2f stop=%.2f)", signal.entry_price, signal.stop_price)
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": "size_zero",
                        "ml_prob": ml_decision.probability,
                        "ml_threshold": ml_decision.threshold,
                        "risk_mult": risk_mult,
                        "warmup": warmup_on,
                        "coin": w.coin,
                    },
                )
                continue

            self.journal.write(
                "decision",
                signal_id,
                {
                    "ts_ms": bar.end_ms,
                    "allowed": True,
                    "reason": f"pass:{ml_gate_mode}:{ml_decision.reason}",
                    "ml_prob": ml_decision.probability,
                    "ml_threshold": ml_decision.threshold,
                    "risk_mult": risk_mult,
                    "warmup": warmup_on,
                    "coin": w.coin,
                },
            )
            self._signal_context.setdefault(signal_id, {})
            self._signal_context[signal_id]["ml_prob"] = ml_decision.probability
            self._signal_context[signal_id]["ml_threshold"] = ml_decision.threshold
            try:
                update = w.executor.submit_entry(signal, signal_id=signal_id, qty=sizing.qty, risk_dollars=sizing.risk_dollars)
            except RuntimeError as exc:
                # SDK/network error mid-submit. submit_entry already logs context.
                # Journal the failure so the signal isn't silently lost.
                self._count_block("submit_entry_error")
                log.warning("submit_entry RuntimeError for signal_id=%s coin=%s: %s",
                            signal_id, w.coin, exc)
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": "submit_entry_error",
                        "error": str(exc),
                        "coin": w.coin,
                    },
                )
                continue

            # The executor may pre-flight-reject (sub-min-notional, sub-min-lot
            # after cap-clamp). Treat as a counted block, not a placed entry.
            if update.event_type == ExecEventType.ENTRY_REJECTED:
                self._count_block("hl_pre_flight_reject")
                log.info("Signal blocked: %s", update.message)
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": update.ts_ms,
                        "allowed": False,
                        "reason": "hl_pre_flight_reject",
                        "message": update.message,
                        "coin": w.coin,
                    },
                )
                continue

            self._entries_placed += 1
            w.entries_placed += 1
            self.journal.write(
                "lifecycle",
                signal_id,
                {
                    "ts_ms": update.ts_ms,
                    "event": update.event_type.value,
                    "message": update.message,
                    "coin": w.coin,
                },
            )
            log.info(
                "%s | coin=%s id=%s label=%s regime=%s conf=%.2f score=%.2f ml_mode=%s ml_prob=%.3f ml_thr=%.3f ml_reason=%s risk_mult=%.2f warmup=%s entry=%.2f stop=%.2f tp1=%.2f tp2=%.2f qty=%.6f",
                update.message,
                w.coin,
                signal_id,
                signal.level_label,
                regime.regime.value,
                signal.confidence,
                signal.signal_score,
                ml_gate_mode,
                ml_decision.probability,
                ml_decision.threshold,
                ml_decision.reason,
                risk_mult,
                warmup_on,
                signal.entry_price,
                signal.stop_price,
                signal.tp1_price,
                signal.tp2_price,
                sizing.qty,
            )

    def _build_market_state(self, ts_ms: int) -> MarketState:
        now_ms = self._now_ms()
        stale_ms = self.cfg.feed.stale_data_sec * 1000
        data_stale = self._last_event_ms == 0 or (now_ms - self._last_event_ms > stale_ms)
        if self.cfg.mode == "replay":
            ws_healthy = self._last_event_ms > 0 and not data_stale
        else:
            ws_healthy = self.feed.last_message_ms > 0 and not data_stale
        return MarketState(
            ts_ms=ts_ms,
            ws_healthy=ws_healthy,
            data_stale=data_stale,
            spread_bps=self._last_spread_bps,
            recent_bar_ranges_pct=list(self._recent_bar_ranges),
            move_30s_pct=self._move_30s_pct(),
        )

    def _trim_recent_trade_prices(self, now_ms: int) -> None:
        cutoff = now_ms - 30_000
        while self._recent_trade_prices and self._recent_trade_prices[0][0] < cutoff:
            self._recent_trade_prices.popleft()

    def _trim_recent_signed_flow(self, now_ms: int) -> None:
        cutoff = now_ms - 30_000
        while self._recent_signed_flow and self._recent_signed_flow[0][0] < cutoff:
            self._recent_signed_flow.popleft()

    def _move_30s_pct(self) -> float:
        if len(self._recent_trade_prices) < 2:
            return 0.0
        first = self._recent_trade_prices[0][1]
        last = self._recent_trade_prices[-1][1]
        if first <= 0:
            return 0.0
        return ((last - first) / first) * 100.0

    def _next_signal_id(self, ts_ms: int) -> str:
        self._signal_seq += 1
        return f"{ts_ms}-{self._signal_seq}"

    def _regime_state(self, ts_ms: int) -> RegimeState:
        closes = list(self._recent_closes)[-self.cfg.strategy.trend_lookback_bars :]
        ranges = list(self._recent_bar_ranges)[-self.cfg.strategy.trend_lookback_bars :]
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return classify_regime(
            closes=closes,
            ranges_pct=ranges,
            spread_bps=self._last_spread_bps,
            move_30s_pct=self._move_30s_pct(),
            trend_threshold_bps=self.cfg.strategy.max_trend_move_bps,
            high_vol_threshold_pct=self.cfg.strategy.max_bar_range_pct,
            illiquid_spread_bps=self.cfg.strategy.max_spread_bps * 1.8,
            hour_utc=dt.hour,
        )

    def _regime_blocks_signal(self, regime: RegimeState) -> bool:
        if regime.regime == Regime.HIGH_VOL and self.cfg.strategy.disable_in_high_vol:
            return True
        if regime.regime == Regime.ILLIQUID and self.cfg.strategy.disable_in_illiquid:
            return True
        return False

    def _ml_decision_mode(self) -> str:
        mode = str(self.cfg.runtime.ml_decision_mode or "rank").strip().lower()
        return mode if mode in {"gate", "rank"} else "rank"

    def _ml_risk_multiplier(self, probability: float, threshold: float) -> float:
        if threshold <= 1e-9:
            return 1.0
        ratio = probability / threshold
        if self._ml_decision_mode() == "gate":
            return max(0.7, min(1.15, ratio))
        return max(0.85, min(1.10, ratio))

    def _signal_features(self, signal, state: MarketState, regime: RegimeState, bar) -> dict[str, float]:
        stop_dist_bps = abs(signal.entry_price - signal.stop_price) / max(signal.entry_price, 1e-9) * 10_000.0
        tp1_dist_bps = abs(signal.tp1_price - signal.entry_price) / max(signal.entry_price, 1e-9) * 10_000.0
        tp2_dist_bps = abs(signal.tp2_price - signal.entry_price) / max(signal.entry_price, 1e-9) * 10_000.0
        stop_dist_abs = abs(signal.entry_price - signal.stop_price)
        rr_tp1 = abs(signal.tp1_price - signal.entry_price) / max(stop_dist_abs, 1e-9)
        rr_tp2 = abs(signal.tp2_price - signal.entry_price) / max(stop_dist_abs, 1e-9)
        flow_ratio = self._signed_flow_ratio(self.cfg.strategy.micro_flow_window_sec * 1000)
        flow_abs = self._signed_flow_abs(self.cfg.strategy.micro_flow_window_sec * 1000)
        queue_imb = self._queue_imbalance()
        micro_delta_bps = self._microprice_delta_bps()
        depth_total = max(0.0, self._last_bid_size) + max(0.0, self._last_ask_size)
        return {
            "side_short": 1.0 if signal.side.value == "short" else 0.0,
            "confidence": signal.confidence,
            "signal_score": signal.signal_score,
            "overshoot_bps": signal.overshoot_bps,
            "reclaim_bps": signal.reclaim_bps,
            "volume_ratio": signal.volume_ratio,
            "wick_ratio": signal.wick_ratio,
            "spread_bps": state.spread_bps,
            "move_30s_pct": state.move_30s_pct,
            "bar_range_pct": bar.range_pct,
            "stop_dist_bps": stop_dist_bps,
            "tp1_dist_bps": tp1_dist_bps,
            "tp2_dist_bps": tp2_dist_bps,
            "rr_tp1": rr_tp1,
            "rr_tp2": rr_tp2,
            "flow_ratio": flow_ratio,
            "flow_abs": flow_abs,
            "queue_imbalance": queue_imb,
            "microprice_delta_bps": micro_delta_bps,
            "book_depth_total": depth_total,
            "regime_trend": 1.0 if regime.regime == Regime.TREND else 0.0,
            "regime_range": 1.0 if regime.regime == Regime.RANGE else 0.0,
            "trend_bps": regime.trend_bps,
            "avg_range_pct": regime.avg_range_pct,
            "session_us": 1.0 if regime.session == "us" else 0.0,
            "session_eu": 1.0 if regime.session == "eu" else 0.0,
            "session_asia": 1.0 if regime.session == "asia" else 0.0,
        }

    def _signed_flow_ratio(self, window_ms: int) -> float:
        signed, total_abs = self._signed_flow_stats(window_ms)
        if total_abs <= 1e-9:
            return 0.0
        return signed / total_abs

    def _signed_flow_abs(self, window_ms: int) -> float:
        _, total_abs = self._signed_flow_stats(window_ms)
        return total_abs

    def _signed_flow_stats(self, window_ms: int) -> tuple[float, float]:
        if window_ms <= 0:
            return 0.0, 0.0
        now_ts = self._recent_signed_flow[-1][0] if self._recent_signed_flow else 0
        if now_ts <= 0:
            return 0.0, 0.0
        cutoff = now_ts - window_ms
        signed = 0.0
        total_abs = 0.0
        for ts, sv in reversed(self._recent_signed_flow):
            if ts < cutoff:
                break
            signed += sv
            total_abs += abs(sv)
        return signed, total_abs

    def _queue_imbalance(self) -> float:
        b = max(0.0, self._last_bid_size)
        a = max(0.0, self._last_ask_size)
        denom = b + a
        if denom <= 1e-9:
            return 0.0
        return (b - a) / denom

    def _microprice_delta_bps(self) -> float:
        bid = self._last_best_bid
        ask = self._last_best_ask
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0.0
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 0.0
        bsz = max(0.0, self._last_bid_size)
        asz = max(0.0, self._last_ask_size)
        denom = bsz + asz
        if denom <= 1e-9:
            return 0.0
        micro = ((ask * bsz) + (bid * asz)) / denom
        return ((micro - mid) / mid) * 10_000.0

    def _microstructure_check(self, side: Side):
        if not self.cfg.strategy.use_micro_confirm:
            return RiskCheck(True, "ok")
        flow_ratio = self._signed_flow_ratio(self.cfg.strategy.micro_flow_window_sec * 1000)
        qimb = self._queue_imbalance()
        min_ofi = max(0.0, self.cfg.strategy.min_ofi_ratio)
        min_q = max(0.0, self.cfg.strategy.min_queue_imbalance)
        warmup_on = self._in_warmup_mode()
        # Always use OR logic — any one microstructure signal confirming is sufficient
        soft_or = True
        if warmup_on and self.cfg.strategy.warmup_micro_relax:
            min_ofi *= max(0.0, self.cfg.strategy.warmup_ofi_scale)
            min_q *= max(0.0, self.cfg.strategy.warmup_qimb_scale)

        if side == Side.LONG:
            flow_ok = flow_ratio >= min_ofi
            q_ok = qimb >= min_q
            micro_ok = self._microprice_delta_bps() > 0  # microprice leaning bullish
            if flow_ok and q_ok:
                return RiskCheck(True, "ok")
            if soft_or and (flow_ok or q_ok or micro_ok):
                return RiskCheck(True, f"micro_softpass:flow={flow_ratio:.3f},qimb={qimb:.3f}")
            if not flow_ok:
                return RiskCheck(False, f"micro_ofi_fail:{flow_ratio:.3f}")
            if not q_ok:
                return RiskCheck(False, f"micro_qimb_fail:{qimb:.3f}")
            return RiskCheck(False, "micro_fail")

        flow_ok = flow_ratio <= -min_ofi
        q_ok = qimb <= -min_q
        micro_ok = self._microprice_delta_bps() < 0  # microprice leaning bearish
        if flow_ok and q_ok:
            return RiskCheck(True, "ok")
        if soft_or and (flow_ok or q_ok or micro_ok):
            return RiskCheck(True, f"micro_softpass:flow={flow_ratio:.3f},qimb={qimb:.3f}")
        if not flow_ok:
            return RiskCheck(False, f"micro_ofi_fail:{flow_ratio:.3f}")
        if not q_ok:
            return RiskCheck(False, f"micro_qimb_fail:{qimb:.3f}")
        return RiskCheck(False, "micro_fail")

    # --- Worker-parameterized helpers (used by _handle_event for multi-coin) ---

    def _trim_recent_trade_prices_w(self, w: CoinWorker, now_ms: int) -> None:
        cutoff = now_ms - 30_000
        while w.recent_trade_prices and w.recent_trade_prices[0][0] < cutoff:
            w.recent_trade_prices.popleft()

    def _trim_recent_signed_flow_w(self, w: CoinWorker, now_ms: int) -> None:
        cutoff = now_ms - 30_000
        while w.recent_signed_flow and w.recent_signed_flow[0][0] < cutoff:
            w.recent_signed_flow.popleft()

    def _move_30s_pct_w(self, w: CoinWorker) -> float:
        if len(w.recent_trade_prices) < 2:
            return 0.0
        first = w.recent_trade_prices[0][1]
        last = w.recent_trade_prices[-1][1]
        if first <= 0:
            return 0.0
        return ((last - first) / first) * 100.0

    def _build_market_state_w(self, w: CoinWorker, ts_ms: int) -> MarketState:
        now_ms = self._now_ms()
        stale_ms = self.cfg.feed.stale_data_sec * 1000
        data_stale = self._last_event_ms == 0 or (now_ms - self._last_event_ms > stale_ms)
        if self.cfg.mode == "replay":
            ws_healthy = self._last_event_ms > 0 and not data_stale
        else:
            ws_healthy = self.feed.last_message_ms > 0 and not data_stale
        return MarketState(
            ts_ms=ts_ms,
            ws_healthy=ws_healthy,
            data_stale=data_stale,
            spread_bps=w.last_spread_bps,
            recent_bar_ranges_pct=list(w.recent_bar_ranges),
            move_30s_pct=self._move_30s_pct_w(w),
        )

    def _regime_state_w(self, w: CoinWorker, ts_ms: int) -> RegimeState:
        closes = list(w.recent_closes)[-self.cfg.strategy.trend_lookback_bars :]
        ranges = list(w.recent_bar_ranges)[-self.cfg.strategy.trend_lookback_bars :]
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return classify_regime(
            closes=closes,
            ranges_pct=ranges,
            spread_bps=w.last_spread_bps,
            move_30s_pct=self._move_30s_pct_w(w),
            trend_threshold_bps=self.cfg.strategy.max_trend_move_bps,
            high_vol_threshold_pct=self.cfg.strategy.max_bar_range_pct,
            illiquid_spread_bps=self.cfg.strategy.max_spread_bps * 1.8,
            hour_utc=dt.hour,
        )

    def _signed_flow_ratio_w(self, w: CoinWorker, window_ms: int) -> float:
        signed, total_abs = self._signed_flow_stats_w(w, window_ms)
        if total_abs <= 1e-9:
            return 0.0
        return signed / total_abs

    def _signed_flow_abs_w(self, w: CoinWorker, window_ms: int) -> float:
        _, total_abs = self._signed_flow_stats_w(w, window_ms)
        return total_abs

    def _signed_flow_stats_w(self, w: CoinWorker, window_ms: int) -> tuple[float, float]:
        if window_ms <= 0:
            return 0.0, 0.0
        now_ts = w.recent_signed_flow[-1][0] if w.recent_signed_flow else 0
        if now_ts <= 0:
            return 0.0, 0.0
        cutoff = now_ts - window_ms
        signed = 0.0
        total_abs = 0.0
        for ts, sv in reversed(w.recent_signed_flow):
            if ts < cutoff:
                break
            signed += sv
            total_abs += abs(sv)
        return signed, total_abs

    def _queue_imbalance_w(self, w: CoinWorker) -> float:
        b = max(0.0, w.last_bid_size)
        a = max(0.0, w.last_ask_size)
        denom = b + a
        if denom <= 1e-9:
            return 0.0
        return (b - a) / denom

    def _microprice_delta_bps_w(self, w: CoinWorker) -> float:
        bid = w.last_best_bid
        ask = w.last_best_ask
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0.0
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 0.0
        bsz = max(0.0, w.last_bid_size)
        asz = max(0.0, w.last_ask_size)
        denom = bsz + asz
        if denom <= 1e-9:
            return 0.0
        micro = ((ask * bsz) + (bid * asz)) / denom
        return ((micro - mid) / mid) * 10_000.0

    def _signal_features_w(self, w: CoinWorker, signal: SweepSignal, state: MarketState, regime: RegimeState, bar) -> dict[str, float]:
        stop_dist_bps = abs(signal.entry_price - signal.stop_price) / max(signal.entry_price, 1e-9) * 10_000.0
        tp1_dist_bps = abs(signal.tp1_price - signal.entry_price) / max(signal.entry_price, 1e-9) * 10_000.0
        tp2_dist_bps = abs(signal.tp2_price - signal.entry_price) / max(signal.entry_price, 1e-9) * 10_000.0
        stop_dist_abs = abs(signal.entry_price - signal.stop_price)
        rr_tp1 = abs(signal.tp1_price - signal.entry_price) / max(stop_dist_abs, 1e-9)
        rr_tp2 = abs(signal.tp2_price - signal.entry_price) / max(stop_dist_abs, 1e-9)
        flow_ratio = self._signed_flow_ratio_w(w, self.cfg.strategy.micro_flow_window_sec * 1000)
        flow_abs = self._signed_flow_abs_w(w, self.cfg.strategy.micro_flow_window_sec * 1000)
        queue_imb = self._queue_imbalance_w(w)
        micro_delta_bps = self._microprice_delta_bps_w(w)
        depth_total = max(0.0, w.last_bid_size) + max(0.0, w.last_ask_size)
        return {
            "side_short": 1.0 if signal.side.value == "short" else 0.0,
            "confidence": signal.confidence,
            "signal_score": signal.signal_score,
            "overshoot_bps": signal.overshoot_bps,
            "reclaim_bps": signal.reclaim_bps,
            "volume_ratio": signal.volume_ratio,
            "wick_ratio": signal.wick_ratio,
            "spread_bps": state.spread_bps,
            "move_30s_pct": state.move_30s_pct,
            "bar_range_pct": bar.range_pct,
            "stop_dist_bps": stop_dist_bps,
            "tp1_dist_bps": tp1_dist_bps,
            "tp2_dist_bps": tp2_dist_bps,
            "rr_tp1": rr_tp1,
            "rr_tp2": rr_tp2,
            "flow_ratio": flow_ratio,
            "flow_abs": flow_abs,
            "queue_imbalance": queue_imb,
            "microprice_delta_bps": micro_delta_bps,
            "book_depth_total": depth_total,
            "regime_trend": 1.0 if regime.regime == Regime.TREND else 0.0,
            "regime_range": 1.0 if regime.regime == Regime.RANGE else 0.0,
            "trend_bps": regime.trend_bps,
            "avg_range_pct": regime.avg_range_pct,
            "session_us": 1.0 if regime.session == "us" else 0.0,
            "session_eu": 1.0 if regime.session == "eu" else 0.0,
            "session_asia": 1.0 if regime.session == "asia" else 0.0,
        }

    def _microstructure_check_w(self, w: CoinWorker, side: Side) -> RiskCheck:
        if not self.cfg.strategy.use_micro_confirm:
            return RiskCheck(True, "ok")
        flow_ratio = self._signed_flow_ratio_w(w, self.cfg.strategy.micro_flow_window_sec * 1000)
        qimb = self._queue_imbalance_w(w)
        min_ofi = max(0.0, self.cfg.strategy.min_ofi_ratio)
        min_q = max(0.0, self.cfg.strategy.min_queue_imbalance)
        warmup_on = self._in_warmup_mode()
        soft_or = True
        if warmup_on and self.cfg.strategy.warmup_micro_relax:
            min_ofi *= max(0.0, self.cfg.strategy.warmup_ofi_scale)
            min_q *= max(0.0, self.cfg.strategy.warmup_qimb_scale)

        if side == Side.LONG:
            flow_ok = flow_ratio >= min_ofi
            q_ok = qimb >= min_q
            micro_ok = self._microprice_delta_bps_w(w) > 0
            if flow_ok and q_ok:
                return RiskCheck(True, "ok")
            if soft_or and (flow_ok or q_ok or micro_ok):
                return RiskCheck(True, f"micro_softpass:flow={flow_ratio:.3f},qimb={qimb:.3f}")
            if not flow_ok:
                return RiskCheck(False, f"micro_ofi_fail:{flow_ratio:.3f}")
            if not q_ok:
                return RiskCheck(False, f"micro_qimb_fail:{qimb:.3f}")
            return RiskCheck(False, "micro_fail")

        flow_ok = flow_ratio <= -min_ofi
        q_ok = qimb <= -min_q
        micro_ok = self._microprice_delta_bps_w(w) < 0
        if flow_ok and q_ok:
            return RiskCheck(True, "ok")
        if soft_or and (flow_ok or q_ok or micro_ok):
            return RiskCheck(True, f"micro_softpass:flow={flow_ratio:.3f},qimb={qimb:.3f}")
        if not flow_ok:
            return RiskCheck(False, f"micro_ofi_fail:{flow_ratio:.3f}")
        if not q_ok:
            return RiskCheck(False, f"micro_qimb_fail:{qimb:.3f}")
        return RiskCheck(False, "micro_fail")

    def _in_funding_blackout(self, ts_ms: int) -> bool:
        if not self.cfg.strategy.use_funding_blackout:
            return False
        blackout = max(0, self.cfg.strategy.funding_blackout_sec)
        if blackout <= 0:
            return False
        sec_of_day = (ts_ms // 1000) % 86_400
        mod = sec_of_day % 28_800  # funding cadence: 8h
        return mod <= blackout or mod >= (28_800 - blackout)

    def _maybe_heartbeat(self) -> None:
        now_ms = self._now_ms()
        if now_ms - self._last_heartbeat_ms < self._heartbeat_interval_ms:
            return
        self._last_heartbeat_ms = now_ms
        exposures: list[str] = []
        for coin, w in self._workers.items():
            if w.executor.position is not None:
                exposures.append(f"{coin}:position:{w.executor.position.side.value}")
            elif w.executor.pending_entry is not None:
                exposures.append(f"{coin}:pending:{w.executor.pending_entry.side.value}")
        exposure = ",".join(exposures) if exposures else "flat"
        first_w = self._first_worker
        # Top-N block reasons sorted by count, formatted compactly for the heartbeat.
        if self._block_reasons:
            top_reasons = sorted(self._block_reasons.items(), key=lambda kv: -kv[1])[:5]
            block_breakdown = "(" + " ".join(f"{k}={v}" for k, v in top_reasons) + ")"
        else:
            block_breakdown = ""
        log.info(
            (
                "Heartbeat events(trade=%d book=%d) bars=%d signals=%d blocked=%d%s "
                "entries(placed=%d filled=%d closed=%d) spread=%.2fbps move30s=%.3f%% "
                "exposure=%s warmup=%s resolved=%d qsize=%d drops=%d coins=%s"
            ),
            self._trade_events,
            self._book_events,
            self._bars_closed,
            self._signals_seen,
            self._signals_blocked,
            block_breakdown,
            self._entries_placed,
            self._entries_filled,
            self._positions_closed,
            first_w.last_spread_bps,
            self._move_30s_pct(),
            exposure,
            self._in_warmup_mode(),
            self._resolved_trades,
            self._event_queue.qsize(),
            self._event_drop_count,
            list(self._workers.keys()),
        )
        for w in self._workers.values():
            diag = w.detector.consume_diagnostics(top_n=10)
            if diag:
                summary = ", ".join(f"{k}={v}" for k, v in diag)
                log.info("Detector diagnostics [%s]: %s", w.coin, summary)
        self._maybe_auto_train()

    def _maybe_refresh_deadmans(self, now_ms: int) -> None:
        """Push HL's schedule_cancel timer forward when the per-executor refresh
        cadence is due. Called per market-tick (NOT per-heartbeat) so the actual
        SDK push fires at the configured deadman_refresh_sec interval with a
        safety margin against timing jitter from the 60s heartbeat."""
        for w in self._workers.values():
            if hasattr(w.executor, "should_refresh_deadman") and w.executor.should_refresh_deadman(now_ms):
                w.executor.refresh_deadman(now_ms)

    def runtime_summary(self) -> dict[str, float | int | dict[str, int]]:
        return {
            "trade_events": self._trade_events,
            "book_events": self._book_events,
            "bars_closed": self._bars_closed,
            "signals_seen": self._signals_seen,
            "signals_blocked": self._signals_blocked,
            "block_reasons": dict(self._block_reasons),
            "entries_placed": self._entries_placed,
            "entries_filled": self._entries_filled,
            "positions_closed": self._positions_closed,
            "resolved_trades": self._resolved_trades,
            "daily_pnl": round(self.risk.daily_pnl, 6),
            "daily_r": round(self.risk.daily_r, 6),
            "equity": round(self.risk.equity, 6),
        }

    def _in_warmup_mode(self) -> bool:
        if not self.cfg.strategy.warmup_enabled:
            return False
        target = max(1, self.cfg.strategy.warmup_target_resolved)
        return self._resolved_trades < target

    def _maybe_auto_train(self) -> None:
        rt = self.cfg.runtime
        if self._suspend_auto_train:
            return
        if not rt.ml_enabled:
            return
        provider = rt.ml_provider.strip().lower()
        if provider not in {"logistic", "ensemble"}:
            return
        if not rt.ml_auto_train:
            return
        now_ms = self._now_ms()
        interval_ms = max(60, rt.ml_auto_train_interval_sec) * 1000
        if now_ms - self._last_auto_train_ms < interval_ms:
            return

        self._last_auto_train_ms = now_ms
        min_resolved = max(1, rt.ml_auto_train_min_resolved)
        if self._resolved_trades < min_resolved:
            log.info(
                "Auto-ML train skipped: resolved_trades=%d < min=%d",
                self._resolved_trades,
                min_resolved,
            )
            return

        min_new = max(1, rt.ml_auto_train_min_new_trades)
        new_since_last = self._resolved_trades - self._last_auto_train_resolved
        if new_since_last < min_new:
            log.info(
                "Auto-ML train skipped: new_resolved=%d < min_new=%d",
                new_since_last,
                min_new,
            )
            return

        script_path = self._project_root / "scripts" / "train_gate.py"
        input_path = self._resolve_project_path(rt.journal_path)
        model_path = self._resolve_project_path(rt.ml_model_path)
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(model_path),
            "--default-min-prob",
            f"{rt.ml_min_prob:.4f}",
            "--min-samples",
            str(min_resolved),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=180,
            )
        except Exception as exc:
            log.warning("Auto-ML train failed to launch: %s", exc)
            return

        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            tail = msg.splitlines()[-1] if msg else f"exit_{proc.returncode}"
            log.warning("Auto-ML train failed: %s", tail)
            return

        rec = self._load_recommended_min_prob(str(model_path))
        if rec is not None and rt.ml_auto_apply_threshold:
            old = rt.ml_min_prob
            rt.ml_min_prob = rec
            self._persist_runtime_ml_state()
            log.info("Auto-ML threshold applied: BOT_ML_MIN_PROB %.3f -> %.3f", old, rec)

        self.ml_gate.reload()
        self._last_auto_train_resolved = self._resolved_trades
        log.info(
            "Auto-ML train complete: resolved=%d new=%d model=%s provider=%s",
            self._resolved_trades,
            new_since_last,
            model_path,
            rt.ml_provider,
        )

    def _load_recommended_min_prob(self, path: str) -> float | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = self._parse_float(raw.get("recommended_min_prob"))
        if value is None:
            return None
        return max(0.0, min(1.0, value))
