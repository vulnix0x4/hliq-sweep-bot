from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import json
import logging
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from hliq_bot.ai.market_data import MarketDataCache
from hliq_bot.ai.memory import AIMemory, MemoryEntry
from hliq_bot.ai.strategy import AIDecisionResult, AIStrategy, read_override_flag
from hliq_bot.analytics.market_capture import MarketCaptureWriter
from hliq_bot.analytics.journal import SignalJournal
from hliq_bot.config import AppConfig
from hliq_bot.data.bar_builder import BarBuilder
from hliq_bot.data.hyperliquid_ws import HyperliquidWsClient
from hliq_bot.execution.order_manager import PaperOrderManager
from hliq_bot.ml.gate import MLGate
from hliq_bot.models import Bar, ClosedTrade, ExecEventType, MarketEvent, MarketState, RiskCheck, Side, SweepSignal
from hliq_bot.risk.governor import RiskGovernor
from hliq_bot.signal.regime import Regime, RegimeState, classify_regime
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.sweep_detector import SweepDetector
from hliq_bot.signal.vwap_tracker import VWAPTracker

log = logging.getLogger(__name__)


def _matches_block_pattern(value: str, patterns: set[str]) -> bool:
    value = value.strip()
    if not value:
        return False
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _matches_coin_level_pair(coin: str, level: str, patterns: set[str]) -> bool:
    pair = f"{coin.strip().upper()}:{level.strip().lower()}"
    return _matches_block_pattern(pair, patterns)


def _matches_coin_session_pair(coin: str, session: str, patterns: set[str]) -> bool:
    pair = f"{coin.strip().upper()}:{session.strip().lower()}"
    return _matches_block_pattern(pair, patterns)


def _matches_coin_session_level_triple(coin: str, session: str, level: str, patterns: set[str]) -> bool:
    triple = f"{coin.strip().upper()}:{session.strip().lower()}:{level.strip().lower()}"
    return _matches_block_pattern(triple, patterns)


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
    # Full Bar objects (not just closes) — needed for the AI strategy's
    # context window. Sized to fit the largest reasonable context_bars.
    recent_bars: deque = field(default_factory=lambda: deque(maxlen=120))
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
        self.run_id = self._make_run_id(config, int(time.time() * 1000))
        self.feed = HyperliquidWsClient(config.feed)
        self.risk = RiskGovernor(config.risk, config.strategy)
        self.ml_gate = MLGate(config.runtime)
        self.journal = SignalJournal(
            config.runtime.journal_path,
            default_context={"run_id": self.run_id, "mode": config.mode},
        )
        self.capture = (
            MarketCaptureWriter(
                config.runtime.market_capture_path,
                max_bytes=config.runtime.market_capture_max_bytes,
                backups=config.runtime.market_capture_backups,
            )
            if config.runtime.market_capture_enabled
            else None
        )
        self._signal_context: dict[str, dict[str, float | str]] = {}

        # Build per-coin workers
        range_maxlen = max(3, config.strategy.circuit_range_bars + 3)
        # recent_closes must hold enough history for BOTH the trend filter and
        # the regime MA. Sizing only on trend_lookback_bars made the regime
        # filter a silent no-op at defaults (deque=25 but ma_bars=30).
        closes_maxlen = max(
            5,
            config.strategy.trend_lookback_bars + 5,
            config.strategy.regime_filter_ma_bars + 5 if config.strategy.regime_filter_enabled else 0,
        )
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
        # Last time we observed WS healthy. Used as a 30s grace window so a
        # momentary feed flap doesn't kill an in-flight AI trade decision.
        self._ws_last_healthy_ms = 0
        self._heartbeat_interval_ms = 60_000
        now_ms = self._now_ms()
        self._write_run_start(now_ms)
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

        # AI strategy: when enabled, polls an LLM per coin on a timer and
        # acts on its decisions instead of the rule-based sweep detector.
        self.ai_strategy: AIStrategy | None = None
        self.ai_memory: AIMemory | None = None
        self.ai_market_data: MarketDataCache | None = None
        if config.ai.enabled:
            mem_path = self._resolve_project_path(config.ai.memory_path)
            self.ai_memory = AIMemory(mem_path, max_entries=config.ai.memory_max_entries)
            loaded_count = self.ai_memory.load()
            # Construct MarketDataCache lazily — uses HL Info SDK. Skip on
            # paper mode without HL credentials; AI will still work but
            # without funding/OI/L2 enrichment.
            try:
                from hyperliquid.info import Info
                from hyperliquid.utils import constants
                api_url = (
                    constants.MAINNET_API_URL
                    if config.live.network == "mainnet"
                    else constants.TESTNET_API_URL
                )
                self.ai_market_data = MarketDataCache(Info(api_url, skip_ws=True))
            except Exception as exc:
                log.warning(
                    "AI market_data cache disabled (HL SDK init failed: %s) — "
                    "funding/OI/L2 enrichment will be absent from prompts.", exc,
                )
            self.ai_strategy = AIStrategy(
                config.ai, memory=self.ai_memory, market_data=self.ai_market_data,
            )
            # Prime the rolling-24h cost tracker from journal so restarting the
            # container doesn't reset the daily-budget cap to $0 — otherwise
            # several restarts in a day let the bot spend N x the cap.
            primed = self.ai_strategy.budget.prime_from_journal(config.runtime.journal_path)
            if primed > 0:
                spent = self.ai_strategy.budget.spent_last_24h()
                log.info(
                    "AI cost tracker primed from journal: %d past calls in 24h window, spent=$%.4f / cap=$%.2f",
                    primed, spent, config.ai.daily_budget_usd,
                )
            # AI strategy wants longer holds than sweep — raise the executor's
            # time-stop so positions can run as long as the AI intends. Don't
            # shrink: the existing value is the operator's choice.
            if config.ai.max_holding_sec > config.strategy.max_holding_sec:
                log.info(
                    "AI mode: raising strategy.max_holding_sec %d -> %d to match BOT_AI_MAX_HOLDING_SEC",
                    config.strategy.max_holding_sec, config.ai.max_holding_sec,
                )
                config.strategy.max_holding_sec = config.ai.max_holding_sec
            stats = self.ai_memory.summary_stats()
            log.info(
                "AI strategy ENABLED: model=%s fallbacks=%s interval=%ds budget=$%.2f/day "
                "max_holding=%ds prompt=%s memory=%s (loaded=%d resolved=%d avg_r=%.3f pnl=%.4f)",
                config.ai.model, config.ai.fallback_models, config.ai.interval_sec,
                config.ai.daily_budget_usd, config.ai.max_holding_sec,
                config.ai.prompt_version, mem_path, loaded_count,
                stats["resolved_trades"], stats["avg_r"], stats["total_pnl"],
            )
        # Buffer of recent AI-driven closed trades for self-reflection in
        # later prompts. Trimmed to last 10 for token efficiency.
        self._ai_recent_outcomes: list[dict] = []

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

                    # Prefer the explicit risk_dollars on the row (schema >=2).
                    # Legacy rows didn't journal it, so derive from pnl/r_mult.
                    # The legacy derivation is fragile (fails on pnl=0, drops
                    # precision) so the explicit field is strongly preferred.
                    explicit_risk = self._parse_float(row.get("risk_dollars"))
                    if explicit_risk is not None and explicit_risk > 0:
                        derived_risk = explicit_risk
                    else:
                        derived_risk = (
                            abs(pnl) / max(abs(r_mult), 1e-6)
                            if abs(r_mult) > 1e-6
                            else 1.0
                        )
                    trade = ClosedTrade(
                        signal_id=sid,
                        side=side,
                        entry_price=0.0,
                        exit_price=0.0,
                        qty=0.0,
                        pnl=pnl,
                        risk_dollars=max(derived_risk, 1e-6),
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
        rt.trade_pause_path = str(self._resolve_project_path(rt.trade_pause_path))
        rt.market_capture_path = str(self._resolve_project_path(rt.market_capture_path))
        rt.ml_model_path = str(self._resolve_project_path(rt.ml_model_path))
        rt.ml_state_path = str(self._resolve_project_path(rt.ml_state_path))
        self.cfg.replay.input_path = str(self._resolve_project_path(self.cfg.replay.input_path))

    def _runtime_pause_reason(self) -> str | None:
        path = Path(self.cfg.runtime.trade_pause_path)
        if not path.exists():
            return None
        try:
            detail = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        except OSError:
            detail = []
        reason = detail[0].strip() if detail else "operator_pause"
        reason = reason[:120] or "operator_pause"
        return f"runtime_pause:{reason}"

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

    def _policy_log_summary(self) -> dict[str, object]:
        s = self.cfg.strategy
        return {
            "allow_coins": sorted(s.allowed_coins),
            "allow_levels": sorted(s.allowed_level_labels),
            "allow_coin_levels": sorted(s.allowed_coin_level_pairs),
            "allow_coin_sessions": sorted(s.allowed_coin_session_pairs),
            "allow_coin_session_levels": sorted(s.allowed_coin_session_level_triples),
            "allow_sessions": sorted(s.allowed_sessions),
            "allow_sides": sorted(s.allowed_sides),
            "block_coins": sorted(s.blocked_coins),
            "block_levels": sorted(s.blocked_level_labels),
            "block_coin_levels": sorted(s.blocked_coin_level_pairs),
            "block_coin_sessions": sorted(s.blocked_coin_session_pairs),
            "block_coin_session_levels": sorted(s.blocked_coin_session_level_triples),
            "block_sessions": sorted(s.blocked_sessions),
            "block_sides": sorted(s.blocked_sides),
            "min_signal_score": s.min_signal_score,
            "min_conf_range": s.min_confidence_range,
            "maker_fee_pct": s.maker_fee_pct,
            "taker_fee_pct": s.taker_fee_pct,
            "paper_entry_slippage_bps": s.paper_entry_slippage_bps,
            "paper_exit_slippage_bps": s.paper_exit_slippage_bps,
            "paper_tp1_is_taker": s.paper_tp1_is_taker,
        }

    async def run(self) -> None:
        log.info(
            (
                "Starting sweep bot run_id=%s mode=%s coins=%s timeframe=%ss ml_enabled=%s ml_mode=%s ml_provider=%s "
                "ml_auto_train=%s interval=%ss min_resolved=%d min_new=%d warmup=%s target=%d journal=%s capture=%s"
            ),
            self.run_id,
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
        log.info("Active operator policy: %s", json.dumps(self._policy_log_summary(), sort_keys=True))
        # Surface micro-check mode so operators are not surprised by low admission rate.
        # Outside warmup, soft_or is False -> every signal requires BOTH OFI and queue-imbalance
        # to clear; combined with min_signal_score / regime filters / allowlists this stacks.
        s = self.cfg.strategy
        soft_or_active = s.warmup_enabled and s.warmup_micro_or_logic
        log.info(
            "Microstructure gating: warmup_enabled=%s warmup_micro_or_logic=%s -> "
            "soft_or=%s (False means OFI AND queue-imbalance must BOTH clear; "
            "low admission is expected). regime_filter=%s min_signal_score=%.2f",
            s.warmup_enabled, s.warmup_micro_or_logic, soft_or_active,
            s.regime_filter_enabled, s.min_signal_score,
        )
        self._warm_start_history()
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

    def _warm_start_history(self) -> None:
        if not self.cfg.runtime.history_warm_start_enabled:
            return
        interval = self._candle_interval()
        if not interval:
            log.warning(
                "History warm-start skipped: unsupported timeframe_sec=%s",
                self.cfg.strategy.timeframe_sec,
            )
            return
        try:
            from hyperliquid.info import Info
            from hyperliquid.utils import constants

            feed_url = self.cfg.feed.ws_url.lower()
            api_url = constants.TESTNET_API_URL if "testnet" in feed_url else constants.MAINNET_API_URL
            info = Info(api_url, skip_ws=True)
        except Exception as exc:
            log.warning("History warm-start skipped: Hyperliquid SDK unavailable: %s", exc)
            return

        bars_requested = max(20, int(self.cfg.runtime.history_warm_start_bars))
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (bars_requested + 5) * self.cfg.strategy.timeframe_sec * 1000
        for coin, worker in self._workers.items():
            try:
                raw = info.candles_snapshot(coin, interval, start_ms, end_ms)
                bars = self._bars_from_candles(raw)
            except Exception as exc:
                log.warning("History warm-start failed for %s: %s", coin, exc)
                continue
            if not bars:
                log.warning("History warm-start found no candles for %s", coin)
                continue
            seeded = worker.detector.seed_history(bars)
            for bar in bars:
                worker.recent_bar_ranges.append(bar.range_pct)
                worker.recent_closes.append(bar.close)
                worker.session_tracker.on_bar(bar)
                worker.vwap_tracker.on_bar(bar)
            log.info(
                "History warm-start seeded %d %s candles for %s range_close=%.4f..%.4f",
                seeded,
                interval,
                coin,
                bars[0].close,
                bars[-1].close,
            )

    def _candle_interval(self) -> str:
        mapping = {
            60: "1m",
            180: "3m",
            300: "5m",
            900: "15m",
            1800: "30m",
            3600: "1h",
        }
        return mapping.get(int(self.cfg.strategy.timeframe_sec), "")

    @staticmethod
    def _bars_from_candles(candles: object) -> list[Bar]:
        if not isinstance(candles, list):
            return []
        bars: list[Bar] = []
        for row in candles:
            if not isinstance(row, dict):
                continue
            try:
                start_ms = int(row["t"])
                end_ms = int(row.get("T", start_ms + 60_000))
                open_px = float(row["o"])
                high = float(row["h"])
                low = float(row["l"])
                close = float(row["c"])
                volume = float(row.get("v", 0.0))
                trades = int(row.get("n", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if open_px <= 0 or high <= 0 or low <= 0 or close <= 0 or high < low:
                continue
            bars.append(
                Bar(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    open=open_px,
                    high=high,
                    low=low,
                    close=close,
                    volume=max(0.0, volume),
                    trade_count=max(0, trades),
                    vwap=close,
                    avg_spread_bps=0.0,
                )
            )
        return sorted(bars, key=lambda b: b.start_ms)

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
                        # Explicit risk denominator + schema version so restore
                        # reads the exact value used to compute r_multiple,
                        # instead of deriving abs(pnl)/abs(r_multiple) which
                        # fails when pnl=0 and is sensitive to rounding.
                        # Schema 2 = "risk_dollars is effective qty*stop_dist
                        # AFTER any notional clamp", matching what governor.py
                        # currently returns from size_position.
                        "risk_dollars": update.closed_trade.risk_dollars,
                        "risk_schema_version": 2,
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
                # Self-reflection buffer for the AI's next prompt — compact
                # record of recent outcomes so it can adapt its tactics.
                if self.ai_strategy is not None:
                    self._ai_recent_outcomes.append({
                        "side": update.closed_trade.side.value,
                        "exit_reason": update.closed_trade.exit_reason,
                        "r_multiple": round(update.closed_trade.r_multiple, 3),
                        "pnl": round(update.closed_trade.pnl, 4),
                        "session": session,
                    })
                    # Keep only last 10 — context budget.
                    if len(self._ai_recent_outcomes) > 10:
                        self._ai_recent_outcomes = self._ai_recent_outcomes[-10:]
                # Persistent memory: attach the outcome to the matching
                # decision so it survives restart and shows up in future
                # prompts as "you decided X and it produced Y".
                if self.ai_memory is not None:
                    ct = update.closed_trade
                    hold_sec = max(0.0, (ct.closed_ms - ct.opened_ms) / 1000.0)
                    self.ai_memory.record_outcome(
                        ct.signal_id,
                        ts_ms=ct.closed_ms,
                        exit_reason=ct.exit_reason,
                        pnl=ct.pnl,
                        r_multiple=ct.r_multiple,
                        hold_sec=hold_sec,
                    )
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
            w.recent_bars.append(bar)
            w.session_tracker.on_bar(bar)
            w.vwap_tracker.on_bar(bar)
            # AI mode: every bar close, check if it's this coin's turn to decide.
            # Skip the rule-based sweep detector entirely when AI is driving.
            if self.ai_strategy is not None:
                self._maybe_ai_decide(w, bar)
                continue
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
            pause_reason = self._runtime_pause_reason()
            if pause_reason is not None:
                self._count_block("runtime_pause")
                self.journal.write(
                    "decision",
                    signal_id,
                    {
                        "ts_ms": bar.end_ms,
                        "allowed": False,
                        "reason": pause_reason,
                        "coin": w.coin,
                    },
                )
                log.info("Signal blocked: %s", pause_reason)
                continue
            block = self._operator_blocklist_check(w.coin, signal, regime.session)
            if block is not None:
                bucket, reason = block
                self._count_block(bucket)
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
        # Track last-healthy timestamp + accept a 30s grace window. A
        # millisecond-scale WS flap shouldn't kill a trade signal that the
        # AI spent 2-3 seconds reasoning over. If the feed is genuinely
        # down for >30s, the strict check still fires.
        if ws_healthy:
            self._ws_last_healthy_ms = now_ms
        recently_healthy = (now_ms - self._ws_last_healthy_ms) <= 30_000
        return MarketState(
            ts_ms=ts_ms,
            ws_healthy=ws_healthy or recently_healthy,
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

    @staticmethod
    def _make_run_id(config: AppConfig, ts_ms: int) -> str:
        configured = config.runtime.run_id.strip()
        if configured:
            return configured
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return f"{dt.strftime('%Y%m%dT%H%M%SZ')}-{config.mode}"

    def _write_run_start(self, ts_ms: int) -> None:
        self.journal.write(
            "run",
            self.run_id,
            {
                "ts_ms": ts_ms,
                "event": "run_start",
                "schema": 2,
                "coins": self.cfg.feed.coins,
                "timeframe_sec": self.cfg.strategy.timeframe_sec,
                "warmup_enabled": self.cfg.strategy.warmup_enabled,
                "ml_enabled": self.cfg.runtime.ml_enabled,
                "ml_provider": self.cfg.runtime.ml_provider,
                "ml_fail_open": self.cfg.runtime.ml_fail_open,
                "risk_per_trade_pct": self.cfg.risk.risk_per_trade_pct,
                "account_equity": self.cfg.risk.account_equity,
                "min_conf_range": self.cfg.strategy.min_confidence_range,
                "min_conf_trend": self.cfg.strategy.min_confidence_trend,
                "min_signal_score": self.cfg.strategy.min_signal_score,
                "maker_fee_pct": self.cfg.strategy.maker_fee_pct,
                "taker_fee_pct": self.cfg.strategy.taker_fee_pct,
                "paper_entry_slippage_bps": self.cfg.strategy.paper_entry_slippage_bps,
                "paper_exit_slippage_bps": self.cfg.strategy.paper_exit_slippage_bps,
                "paper_tp1_is_taker": self.cfg.strategy.paper_tp1_is_taker,
                "allowed_coins": sorted(self.cfg.strategy.allowed_coins),
                "allowed_level_labels": sorted(self.cfg.strategy.allowed_level_labels),
                "allowed_coin_level_pairs": sorted(self.cfg.strategy.allowed_coin_level_pairs),
                "allowed_coin_session_pairs": sorted(self.cfg.strategy.allowed_coin_session_pairs),
                "allowed_coin_session_level_triples": sorted(self.cfg.strategy.allowed_coin_session_level_triples),
                "allowed_sessions": sorted(self.cfg.strategy.allowed_sessions),
                "allowed_sides": sorted(self.cfg.strategy.allowed_sides),
                "blocked_coins": sorted(self.cfg.strategy.blocked_coins),
                "blocked_level_labels": sorted(self.cfg.strategy.blocked_level_labels),
                "blocked_coin_level_pairs": sorted(self.cfg.strategy.blocked_coin_level_pairs),
                "blocked_coin_session_pairs": sorted(self.cfg.strategy.blocked_coin_session_pairs),
                "blocked_coin_session_level_triples": sorted(self.cfg.strategy.blocked_coin_session_level_triples),
                "blocked_sessions": sorted(self.cfg.strategy.blocked_sessions),
                "blocked_sides": sorted(self.cfg.strategy.blocked_sides),
                "restored_trades": self._resolved_trades,
            },
        )

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

    def _operator_blocklist_check(
        self,
        coin: str,
        signal: SweepSignal,
        session: str,
    ) -> tuple[str, str] | None:
        coin_key = coin.strip().upper()
        allowed_coins = self.cfg.strategy.allowed_coins
        if allowed_coins and not _matches_block_pattern(coin_key, allowed_coins):
            return "allow_coin_miss", f"allow_coin_miss:{coin_key}"

        blocked_coins = self.cfg.strategy.blocked_coins
        if _matches_block_pattern(coin_key, blocked_coins):
            return "block_coin", f"block_coin:{coin_key}"

        level_key = signal.level_label.strip().lower()
        allowed_levels = self.cfg.strategy.allowed_level_labels
        if allowed_levels and not _matches_block_pattern(level_key, allowed_levels):
            return "allow_level_miss", f"allow_level_miss:{level_key}"

        blocked_levels = self.cfg.strategy.blocked_level_labels
        if _matches_block_pattern(level_key, blocked_levels):
            return "block_level", f"block_level:{level_key}"

        allowed_coin_levels = self.cfg.strategy.allowed_coin_level_pairs
        if allowed_coin_levels and not _matches_coin_level_pair(coin_key, level_key, allowed_coin_levels):
            return "allow_coin_level_miss", f"allow_coin_level_miss:{coin_key}:{level_key}"

        blocked_coin_levels = self.cfg.strategy.blocked_coin_level_pairs
        if _matches_coin_level_pair(coin_key, level_key, blocked_coin_levels):
            return "block_coin_level", f"block_coin_level:{coin_key}:{level_key}"

        session_key = session.strip().lower()
        allowed_coin_sessions = self.cfg.strategy.allowed_coin_session_pairs
        if allowed_coin_sessions and not _matches_coin_session_pair(coin_key, session_key, allowed_coin_sessions):
            return "allow_coin_session_miss", f"allow_coin_session_miss:{coin_key}:{session_key}"

        blocked_coin_sessions = self.cfg.strategy.blocked_coin_session_pairs
        if _matches_coin_session_pair(coin_key, session_key, blocked_coin_sessions):
            return "block_coin_session", f"block_coin_session:{coin_key}:{session_key}"

        allowed_coin_session_levels = self.cfg.strategy.allowed_coin_session_level_triples
        if allowed_coin_session_levels and not _matches_coin_session_level_triple(
            coin_key, session_key, level_key, allowed_coin_session_levels
        ):
            return "allow_coin_session_level_miss", f"allow_coin_session_level_miss:{coin_key}:{session_key}:{level_key}"

        blocked_coin_session_levels = self.cfg.strategy.blocked_coin_session_level_triples
        if _matches_coin_session_level_triple(coin_key, session_key, level_key, blocked_coin_session_levels):
            return "block_coin_session_level", f"block_coin_session_level:{coin_key}:{session_key}:{level_key}"

        allowed_sessions = self.cfg.strategy.allowed_sessions
        if allowed_sessions and not _matches_block_pattern(session_key, allowed_sessions):
            return "allow_session_miss", f"allow_session_miss:{session_key}"

        blocked_sessions = self.cfg.strategy.blocked_sessions
        if _matches_block_pattern(session_key, blocked_sessions):
            return "block_session", f"block_session:{session_key}"

        side_key = signal.side.value.strip().lower()
        allowed_sides = self.cfg.strategy.allowed_sides
        if allowed_sides and not _matches_block_pattern(side_key, allowed_sides):
            return "allow_side_miss", f"allow_side_miss:{side_key}"

        blocked_sides = self.cfg.strategy.blocked_sides
        if _matches_block_pattern(side_key, blocked_sides):
            return "block_side", f"block_side:{side_key}"

        return None

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
        soft_or = warmup_on and self.cfg.strategy.warmup_micro_or_logic
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
        # 30s grace window — see _build_market_state for rationale.
        if ws_healthy:
            self._ws_last_healthy_ms = now_ms
        recently_healthy = (now_ms - self._ws_last_healthy_ms) <= 30_000
        return MarketState(
            ts_ms=ts_ms,
            ws_healthy=ws_healthy or recently_healthy,
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
        soft_or = warmup_on and self.cfg.strategy.warmup_micro_or_logic
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

    def _maybe_ai_decide(self, w: CoinWorker, bar: Bar) -> None:
        """Poll the AI strategy for this coin if its decision cadence has elapsed."""
        if self.ai_strategy is None:
            return
        now_ms = bar.end_ms
        if not self.ai_strategy.should_decide(w.coin, now_ms):
            return
        # Operator override flag — short-circuits the LLM call when set.
        override = read_override_flag(self.cfg.runtime.runtime_dir)
        if override == "close_all":
            # Force close every open position; don't waste an LLM call.
            self.ai_strategy._last_decision_ms[w.coin] = now_ms
            position = getattr(w.executor, "position", None)
            if position is not None:
                decision_id = self._next_signal_id(bar.end_ms)
                self.journal.write("ai_decision", decision_id, {
                    "ts_ms": bar.end_ms, "coin": w.coin, "action": "close",
                    "reasoning": "operator override: close_all",
                    "confidence": 1.0, "cost_usd": 0.0, "latency_ms": 0,
                    "model": "operator", "skip_reason": "ai_override:close_all",
                    "prompt_version": self.cfg.ai.prompt_version,
                })
                self._ai_force_close(w, bar, decision_id, "operator override: close_all")
            return
        if override in {"pause", "no_new"}:
            # Don't call the LLM at all — but DO keep managing existing
            # positions via the regular tick handler. Just skip AI poll.
            self.ai_strategy._last_decision_ms[w.coin] = now_ms
            return
        result = self.ai_strategy.decide_for_coin(
            w,
            bars=list(w.recent_bars),
            now_ms=now_ms,
            account_equity=self.risk.equity,
            daily_pnl=self.risk.daily_pnl,
            daily_r=self.risk.daily_r,
            recent_outcomes=list(self._ai_recent_outcomes),
            workers_by_coin=dict(self._workers),
        )
        self._handle_ai_decision(w, bar, result)

    def _handle_ai_decision(self, w: CoinWorker, bar: Bar, result: AIDecisionResult) -> None:
        """Journal the AI decision and act on it."""
        # Journal every decision (including hold / skip / error) for offline analysis.
        decision_id = self._next_signal_id(bar.end_ms)
        self.journal.write(
            "ai_decision",
            decision_id,
            {
                "ts_ms": bar.end_ms,
                "coin": w.coin,
                "action": result.action,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "model": result.model,
                "prompt_version": self.cfg.ai.prompt_version,
                "error": result.error,
                "skip_reason": result.skip_reason,
                "budget_spent_24h": round(self.ai_strategy.budget.spent_last_24h(now_ms=bar.end_ms), 6) if self.ai_strategy else 0.0,
            },
        )
        # Persistent memory: record every actionable (non-skip/error) decision.
        if self.ai_memory is not None and result.action in {"open_long", "open_short", "close", "hold"}:
            entry = MemoryEntry(
                decision_id=decision_id,
                ts_ms=bar.end_ms,
                coin=w.coin,
                action=result.action,
                reasoning=result.reasoning,
                confidence=result.confidence,
                stop_price=result.signal.stop_price if result.signal else None,
                tp1_price=result.signal.tp1_price if result.signal else None,
                tp2_price=result.signal.tp2_price if result.signal else None,
                entry_price=result.signal.entry_price if result.signal else None,
            )
            self.ai_memory.record_decision(entry)
        if result.action in {"hold", "skipped", "error"}:
            return

        if result.action == "close":
            self._ai_force_close(w, bar, decision_id, result.reasoning)
            return

        if result.action == "move_stop_to_breakeven":
            self._ai_move_stop_to_be(w, bar, decision_id, result.reasoning)
            return

        if result.action == "modify_stop":
            self._ai_modify_stop(w, bar, decision_id, result)
            return

        if result.action == "scale_out":
            self._ai_scale_out(w, bar, decision_id, result)
            return

        if result.action == "add_to_position":
            self._ai_add_to_position(w, bar, decision_id, result)
            return

        if result.action in {"open_long", "open_short"} and result.signal is not None:
            self._ai_open_position(w, bar, decision_id, result)

    def _ai_force_close(self, w: CoinWorker, bar: Bar, decision_id: str, reason: str) -> None:
        """Force-close an open position. Done by tightening stop_price to the
        current mid so the executor's normal stop logic fires on next tick.
        Idempotent: no-op if there's no position."""
        executor = w.executor
        position = getattr(executor, "position", None)
        if position is None:
            log.info("AI close requested for %s but no open position; skipping", w.coin)
            return
        mid = (w.last_best_bid + w.last_best_ask) / 2.0 if (w.last_best_bid and w.last_best_ask) else bar.close
        # Set stop to current price on the LOSING side so next trade triggers exit.
        # Tiny offset of 1 bps to ensure the comparison fires immediately.
        if position.side == Side.LONG:
            position.stop_price = mid * 1.0001
        else:
            position.stop_price = mid * 0.9999
        # For HL executor, propagate to the resting native stop too.
        if hasattr(executor, "_replace_native_stop"):
            try:
                executor._replace_native_stop()
            except Exception as exc:
                log.warning("AI force-close: native stop replace failed for %s: %s", w.coin, exc)
        self.journal.write(
            "decision",
            decision_id,
            {
                "ts_ms": bar.end_ms,
                "allowed": False,
                "coin": w.coin,
                "reason": "ai_close",
                "ai_reasoning": reason,
            },
        )
        log.info("AI force-close armed for %s (mid=%.6f): %s", w.coin, mid, reason[:120])

    def _ai_open_position(self, w: CoinWorker, bar: Bar, decision_id: str, result: AIDecisionResult) -> None:
        """Open a position from an AI decision. Reuses the existing safety stack:
        operator pause, blocklist, risk governor, executor pre-flight."""
        signal = result.signal
        if signal is None:
            return
        # Correlation gate: most alts are highly correlated, so 4 concurrent
        # longs is stealth 4x leverage in the same direction. Block when
        # same-side concurrent open positions already at the configured cap.
        intended_side = signal.side.value  # "long" or "short"
        same_side_count = sum(
            1 for other_w in self._workers.values()
            if other_w is not w
            and (pos := getattr(other_w.executor, "position", None)) is not None
            and pos.side.value == intended_side
        )
        if same_side_count >= self.cfg.ai.max_concurrent_same_side:
            self.journal.write(
                "decision", decision_id,
                {
                    "ts_ms": bar.end_ms, "allowed": False, "coin": w.coin,
                    "reason": f"ai_correlation_cap:{intended_side}:{same_side_count}",
                    "ai_reasoning": result.reasoning,
                },
            )
            log.info(
                "AI %s blocked: %d %s positions already open (cap=%d)",
                result.action, same_side_count, intended_side,
                self.cfg.ai.max_concurrent_same_side,
            )
            return
        # Reject if already exposed: AI is not allowed to ladder. To switch
        # sides, AI must `close` first; next cycle can re-evaluate.
        if w.executor.has_exposure():
            self.journal.write(
                "decision",
                decision_id,
                {
                    "ts_ms": bar.end_ms,
                    "allowed": False,
                    "coin": w.coin,
                    "reason": "ai_open_blocked_existing_exposure",
                    "ai_reasoning": result.reasoning,
                },
            )
            log.info("AI %s blocked: existing exposure on %s", result.action, w.coin)
            return
        # Honor operator pause (but not the sweep-strategy edge_check pause —
        # that gate doesn't apply to AI signals).
        pause = self._runtime_pause_reason()
        if pause is not None and "edge_check" not in pause:
            self.journal.write(
                "decision",
                decision_id,
                {
                    "ts_ms": bar.end_ms,
                    "allowed": False,
                    "coin": w.coin,
                    "reason": pause,
                    "ai_reasoning": result.reasoning,
                },
            )
            log.info("AI %s blocked by operator pause: %s", result.action, pause)
            return
        # Risk governor (daily loss, cooldowns, etc.)
        state = self._build_market_state_w(w, bar.end_ms)
        risk_check = self.risk.can_open_new_trade(state)
        if not risk_check.allowed:
            self.journal.write(
                "decision",
                decision_id,
                {
                    "ts_ms": bar.end_ms,
                    "allowed": False,
                    "coin": w.coin,
                    "reason": f"risk_governor:{risk_check.reason}",
                    "ai_reasoning": result.reasoning,
                },
            )
            log.info("AI %s blocked by risk governor: %s", result.action, risk_check.reason)
            return
        # Volatility-targeted sizing — when vol is high, scale risk down so
        # dollar PnL variance is roughly constant across regimes. Inputs from
        # the same context object the AI saw, so the AI's risk-talk matches
        # what actually gets sized.
        vol_scale = self._ai_vol_target_scale_for(w, bar.end_ms)
        sizing = self.risk.size_position(
            signal.entry_price,
            signal.stop_price,
            risk_multiplier=self.cfg.ai.risk_multiplier * vol_scale,
        )
        if sizing.qty <= 0:
            self.journal.write(
                "decision",
                decision_id,
                {
                    "ts_ms": bar.end_ms,
                    "allowed": False,
                    "coin": w.coin,
                    "reason": "size_zero",
                    "ai_reasoning": result.reasoning,
                },
            )
            log.info("AI %s blocked: sizer returned 0 qty (risk too small?)", result.action)
            return
        # Submit. Reuses the executor's full safety pre-flight.
        try:
            update = w.executor.submit_entry(
                signal,
                signal_id=decision_id,
                qty=sizing.qty,
                risk_dollars=sizing.risk_dollars,
            )
        except Exception as exc:
            log.warning("AI submit_entry error for %s: %s", w.coin, exc, exc_info=True)
            self.journal.write(
                "decision",
                decision_id,
                {
                    "ts_ms": bar.end_ms,
                    "allowed": False,
                    "coin": w.coin,
                    "reason": f"submit_entry_error:{type(exc).__name__}",
                    "ai_reasoning": result.reasoning,
                },
            )
            return
        self.journal.write(
            "decision",
            decision_id,
            {
                "ts_ms": bar.end_ms,
                "allowed": update.event_type == ExecEventType.ENTRY_PLACED,
                "coin": w.coin,
                "reason": "ai_open",
                "ai_reasoning": result.reasoning,
                "qty": sizing.qty,
                "risk_dollars": sizing.risk_dollars,
            },
        )
        self.journal.write(
            "lifecycle",
            decision_id,
            {
                "ts_ms": update.ts_ms,
                "event": update.event_type.value,
                "message": update.message,
                "coin": w.coin,
            },
        )
        if update.event_type == ExecEventType.ENTRY_PLACED:
            self._entries_placed += 1
            w.entries_placed += 1
        log.info("AI %s on %s: %s", result.action, w.coin, update.message)

    def _ai_vol_target_scale_for(self, w: CoinWorker, now_ms: int) -> float:
        """Compute vol-targeted risk scale for the focus coin.

        Returns 1.0 (no scaling) when vol-targeting is disabled or there's
        insufficient data. Otherwise returns target_vol / actual_vol clamped
        to [min_scale, max_scale]. Higher actual vol -> smaller position.
        """
        cfg_ai = self.cfg.ai
        if not cfg_ai.vol_target_enabled:
            return 1.0
        # Reuse the same calculation as the context builder so the AI's
        # reasoning matches the sizing it actually gets.
        from hliq_bot.ai.context import _realized_vol_bps
        actual = _realized_vol_bps(
            list(w.recent_trade_prices), window_ms=5 * 60 * 1000, now_ms=now_ms,
        )
        if actual <= 0:
            return 1.0
        target = max(1.0, cfg_ai.vol_target_bps)
        # Cap actual at target/max_scale so we don't size to 0 in a flash crash;
        # cap below at target/min_scale so we don't blow up in a flat tape.
        actual_clamped = max(target / cfg_ai.vol_target_max_scale, actual)
        actual_clamped = min(target / cfg_ai.vol_target_min_scale, actual_clamped)
        scale = target / actual_clamped
        return max(cfg_ai.vol_target_min_scale, min(cfg_ai.vol_target_max_scale, scale))

    def _ai_move_stop_to_be(self, w: CoinWorker, bar: Bar, decision_id: str, reason: str) -> None:
        position = getattr(w.executor, "position", None)
        if position is None:
            return
        old = position.stop_price
        position.stop_price = position.entry_price
        if hasattr(w.executor, "_replace_native_stop"):
            try:
                w.executor._replace_native_stop()
            except Exception as exc:
                log.warning("AI move_stop_to_be: native stop replace failed: %s", exc)
        self.journal.write(
            "decision", decision_id,
            {
                "ts_ms": bar.end_ms, "allowed": True, "coin": w.coin,
                "reason": "ai_move_stop_to_be",
                "ai_reasoning": reason,
                "old_stop": old, "new_stop": position.stop_price,
            },
        )
        log.info("AI move_stop_to_be %s: %.6f -> %.6f", w.coin, old, position.stop_price)

    def _ai_modify_stop(self, w: CoinWorker, bar: Bar, decision_id: str, result: AIDecisionResult) -> None:
        position = getattr(w.executor, "position", None)
        if position is None or result.new_stop_price is None:
            return
        old = position.stop_price
        new = result.new_stop_price
        # Validate strict improvement: never widen the stop.
        if position.side == Side.LONG and new <= old:
            log.info("AI modify_stop %s rejected: new=%.6f not strictly above old=%.6f (long)", w.coin, new, old)
            return
        if position.side == Side.SHORT and new >= old:
            log.info("AI modify_stop %s rejected: new=%.6f not strictly below old=%.6f (short)", w.coin, new, old)
            return
        # Validate new stop is still on the LOSING side of entry.
        if position.side == Side.LONG and new >= position.entry_price:
            # Allow if entry-or-better (move to BE+) but only if currently profitable
            mid = (w.last_best_bid + w.last_best_ask) / 2.0 if (w.last_best_bid and w.last_best_ask) else bar.close
            if mid <= position.entry_price:
                log.info("AI modify_stop %s rejected: new>=entry but not yet profitable", w.coin)
                return
        if position.side == Side.SHORT and new <= position.entry_price:
            mid = (w.last_best_bid + w.last_best_ask) / 2.0 if (w.last_best_bid and w.last_best_ask) else bar.close
            if mid >= position.entry_price:
                log.info("AI modify_stop %s rejected: new<=entry but not yet profitable", w.coin)
                return
        position.stop_price = new
        if hasattr(w.executor, "_replace_native_stop"):
            try:
                w.executor._replace_native_stop()
            except Exception as exc:
                log.warning("AI modify_stop: native stop replace failed: %s", exc)
        self.journal.write(
            "decision", decision_id,
            {
                "ts_ms": bar.end_ms, "allowed": True, "coin": w.coin,
                "reason": "ai_modify_stop",
                "ai_reasoning": result.reasoning,
                "old_stop": old, "new_stop": new,
            },
        )
        log.info("AI modify_stop %s: %.6f -> %.6f", w.coin, old, new)

    def _ai_scale_out(self, w: CoinWorker, bar: Bar, decision_id: str, result: AIDecisionResult) -> None:
        position = getattr(w.executor, "position", None)
        if position is None or result.scale_fraction is None:
            return
        if position.qty_remaining <= 0:
            return
        # For HL executor, use the existing partial-close machinery.
        # For paper executor, mutate qty + realize fees/pnl ourselves
        # (simpler than adding a partial-close to PaperOrderManager).
        if hasattr(w.executor, "_partial_close_tp1"):
            # HL: temporarily override the partial fraction by faking TP1 logic.
            # Easier approach: call market_close at scale_fraction directly.
            mid = (w.last_best_bid + w.last_best_ask) / 2.0 if (w.last_best_bid and w.last_best_ask) else bar.close
            self._ai_hl_scale_out(w, bar, decision_id, result, mid)
        else:
            self._ai_paper_scale_out(w, bar, decision_id, result)

    def _ai_hl_scale_out(self, w: CoinWorker, bar: Bar, decision_id: str, result: AIDecisionResult, mid: float) -> None:
        """HL scale-out: market_close a fraction of remaining qty."""
        position = w.executor.position
        partial_qty = position.qty_remaining * result.scale_fraction
        if partial_qty <= 0:
            return
        try:
            exchange = w.executor._ensure_exchange()
            res = exchange.market_close(coin=w.executor.coin, sz=partial_qty, slippage=0.005)
        except Exception as exc:
            log.warning("AI scale_out market_close failed for %s: %s", w.coin, exc)
            return
        # Best-effort fill extraction; if no fill, log and skip mutation.
        extracted = w.executor._extract_fill(res, default_px=mid) if res is not None else None
        if extracted is None:
            log.warning("AI scale_out %s: no fill in response; skipping local mutation", w.coin)
            return
        actual_qty, fill_px = extracted
        if position.side == Side.LONG:
            partial_pnl = (fill_px - position.entry_price) * actual_qty
        else:
            partial_pnl = (position.entry_price - fill_px) * actual_qty
        partial_fee = (fill_px * actual_qty) * w.executor.cfg.taker_fee_pct
        position.realized_pnl += partial_pnl
        position.realized_fees += partial_fee
        position.qty_remaining -= actual_qty
        if hasattr(w.executor, "_replace_native_stop"):
            try:
                w.executor._replace_native_stop()
            except Exception:
                pass
        self.journal.write(
            "decision", decision_id,
            {
                "ts_ms": bar.end_ms, "allowed": True, "coin": w.coin,
                "reason": "ai_scale_out",
                "ai_reasoning": result.reasoning,
                "scale_fraction": result.scale_fraction,
                "filled_qty": actual_qty, "fill_px": fill_px,
                "partial_pnl": partial_pnl,
            },
        )
        log.info("AI scale_out %s: %.6f @ %.6f pnl=%.4f", w.coin, actual_qty, fill_px, partial_pnl)

    def _ai_paper_scale_out(self, w: CoinWorker, bar: Bar, decision_id: str, result: AIDecisionResult) -> None:
        """Paper scale-out: mutate qty + realize PnL at current mid."""
        position = w.executor.position
        partial_qty = position.qty_remaining * result.scale_fraction
        if partial_qty <= 0:
            return
        mid = (w.last_best_bid + w.last_best_ask) / 2.0 if (w.last_best_bid and w.last_best_ask) else bar.close
        if position.side == Side.LONG:
            partial_pnl = (mid - position.entry_price) * partial_qty
        else:
            partial_pnl = (position.entry_price - mid) * partial_qty
        partial_fee = (mid * partial_qty) * self.cfg.strategy.taker_fee_pct
        position.realized_pnl += partial_pnl
        position.realized_fees += partial_fee
        position.qty_remaining -= partial_qty
        self.journal.write(
            "decision", decision_id,
            {
                "ts_ms": bar.end_ms, "allowed": True, "coin": w.coin,
                "reason": "ai_scale_out_paper",
                "ai_reasoning": result.reasoning,
                "scale_fraction": result.scale_fraction,
                "filled_qty": partial_qty, "fill_px": mid,
                "partial_pnl": partial_pnl,
            },
        )
        log.info("AI paper scale_out %s: %.6f @ %.6f pnl=%.4f", w.coin, partial_qty, mid, partial_pnl)

    def _ai_add_to_position(self, w: CoinWorker, bar: Bar, decision_id: str, result: AIDecisionResult) -> None:
        """Add to a winning position. Only allowed when current unrealized R >= 0.5."""
        position = getattr(w.executor, "position", None)
        if position is None or result.signal is None or result.add_qty_fraction is None:
            return
        mid = (w.last_best_bid + w.last_best_ask) / 2.0 if (w.last_best_bid and w.last_best_ask) else bar.close
        if position.side == Side.LONG:
            unrealized = (mid - position.entry_price) * position.qty_remaining
        else:
            unrealized = (position.entry_price - mid) * position.qty_remaining
        if position.risk_dollars > 0:
            unrealized_r = unrealized / position.risk_dollars
            if unrealized_r < 0.5:
                self.journal.write(
                    "decision", decision_id,
                    {
                        "ts_ms": bar.end_ms, "allowed": False, "coin": w.coin,
                        "reason": f"ai_add_blocked_unrealized_r:{unrealized_r:.2f}",
                        "ai_reasoning": result.reasoning,
                    },
                )
                log.info("AI add_to_position %s blocked: unrealized_r=%.2f < 0.5", w.coin, unrealized_r)
                return
        # Sized as a fresh entry at reduced risk via add_qty_fraction.
        sizing = self.risk.size_position(
            result.signal.entry_price,
            result.signal.stop_price,
            risk_multiplier=self.cfg.ai.risk_multiplier * result.add_qty_fraction,
        )
        if sizing.qty <= 0:
            return
        # NOTE: HL/paper executors don't currently support "add to existing
        # position" cleanly — submitting a fresh order while a position is
        # open would create a second leg. For safety, log + skip until we add
        # explicit add-on support to the executor.
        self.journal.write(
            "decision", decision_id,
            {
                "ts_ms": bar.end_ms, "allowed": False, "coin": w.coin,
                "reason": "ai_add_not_implemented",
                "ai_reasoning": result.reasoning,
                "would_add_qty": sizing.qty,
            },
        )
        log.warning(
            "AI add_to_position requested for %s (qty=%.8f) — executor doesn't yet support "
            "add-ons; logging intent only. Add ExecutorAddOn API to enable.",
            w.coin, sizing.qty,
        )

    def _maybe_refresh_deadmans(self, now_ms: int) -> None:
        """Push HL's schedule_cancel timer forward when the per-executor refresh
        cadence is due. Called per market-tick (NOT per-heartbeat) so the actual
        SDK push fires at the configured deadman_refresh_sec interval with a
        safety margin against timing jitter from the 60s heartbeat."""
        for w in self._workers.values():
            has_exposure = getattr(w.executor, "has_exposure", None)
            if callable(has_exposure) and not has_exposure():
                continue
            if hasattr(w.executor, "should_refresh_deadman") and w.executor.should_refresh_deadman(now_ms):
                w.executor.refresh_deadman(now_ms)

    def runtime_summary(self) -> dict[str, str | float | int | dict[str, int]]:
        return {
            "run_id": self.run_id,
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
