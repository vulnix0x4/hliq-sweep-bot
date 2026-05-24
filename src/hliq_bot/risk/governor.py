from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from collections import defaultdict

from hliq_bot.config import RiskConfig, StrategyConfig
from hliq_bot.models import ClosedTrade, MarketState, PositionSize, RiskCheck, Side


@dataclass(slots=True)
class _DailyStats:
    day_key: str
    pnl: float = 0.0
    r_sum: float = 0.0
    trades: int = 0


class RiskGovernor:
    def __init__(self, risk_cfg: RiskConfig, strategy_cfg: StrategyConfig) -> None:
        self.risk_cfg = risk_cfg
        self.strategy_cfg = strategy_cfg
        self._equity = risk_cfg.account_equity
        day_key = self._day_key_from_ms(int(datetime.now(tz=timezone.utc).timestamp() * 1000))
        self._daily = _DailyStats(day_key=day_key)
        self._recent_r = deque(maxlen=max(5, risk_cfg.perf_window_trades))
        self._recent_r_long = deque(maxlen=max(5, risk_cfg.perf_window_trades))
        self._recent_r_short = deque(maxlen=max(5, risk_cfg.perf_window_trades))
        self._recent_r_by_session: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(5, risk_cfg.perf_window_trades))
        )
        self._recent_r_by_level: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(5, risk_cfg.perf_window_trades))
        )
        self._last_loss_ts_ms = 0
        self._last_hard_loss_ts_ms = 0
        self._last_hard_loss_side: dict[Side, int] = {}
        self._last_hard_loss_level: dict[str, int] = {}
        self._side_edge_pause_until_ms: dict[Side, int] = {}

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def daily_r(self) -> float:
        return self._daily.r_sum

    @property
    def daily_pnl(self) -> float:
        return self._daily.pnl

    def size_position(self, entry_price: float, stop_price: float, risk_multiplier: float = 1.0) -> PositionSize:
        stop_distance_abs = abs(entry_price - stop_price)
        if entry_price <= 0 or stop_distance_abs <= 0:
            return PositionSize(qty=0.0, notional=0.0, risk_dollars=0.0, stop_distance_abs=stop_distance_abs)

        risk_multiplier = max(self.risk_cfg.risk_mult_min, min(self.risk_cfg.risk_mult_max, risk_multiplier))
        risk_dollars = self._equity * (self.risk_cfg.risk_per_trade_pct / 100.0) * risk_multiplier
        raw_qty = risk_dollars / stop_distance_abs
        raw_notional = raw_qty * entry_price

        max_notional = self._equity * self.risk_cfg.max_leverage
        if raw_notional > max_notional:
            raw_notional = max_notional
            raw_qty = raw_notional / entry_price
        effective_risk_dollars = raw_qty * stop_distance_abs

        if raw_qty < self.risk_cfg.min_qty:
            return PositionSize(qty=0.0, notional=0.0, risk_dollars=0.0, stop_distance_abs=stop_distance_abs)

        return PositionSize(
            qty=raw_qty,
            notional=raw_notional,
            risk_dollars=effective_risk_dollars,
            stop_distance_abs=stop_distance_abs,
        )

    def can_open_new_trade(self, state: MarketState) -> RiskCheck:
        self._roll_day(state.ts_ms)

        if self._daily.r_sum <= -abs(self.risk_cfg.daily_loss_limit_r):
            return RiskCheck(False, "daily loss limit reached")
        if self._in_hard_loss_cooldown(state.ts_ms):
            return RiskCheck(False, "hard loss cooldown")
        if self._in_loss_cooldown(state.ts_ms):
            return RiskCheck(False, "loss cooldown")
        avg_r = self._avg_recent_r()
        if len(self._recent_r) >= max(1, self.risk_cfg.edge_pause_min_trades) and avg_r <= self.risk_cfg.edge_pause_avg_r:
            return RiskCheck(False, f"edge pause avg_r={avg_r:.2f}")
        if not state.ws_healthy:
            return RiskCheck(False, "ws unhealthy")
        if state.data_stale:
            return RiskCheck(False, "data stale")
        if state.spread_bps > self.strategy_cfg.max_spread_bps:
            return RiskCheck(False, "spread too wide")
        if abs(state.move_30s_pct) >= self.strategy_cfg.news_spike_30s_pct:
            return RiskCheck(False, "news spike circuit breaker")
        ranges = state.recent_bar_ranges_pct[-self.strategy_cfg.circuit_range_bars :]
        if len(ranges) >= self.strategy_cfg.circuit_range_bars and all(
            r >= self.strategy_cfg.max_bar_range_pct for r in ranges
        ):
            return RiskCheck(False, "volatility circuit breaker")
        return RiskCheck(True, "ok")

    def register_closed_trade(self, trade: ClosedTrade, session: str = "", level_label: str = "") -> None:
        self._roll_day(trade.closed_ms)
        self._equity += trade.pnl
        self._daily.pnl += trade.pnl
        self._daily.r_sum += trade.r_multiple
        self._daily.trades += 1
        self._recent_r.append(trade.r_multiple)
        if trade.side == Side.LONG:
            self._recent_r_long.append(trade.r_multiple)
        else:
            self._recent_r_short.append(trade.r_multiple)
        self._side_edge_pause_until_ms.pop(trade.side, None)
        session = (session or "").strip().lower()
        if session:
            self._recent_r_by_session[session].append(trade.r_multiple)
        level_label = (level_label or "").strip().lower()
        if level_label:
            self._recent_r_by_level[level_label].append(trade.r_multiple)
        if trade.pnl < 0 or trade.r_multiple < 0:
            self._last_loss_ts_ms = trade.closed_ms
        if trade.r_multiple <= self.risk_cfg.hard_loss_r:
            self._last_hard_loss_ts_ms = trade.closed_ms
        if trade.r_multiple <= self.risk_cfg.side_hard_loss_r:
            self._last_hard_loss_side[trade.side] = trade.closed_ms
        if level_label and trade.r_multiple <= self.risk_cfg.level_hard_loss_r:
            self._last_hard_loss_level[level_label] = trade.closed_ms

    def performance_multiplier(self) -> float:
        if len(self._recent_r) < max(1, self.risk_cfg.min_trades_for_perf_scaling):
            return 1.0
        avg_r = self._avg_recent_r()
        if avg_r >= 0.2:
            return 1.12
        if avg_r >= 0.05:
            return 1.05
        if avg_r <= -0.2:
            return 0.6
        if avg_r <= -0.05:
            return 0.8
        return 1.0

    def regime_multiplier(self, regime: str, session: str) -> float:
        regime_mult = {
            "range": 1.05,
            "trend": 0.85,
            "high_vol": 0.0,
            "illiquid": 0.0,
        }.get(regime, 1.0)
        session_mult = self._session_performance_multiplier(session)
        return regime_mult * session_mult

    def _session_performance_multiplier(self, session: str) -> float:
        s = (session or "").strip().lower()
        if not s:
            return 1.0
        recent = self._recent_r_by_session.get(s)
        min_n = max(1, self.risk_cfg.min_trades_for_perf_scaling)
        if recent is None or len(recent) < min_n:
            return 1.0  # neutral until we have data
        avg_r = sum(recent) / len(recent)
        mult = 1.0 + max(-0.3, min(0.3, avg_r * 0.6))
        return max(0.5, min(1.3, mult))

    def can_trade_side(self, side: Side, ts_ms: int | None = None) -> RiskCheck:
        if self._in_side_hard_loss_cooldown(side, ts_ms):
            return RiskCheck(False, f"{side.value}_hard_loss_cooldown")

        now_ms = self._resolve_now_ms(ts_ms)
        pause_until = self._side_edge_pause_until_ms.get(side, 0)
        if pause_until > now_ms:
            return RiskCheck(False, f"{side.value}_edge_pause_cooldown")

        recent = self._recent_r_long if side == Side.LONG else self._recent_r_short
        min_n = max(1, self.risk_cfg.side_edge_pause_min_trades)

        if pause_until > 0 and pause_until <= now_ms:
            # Release stale side pause and require fresh outcomes before re-pausing.
            recent.clear()
            self._side_edge_pause_until_ms.pop(side, None)

        if len(recent) < min_n:
            return RiskCheck(True, "ok")

        avg_r = sum(recent) / len(recent)
        if avg_r <= self.risk_cfg.side_edge_pause_avg_r:
            cooldown_sec = max(0, self.risk_cfg.side_edge_pause_cooldown_sec)
            if cooldown_sec > 0:
                self._side_edge_pause_until_ms[side] = now_ms + (cooldown_sec * 1000)
            return RiskCheck(False, f"{side.value}_edge_pause avg_r={avg_r:.2f}")
        return RiskCheck(True, "ok")

    def can_trade_session(self, session: str) -> RiskCheck:
        s = (session or "").strip().lower()
        if not s:
            return RiskCheck(True, "ok")
        recent = self._recent_r_by_session.get(s)
        min_n = max(1, self.risk_cfg.session_edge_pause_min_trades)
        if recent is None or len(recent) < min_n:
            return RiskCheck(True, "ok")
        avg_r = sum(recent) / len(recent)
        if avg_r <= self.risk_cfg.session_edge_pause_avg_r:
            return RiskCheck(False, f"session_edge_pause:{s} avg_r={avg_r:.2f}")
        return RiskCheck(True, "ok")

    def can_trade_level(self, level_label: str, ts_ms: int | None = None) -> RiskCheck:
        label = (level_label or "").strip().lower()
        if not label:
            return RiskCheck(True, "ok")
        if self._in_level_hard_loss_cooldown(label, ts_ms):
            return RiskCheck(False, f"level_hard_loss_pause:{label}")
        recent = self._recent_r_by_level.get(label)
        min_n = max(1, self.risk_cfg.level_edge_pause_min_trades)
        if recent is None or len(recent) < min_n:
            return RiskCheck(True, "ok")
        avg_r = sum(recent) / len(recent)
        if avg_r <= self.risk_cfg.level_edge_pause_avg_r:
            return RiskCheck(False, f"level_edge_pause:{label} avg_r={avg_r:.2f}")
        return RiskCheck(True, "ok")

    def _avg_recent_r(self) -> float:
        if not self._recent_r:
            return 0.0
        return sum(self._recent_r) / len(self._recent_r)

    def _in_loss_cooldown(self, now_ms: int) -> bool:
        if self.risk_cfg.loss_cooldown_sec <= 0:
            return False
        if self._last_loss_ts_ms <= 0:
            return False
        return (now_ms - self._last_loss_ts_ms) < (self.risk_cfg.loss_cooldown_sec * 1000)

    def _in_hard_loss_cooldown(self, now_ms: int) -> bool:
        if self.risk_cfg.hard_loss_cooldown_sec <= 0:
            return False
        if self._last_hard_loss_ts_ms <= 0:
            return False
        return (now_ms - self._last_hard_loss_ts_ms) < (self.risk_cfg.hard_loss_cooldown_sec * 1000)

    def _in_side_hard_loss_cooldown(self, side: Side, ts_ms: int | None) -> bool:
        cooldown_sec = max(0, self.risk_cfg.side_hard_loss_cooldown_sec)
        if cooldown_sec <= 0:
            return False
        last_ms = self._last_hard_loss_side.get(side, 0)
        if last_ms <= 0:
            return False
        now_ms = self._resolve_now_ms(ts_ms)
        return (now_ms - last_ms) < (cooldown_sec * 1000)

    def _in_level_hard_loss_cooldown(self, level_label: str, ts_ms: int | None) -> bool:
        cooldown_sec = max(0, self.risk_cfg.level_hard_loss_cooldown_sec)
        if cooldown_sec <= 0:
            return False
        last_ms = self._last_hard_loss_level.get(level_label, 0)
        if last_ms <= 0:
            return False
        now_ms = self._resolve_now_ms(ts_ms)
        return (now_ms - last_ms) < (cooldown_sec * 1000)

    def _resolve_now_ms(self, ts_ms: int | None) -> int:
        if ts_ms is not None and ts_ms > 0:
            return ts_ms
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    def _roll_day(self, ts_ms: int) -> None:
        day_key = self._day_key_from_ms(ts_ms)
        if day_key != self._daily.day_key:
            self._daily = _DailyStats(day_key=day_key)
            self._recent_r_by_session.clear()
            self._recent_r_by_level.clear()

    def _day_key_from_ms(self, ts_ms: int) -> str:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
