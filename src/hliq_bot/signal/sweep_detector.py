from __future__ import annotations

from collections import Counter
from collections import deque

from hliq_bot.config import LevelConfig, StrategyConfig
from hliq_bot.models import Bar, Side, SweepSignal
from hliq_bot.signal.levels import derive_levels
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.vwap_tracker import VWAPTracker

_LEVEL_TYPE_WEIGHT: dict[str, float] = {
    "pdh": 0.05,
    "pdl": 0.05,
    "vwap_daily": 0.04,
    "session_open_current": 0.03,
    "session_open_prior": 0.03,
    "prior_session_high": 0.02,
    "prior_session_low": 0.02,
}


def _level_type_bonus(label: str) -> float:
    for prefix, bonus in _LEVEL_TYPE_WEIGHT.items():
        if label == prefix or label.startswith(prefix):
            return bonus
    if label.startswith("round_"):
        return 0.01
    return 0.0


class SweepDetector:
    def __init__(
        self,
        config: StrategyConfig,
        level_config: LevelConfig | None = None,
        session_tracker: SessionTracker | None = None,
        vwap_tracker: VWAPTracker | None = None,
        coin: str = "BTC",
    ) -> None:
        self.cfg = config
        self._level_config = level_config
        self._session_tracker = session_tracker
        self._vwap_tracker = vwap_tracker
        self._coin = coin
        self._bars_1h = max(1, int((60 * 60) / max(1, config.timeframe_sec)))
        bars_15m = max(1, int((15 * 60) / max(1, config.timeframe_sec)))
        self._min_history_bars = max(
            3,
            bars_15m,
        )
        max_bars = max(
            500,
            int((60 * 60) / max(1, config.timeframe_sec)) + config.volume_lookback_bars + 10,
        )
        self._history: deque[Bar] = deque(maxlen=max_bars)
        self._diag_counts: Counter[str] = Counter()
        self._tp1_bps: float = config.tp1_bps
        self._tp2_bps: float = config.tp2_bps

    def on_bar(self, bar: Bar) -> SweepSignal | None:
        history = list(self._history)
        signal = None

        if len(history) < self._min_history_bars:
            self._diag_counts["skip_history"] += 1
        else:
            levels = derive_levels(
                history,
                self.cfg.timeframe_sec,
                self.cfg.equal_level_band_bps,
                level_config=self._level_config,
                session_tracker=self._session_tracker,
                vwap_tracker=self._vwap_tracker,
                current_price=bar.close,
                coin=self._coin,
            )
            avg_vol = self._avg_volume(history)
            self._tp1_bps, self._tp2_bps = self._compute_tp_bps(history)
            if avg_vol <= 0:
                self._diag_counts["skip_avg_volume"] += 1
            else:
                signal = self._find_signal(bar, levels.short_levels, levels.long_levels, avg_vol)
                if signal is None and self._is_trending(history):
                    self._diag_counts["trend_context"] += 1

        self._history.append(bar)
        return signal

    def consume_diagnostics(self, top_n: int = 8) -> list[tuple[str, int]]:
        if not self._diag_counts:
            return []
        out = self._diag_counts.most_common(max(1, top_n))
        self._diag_counts.clear()
        return out

    def _find_signal(
        self,
        bar: Bar,
        short_levels: list[tuple[str, float]],
        long_levels: list[tuple[str, float]],
        avg_vol: float,
    ) -> SweepSignal | None:
        if bar.avg_spread_bps > self.cfg.max_spread_bps:
            self._diag_counts["skip_spread"] += 1
            return None

        short_signal = None if self.cfg.long_only else self._short_signal(bar, short_levels, avg_vol)
        long_signal = self._long_signal(bar, long_levels, avg_vol)
        if short_signal is None and long_signal is None:
            return None
        if short_signal is not None and long_signal is not None:
            if short_signal.signal_score >= long_signal.signal_score:
                self._diag_counts["signal_short"] += 1
                return short_signal
            self._diag_counts["signal_long"] += 1
            return long_signal
        if short_signal is not None:
            self._diag_counts["signal_short"] += 1
            return short_signal
        self._diag_counts["signal_long"] += 1
        return long_signal

    def _short_signal(
        self,
        bar: Bar,
        short_levels: list[tuple[str, float]],
        avg_vol: float,
    ) -> SweepSignal | None:
        best: SweepSignal | None = None
        for label, level in short_levels:
            overshoot_bps = ((bar.high - level) / level) * 10_000.0
            if overshoot_bps < self.cfg.min_sweep_bps or overshoot_bps > self.cfg.max_sweep_bps:
                self._diag_counts["short_reject_sweep"] += 1
                continue
            if bar.close >= level:
                self._diag_counts["short_reject_close"] += 1
                continue
            reclaim_bps = ((level - bar.close) / level) * 10_000.0
            if reclaim_bps < self.cfg.min_reclaim_bps:
                self._diag_counts["short_reject_reclaim"] += 1
                continue
            if not self._volume_spike(bar.volume, avg_vol):
                self._diag_counts["short_reject_volume"] += 1
                continue
            if self._wick_ratio_short(bar) < self.cfg.wick_body_ratio_min:
                self._diag_counts["short_reject_wick"] += 1
                continue
            entry = level * (1 - self.cfg.retest_entry_offset_bps / 10_000.0)
            stop = bar.high * (1 + self.cfg.stop_buffer_bps / 10_000.0)
            tp1 = entry * (1 - self._tp1_bps / 10_000.0)
            tp2 = entry * (1 - self._tp2_bps / 10_000.0)
            stop_dist = abs(entry - stop)
            rr_tp1, rr_tp2 = self._rr_values(entry=entry, stop=stop, tp1=tp1, tp2=tp2)
            if not self._risk_reward_ok(entry=entry, stop=stop, tp1=tp1, tp2=tp2):
                self._diag_counts["short_reject_rr"] += 1
                continue
            wick_ratio = self._wick_ratio_short(bar)
            volume_ratio = bar.volume / max(avg_vol, 1e-9)
            conf = self._confidence(
                wick_ratio=wick_ratio,
                volume_ratio=volume_ratio,
                overshoot_ratio=min(overshoot_bps / max(self.cfg.min_sweep_bps, 1e-9), 3.0),
            )
            score = self._signal_score(
                confidence=conf,
                reclaim_bps=reclaim_bps,
                volume_ratio=volume_ratio,
                wick_ratio=wick_ratio,
                overshoot_bps=overshoot_bps,
                rr_tp1=rr_tp1,
                rr_tp2=rr_tp2,
                stop_distance_abs=stop_dist,
                entry_price=entry,
            )
            bonus = _level_type_bonus(label)
            score = max(0.0, min(1.0, score + bonus))
            signal = SweepSignal(
                side=Side.SHORT,
                level=level,
                level_label=label,
                sweep_extreme=bar.high,
                entry_price=entry,
                stop_price=stop,
                tp1_price=tp1,
                tp2_price=tp2,
                confidence=conf,
                overshoot_bps=overshoot_bps,
                reclaim_bps=reclaim_bps,
                volume_ratio=volume_ratio,
                wick_ratio=wick_ratio,
                signal_score=score,
                reason=f"short sweep+reject @ {label}",
                created_ms=bar.end_ms,
            )
            if best is None or signal.signal_score > best.signal_score:
                best = signal
        return best

    def _long_signal(
        self,
        bar: Bar,
        long_levels: list[tuple[str, float]],
        avg_vol: float,
    ) -> SweepSignal | None:
        best: SweepSignal | None = None
        for label, level in long_levels:
            overshoot_bps = ((level - bar.low) / level) * 10_000.0
            if overshoot_bps < self.cfg.min_sweep_bps or overshoot_bps > self.cfg.max_sweep_bps:
                self._diag_counts["long_reject_sweep"] += 1
                continue
            if bar.close <= level:
                self._diag_counts["long_reject_close"] += 1
                continue
            reclaim_bps = ((bar.close - level) / level) * 10_000.0
            if reclaim_bps < self.cfg.min_reclaim_bps:
                self._diag_counts["long_reject_reclaim"] += 1
                continue
            if not self._volume_spike(bar.volume, avg_vol):
                self._diag_counts["long_reject_volume"] += 1
                continue
            if self._wick_ratio_long(bar) < self.cfg.wick_body_ratio_min:
                self._diag_counts["long_reject_wick"] += 1
                continue
            entry = level * (1 + self.cfg.retest_entry_offset_bps / 10_000.0)
            stop = bar.low * (1 - self.cfg.stop_buffer_bps / 10_000.0)
            tp1 = entry * (1 + self._tp1_bps / 10_000.0)
            tp2 = entry * (1 + self._tp2_bps / 10_000.0)
            stop_dist = abs(entry - stop)
            rr_tp1, rr_tp2 = self._rr_values(entry=entry, stop=stop, tp1=tp1, tp2=tp2)
            if not self._risk_reward_ok(entry=entry, stop=stop, tp1=tp1, tp2=tp2):
                self._diag_counts["long_reject_rr"] += 1
                continue
            wick_ratio = self._wick_ratio_long(bar)
            volume_ratio = bar.volume / max(avg_vol, 1e-9)
            conf = self._confidence(
                wick_ratio=wick_ratio,
                volume_ratio=volume_ratio,
                overshoot_ratio=min(overshoot_bps / max(self.cfg.min_sweep_bps, 1e-9), 3.0),
            )
            score = self._signal_score(
                confidence=conf,
                reclaim_bps=reclaim_bps,
                volume_ratio=volume_ratio,
                wick_ratio=wick_ratio,
                overshoot_bps=overshoot_bps,
                rr_tp1=rr_tp1,
                rr_tp2=rr_tp2,
                stop_distance_abs=stop_dist,
                entry_price=entry,
            )
            bonus = _level_type_bonus(label)
            score = max(0.0, min(1.0, score + bonus))
            signal = SweepSignal(
                side=Side.LONG,
                level=level,
                level_label=label,
                sweep_extreme=bar.low,
                entry_price=entry,
                stop_price=stop,
                tp1_price=tp1,
                tp2_price=tp2,
                confidence=conf,
                overshoot_bps=overshoot_bps,
                reclaim_bps=reclaim_bps,
                volume_ratio=volume_ratio,
                wick_ratio=wick_ratio,
                signal_score=score,
                reason=f"long sweep+reject @ {label}",
                created_ms=bar.end_ms,
            )
            if best is None or signal.signal_score > best.signal_score:
                best = signal
        return best

    def _avg_volume(self, history: list[Bar]) -> float:
        look = history[-self.cfg.volume_lookback_bars :]
        if not look:
            return 0.0
        return sum(b.volume for b in look) / len(look)

    def _atr_bps(self, history: list[Bar]) -> float:
        look = history[-self.cfg.volume_lookback_bars :]
        if not look:
            return 0.0
        # range_pct is in % (e.g., 0.2 means 0.2%), multiply by 100 to get bps
        ranges = [b.range_pct * 100.0 for b in look]
        return sum(ranges) / len(ranges)

    def _compute_tp_bps(self, history: list[Bar]) -> tuple[float, float]:
        if not self.cfg.use_atr_targets:
            return self._tp1_bps, self._tp2_bps
        atr = self._atr_bps(history)
        if atr <= 0:
            return self._tp1_bps, self._tp2_bps
        tp1 = max(self.cfg.min_tp1_bps, atr * self.cfg.tp1_atr_mult)
        tp2 = max(self.cfg.min_tp2_bps, atr * self.cfg.tp2_atr_mult)
        return tp1, tp2

    def _is_trending(self, history: list[Bar]) -> bool:
        look = history[-self.cfg.trend_lookback_bars :]
        if len(look) < 2:
            return False
        move_bps = abs((look[-1].close - look[0].close) / look[0].close) * 10_000.0
        return move_bps >= self.cfg.max_trend_move_bps

    def _volume_spike(self, volume: float, avg_volume: float) -> bool:
        return volume >= avg_volume * self.cfg.volume_spike_mult

    def _wick_ratio_short(self, bar: Bar) -> float:
        body_floor = max((bar.high - bar.low) * 0.01, 1e-9)
        body = max(abs(bar.close - bar.open), body_floor)
        upper_wick = max(0.0, bar.high - max(bar.open, bar.close))
        return min(upper_wick / body, 20.0)

    def _wick_ratio_long(self, bar: Bar) -> float:
        body_floor = max((bar.high - bar.low) * 0.01, 1e-9)
        body = max(abs(bar.close - bar.open), body_floor)
        lower_wick = max(0.0, min(bar.open, bar.close) - bar.low)
        return min(lower_wick / body, 20.0)

    def _confidence(self, wick_ratio: float, volume_ratio: float, overshoot_ratio: float) -> float:
        score = 0.0
        score += min(wick_ratio / max(self.cfg.wick_body_ratio_min, 1e-9), 2.0) * 0.45
        score += min(volume_ratio / max(self.cfg.volume_spike_mult, 1e-9), 2.0) * 0.35
        score += min(overshoot_ratio, 2.0) * 0.2
        return max(0.0, min(score / 2.0, 1.0))

    def _signal_score(
        self,
        confidence: float,
        reclaim_bps: float,
        volume_ratio: float,
        wick_ratio: float,
        overshoot_bps: float,
        rr_tp1: float,
        rr_tp2: float,
        stop_distance_abs: float,
        entry_price: float,
    ) -> float:
        """Rank a sweep signal by the features that empirically predict outcomes.

        Based on Spearman + quartile analysis of 185 historical outcomes:
        - wick_ratio: Q4-Q1 = +0.285R — strongest bar-level predictor (highest weight)
        - reclaim_bps: modestly positive
        - volume_ratio: SWEET-SPOT around 2x; high volume (>5x) LOSES (Q4-Q1 = -0.40R)
        - overshoot_bps: SWEET-SPOT around 6bps; deep overshoots (>12bps) LOSE (Q4-Q1 = -0.45R)
        - confidence: ρ = -0.12 (ANTI-predictive) — dropped from positive contribution
        - rr_tp2:     ρ = -0.305 (MOST anti-predictive) — dropped from positive contribution

        The old formula gave 45% weight to `confidence` and 15% to rr — both anti-predictive.
        This rewrite routes weight to features that actually predict wins.
        """
        # Primary: wick_ratio. Saturate at 6x (empirical strong rejection candle).
        wick_norm = min(wick_ratio / 6.0, 1.0)

        # Secondary: reclaim strength. Saturate at 3x the min_reclaim threshold.
        reclaim_cap = max(self.cfg.min_reclaim_bps * 3.0, 1e-9)
        reclaim_norm = min(reclaim_bps / reclaim_cap, 1.0)

        # Volume sweet-spot: peaks at ~2x avg, degrades either side. Top quartile (>5x) LOSES.
        vol_sweet = max(0.0, 1.0 - abs(volume_ratio - 2.0) / 3.0)

        # Overshoot sweet-spot: best around 6bps. Deep sweeps (Q4) lose.
        overshoot_sweet = max(0.0, 1.0 - abs(overshoot_bps - 6.0) / 10.0)

        score = (
            (wick_norm * 0.45)
            + (reclaim_norm * 0.25)
            + (vol_sweet * 0.15)
            + (overshoot_sweet * 0.15)
        )

        # Explicit penalty for extreme overshoot beyond the sweet-spot floor.
        if overshoot_bps > 12.0:
            score -= 0.15 * min((overshoot_bps - 12.0) / 12.0, 1.0)

        # Explicit penalty for extreme volume (>5x historically LOSES).
        if volume_ratio > 5.0:
            score -= 0.15 * min((volume_ratio - 5.0) / 10.0, 1.0)

        return max(0.0, min(score, 1.0))

    def _risk_reward_ok(self, entry: float, stop: float, tp1: float, tp2: float) -> bool:
        if entry <= 0:
            return False
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return False
        stop_dist_bps = stop_dist / entry * 10_000.0
        if stop_dist_bps < self.cfg.min_stop_distance_bps:
            return False
        if stop_dist_bps > self.cfg.max_stop_distance_bps:
            return False

        rr_tp1, rr_tp2 = self._rr_values(entry=entry, stop=stop, tp1=tp1, tp2=tp2)
        if rr_tp1 < self.cfg.min_rr_tp1:
            return False
        if rr_tp2 < self.cfg.min_rr_tp2:
            return False
        return True

    def _rr_values(self, entry: float, stop: float, tp1: float, tp2: float) -> tuple[float, float]:
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return 0.0, 0.0
        rr_tp1 = abs(tp1 - entry) / stop_dist
        rr_tp2 = abs(tp2 - entry) / stop_dist
        return rr_tp1, rr_tp2
