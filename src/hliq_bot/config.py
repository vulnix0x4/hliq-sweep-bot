from __future__ import annotations

from dataclasses import dataclass
import os


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class FeedConfig:
    ws_url: str = "wss://api.hyperliquid.xyz/ws"
    coins_str: str = "BTC"  # comma-separated, e.g. "BTC,ETH,SOL"
    subscribe_user: bool = False
    user_address: str = ""
    reconnect_backoff_sec: float = 2.0
    stale_data_sec: int = 5

    @property
    def coins(self) -> list[str]:
        return [c.strip().upper() for c in self.coins_str.split(",") if c.strip()]

    @property
    def coin(self) -> str:
        """Backward compat: return the first coin."""
        coins = self.coins
        return coins[0] if coins else "BTC"


@dataclass(slots=True)
class StrategyConfig:
    timeframe_sec: int = 60
    min_sweep_bps: float = 4.0
    max_sweep_bps: float = 16.0
    min_reclaim_bps: float = 2.0
    equal_level_band_bps: float = 6.0
    wick_body_ratio_min: float = 1.3
    volume_lookback_bars: int = 20
    volume_spike_mult: float = 1.0
    retest_entry_offset_bps: float = 0.0
    stop_buffer_bps: float = 12.0
    min_stop_distance_bps: float = 6.0
    max_stop_distance_bps: float = 55.0
    tp1_bps: float = 80.0
    tp2_bps: float = 180.0
    min_rr_tp1: float = 0.6
    min_rr_tp2: float = 1.2
    break_even_progress_tp1_frac: float = 0.45
    time_stop_sec: int = 240
    max_holding_sec: int = 1800
    pending_entry_expiry_sec: int = 120
    entry_touch_tolerance_bps: float = 1.0
    max_spread_bps: float = 6.0
    trend_lookback_bars: int = 20
    max_trend_move_bps: float = 120.0
    max_bar_range_pct: float = 0.5
    news_spike_30s_pct: float = 0.5
    circuit_range_bars: int = 3
    min_confidence_range: float = 0.55
    min_confidence_trend: float = 0.72
    disable_in_high_vol: bool = True
    disable_in_illiquid: bool = True
    use_micro_confirm: bool = True
    micro_flow_window_sec: int = 5
    min_ofi_ratio: float = 0.06
    min_queue_imbalance: float = 0.03
    use_funding_blackout: bool = True
    funding_blackout_sec: int = 90
    warmup_enabled: bool = True
    warmup_target_resolved: int = 12
    warmup_micro_relax: bool = True
    warmup_ofi_scale: float = 0.5
    warmup_qimb_scale: float = 0.5
    warmup_micro_or_logic: bool = True
    warmup_risk_mult_cap: float = 0.75
    use_atr_targets: bool = True
    tp1_atr_mult: float = 0.6
    tp2_atr_mult: float = 1.4
    min_tp1_bps: float = 25.0
    min_tp2_bps: float = 50.0
    trail_after_tp1: bool = True
    trail_factor: float = 0.5
    early_exit_sec: int = 120
    early_exit_r_threshold: float = -0.3


@dataclass(slots=True)
class LevelConfig:
    pdh_pdl: bool = True
    session_open: bool = True
    round_numbers: bool = True
    vwap: bool = True
    prior_session: bool = True
    round_number_range_pct: float = 1.5


@dataclass(slots=True)
class RiskConfig:
    account_equity: float = 1000.0
    risk_per_trade_pct: float = 0.75
    daily_loss_limit_r: float = 2.5
    max_leverage: float = 8.0
    min_qty: float = 0.0001
    max_open_positions: int = 1
    max_positions_per_coin: int = 1
    portfolio_max_positions: int = 3
    perf_window_trades: int = 20
    min_trades_for_perf_scaling: int = 8
    risk_mult_min: float = 0.4
    risk_mult_max: float = 1.25
    loss_cooldown_sec: int = 120
    hard_loss_r: float = -0.9
    hard_loss_cooldown_sec: int = 300
    side_hard_loss_r: float = -0.9
    side_hard_loss_cooldown_sec: int = 1800
    level_hard_loss_r: float = -0.9
    level_hard_loss_cooldown_sec: int = 21600
    edge_pause_avg_r: float = -0.25
    edge_pause_min_trades: int = 12
    side_edge_pause_avg_r: float = -0.25
    side_edge_pause_min_trades: int = 3
    side_edge_pause_cooldown_sec: int = 900
    session_edge_pause_avg_r: float = -0.25
    session_edge_pause_min_trades: int = 2
    level_edge_pause_avg_r: float = -0.25
    level_edge_pause_min_trades: int = 2


@dataclass(slots=True)
class RuntimeConfig:
    runtime_dir: str = "runtime"
    journal_path: str = "runtime/signals.jsonl"
    market_capture_enabled: bool = False
    market_capture_path: str = "runtime/market_events.jsonl"
    ml_state_path: str = "runtime/ml_state.json"
    ml_enabled: bool = False
    ml_decision_mode: str = "rank"
    ml_provider: str = "logistic"
    ml_model_path: str = "runtime/models/gate_model.json"
    ml_model_min_samples: int = 20
    ml_min_prob: float = 0.55
    ml_threshold_floor: float = 0.52
    ml_threshold_cap: float = 0.85
    ml_adaptive_threshold: bool = True
    ml_adaptive_window: int = 40
    ml_adaptive_min_trades: int = 12
    ml_adaptive_step: float = 0.03
    ml_ensemble_codex_weight: float = 0.65
    ml_auto_train: bool = True
    ml_auto_train_interval_sec: int = 1800
    ml_auto_train_min_resolved: int = 30
    ml_auto_train_min_new_trades: int = 6
    ml_auto_apply_threshold: bool = True
    ml_fail_open: bool = False
    codex_cmd: str = "codex"
    codex_model: str = "gpt-5.3-codex"
    codex_profile: str = ""
    codex_timeout_sec: float = 20.0
    codex_min_interval_sec: int = 15
    codex_max_calls_hourly: int = 120


@dataclass(slots=True)
class ReplayConfig:
    input_path: str = "runtime/market_events.jsonl"


@dataclass(slots=True)
class AppConfig:
    mode: str
    feed: FeedConfig
    strategy: StrategyConfig
    risk: RiskConfig
    runtime: RuntimeConfig
    replay: ReplayConfig
    levels: LevelConfig


def load_config() -> AppConfig:
    feed = FeedConfig(
        ws_url=_env_str("HL_WS_URL", "wss://api.hyperliquid.xyz/ws"),
        coins_str=_env_str("HL_COINS", _env_str("HL_COIN", "BTC")).upper(),
        subscribe_user=_env_bool("HL_SUBSCRIBE_USER", False),
        user_address=_env_str("HL_USER_ADDRESS", ""),
        reconnect_backoff_sec=_env_float("HL_RECONNECT_BACKOFF_SEC", 2.0),
        stale_data_sec=_env_int("HL_STALE_DATA_SEC", 5),
    )

    strategy = StrategyConfig(
        timeframe_sec=_env_int("BOT_TIMEFRAME_SEC", 60),
        min_sweep_bps=_env_float("BOT_MIN_SWEEP_BPS", 4.0),
        max_sweep_bps=_env_float("BOT_MAX_SWEEP_BPS", 16.0),
        min_reclaim_bps=_env_float("BOT_MIN_RECLAIM_BPS", 2.0),
        equal_level_band_bps=_env_float("BOT_EQUAL_LEVEL_BPS", 6.0),
        wick_body_ratio_min=_env_float("BOT_WICK_BODY_RATIO_MIN", 1.3),
        volume_lookback_bars=_env_int("BOT_VOLUME_LOOKBACK_BARS", 20),
        volume_spike_mult=_env_float("BOT_VOLUME_SPIKE_MULT", 1.0),
        retest_entry_offset_bps=_env_float("BOT_RETEST_OFFSET_BPS", 0.0),
        stop_buffer_bps=_env_float("BOT_STOP_BUFFER_BPS", 12.0),
        min_stop_distance_bps=_env_float("BOT_MIN_STOP_DISTANCE_BPS", 6.0),
        max_stop_distance_bps=_env_float("BOT_MAX_STOP_DISTANCE_BPS", 55.0),
        tp1_bps=_env_float("BOT_TP1_BPS", 80.0),
        tp2_bps=_env_float("BOT_TP2_BPS", 180.0),
        min_rr_tp1=_env_float("BOT_MIN_RR_TP1", 0.6),
        min_rr_tp2=_env_float("BOT_MIN_RR_TP2", 1.2),
        break_even_progress_tp1_frac=_env_float("BOT_BREAK_EVEN_PROGRESS_TP1_FRAC", 0.45),
        time_stop_sec=_env_int("BOT_TIME_STOP_SEC", 240),
        max_holding_sec=_env_int("BOT_MAX_HOLDING_SEC", 1800),
        pending_entry_expiry_sec=_env_int("BOT_ENTRY_EXPIRY_SEC", 120),
        entry_touch_tolerance_bps=_env_float("BOT_ENTRY_TOUCH_TOL_BPS", 1.0),
        max_spread_bps=_env_float("BOT_MAX_SPREAD_BPS", 6.0),
        trend_lookback_bars=_env_int("BOT_TREND_LOOKBACK_BARS", 20),
        max_trend_move_bps=_env_float("BOT_MAX_TREND_MOVE_BPS", 120.0),
        max_bar_range_pct=_env_float("BOT_MAX_BAR_RANGE_PCT", 0.5),
        news_spike_30s_pct=_env_float("BOT_NEWS_SPIKE_30S_PCT", 0.5),
        circuit_range_bars=_env_int("BOT_CIRCUIT_RANGE_BARS", 3),
        min_confidence_range=_env_float("BOT_MIN_CONF_RANGE", 0.55),
        min_confidence_trend=_env_float("BOT_MIN_CONF_TREND", 0.72),
        disable_in_high_vol=_env_bool("BOT_DISABLE_HIGH_VOL", True),
        disable_in_illiquid=_env_bool("BOT_DISABLE_ILLIQUID", True),
        use_micro_confirm=_env_bool("BOT_USE_MICRO_CONFIRM", True),
        micro_flow_window_sec=_env_int("BOT_MICRO_FLOW_WINDOW_SEC", 5),
        min_ofi_ratio=_env_float("BOT_MIN_OFI_RATIO", 0.06),
        min_queue_imbalance=_env_float("BOT_MIN_QUEUE_IMBALANCE", 0.03),
        use_funding_blackout=_env_bool("BOT_USE_FUNDING_BLACKOUT", True),
        funding_blackout_sec=_env_int("BOT_FUNDING_BLACKOUT_SEC", 90),
        warmup_enabled=_env_bool("BOT_WARMUP_ENABLED", True),
        warmup_target_resolved=_env_int("BOT_WARMUP_TARGET_RESOLVED", 12),
        warmup_micro_relax=_env_bool("BOT_WARMUP_MICRO_RELAX", True),
        warmup_ofi_scale=_env_float("BOT_WARMUP_OFI_SCALE", 0.5),
        warmup_qimb_scale=_env_float("BOT_WARMUP_QIMB_SCALE", 0.5),
        warmup_micro_or_logic=_env_bool("BOT_WARMUP_MICRO_OR_LOGIC", True),
        warmup_risk_mult_cap=_env_float("BOT_WARMUP_RISK_MULT_CAP", 0.75),
        use_atr_targets=_env_bool("BOT_USE_ATR_TARGETS", True),
        tp1_atr_mult=_env_float("BOT_TP1_ATR_MULT", 0.6),
        tp2_atr_mult=_env_float("BOT_TP2_ATR_MULT", 1.4),
        min_tp1_bps=_env_float("BOT_MIN_TP1_BPS", 25.0),
        min_tp2_bps=_env_float("BOT_MIN_TP2_BPS", 50.0),
        trail_after_tp1=_env_bool("BOT_TRAIL_AFTER_TP1", True),
        trail_factor=_env_float("BOT_TRAIL_FACTOR", 0.5),
        early_exit_sec=_env_int("BOT_EARLY_EXIT_SEC", 120),
        early_exit_r_threshold=_env_float("BOT_EARLY_EXIT_R_THRESHOLD", -0.3),
    )

    risk = RiskConfig(
        account_equity=_env_float("BOT_ACCOUNT_EQUITY", 1000.0),
        risk_per_trade_pct=_env_float("BOT_RISK_PER_TRADE_PCT", 0.75),
        daily_loss_limit_r=_env_float("BOT_DAILY_LOSS_LIMIT_R", 2.5),
        max_leverage=_env_float("BOT_MAX_LEVERAGE", 8.0),
        min_qty=_env_float("BOT_MIN_QTY", 0.0001),
        max_open_positions=_env_int("BOT_MAX_OPEN_POSITIONS", 1),
        max_positions_per_coin=_env_int("BOT_MAX_POSITIONS_PER_COIN", 1),
        portfolio_max_positions=_env_int("BOT_PORTFOLIO_MAX_POSITIONS", 3),
        perf_window_trades=_env_int("BOT_PERF_WINDOW_TRADES", 20),
        min_trades_for_perf_scaling=_env_int("BOT_MIN_TRADES_FOR_PERF", 8),
        risk_mult_min=_env_float("BOT_RISK_MULT_MIN", 0.4),
        risk_mult_max=_env_float("BOT_RISK_MULT_MAX", 1.25),
        loss_cooldown_sec=_env_int("BOT_LOSS_COOLDOWN_SEC", 120),
        hard_loss_r=_env_float("BOT_HARD_LOSS_R", -0.9),
        hard_loss_cooldown_sec=_env_int("BOT_HARD_LOSS_COOLDOWN_SEC", 300),
        side_hard_loss_r=_env_float("BOT_SIDE_HARD_LOSS_R", -0.9),
        side_hard_loss_cooldown_sec=_env_int("BOT_SIDE_HARD_LOSS_COOLDOWN_SEC", 1800),
        level_hard_loss_r=_env_float("BOT_LEVEL_HARD_LOSS_R", -0.9),
        level_hard_loss_cooldown_sec=_env_int("BOT_LEVEL_HARD_LOSS_COOLDOWN_SEC", 21600),
        edge_pause_avg_r=_env_float("BOT_EDGE_PAUSE_AVG_R", -0.25),
        edge_pause_min_trades=_env_int("BOT_EDGE_PAUSE_MIN_TRADES", 12),
        side_edge_pause_avg_r=_env_float("BOT_SIDE_EDGE_PAUSE_AVG_R", -0.25),
        side_edge_pause_min_trades=_env_int("BOT_SIDE_EDGE_PAUSE_MIN_TRADES", 3),
        side_edge_pause_cooldown_sec=_env_int("BOT_SIDE_EDGE_PAUSE_COOLDOWN_SEC", 900),
        session_edge_pause_avg_r=_env_float("BOT_SESSION_EDGE_PAUSE_AVG_R", -0.25),
        session_edge_pause_min_trades=_env_int("BOT_SESSION_EDGE_PAUSE_MIN_TRADES", 2),
        level_edge_pause_avg_r=_env_float("BOT_LEVEL_EDGE_PAUSE_AVG_R", -0.25),
        level_edge_pause_min_trades=_env_int("BOT_LEVEL_EDGE_PAUSE_MIN_TRADES", 2),
    )

    runtime = RuntimeConfig(
        runtime_dir=_env_str("BOT_RUNTIME_DIR", "runtime"),
        journal_path=_env_str("BOT_JOURNAL_PATH", "runtime/signals.jsonl"),
        market_capture_enabled=_env_bool("BOT_MARKET_CAPTURE_ENABLED", False),
        market_capture_path=_env_str("BOT_MARKET_CAPTURE_PATH", "runtime/market_events.jsonl"),
        ml_state_path=_env_str("BOT_ML_STATE_PATH", "runtime/ml_state.json"),
        ml_enabled=_env_bool("BOT_ML_ENABLED", False),
        ml_decision_mode=_env_str("BOT_ML_DECISION_MODE", "rank").lower(),
        ml_provider=_env_str("BOT_ML_PROVIDER", "logistic").lower(),
        ml_model_path=_env_str("BOT_ML_MODEL_PATH", "runtime/models/gate_model.json"),
        ml_model_min_samples=_env_int("BOT_ML_MODEL_MIN_SAMPLES", 20),
        ml_min_prob=_env_float("BOT_ML_MIN_PROB", 0.55),
        ml_threshold_floor=_env_float("BOT_ML_THRESHOLD_FLOOR", 0.52),
        ml_threshold_cap=_env_float("BOT_ML_THRESHOLD_CAP", 0.85),
        ml_adaptive_threshold=_env_bool("BOT_ML_ADAPTIVE_THRESHOLD", True),
        ml_adaptive_window=_env_int("BOT_ML_ADAPTIVE_WINDOW", 40),
        ml_adaptive_min_trades=_env_int("BOT_ML_ADAPTIVE_MIN_TRADES", 12),
        ml_adaptive_step=_env_float("BOT_ML_ADAPTIVE_STEP", 0.03),
        ml_ensemble_codex_weight=_env_float("BOT_ML_ENSEMBLE_CODEX_WEIGHT", 0.65),
        ml_auto_train=_env_bool("BOT_ML_AUTO_TRAIN", True),
        ml_auto_train_interval_sec=_env_int("BOT_ML_AUTO_TRAIN_INTERVAL_SEC", 1800),
        ml_auto_train_min_resolved=_env_int("BOT_ML_AUTO_TRAIN_MIN_RESOLVED", 30),
        ml_auto_train_min_new_trades=_env_int("BOT_ML_AUTO_TRAIN_MIN_NEW_TRADES", 6),
        ml_auto_apply_threshold=_env_bool("BOT_ML_AUTO_APPLY_THRESHOLD", True),
        ml_fail_open=_env_bool("BOT_ML_FAIL_OPEN", False),
        codex_cmd=_env_str("BOT_CODEX_CMD", "codex"),
        codex_model=_env_str("BOT_CODEX_MODEL", "gpt-5.3-codex"),
        codex_profile=_env_str("BOT_CODEX_PROFILE", ""),
        codex_timeout_sec=_env_float("BOT_CODEX_TIMEOUT_SEC", 20.0),
        codex_min_interval_sec=_env_int("BOT_CODEX_MIN_INTERVAL_SEC", 15),
        codex_max_calls_hourly=_env_int("BOT_CODEX_MAX_CALLS_HOURLY", 120),
    )

    replay = ReplayConfig(
        input_path=_env_str("BOT_REPLAY_INPUT_PATH", "runtime/market_events.jsonl"),
    )

    levels = LevelConfig(
        pdh_pdl=_env_bool("BOT_LEVELS_PDH_PDL", True),
        session_open=_env_bool("BOT_LEVELS_SESSION_OPEN", True),
        round_numbers=_env_bool("BOT_LEVELS_ROUND_NUMBERS", True),
        vwap=_env_bool("BOT_LEVELS_VWAP", True),
        prior_session=_env_bool("BOT_LEVELS_PRIOR_SESSION", True),
        round_number_range_pct=_env_float("BOT_ROUND_NUMBER_RANGE_PCT", 1.5),
    )

    return AppConfig(
        mode=_env_str("BOT_MODE", "paper").lower(),
        feed=feed,
        strategy=strategy,
        risk=risk,
        runtime=runtime,
        replay=replay,
        levels=levels,
    )
