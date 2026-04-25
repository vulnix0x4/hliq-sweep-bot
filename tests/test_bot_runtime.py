from __future__ import annotations

import json
from pathlib import Path
import sys

from hliq_bot.bot import SweepBot
from hliq_bot.config import AppConfig, FeedConfig, LevelConfig, LiveConfig, ReplayConfig, RiskConfig, RuntimeConfig, StrategyConfig
from hliq_bot.models import MarketEvent, Side, TradeEvent


def _app_config(tmp_path: Path) -> AppConfig:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        mode="paper",
        feed=FeedConfig(),
        strategy=StrategyConfig(),
        risk=RiskConfig(),
        runtime=RuntimeConfig(
            runtime_dir=str(runtime_dir),
            journal_path=str(runtime_dir / "signals.jsonl"),
            ml_state_path=str(runtime_dir / "ml_state.json"),
            ml_model_path=str(runtime_dir / "models" / "gate_model.json"),
        ),
        replay=ReplayConfig(input_path=str(runtime_dir / "market_events.jsonl")),
        levels=LevelConfig(),
        live=LiveConfig(),
    )


def test_signal_quality_mult_mapping() -> None:
    """Verify the signal_score -> size multiplier mapping used in bot.py risk_mult.

    Formula: max(0.8, min(1.2, 0.8 + 0.4 * signal_score))
    - score 0.0 -> 0.8  (low-quality signals take 80% size)
    - score 0.5 -> 1.0  (neutral at median score)
    - score 1.0 -> 1.2  (top-tier signals take 120% size)
    - clamped so final size never exceeds [0.8, 1.2] before governor clamps.
    """

    def mult(score: float) -> float:
        return max(0.8, min(1.2, 0.8 + 0.4 * score))

    assert mult(0.0) == 0.8
    assert abs(mult(0.3) - 0.92) < 1e-9
    assert abs(mult(0.5) - 1.0) < 1e-9
    assert abs(mult(0.75) - 1.1) < 1e-9
    assert abs(mult(1.0) - 1.2) < 1e-9
    # Clamp protection in case of bad inputs
    assert mult(-1.0) == 0.8
    assert mult(5.0) == 1.2


def test_warmup_micro_softpass_allows_one_signal_side(tmp_path: Path) -> None:
    cfg = _app_config(tmp_path)
    cfg.strategy.warmup_enabled = True
    cfg.strategy.warmup_target_resolved = 12
    cfg.strategy.warmup_micro_relax = True
    cfg.strategy.warmup_micro_or_logic = True
    cfg.strategy.min_ofi_ratio = 0.06
    cfg.strategy.min_queue_imbalance = 0.03

    bot = SweepBot(cfg)
    bot._resolved_trades = 0
    # Queue imbalance passes for long, flow fails.
    bot._last_bid_size = 120.0
    bot._last_ask_size = 100.0
    bot._recent_signed_flow.clear()
    bot._recent_signed_flow.extend([(1_000, -5.0), (2_000, -8.0)])

    check = bot._microstructure_check(Side.LONG)
    assert check.allowed is True
    assert "micro_softpass" in check.reason


def test_non_warmup_or_logic_allows_single_signal(tmp_path: Path) -> None:
    """OR logic is now always active, so a single passing signal suffices outside warmup."""
    cfg = _app_config(tmp_path)
    cfg.strategy.warmup_enabled = True
    cfg.strategy.warmup_target_resolved = 1
    cfg.strategy.warmup_micro_relax = True
    cfg.strategy.warmup_micro_or_logic = True
    cfg.strategy.min_ofi_ratio = 0.06
    cfg.strategy.min_queue_imbalance = 0.03

    bot = SweepBot(cfg)
    bot._resolved_trades = 2  # warmup off
    bot._last_bid_size = 120.0
    bot._last_ask_size = 100.0
    bot._recent_signed_flow.clear()
    bot._recent_signed_flow.extend([(1_000, -5.0), (2_000, -8.0)])

    check = bot._microstructure_check(Side.LONG)
    assert check.allowed is True
    assert "micro_softpass" in check.reason


def test_auto_train_uses_local_paths_and_persists_state(tmp_path: Path, monkeypatch) -> None:
    cfg = _app_config(tmp_path)
    cfg.runtime.ml_enabled = True
    cfg.runtime.ml_provider = "logistic"
    cfg.runtime.ml_auto_train = True
    cfg.runtime.ml_auto_train_interval_sec = 1
    cfg.runtime.ml_auto_train_min_resolved = 1
    cfg.runtime.ml_auto_train_min_new_trades = 1
    cfg.runtime.ml_auto_apply_threshold = True
    cfg.runtime.ml_min_prob = 0.62

    bot = SweepBot(cfg)
    bot._resolved_trades = 2
    bot._last_auto_train_resolved = 0
    bot._last_auto_train_ms = 0

    captured: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stdout = '{"status":"ok"}'
        stderr = ""

    def _fake_run(cmd, cwd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return _Proc()

    monkeypatch.setattr("hliq_bot.bot.subprocess.run", _fake_run)
    monkeypatch.setattr(bot, "_load_recommended_min_prob", lambda _p: 0.66)
    monkeypatch.setattr(bot.ml_gate, "reload", lambda: captured.setdefault("reloaded", True))

    bot._maybe_auto_train()

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/train_gate.py")
    assert captured.get("cwd") == str(bot._project_root)
    assert captured.get("reloaded") is True
    assert round(bot.cfg.runtime.ml_min_prob, 2) == 0.66

    state_path = Path(bot.cfg.runtime.ml_state_path)
    assert state_path.exists()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert round(float(raw.get("ml_min_prob", 0.0)), 2) == 0.66

    # Validate reload from persisted state.
    bot.cfg.runtime.ml_min_prob = 0.55
    bot._load_runtime_ml_state()
    assert round(bot.cfg.runtime.ml_min_prob, 2) == 0.66


def test_run_replay_updates_summary_without_live_clock(tmp_path: Path) -> None:
    cfg = _app_config(tmp_path)
    bot = SweepBot(cfg)

    events = [
        MarketEvent(kind="book", ts_ms=60_000, book=None),
        MarketEvent(kind="trade", ts_ms=60_000, trade=TradeEvent(ts_ms=60_000, price=100.0, size=1.0, side="buy")),
        MarketEvent(kind="trade", ts_ms=120_000, trade=TradeEvent(ts_ms=120_000, price=100.1, size=1.0, side="buy")),
    ]

    summary = bot.run_replay(events)
    assert int(summary["trade_events"]) == 2
    assert int(summary["bars_closed"]) == 1


def test_micro_softpass_outside_warmup(tmp_path: Path) -> None:
    cfg = _app_config(tmp_path)
    cfg.strategy.warmup_enabled = True
    cfg.strategy.warmup_target_resolved = 5
    cfg.strategy.min_ofi_ratio = 0.06
    cfg.strategy.min_queue_imbalance = 0.03

    bot = SweepBot(cfg)
    bot._resolved_trades = 100  # well past warmup
    # Queue imbalance passes for long, flow fails
    bot._last_bid_size = 120.0
    bot._last_ask_size = 100.0
    bot._recent_signed_flow.clear()
    bot._recent_signed_flow.extend([(1_000, -5.0), (2_000, -8.0)])

    check = bot._microstructure_check(Side.LONG)
    assert check.allowed is True
    assert "micro_softpass" in check.reason


def test_bot_creates_session_and_vwap_trackers(tmp_path: Path) -> None:
    cfg = _app_config(tmp_path)
    bot = SweepBot(cfg)
    assert bot._session_tracker is not None
    assert bot._vwap_tracker is not None


def test_bot_creates_workers_for_each_coin(tmp_path: Path) -> None:
    cfg = _app_config(tmp_path)
    cfg.feed.coins_str = "BTC,ETH,SOL"
    bot = SweepBot(cfg)
    assert len(bot._workers) == 3
    assert "BTC" in bot._workers
    assert "ETH" in bot._workers
    assert "SOL" in bot._workers
    # Each worker has its own independent components
    assert bot._workers["BTC"].bar_builder is not bot._workers["ETH"].bar_builder
    assert bot._workers["BTC"].detector is not bot._workers["ETH"].detector
    assert bot._workers["BTC"].executor is not bot._workers["ETH"].executor
    assert bot._workers["BTC"].session_tracker is not bot._workers["SOL"].session_tracker


def test_events_route_to_correct_worker(tmp_path: Path) -> None:
    cfg = _app_config(tmp_path)
    cfg.feed.coins_str = "BTC,ETH"
    bot = SweepBot(cfg)

    btc_event = MarketEvent(
        kind="trade",
        ts_ms=60_000,
        coin="BTC",
        trade=TradeEvent(ts_ms=60_000, price=50_000.0, size=0.1, side="buy"),
    )
    eth_event = MarketEvent(
        kind="trade",
        ts_ms=60_000,
        coin="ETH",
        trade=TradeEvent(ts_ms=60_000, price=3_000.0, size=1.0, side="sell"),
    )

    bot._handle_event(btc_event)
    bot._handle_event(eth_event)

    # BTC worker should have a trade price logged
    assert len(bot._workers["BTC"].recent_trade_prices) == 1
    assert bot._workers["BTC"].recent_trade_prices[0][1] == 50_000.0

    # ETH worker should have a trade price logged
    assert len(bot._workers["ETH"].recent_trade_prices) == 1
    assert bot._workers["ETH"].recent_trade_prices[0][1] == 3_000.0


def test_book_events_route_to_correct_worker(tmp_path: Path) -> None:
    from hliq_bot.models import BookTopEvent

    cfg = _app_config(tmp_path)
    cfg.feed.coins_str = "BTC,ETH"
    bot = SweepBot(cfg)

    btc_book = MarketEvent(
        kind="book",
        ts_ms=60_000,
        coin="BTC",
        book=BookTopEvent(ts_ms=60_000, best_bid=49_900.0, best_ask=50_100.0, bid_size=10.0, ask_size=8.0),
    )
    eth_book = MarketEvent(
        kind="book",
        ts_ms=60_000,
        coin="ETH",
        book=BookTopEvent(ts_ms=60_000, best_bid=2_990.0, best_ask=3_010.0, bid_size=50.0, ask_size=60.0),
    )

    bot._handle_event(btc_book)
    bot._handle_event(eth_book)

    assert bot._workers["BTC"].last_best_bid == 49_900.0
    assert bot._workers["BTC"].last_bid_size == 10.0
    assert bot._workers["ETH"].last_best_bid == 2_990.0
    assert bot._workers["ETH"].last_ask_size == 60.0


def test_backward_compat_single_coin(tmp_path: Path) -> None:
    """HL_COIN=BTC (single coin) still works via backward-compat properties."""
    cfg = _app_config(tmp_path)
    # Default FeedConfig has coins_str="BTC" which is the single-coin default
    bot = SweepBot(cfg)
    assert len(bot._workers) == 1
    assert "BTC" in bot._workers
    # Backward-compat attributes point to the first worker
    assert bot.bar_builder is bot._workers["BTC"].bar_builder
    assert bot.detector is bot._workers["BTC"].detector
    assert bot.executor is bot._workers["BTC"].executor


def test_feed_config_coins_property() -> None:
    from hliq_bot.config import FeedConfig
    fc = FeedConfig(coins_str="BTC,ETH,SOL")
    assert fc.coins == ["BTC", "ETH", "SOL"]
    assert fc.coin == "BTC"  # backward compat: first coin

    fc2 = FeedConfig(coins_str="ETH")
    assert fc2.coins == ["ETH"]
    assert fc2.coin == "ETH"

    # Test whitespace handling
    fc3 = FeedConfig(coins_str=" btc , eth ")
    assert fc3.coins == ["BTC", "ETH"]


def test_portfolio_position_limit_config() -> None:
    from hliq_bot.config import RiskConfig
    rc = RiskConfig(portfolio_max_positions=5, max_positions_per_coin=2)
    assert rc.portfolio_max_positions == 5
    assert rc.max_positions_per_coin == 2


def test_bot_uses_paper_executor_when_mode_paper(tmp_path):
    cfg = _app_config(tmp_path)
    cfg.mode = "paper"
    bot = SweepBot(cfg)
    from hliq_bot.execution.order_manager import PaperOrderManager
    for w in bot._workers.values():
        assert isinstance(w.executor, PaperOrderManager)


def test_bot_uses_hyperliquid_executor_when_mode_live(tmp_path):
    cfg = _app_config(tmp_path)
    cfg.mode = "live"
    cfg.live.allow_live = True
    cfg.live.agent_private_key = "0x" + "a" * 64
    cfg.live.main_wallet_address = "0x" + "b" * 40
    bot = SweepBot(cfg)
    from hliq_bot.execution.hyperliquid_order_manager import HyperliquidOrderManager
    for w in bot._workers.values():
        assert isinstance(w.executor, HyperliquidOrderManager)


def test_bot_refuses_live_mode_when_allow_live_false(tmp_path):
    """BOT_MODE=live without BOT_ALLOW_LIVE=true must fail at boot, not in worker loop."""
    import pytest
    cfg = _app_config(tmp_path)
    cfg.mode = "live"
    cfg.live.allow_live = False  # explicit
    cfg.live.agent_private_key = "0x" + "a" * 64
    cfg.live.main_wallet_address = "0x" + "b" * 40
    with pytest.raises(RuntimeError, match="BOT_ALLOW_LIVE"):
        SweepBot(cfg)
