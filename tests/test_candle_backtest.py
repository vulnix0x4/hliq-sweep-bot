from types import SimpleNamespace

import pytest

import scripts.candle_backtest as candle_backtest
from hliq_bot.config import StrategyConfig
from hliq_bot.models import Bar, Side


def _trade(r, session="us", side="long", family="prior_15m_low"):
    return {"r": r, "session": session, "side": side, "family": family}


def test_load_env_respects_existing_shell_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("BOT_MIN_SIGNAL_SCORE=0.50\nBOT_MIN_CONF_RANGE=0.65\n", encoding="utf-8")
    monkeypatch.setenv("BOT_MIN_SIGNAL_SCORE", "0.60")
    monkeypatch.delenv("BOT_MIN_CONF_RANGE", raising=False)

    candle_backtest._load_env(env_file)

    assert candle_backtest.os.environ["BOT_MIN_SIGNAL_SCORE"] == "0.60"
    assert candle_backtest.os.environ["BOT_MIN_CONF_RANGE"] == "0.65"


def test_candle_fetch_retries_rate_limit(monkeypatch):
    calls = []

    class RateLimitError(Exception):
        status_code = 429

    class FakeInfo:
        def candles_snapshot(self, coin, interval, start_ms, end_ms):
            calls.append((coin, interval, start_ms, end_ms))
            if len(calls) == 1:
                raise RateLimitError()
            return [{"t": start_ms, "T": end_ms, "o": "1", "h": "1", "l": "1", "c": "1"}]

    monkeypatch.setattr(candle_backtest.time, "sleep", lambda _seconds: None)

    rows = candle_backtest._candles_snapshot_with_retry(FakeInfo(), "BTC", 100, 200)

    assert rows == [{"t": 100, "T": 200, "o": "1", "h": "1", "l": "1", "c": "1"}]
    assert len(calls) == 2


def test_candle_fetch_retries_sdk_tuple_rate_limit(monkeypatch):
    calls = []

    class SdkClientError(Exception):
        pass

    class FakeInfo:
        def candles_snapshot(self, coin, interval, start_ms, end_ms):
            calls.append((coin, interval, start_ms, end_ms))
            if len(calls) == 1:
                raise SdkClientError((429, None, "null", None, {}))
            return [{"t": start_ms, "T": end_ms, "o": "1", "h": "1", "l": "1", "c": "1"}]

    monkeypatch.setattr(candle_backtest.time, "sleep", lambda _seconds: None)

    rows = candle_backtest._candles_snapshot_with_retry(FakeInfo(), "BTC", 100, 200)

    assert rows == [{"t": 100, "T": 200, "o": "1", "h": "1", "l": "1", "c": "1"}]
    assert len(calls) == 2


def test_candle_fetch_does_not_retry_non_retryable_errors(monkeypatch):
    calls = []

    class BadRequestError(Exception):
        status_code = 400

    class FakeInfo:
        def candles_snapshot(self, coin, interval, start_ms, end_ms):
            calls.append((coin, interval, start_ms, end_ms))
            raise BadRequestError()

    monkeypatch.setattr(candle_backtest.time, "sleep", lambda _seconds: None)

    with pytest.raises(BadRequestError):
        candle_backtest._candles_snapshot_with_retry(FakeInfo(), "BTC", 100, 200)

    assert len(calls) == 1


def test_current_session_recommendation_uses_only_current_session_trades():
    trades = [
        _trade(-1.0, session="asia"),
        _trade(-1.0, session="asia"),
        _trade(1.4, session="us"),
        _trade(1.2, session="us"),
        _trade(1.1, session="us"),
    ]

    unrestricted = candle_backtest._best_recommendation("BTC", trades, 3)
    current_session = candle_backtest._best_recommendation(
        "BTC",
        trades,
        2,
        current_session="asia",
    )

    assert unrestricted is not None
    assert unrestricted[5] == ("us",)
    assert current_session is None


def test_current_session_recommendation_returns_session_pure_candidate():
    trades = [
        _trade(1.2, session="asia", family="equal_low_*"),
        _trade(0.8, session="asia", family="equal_low_*"),
        _trade(-1.0, session="us", family="prior_15m_low"),
        _trade(1.8, session="us", family="prior_15m_low"),
    ]

    rec = candle_backtest._best_recommendation(
        "BTC",
        trades,
        2,
        current_session="asia",
    )

    assert rec is not None
    assert rec[0] == "BTC"
    assert rec[1] == 2
    assert rec[5] == ("asia",)
    assert rec[7] == ("equal_low_*",)


def test_policy_search_can_be_filtered_to_current_session_rows():
    trades = [
        _trade(1.2, session="asia", family="equal_low_*"),
        _trade(0.8, session="asia", family="equal_low_*"),
        _trade(1.6, session="us", family="prior_15m_low"),
        _trade(1.5, session="us", family="prior_15m_low"),
        _trade(1.4, session="us", family="prior_15m_low"),
    ]

    ranked = candle_backtest._rank_policy_subsets(
        [row for row in trades if row["session"] == "asia"],
        2,
    )

    assert ranked
    assert all(row[4] == ("asia",) for row in ranked)


def test_recommendation_rejects_under_sampled_coin():
    trades = [
        _trade(1.2, session="asia", family="equal_low_*"),
        _trade(0.8, session="asia", family="equal_low_*"),
        _trade(0.7, session="asia", family="prior_15m_low"),
        _trade(0.6, session="asia", family="prior_15m_low"),
    ]

    rec = candle_backtest._best_recommendation(
        "DOGE",
        trades,
        3,
        current_session="asia",
        min_coin_trades=5,
        min_session_trades=2,
        min_level_trades=2,
    )

    assert rec is None


def test_recommendation_rejects_weak_level_family_inside_candidate():
    trades = [
        _trade(1.6, session="asia", family="prior_15m_low"),
        _trade(1.4, session="asia", family="prior_15m_low"),
        _trade(1.3, session="asia", family="prior_15m_low"),
        _trade(-0.4, session="asia", family="equal_low_*"),
        _trade(-0.2, session="asia", family="equal_low_*"),
        _trade(1.8, session="asia", family="vwap_daily"),
        _trade(1.7, session="asia", family="vwap_daily"),
    ]

    rec = candle_backtest._best_recommendation(
        "DOGE",
        trades,
        3,
        current_session="asia",
        min_coin_trades=5,
        min_session_trades=2,
        min_level_trades=2,
    )

    assert rec is not None
    assert rec[7] == ("prior_15m_low", "vwap_daily")


def test_simulate_trade_subtracts_configured_fee_drag_from_r():
    # Simulator now starts at fill_index+1 (can't know intra-bar trajectory on
    # the bar that filled the entry) and books TP2 as a market_close (taker)
    # not maker, to match live execution via HyperliquidOrderManager.
    cfg = SimpleNamespace(strategy=StrategyConfig())
    signal = SimpleNamespace(
        side=Side.LONG,
        entry_price=100.0,
        stop_price=99.0,
        tp1_price=101.0,
        tp2_price=102.0,
    )
    bars = [
        Bar(0, 60_000, 100.0, 100.0, 100.0, 100.0, 1.0, 1, 100.0, 0.1),       # signal bar
        Bar(60_000, 120_000, 100.0, 100.0, 100.0, 100.0, 1.0, 1, 100.0, 0.1), # entry fills here
        # tp1+tp2 hit; bar.low=100.5 stays above entry_fill=100 so the new
        # post-TP1 BE stop re-check doesn't fire.
        Bar(120_000, 180_000, 100.5, 102.0, 100.5, 102.0, 1.0, 1, 101.0, 0.1),
    ]

    filled, r_value, outcome = candle_backtest._simulate_trade(cfg, bars, 0, signal)

    slip = cfg.strategy.paper_exit_slippage_bps / 10_000.0
    tp1_fill = 101.0 * (1.0 - slip)
    tp2_fill = 102.0 * (1.0 - slip)  # taker now → slippage applies
    gross_r = 0.5 * (tp1_fill - 100.0) + 0.5 * (tp2_fill - 100.0)
    fee_r = (
        100.0 * cfg.strategy.maker_fee_pct           # entry maker
        + 0.5 * tp1_fill * cfg.strategy.taker_fee_pct  # tp1 partial taker
        + 0.5 * tp2_fill * cfg.strategy.taker_fee_pct  # tp2 final taker (was maker)
    )
    assert filled is True
    assert outcome == "tp2"
    assert r_value == pytest.approx(gross_r - fee_r)
