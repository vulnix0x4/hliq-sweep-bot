from types import SimpleNamespace

import scripts.go_live_check as go_live_check
from hliq_bot.config import AppConfig, FeedConfig, LevelConfig, LiveConfig, ReplayConfig, RiskConfig, RuntimeConfig, StrategyConfig


def test_journal_summary_reports_pending_and_open_paper_exposure():
    summary = go_live_check._journal_summary(
        [
            {"event_type": "run", "event": "run_start", "run_id": "r"},
            {"event_type": "lifecycle", "run_id": "r", "event": "entry_placed"},
            {"event_type": "lifecycle", "run_id": "r", "event": "entry_placed"},
            {"event_type": "lifecycle", "run_id": "r", "event": "entry_filled"},
        ],
        run_id="r",
    )

    assert summary["entries_placed"] == 2
    assert summary["entries_filled"] == 1
    assert summary["pending_entries"] == 1
    assert summary["open_paper_positions"] == 1


def test_active_policy_run_ids_match_exact_current_policy():
    cfg = AppConfig(
        mode="paper",
        feed=FeedConfig(coins_str="BTC"),
        strategy=StrategyConfig(
            allowed_coins_str="BTC",
            allowed_level_labels_str="equal_high_*,prior_15m_low",
            allowed_sessions_str="us",
            allowed_coin_session_pairs_str="BTC:us",
            allowed_coin_session_level_triples_str="BTC:us:prior_15m_low",
            allowed_sides_str="long,short",
            blocked_coins_str="ETH,SOL",
            blocked_level_labels_str="vwap_daily",
            blocked_sessions_str="asia,late",
            blocked_coin_session_pairs_str="BTC:asia",
            blocked_coin_session_level_triples_str="BTC:asia:prior_15m_low",
            min_signal_score=0.5,
            min_confidence_range=0.65,
            maker_fee_pct=0.00015,
            taker_fee_pct=0.00045,
            paper_exit_slippage_bps=1.5,
            paper_tp1_is_taker=True,
        ),
        risk=RiskConfig(account_equity=50.0, risk_per_trade_pct=0.5),
        runtime=RuntimeConfig(),
        replay=ReplayConfig(),
        levels=LevelConfig(),
        live=LiveConfig(),
    )
    matching_run = {
        "event_type": "run",
        "event": "run_start",
        "run_id": "match",
        "coins": ["BTC"],
        "risk_per_trade_pct": 0.5,
        "account_equity": 50.0,
        "min_conf_range": 0.65,
        "min_conf_trend": cfg.strategy.min_confidence_trend,
        "min_signal_score": 0.5,
        "maker_fee_pct": 0.00015,
        "taker_fee_pct": 0.00045,
        "paper_entry_slippage_bps": 0.0,
        "paper_exit_slippage_bps": 1.5,
        "paper_tp1_is_taker": True,
        "allowed_coins": ["BTC"],
        "allowed_level_labels": ["equal_high_*", "prior_15m_low"],
        "allowed_sessions": ["us"],
        "allowed_coin_session_pairs": ["BTC:us"],
        "allowed_coin_session_level_triples": ["BTC:us:prior_15m_low"],
        "allowed_sides": ["long", "short"],
        "blocked_coins": ["ETH", "SOL"],
        "blocked_level_labels": ["vwap_daily"],
        "blocked_sessions": ["asia", "late"],
        "blocked_coin_session_pairs": ["BTC:asia"],
        "blocked_coin_session_level_triples": ["BTC:asia:prior_15m_low"],
        "blocked_sides": [],
    }
    stale_run = {**matching_run, "run_id": "stale", "min_signal_score": 0.4}

    assert go_live_check._active_policy_run_ids([matching_run, stale_run], cfg) == {"match"}


def test_journal_summary_can_aggregate_explicit_run_ids():
    summary = go_live_check._journal_summary(
        [
            {"event_type": "run", "event": "run_start", "run_id": "a"},
            {"event_type": "outcome", "run_id": "a", "signal_id": "a-1", "pnl": 1.0, "r_multiple": 1.0},
            {"event_type": "run", "event": "run_start", "run_id": "b"},
            {"event_type": "outcome", "run_id": "b", "signal_id": "b-1", "pnl": 2.0, "r_multiple": 0.5},
            {"event_type": "run", "event": "run_start", "run_id": "c"},
            {"event_type": "outcome", "run_id": "c", "signal_id": "c-1", "pnl": -9.0, "r_multiple": -9.0},
        ],
        run_ids={"a", "b"},
    )

    assert summary["closed_trades"] == 2
    assert summary["net_pnl"] == 3.0
    assert summary["avg_r"] == 0.75


def test_parse_candle_total_extracts_recent_edge_metrics():
    total = go_live_check._parse_candle_total(
        "noise\nTOTAL: trades=9 avg_r=0.978 win_rate=66.7% profit_factor=3.93 span_days=3.6\n"
    )

    assert total["trades"] == 9
    assert total["avg_r"] == 0.978
    assert total["profit_factor"] == 3.93
    assert total["span_days"] == 3.6


def test_parse_candle_coin_totals_extracts_per_coin_metrics():
    totals = go_live_check._parse_candle_coin_totals(
        "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
        "DOGE: trades=5 avg_r=1.064 win_rate=80.0% profit_factor=4.73\n"
        "TOTAL: trades=12 avg_r=0.988 win_rate=75.0% profit_factor=3.71 span_days=3.5\n"
    )

    assert totals["BTC"]["trades"] == 7
    assert totals["BTC"]["avg_r"] == 0.933
    assert totals["DOGE"]["profit_factor"] == 4.73


def test_parse_candle_slices_extracts_session_side_level_metrics():
    slices = go_live_check._parse_candle_slices(
        "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
        "slice us:short:equal_high_*: trades=2 avg_r=1.213 win_rate=50.0% pf=2.64\n"
        "HYPE: trades=12 avg_r=0.204 win_rate=41.7% profit_factor=1.28\n"
        "slice eu:short:equal_high_*: trades=1 avg_r=0.770 win_rate=100.0% pf=inf\n"
    )

    assert slices[0]["coin"] == "BTC"
    assert slices[0]["session"] == "us"
    assert slices[0]["side"] == "short"
    assert slices[0]["level"] == "equal_high_*"
    assert slices[0]["trades"] == 2
    assert slices[0]["avg_r"] == 1.213
    assert slices[1]["coin"] == "HYPE"
    assert slices[1]["session"] == "eu"
    assert slices[1]["pf"] == go_live_check.math.inf


def test_candle_edge_check_passes_positive_recent_edge(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="TOTAL: trades=9 avg_r=0.978 win_rate=66.7% profit_factor=3.93 span_days=3.6\n",
            stderr="",
        )

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)

    check = go_live_check._candle_edge_check(
        days=7,
        min_trades=5,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        min_span_days=3.0,
    )

    assert check.name == "recent_candle_edge"
    assert check.ok is True
    assert "trades=9" in check.detail


def test_candle_edge_check_fails_negative_recent_edge(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="TOTAL: trades=8 avg_r=-0.120 win_rate=37.5% profit_factor=0.80 span_days=3.6\n",
            stderr="",
        )

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)

    check = go_live_check._candle_edge_check(
        days=7,
        min_trades=5,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        min_span_days=3.0,
    )

    assert check.name == "recent_candle_edge"
    assert check.ok is False
    assert "avg_r=-0.120" in check.detail


def test_candle_edge_check_fails_closed_on_backtest_error(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="network unavailable\n")

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)

    check = go_live_check._candle_edge_check(
        days=7,
        min_trades=5,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        min_span_days=3.0,
    )

    assert check.name == "recent_candle_edge"
    assert check.ok is False
    assert "failed" in check.detail


def test_candle_edge_check_fails_when_actual_span_too_short(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="TOTAL: trades=9 avg_r=0.978 win_rate=66.7% profit_factor=3.93 span_days=1.5\n",
            stderr="",
        )

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)

    check = go_live_check._candle_edge_check(
        days=7,
        min_trades=5,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        min_span_days=3.0,
    )

    assert check.ok is False
    assert "span_days=1.5" in check.detail


def test_candle_level_sample_check_passes_positive_allowed_levels(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "slice us:short:equal_high_*: trades=2 avg_r=1.213 win_rate=50.0% pf=2.64\n"
                "slice us:long:equal_low_*: trades=2 avg_r=0.561 win_rate=50.0% pf=1.75\n"
                "slice us:long:prior_15m_low: trades=5 avg_r=1.082 win_rate=80.0% pf=4.73\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(
                allowed_level_labels={"equal_high_*", "equal_low_*", "prior_15m_low"}
            )
        ),
    )

    check = go_live_check._candle_level_sample_check(
        days=7,
        min_level_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
    )

    assert check.name == "candle_level_samples"
    assert check.ok is True
    assert "equal_low_*=trades:2" in check.detail


def test_candle_level_sample_check_fails_weak_allowed_level(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "slice us:short:equal_high_*: trades=2 avg_r=1.213 win_rate=50.0% pf=2.64\n"
                "slice us:long:equal_low_*: trades=1 avg_r=-0.100 win_rate=0.0% pf=0.50\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(allowed_level_labels={"equal_high_*", "equal_low_*"})
        ),
    )

    check = go_live_check._candle_level_sample_check(
        days=7,
        min_level_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
    )

    assert check.ok is False
    assert "equal_low_*=trades:1" in check.detail


def test_candle_coin_sample_check_passes_positive_allowed_coins(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
                "DOGE: trades=5 avg_r=1.064 win_rate=80.0% profit_factor=4.73\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(strategy=SimpleNamespace(allowed_coins={"BTC", "DOGE"})),
    )

    check = go_live_check._candle_coin_sample_check(
        days=7,
        min_coin_trades=5,
        min_avg_r=0.05,
        min_profit_factor=1.2,
    )

    assert check.name == "candle_coin_samples"
    assert check.ok is True
    assert "BTC=trades:7" in check.detail
    assert "DOGE=trades:5" in check.detail


def test_candle_coin_sample_check_fails_weak_allowed_coin(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
                "DOGE: trades=1 avg_r=-0.200 win_rate=0.0% profit_factor=0.00\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(go_live_check.subprocess, "run", fake_run)
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(strategy=SimpleNamespace(allowed_coins={"BTC", "DOGE"})),
    )

    check = go_live_check._candle_coin_sample_check(
        days=7,
        min_coin_trades=5,
        min_avg_r=0.05,
        min_profit_factor=1.2,
    )

    assert check.ok is False
    assert "DOGE=trades:1" in check.detail


def test_candle_coin_level_sample_check_passes_allowed_pairs(monkeypatch):
    output = (
        "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
        "slice us:long:prior_15m_low: trades=5 avg_r=1.082 win_rate=80.0% pf=4.73\n"
        "slice us:long:equal_low_*: trades=2 avg_r=0.561 win_rate=50.0% pf=1.75\n"
        "HYPE: trades=12 avg_r=0.204 win_rate=41.7% profit_factor=1.28\n"
        "slice us:long:equal_low_*: trades=12 avg_r=0.204 win_rate=41.7% pf=1.28\n"
    )
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(
                allowed_coin_level_pairs={"BTC:prior_15m_low", "HYPE:equal_low_*"}
            )
        ),
    )

    check = go_live_check._candle_coin_level_sample_check(
        days=7,
        min_pair_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        output=output,
    )

    assert check.ok is True
    assert "HYPE:equal_low_*=trades:12" in check.detail


def test_candle_coin_level_sample_check_fails_weak_pair(monkeypatch):
    output = (
        "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
        "slice us:long:prior_15m_low: trades=5 avg_r=1.082 win_rate=80.0% pf=4.73\n"
        "HYPE: trades=12 avg_r=0.204 win_rate=41.7% profit_factor=1.28\n"
        "slice us:long:equal_low_*: trades=1 avg_r=0.204 win_rate=100.0% pf=1.28\n"
    )
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(
                allowed_coin_level_pairs={"BTC:prior_15m_low", "HYPE:equal_low_*"}
            )
        ),
    )

    check = go_live_check._candle_coin_level_sample_check(
        days=7,
        min_pair_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        output=output,
    )

    assert check.ok is False
    assert "HYPE:equal_low_*=trades:1" in check.detail


def test_candle_coin_session_sample_check_passes_allowed_pairs(monkeypatch):
    output = (
        "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
        "slice us:long:prior_15m_low: trades=5 avg_r=1.082 win_rate=80.0% pf=4.73\n"
        "LINK: trades=6 avg_r=1.739 win_rate=83.3% profit_factor=8.64\n"
        "slice asia:short:equal_high_*: trades=2 avg_r=2.000 win_rate=100.0% pf=inf\n"
        "slice eu:long:prior_15m_low: trades=2 avg_r=0.869 win_rate=100.0% pf=inf\n"
    )
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(
                allowed_coin_session_pairs={"BTC:us", "LINK:asia", "LINK:eu"}
            )
        ),
    )

    check = go_live_check._candle_coin_session_sample_check(
        days=7,
        min_pair_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        output=output,
    )

    assert check.ok is True
    assert "LINK:asia=trades:2" in check.detail


def test_candle_coin_session_sample_check_fails_weak_pair(monkeypatch):
    output = (
        "BTC: trades=7 avg_r=0.933 win_rate=71.4% profit_factor=3.21\n"
        "slice us:long:prior_15m_low: trades=5 avg_r=1.082 win_rate=80.0% pf=4.73\n"
        "LINK: trades=1 avg_r=-0.500 win_rate=0.0% profit_factor=0.00\n"
        "slice asia:short:equal_high_*: trades=1 avg_r=-0.500 win_rate=0.0% pf=0.00\n"
    )
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(allowed_coin_session_pairs={"BTC:us", "LINK:asia"})
        ),
    )

    check = go_live_check._candle_coin_session_sample_check(
        days=7,
        min_pair_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        output=output,
    )

    assert check.ok is False
    assert "LINK:asia=trades:1" in check.detail


def test_candle_coin_session_level_sample_check_passes_allowed_triples(monkeypatch):
    output = (
        "HYPE: trades=17 avg_r=0.300 win_rate=50.0% profit_factor=1.50\n"
        "slice asia:short:equal_high_*: trades=5 avg_r=0.387 win_rate=60.0% pf=1.66\n"
        "slice us:long:equal_low_*: trades=12 avg_r=0.204 win_rate=41.7% pf=1.28\n"
    )
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(
                allowed_coin_session_level_triples={"HYPE:asia:equal_high_*", "HYPE:us:equal_low_*"}
            )
        ),
    )

    check = go_live_check._candle_coin_session_level_sample_check(
        days=7,
        min_triple_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        output=output,
    )

    assert check.ok is True
    assert "HYPE:asia:equal_high_*=trades:5" in check.detail


def test_candle_coin_session_level_sample_check_fails_weak_triple(monkeypatch):
    output = (
        "HYPE: trades=1 avg_r=-1.000 win_rate=0.0% profit_factor=0.00\n"
        "slice asia:short:equal_high_*: trades=1 avg_r=-1.000 win_rate=0.0% pf=0.00\n"
    )
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=SimpleNamespace(allowed_coin_session_level_triples={"HYPE:asia:equal_high_*"})
        ),
    )

    check = go_live_check._candle_coin_session_level_sample_check(
        days=7,
        min_triple_trades=2,
        min_avg_r=0.05,
        min_profit_factor=1.2,
        output=output,
    )

    assert check.ok is False
    assert "HYPE:asia:equal_high_*=trades:1" in check.detail


def test_paper_notional_cap_alignment_formula():
    risk = RiskConfig(account_equity=50.0, max_leverage=1.0)

    assert risk.account_equity * risk.max_leverage <= 50.0


def test_sizing_floor_check_passes_current_small_account(monkeypatch):
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=StrategyConfig(max_stop_distance_bps=55.0),
            risk=RiskConfig(
                account_equity=50.0,
                risk_per_trade_pct=0.5,
                max_leverage=1.0,
                risk_mult_min=0.4,
            ),
            live=LiveConfig(max_notional_per_trade=50.0),
        ),
    )

    check = go_live_check._sizing_floor_check()

    assert check.ok is True
    assert "min_notional=10.00" in check.detail


def test_sizing_floor_check_fails_when_worst_case_order_too_small(monkeypatch):
    monkeypatch.setattr(
        go_live_check,
        "load_config",
        lambda: SimpleNamespace(
            strategy=StrategyConfig(max_stop_distance_bps=55.0),
            risk=RiskConfig(
                account_equity=10.0,
                risk_per_trade_pct=0.1,
                max_leverage=1.0,
                risk_mult_min=0.4,
            ),
            live=LiveConfig(max_notional_per_trade=10.0),
        ),
    )

    check = go_live_check._sizing_floor_check()

    assert check.ok is False
    assert "worst_case_notional" in check.detail
