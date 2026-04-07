from __future__ import annotations

from hliq_bot.config import RiskConfig, StrategyConfig
from hliq_bot.models import ClosedTrade, MarketState, Side
from hliq_bot.risk.governor import RiskGovernor


def _market_state(ts_ms: int = 0) -> MarketState:
    return MarketState(
        ts_ms=ts_ms,
        ws_healthy=True,
        data_stale=False,
        spread_bps=1.0,
        recent_bar_ranges_pct=[0.3, 0.4, 0.5],
        move_30s_pct=0.1,
    )


def _closed_trade(ts_ms: int, r_multiple: float) -> ClosedTrade:
    pnl = 10.0 * r_multiple
    return ClosedTrade(
        side=Side.LONG,
        entry_price=100.0,
        exit_price=100.0 + pnl / 10.0,
        qty=10.0,
        pnl=pnl,
        risk_dollars=10.0,
        r_multiple=r_multiple,
        opened_ms=ts_ms - 5_000,
        closed_ms=ts_ms,
        exit_reason="test",
    )


def test_position_sizing_uses_risk_cap() -> None:
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            max_leverage=5.0,
            min_qty=0.0001,
        ),
        strategy_cfg=StrategyConfig(),
    )
    size = gov.size_position(entry_price=100.0, stop_price=99.5)
    # $10 risk / $0.5 stop distance = 20 units
    assert round(size.qty, 6) == 20.0
    assert round(size.notional, 6) == 2000.0


def test_daily_loss_limit_blocks_new_trades() -> None:
    risk_cfg = RiskConfig(
        account_equity=1000.0,
        risk_per_trade_pct=1.0,
        daily_loss_limit_r=3.0,
    )
    gov = RiskGovernor(risk_cfg=risk_cfg, strategy_cfg=StrategyConfig())
    ts = 1_700_000_000_000

    for i in range(3):
        gov.register_closed_trade(
            ClosedTrade(
                side=Side.LONG,
                entry_price=100.0,
                exit_price=99.5,
                qty=20.0,
                pnl=-10.0,
                risk_dollars=10.0,
                r_multiple=-1.0,
                opened_ms=ts + i * 10_000,
                closed_ms=ts + i * 10_000 + 5_000,
                exit_reason="stop_loss",
            )
        )

    check = gov.can_open_new_trade(_market_state(ts_ms=ts + 60_000))
    assert check.allowed is False
    assert "daily loss" in check.reason


def test_loss_cooldown_blocks_after_loss() -> None:
    ts = 1_700_100_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            loss_cooldown_sec=120,
            hard_loss_cooldown_sec=0,
            edge_pause_min_trades=50,
        ),
        strategy_cfg=StrategyConfig(),
    )
    gov.register_closed_trade(_closed_trade(ts, r_multiple=-1.0))
    blocked = gov.can_open_new_trade(_market_state(ts_ms=ts + 30_000))
    assert blocked.allowed is False
    assert "cooldown" in blocked.reason

    allowed = gov.can_open_new_trade(_market_state(ts_ms=ts + 125_000))
    assert allowed.allowed is True


def test_edge_pause_blocks_on_bad_recent_expectancy() -> None:
    ts = 1_700_200_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            edge_pause_avg_r=-0.2,
            edge_pause_min_trades=3,
            loss_cooldown_sec=0,
            daily_loss_limit_r=20.0,
        ),
        strategy_cfg=StrategyConfig(),
    )
    gov.register_closed_trade(_closed_trade(ts + 1_000, r_multiple=-0.5))
    gov.register_closed_trade(_closed_trade(ts + 2_000, r_multiple=-0.4))
    gov.register_closed_trade(_closed_trade(ts + 3_000, r_multiple=-0.3))

    check = gov.can_open_new_trade(_market_state(ts_ms=ts + 60_000))
    assert check.allowed is False
    assert "edge pause" in check.reason


def test_performance_multiplier_waits_for_min_trades() -> None:
    ts = 1_700_300_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            min_trades_for_perf_scaling=4,
            perf_window_trades=20,
        ),
        strategy_cfg=StrategyConfig(),
    )
    gov.register_closed_trade(_closed_trade(ts + 1_000, r_multiple=0.5))
    gov.register_closed_trade(_closed_trade(ts + 2_000, r_multiple=0.4))
    assert gov.performance_multiplier() == 1.0

    gov.register_closed_trade(_closed_trade(ts + 3_000, r_multiple=0.3))
    gov.register_closed_trade(_closed_trade(ts + 4_000, r_multiple=0.2))
    assert gov.performance_multiplier() > 1.0


def test_side_edge_pause_blocks_only_weak_side() -> None:
    ts = 1_700_400_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            side_edge_pause_avg_r=-0.2,
            side_edge_pause_min_trades=3,
            loss_cooldown_sec=0,
            edge_pause_min_trades=50,
            daily_loss_limit_r=20.0,
        ),
        strategy_cfg=StrategyConfig(),
    )
    gov.register_closed_trade(_closed_trade(ts + 1_000, r_multiple=-0.5))
    gov.register_closed_trade(_closed_trade(ts + 2_000, r_multiple=-0.4))
    gov.register_closed_trade(_closed_trade(ts + 3_000, r_multiple=-0.3))

    blocked_long = gov.can_trade_side(Side.LONG)
    assert blocked_long.allowed is False
    assert "long_edge_pause" in blocked_long.reason

    allowed_short = gov.can_trade_side(Side.SHORT)
    assert allowed_short.allowed is True


def test_side_edge_pause_cooldown_expires_and_releases_side() -> None:
    ts = 1_700_450_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            side_edge_pause_avg_r=-0.2,
            side_edge_pause_min_trades=3,
            side_edge_pause_cooldown_sec=60,
            loss_cooldown_sec=0,
            edge_pause_min_trades=50,
            daily_loss_limit_r=20.0,
        ),
        strategy_cfg=StrategyConfig(),
    )

    gov.register_closed_trade(_closed_trade(ts + 1_000, r_multiple=-0.5))
    gov.register_closed_trade(_closed_trade(ts + 2_000, r_multiple=-0.4))
    gov.register_closed_trade(_closed_trade(ts + 3_000, r_multiple=-0.3))

    first_block = gov.can_trade_side(Side.LONG, ts_ms=ts + 10_000)
    assert first_block.allowed is False
    assert "long_edge_pause" in first_block.reason

    cooldown_block = gov.can_trade_side(Side.LONG, ts_ms=ts + 40_000)
    assert cooldown_block.allowed is False
    assert "long_edge_pause_cooldown" in cooldown_block.reason

    released = gov.can_trade_side(Side.LONG, ts_ms=ts + 80_000)
    assert released.allowed is True

    # Side history is cleared on release, so it should not immediately re-block.
    still_allowed = gov.can_trade_side(Side.LONG, ts_ms=ts + 81_000)
    assert still_allowed.allowed is True


def test_session_and_level_edge_pause() -> None:
    ts = 1_700_500_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            session_edge_pause_avg_r=-0.2,
            session_edge_pause_min_trades=2,
            level_edge_pause_avg_r=-0.2,
            level_edge_pause_min_trades=2,
            loss_cooldown_sec=0,
            edge_pause_min_trades=50,
            daily_loss_limit_r=20.0,
        ),
        strategy_cfg=StrategyConfig(),
    )
    gov.register_closed_trade(_closed_trade(ts + 1_000, r_multiple=-0.5), session="asia", level_label="prior_15m_low")
    gov.register_closed_trade(_closed_trade(ts + 2_000, r_multiple=-0.4), session="asia", level_label="prior_15m_low")

    blocked_session = gov.can_trade_session("asia")
    assert blocked_session.allowed is False
    assert "session_edge_pause" in blocked_session.reason

    blocked_level = gov.can_trade_level("prior_15m_low")
    assert blocked_level.allowed is False
    assert "level_edge_pause" in blocked_level.reason

    allowed_other = gov.can_trade_session("us")
    assert allowed_other.allowed is True


def test_hard_loss_cooldown_blocks_new_trade_temporarily() -> None:
    ts = 1_700_600_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            hard_loss_r=-0.8,
            hard_loss_cooldown_sec=120,
            loss_cooldown_sec=0,
            edge_pause_min_trades=50,
            daily_loss_limit_r=20.0,
        ),
        strategy_cfg=StrategyConfig(),
    )
    gov.register_closed_trade(_closed_trade(ts + 1_000, r_multiple=-0.95))

    blocked = gov.can_open_new_trade(_market_state(ts_ms=ts + 30_000))
    assert blocked.allowed is False
    assert "hard loss cooldown" in blocked.reason

    allowed = gov.can_open_new_trade(_market_state(ts_ms=ts + 130_000))
    assert allowed.allowed is True


def test_side_and_level_hard_loss_cooldowns_are_scoped() -> None:
    ts = 1_700_700_000_000
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            side_hard_loss_r=-0.8,
            side_hard_loss_cooldown_sec=180,
            level_hard_loss_r=-0.8,
            level_hard_loss_cooldown_sec=180,
            loss_cooldown_sec=0,
            edge_pause_min_trades=50,
            side_edge_pause_min_trades=50,
            level_edge_pause_min_trades=50,
            daily_loss_limit_r=20.0,
        ),
        strategy_cfg=StrategyConfig(),
    )
    gov.register_closed_trade(
        ClosedTrade(
            side=Side.LONG,
            entry_price=100.0,
            exit_price=99.0,
            qty=10.0,
            pnl=-10.0,
            risk_dollars=10.0,
            r_multiple=-1.0,
            opened_ms=ts,
            closed_ms=ts + 1_000,
            exit_reason="stop_loss",
        ),
        level_label="prior_15m_low",
    )

    blocked_long = gov.can_trade_side(Side.LONG, ts_ms=ts + 30_000)
    assert blocked_long.allowed is False
    assert "long_hard_loss_cooldown" in blocked_long.reason

    allowed_short = gov.can_trade_side(Side.SHORT, ts_ms=ts + 30_000)
    assert allowed_short.allowed is True

    blocked_level = gov.can_trade_level("prior_15m_low", ts_ms=ts + 30_000)
    assert blocked_level.allowed is False
    assert "level_hard_loss_pause" in blocked_level.reason

    allowed_other_level = gov.can_trade_level("prior_15m_high", ts_ms=ts + 30_000)
    assert allowed_other_level.allowed is True

    # Cooldowns expire.
    assert gov.can_trade_side(Side.LONG, ts_ms=ts + 300_000).allowed is True
    assert gov.can_trade_level("prior_15m_low", ts_ms=ts + 300_000).allowed is True
