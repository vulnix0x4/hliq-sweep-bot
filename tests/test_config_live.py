import os
from hliq_bot.config import load_config


def test_live_mode_defaults_paper(monkeypatch):
    """Default config must be paper. Live requires explicit opt-in."""
    monkeypatch.delenv("BOT_MODE", raising=False)
    monkeypatch.delenv("BOT_ALLOW_LIVE", raising=False)
    cfg = load_config()
    assert cfg.mode == "paper"
    assert cfg.live.allow_live is False
    assert cfg.live.network == "testnet"
    assert cfg.live.agent_private_key == ""
    assert cfg.live.main_wallet_address == ""


def test_load_config_default_maker_fee_matches_retail_paid_fee(monkeypatch):
    monkeypatch.delenv("BOT_MAKER_FEE_PCT", raising=False)
    cfg = load_config()
    assert cfg.strategy.maker_fee_pct == 0.00015


def test_live_mode_loads_from_env(monkeypatch):
    monkeypatch.setenv("BOT_MODE", "live")
    monkeypatch.setenv("BOT_ALLOW_LIVE", "true")
    monkeypatch.setenv("HL_NETWORK", "mainnet")
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", "0x" + "a" * 64)
    monkeypatch.setenv("HL_MAIN_WALLET_ADDRESS", "0x" + "b" * 40)
    cfg = load_config()
    assert cfg.mode == "live"
    assert cfg.live.allow_live is True
    assert cfg.live.network == "mainnet"
    assert cfg.live.agent_private_key.startswith("0x")
    assert cfg.live.main_wallet_address.startswith("0x")


def test_coin_level_pair_policy_normalizes_coin_only(monkeypatch):
    monkeypatch.setenv("BOT_ALLOW_COIN_LEVELS", "btc:prior_15m_low,HYPE:equal_low_*")
    monkeypatch.setenv("BOT_BLOCK_COIN_LEVELS", "hype:prior_15m_low")

    cfg = load_config()

    assert cfg.strategy.allowed_coin_level_pairs == {"BTC:prior_15m_low", "HYPE:equal_low_*"}
    assert cfg.strategy.blocked_coin_level_pairs == {"HYPE:prior_15m_low"}


def test_coin_session_pair_policy_normalizes_coin_only(monkeypatch):
    monkeypatch.setenv("BOT_ALLOW_COIN_SESSIONS", "btc:us,link:asia")
    monkeypatch.setenv("BOT_BLOCK_COIN_SESSIONS", "hype:us")

    cfg = load_config()

    assert cfg.strategy.allowed_coin_session_pairs == {"BTC:us", "LINK:asia"}
    assert cfg.strategy.blocked_coin_session_pairs == {"HYPE:us"}


def test_coin_session_level_policy_normalizes_coin_session_level(monkeypatch):
    monkeypatch.setenv("BOT_ALLOW_COIN_SESSION_LEVELS", "hype:asia:equal_high_*,btc:us:prior_15m_low")
    monkeypatch.setenv("BOT_BLOCK_COIN_SESSION_LEVELS", "hype:asia:equal_low_*")

    cfg = load_config()

    assert cfg.strategy.allowed_coin_session_level_triples == {
        "BTC:us:prior_15m_low",
        "HYPE:asia:equal_high_*",
    }
    assert cfg.strategy.blocked_coin_session_level_triples == {"HYPE:asia:equal_low_*"}


def test_live_config_rejects_bad_deadman_timing(monkeypatch):
    monkeypatch.setenv("BOT_MODE", "live")
    monkeypatch.setenv("BOT_ALLOW_LIVE", "true")
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", "0x" + "a" * 64)
    monkeypatch.setenv("HL_DEADMAN_CANCEL_SEC", "30")
    monkeypatch.setenv("HL_DEADMAN_REFRESH_SEC", "30")  # equal -> bad
    import pytest
    with pytest.raises(ValueError, match="deadman"):
        load_config()


def test_live_config_rejects_insufficient_safety_margin(monkeypatch):
    monkeypatch.setenv("HL_DEADMAN_CANCEL_SEC", "40")
    monkeypatch.setenv("HL_DEADMAN_REFRESH_SEC", "30")  # 40 < 2*30 = 60 -> bad
    import pytest
    with pytest.raises(ValueError, match="2 \\* deadman_refresh_sec"):
        load_config()
