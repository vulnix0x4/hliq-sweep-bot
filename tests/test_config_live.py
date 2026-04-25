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
