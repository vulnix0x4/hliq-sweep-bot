import subprocess

import scripts.policy_candidate_check as policy_candidate_check


def test_candidate_env_clears_policy_and_applies_overrides():
    base = {
        "HL_COINS": "BTC,ETH",
        "BOT_ALLOW_COINS": "BTC,ETH",
        "BOT_BLOCK_SESSIONS": "late",
        "UNRELATED": "keep",
    }

    env = policy_candidate_check._candidate_env(
        base,
        ["HL_COINS=TON", "BOT_ALLOW_COINS=TON", "BOT_ALLOW_SESSIONS=eu"],
        clear_policy=True,
    )

    assert env["HL_COINS"] == "TON"
    assert env["BOT_ALLOW_COINS"] == "TON"
    assert env["BOT_ALLOW_SESSIONS"] == "eu"
    assert env["BOT_BLOCK_SESSIONS"] == "late"
    assert env["BOT_ALLOW_LEVEL_LABELS"] == ""
    assert env["UNRELATED"] == "keep"


def test_parse_assignment_rejects_bad_input():
    try:
        policy_candidate_check._parse_assignment("NOT_AN_ASSIGNMENT")
    except ValueError as exc:
        assert "KEY=VALUE" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_main_invokes_policy_freshness_with_candidate_env(monkeypatch, capsys):
    calls = {}

    def fake_call(cmd, cwd, env):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["env"] = env
        return 0

    monkeypatch.setattr(policy_candidate_check.sys, "argv", [
        "policy_candidate_check.py",
        "--clear-policy",
        "--set",
        "HL_COINS=TON",
        "--set",
        "BOT_ALLOW_COINS=TON",
        "--min-trades",
        "12",
    ])
    monkeypatch.setattr(subprocess, "call", fake_call)

    assert policy_candidate_check.main() == 0

    assert "policy_freshness_check.py" in str(calls["cmd"])
    assert "--min-trades" in calls["cmd"]
    assert "12" in calls["cmd"]
    assert calls["cwd"] == policy_candidate_check.ROOT
    assert calls["env"]["HL_COINS"] == "TON"
    assert calls["env"]["BOT_ALLOW_COINS"] == "TON"
    assert calls["env"]["BOT_ALLOW_LEVEL_LABELS"] == ""
    assert "candidate assignments" in capsys.readouterr().out
