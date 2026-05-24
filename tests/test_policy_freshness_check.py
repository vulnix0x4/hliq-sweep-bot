from types import SimpleNamespace

import scripts.policy_freshness_check as policy_freshness_check


def test_policy_freshness_requires_edge_session_and_level_samples(monkeypatch, capsys):
    monkeypatch.setattr(
        policy_freshness_check.sys,
        "argv",
        ["policy_freshness_check.py"],
    )
    monkeypatch.setattr(policy_freshness_check, "_load_env_file", lambda _path: None)
    monkeypatch.setattr(policy_freshness_check, "_run_candle_backtest", lambda _days: ("out", ""))
    monkeypatch.setattr(
        policy_freshness_check,
        "_edge_check_from_output",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, name="recent_candle_edge", detail="ok"),
    )
    monkeypatch.setattr(
        policy_freshness_check,
        "_session_check_from_output",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, name="candle_session_samples", detail="ok"),
    )
    monkeypatch.setattr(
        policy_freshness_check,
        "_coin_check_from_output",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, name="candle_coin_samples", detail="ok"),
    )
    monkeypatch.setattr(
        policy_freshness_check,
        "_level_check_from_output",
        lambda *_args, **_kwargs: SimpleNamespace(ok=False, name="candle_level_samples", detail="weak"),
    )

    assert policy_freshness_check.main() == 1

    out = capsys.readouterr().out
    assert "status: FAIL" in out
    assert "[PASS] recent_candle_edge" in out
    assert "[PASS] candle_session_samples" in out
    assert "[PASS] candle_coin_samples" in out
    assert "[FAIL] candle_level_samples" in out


def test_policy_freshness_passes_when_all_checks_pass(monkeypatch, capsys):
    monkeypatch.setattr(
        policy_freshness_check.sys,
        "argv",
        [
            "policy_freshness_check.py",
            "--min-session-trades",
            "3",
            "--min-coin-trades",
            "5",
            "--min-level-trades",
            "4",
        ],
    )
    monkeypatch.setattr(policy_freshness_check, "_load_env_file", lambda _path: None)
    monkeypatch.setattr(policy_freshness_check, "_run_candle_backtest", lambda _days: ("out", ""))
    seen = {}
    monkeypatch.setattr(
        policy_freshness_check,
        "_edge_check_from_output",
        lambda *_args, **kwargs: SimpleNamespace(ok=True, name="recent_candle_edge", detail=str(kwargs)),
    )

    def fake_session_check(*_args, **kwargs):
        seen["session"] = kwargs
        return SimpleNamespace(ok=True, name="candle_session_samples", detail="ok")

    def fake_level_check(*_args, **kwargs):
        seen["level"] = kwargs
        return SimpleNamespace(ok=True, name="candle_level_samples", detail="ok")

    monkeypatch.setattr(policy_freshness_check, "_session_check_from_output", fake_session_check)
    def fake_coin_check(*_args, **kwargs):
        seen["coin"] = kwargs
        return SimpleNamespace(ok=True, name="candle_coin_samples", detail="ok")

    monkeypatch.setattr(policy_freshness_check, "_coin_check_from_output", fake_coin_check)
    monkeypatch.setattr(policy_freshness_check, "_level_check_from_output", fake_level_check)

    assert policy_freshness_check.main() == 0

    assert seen["session"]["min_session_trades"] == 3
    assert seen["coin"]["min_coin_trades"] == 5
    assert seen["level"]["min_level_trades"] == 4
    assert "status: PASS" in capsys.readouterr().out
