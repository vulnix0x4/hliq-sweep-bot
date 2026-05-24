import scripts.session_profit_watch as session_profit_watch
from datetime import datetime, timezone


def test_seconds_until_disallowed_stops_at_session_boundary():
    now = datetime(2026, 5, 12, 21, 30, tzinfo=timezone.utc)

    assert session_profit_watch._seconds_until_disallowed(now, {"us"}) == 1800


def test_seconds_until_disallowed_handles_adjacent_allowed_sessions():
    now = datetime(2026, 5, 12, 12, 30, tzinfo=timezone.utc)

    assert session_profit_watch._seconds_until_disallowed(now, {"eu", "us"}) == 34_200


def test_session_profit_watch_dry_run_does_not_sleep(monkeypatch, capsys):
    monkeypatch.setenv("BOT_ALLOW_SESSIONS", "eu,us")
    monkeypatch.setattr(
        session_profit_watch.sys,
        "argv",
        ["session_profit_watch.py", "--dry-run"],
    )

    assert session_profit_watch.main() == 0

    out = capsys.readouterr().out
    assert "Waiting" in out
    assert "EU,US session" in out


def test_session_profit_watch_respects_max_wait(monkeypatch, capsys):
    monkeypatch.setenv("BOT_ALLOW_SESSIONS", "us")
    monkeypatch.setattr(
        session_profit_watch.sys,
        "argv",
        ["session_profit_watch.py", "--session", "us", "--max-wait-sec", "0"],
    )

    assert session_profit_watch.main() == 2

    err = capsys.readouterr().err
    assert "Refusing to sleep longer" in err


def test_session_profit_watch_loop_retries_until_success(monkeypatch):
    monkeypatch.setenv("BOT_ALLOW_SESSIONS", "eu,us")
    monkeypatch.setattr(
        session_profit_watch.sys,
        "argv",
        [
            "session_profit_watch.py",
            "--loop",
            "--skip-edge-check",
            "--timeout-sec",
            "1",
            "--poll-sec",
            "1",
        ],
    )
    monkeypatch.setattr(session_profit_watch.time, "sleep", lambda _seconds: None)
    calls = iter([2, 0])
    monkeypatch.setattr(session_profit_watch.subprocess, "call", lambda _cmd: next(calls))

    assert session_profit_watch.main() == 0


def test_session_profit_watch_clears_pause_after_edge_check_passes(monkeypatch, tmp_path):
    pause_path = tmp_path / "pause.flag"
    monkeypatch.setenv("BOT_ALLOW_SESSIONS", "eu,us")
    monkeypatch.setenv("BOT_TRADE_PAUSE_PATH", str(pause_path))
    monkeypatch.setattr(
        session_profit_watch.sys,
        "argv",
        [
            "session_profit_watch.py",
            "--timeout-sec",
            "1",
            "--poll-sec",
            "1",
        ],
    )
    monkeypatch.setattr(session_profit_watch.time, "sleep", lambda _seconds: None)
    calls = []

    def fake_call(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(session_profit_watch.subprocess, "call", fake_call)

    assert session_profit_watch.main() == 0
    assert len(calls) == 2
    assert "policy_freshness_check.py" in str(calls[0])
    assert "--min-session-trades" in calls[0]
    assert "--min-coin-trades" in calls[0]
    assert "--min-level-trades" in calls[0]
    assert "proof_watch.py" in str(calls[1])
    assert "--active-policy-runs" in calls[1]
    assert not pause_path.exists()


def test_session_profit_watch_edge_failure_backs_off_inside_allowed_session(monkeypatch, tmp_path):
    pause_path = tmp_path / "pause.flag"
    monkeypatch.setenv("BOT_ALLOW_SESSIONS", "us")
    monkeypatch.setenv("BOT_TRADE_PAUSE_PATH", str(pause_path))
    monkeypatch.setattr(
        session_profit_watch.sys,
        "argv",
        [
            "session_profit_watch.py",
            "--loop",
            "--max-wait-sec",
            "10",
            "--edge-retry-sec",
            "300",
        ],
    )
    monkeypatch.setattr(
        session_profit_watch,
        "_next_allowed",
        lambda now, _sessions: now,
    )
    monkeypatch.setattr(session_profit_watch, "_session_from_hour", lambda _hour: "us")
    monkeypatch.setattr(
        session_profit_watch,
        "_seconds_until_disallowed",
        lambda _now, _sessions: 120,
    )
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(session_profit_watch.time, "sleep", fake_sleep)
    monkeypatch.setattr(session_profit_watch.subprocess, "call", lambda _cmd: 1)

    try:
        session_profit_watch.main()
    except KeyboardInterrupt:
        pass

    assert sleeps[-1] == 120
    assert pause_path.read_text(encoding="utf-8").strip() == "edge_check_failed"
