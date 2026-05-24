import json
import subprocess

import scripts.rejected_signal_check as rejected_signal_check


def test_candidate_assignments_use_level_family():
    candidate = {
        "coin": "spx",
        "session": "eu",
        "side": "short",
        "level_label": "equal_high_4",
    }

    assert rejected_signal_check._candidate_assignments(candidate) == [
        "HL_COINS=SPX",
        "BOT_ALLOW_COINS=SPX",
        "BOT_ALLOW_LEVEL_LABELS=equal_high_*",
        "BOT_ALLOW_COIN_LEVELS=SPX:equal_high_*",
        "BOT_ALLOW_COIN_SESSIONS=SPX:eu",
        "BOT_ALLOW_COIN_SESSION_LEVELS=SPX:eu:equal_high_*",
        "BOT_ALLOW_SESSIONS=eu",
        "BOT_ALLOW_SIDES=short",
    ]


def test_blocked_candidates_pair_candidates_and_decisions():
    rows = [
        {"event_type": "candidate", "signal_id": "a", "run_id": "r1", "coin": "TON"},
        {"event_type": "decision", "signal_id": "a", "run_id": "r1", "allowed": False, "reason": "blocked"},
        {"event_type": "candidate", "signal_id": "b", "run_id": "r1", "coin": "SPX"},
        {"event_type": "decision", "signal_id": "b", "run_id": "r1", "allowed": True, "reason": ""},
    ]

    blocked = rejected_signal_check._blocked_candidates(rows, active_policy_only=False)

    assert len(blocked) == 1
    assert blocked[0][0]["signal_id"] == "a"
    assert blocked[0][1]["reason"] == "blocked"


def test_main_checks_latest_blocked_signal(tmp_path, monkeypatch):
    journal = tmp_path / "signals.jsonl"
    rows = [
        {
            "event_type": "candidate",
            "signal_id": "sig",
            "run_id": "run",
            "coin": "SPX",
            "session": "eu",
            "side": "short",
            "level_label": "equal_high_4",
        },
        {
            "event_type": "decision",
            "signal_id": "sig",
            "run_id": "run",
            "allowed": False,
            "reason": "allow_coin_level_miss:SPX:equal_high_4",
        },
    ]
    journal.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    calls = {}

    def fake_call(cmd, cwd):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return 1

    monkeypatch.setattr(rejected_signal_check.sys, "argv", [
        "rejected_signal_check.py",
        "--all-runs",
        "--input",
        str(journal),
    ])
    monkeypatch.setattr(subprocess, "call", fake_call)

    assert rejected_signal_check.main() == 1

    assert calls["cwd"] == rejected_signal_check.ROOT
    assert "policy_candidate_check.py" in str(calls["cmd"])
    assert "BOT_ALLOW_COIN_SESSION_LEVELS=SPX:eu:equal_high_*" in calls["cmd"]
