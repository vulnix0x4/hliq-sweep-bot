from pathlib import Path

import scripts.replay_capture as replay_capture


def test_load_env_respects_existing_shell_override(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BOT_ALLOW_SESSIONS=us\n"
        "BOT_RISK_PER_TRADE_PCT=0.50\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BOT_ALLOW_SESSIONS", "asia")
    monkeypatch.delenv("BOT_RISK_PER_TRADE_PCT", raising=False)

    replay_capture._load_env(env_file)

    assert replay_capture.os.environ["BOT_ALLOW_SESSIONS"] == "asia"
    assert replay_capture.os.environ["BOT_RISK_PER_TRADE_PCT"] == "0.50"


def test_ignore_runtime_pause_uses_isolated_pause_path(tmp_path: Path, monkeypatch):
    replay_input = tmp_path / "events.jsonl"
    replay_journal = tmp_path / "replay.jsonl"
    replay_input.write_text("", encoding="utf-8")
    seen = {}

    class FakeBot:
        def __init__(self, cfg):
            seen["pause_path"] = cfg.runtime.trade_pause_path

        def run_replay(self, events):
            seen["events"] = list(events)
            return {"ok": 1}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(replay_capture, "ROOT", tmp_path)
    monkeypatch.setattr(replay_capture, "SweepBot", FakeBot)
    monkeypatch.setattr(replay_capture, "load_market_events", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        replay_capture.sys,
        "argv",
        [
            "replay_capture.py",
            "--input",
            str(replay_input),
            "--journal",
            str(replay_journal),
            "--ignore-runtime-pause",
        ],
    )

    assert replay_capture.main() == 0
    assert seen["pause_path"].endswith("runtime/replay_pause_ignored.flag")
