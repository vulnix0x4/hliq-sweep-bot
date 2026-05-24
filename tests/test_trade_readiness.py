import json
import subprocess

import scripts.trade_readiness as trade_readiness


def test_trade_readiness_reports_collecting_but_not_live_ready(monkeypatch, capsys):
    def fake_run(cmd, cwd, check, capture_output, text):
        assert cwd == trade_readiness.ROOT
        if "proof_watch.py" in str(cmd):
            payload = {
                "runtime_paused": False,
                "session": "eu",
                "session_allowed": True,
                "candidates": 1,
                "allowed": 0,
                "placed": 0,
                "filled": 0,
                "closed": 0,
                "net_pnl": 0.0,
                "avg_r": 0.0,
                "pending_entries": 0,
                "open_paper_positions": 0,
                "active_position_files": 0,
            }
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload) + "\n", stderr="")
        out = "\n".join([
            "Go-Live Check",
            "[PASS] live_disarmed_now: mode=paper allow_live=False",
            "[FAIL] paper_sample_size: closed_trades=0 min=50",
            "[FAIL] paper_expectancy: avg_r=0.000 min=0.100",
        ])
        return subprocess.CompletedProcess(cmd, 1, stdout=out, stderr="")

    monkeypatch.setattr(trade_readiness.sys, "argv", ["trade_readiness.py"])
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert trade_readiness.main() == 0

    out = capsys.readouterr().out
    assert "paper_collecting: yes" in out
    assert "live_ready: no" in out
    assert "[FAIL] paper_sample_size" in out


def test_trade_readiness_json_marks_paused_not_collecting(monkeypatch, capsys):
    def fake_run(cmd, cwd, check, capture_output, text):
        if "proof_watch.py" in str(cmd):
            payload = {
                "runtime_paused": True,
                "session": "eu",
                "session_allowed": True,
                "pending_entries": 0,
                "open_paper_positions": 0,
                "active_position_files": 0,
            }
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload) + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="Go-Live Check\nstatus: PASS\n", stderr="")

    monkeypatch.setattr(trade_readiness.sys, "argv", ["trade_readiness.py", "--json"])
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert trade_readiness.main() == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["paper_collecting"] is False
    assert payload["runtime_paused"] is True
    assert payload["live_ready"] is True
