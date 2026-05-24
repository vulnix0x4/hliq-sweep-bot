import json
from datetime import datetime, timezone

import scripts.proof_watch as proof_watch


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _run_start(run_id, *, score=0.5):
    return {
        "event_type": "run",
        "event": "run_start",
        "run_id": run_id,
        "coins": ["BTC"],
        "risk_per_trade_pct": 0.5,
        "account_equity": 50.0,
        "min_conf_range": 0.65,
        "min_conf_trend": 0.72,
        "min_signal_score": score,
        "maker_fee_pct": 0.00015,
        "taker_fee_pct": 0.00045,
        "paper_entry_slippage_bps": 0.0,
        "paper_exit_slippage_bps": 1.5,
        "paper_tp1_is_taker": True,
        "allowed_coins": ["BTC"],
        "allowed_level_labels": ["equal_high_*", "equal_low_*", "prior_15m_low"],
        "allowed_sessions": ["us"],
        "allowed_sides": ["long", "short"],
        "blocked_coins": ["ETH", "SOL"],
        "blocked_level_labels": ["prior_15m_high", "prior_1h_high", "prior_1h_low", "session_open_current", "vwap_daily"],
        "blocked_sessions": ["asia", "late"],
        "blocked_sides": [],
    }


def test_state_uses_latest_run_and_summarizes_profit(monkeypatch, tmp_path):
    monkeypatch.delenv("BOT_ALLOW_SESSIONS", raising=False)
    monkeypatch.setenv("BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    journal = tmp_path / "signals.jsonl"
    _write_rows(
        journal,
        [
            {"event_type": "run", "event": "run_start", "run_id": "old"},
            {"event_type": "outcome", "run_id": "old", "signal_id": "old-1", "pnl": -1, "r_multiple": -1},
            {"event_type": "run", "event": "run_start", "run_id": "new"},
            {"event_type": "candidate", "run_id": "new", "signal_id": "sig-1"},
            {"event_type": "decision", "run_id": "new", "signal_id": "sig-1", "allowed": True},
            {"event_type": "lifecycle", "run_id": "new", "event": "entry_placed"},
            {"event_type": "lifecycle", "run_id": "new", "event": "entry_filled"},
            {"event_type": "outcome", "run_id": "new", "signal_id": "sig-1", "pnl": 2.5, "r_multiple": 1.25},
        ],
    )

    state = proof_watch._state(journal, "")

    assert state.run_id == "new"
    assert state.candidates == 1
    assert state.decisions == 1
    assert state.allowed == 1
    assert state.placed == 1
    assert state.filled == 1
    assert state.closed == 1
    assert state.pending_entries == 0
    assert state.open_paper_positions == 0
    assert state.active_positions == 0
    assert state.net_pnl == 2.5
    assert state.avg_r == 1.25


def test_state_can_select_specific_run(monkeypatch, tmp_path):
    monkeypatch.delenv("BOT_ALLOW_SESSIONS", raising=False)
    monkeypatch.setenv("BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    journal = tmp_path / "signals.jsonl"
    _write_rows(
        journal,
        [
            {"event_type": "run", "event": "run_start", "run_id": "a"},
            {"event_type": "outcome", "run_id": "a", "signal_id": "a-1", "pnl": 1, "r_multiple": 1},
            {"event_type": "run", "event": "run_start", "run_id": "b"},
            {"event_type": "outcome", "run_id": "b", "signal_id": "b-1", "pnl": -2, "r_multiple": -1},
        ],
    )

    state = proof_watch._state(journal, "a")

    assert state.run_id == "a"
    assert state.closed == 1
    assert state.net_pnl == 1


def test_state_can_aggregate_active_policy_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("HL_COINS", "BTC")
    monkeypatch.setenv("BOT_ALLOW_COINS", "BTC")
    monkeypatch.setenv("BOT_ALLOW_LEVEL_LABELS", "equal_high_*,equal_low_*,prior_15m_low")
    monkeypatch.setenv("BOT_ALLOW_SESSIONS", "us")
    monkeypatch.setenv("BOT_ALLOW_SIDES", "long,short")
    monkeypatch.setenv("BOT_BLOCK_COINS", "ETH,SOL")
    monkeypatch.setenv("BOT_BLOCK_LEVEL_LABELS", "prior_15m_high,prior_1h_low,prior_1h_high,session_open_current,vwap_daily")
    monkeypatch.setenv("BOT_BLOCK_SESSIONS", "asia,late")
    monkeypatch.setenv("BOT_ACCOUNT_EQUITY", "50")
    monkeypatch.setenv("BOT_RISK_PER_TRADE_PCT", "0.50")
    monkeypatch.setenv("BOT_MIN_CONF_RANGE", "0.65")
    monkeypatch.setenv("BOT_MIN_CONF_TREND", "0.72")
    monkeypatch.setenv("BOT_MIN_SIGNAL_SCORE", "0.50")
    monkeypatch.setenv("BOT_MAKER_FEE_PCT", "0.00015")
    monkeypatch.setenv("BOT_TAKER_FEE_PCT", "0.00045")
    monkeypatch.setenv("BOT_PAPER_ENTRY_SLIPPAGE_BPS", "0.0")
    monkeypatch.setenv("BOT_PAPER_EXIT_SLIPPAGE_BPS", "1.5")
    monkeypatch.setenv("BOT_PAPER_TP1_IS_TAKER", "true")
    monkeypatch.setenv("BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    journal = tmp_path / "signals.jsonl"
    _write_rows(
        journal,
        [
            _run_start("match-a"),
            {"event_type": "outcome", "run_id": "match-a", "signal_id": "a-1", "pnl": 1.0, "r_multiple": 1.0},
            _run_start("match-b"),
            {"event_type": "outcome", "run_id": "match-b", "signal_id": "b-1", "pnl": 2.0, "r_multiple": 0.5},
            _run_start("stale", score=0.4),
            {"event_type": "outcome", "run_id": "stale", "signal_id": "c-1", "pnl": -9.0, "r_multiple": -9.0},
        ],
    )

    state = proof_watch._state(journal, "", active_policy_runs=True)

    assert state.run_id == "active_policy(2 runs)"
    assert state.closed == 2
    assert state.net_pnl == 3.0
    assert state.avg_r == 0.75


def test_state_reports_active_position_files(monkeypatch, tmp_path):
    monkeypatch.delenv("BOT_ALLOW_SESSIONS", raising=False)
    runtime_dir = tmp_path / "runtime"
    active_dir = runtime_dir / "active_positions"
    active_dir.mkdir(parents=True)
    (active_dir / "BTC.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("BOT_RUNTIME_DIR", str(runtime_dir))
    journal = tmp_path / "signals.jsonl"
    _write_rows(journal, [{"event_type": "run", "event": "run_start", "run_id": "r"}])

    state = proof_watch._state(journal, "")

    assert state.active_positions == 1


def test_state_reports_runtime_pause_file(monkeypatch, tmp_path):
    monkeypatch.delenv("BOT_ALLOW_SESSIONS", raising=False)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True)
    pause_path = runtime_dir / "trade_pause.flag"
    pause_path.write_text("edge_check_pending\n", encoding="utf-8")
    monkeypatch.setenv("BOT_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("BOT_TRADE_PAUSE_PATH", str(pause_path))
    journal = tmp_path / "signals.jsonl"
    _write_rows(journal, [{"event_type": "run", "event": "run_start", "run_id": "r"}])

    state = proof_watch._state(journal, "")

    assert state.runtime_paused is True
    assert state.runtime_pause_reason == "edge_check_pending"


def test_state_reports_pending_and_open_paper_exposure(monkeypatch, tmp_path):
    monkeypatch.delenv("BOT_ALLOW_SESSIONS", raising=False)
    monkeypatch.setenv("BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    journal = tmp_path / "signals.jsonl"
    _write_rows(
        journal,
        [
            {"event_type": "run", "event": "run_start", "run_id": "r"},
            {"event_type": "lifecycle", "run_id": "r", "event": "entry_placed"},
            {"event_type": "lifecycle", "run_id": "r", "event": "entry_placed"},
            {"event_type": "lifecycle", "run_id": "r", "event": "entry_filled"},
        ],
    )

    state = proof_watch._state(journal, "")

    assert state.pending_entries == 1
    assert state.open_paper_positions == 1


def test_print_state_json_contains_money_and_exposure_fields(capsys):
    state = proof_watch.ProofState(
        run_id="r",
        rows=1,
        candidates=0,
        decisions=0,
        allowed=0,
        placed=0,
        filled=0,
        closed=0,
        pending_entries=0,
        open_paper_positions=0,
        active_positions=0,
        net_pnl=0.0,
        avg_r=0.0,
        session="asia",
        session_allowed=False,
        next_allowed_utc=datetime(2026, 5, 12, 13, 0, tzinfo=timezone.utc),
        runtime_paused=True,
        runtime_pause_reason="edge_check_pending",
    )

    proof_watch._print_state_json("label", state)
    payload = json.loads(capsys.readouterr().out)

    assert payload["run_id"] == "r"
    assert payload["net_pnl"] == 0.0
    assert payload["active_positions"] == 0
    assert payload["open_paper_positions"] == 0
    assert payload["runtime_paused"] is True
    assert payload["runtime_pause_reason"] == "edge_check_pending"
    assert payload["next_allowed_utc"] == "2026-05-12 13:00:00"


def test_once_can_require_positive_pnl(monkeypatch, tmp_path):
    monkeypatch.setenv("BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    journal = tmp_path / "signals.jsonl"
    _write_rows(
        journal,
        [
            {"event_type": "run", "event": "run_start", "run_id": "r"},
            {"event_type": "outcome", "run_id": "r", "signal_id": "s", "pnl": 0.0, "r_multiple": 0.0},
        ],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "proof_watch.py",
            "--input",
            str(journal),
            "--once",
            "--require-positive-pnl",
        ],
    )

    assert proof_watch.main() == 1
