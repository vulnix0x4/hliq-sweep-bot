#!/usr/bin/env python3
"""Summarize paper collection and live-readiness blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    return proc.returncode, proc.stdout


def _proof_state() -> tuple[int, dict[str, object]]:
    rc, out = _run([
        sys.executable,
        str(ROOT / "scripts" / "proof_watch.py"),
        "--once",
        "--active-policy-runs",
        "--json",
    ])
    try:
        return rc, json.loads(out.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return rc or 1, {"error": out.strip() or "proof_watch returned no JSON"}


def _go_live_output() -> tuple[int, str, list[str]]:
    rc, out = _run([
        sys.executable,
        str(ROOT / "scripts" / "go_live_check.py"),
        "--active-policy-runs",
        "--min-candle-session-trades",
        "2",
        "--min-candle-coin-trades",
        "5",
        "--min-candle-level-trades",
        "2",
    ])
    failures = [line for line in out.splitlines() if line.startswith("[FAIL]")]
    return rc, out, failures


def _as_int(state: dict[str, object], key: str) -> int:
    try:
        return int(state.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _as_float(state: dict[str, object], key: str) -> float:
    try:
        return float(state.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Show whether paper collection is healthy and live is allowed.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    proof_rc, state = _proof_state()
    go_live_rc, _go_live_text, failures = _go_live_output()
    unsafe_exposure = _as_int(state, "pending_entries") > 0 or _as_int(state, "open_paper_positions") > 0 or _as_int(state, "active_position_files") > 0
    paused = bool(state.get("runtime_paused", False))
    paper_collecting = proof_rc == 0 and not unsafe_exposure and not paused
    live_ready = go_live_rc == 0
    payload = {
        "paper_collecting": paper_collecting,
        "live_ready": live_ready,
        "runtime_paused": paused,
        "session": state.get("session", ""),
        "session_allowed": bool(state.get("session_allowed", False)),
        "candidates": _as_int(state, "candidates"),
        "allowed": _as_int(state, "allowed"),
        "placed": _as_int(state, "placed"),
        "filled": _as_int(state, "filled"),
        "closed": _as_int(state, "closed"),
        "paper_pnl": _as_float(state, "net_pnl"),
        "avg_r": _as_float(state, "avg_r"),
        "pending_entries": _as_int(state, "pending_entries"),
        "open_paper_positions": _as_int(state, "open_paper_positions"),
        "active_position_files": _as_int(state, "active_position_files"),
        "live_blockers": failures,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("Trade Readiness")
        print("=" * 40)
        print(f"paper_collecting: {'yes' if paper_collecting else 'no'}")
        print(f"live_ready: {'yes' if live_ready else 'no'}")
        print(f"session: {payload['session']} allowed={payload['session_allowed']}")
        print(
            "funnel: "
            f"candidates={payload['candidates']} allowed={payload['allowed']} "
            f"placed={payload['placed']} filled={payload['filled']} closed={payload['closed']}"
        )
        print(f"paper_pnl: {payload['paper_pnl']:.2f} avg_r={payload['avg_r']:.3f}")
        print(
            "exposure: "
            f"pending={payload['pending_entries']} open_paper={payload['open_paper_positions']} "
            f"active_files={payload['active_position_files']}"
        )
        if failures:
            print("live_blockers:")
            for failure in failures:
                print(f"  {failure}")
    return 0 if paper_collecting else 1


if __name__ == "__main__":
    raise SystemExit(main())
