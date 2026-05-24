#!/usr/bin/env python3
"""Validate whether the latest blocked signal belongs in the operator policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from hliq_bot.config import load_config
    from scripts.candle_backtest import _level_family
    from scripts.go_live_check import _active_policy_run_ids
    from scripts.proof_watch import _load_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from hliq_bot.config import load_config
    from candle_backtest import _level_family  # type: ignore
    from go_live_check import _active_policy_run_ids  # type: ignore
    from proof_watch import _load_env  # type: ignore


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _blocked_candidates(rows: list[dict[str, Any]], *, active_policy_only: bool) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if active_policy_only:
        run_ids = _active_policy_run_ids(rows, load_config())
        rows = [row for row in rows if str(row.get("run_id", "")).strip() in run_ids]
    candidates: dict[str, dict[str, Any]] = {}
    blocked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        signal_id = str(row.get("signal_id", "")).strip()
        if not signal_id:
            continue
        if row.get("event_type") == "candidate":
            candidates[signal_id] = row
            continue
        if row.get("event_type") != "decision" or bool(row.get("allowed", False)):
            continue
        candidate = candidates.get(signal_id)
        if candidate is not None:
            blocked.append((candidate, row))
    return blocked


def _candidate_assignments(candidate: dict[str, Any]) -> list[str]:
    coin = str(candidate["coin"]).upper()
    session = str(candidate["session"]).lower()
    side = str(candidate["side"]).lower()
    level = _level_family(str(candidate["level_label"]).lower())
    return [
        f"HL_COINS={coin}",
        f"BOT_ALLOW_COINS={coin}",
        f"BOT_ALLOW_LEVEL_LABELS={level}",
        f"BOT_ALLOW_COIN_LEVELS={coin}:{level}",
        f"BOT_ALLOW_COIN_SESSIONS={coin}:{session}",
        f"BOT_ALLOW_COIN_SESSION_LEVELS={coin}:{session}:{level}",
        f"BOT_ALLOW_SESSIONS={session}",
        f"BOT_ALLOW_SIDES={side}",
    ]


def _print_signal(candidate: dict[str, Any], decision: dict[str, Any], assignments: list[str]) -> None:
    print("Rejected Signal Check")
    print("=" * 40)
    print(f"signal_id: {candidate.get('signal_id', '')}")
    print(f"run_id: {candidate.get('run_id', '')}")
    print(f"coin: {candidate.get('coin', '')}")
    print(f"session: {candidate.get('session', '')}")
    print(f"side: {candidate.get('side', '')}")
    print(f"level: {candidate.get('level_label', '')}")
    print(f"reject_reason: {decision.get('reason', '')}")
    print("candidate policy:")
    for item in assignments:
        print(f"  {item}")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict policy validation for the latest blocked signal slice.")
    parser.add_argument("--input", default="runtime/signals.jsonl")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Search all journal rows instead of rows matching the active operator policy.",
    )
    args = parser.parse_args()

    _load_env(ROOT / ".env")
    rows = _read_rows(ROOT / args.input)
    blocked = _blocked_candidates(rows, active_policy_only=not args.all_runs)
    if not blocked:
        print("No blocked signals found for the selected run set.")
        return 1
    candidate, decision = blocked[-1]
    assignments = _candidate_assignments(candidate)
    _print_signal(candidate, decision, assignments)
    cmd = [sys.executable, str(ROOT / "scripts" / "policy_candidate_check.py"), "--clear-policy"]
    for assignment in assignments:
        cmd.extend(["--set", assignment])
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
