#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hliq_bot.config import load_config  # noqa: E402

try:
    from scripts.go_live_check import _active_policy_run_ids  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from go_live_check import _active_policy_run_ids  # type: ignore  # noqa: E402


@dataclass(slots=True)
class ProofState:
    run_id: str
    rows: int
    candidates: int
    decisions: int
    allowed: int
    placed: int
    filled: int
    closed: int
    pending_entries: int
    open_paper_positions: int
    active_positions: int
    net_pnl: float
    avg_r: float
    session: str
    session_allowed: bool
    next_allowed_utc: datetime | None
    runtime_paused: bool
    runtime_pause_reason: str


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _last_run_id(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        if row.get("event_type") == "run" and row.get("event") == "run_start":
            run_id = str(row.get("run_id", "")).strip()
            if run_id:
                return run_id
    return ""


def _session_from_hour(hour: int) -> str:
    if hour < 7:
        return "asia"
    if hour < 13:
        return "eu"
    if hour < 22:
        return "us"
    return "late"


def _next_allowed(now: datetime, allowed: set[str]) -> datetime | None:
    if not allowed:
        return now
    probe = now.replace(minute=0, second=0, microsecond=0)
    if probe < now:
        probe += timedelta(hours=1)
    for i in range(0, 48):
        candidate = probe + timedelta(hours=i)
        if _session_from_hour(candidate.hour) in allowed:
            return candidate
    return None


def _active_position_files(runtime_dir: str) -> list[Path]:
    path = (ROOT / runtime_dir / "active_positions").resolve()
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file())


def _resolve_runtime_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _runtime_pause_state(raw_path: str) -> tuple[bool, str]:
    path = _resolve_runtime_path(raw_path)
    if not path.exists():
        return False, ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return True, "unreadable_pause_file"
    reason = lines[0].strip() if lines else "operator_pause"
    return True, reason or "operator_pause"


def _state(path: Path, run_id: str, *, active_policy_runs: bool = False) -> ProofState:
    rows = _read_rows(path)
    cfg = load_config()
    selected_run_id = run_id.strip()
    if active_policy_runs:
        run_ids = _active_policy_run_ids(rows, cfg)
        selected_run_id = f"active_policy({len(run_ids)} runs)"
        rows = [r for r in rows if str(r.get("run_id", "")).strip() in run_ids]
    else:
        selected_run_id = selected_run_id or _last_run_id(rows)
        if selected_run_id:
            rows = [r for r in rows if str(r.get("run_id", "")).strip() == selected_run_id]

    lifecycle = Counter()
    by_id: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        sid = str(row.get("signal_id", "")).strip()
        event_type = str(row.get("event_type", "")).strip()
        if event_type == "lifecycle":
            lifecycle[str(row.get("event", "unknown"))] += 1
        if not sid:
            continue
        if event_type == "candidate":
            by_id[sid]["candidate"] = row
        elif event_type == "decision":
            by_id[sid]["decision"] = row
        elif event_type == "outcome":
            by_id[sid]["outcome"] = row

    candidates = 0
    decisions = 0
    allowed = 0
    closed = 0
    net_pnl = 0.0
    r_sum = 0.0
    for rec in by_id.values():
        if "candidate" in rec:
            candidates += 1
        dec = rec.get("decision")
        if dec is not None:
            decisions += 1
            if bool(dec.get("allowed", False)):
                allowed += 1
        out = rec.get("outcome")
        if out is not None:
            closed += 1
            try:
                net_pnl += float(out.get("pnl", 0.0))
            except (TypeError, ValueError):
                pass
            try:
                r_sum += float(out.get("r_multiple", 0.0))
            except (TypeError, ValueError):
                pass

    active_positions = _active_position_files(cfg.runtime.runtime_dir)
    runtime_paused, runtime_pause_reason = _runtime_pause_state(cfg.runtime.trade_pause_path)
    now = datetime.now(timezone.utc)
    session = _session_from_hour(now.hour)
    session_allowed = (not cfg.strategy.allowed_sessions) or (session in cfg.strategy.allowed_sessions)
    return ProofState(
        run_id=selected_run_id,
        rows=len(rows),
        candidates=candidates,
        decisions=decisions,
        allowed=allowed,
        placed=lifecycle.get("entry_placed", 0),
        filled=lifecycle.get("entry_filled", 0),
        closed=closed,
        pending_entries=max(lifecycle.get("entry_placed", 0) - lifecycle.get("entry_filled", 0), 0),
        open_paper_positions=max(lifecycle.get("entry_filled", 0) - closed, 0),
        active_positions=len(active_positions),
        net_pnl=net_pnl,
        avg_r=(r_sum / closed) if closed else 0.0,
        session=session,
        session_allowed=session_allowed,
        next_allowed_utc=_next_allowed(now, cfg.strategy.allowed_sessions),
        runtime_paused=runtime_paused,
        runtime_pause_reason=runtime_pause_reason,
    )


def _print_state(label: str, state: ProofState) -> None:
    print(f"{label}: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"run_id: {state.run_id or 'n/a'}")
    print(
        "funnel: "
        f"candidates={state.candidates} decisions={state.decisions} allowed={state.allowed} "
        f"placed={state.placed} filled={state.filled} closed={state.closed}"
    )
    print(f"paper_pnl: {state.net_pnl:.2f} avg_r: {state.avg_r:.3f}")
    has_exposure = state.active_positions > 0 or state.open_paper_positions > 0
    print(
        f"exposure: {'open' if has_exposure else 'flat'} "
        f"pending_entries={state.pending_entries} "
        f"open_paper_positions={state.open_paper_positions} "
        f"active_position_files={state.active_positions}"
    )
    print(f"session: {state.session} allowed_now={state.session_allowed}")
    if state.runtime_paused:
        print(f"runtime_pause: true reason={state.runtime_pause_reason}")
    else:
        print("runtime_pause: false")
    if state.next_allowed_utc and not state.session_allowed:
        hours = (state.next_allowed_utc - datetime.now(timezone.utc)).total_seconds() / 3600.0
        print(
            "next_allowed_utc: "
            f"{state.next_allowed_utc.strftime('%Y-%m-%d %H:%M:%S')} "
            f"({max(hours, 0.0):.2f}h)"
        )
        print(f"next_allowed_local: {state.next_allowed_utc.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()


def _print_state_json(label: str, state: ProofState) -> None:
    payload = dataclasses.asdict(state)
    payload["label"] = label
    payload["observed_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if state.next_allowed_utc is not None:
        payload["next_allowed_utc"] = state.next_allowed_utc.strftime("%Y-%m-%d %H:%M:%S")
        payload["next_allowed_local"] = state.next_allowed_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(json.dumps(payload, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded watch for fresh paper-trade proof in the current run."
    )
    parser.add_argument("--input", default="runtime/signals.jsonl")
    parser.add_argument("--run-id", default="", help="Watch a specific run_id; default is latest.")
    parser.add_argument(
        "--active-policy-runs",
        action="store_true",
        help="Watch all runs whose run_start policy exactly matches the active .env policy.",
    )
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument(
        "--exit-on",
        choices=("any", "entry", "close"),
        default="any",
        help="Event that ends the watch successfully. Default any preserves legacy entry-or-close behavior.",
    )
    parser.add_argument(
        "--require-profit",
        action="store_true",
        help="With --exit-on close, only succeed when net paper PnL increased above the start value.",
    )
    parser.add_argument("--once", action="store_true", help="Print the current proof state and exit.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON status.")
    parser.add_argument(
        "--require-positive-pnl",
        action="store_true",
        help="Exit nonzero unless the selected run has positive net paper PnL.",
    )
    args = parser.parse_args()

    _load_env(ROOT / ".env")
    path = Path(args.input)
    start = _state(path, args.run_id, active_policy_runs=args.active_policy_runs)
    printer = _print_state_json if args.json else _print_state
    printer("Proof Watch Start", start)
    if args.once:
        return 0 if (not args.require_positive_pnl or start.net_pnl > 0.0) else 1

    deadline = time.monotonic() + max(1, args.timeout_sec)
    poll_sec = max(1, args.poll_sec)
    last = start
    while time.monotonic() < deadline:
        time.sleep(min(poll_sec, max(0.0, deadline - time.monotonic())))
        current = _state(
            path,
            "" if args.active_policy_runs else (args.run_id or start.run_id),
            active_policy_runs=args.active_policy_runs,
        )
        if (
            current.placed != last.placed
            or current.filled != last.filled
            or current.closed != last.closed
            or current.session_allowed != last.session_allowed
        ):
            printer("Proof Watch Update", current)
            last = current
        entry_seen = current.placed > start.placed
        close_seen = current.closed > start.closed
        profit_seen = current.net_pnl > start.net_pnl
        if args.exit_on in {"any", "close"} and close_seen:
            if not args.require_profit or profit_seen:
                return 0
        if args.exit_on in {"any", "entry"} and entry_seen:
            return 0

    final = _state(
        path,
        "" if args.active_policy_runs else (args.run_id or start.run_id),
        active_policy_runs=args.active_policy_runs,
    )
    printer("Proof Watch Timeout", final)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
