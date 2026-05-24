#!/usr/bin/env python3
"""Wait for an allowed strategy session, then watch for profitable paper proof."""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from hliq_bot.config import load_config
    from scripts.session_status import _load_env, _next_allowed, _session_from_hour
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    SRC = Path(__file__).resolve().parents[1] / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from hliq_bot.config import load_config
    from session_status import _load_env, _next_allowed, _session_from_hour


ROOT = Path(__file__).resolve().parents[1]


def _pause_path() -> Path:
    raw = os.environ.get("BOT_TRADE_PAUSE_PATH", "runtime/trade_pause.flag")
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


# Track our own writes so cleanup hooks only delete flags WE created (never an
# operator-set pause). Reset to None whenever we explicitly clear.
_OUR_PAUSE_REASON: str | None = None


def _set_pause(reason: str) -> None:
    global _OUR_PAUSE_REASON
    path = _pause_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{reason}\n", encoding="utf-8")
    _OUR_PAUSE_REASON = reason


def _clear_pause() -> None:
    global _OUR_PAUSE_REASON
    try:
        _pause_path().unlink()
    except FileNotFoundError:
        pass
    finally:
        _OUR_PAUSE_REASON = None


def _cleanup_on_exit() -> None:
    """Clear our own edge_check_pending flag on process exit so a crash doesn't
    silently halt the bot. We only clear if WE wrote 'edge_check_pending' —
    'edge_check_failed' and operator pauses are preserved."""
    if _OUR_PAUSE_REASON != "edge_check_pending":
        return
    path = _pause_path()
    if not path.exists():
        return
    try:
        first = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if first and first[0].strip() == "edge_check_pending":
            path.unlink()
    except OSError:
        pass


def _install_exit_handlers() -> None:
    atexit.register(_cleanup_on_exit)
    # SIGTERM is what docker sends on container stop; SIGINT is Ctrl-C.
    def _handler(signum, frame):  # type: ignore[no-untyped-def]
        _cleanup_on_exit()
        # Re-raise so the default handler can exit cleanly.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Non-main thread or platform without this signal; safe to ignore.
            pass


def _seconds_until_disallowed(now: datetime, sessions: set[str]) -> int:
    if not sessions or _session_from_hour(now.hour) not in sessions:
        return 1
    # Strategy sessions change only on UTC hour boundaries. Walk hour by hour
    # and stop at the first boundary whose session is no longer allowed.
    boundary = now.replace(minute=0, second=0, microsecond=0)
    if boundary <= now:
        boundary += timedelta(hours=1)
    for i in range(0, 48):
        candidate = boundary + timedelta(hours=i)
        if _session_from_hour(candidate.hour) not in sessions:
            return max(1, int((candidate - now).total_seconds()))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait until a strategy session opens, then run proof_watch for profit."
    )
    parser.add_argument(
        "--session",
        default="auto",
        help="Comma-separated strategy session(s) to wait for, or auto for BOT_ALLOW_SESSIONS.",
    )
    parser.add_argument("--max-wait-sec", type=int, default=86_400)
    parser.add_argument("--timeout-sec", type=int, default=28_800)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--input", default="runtime/signals.jsonl")
    parser.add_argument("--skip-edge-check", action="store_true")
    parser.add_argument("--edge-days", type=int, default=7)
    parser.add_argument("--edge-min-trades", type=int, default=5)
    parser.add_argument("--edge-min-avg-r", type=float, default=0.05)
    parser.add_argument("--edge-min-profit-factor", type=float, default=1.20)
    parser.add_argument("--edge-min-span-days", type=float, default=3.5)
    parser.add_argument("--edge-min-session-trades", type=int, default=2)
    parser.add_argument("--edge-min-coin-trades", type=int, default=5)
    parser.add_argument("--edge-min-level-trades", type=int, default=2)
    parser.add_argument(
        "--edge-retry-sec",
        type=int,
        default=300,
        help="When looping, wait this long before retrying a failed edge check inside an allowed session.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep watching future allowed sessions until a profitable close is observed.",
    )
    args = parser.parse_args()

    _load_env(ROOT / ".env")
    _install_exit_handlers()
    session_arg = args.session.strip().lower()
    if session_arg == "auto":
        sessions = set(load_config().strategy.allowed_sessions)
    else:
        sessions = {s.strip().lower() for s in session_arg.split(",") if s.strip()}
    if not sessions:
        print("session must not be empty; set --session or BOT_ALLOW_SESSIONS", file=sys.stderr)
        return 2

    session_label = ",".join(sorted(s.upper() for s in sessions))
    while True:
        now = datetime.now(timezone.utc)
        next_allowed = _next_allowed(now, sessions)
        wait_sec = max(1, int((next_allowed - now).total_seconds())) if next_allowed else 1
        wait_until = next_allowed.strftime("%Y-%m-%d %H:%M:%S UTC") if next_allowed else "unknown"
        print(
            f"Waiting {wait_sec}s until next {session_label} session ({wait_until}), "
            "then watching for profitable close.",
            flush=True,
        )
        if wait_sec > args.max_wait_sec:
            print(
                f"Refusing to sleep longer than --max-wait-sec={args.max_wait_sec}.",
                file=sys.stderr,
            )
            return 2
        if args.dry_run:
            return 0

        if not args.skip_edge_check:
            _set_pause("edge_check_pending")
        time.sleep(wait_sec)
        if not args.skip_edge_check:
            _set_pause("edge_check_pending")
            rc = subprocess.call(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "policy_freshness_check.py"),
                    "--days",
                    str(args.edge_days),
                    "--min-trades",
                    str(args.edge_min_trades),
                    "--min-avg-r",
                    str(args.edge_min_avg_r),
                    "--min-profit-factor",
                    str(args.edge_min_profit_factor),
                    "--min-span-days",
                    str(args.edge_min_span_days),
                    "--min-session-trades",
                    str(args.edge_min_session_trades),
                    "--min-coin-trades",
                    str(args.edge_min_coin_trades),
                    "--min-level-trades",
                    str(args.edge_min_level_trades),
                ]
            )
            if rc != 0:
                _set_pause("edge_check_failed")
                if not args.loop:
                    return rc
                print(f"Policy freshness check exited {rc}; continuing loop.", flush=True)
                retry_sec = max(1, int(args.edge_retry_sec))
                if _session_from_hour(datetime.now(timezone.utc).hour) in sessions:
                    sleep_sec = min(
                        retry_sec,
                        _seconds_until_disallowed(datetime.now(timezone.utc), sessions),
                    )
                    print(f"Retrying policy freshness check in {sleep_sec}s.", flush=True)
                    time.sleep(sleep_sec)
                continue
            _clear_pause()
        watch_timeout_sec = min(args.timeout_sec, _seconds_until_disallowed(datetime.now(timezone.utc), sessions))
        rc = subprocess.call(
            [
                sys.executable,
                str(ROOT / "scripts" / "proof_watch.py"),
                "--input",
                args.input,
                "--active-policy-runs",
                "--exit-on",
                "close",
                "--require-profit",
                "--timeout-sec",
                str(watch_timeout_sec),
                "--poll-sec",
                str(args.poll_sec),
            ]
        )
        if rc != 0 and not args.skip_edge_check:
            _set_pause("edge_check_pending")
        if rc == 0 or not args.loop:
            return rc
        print(f"Proof watch exited {rc}; continuing loop.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
