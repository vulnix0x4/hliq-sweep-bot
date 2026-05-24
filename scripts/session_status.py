#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hliq_bot.config import load_config  # noqa: E402


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
    if _session_from_hour(now.hour) in allowed:
        return now
    probe = now.replace(minute=0, second=0, microsecond=0)
    if probe < now:
        probe += timedelta(hours=1)
    for i in range(0, 48):
        candidate = probe + timedelta(hours=i)
        if _session_from_hour(candidate.hour) in allowed:
            return candidate
    return None


def main() -> int:
    _load_env(ROOT / ".env")
    cfg = load_config()
    now = datetime.now(timezone.utc)
    session = _session_from_hour(now.hour)
    allowed = cfg.strategy.allowed_sessions
    is_allowed = (not allowed) or (session in allowed)
    next_allowed = _next_allowed(now, allowed)

    print("Session Status")
    print("=" * 40)
    print(f"utc_now: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"current_session: {session}")
    print(f"allowed_sessions: {sorted(allowed) if allowed else ['all']}")
    print(f"session_allowed_now: {is_allowed}")
    if next_allowed and not is_allowed:
        delta = next_allowed - now
        hours = delta.total_seconds() / 3600.0
        print(f"next_allowed_utc: {next_allowed.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"next_allowed_local: {next_allowed.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"next_allowed_in_hours: {hours:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
