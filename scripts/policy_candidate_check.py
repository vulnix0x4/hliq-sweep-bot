#!/usr/bin/env python3
"""Validate a proposed operator policy without editing .env."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

POLICY_KEYS = {
    "HL_COINS",
    "BOT_ALLOW_COINS",
    "BOT_ALLOW_LEVEL_LABELS",
    "BOT_ALLOW_COIN_LEVELS",
    "BOT_ALLOW_COIN_SESSIONS",
    "BOT_ALLOW_COIN_SESSION_LEVELS",
    "BOT_ALLOW_SESSIONS",
    "BOT_ALLOW_SIDES",
    "BOT_BLOCK_COINS",
    "BOT_BLOCK_LEVEL_LABELS",
    "BOT_BLOCK_COIN_LEVELS",
    "BOT_BLOCK_COIN_SESSIONS",
    "BOT_BLOCK_COIN_SESSION_LEVELS",
    "BOT_BLOCK_SESSIONS",
    "BOT_BLOCK_SIDES",
}


def _parse_assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"candidate assignment must be KEY=VALUE: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"candidate assignment has an empty key: {raw}")
    if not key.replace("_", "").isalnum() or key[0].isdigit():
        raise ValueError(f"candidate assignment has an invalid key: {key}")
    return key, value.strip().strip('"').strip("'")


def _candidate_env(base: dict[str, str], assignments: list[str], *, clear_policy: bool) -> dict[str, str]:
    env = dict(base)
    if clear_policy:
        for key in POLICY_KEYS:
            env[key] = ""
        env["BOT_BLOCK_SESSIONS"] = "late"
    for raw in assignments:
        key, value = _parse_assignment(raw)
        env[key] = value
    return env


def _print_candidate(assignments: list[str], *, clear_policy: bool) -> None:
    print("Policy Candidate Check")
    print("=" * 40)
    print(f"clear_policy: {clear_policy}")
    if not assignments:
        print("candidate: active .env policy")
        return
    print("candidate assignments:")
    for raw in assignments:
        key, value = _parse_assignment(raw)
        print(f"  {key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the policy freshness gate against proposed KEY=VALUE overrides.")
    parser.add_argument(
        "--set",
        dest="assignments",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Candidate environment override. Repeat for each policy key.",
    )
    parser.add_argument(
        "--clear-policy",
        action="store_true",
        help="Blank all allow/block policy keys before applying --set overrides.",
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-avg-r", type=float, default=0.10)
    parser.add_argument("--min-profit-factor", type=float, default=1.50)
    parser.add_argument("--min-span-days", type=float, default=3.5)
    parser.add_argument("--min-session-trades", type=int, default=5)
    parser.add_argument("--min-coin-trades", type=int, default=10)
    parser.add_argument("--min-level-trades", type=int, default=5)
    args = parser.parse_args()

    try:
        _print_candidate(args.assignments, clear_policy=args.clear_policy)
        env = _candidate_env(os.environ, args.assignments, clear_policy=args.clear_policy)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.flush()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "policy_freshness_check.py"),
        "--days",
        str(max(1, args.days)),
        "--min-trades",
        str(max(1, args.min_trades)),
        "--min-avg-r",
        str(args.min_avg_r),
        "--min-profit-factor",
        str(args.min_profit_factor),
        "--min-span-days",
        str(max(0.0, args.min_span_days)),
        "--min-session-trades",
        str(max(1, args.min_session_trades)),
        "--min-coin-trades",
        str(max(1, args.min_coin_trades)),
        "--min-level-trades",
        str(max(1, args.min_level_trades)),
    ]
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
