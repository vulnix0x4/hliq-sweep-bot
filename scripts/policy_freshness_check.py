#!/usr/bin/env python3
"""Fail closed when the active policy no longer has recent candle edge."""

from __future__ import annotations

import argparse
import fnmatch
import math
import os
from pathlib import Path
import subprocess
import sys

try:
    from scripts.go_live_check import (
        Check,
        _candle_coin_session_sample_check,
        _candle_coin_session_level_sample_check,
        _parse_candle_coin_totals,
        _parse_candle_slices,
        _parse_candle_total,
        _weighted_slice_stats,
        _load_env_file,
    )
    from hliq_bot.config import load_config
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from go_live_check import (
        Check,
        _candle_coin_session_sample_check,
        _candle_coin_session_level_sample_check,
        _parse_candle_coin_totals,
        _parse_candle_slices,
        _parse_candle_total,
        _weighted_slice_stats,
        _load_env_file,
    )
    from hliq_bot.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _candle_timeout_sec() -> int:
    raw = os.getenv("BOT_EDGE_CANDLE_TIMEOUT_SEC", "180")
    try:
        return max(30, int(raw))
    except ValueError:
        return 180


def _run_candle_backtest(days: int) -> tuple[str, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "candle_backtest.py"),
        "--days",
        str(days),
        "--slice-limit",
        "0",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=_candle_timeout_sec(),
        )
    except Exception as exc:
        return "", f"candle backtest error: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        msg = detail[-1] if detail else f"exit={proc.returncode}"
        return "", f"candle backtest failed: {msg}"
    return proc.stdout, ""


def _edge_check_from_output(
    output: str,
    *,
    days: int,
    min_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
    min_span_days: float,
) -> Check:
    total = _parse_candle_total(output)
    if not total:
        return Check("recent_candle_edge", False, "candle backtest did not report TOTAL")
    trades = int(total.get("trades", 0))
    avg_r = float(total.get("avg_r", 0.0))
    pf = float(total.get("profit_factor", 0.0))
    span_days = float(total.get("span_days", 0.0))
    ok = trades >= min_trades and avg_r >= min_avg_r and pf >= min_profit_factor and span_days >= min_span_days
    pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
    return Check(
        "recent_candle_edge",
        ok,
        (
            f"days={days} trades={trades} min_trades={min_trades} "
            f"avg_r={avg_r:.3f} min_avg_r={min_avg_r:.3f} "
            f"profit_factor={pf_text} min_profit_factor={min_profit_factor:.2f} "
            f"span_days={span_days:.1f} min_span_days={min_span_days:.1f}"
        ),
    )


def _session_check_from_output(output: str, *, min_session_trades: int) -> Check:
    slices = _parse_candle_slices(output)
    by_session: dict[str, int] = {}
    for item in slices:
        session = str(item.get("session", ""))
        trades = int(item.get("trades", 0))
        by_session[session] = by_session.get(session, 0) + trades
    active_sessions = load_config().strategy.allowed_sessions
    weak = {
        session: by_session.get(session, 0)
        for session in sorted(active_sessions)
        if by_session.get(session, 0) < min_session_trades
    }
    detail = " ".join(f"{session}={by_session.get(session, 0)}" for session in sorted(active_sessions))
    return Check(
        "candle_session_samples",
        not weak,
        f"{detail or 'no_allowed_sessions'} min_session_trades={min_session_trades}",
    )


def _coin_check_from_output(
    output: str,
    *,
    min_coin_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
) -> Check:
    allowed_coins = sorted(load_config().strategy.allowed_coins)
    if not allowed_coins:
        return Check("candle_coin_samples", False, "no_allowed_coins")
    totals = _parse_candle_coin_totals(output)
    weak: list[str] = []
    details: list[str] = []
    for coin in allowed_coins:
        total = totals.get(coin.upper(), {})
        trades = int(total.get("trades", 0))
        avg_r = float(total.get("avg_r", 0.0))
        pf = float(total.get("profit_factor", 0.0))
        pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
        details.append(f"{coin}=trades:{trades},avg_r:{avg_r:.3f},pf:{pf_text}")
        if trades < min_coin_trades or avg_r < min_avg_r or pf < min_profit_factor:
            weak.append(coin)
    return Check(
        "candle_coin_samples",
        not weak,
        (
            " ".join(details)
            + f" min_coin_trades={min_coin_trades} min_avg_r={min_avg_r:.3f} "
            + f"min_profit_factor={min_profit_factor:.2f}"
        ),
    )


def _level_check_from_output(
    output: str,
    *,
    min_level_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
) -> Check:
    allowed_levels = sorted(load_config().strategy.allowed_level_labels)
    if not allowed_levels:
        return Check("candle_level_samples", False, "no_allowed_levels")
    slices = _parse_candle_slices(output)
    weak: list[str] = []
    details: list[str] = []
    for pattern in allowed_levels:
        matches = [item for item in slices if fnmatch.fnmatchcase(str(item.get("level", "")), pattern)]
        trades, avg_r, min_pf = _weighted_slice_stats(matches)
        pf_text = "inf" if math.isinf(min_pf) else f"{min_pf:.2f}"
        details.append(f"{pattern}=trades:{trades},avg_r:{avg_r:.3f},min_pf:{pf_text}")
        if trades < min_level_trades or avg_r < min_avg_r or min_pf < min_profit_factor:
            weak.append(pattern)
    return Check(
        "candle_level_samples",
        not weak,
        (
            " ".join(details)
            + f" min_level_trades={min_level_trades} min_avg_r={min_avg_r:.3f} "
            + f"min_profit_factor={min_profit_factor:.2f}"
        ),
    )


def _coin_level_check_from_output(
    output: str,
    *,
    min_pair_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
) -> Check:
    pairs = sorted(load_config().strategy.allowed_coin_level_pairs)
    if not pairs:
        return Check("candle_coin_level_samples", True, "no_allowed_coin_levels")
    slices = _parse_candle_slices(output)
    weak: list[str] = []
    details: list[str] = []
    for pair in pairs:
        coin, level_pattern = pair.split(":", 1)
        matches = [
            item for item in slices
            if str(item.get("coin", "")).upper() == coin
            and fnmatch.fnmatchcase(str(item.get("level", "")), level_pattern)
        ]
        trades, avg_r, min_pf = _weighted_slice_stats(matches)
        pf_text = "inf" if math.isinf(min_pf) else f"{min_pf:.2f}"
        details.append(f"{pair}=trades:{trades},avg_r:{avg_r:.3f},min_pf:{pf_text}")
        if trades < min_pair_trades or avg_r < min_avg_r or min_pf < min_profit_factor:
            weak.append(pair)
    return Check(
        "candle_coin_level_samples",
        not weak,
        (
            " ".join(details)
            + f" min_pair_trades={min_pair_trades} min_avg_r={min_avg_r:.3f} "
            + f"min_profit_factor={min_profit_factor:.2f}"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-avg-r", type=float, default=0.05)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--min-span-days", type=float, default=3.5)
    parser.add_argument("--min-session-trades", type=int, default=2)
    parser.add_argument("--min-coin-trades", type=int, default=5)
    parser.add_argument("--min-level-trades", type=int, default=2)
    args = parser.parse_args()

    _load_env_file(ROOT / args.env_file)
    days = max(1, args.days)
    output, error = _run_candle_backtest(days)
    if error:
        checks = [
            Check("recent_candle_edge", False, error),
            Check("candle_session_samples", False, error),
            Check("candle_coin_samples", False, error),
            Check("candle_level_samples", False, error),
            Check("candle_coin_level_samples", False, error),
            Check("candle_coin_session_samples", False, error),
            Check("candle_coin_session_level_samples", False, error),
        ]
    else:
        checks = [
            _edge_check_from_output(
                output,
            days=days,
            min_trades=max(1, args.min_trades),
            min_avg_r=args.min_avg_r,
            min_profit_factor=args.min_profit_factor,
            min_span_days=max(0.0, args.min_span_days),
            ),
            _session_check_from_output(
                output,
            min_session_trades=max(1, args.min_session_trades),
            ),
            _coin_check_from_output(
                output,
            min_coin_trades=max(1, args.min_coin_trades),
            min_avg_r=args.min_avg_r,
            min_profit_factor=args.min_profit_factor,
            ),
            _level_check_from_output(
                output,
            min_level_trades=max(1, args.min_level_trades),
            min_avg_r=args.min_avg_r,
            min_profit_factor=args.min_profit_factor,
            ),
            _coin_level_check_from_output(
                output,
            min_pair_trades=max(1, args.min_level_trades),
            min_avg_r=args.min_avg_r,
            min_profit_factor=args.min_profit_factor,
            ),
            _candle_coin_session_sample_check(
                days=days,
                min_pair_trades=max(1, args.min_session_trades),
                min_avg_r=args.min_avg_r,
                min_profit_factor=args.min_profit_factor,
                output=output,
            ),
            _candle_coin_session_level_sample_check(
                days=days,
                min_triple_trades=max(1, args.min_level_trades),
                min_avg_r=args.min_avg_r,
                min_profit_factor=args.min_profit_factor,
                output=output,
            ),
        ]
    ok = all(check.ok for check in checks)
    print("Policy Freshness Check")
    print("=" * 40)
    print(f"status: {'PASS' if ok else 'FAIL'}")
    for check in checks:
        print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
