#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import median
import re
import sys

RE_CLOSED = re.compile(
    r"position closed \((?P<reason>[^)]+)\): pnl=(?P<pnl>-?\d+(?:\.\d+)?) r=(?P<r>-?\d+(?:\.\d+)?)(?: hold_sec=(?P<hold_sec>-?\d+(?:\.\d+)?))?"
)
RE_PLACED = re.compile(r"paper entry placed")
RE_FILLED = re.compile(r"paper entry filled")
RE_EXPIRED = re.compile(r"pending entry expired")
RE_PARTIAL = re.compile(r"tp1 partial")
RE_BLOCKED = re.compile(r"Signal blocked: (?P<reason>.+)")


def _line_iter(path: str | None):
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
        return
    for line in sys.stdin:
        yield line.rstrip("\n")


def _extract_day(line: str) -> str:
    if len(line) >= 10 and line[4] == "-" and line[7] == "-":
        return line[:10]
    return "unknown"


def _max_drawdown(pnls: list[float]) -> float:
    peak = 0.0
    eq = 0.0
    max_dd = 0.0
    for pnl in pnls:
        eq += pnl
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize paper bot performance from log lines.")
    parser.add_argument("path", nargs="?", help="Optional log file path. Reads stdin if omitted.")
    args = parser.parse_args()

    closed: list[tuple[float, float, str, str]] = []
    entries_placed = 0
    entries_filled = 0
    pending_expired = 0
    tp1_partial = 0
    blocked = Counter()
    daily_pnl: dict[str, float] = defaultdict(float)
    hold_secs: list[float] = []

    for line in _line_iter(args.path):
        day = _extract_day(line)

        if RE_PLACED.search(line):
            entries_placed += 1
        if RE_FILLED.search(line):
            entries_filled += 1
        if RE_EXPIRED.search(line):
            pending_expired += 1
        if RE_PARTIAL.search(line):
            tp1_partial += 1

        m_blocked = RE_BLOCKED.search(line)
        if m_blocked:
            blocked[m_blocked.group("reason").strip()] += 1

        m = RE_CLOSED.search(line)
        if not m:
            continue

        pnl = float(m.group("pnl"))
        r_val = float(m.group("r"))
        reason = m.group("reason").strip()
        closed.append((pnl, r_val, reason, day))
        daily_pnl[day] += pnl
        hold_sec = m.group("hold_sec")
        if hold_sec is not None:
            hold_secs.append(float(hold_sec))

    print("Paper Bot Stats")
    print("=" * 40)
    print(f"entries_placed: {entries_placed}")
    print(f"entries_filled: {entries_filled}")
    print(f"pending_expired: {pending_expired}")
    print(f"tp1_partial: {tp1_partial}")
    print(f"closed_trades: {len(closed)}")
    if entries_placed > 0:
        print(f"fill_rate: {(entries_filled / entries_placed) * 100.0:.1f}%")
        print(f"expiry_rate: {(pending_expired / entries_placed) * 100.0:.1f}%")

    if not closed:
        print("No closed trades found yet.")
        if blocked:
            print("\nSignal blocks:")
            for reason, n in blocked.most_common():
                print(f"- {reason}: {n}")
        return 0

    pnls = [x[0] for x in closed]
    r_vals = [x[1] for x in closed]
    reasons = Counter(x[2] for x in closed)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)

    net = sum(pnls)
    win_rate = (len(wins) / len(pnls)) * 100.0 if pnls else 0.0
    avg_pnl = net / len(pnls)
    avg_r = sum(r_vals) / len(r_vals)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    max_dd = _max_drawdown(pnls)

    print(f"net_pnl: {net:.2f}")
    print(f"win_rate: {win_rate:.1f}%")
    print(f"profit_factor: {pf:.2f}")
    print(f"avg_pnl_per_trade: {avg_pnl:.2f}")
    print(f"avg_r_per_trade: {avg_r:.3f}")
    print(f"max_drawdown_est: {max_dd:.2f}")
    if hold_secs:
        print(f"avg_hold_sec: {sum(hold_secs)/len(hold_secs):.1f}")
        print(f"median_hold_sec: {median(hold_secs):.1f}")
        print(f"max_hold_sec: {max(hold_secs):.1f}")

    print("\nExit reasons:")
    for reason, n in reasons.most_common():
        print(f"- {reason}: {n}")

    print("\nDaily pnl:")
    for day in sorted(daily_pnl):
        print(f"- {day}: {daily_pnl[day]:.2f}")

    if blocked:
        print("\nSignal blocks:")
        for reason, n in blocked.most_common():
            print(f"- {reason}: {n}")

    hints: list[str] = []
    if entries_placed >= 5 and entries_filled / max(entries_placed, 1) < 0.5:
        hints.append("Low fill rate: consider increasing BOT_ENTRY_TOUCH_TOL_BPS (e.g., 1 -> 2) or BOT_ENTRY_EXPIRY_SEC.")
    if len(closed) >= 4 and reasons.get("time_stop", 0) / len(closed) >= 0.5:
        hints.append("Many time_stop exits: consider slightly lower TP targets or a shorter BOT_MAX_HOLDING_SEC.")
    if len(closed) >= 4 and reasons.get("stop_loss", 0) / len(closed) >= 0.5 and pf < 1.0:
        hints.append("Stop-loss heavy and PF<1: tighten filters (higher BOT_VOLUME_SPIKE_MULT or lower BOT_MAX_TREND_MOVE_BPS).")
    if max_dd >= 3 * abs(avg_pnl) and len(closed) >= 4:
        hints.append("Drawdown is high versus expectancy: reduce BOT_RISK_PER_TRADE_PCT until edge stabilizes.")

    if hints:
        print("\nTuning hints:")
        for hint in hints:
            print(f"- {hint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
