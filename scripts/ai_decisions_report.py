#!/usr/bin/env python3
"""Cross-reference AI decisions with trade outcomes.

Reads runtime/signals.jsonl, pairs each ai_decision row with its outcome
(via signal_id), and reports which prompt patterns / models / sessions /
coins produced winners.

USAGE
-----
    python scripts/ai_decisions_report.py
    python scripts/ai_decisions_report.py --input runtime/signals.jsonl --since 2026-05-23
    python scripts/ai_decisions_report.py --by reasoning_word --min-trades 3
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _iter_rows(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _ts_filter(since_ms: int | None, until_ms: int | None):
    def fn(row: dict[str, Any]) -> bool:
        ts = int(row.get("ts_ms") or 0)
        if since_ms is not None and ts < since_ms:
            return False
        if until_ms is not None and ts > until_ms:
            return False
        return True
    return fn


def _parse_date(s: str | None) -> int | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        print(f"warn: invalid date {s!r}, expected YYYY-MM-DD", file=sys.stderr)
        return None


def _summary(rs: list[float]) -> tuple[int, float, float, float]:
    if not rs:
        return 0, 0.0, 0.0, 0.0
    avg = sum(rs) / len(rs)
    wins = sum(1 for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    pf = (sum(r for r in rs if r > 0) / losses) if losses > 0 else (math.inf if any(r > 0 for r in rs) else 0.0)
    return len(rs), avg, wins / len(rs) * 100.0, pf


def _bucket_key_for(decision: dict[str, Any], outcome: dict[str, Any], by: str) -> str | None:
    """Return the grouping key for one decision based on --by selector."""
    if by == "action":
        return decision.get("action") or "unknown"
    if by == "model":
        return decision.get("model") or "unknown"
    if by == "coin":
        return outcome.get("coin") or decision.get("coin") or "unknown"
    if by == "session":
        return outcome.get("session") or ""
    if by == "prompt_version":
        return decision.get("prompt_version") or "unknown"
    if by == "confidence_bucket":
        c = float(decision.get("confidence") or 0)
        if c < 0.3: return "0.0-0.3"
        if c < 0.5: return "0.3-0.5"
        if c < 0.7: return "0.5-0.7"
        if c < 0.85: return "0.7-0.85"
        return "0.85-1.0"
    if by == "reasoning_word":
        # Bucket by salient words in reasoning — useful for prompt iteration
        text = str(decision.get("reasoning") or "").lower()
        # keep only words ≥ 5 chars that aren't stopwordy
        words = re.findall(r"\b[a-z]{5,}\b", text)
        return ",".join(sorted(set(words[:3]))) if words else "(no_reasoning)"
    return "all"


def main() -> int:
    parser = argparse.ArgumentParser(description="AI decisions vs outcomes report.")
    parser.add_argument("--input", default="runtime/signals.jsonl")
    parser.add_argument("--since", help="ISO date e.g. 2026-05-23 (inclusive)")
    parser.add_argument("--until", help="ISO date (exclusive)")
    parser.add_argument(
        "--by",
        choices=["action", "model", "coin", "session", "prompt_version",
                 "confidence_bucket", "reasoning_word", "all"],
        default="action",
    )
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    in_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    if not in_path.exists():
        print(f"error: {in_path} does not exist", file=sys.stderr)
        return 2

    since_ms = _parse_date(args.since)
    until_ms = _parse_date(args.until)
    ts_ok = _ts_filter(since_ms, until_ms)

    decisions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    decision_count = 0
    action_tally: Counter = Counter()
    cost_total = 0.0
    error_count = 0
    skip_count = 0

    for row in _iter_rows(in_path):
        if not ts_ok(row):
            continue
        et = row.get("event_type")
        sid = str(row.get("signal_id") or "")
        if et == "ai_decision":
            decision_count += 1
            action_tally[str(row.get("action") or "")] += 1
            cost_total += float(row.get("cost_usd") or 0)
            if row.get("error"):
                error_count += 1
            if row.get("skip_reason"):
                skip_count += 1
            decisions[sid] = row
        elif et == "outcome" and sid:
            outcomes[sid] = row

    if not decisions:
        print("No ai_decision rows in journal.")
        return 0

    print(f"AI Decisions Report ({in_path})")
    print("=" * 60)
    print(f"Decisions: {decision_count}  errors: {error_count}  skipped: {skip_count}")
    print(f"Total cost: ${cost_total:.4f}")
    print()
    print("Action breakdown:")
    for action, n in action_tally.most_common():
        print(f"  {action:25s} {n}")
    print()

    # Pair decisions with outcomes
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sid, d in decisions.items():
        o = outcomes.get(sid)
        if o is None:
            continue
        paired.append((d, o))

    if not paired:
        print("No paired decisions+outcomes yet — wait for more trades to close.")
        return 0

    rs_all = [float(o.get("r_multiple") or 0) for _, o in paired]
    n, avg, wr, pf = _summary(rs_all)
    pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
    print(f"Overall outcomes ({n} paired): avg_r={avg:+.3f} win_rate={wr:.1f}% pf={pf_text}")
    print()

    # Group by selector
    buckets: dict[str, list[float]] = defaultdict(list)
    for d, o in paired:
        key = _bucket_key_for(d, o, args.by) or "(none)"
        buckets[key].append(float(o.get("r_multiple") or 0))

    rows = []
    for key, rs in buckets.items():
        n, avg, wr, pf = _summary(rs)
        if n < args.min_trades:
            continue
        rows.append((key, n, avg, wr, pf))
    rows.sort(key=lambda r: -r[2])

    print(f"Grouped by {args.by} (min_trades={args.min_trades}):")
    print(f"  {'key':<40s} {'n':>4s} {'avg_r':>8s} {'win%':>6s} {'pf':>7s}")
    for key, n, avg, wr, pf in rows[: args.top_n]:
        pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
        print(f"  {key[:40]:<40s} {n:>4d} {avg:>+8.3f} {wr:>5.1f}% {pf_text:>7s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
