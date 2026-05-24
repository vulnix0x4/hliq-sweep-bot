#!/usr/bin/env python3
"""Replay historical Hyperliquid candles through the AI trading strategy.

Used to iterate on prompts / models BEFORE deploying to live paper. Every
decision goes through the actual AIStrategy module + prompt, so what you see
here is what you'd see live (modulo OHLC vs tick simulation).

Costs real LLM money — defaults to a small window. Use --dry-run to estimate
the call count and cost without sending requests.

USAGE
-----
    # Estimate cost without spending money
    python scripts/ai_backtest.py --days 1 --coins HYPE --dry-run

    # Actual backtest (needs OPENROUTER_API_KEY)
    python scripts/ai_backtest.py --days 1 --coins HYPE,NEAR --interval-min 15

OUTPUT
------
Per-coin: number of decisions, action breakdown, simulated trades, total R.
A summary at the bottom + optional --json-out for machine consumption.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from hliq_bot.models import Bar, OpenPosition, Side  # noqa: E402
from hliq_bot.signal.session_tracker import SessionTracker  # noqa: E402
from hliq_bot.signal.vwap_tracker import VWAPTracker  # noqa: E402


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


def _bar_from_row(row: dict[str, Any]) -> Bar | None:
    try:
        start_ms = int(row["t"])
        end_ms = int(row["T"])
        open_px = float(row["o"])
        high = float(row["h"])
        low = float(row["l"])
        close = float(row["c"])
        volume = float(row.get("v", 0.0))
        trades = int(row.get("n", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if open_px <= 0 or high < low or close <= 0:
        return None
    return Bar(
        start_ms=start_ms, end_ms=end_ms,
        open=open_px, high=high, low=low, close=close,
        volume=max(0.0, volume), trade_count=max(0, trades),
        vwap=close, avg_spread_bps=1.0,
    )


def _fetch_candles(info: Any, coin: str, start_ms: int, end_ms: int) -> list[Bar]:
    """Pull 1m candles in 12h chunks (HL API limit)."""
    out: dict[int, dict[str, Any]] = {}
    chunk = 12 * 60 * 60 * 1000
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(end_ms, cursor + chunk)
        try:
            rows = info.candles_snapshot(coin, "1m", cursor, chunk_end)
        except Exception as exc:
            print(f"warn: candles fetch failed for {coin} @ {cursor}: {exc}", file=sys.stderr)
            rows = []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    try:
                        out[int(row["t"])] = row
                    except (KeyError, TypeError, ValueError):
                        continue
        cursor = chunk_end + 1
    bars = [b for k in sorted(out) if (b := _bar_from_row(out[k])) is not None]
    return bars


class _SimWorker:
    """Minimal worker stub matching the duck-typed surface AIStrategy expects."""

    def __init__(self, coin: str) -> None:
        from collections import deque
        self.coin = coin
        self.last_spread_bps = 1.0
        self.last_best_bid = 0.0
        self.last_best_ask = 0.0
        self.recent_signed_flow: deque = deque(maxlen=2000)
        self.recent_trade_prices: deque = deque(maxlen=2000)
        self.recent_bars: deque = deque(maxlen=120)
        self.session_tracker = SessionTracker()
        self.vwap_tracker = VWAPTracker()

        class _Exec:
            position: OpenPosition | None = None
            pending_entry = None
            def has_exposure(self) -> bool:
                return self.position is not None
        self.executor = _Exec()


def _simulate_position(position: OpenPosition, bars: list[Bar], start_idx: int, max_hold_sec: int) -> tuple[float, str, int]:
    """Walk bars after start_idx until stop/TP1/TP2 hits or max_hold elapses.

    Returns (r_multiple, exit_reason, exit_bar_idx).
    Pessimistic OHLC: if bar.low/high touches stop and TP in same bar, assume stop wins.
    """
    risk = abs(position.entry_price - position.stop_price) * position.qty_remaining
    if risk <= 0:
        return 0.0, "bad_risk", start_idx
    max_bars = max(1, max_hold_sec // 60)
    end_idx = min(start_idx + max_bars, len(bars))
    for idx in range(start_idx, end_idx):
        b = bars[idx]
        if position.side == Side.LONG:
            if b.low <= position.stop_price:
                pnl = (position.stop_price - position.entry_price) * position.qty_remaining
                return pnl / risk, "stop", idx
            if b.high >= position.tp2_price and position.tp2_price > 0:
                pnl = (position.tp2_price - position.entry_price) * position.qty_remaining
                return pnl / risk, "tp2", idx
            if b.high >= position.tp1_price and not position.tp1_filled:
                # Mark TP1 hit, move stop to BE, halve qty
                position.tp1_filled = True
                position.qty_remaining *= 0.5
                position.stop_price = position.entry_price
        else:
            if b.high >= position.stop_price:
                pnl = (position.entry_price - position.stop_price) * position.qty_remaining
                return pnl / risk, "stop", idx
            if b.low <= position.tp2_price and position.tp2_price > 0:
                pnl = (position.entry_price - position.tp2_price) * position.qty_remaining
                return pnl / risk, "tp2", idx
            if b.low <= position.tp1_price and not position.tp1_filled:
                position.tp1_filled = True
                position.qty_remaining *= 0.5
                position.stop_price = position.entry_price
    # max hold reached — close at last bar's close
    last = bars[end_idx - 1]
    if position.side == Side.LONG:
        pnl = (last.close - position.entry_price) * position.qty_remaining
    else:
        pnl = (position.entry_price - last.close) * position.qty_remaining
    return pnl / risk, "max_hold", end_idx - 1


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-strategy backtest over HL 1m candles.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--coins", default="", help="Comma-separated; default uses HL_COINS.")
    parser.add_argument("--interval-min", type=int, default=15, help="AI poll cadence (min). Default 15.")
    parser.add_argument("--max-hold-sec", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate call count + cost; don't send LLM requests.")
    parser.add_argument("--json-out", default="", help="Write summary JSON to this path.")
    parser.add_argument("--max-decisions", type=int, default=0,
                        help="Hard cap on LLM calls (cost safety). 0 = unlimited.")
    args = parser.parse_args()

    _load_env(ROOT / args.env_file)
    cfg = load_config()
    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        print("error: OPENROUTER_API_KEY not set (or use --dry-run)", file=sys.stderr)
        return 2

    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    api_url = constants.TESTNET_API_URL if "testnet" in cfg.feed.ws_url.lower() else constants.MAINNET_API_URL
    info = Info(api_url, skip_ws=True)
    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()] or cfg.feed.coins
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(1, args.days) * 24 * 60 * 60 * 1000
    interval_bars = max(1, args.interval_min)  # 1m bars per interval

    # Force AI on for the backtest config (but never construct live executor)
    cfg.ai.enabled = True
    if not args.dry_run:
        from hliq_bot.ai.market_data import MarketDataCache
        from hliq_bot.ai.strategy import AIStrategy
        market_data = MarketDataCache(info)
        strategy = AIStrategy(cfg.ai, market_data=market_data)

    print(f"AI Backtest (days={args.days}, interval={args.interval_min}m, dry_run={args.dry_run})")
    print(f"  coins={coins}  model={cfg.ai.model}")
    print("-" * 60)

    summary: dict[str, Any] = {"per_coin": {}, "total_decisions": 0, "total_cost_usd": 0.0,
                                "total_trades": 0, "total_r": 0.0}
    total_decisions = 0

    for coin in coins:
        print(f"Loading {coin} candles...", flush=True)
        bars = _fetch_candles(info, coin, start_ms, end_ms)
        if not bars:
            print(f"  {coin}: no candles available, skipping")
            continue
        print(f"  {coin}: {len(bars)} bars from {datetime.fromtimestamp(bars[0].start_ms/1000, tz=timezone.utc):%Y-%m-%d %H:%M} "
              f"to {datetime.fromtimestamp(bars[-1].end_ms/1000, tz=timezone.utc):%Y-%m-%d %H:%M}")

        worker = _SimWorker(coin)
        coin_decisions = 0
        actions: dict[str, int] = {}
        trades: list[dict[str, Any]] = []
        coin_cost = 0.0

        for i in range(0, len(bars) - interval_bars, interval_bars):
            if args.max_decisions and total_decisions >= args.max_decisions:
                break
            # Feed all bars up to this point into trackers
            bar = bars[i]
            worker.session_tracker.on_bar(bar)
            worker.vwap_tracker.on_bar(bar)
            worker.recent_bars.append(bar)
            worker.recent_trade_prices.append((bar.end_ms, bar.close))
            # Synthesize last_best_bid/ask from close +/- 1bp
            worker.last_best_bid = bar.close * 0.9999
            worker.last_best_ask = bar.close * 1.0001
            worker.last_spread_bps = 2.0

            if args.dry_run:
                actions["dry_run"] = actions.get("dry_run", 0) + 1
                coin_decisions += 1
                total_decisions += 1
                continue

            result = strategy.decide_for_coin(
                worker,
                bars=list(worker.recent_bars),
                now_ms=bar.end_ms,
                account_equity=cfg.risk.account_equity,
                daily_pnl=0.0,
                daily_r=0.0,
                recent_outcomes=[],
            )
            actions[result.action] = actions.get(result.action, 0) + 1
            coin_decisions += 1
            total_decisions += 1
            coin_cost += result.cost_usd

            # Open a simulated position only on open_long/open_short
            if result.action in {"open_long", "open_short"} and result.signal is not None and worker.executor.position is None:
                from hliq_bot.models import OpenPosition
                qty = 0.01  # nominal for ratio math; risk math uses qty * stop_dist
                worker.executor.position = OpenPosition(
                    signal_id=f"bt-{coin}-{i}",
                    side=result.signal.side,
                    entry_price=result.signal.entry_price,
                    stop_price=result.signal.stop_price,
                    tp1_price=result.signal.tp1_price,
                    tp2_price=result.signal.tp2_price,
                    opened_ms=bar.end_ms,
                    qty_initial=qty, qty_remaining=qty,
                    risk_dollars=abs(result.signal.entry_price - result.signal.stop_price) * qty,
                    coin=coin,
                    best_price=result.signal.entry_price,
                    worst_price=result.signal.entry_price,
                )
                # Simulate forward
                r_mult, exit_reason, exit_idx = _simulate_position(
                    worker.executor.position, bars, i + 1, args.max_hold_sec,
                )
                trades.append({
                    "i": i, "side": result.signal.side.value, "r": r_mult,
                    "exit": exit_reason, "exit_bar": exit_idx,
                    "reasoning": result.reasoning[:120],
                })
                worker.executor.position = None

            if coin_decisions % 10 == 0:
                print(f"  {coin}: {coin_decisions} decisions so far  cost=${coin_cost:.3f}", flush=True)

        # Summarize coin
        coin_r = sum(t["r"] for t in trades)
        avg_r = coin_r / len(trades) if trades else 0.0
        wins = sum(1 for t in trades if t["r"] > 0)
        win_rate = (wins / len(trades) * 100.0) if trades else 0.0
        summary["per_coin"][coin] = {
            "decisions": coin_decisions, "actions": actions,
            "trades": len(trades), "avg_r": round(avg_r, 3),
            "win_rate_pct": round(win_rate, 1), "total_r": round(coin_r, 3),
            "cost_usd": round(coin_cost, 4),
        }
        summary["total_decisions"] += coin_decisions
        summary["total_cost_usd"] += coin_cost
        summary["total_trades"] += len(trades)
        summary["total_r"] += coin_r
        print(f"  {coin}: decisions={coin_decisions} actions={actions} trades={len(trades)} "
              f"avg_r={avg_r:+.3f} win_rate={win_rate:.1f}% cost=${coin_cost:.4f}")

    # Final
    print()
    print(f"TOTAL: decisions={summary['total_decisions']} trades={summary['total_trades']} "
          f"total_r={summary['total_r']:+.3f} cost_usd=${summary['total_cost_usd']:.4f}")
    if args.dry_run and not args.dry_run is False:
        # Rough cost estimate using current model pricing
        from hliq_bot.ai.client import _cost_for
        est = _cost_for(cfg.ai.model, prompt_tokens=3000, completion_tokens=200) * summary["total_decisions"]
        print(f"  ESTIMATED COST if not dry-run: ~${est:.2f} (assuming ~3k input + 200 output tokens/call)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"  Summary written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
