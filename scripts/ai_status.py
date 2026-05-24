#!/usr/bin/env python3
"""Quick status of the running AI: open positions, memory stats, recent
decisions, AI override state. Read-only — safe to run anytime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="AI strategy status snapshot.")
    parser.add_argument("--runtime-dir", default="runtime")
    parser.add_argument("--recent", type=int, default=10, help="Recent decisions to show")
    parser.add_argument("--json", action="store_true", help="Emit JSON, not text")
    args = parser.parse_args()

    runtime = ROOT / args.runtime_dir
    pause_flag = runtime / "trade_pause.flag"
    override_flag = runtime / "ai_override.flag"
    memory_path = runtime / "ai_memory.jsonl"
    journal = runtime / "signals.jsonl"

    # Active positions (from on-disk state files)
    positions = []
    state_dir = runtime / "active_positions"
    if state_dir.exists():
        for p in state_dir.iterdir():
            if p.is_file() and p.suffix == ".json":
                try:
                    positions.append(json.loads(p.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue

    # Memory summary
    mem_rows = _load_jsonl(memory_path)
    decisions_mem = [r for r in mem_rows if r.get("kind") == "decision"]
    outcomes_mem = [r for r in mem_rows if r.get("kind") == "outcome"]
    by_id = {r["decision_id"]: r for r in decisions_mem if "decision_id" in r}
    for o in outcomes_mem:
        did = o.get("decision_id")
        if did and did in by_id:
            by_id[did].update({k: v for k, v in o.items() if k.startswith("outcome_")})
    resolved = [d for d in by_id.values() if "outcome_r_multiple" in d]
    wins = [d for d in resolved if (d.get("outcome_r_multiple") or 0) > 0]
    total_pnl = sum((d.get("outcome_pnl") or 0) for d in resolved)
    avg_r = sum((d.get("outcome_r_multiple") or 0) for d in resolved) / len(resolved) if resolved else 0.0

    # Recent decisions from journal
    j_rows = _load_jsonl(journal)
    ai_decisions = [r for r in j_rows if r.get("event_type") == "ai_decision"]
    recent = ai_decisions[-args.recent:] if ai_decisions else []
    cost_24h = sum(r.get("cost_usd") or 0 for r in ai_decisions
                   if (r.get("ts_ms") or 0) > (datetime.now(tz=timezone.utc).timestamp() - 86400) * 1000)

    # Override flag
    override = pause = None
    if override_flag.exists():
        try:
            override = override_flag.read_text(encoding="utf-8").strip()
        except OSError:
            override = "(unreadable)"
    if pause_flag.exists():
        try:
            pause = pause_flag.read_text(encoding="utf-8").strip()
        except OSError:
            pause = "(unreadable)"

    status = {
        "runtime_dir": str(runtime),
        "open_positions": positions,
        "memory": {
            "total_decisions": len(by_id),
            "resolved_trades": len(resolved),
            "wins": len(wins),
            "win_rate_pct": round(len(wins) / len(resolved) * 100, 1) if resolved else 0.0,
            "avg_r": round(avg_r, 3),
            "total_pnl": round(total_pnl, 4),
        },
        "cost_24h_usd": round(cost_24h, 4),
        "override_flag": override,
        "pause_flag": pause,
        "recent_decisions": [
            {
                "ts": datetime.fromtimestamp((r.get("ts_ms") or 0) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "coin": r.get("coin"),
                "action": r.get("action"),
                "confidence": r.get("confidence"),
                "reasoning": (r.get("reasoning") or "")[:160],
            }
            for r in recent
        ],
    }

    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    print("AI Status")
    print("=" * 60)
    print(f"Override flag : {override or '(none — trading normally)'}")
    print(f"Pause flag    : {pause or '(none)'}")
    print(f"Cost (24h)    : ${cost_24h:.4f}")
    print()
    print(f"Memory: {len(by_id)} decisions ({len(resolved)} resolved, {len(wins)} wins, "
          f"win_rate {round(len(wins)/len(resolved)*100,1) if resolved else 0:.1f}%) "
          f"avg_r={avg_r:+.3f} total_pnl=${total_pnl:.4f}")
    print()
    print(f"Open positions ({len(positions)}):")
    for p in positions:
        print(f"  {p.get('coin'):>6s} {p.get('side'):>5s} "
              f"qty={p.get('qty_remaining'):.6f} entry={p.get('entry_price'):.4f} "
              f"stop={p.get('stop_price'):.4f} (signal={p.get('signal_id')})")
    print()
    print(f"Recent decisions (last {len(recent)}):")
    for r in status["recent_decisions"]:
        print(f"  {r['ts']:>23s}  {r['coin']:>6s}  {r['action']:>22s}  conf={r['confidence']}  {r['reasoning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
