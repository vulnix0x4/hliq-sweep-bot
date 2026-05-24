#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hliq_bot.config import load_config  # noqa: E402


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _matches(value: str, patterns: set[str]) -> bool:
    value = value.strip()
    return bool(value) and any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _matches_coin_level(coin: str, level: str, patterns: set[str]) -> bool:
    return _matches(f"{coin.strip().upper()}:{level.strip().lower()}", patterns)


def _matches_coin_session(coin: str, session: str, patterns: set[str]) -> bool:
    return _matches(f"{coin.strip().upper()}:{session.strip().lower()}", patterns)


def _matches_coin_session_level(coin: str, session: str, level: str, patterns: set[str]) -> bool:
    return _matches(f"{coin.strip().upper()}:{session.strip().lower()}:{level.strip().lower()}", patterns)


def _selected_by_policy(candidate: dict[str, Any]) -> tuple[bool, str]:
    cfg = load_config()
    coin = str(candidate.get("coin", "")).strip().upper()
    if cfg.strategy.allowed_coins and not _matches(coin, cfg.strategy.allowed_coins):
        return False, f"allow_coin_miss:{coin}"
    if _matches(coin, cfg.strategy.blocked_coins):
        return False, f"block_coin:{coin}"

    level = str(candidate.get("level_label", "")).strip().lower()
    if cfg.strategy.allowed_level_labels and not _matches(level, cfg.strategy.allowed_level_labels):
        return False, f"allow_level_miss:{level}"
    if _matches(level, cfg.strategy.blocked_level_labels):
        return False, f"block_level:{level}"
    if cfg.strategy.allowed_coin_level_pairs and not _matches_coin_level(coin, level, cfg.strategy.allowed_coin_level_pairs):
        return False, f"allow_coin_level_miss:{coin}:{level}"
    if _matches_coin_level(coin, level, cfg.strategy.blocked_coin_level_pairs):
        return False, f"block_coin_level:{coin}:{level}"

    session = str(candidate.get("session", "")).strip().lower()
    if cfg.strategy.allowed_coin_session_pairs and not _matches_coin_session(coin, session, cfg.strategy.allowed_coin_session_pairs):
        return False, f"allow_coin_session_miss:{coin}:{session}"
    if _matches_coin_session(coin, session, cfg.strategy.blocked_coin_session_pairs):
        return False, f"block_coin_session:{coin}:{session}"
    if cfg.strategy.allowed_coin_session_level_triples and not _matches_coin_session_level(
        coin, session, level, cfg.strategy.allowed_coin_session_level_triples
    ):
        return False, f"allow_coin_session_level_miss:{coin}:{session}:{level}"
    if _matches_coin_session_level(coin, session, level, cfg.strategy.blocked_coin_session_level_triples):
        return False, f"block_coin_session_level:{coin}:{session}:{level}"
    if cfg.strategy.allowed_sessions and not _matches(session, cfg.strategy.allowed_sessions):
        return False, f"allow_session_miss:{session}"
    if _matches(session, cfg.strategy.blocked_sessions):
        return False, f"block_session:{session}"

    side = str(candidate.get("side", "")).strip().lower()
    if cfg.strategy.allowed_sides and not _matches(side, cfg.strategy.allowed_sides):
        return False, f"allow_side_miss:{side}"
    if _matches(side, cfg.strategy.blocked_sides):
        return False, f"block_side:{side}"

    score = _safe_float(candidate.get("signal_score"))
    if score < cfg.strategy.min_signal_score:
        return False, f"signal_score<{cfg.strategy.min_signal_score:.2f}"

    regime = str(candidate.get("regime", "")).strip().lower()
    conf_floor = cfg.strategy.min_confidence_trend if regime == "trend" else cfg.strategy.min_confidence_range
    confidence = _safe_float(candidate.get("confidence"))
    if confidence < conf_floor:
        return False, f"confidence<{conf_floor:.2f}"

    return True, "selected"


def _summarize(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, float | int]:
    pnls = [_safe_float(out.get("pnl")) for _, out in rows]
    rs = [_safe_float(out.get("r_multiple")) for _, out in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0
    return {
        "trades": len(rows),
        "wins": len(wins),
        "net_pnl": sum(pnls),
        "win_rate": (len(wins) / len(rows)) * 100.0 if rows else 0.0,
        "avg_r": (sum(rs) / len(rs)) if rs else 0.0,
        "profit_factor": profit_factor,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply current policy filters to historical journal outcomes.")
    parser.add_argument("--input", default="runtime/signals.jsonl")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--show-selected", action="store_true")
    parser.add_argument("--min-selected", type=int, default=20)
    args = parser.parse_args()

    _load_env_file(ROOT / args.env_file)
    rows = _rows(ROOT / args.input)
    candidates = {str(r.get("signal_id", "")): r for r in rows if r.get("event_type") == "candidate"}
    outcomes = [r for r in rows if r.get("event_type") == "outcome"]

    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejected: dict[str, int] = {}
    for outcome in outcomes:
        candidate = candidates.get(str(outcome.get("signal_id", "")))
        if not candidate:
            continue
        allowed, reason = _selected_by_policy(candidate)
        if allowed:
            paired.append((candidate, outcome))
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    summary = _summarize(paired)
    print("Policy Report")
    print("=" * 40)
    print(f"input: {args.input}")
    print(f"closed_trades_in_input: {len(outcomes)}")
    print(f"selected_trades: {summary['trades']}")
    print(f"net_pnl: {summary['net_pnl']:.4f}")
    print(f"win_rate: {summary['win_rate']:.1f}%")
    print(f"avg_r: {summary['avg_r']:.3f}")
    pf = summary["profit_factor"]
    print(f"profit_factor: {'inf' if math.isinf(float(pf)) else f'{float(pf):.2f}'}")
    if int(summary["trades"]) < args.min_selected:
        print(f"sample_warning: selected_trades<{args.min_selected} (hypothesis only)")
    if rejected:
        print("\nTop rejects")
        print("-" * 40)
        for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1])[:12]:
            print(f"- {reason}: {count}")

    if args.show_selected and paired:
        print("\nSelected trades")
        print("-" * 40)
        for candidate, outcome in paired:
            print(
                "{coin} {side} {session} {level} pnl={pnl:.4f} r={r:.3f} score={score:.3f}".format(
                    coin=outcome.get("coin", candidate.get("coin", "")),
                    side=outcome.get("side", candidate.get("side", "")),
                    session=outcome.get("session", candidate.get("session", "")),
                    level=outcome.get("level_label", candidate.get("level_label", "")),
                    pnl=_safe_float(outcome.get("pnl")),
                    r=_safe_float(outcome.get("r_multiple")),
                    score=_safe_float(candidate.get("signal_score")),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
