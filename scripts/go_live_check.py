#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
import math
import os
from pathlib import Path
import sys
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hliq_bot.config import load_config  # noqa: E402
from hliq_bot.execution.hyperliquid_order_manager import (  # noqa: E402
    HL_MIN_ORDER_NOTIONAL,
    HyperliquidOrderManager,
)


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


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


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _last_run_id(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        if row.get("event_type") == "run" and row.get("event") == "run_start":
            run_id = str(row.get("run_id", "")).strip()
            if run_id:
                return run_id
    return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _journal_summary(
    rows: list[dict[str, Any]],
    *,
    run_id: str = "",
    run_ids: set[str] | None = None,
) -> dict[str, float | int | str]:
    filtered = [
        row for row in rows
        if (
            (run_ids is not None and str(row.get("run_id", "")).strip() in run_ids)
            or (run_ids is None and (not run_id or str(row.get("run_id", "")).strip() == run_id))
        )
    ]
    candidates = 0
    decisions = 0
    allowed = 0
    closed = 0
    wins = 0
    r_sum = 0.0
    pnl_sum = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    lifecycle: dict[str, int] = {}

    for row in filtered:
        event_type = str(row.get("event_type", ""))
        if event_type == "candidate":
            candidates += 1
        elif event_type == "decision":
            decisions += 1
            if bool(row.get("allowed", False)):
                allowed += 1
        elif event_type == "lifecycle":
            event = str(row.get("event", "unknown"))
            lifecycle[event] = lifecycle.get(event, 0) + 1
        elif event_type == "outcome":
            pnl = _safe_float(row.get("pnl"))
            r_val = _safe_float(row.get("r_multiple"))
            closed += 1
            pnl_sum += pnl
            r_sum += r_val
            if pnl > 0:
                wins += 1
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    return {
        "run_id": ",".join(sorted(run_ids)) if run_ids is not None else run_id,
        "rows": len(filtered),
        "candidates": candidates,
        "decisions": decisions,
        "allowed": allowed,
        "entries_placed": lifecycle.get("entry_placed", 0),
        "entries_filled": lifecycle.get("entry_filled", 0),
        "pending_entries": max(lifecycle.get("entry_placed", 0) - lifecycle.get("entry_filled", 0), 0),
        "open_paper_positions": max(lifecycle.get("entry_filled", 0) - closed, 0),
        "closed_trades": closed,
        "win_rate": (wins / closed) * 100.0 if closed else 0.0,
        "avg_r": r_sum / closed if closed else 0.0,
        "net_pnl": pnl_sum,
        "profit_factor": profit_factor,
    }


def _policy_key_from_config(cfg: Any) -> dict[str, Any]:
    strategy = cfg.strategy
    risk = cfg.risk
    return {
        "coins": sorted(cfg.feed.coins),
        "risk_per_trade_pct": risk.risk_per_trade_pct,
        "account_equity": risk.account_equity,
        "min_conf_range": strategy.min_confidence_range,
        "min_conf_trend": strategy.min_confidence_trend,
        "min_signal_score": strategy.min_signal_score,
        "maker_fee_pct": strategy.maker_fee_pct,
        "taker_fee_pct": strategy.taker_fee_pct,
        "paper_entry_slippage_bps": strategy.paper_entry_slippage_bps,
        "paper_exit_slippage_bps": strategy.paper_exit_slippage_bps,
        "paper_tp1_is_taker": strategy.paper_tp1_is_taker,
        "allowed_coins": sorted(strategy.allowed_coins),
        "allowed_level_labels": sorted(strategy.allowed_level_labels),
        "allowed_coin_level_pairs": sorted(strategy.allowed_coin_level_pairs),
        "allowed_coin_session_pairs": sorted(strategy.allowed_coin_session_pairs),
        "allowed_coin_session_level_triples": sorted(strategy.allowed_coin_session_level_triples),
        "allowed_sessions": sorted(strategy.allowed_sessions),
        "allowed_sides": sorted(strategy.allowed_sides),
        "blocked_coins": sorted(strategy.blocked_coins),
        "blocked_level_labels": sorted(strategy.blocked_level_labels),
        "blocked_coin_level_pairs": sorted(strategy.blocked_coin_level_pairs),
        "blocked_coin_session_pairs": sorted(strategy.blocked_coin_session_pairs),
        "blocked_coin_session_level_triples": sorted(strategy.blocked_coin_session_level_triples),
        "blocked_sessions": sorted(strategy.blocked_sessions),
        "blocked_sides": sorted(strategy.blocked_sides),
    }


def _normal_pair_values(raw: Any) -> list[str]:
    out: list[str] = []
    for value in raw or []:
        text = str(value).strip()
        if ":" not in text:
            continue
        coin, level = text.split(":", 1)
        coin = coin.strip().upper()
        level = level.strip().lower()
        if coin and level:
            out.append(f"{coin}:{level}")
    return sorted(out)


def _normal_triple_values(raw: Any) -> list[str]:
    out: list[str] = []
    for value in raw or []:
        text = str(value).strip()
        if text.count(":") < 2:
            continue
        coin, session, level = text.split(":", 2)
        coin = coin.strip().upper()
        session = session.strip().lower()
        level = level.strip().lower()
        if coin and session and level:
            out.append(f"{coin}:{session}:{level}")
    return sorted(out)


def _policy_key_from_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "coins": sorted(str(c).upper() for c in row.get("coins", []) if str(c).strip()),
        "risk_per_trade_pct": _safe_float(row.get("risk_per_trade_pct")),
        "account_equity": _safe_float(row.get("account_equity")),
        "min_conf_range": _safe_float(row.get("min_conf_range")),
        "min_conf_trend": _safe_float(row.get("min_conf_trend")),
        "min_signal_score": _safe_float(row.get("min_signal_score")),
        "maker_fee_pct": _safe_float(row.get("maker_fee_pct")),
        "taker_fee_pct": _safe_float(row.get("taker_fee_pct")),
        "paper_entry_slippage_bps": _safe_float(row.get("paper_entry_slippage_bps")),
        "paper_exit_slippage_bps": _safe_float(row.get("paper_exit_slippage_bps")),
        "paper_tp1_is_taker": bool(row.get("paper_tp1_is_taker", False)),
        "allowed_coins": sorted(str(v).upper() for v in row.get("allowed_coins", []) if str(v).strip()),
        "allowed_level_labels": sorted(str(v).lower() for v in row.get("allowed_level_labels", []) if str(v).strip()),
        "allowed_coin_level_pairs": _normal_pair_values(row.get("allowed_coin_level_pairs", [])),
        "allowed_coin_session_pairs": _normal_pair_values(row.get("allowed_coin_session_pairs", [])),
        "allowed_coin_session_level_triples": _normal_triple_values(row.get("allowed_coin_session_level_triples", [])),
        "allowed_sessions": sorted(str(v).lower() for v in row.get("allowed_sessions", []) if str(v).strip()),
        "allowed_sides": sorted(str(v).lower() for v in row.get("allowed_sides", []) if str(v).strip()),
        "blocked_coins": sorted(str(v).upper() for v in row.get("blocked_coins", []) if str(v).strip()),
        "blocked_level_labels": sorted(str(v).lower() for v in row.get("blocked_level_labels", []) if str(v).strip()),
        "blocked_coin_level_pairs": _normal_pair_values(row.get("blocked_coin_level_pairs", [])),
        "blocked_coin_session_pairs": _normal_pair_values(row.get("blocked_coin_session_pairs", [])),
        "blocked_coin_session_level_triples": _normal_triple_values(row.get("blocked_coin_session_level_triples", [])),
        "blocked_sessions": sorted(str(v).lower() for v in row.get("blocked_sessions", []) if str(v).strip()),
        "blocked_sides": sorted(str(v).lower() for v in row.get("blocked_sides", []) if str(v).strip()),
    }


def _active_policy_run_ids(rows: list[dict[str, Any]], cfg: Any) -> set[str]:
    current = _policy_key_from_config(cfg)
    out: set[str] = set()
    for row in rows:
        if row.get("event_type") != "run" or row.get("event") != "run_start":
            continue
        run_id = str(row.get("run_id", "")).strip()
        if run_id and _policy_key_from_run(row) == current:
            out.add(run_id)
    return out


def _active_position_files(runtime_dir: str) -> list[Path]:
    # Must match HyperliquidOrderManager — it reads BOT_STATE_DIR (defaulting to
    # runtime/active_positions) for the position state files. If an operator
    # sets BOT_STATE_DIR but leaves BOT_RUNTIME_DIR alone, scanning under
    # runtime_dir would silently miss live exposure and let the go-live gate
    # pass while positions are still open.
    raw = os.environ.get("BOT_STATE_DIR", "")
    if raw:
        state_path = Path(raw)
        path = state_path if state_path.is_absolute() else (ROOT / state_path)
    else:
        path = ROOT / runtime_dir / "active_positions"
    path = path.resolve()
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file())


def _parse_candle_total(output: str) -> dict[str, float | int | str]:
    for raw in output.splitlines():
        line = raw.strip()
        if not line.startswith("TOTAL:"):
            continue
        parts: dict[str, float | int | str] = {}
        for token in line.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "trades":
                try:
                    parts[key] = int(value)
                except ValueError:
                    parts[key] = 0
            elif key in {"avg_r", "win_rate", "profit_factor", "span_days"}:
                try:
                    parts[key] = math.inf if value == "inf" else float(value)
                except ValueError:
                    parts[key] = 0.0
        return parts
    return {}


def _parse_candle_coin_totals(output: str) -> dict[str, dict[str, float | int | str]]:
    totals: dict[str, dict[str, float | int | str]] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.startswith("TOTAL:") or line.startswith("slice "):
            continue
        head, sep, rest = line.partition(": ")
        if not sep or not head.isupper():
            continue
        parts: dict[str, float | int | str] = {}
        for token in rest.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "trades":
                try:
                    parts[key] = int(value)
                except ValueError:
                    parts[key] = 0
            elif key in {"avg_r", "win_rate", "profit_factor"}:
                try:
                    parts[key] = math.inf if value == "inf" else float(value)
                except ValueError:
                    parts[key] = 0.0
        if "trades" in parts:
            totals[head] = parts
    return totals


def _parse_candle_slices(output: str) -> list[dict[str, float | int | str]]:
    slices: list[dict[str, float | int | str]] = []
    current_coin = ""
    for raw in output.splitlines():
        line = raw.strip()
        if line and not line.startswith("TOTAL:") and not line.startswith("slice "):
            head, sep, _rest = line.partition(": ")
            if sep and head.isupper():
                current_coin = head
            continue
        if not line.startswith("slice "):
            continue
        head, sep, rest = line.rpartition(": ")
        if not sep:
            continue
        parts = head.split()
        if len(parts) != 2:
            continue
        key_parts = parts[1].split(":")
        if len(key_parts) < 3:
            continue
        item: dict[str, float | int | str] = {
            "coin": current_coin,
            "session": key_parts[0],
            "side": key_parts[1],
            "level": ":".join(key_parts[2:]),
        }
        for token in rest.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "trades":
                try:
                    item[key] = int(value)
                except ValueError:
                    item[key] = 0
            elif key in {"avg_r", "pf"}:
                try:
                    item[key] = math.inf if value == "inf" else float(value)
                except ValueError:
                    item[key] = 0.0
        slices.append(item)
    return slices


def _weighted_slice_stats(
    matches: list[dict[str, float | int | str]],
) -> tuple[int, float, float]:
    trades = sum(int(item.get("trades", 0)) for item in matches)
    if trades <= 0:
        return 0, 0.0, 0.0
    avg_r = sum(
        int(item.get("trades", 0)) * float(item.get("avg_r", 0.0))
        for item in matches
    ) / trades
    pf_values = [float(item.get("pf", 0.0)) for item in matches]
    min_pf = min(pf_values) if pf_values else 0.0
    return trades, avg_r, min_pf


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
        try:
            timeout_sec = max(30, int(os.getenv("BOT_EDGE_CANDLE_TIMEOUT_SEC", "180")))
        except ValueError:
            timeout_sec = 180
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except Exception as exc:
        return "", f"candle backtest error: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        msg = detail[-1] if detail else f"exit={proc.returncode}"
        return "", f"candle backtest failed: {msg}"
    return proc.stdout, ""


def _candle_edge_check(
    *,
    days: int,
    min_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
    min_span_days: float,
    output: str | None = None,
) -> Check:
    if output is None:
        output, error = _run_candle_backtest(days)
        if error:
            return Check("recent_candle_edge", False, error)

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


def _candle_session_sample_check(
    *,
    days: int,
    min_session_trades: int,
    output: str | None = None,
) -> Check:
    if output is None:
        output, error = _run_candle_backtest(days)
        if error:
            return Check("candle_session_samples", False, error)

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
    detail = " ".join(
        f"{session}={by_session.get(session, 0)}" for session in sorted(active_sessions)
    )
    ok = not weak
    return Check(
        "candle_session_samples",
        ok,
        f"{detail or 'no_allowed_sessions'} min_session_trades={min_session_trades}",
    )


def _candle_coin_sample_check(
    *,
    days: int,
    min_coin_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
    output: str | None = None,
) -> Check:
    if output is None:
        output, error = _run_candle_backtest(days)
        if error:
            return Check("candle_coin_samples", False, error)

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

    ok = not weak
    return Check(
        "candle_coin_samples",
        ok,
        (
            " ".join(details)
            + f" min_coin_trades={min_coin_trades} min_avg_r={min_avg_r:.3f} "
            + f"min_profit_factor={min_profit_factor:.2f}"
        ),
    )


def _candle_level_sample_check(
    *,
    days: int,
    min_level_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
    output: str | None = None,
) -> Check:
    if output is None:
        output, error = _run_candle_backtest(days)
        if error:
            return Check("candle_level_samples", False, error)

    allowed_levels = sorted(load_config().strategy.allowed_level_labels)
    if not allowed_levels:
        return Check("candle_level_samples", False, "no_allowed_levels")

    slices = _parse_candle_slices(output)
    weak: list[str] = []
    details: list[str] = []
    for pattern in allowed_levels:
        matches = [
            item for item in slices
            if fnmatch.fnmatchcase(str(item.get("level", "")), pattern)
        ]
        trades, avg_r, min_pf = _weighted_slice_stats(matches)
        pf_text = "inf" if math.isinf(min_pf) else f"{min_pf:.2f}"
        details.append(f"{pattern}=trades:{trades},avg_r:{avg_r:.3f},min_pf:{pf_text}")
        if trades < min_level_trades or avg_r < min_avg_r or min_pf < min_profit_factor:
            weak.append(pattern)

    ok = not weak
    return Check(
        "candle_level_samples",
        ok,
        (
            " ".join(details)
            + f" min_level_trades={min_level_trades} min_avg_r={min_avg_r:.3f} "
            + f"min_profit_factor={min_profit_factor:.2f}"
        ),
    )


def _candle_coin_level_sample_check(
    *,
    days: int,
    min_pair_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
    output: str | None = None,
) -> Check:
    if output is None:
        output, error = _run_candle_backtest(days)
        if error:
            return Check("candle_coin_level_samples", False, error)

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


def _candle_coin_session_sample_check(
    *,
    days: int,
    min_pair_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
    output: str | None = None,
) -> Check:
    pairs = sorted(load_config().strategy.allowed_coin_session_pairs)
    if not pairs:
        return Check("candle_coin_session_samples", True, "no_allowed_coin_sessions")
    if output is None:
        output, error = _run_candle_backtest(days=7)
        if error:
            return Check("candle_coin_session_samples", False, error)
    slices = _parse_candle_slices(output)
    weak: list[str] = []
    details: list[str] = []
    for pair in pairs:
        coin, session_pattern = pair.split(":", 1)
        matches = [
            item for item in slices
            if str(item.get("coin", "")).upper() == coin
            and fnmatch.fnmatchcase(str(item.get("session", "")), session_pattern)
        ]
        trades, avg_r, min_pf = _weighted_slice_stats(matches)
        pf_text = "inf" if math.isinf(min_pf) else f"{min_pf:.2f}"
        details.append(f"{pair}=trades:{trades},avg_r:{avg_r:.3f},min_pf:{pf_text}")
        if trades < min_pair_trades or avg_r < min_avg_r or min_pf < min_profit_factor:
            weak.append(pair)
    return Check(
        "candle_coin_session_samples",
        not weak,
        (
            " ".join(details)
            + f" min_pair_trades={min_pair_trades} min_avg_r={min_avg_r:.3f} "
            + f"min_profit_factor={min_profit_factor:.2f}"
        ),
    )


def _candle_coin_session_level_sample_check(
    *,
    days: int,
    min_triple_trades: int,
    min_avg_r: float,
    min_profit_factor: float,
    output: str | None = None,
) -> Check:
    triples = sorted(load_config().strategy.allowed_coin_session_level_triples)
    if not triples:
        return Check("candle_coin_session_level_samples", True, "no_allowed_coin_session_levels")
    if output is None:
        output, error = _run_candle_backtest(days=days)
        if error:
            return Check("candle_coin_session_level_samples", False, error)
    slices = _parse_candle_slices(output)
    weak: list[str] = []
    details: list[str] = []
    for triple in triples:
        coin, session_pattern, level_pattern = triple.split(":", 2)
        matches = [
            item for item in slices
            if str(item.get("coin", "")).upper() == coin
            and fnmatch.fnmatchcase(str(item.get("session", "")), session_pattern)
            and fnmatch.fnmatchcase(str(item.get("level", "")), level_pattern)
        ]
        trades, avg_r, min_pf = _weighted_slice_stats(matches)
        pf_text = "inf" if math.isinf(min_pf) else f"{min_pf:.2f}"
        details.append(f"{triple}=trades:{trades},avg_r:{avg_r:.3f},min_pf:{pf_text}")
        if trades < min_triple_trades or avg_r < min_avg_r or min_pf < min_profit_factor:
            weak.append(triple)
    return Check(
        "candle_coin_session_level_samples",
        not weak,
        (
            " ".join(details)
            + f" min_triple_trades={min_triple_trades} min_avg_r={min_avg_r:.3f} "
            + f"min_profit_factor={min_profit_factor:.2f}"
        ),
    )


def _sizing_floor_check() -> Check:
    cfg = load_config()
    # Worst-case notional occurs at the widest allowed stop distance and smallest
    # allowed dynamic risk multiplier. For a percent stop, entry price cancels out:
    # notional = risk_dollars / stop_distance_pct.
    stop_distance_pct = max(cfg.strategy.max_stop_distance_bps / 10_000.0, 1e-12)
    risk_dollars = (
        cfg.risk.account_equity
        * (cfg.risk.risk_per_trade_pct / 100.0)
        * max(0.0, cfg.risk.risk_mult_min)
    )
    unconstrained_notional = risk_dollars / stop_distance_pct
    max_paper_notional = cfg.risk.account_equity * cfg.risk.max_leverage
    effective_notional = min(unconstrained_notional, max_paper_notional, cfg.live.max_notional_per_trade)
    ok = effective_notional >= HL_MIN_ORDER_NOTIONAL
    return Check(
        "sizing_above_hl_min_notional",
        ok,
        (
            f"worst_case_notional={effective_notional:.2f} "
            f"min_notional={HL_MIN_ORDER_NOTIONAL:.2f} "
            f"risk_dollars={risk_dollars:.4f} "
            f"max_stop_bps={cfg.strategy.max_stop_distance_bps:.1f}"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Hard pass/fail readiness check before enabling real-money live mode.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--journal", default="runtime/signals.jsonl")
    parser.add_argument("--run-id", default="", help="Use this run_id. Defaults to the latest run_start if present.")
    parser.add_argument("--all-runs", action="store_true", help="Evaluate the whole journal instead of the latest run.")
    parser.add_argument(
        "--active-policy-runs",
        action="store_true",
        help="Evaluate all runs whose run_start policy exactly matches the active .env policy.",
    )
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--min-avg-r", type=float, default=0.10)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--max-risk-pct", type=float, default=0.50)
    parser.add_argument("--max-live-notional", type=float, default=50.0)
    parser.add_argument("--skip-candle-edge", action="store_true", help="Skip recent candle evidence gate.")
    parser.add_argument("--candle-days", type=int, default=7)
    parser.add_argument("--min-candle-trades", type=int, default=5)
    parser.add_argument("--min-candle-avg-r", type=float, default=0.05)
    parser.add_argument("--min-candle-profit-factor", type=float, default=1.20)
    parser.add_argument("--min-candle-span-days", type=float, default=3.5)
    parser.add_argument("--min-candle-session-trades", type=int, default=0)
    parser.add_argument("--min-candle-coin-trades", type=int, default=0)
    parser.add_argument("--min-candle-level-trades", type=int, default=0)
    parser.add_argument(
        "--allow-code-watched-stops",
        action="store_true",
        help="Permit live readiness with bot-watched stops. Intended only for tiny pilot size.",
    )
    args = parser.parse_args()

    _load_env_file(ROOT / args.env_file)
    cfg = load_config()

    rows = _read_rows(ROOT / args.journal)
    run_id = ""
    active_policy_run_ids: set[str] | None = None
    if args.active_policy_runs:
        active_policy_run_ids = _active_policy_run_ids(rows, cfg)
    elif not args.all_runs:
        run_id = args.run_id.strip() or _last_run_id(rows)
    summary = _journal_summary(rows, run_id=run_id, run_ids=active_policy_run_ids)
    active_positions = _active_position_files(cfg.runtime.runtime_dir)
    candle_output: str | None = None
    candle_error = ""
    if not args.skip_candle_edge:
        candle_output, candle_error = _run_candle_backtest(max(1, args.candle_days))

    checks: list[Check] = [
        Check(
            "live_disarmed_now",
            cfg.mode != "live" or not cfg.live.allow_live,
            f"mode={cfg.mode} allow_live={cfg.live.allow_live}",
        ),
        Check(
            "testnet_until_ready",
            cfg.live.network == "testnet",
            f"HL_NETWORK={cfg.live.network}",
        ),
        Check(
            "warmup_disabled",
            not cfg.strategy.warmup_enabled,
            f"BOT_WARMUP_ENABLED={cfg.strategy.warmup_enabled}",
        ),
        Check(
            "ml_fail_closed",
            not cfg.runtime.ml_fail_open,
            f"BOT_ML_FAIL_OPEN={cfg.runtime.ml_fail_open}",
        ),
        Check(
            "risk_cap_small",
            cfg.risk.risk_per_trade_pct <= args.max_risk_pct,
            f"risk_per_trade_pct={cfg.risk.risk_per_trade_pct:.3f} max={args.max_risk_pct:.3f}",
        ),
        Check(
            "live_notional_cap",
            cfg.live.max_notional_per_trade <= args.max_live_notional,
            f"max_notional={cfg.live.max_notional_per_trade:.2f} max={args.max_live_notional:.2f}",
        ),
        Check(
            "paper_notional_matches_live_cap",
            cfg.risk.account_equity * cfg.risk.max_leverage <= cfg.live.max_notional_per_trade,
            (
                f"paper_max_notional={cfg.risk.account_equity * cfg.risk.max_leverage:.2f} "
                f"live_max_notional={cfg.live.max_notional_per_trade:.2f}"
            ),
        ),
        _sizing_floor_check(),
        Check(
            "realistic_fee_model",
            cfg.strategy.maker_fee_pct >= 0.0 and cfg.strategy.taker_fee_pct > 0.0,
            f"maker={cfg.strategy.maker_fee_pct:.6f} taker={cfg.strategy.taker_fee_pct:.6f}",
        ),
        Check(
            "bad_slices_blocked",
            bool(
                cfg.strategy.allowed_coins
                or cfg.strategy.allowed_level_labels
                or cfg.strategy.allowed_coin_level_pairs
                or cfg.strategy.allowed_coin_session_pairs
                or cfg.strategy.allowed_coin_session_level_triples
                or cfg.strategy.allowed_sessions
                or cfg.strategy.allowed_sides
                or cfg.strategy.blocked_coins
                or cfg.strategy.blocked_level_labels
                or cfg.strategy.blocked_coin_level_pairs
                or cfg.strategy.blocked_coin_session_pairs
                or cfg.strategy.blocked_coin_session_level_triples
                or cfg.strategy.blocked_sessions
                or cfg.strategy.blocked_sides
            ),
            (
                f"allow_coins={sorted(cfg.strategy.allowed_coins)} "
                f"allow_levels={sorted(cfg.strategy.allowed_level_labels)} "
                f"allow_coin_levels={sorted(cfg.strategy.allowed_coin_level_pairs)} "
                f"allow_coin_sessions={sorted(cfg.strategy.allowed_coin_session_pairs)} "
                f"allow_coin_session_levels={sorted(cfg.strategy.allowed_coin_session_level_triples)} "
                f"allow_sessions={sorted(cfg.strategy.allowed_sessions)} "
                f"allow_sides={sorted(cfg.strategy.allowed_sides)} "
                f"block_coins={sorted(cfg.strategy.blocked_coins)} "
                f"block_levels={sorted(cfg.strategy.blocked_level_labels)} "
                f"block_coin_levels={sorted(cfg.strategy.blocked_coin_level_pairs)} "
                f"block_coin_sessions={sorted(cfg.strategy.blocked_coin_session_pairs)} "
                f"block_coin_session_levels={sorted(cfg.strategy.blocked_coin_session_level_triples)} "
                f"block_sessions={sorted(cfg.strategy.blocked_sessions)} "
                f"block_sides={sorted(cfg.strategy.blocked_sides)}"
            ),
        ),
        Check(
            "no_active_position_files",
            len(active_positions) == 0,
            f"active_files={len(active_positions)}",
        ),
        Check(
            "no_paper_exposure",
            int(summary["pending_entries"]) == 0 and int(summary["open_paper_positions"]) == 0,
            (
                f"pending_entries={summary['pending_entries']} "
                f"open_paper_positions={summary['open_paper_positions']}"
            ),
        ),
        Check(
            "native_or_tiny_stop_policy",
            bool(getattr(HyperliquidOrderManager, "NATIVE_STOPS_SUPPORTED", False)) or args.allow_code_watched_stops,
            (
                "native exchange stops supported"
                if getattr(HyperliquidOrderManager, "NATIVE_STOPS_SUPPORTED", False)
                else "native exchange stops are not implemented; pass --allow-code-watched-stops only for tiny pilot size"
            ),
        ),
        (
            Check("recent_candle_edge", True, "skipped by --skip-candle-edge")
            if args.skip_candle_edge
            else Check("recent_candle_edge", False, candle_error)
            if candle_error
            else _candle_edge_check(
                days=max(1, args.candle_days),
                min_trades=max(1, args.min_candle_trades),
                min_avg_r=args.min_candle_avg_r,
                min_profit_factor=args.min_candle_profit_factor,
                min_span_days=max(0.0, args.min_candle_span_days),
                output=candle_output,
            )
        ),
        (
            Check("candle_session_samples", True, "disabled")
            if args.skip_candle_edge or args.min_candle_session_trades <= 0
            else Check("candle_session_samples", False, candle_error)
            if candle_error
            else _candle_session_sample_check(
                days=args.candle_days,
                min_session_trades=args.min_candle_session_trades,
                output=candle_output,
            )
        ),
        (
            Check("candle_coin_samples", True, "disabled")
            if args.skip_candle_edge or args.min_candle_coin_trades <= 0
            else Check("candle_coin_samples", False, candle_error)
            if candle_error
            else _candle_coin_sample_check(
                days=args.candle_days,
                min_coin_trades=args.min_candle_coin_trades,
                min_avg_r=args.min_candle_avg_r,
                min_profit_factor=args.min_candle_profit_factor,
                output=candle_output,
            )
        ),
        (
            Check("candle_level_samples", True, "disabled")
            if args.skip_candle_edge or args.min_candle_level_trades <= 0
            else Check("candle_level_samples", False, candle_error)
            if candle_error
            else _candle_level_sample_check(
                days=args.candle_days,
                min_level_trades=args.min_candle_level_trades,
                min_avg_r=args.min_candle_avg_r,
                min_profit_factor=args.min_candle_profit_factor,
                output=candle_output,
            )
        ),
        (
            Check("candle_coin_level_samples", True, "disabled")
            if args.skip_candle_edge or args.min_candle_level_trades <= 0
            else Check("candle_coin_level_samples", False, candle_error)
            if candle_error
            else _candle_coin_level_sample_check(
                days=args.candle_days,
                min_pair_trades=args.min_candle_level_trades,
                min_avg_r=args.min_candle_avg_r,
                min_profit_factor=args.min_candle_profit_factor,
                output=candle_output,
            )
        ),
        (
            Check("candle_coin_session_samples", True, "disabled")
            if args.skip_candle_edge or args.min_candle_session_trades <= 0
            else Check("candle_coin_session_samples", False, candle_error)
            if candle_error
            else _candle_coin_session_sample_check(
                days=args.candle_days,
                min_pair_trades=args.min_candle_session_trades,
                min_avg_r=args.min_candle_avg_r,
                min_profit_factor=args.min_candle_profit_factor,
                output=candle_output,
            )
        ),
        (
            Check("candle_coin_session_level_samples", True, "disabled")
            if args.skip_candle_edge or args.min_candle_level_trades <= 0
            else Check("candle_coin_session_level_samples", False, candle_error)
            if candle_error
            else _candle_coin_session_level_sample_check(
                days=args.candle_days,
                min_triple_trades=args.min_candle_level_trades,
                min_avg_r=args.min_candle_avg_r,
                min_profit_factor=args.min_candle_profit_factor,
                output=candle_output,
            )
        ),
        Check(
            "paper_sample_size",
            int(summary["closed_trades"]) >= args.min_trades,
            f"closed_trades={summary['closed_trades']} min={args.min_trades}",
        ),
        Check(
            "paper_expectancy",
            float(summary["avg_r"]) >= args.min_avg_r,
            f"avg_r={float(summary['avg_r']):.3f} min={args.min_avg_r:.3f}",
        ),
        Check(
            "paper_profit_factor",
            float(summary["profit_factor"]) >= args.min_profit_factor,
            f"profit_factor={float(summary['profit_factor']):.3f} min={args.min_profit_factor:.3f}",
        ),
    ]

    ok = all(check.ok for check in checks)
    print("Go-Live Check")
    print("========================================")
    print(f"status: {'PASS' if ok else 'FAIL'}")
    print(f"journal: {args.journal}")
    if active_policy_run_ids is not None:
        print(f"run_id: active_policy({len(active_policy_run_ids)} runs)")
    else:
        print(f"run_id: {run_id or 'all'}")
    print(
        "sample: "
        f"rows={summary['rows']} candidates={summary['candidates']} allowed={summary['allowed']} "
        f"placed={summary['entries_placed']} filled={summary['entries_filled']} "
        f"pending={summary['pending_entries']} open_paper={summary['open_paper_positions']} "
        f"closed={summary['closed_trades']} win_rate={float(summary['win_rate']):.1f}% "
        f"avg_r={float(summary['avg_r']):.3f} profit_factor={float(summary['profit_factor']):.3f}"
    )
    print()
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
