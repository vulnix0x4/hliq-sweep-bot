#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import itertools
import json
import math
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
from hliq_bot.models import Bar, Side  # noqa: E402
from hliq_bot.signal.session_tracker import SessionTracker  # noqa: E402
from hliq_bot.signal.sweep_detector import SweepDetector  # noqa: E402
from hliq_bot.signal.vwap_tracker import VWAPTracker  # noqa: E402

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


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


def _matches(value: str, patterns: set[str]) -> bool:
    value = value.strip()
    return bool(value) and any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _matches_coin_level(coin: str, level: str, patterns: set[str]) -> bool:
    return _matches(f"{coin.strip().upper()}:{level.strip().lower()}", patterns)


def _matches_coin_session(coin: str, session: str, patterns: set[str]) -> bool:
    return _matches(f"{coin.strip().upper()}:{session.strip().lower()}", patterns)


def _matches_coin_session_level(coin: str, session: str, level: str, patterns: set[str]) -> bool:
    return _matches(f"{coin.strip().upper()}:{session.strip().lower()}:{level.strip().lower()}", patterns)


def _level_family(label: str) -> str:
    if label.startswith("equal_low_"):
        return "equal_low_*"
    if label.startswith("equal_high_"):
        return "equal_high_*"
    return label


def _session(ts_ms: int) -> str:
    hour = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).hour
    if hour < 7:
        return "asia"
    if hour < 13:
        return "eu"
    if hour < 22:
        return "us"
    return "late"


def _current_session() -> str:
    return _session(int(time.time() * 1000))


def _fmt_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


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
    if open_px <= 0 or high <= 0 or low <= 0 or close <= 0 or high < low:
        return None
    return Bar(
        start_ms=start_ms,
        end_ms=end_ms,
        open=open_px,
        high=high,
        low=low,
        close=close,
        volume=max(0.0, volume),
        trade_count=max(0, trades),
        vwap=close,
        avg_spread_bps=0.1,
    )


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    if getattr(exc, "args", None):
        first = exc.args[0]
        if isinstance(first, int):
            return first
        if isinstance(first, tuple) and first and isinstance(first[0], int):
            return first[0]
    return None


def _candles_snapshot_with_retry(
    info: Any,
    coin: str,
    start_ms: int,
    end_ms: int,
    *,
    attempts: int = 4,
    base_sleep_sec: float = 1.0,
) -> Any:
    last_exc: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return info.candles_snapshot(coin, "1m", start_ms, end_ms)
        except Exception as exc:
            status = _status_code(exc)
            if status not in _RETRYABLE_STATUS_CODES or attempt >= attempts:
                raise
            last_exc = exc
            sleep_sec = base_sleep_sec * (2 ** (attempt - 1))
            print(
                f"warning: candle fetch {coin} {start_ms}->{end_ms} returned HTTP {status}; "
                f"retrying in {sleep_sec:.1f}s ({attempt}/{attempts})",
                file=sys.stderr,
            )
            time.sleep(sleep_sec)
    if last_exc is not None:
        raise last_exc
    return []


def _cache_file(cache_dir: Path, coin: str, days: int) -> Path:
    return cache_dir / f"{coin.upper()}_1m_{max(1, days)}d.json"


def _read_candle_cache(cache_dir: Path, coin: str, days: int, start_ms: int, end_ms: int, ttl_sec: int) -> tuple[list[dict[str, Any]], list[int]] | None:
    if ttl_sec <= 0:
        return None
    path = _cache_file(cache_dir, coin, days)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        fetched_ms = int(payload.get("fetched_ms", 0))
        cached_start = int(payload.get("start_ms", 0))
        cached_end = int(payload.get("end_ms", 0))
        rows = payload.get("rows", [])
        counts = payload.get("counts", [])
    except (TypeError, ValueError):
        return None
    now_ms = int(time.time() * 1000)
    if now_ms - fetched_ms > ttl_sec * 1000:
        return None
    # Permit the latest cached candle set to lag by the TTL; this avoids
    # repeated full-window downloads while keeping freshness bounded.
    if cached_start > start_ms or cached_end < end_ms - (ttl_sec * 1000):
        return None
    if not isinstance(rows, list) or not isinstance(counts, list):
        return None
    return [row for row in rows if isinstance(row, dict)], [int(c) for c in counts if isinstance(c, int)]


def _write_candle_cache(
    cache_dir: Path,
    coin: str,
    days: int,
    start_ms: int,
    end_ms: int,
    rows: list[dict[str, Any]],
    counts: list[int],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file(cache_dir, coin, days)
    payload = {
        "coin": coin.upper(),
        "interval": "1m",
        "days": max(1, days),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "fetched_ms": int(time.time() * 1000),
        "counts": counts,
        "rows": rows,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def _fetch_candles(
    info: Any,
    coin: str,
    start_ms: int,
    end_ms: int,
    *,
    days: int,
    cache_sec: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    cache_dir = ROOT / "runtime" / "candle_cache"
    cached = _read_candle_cache(cache_dir, coin, days, start_ms, end_ms, cache_sec)
    if cached is not None:
        return cached
    # Hyperliquid caps candle snapshot responses. Fetch in small chunks so
    # --days has the best chance to expand the sample, and return per-chunk
    # counts so reports reveal any upstream history limit.
    chunk_ms = 12 * 60 * 60 * 1000
    out: dict[int, dict[str, Any]] = {}
    counts: list[int] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(end_ms, cursor + chunk_ms)
        rows = _candles_snapshot_with_retry(info, coin, cursor, chunk_end)
        count = 0
        if isinstance(rows, list):
            count = len(rows)
            for row in rows:
                if isinstance(row, dict):
                    try:
                        out[int(row["t"])] = row
                    except (KeyError, TypeError, ValueError):
                        continue
        counts.append(count)
        cursor = chunk_end + 1
    rows = [out[k] for k in sorted(out)]
    if cache_sec > 0:
        _write_candle_cache(cache_dir, coin, days, start_ms, end_ms, rows, counts)
    return rows, counts


def _invert_signal(signal):
    """Return a new SweepSignal that flips side and mirrors stop/TPs around entry.

    Used by the --invert-signals experiment. Thesis: if the bot reliably stops
    out on its own signals, the inverse direction (fading the reclaim, riding
    the continuation) may have positive edge. Mirroring around entry keeps the
    R:R ratio identical so the comparison is apples-to-apples.
    """
    from hliq_bot.models import SweepSignal, Side
    e = signal.entry_price
    new_side = Side.SHORT if signal.side == Side.LONG else Side.LONG
    # Mirror: dist from entry stays the same magnitude, just flips sign.
    new_stop = e + (e - signal.stop_price)
    new_tp1  = e + (e - signal.tp1_price)
    new_tp2  = e + (e - signal.tp2_price)
    return SweepSignal(
        side=new_side,
        level=signal.level,
        level_label=signal.level_label,
        sweep_extreme=signal.sweep_extreme,
        entry_price=e,
        stop_price=new_stop,
        tp1_price=new_tp1,
        tp2_price=new_tp2,
        confidence=signal.confidence,
        reason=f"inverted({signal.reason})",
        created_ms=signal.created_ms,
        coin=signal.coin,
        overshoot_bps=signal.overshoot_bps,
        reclaim_bps=signal.reclaim_bps,
        volume_ratio=signal.volume_ratio,
        wick_ratio=signal.wick_ratio,
        signal_score=signal.signal_score,
    )


def _selected(cfg, coin: str, signal, session: str, *, ignore_operator_policy: bool = False) -> tuple[bool, str]:
    coin_key = coin.upper()
    if not ignore_operator_policy:
        if cfg.strategy.allowed_coins and not _matches(coin_key, cfg.strategy.allowed_coins):
            return False, f"allow_coin_miss:{coin_key}"
        if _matches(coin_key, cfg.strategy.blocked_coins):
            return False, f"block_coin:{coin_key}"

    level = signal.level_label.lower()
    if not ignore_operator_policy:
        if cfg.strategy.allowed_level_labels and not _matches(level, cfg.strategy.allowed_level_labels):
            return False, f"allow_level_miss:{level}"
        if _matches(level, cfg.strategy.blocked_level_labels):
            return False, f"block_level:{level}"
        if cfg.strategy.allowed_coin_level_pairs and not _matches_coin_level(coin_key, level, cfg.strategy.allowed_coin_level_pairs):
            return False, f"allow_coin_level_miss:{coin_key}:{level}"
        if _matches_coin_level(coin_key, level, cfg.strategy.blocked_coin_level_pairs):
            return False, f"block_coin_level:{coin_key}:{level}"

    if not ignore_operator_policy:
        if cfg.strategy.allowed_coin_session_pairs and not _matches_coin_session(coin_key, session, cfg.strategy.allowed_coin_session_pairs):
            return False, f"allow_coin_session_miss:{coin_key}:{session}"
        if _matches_coin_session(coin_key, session, cfg.strategy.blocked_coin_session_pairs):
            return False, f"block_coin_session:{coin_key}:{session}"
        if cfg.strategy.allowed_coin_session_level_triples and not _matches_coin_session_level(
            coin_key, session, level, cfg.strategy.allowed_coin_session_level_triples
        ):
            return False, f"allow_coin_session_level_miss:{coin_key}:{session}:{level}"
        if _matches_coin_session_level(coin_key, session, level, cfg.strategy.blocked_coin_session_level_triples):
            return False, f"block_coin_session_level:{coin_key}:{session}:{level}"
        if cfg.strategy.allowed_sessions and not _matches(session, cfg.strategy.allowed_sessions):
            return False, f"allow_session_miss:{session}"
        if _matches(session, cfg.strategy.blocked_sessions):
            return False, f"block_session:{session}"

    side = signal.side.value
    if not ignore_operator_policy:
        if cfg.strategy.allowed_sides and not _matches(side, cfg.strategy.allowed_sides):
            return False, f"allow_side_miss:{side}"
        if _matches(side, cfg.strategy.blocked_sides):
            return False, f"block_side:{side}"

    if signal.signal_score < cfg.strategy.min_signal_score:
        return False, f"signal_score<{cfg.strategy.min_signal_score:.2f}"
    conf_floor = cfg.strategy.min_confidence_range
    if signal.confidence < conf_floor:
        return False, f"confidence<{conf_floor:.2f}"
    return True, "selected"


def _apply_slippage(side: Side, price: float, slippage_bps: float, *, is_exit: bool) -> float:
    if slippage_bps <= 0:
        return price
    factor = slippage_bps / 10_000.0
    if side == Side.LONG:
        return price * (1.0 - factor) if is_exit else price * (1.0 + factor)
    return price * (1.0 + factor) if is_exit else price * (1.0 - factor)


def _net_r(
    cfg,
    side: Side,
    entry_fill: float,
    risk: float,
    exits: list[tuple[float, float, bool]],
) -> float:
    risk = max(risk, 1e-12)
    gross = 0.0
    exit_fees = 0.0
    for raw_exit, weight, is_maker in exits:
        exit_fill = raw_exit if is_maker else _apply_slippage(
            side,
            raw_exit,
            cfg.strategy.paper_exit_slippage_bps,
            is_exit=True,
        )
        if side == Side.LONG:
            gross += weight * ((exit_fill - entry_fill) / risk)
        else:
            gross += weight * ((entry_fill - exit_fill) / risk)
        fee_pct = cfg.strategy.maker_fee_pct if is_maker else cfg.strategy.taker_fee_pct
        exit_fees += exit_fill * weight * fee_pct
    entry_fee = entry_fill * cfg.strategy.maker_fee_pct
    return gross - ((entry_fee + exit_fees) / risk)


def _simulate_trade(cfg, bars: list[Bar], signal_index: int, signal) -> tuple[bool, float, str]:
    entry = signal.entry_price
    stop = signal.stop_price
    tp1 = signal.tp1_price
    tp2 = signal.tp2_price
    entry_fill = _apply_slippage(
        signal.side,
        entry,
        cfg.strategy.paper_entry_slippage_bps,
        is_exit=False,
    )
    risk = max(abs(entry - stop), abs(entry_fill - stop))
    if risk <= 0:
        return False, 0.0, "bad_risk"

    tolerance = cfg.strategy.entry_touch_tolerance_bps / 10_000.0 * entry
    expiry_bars = max(1, math.ceil(cfg.strategy.pending_entry_expiry_sec / cfg.strategy.timeframe_sec))
    fill_index = -1
    for idx in range(signal_index + 1, min(signal_index + 1 + expiry_bars, len(bars))):
        bar = bars[idx]
        if signal.side == Side.LONG and bar.low <= entry + tolerance:
            fill_index = idx
            break
        if signal.side == Side.SHORT and bar.high >= entry - tolerance:
            fill_index = idx
            break
    if fill_index < 0:
        return False, 0.0, "expired"

    def close_r(price: float) -> float:
        if signal.side == Side.LONG:
            return (price - entry_fill) / risk
        return (entry_fill - price) / risk

    hold_bars = max(1, math.ceil(cfg.strategy.max_holding_sec / cfg.strategy.timeframe_sec))
    profit_take_bars = (
        math.ceil(cfg.strategy.profit_take_sec / cfg.strategy.timeframe_sec)
        if cfg.strategy.profit_take_sec > 0
        else 0
    )
    tp1_hit = False
    active_stop = stop
    last_index = min(fill_index + hold_bars, len(bars) - 1)
    # Start iterating at fill_index+1: on the entry bar itself we cannot know
    # whether the post-fill move went stop-direction or TP-direction first,
    # so checking the entry bar's extremes would systematically over-count
    # stops (LONG: any bar where bar.low<=stop was booked as a loss even if
    # entry only filled when price was already on its way to TP).
    for idx in range(fill_index + 1, last_index + 1):
        bar = bars[idx]
        if signal.side == Side.LONG:
            if bar.low <= active_stop:
                if not tp1_hit:
                    return True, _net_r(cfg, signal.side, entry_fill, risk, [(active_stop, 1.0, False)]), "stop"
                tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
                return True, _net_r(
                    cfg,
                    signal.side,
                    entry_fill,
                    risk,
                    [(tp1, 0.5, tp1_is_maker), (active_stop, 0.5, False)],
                ), "stop_after_tp1"
            if bar.high >= tp1:
                tp1_hit = True
                active_stop = entry_fill
                # Re-check the BE stop within the SAME bar. Without this, a bar
                # that touched TP1 then retraced below entry was booked as
                # TP1→TP2 (if it also touched tp2) or TP1→time, both ignoring
                # the BE stop that would have actually fired.
                if bar.low <= active_stop:
                    tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
                    return True, _net_r(
                        cfg,
                        signal.side,
                        entry_fill,
                        risk,
                        [(tp1, 0.5, tp1_is_maker), (active_stop, 0.5, False)],
                    ), "stop_after_tp1"
            if tp1_hit and bar.high >= tp2:
                tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
                # TP2 in live mode is closed via market_close (taker fee +
                # slippage), NOT a resting maker limit — was previously
                # mis-marked as (True) which inflated avg_r for the go-live gate.
                return True, _net_r(
                    cfg,
                    signal.side,
                    entry_fill,
                    risk,
                    [(tp1, 0.5, tp1_is_maker), (tp2, 0.5, False)],
                ), "tp2"
        else:
            if bar.high >= active_stop:
                if not tp1_hit:
                    return True, _net_r(cfg, signal.side, entry_fill, risk, [(active_stop, 1.0, False)]), "stop"
                tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
                return True, _net_r(
                    cfg,
                    signal.side,
                    entry_fill,
                    risk,
                    [(tp1, 0.5, tp1_is_maker), (active_stop, 0.5, False)],
                ), "stop_after_tp1"
            if bar.low <= tp1:
                tp1_hit = True
                active_stop = entry_fill
                if bar.high >= active_stop:
                    tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
                    return True, _net_r(
                        cfg,
                        signal.side,
                        entry_fill,
                        risk,
                        [(tp1, 0.5, tp1_is_maker), (active_stop, 0.5, False)],
                    ), "stop_after_tp1"
            if tp1_hit and bar.low <= tp2:
                tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
                return True, _net_r(
                    cfg,
                    signal.side,
                    entry_fill,
                    risk,
                    [(tp1, 0.5, tp1_is_maker), (tp2, 0.5, False)],
                ), "tp2"
        if (
            profit_take_bars > 0
            and idx - fill_index >= profit_take_bars
            and close_r(bar.close) >= cfg.strategy.profit_take_min_r
        ):
            if tp1_hit:
                tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
                return True, _net_r(
                    cfg,
                    signal.side,
                    entry_fill,
                    risk,
                    [(tp1, 0.5, tp1_is_maker), (bar.close, 0.5, False)],
                ), "profit_take_after_tp1"
            return True, _net_r(
                cfg,
                signal.side,
                entry_fill,
                risk,
                [(bar.close, 1.0, False)],
            ), "profit_take"

    close = bars[last_index].close
    if signal.side == Side.LONG:
        if tp1_hit:
            tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
            return True, _net_r(
                cfg,
                signal.side,
                entry_fill,
                risk,
                [(tp1, 0.5, tp1_is_maker), (close, 0.5, False)],
            ), "time"
    else:
        if tp1_hit:
            tp1_is_maker = not cfg.strategy.paper_tp1_is_taker
            return True, _net_r(
                cfg,
                signal.side,
                entry_fill,
                risk,
                [(tp1, 0.5, tp1_is_maker), (close, 0.5, False)],
            ), "time"
    return True, _net_r(cfg, signal.side, entry_fill, risk, [(close, 1.0, False)]), "time"


def _summary(values: list[float]) -> tuple[int, float, float, float]:
    if not values:
        return 0, 0.0, 0.0, 0.0
    gross_profit = sum(v for v in values if v > 0)
    gross_loss = -sum(v for v in values if v <= 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    return len(values), sum(values) / len(values), (sum(1 for v in values if v > 0) / len(values)) * 100.0, pf


def _rank_policy_subsets(
    trades: list[dict[str, Any]],
    min_trades: int,
) -> list[tuple[int, float, float, float, tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    sessions = sorted({str(t["session"]) for t in trades})
    sides = sorted({str(t["side"]) for t in trades})
    families = sorted({str(t["family"]) for t in trades})
    results: list[tuple[int, float, float, float, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for sess_count in range(1, len(sessions) + 1):
        for sess_subset in itertools.combinations(sessions, sess_count):
            for side_count in range(1, len(sides) + 1):
                for side_subset in itertools.combinations(sides, side_count):
                    for family_count in range(1, min(4, len(families)) + 1):
                        for family_subset in itertools.combinations(families, family_count):
                            selected = [
                                float(t["r"])
                                for t in trades
                                if t["session"] in sess_subset
                                and t["side"] in side_subset
                                and t["family"] in family_subset
                            ]
                            n, avg_r, win_rate, pf = _summary(selected)
                            if n >= min_trades and avg_r > 0.0 and pf >= 1.2:
                                results.append((n, avg_r, win_rate, pf, sess_subset, side_subset, family_subset))
    return sorted(
        results,
        # Prefer enough observations, but do not let a larger weak sample bury a
        # smaller much cleaner slice.
        key=lambda row: (row[1] * min(row[0] / 10.0, 1.0), row[3], row[0]),
        reverse=True,
    )


def _positive_sample(values: list[float], min_trades: int) -> bool:
    n, avg_r, _win_rate, pf = _summary(values)
    return n >= min_trades and avg_r > 0.0 and pf >= 1.2


def _passes_recommendation_sample_gates(
    trades: list[dict[str, Any]],
    sessions: tuple[str, ...],
    sides: tuple[str, ...],
    families: tuple[str, ...],
    *,
    min_session_trades: int,
    min_level_trades: int,
) -> bool:
    selected = [
        row
        for row in trades
        if row["session"] in sessions and row["side"] in sides and row["family"] in families
    ]
    for session in sessions:
        if not _positive_sample([float(row["r"]) for row in selected if row["session"] == session], min_session_trades):
            return False
    for family in families:
        if not _positive_sample([float(row["r"]) for row in selected if row["family"] == family], min_level_trades):
            return False
    return True


def _print_policy_search(trades: list[dict[str, Any]], min_trades: int) -> list[tuple[int, float, float, float, tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    ranked = _rank_policy_subsets(trades, min_trades)
    if not ranked:
        print(f"No positive subsets with trades>={min_trades}.")
        return []
    print()
    print("Top Policy Subsets")
    print("------------------")
    for n, avg_r, win_rate, pf, sess_subset, side_subset, family_subset in ranked[:12]:
        pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
        print(
            "trades={n} avg_r={avg_r:.3f} win_rate={win_rate:.1f}% pf={pf} "
            "sessions={sessions} sides={sides} levels={levels}".format(
                n=n,
                avg_r=avg_r,
                win_rate=win_rate,
                pf=pf_text,
                sessions=",".join(sess_subset),
                sides=",".join(side_subset),
                levels=",".join(family_subset),
            )
        )
    return ranked


def _print_recommended_env(
    rec: tuple[str, int, float, float, float, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None,
) -> None:
    print()
    print("Recommended Env Candidate")
    print("-------------------------")
    if rec is None:
        print("No positive recommendation met the search constraints.")
        return
    coin, n, avg_r, win_rate, pf, sessions, sides, levels = rec
    pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
    print(f"# evidence: trades={n} avg_r={avg_r:.3f} win_rate={win_rate:.1f}% pf={pf_text}")
    print(f"HL_COINS={coin}")
    print(f"BOT_ALLOW_COINS={coin}")
    print(f"BOT_ALLOW_SESSIONS={','.join(sessions)}")
    print(f"BOT_ALLOW_SIDES={','.join(sides)}")
    print(f"BOT_ALLOW_LEVEL_LABELS={','.join(levels)}")


def _best_recommendation(
    coin: str,
    trade_rows: list[dict[str, Any]],
    min_trades: int,
    *,
    current_session: str = "",
    min_coin_trades: int = 1,
    min_session_trades: int = 1,
    min_level_trades: int = 1,
) -> tuple[str, int, float, float, float, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    rows = trade_rows
    if current_session:
        rows = [row for row in trade_rows if row["session"] == current_session]
    ranked = _rank_policy_subsets(rows, min_trades)
    for n, avg_r, win_rate, pf, sessions, sides, families in ranked:
        if n < min_coin_trades:
            continue
        if not _passes_recommendation_sample_gates(
            rows,
            sessions,
            sides,
            families,
            min_session_trades=min_session_trades,
            min_level_trades=min_level_trades,
        ):
            continue
        return (coin, n, avg_r, win_rate, pf, sessions, sides, families)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Approximate current policy expectancy from Hyperliquid candles.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--coins", default="", help="Comma-separated override; default uses HL_COINS.")
    parser.add_argument("--search", action="store_true", help="Rank positive session/side/level policy subsets.")
    parser.add_argument("--search-min-trades", type=int, default=5)
    parser.add_argument(
        "--ignore-operator-policy",
        action="store_true",
        help="Ignore allow/block coin/session/side/level policy for analysis; score/confidence floors still apply.",
    )
    parser.add_argument("--recommend-env", action="store_true", help="Print the strongest .env candidate from search results.")
    parser.add_argument("--recommend-min-coin-trades", type=int, default=5)
    parser.add_argument("--recommend-min-session-trades", type=int, default=2)
    parser.add_argument("--recommend-min-level-trades", type=int, default=2)
    parser.add_argument(
        "--search-current-session",
        action="store_true",
        help="Restrict printed search subsets to the current UTC strategy session.",
    )
    parser.add_argument(
        "--recommend-current-session",
        action="store_true",
        help="Only recommend policies whose session set includes the current UTC strategy session.",
    )
    parser.add_argument(
        "--cache-sec",
        type=int,
        default=int(os.getenv("BOT_CANDLE_CACHE_SEC", "300") or "300"),
        help="Reuse recent raw candle downloads for this many seconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--slice-limit",
        type=int,
        default=8,
        help="Maximum per-coin slice rows to print. Use 0 to print every slice for machine gates.",
    )
    parser.add_argument(
        "--invert-signals",
        action="store_true",
        help="EXPERIMENTAL: flip side and mirror stop/TPs around entry on every "
        "signal before simulation. Use to test the 'fade the reclaim, ride the "
        "continuation' thesis when the baseline strategy is reliably stopping out.",
    )
    args = parser.parse_args()

    _load_env(ROOT / args.env_file)
    cfg = load_config()

    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    api_url = constants.TESTNET_API_URL if "testnet" in cfg.feed.ws_url.lower() else constants.MAINNET_API_URL
    info = Info(api_url, skip_ws=True)
    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()] or cfg.feed.coins
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(1, args.days) * 24 * 60 * 60 * 1000

    all_r: list[float] = []
    recommendations: list[tuple[str, int, float, float, float, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    max_span_days = 0.0
    print("Candle Backtest")
    print("=" * 40)
    print(f"days: {args.days}")
    print(f"coins: {coins}")
    print("note: OHLC ordering is conservative when stop/target share a candle; R includes configured fees/slippage; use as a filter, not proof.")

    for coin in coins:
        raw, fetch_counts = _fetch_candles(
            info,
            coin,
            start_ms,
            end_ms,
            days=max(1, args.days),
            cache_sec=max(0, args.cache_sec),
        )
        bars = [bar for row in raw if (bar := _bar_from_row(row)) is not None]
        st = SessionTracker()
        vt = VWAPTracker()
        detector = SweepDetector(cfg.strategy, cfg.levels, st, vt, coin)
        r_values: list[float] = []
        candidates = 0
        selected = 0
        filled = 0
        reject_counts: dict[str, int] = {}
        by_slice: dict[str, list[float]] = {}
        trade_rows: list[dict[str, Any]] = []

        for idx, bar in enumerate(bars[:-1]):
            st.on_bar(bar)
            vt.on_bar(bar)
            signal = detector.on_bar(bar)
            if signal is None:
                continue
            candidates += 1
            if args.invert_signals:
                signal = _invert_signal(signal)
            sess = _session(signal.created_ms)
            ok, reason = _selected(cfg, coin, signal, sess, ignore_operator_policy=args.ignore_operator_policy)
            if not ok:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            selected += 1
            did_fill, r_val, outcome = _simulate_trade(cfg, bars, idx, signal)
            if not did_fill:
                reject_counts[outcome] = reject_counts.get(outcome, 0) + 1
                continue
            filled += 1
            r_values.append(r_val)
            slice_key = f"{sess}:{signal.side.value}:{_level_family(signal.level_label)}"
            by_slice.setdefault(slice_key, []).append(r_val)
            trade_rows.append(
                {
                    "r": r_val,
                    "coin": coin,
                    "session": sess,
                    "side": signal.side.value,
                    "family": _level_family(signal.level_label),
                    "level": signal.level_label,
                    "outcome": outcome,
                }
            )

        all_r.extend(r_values)
        n, avg_r, win_rate, pf = _summary(r_values)
        pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
        print()
        if bars:
            span_days = (bars[-1].end_ms - bars[0].start_ms) / (24 * 60 * 60 * 1000)
            max_span_days = max(max_span_days, span_days)
            print(
                f"{coin}: bars={len(bars)} span_days={span_days:.1f} "
                f"from={_fmt_utc(bars[0].start_ms)} to={_fmt_utc(bars[-1].end_ms)} "
                f"candidates={candidates} selected={selected} filled={filled}"
            )
        else:
            print(f"{coin}: bars=0 candidates={candidates} selected={selected} filled={filled}")
        print(f"{coin}: trades={n} avg_r={avg_r:.3f} win_rate={win_rate:.1f}% profit_factor={pf_text}")
        if fetch_counts:
            nonempty = sum(1 for count in fetch_counts if count > 0)
            empty = len(fetch_counts) - nonempty
            print(
                f"{coin}: candle_fetch_chunks={len(fetch_counts)} nonempty={nonempty} empty={empty} "
                f"max_chunk_rows={max(fetch_counts)}"
            )
        if reject_counts:
            top = sorted(reject_counts.items(), key=lambda kv: -kv[1])[:8]
            print("top_rejects: " + ", ".join(f"{k}={v}" for k, v in top))
        slice_rows = sorted(by_slice.items(), key=lambda kv: _summary(kv[1])[1], reverse=True)
        if args.slice_limit > 0:
            slice_rows = slice_rows[: args.slice_limit]
        for key, vals in slice_rows:
            sn, savg, swr, spf = _summary(vals)
            spf_text = "inf" if math.isinf(spf) else f"{spf:.2f}"
            print(f"slice {key}: trades={sn} avg_r={savg:.3f} win_rate={swr:.1f}% pf={spf_text}")
        search_rows = trade_rows
        current_session = _current_session() if (args.search_current_session or args.recommend_current_session) else ""
        if current_session and args.search_current_session:
            search_rows = [row for row in trade_rows if row["session"] == current_session]
        if args.search:
            ranked = _print_policy_search(search_rows, max(1, args.search_min_trades))
            if ranked:
                rec = _best_recommendation(
                    coin,
                    trade_rows,
                    max(1, args.search_min_trades),
                    current_session=current_session if args.recommend_current_session else "",
                    min_coin_trades=max(1, args.recommend_min_coin_trades),
                    min_session_trades=max(1, args.recommend_min_session_trades),
                    min_level_trades=max(1, args.recommend_min_level_trades),
                )
                if rec is not None:
                    recommendations.append(rec)

    n, avg_r, win_rate, pf = _summary(all_r)
    pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
    print()
    print(
        f"TOTAL: trades={n} avg_r={avg_r:.3f} win_rate={win_rate:.1f}% "
        f"profit_factor={pf_text} span_days={max_span_days:.1f}"
    )
    if n < 20:
        print("sample_warning: candle trades<20 (hypothesis only)")
    if args.recommend_env:
        best = None
        if recommendations:
            best = sorted(
                recommendations,
                key=lambda row: (row[2] * min(row[1] / 10.0, 1.0), row[4], row[1]),
                reverse=True,
            )[0] if recommendations else None
        _print_recommended_env(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
