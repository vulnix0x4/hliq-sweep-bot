#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Agg:
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0
    r_sum: float = 0.0

    def add(self, pnl: float, r_val: float) -> None:
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        self.pnl += pnl
        self.r_sum += r_val

    @property
    def win_rate(self) -> float:
        if self.trades <= 0:
            return 0.0
        return (self.wins / self.trades) * 100.0

    @property
    def avg_r(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.r_sum / self.trades


@dataclass(slots=True)
class ProbBucketAgg:
    trades: int = 0
    wins: int = 0
    r_sum: float = 0.0

    def add(self, r_val: float) -> None:
        self.trades += 1
        if r_val > 0:
            self.wins += 1
        self.r_sum += r_val

    @property
    def win_rate(self) -> float:
        if self.trades <= 0:
            return 0.0
        return (self.wins / self.trades) * 100.0

    @property
    def avg_r(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.r_sum / self.trades


def _read_rows(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _last_run_id(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        if row.get("event_type") == "run" and row.get("event") == "run_start":
            run_id = str(row.get("run_id", "")).strip()
            if run_id:
                return run_id
    return ""


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str = "",
    since_ts_ms: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if run_id and str(row.get("run_id", "")).strip() != run_id:
            continue
        if since_ts_ms > 0 and _safe_int(row.get("ts_ms")) < since_ts_ms:
            continue
        out.append(row)
    return out


def _fmt_rate(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{(num / den) * 100.0:.1f}%"


def _print_top_aggs(title: str, aggs: dict[str, Agg], n: int = 8) -> None:
    if not aggs:
        return
    print()
    print(title)
    print("-" * len(title))
    ranked = sorted(aggs.items(), key=lambda kv: kv[1].trades, reverse=True)[:n]
    for key, a in ranked:
        print(
            f"- {key}: trades={a.trades} win_rate={a.win_rate:.1f}% avg_r={a.avg_r:.3f} pnl={a.pnl:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze signal journal funnel and performance.")
    parser.add_argument("--input", default="runtime/signals.jsonl")
    parser.add_argument("--run-id", default="", help="Only include rows for this run_id.")
    parser.add_argument("--last-run", action="store_true", help="Only include rows for the latest journaled run_start.")
    parser.add_argument("--since-ts-ms", type=int, default=0, help="Only include rows at/after this timestamp.")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    if not rows:
        print(f"No journal rows found at {args.input}")
        return 0
    selected_run_id = args.run_id.strip()
    if args.last_run:
        selected_run_id = _last_run_id(rows) or selected_run_id
    rows = _filter_rows(rows, run_id=selected_run_id, since_ts_ms=max(0, args.since_ts_ms))
    if not rows:
        print(f"No journal rows matched filters at {args.input}")
        return 0

    by_id: dict[str, dict[str, Any]] = defaultdict(dict)
    denied_reasons = Counter()
    lifecycle_events = Counter()

    for row in rows:
        sid = str(row.get("signal_id", ""))
        if not sid:
            continue
        et = str(row.get("event_type", ""))
        if et == "candidate":
            by_id[sid]["candidate"] = row
        elif et == "decision":
            by_id[sid]["decision"] = row
            if not bool(row.get("allowed", False)):
                denied_reasons[str(row.get("reason", "unknown"))] += 1
        elif et == "outcome":
            by_id[sid]["outcome"] = row
        elif et == "lifecycle":
            evt = str(row.get("event", "unknown"))
            lifecycle_events[evt] += 1

    candidates = 0
    decisions = 0
    allowed = 0
    closed = 0
    wins = 0
    net_pnl = 0.0
    r_sum = 0.0

    agg_regime: dict[str, Agg] = defaultdict(Agg)
    agg_session: dict[str, Agg] = defaultdict(Agg)
    agg_level: dict[str, Agg] = defaultdict(Agg)
    ml_prob_buckets: dict[str, ProbBucketAgg] = defaultdict(ProbBucketAgg)
    ml_accepted_probs: list[float] = []
    ml_accepted_thresholds: list[float] = []
    signal_scores: list[float] = []
    closed_signal_scores: list[float] = []
    mfe_vals: list[float] = []
    mae_vals: list[float] = []
    bar_range_vals: list[float] = []
    move_30s_vals: list[float] = []
    abs_r_vals: list[float] = []
    allow_by_regime: Counter = Counter()
    cand_by_regime: Counter = Counter()

    for sid, rec in by_id.items():
        cand = rec.get("candidate")
        dec = rec.get("decision")
        out = rec.get("outcome")

        if cand is not None:
            candidates += 1
            regime = str(cand.get("regime", "unknown"))
            cand_by_regime[regime] += 1
            try:
                signal_scores.append(float(cand.get("signal_score", 0.0)))
            except (TypeError, ValueError):
                pass
            features = cand.get("features") or {}
            if isinstance(features, dict):
                try:
                    bar_range_vals.append(abs(float(features.get("bar_range_pct", 0.0))))
                except (TypeError, ValueError):
                    pass
                try:
                    move_30s_vals.append(abs(float(features.get("move_30s_pct", 0.0))))
                except (TypeError, ValueError):
                    pass
        if dec is not None:
            decisions += 1
            if bool(dec.get("allowed", False)):
                allowed += 1
                ml_prob = dec.get("ml_prob")
                if ml_prob is not None:
                    try:
                        ml_accepted_probs.append(float(ml_prob))
                    except (TypeError, ValueError):
                        pass
                ml_thr = dec.get("ml_threshold")
                if ml_thr is not None:
                    try:
                        ml_accepted_thresholds.append(float(ml_thr))
                    except (TypeError, ValueError):
                        pass
                if cand is not None:
                    allow_by_regime[str(cand.get("regime", "unknown"))] += 1
        if out is not None:
            closed += 1
            pnl = float(out.get("pnl", 0.0))
            r_val = float(out.get("r_multiple", 0.0))
            net_pnl += pnl
            r_sum += r_val
            abs_r_vals.append(abs(r_val))
            if pnl > 0:
                wins += 1

            regime = "unknown"
            session = "unknown"
            level = "unknown"
            if cand is not None:
                regime = str(cand.get("regime", "unknown"))
                session = str(cand.get("session", "unknown"))
                level = str(cand.get("level_label", "unknown"))
                try:
                    closed_signal_scores.append(float(cand.get("signal_score", 0.0)))
                except (TypeError, ValueError):
                    pass
            agg_regime[regime].add(pnl, r_val)
            agg_session[session].add(pnl, r_val)
            agg_level[level].add(pnl, r_val)
            try:
                mfe_vals.append(float(out.get("mfe_pnl", 0.0)))
            except (TypeError, ValueError):
                pass
            try:
                mae_vals.append(float(out.get("mae_pnl", 0.0)))
            except (TypeError, ValueError):
                pass
            if dec is not None and bool(dec.get("allowed", False)):
                try:
                    ml_prob = float(dec.get("ml_prob", out.get("ml_prob", 0.0)))
                except (TypeError, ValueError):
                    ml_prob = 0.0
                if math.isfinite(ml_prob) and ml_prob > 0.0:
                    lo = math.floor(ml_prob * 10.0) / 10.0
                    lo = max(0.0, min(0.9, lo))
                    hi = lo + 0.1
                    label = f"{lo:.1f}-{hi:.1f}"
                    ml_prob_buckets[label].add(r_val)

    placed = lifecycle_events.get("entry_placed", 0)
    filled = lifecycle_events.get("entry_filled", 0)
    expired = lifecycle_events.get("order_canceled", 0)
    partial = lifecycle_events.get("partial_tp", 0)

    print("Journal Report")
    print("========================================")
    if selected_run_id:
        print(f"run_id: {selected_run_id}")
    if args.since_ts_ms > 0:
        print(f"since_ts_ms: {args.since_ts_ms}")
    print(f"journal_rows: {len(rows)}")
    print(f"signals_candidate: {candidates}")
    print(f"signals_decided: {decisions}")
    print(f"signals_allowed: {allowed} ({_fmt_rate(allowed, decisions)})")
    print(f"entries_placed: {placed}")
    print(f"entries_filled: {filled} ({_fmt_rate(filled, placed)})")
    print(f"entries_expired: {expired} ({_fmt_rate(expired, placed)})")
    print(f"tp1_partial: {partial}")
    print(f"closed_trades: {closed}")

    if closed > 0:
        print(f"net_pnl: {net_pnl:.2f}")
        print(f"win_rate: {_fmt_rate(wins, closed)}")
        print(f"avg_r_per_trade: {r_sum / closed:.3f}")
    if ml_accepted_probs:
        avg_prob = sum(ml_accepted_probs) / len(ml_accepted_probs)
        print(f"avg_ml_prob_allowed: {avg_prob:.3f}")
    if ml_accepted_thresholds:
        avg_thr = sum(ml_accepted_thresholds) / len(ml_accepted_thresholds)
        print(f"avg_ml_threshold_allowed: {avg_thr:.3f}")
    if signal_scores:
        print(f"avg_signal_score_candidate: {sum(signal_scores) / len(signal_scores):.3f}")
    if closed_signal_scores:
        print(f"avg_signal_score_closed: {sum(closed_signal_scores) / len(closed_signal_scores):.3f}")
    if mfe_vals:
        print(f"avg_mfe_pnl: {sum(mfe_vals) / len(mfe_vals):.2f}")
    if mae_vals:
        print(f"avg_mae_pnl: {sum(mae_vals) / len(mae_vals):.2f}")

    sanity_warnings: list[str] = []
    max_abs_r = max(abs_r_vals) if abs_r_vals else 0.0
    max_bar_range = max(bar_range_vals) if bar_range_vals else 0.0
    max_move_30s = max(move_30s_vals) if move_30s_vals else 0.0
    if max_abs_r > 25.0:
        sanity_warnings.append(f"max_abs_r={max_abs_r:.3f} exceeds 25; replay/outcome data may be distorted")
    if max_bar_range > 10.0:
        sanity_warnings.append(
            f"max_bar_range_pct={max_bar_range:.3f} exceeds 10%; captured prices may have scale jumps"
        )
    if max_move_30s > 10.0:
        sanity_warnings.append(
            f"max_abs_move_30s_pct={max_move_30s:.3f} exceeds 10%; captured prices may have scale jumps"
        )
    if sanity_warnings:
        print()
        print("Data sanity warnings")
        print("--------------------")
        for warning in sanity_warnings:
            print(f"- {warning}")

    if denied_reasons:
        print()
        print("Top block reasons")
        print("-----------------")
        for reason, n in denied_reasons.most_common(10):
            print(f"- {reason}: {n}")

    if cand_by_regime:
        print()
        print("Allow rate by regime")
        print("--------------------")
        for regime, total in cand_by_regime.items():
            ok = allow_by_regime.get(regime, 0)
            print(f"- {regime}: allowed={ok}/{total} ({_fmt_rate(ok, total)})")

    _print_top_aggs("Expectancy by regime", agg_regime, n=6)
    _print_top_aggs("Expectancy by session", agg_session, n=6)
    _print_top_aggs("Expectancy by level label", agg_level, n=10)

    if ml_prob_buckets:
        print()
        print("ML prob bucket expectancy")
        print("------------------------")
        for bucket in sorted(ml_prob_buckets):
            a = ml_prob_buckets[bucket]
            print(f"- {bucket}: trades={a.trades} win_rate={a.win_rate:.1f}% avg_r={a.avg_r:.3f}")

    hints: list[str] = []
    if placed >= 6 and filled / max(placed, 1) < 0.45:
        hints.append("Fill conversion is weak: raise BOT_ENTRY_TOUCH_TOL_BPS or BOT_ENTRY_EXPIRY_SEC slightly.")
    if closed >= 8 and (r_sum / closed) < 0:
        hints.append("Negative expectancy: increase BOT_MIN_CONF_RANGE / BOT_MIN_RECLAIM_BPS and reduce BOT_RISK_PER_TRADE_PCT.")
    if closed >= 8 and agg_regime.get("trend", Agg()).avg_r < agg_regime.get("range", Agg()).avg_r:
        hints.append("Trend regime underperforming: raise BOT_MIN_CONF_TREND or keep trend risk multiplier below 1.0.")
    if denied_reasons.get("loss cooldown", 0) > 0:
        hints.append("Loss cooldown triggered: this is expected in rough tape; do not widen risk until expectancy recovers.")

    if hints:
        print()
        print("Tuning hints")
        print("-----------")
        for hint in hints:
            print(f"- {hint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
