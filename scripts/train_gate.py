#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


def _read_events(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _build_dataset(events: list[dict[str, Any]]) -> tuple[list[dict[str, float]], list[int], list[float], list[str]]:
    by_id: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in events:
        sid = str(row.get("signal_id", ""))
        if not sid:
            continue
        et = row.get("event_type")
        if et == "candidate":
            by_id[sid]["features"] = row.get("features", {})
            by_id[sid]["ts_ms"] = int(row.get("ts_ms", 0))
        elif et == "decision":
            by_id[sid]["decision"] = bool(row.get("allowed", False))
        elif et == "outcome":
            by_id[sid]["r"] = float(row.get("r_multiple", 0.0))

    rows: list[tuple[int, str, dict[str, float], int, float]] = []
    for sid, rec in by_id.items():
        feats = rec.get("features")
        decision = rec.get("decision")
        r_mult = rec.get("r")
        if not isinstance(feats, dict):
            continue
        if decision is not True:
            continue
        if r_mult is None:
            continue

        xrow: dict[str, float] = {}
        for k, v in feats.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            xrow[k] = fv
        y = 1 if float(r_mult) > 0.0 else 0
        rows.append((int(rec.get("ts_ms", 0)), sid, xrow, y, float(r_mult)))

    rows.sort(key=lambda x: (x[0], x[1]))
    xs = [row[2] for row in rows]
    ys = [row[3] for row in rows]
    rs = [row[4] for row in rows]
    ids = [row[1] for row in rows]
    return xs, ys, rs, ids


def _fit_logistic(xs: list[dict[str, float]], ys: list[int], epochs: int, lr: float, l2: float) -> dict[str, Any]:
    features = sorted({k for row in xs for k in row})
    if not features:
        raise ValueError("No feature columns found.")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for f in features:
        vals = [row.get(f, 0.0) for row in xs]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals))
        sigma = math.sqrt(var) if var > 1e-12 else 1.0
        means[f] = mu
        stds[f] = sigma

    w = {f: 0.0 for f in features}
    b = 0.0

    for _ in range(max(1, epochs)):
        grad_w = {f: 0.0 for f in features}
        grad_b = 0.0
        n = len(xs)
        for row, y in zip(xs, ys):
            z = b
            for f in features:
                x = (row.get(f, 0.0) - means[f]) / stds[f]
                z += w[f] * x
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y
            grad_b += err
            for f in features:
                x = (row.get(f, 0.0) - means[f]) / stds[f]
                grad_w[f] += err * x

        grad_b /= n
        for f in features:
            grad_w[f] = (grad_w[f] / n) + l2 * w[f]
            w[f] -= lr * grad_w[f]
        b -= lr * grad_b

    return {
        "model_type": "logistic_v1",
        "features": features,
        "means": means,
        "stds": stds,
        "weights": w,
        "intercept": b,
        "train_samples": len(xs),
        "positive_rate": (sum(ys) / len(ys)) if ys else 0.0,
    }


def _predict_prob(model: dict[str, Any], row: dict[str, float]) -> float:
    z = float(model.get("intercept", 0.0))
    features = list(model.get("features", []))
    means = dict(model.get("means", {}))
    stds = dict(model.get("stds", {}))
    weights = dict(model.get("weights", {}))
    for f in features:
        mu = float(means.get(f, 0.0))
        sigma = float(stds.get(f, 1.0))
        if sigma <= 1e-9:
            sigma = 1.0
        x = (float(row.get(f, 0.0)) - mu) / sigma
        z += float(weights.get(f, 0.0)) * x
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _split_idx(n: int) -> int:
    if n <= 1:
        return n
    return max(1, min(n - 1, int(n * 0.8)))


def _optimize_threshold(probs: list[float], rs: list[float], base: float) -> tuple[float, dict[str, float | int]]:
    n = len(probs)
    if n <= 0:
        return base, {
            "val_samples": 0,
            "selected": 0,
            "selected_rate": 0.0,
            "selected_avg_r": 0.0,
            "selected_win_rate": 0.0,
            "brier": 0.0,
        }

    wins = [1.0 if r > 0.0 else 0.0 for r in rs]
    brier = sum((p - y) ** 2 for p, y in zip(probs, wins)) / n
    min_selected = max(5, int(0.08 * n))
    best_thr = base
    best_score = float("-inf")
    best_stats: dict[str, float | int] = {}

    for thr in [round(x / 100.0, 2) for x in range(50, 91)]:
        chosen = [r for p, r in zip(probs, rs) if p >= thr]
        if len(chosen) < min_selected:
            continue
        coverage = len(chosen) / n
        avg_r = sum(chosen) / len(chosen)
        win_rate = sum(1.0 if r > 0.0 else 0.0 for r in chosen) / len(chosen)
        score = avg_r * math.sqrt(max(coverage, 1e-9))
        if score > best_score:
            best_score = score
            best_thr = thr
            best_stats = {
                "val_samples": n,
                "selected": len(chosen),
                "selected_rate": coverage,
                "selected_avg_r": avg_r,
                "selected_win_rate": win_rate,
                "brier": brier,
            }

    if not best_stats:
        chosen = [r for p, r in zip(probs, rs) if p >= base]
        sel_n = len(chosen)
        best_stats = {
            "val_samples": n,
            "selected": sel_n,
            "selected_rate": (sel_n / n) if n > 0 else 0.0,
            "selected_avg_r": (sum(chosen) / sel_n) if sel_n > 0 else 0.0,
            "selected_win_rate": (
                sum(1.0 if r > 0.0 else 0.0 for r in chosen) / sel_n if sel_n > 0 else 0.0
            ),
            "brier": brier,
        }
    return best_thr, best_stats


def _round_floats(obj: dict[str, float | int]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for k, v in obj.items():
        if isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


def _clamp_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train lightweight ML gate model from journaled signals.")
    parser.add_argument("--input", default="runtime/signals.jsonl")
    parser.add_argument("--output", default="runtime/models/gate_model.json")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--default-min-prob", type=float, default=0.62)
    parser.add_argument("--min-samples", type=int, default=30)
    args = parser.parse_args()

    events = _read_events(args.input)
    xs, ys, rs, _ = _build_dataset(events)
    min_samples = max(1, int(args.min_samples))
    if len(xs) < min_samples:
        raise SystemExit(f"Need at least {min_samples} resolved trades to train. Found {len(xs)}.")

    split = _split_idx(len(xs))
    train_x = xs[:split]
    train_y = ys[:split]
    val_x = xs[split:]
    val_r = rs[split:]

    model = _fit_logistic(train_x, train_y, epochs=args.epochs, lr=args.lr, l2=args.l2)
    if val_x:
        val_probs = [_predict_prob(model, row) for row in val_x]
        best_thr, val_stats = _optimize_threshold(
            probs=val_probs,
            rs=val_r,
            base=_clamp_threshold(args.default_min_prob),
        )
        model["recommended_min_prob"] = round(_clamp_threshold(best_thr), 2)
        model["validation"] = _round_floats(val_stats)
    else:
        model["recommended_min_prob"] = round(_clamp_threshold(args.default_min_prob), 2)
        model["validation"] = {}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(out),
                "train_samples": model["train_samples"],
                "positive_rate": round(model["positive_rate"], 4),
                "recommended_min_prob": model["recommended_min_prob"],
                "validation": model["validation"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
