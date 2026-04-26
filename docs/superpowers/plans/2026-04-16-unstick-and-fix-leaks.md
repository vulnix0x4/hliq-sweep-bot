# Unstick The Bot + Plug The Biggest Leaks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume trading (the bot has been self-muted since early March) and capture a materially larger fraction of the MFE the detector already generates.

**Architecture:** Five surgical changes, all low-risk and locally testable. No new features, no new strategies. Every change either fixes a bug, loosens an over-tight gate, or corrects a parameter whose data-grounded wrongness we've already measured.

**Tech Stack:** Python 3.11+, stdlib + websockets. Tests via pytest with `PYTHONPATH=src`. Bot runs in Docker compose; restart via `./scripts/botctl.sh down && ./scripts/botctl.sh up`.

---

## Context (evidence that justified each change)

- 59 closed trades, +4.11R total, **MFE capture rate 6.3%** ($131.57 favorable → $8.23 kept).
- `time_stop` (240s) exits 35/59 trades with avg MFE $+1.26 and avg R −0.10. All 6 winners reached `max_hold` (1800s).
- 3/4 UTC sessions permanently paused on 4-trade windows from 2026-03-06. One −1R BTC short from that day alone has blocked 553 US-session candidates.
- `wick_ratio` has a numerical artifact on doji bars (body floor 1e-9 → ratios of 10^10), polluting features and saturating the (disabled) ML gate.
- `market_capture.py` drops the `coin` field, making multi-coin replay unreliable for data >2026-04-06.

## Task 1 — Governor: clear session/level deques on UTC day roll

**Files:**
- Modify: `src/hliq_bot/risk/governor.py:269-272`
- Test:   `tests/test_risk_governor.py`

- [ ] **Step 1.1: Write failing test** — verify that crossing UTC midnight clears per-session and per-level R deques

```python
def test_roll_day_clears_session_and_level_deques() -> None:
    gov = RiskGovernor(
        risk_cfg=RiskConfig(
            account_equity=1000.0,
            risk_per_trade_pct=1.0,
            session_edge_pause_avg_r=-0.2,
            session_edge_pause_min_trades=2,
            level_edge_pause_avg_r=-0.2,
            level_edge_pause_min_trades=2,
            loss_cooldown_sec=0,
            edge_pause_min_trades=50,
            daily_loss_limit_r=20.0,
        ),
        strategy_cfg=StrategyConfig(),
    )
    day1 = 1_700_000_000_000  # UTC timestamp inside day 1
    gov.register_closed_trade(
        _closed_trade(day1 + 1_000, r_multiple=-0.5),
        session="us",
        level_label="prior_15m_low",
    )
    gov.register_closed_trade(
        _closed_trade(day1 + 2_000, r_multiple=-0.4),
        session="us",
        level_label="prior_15m_low",
    )

    # Before day roll, both gates block
    assert gov.can_trade_session("us").allowed is False
    assert gov.can_trade_level("prior_15m_low").allowed is False

    # Advance more than one UTC day
    day3 = day1 + 2 * 86_400_000
    gov.can_open_new_trade(_market_state(ts_ms=day3))

    # After day roll, both gates release
    assert gov.can_trade_session("us").allowed is True
    assert gov.can_trade_level("prior_15m_low").allowed is True
```

- [ ] **Step 1.2: Run test, confirm FAIL** (`pytest tests/test_risk_governor.py::test_roll_day_clears_session_and_level_deques -v`)
- [ ] **Step 1.3: Patch `_roll_day`**

```python
def _roll_day(self, ts_ms: int) -> None:
    day_key = self._day_key_from_ms(ts_ms)
    if day_key != self._daily.day_key:
        self._daily = _DailyStats(day_key=day_key)
        self._recent_r_by_session.clear()
        self._recent_r_by_level.clear()
```

- [ ] **Step 1.4: Run test, confirm PASS**
- [ ] **Step 1.5: Run full suite to confirm no regression**

## Task 2 — Detector: fix wick_ratio numerical artifact on doji bars

**Files:**
- Modify: `src/hliq_bot/signal/sweep_detector.py:310-318`
- Test:   `tests/test_sweep_detector.py`

- [ ] **Step 2.1: Write failing tests**

```python
def test_wick_ratio_is_bounded_on_doji_bars() -> None:
    det = SweepDetector(StrategyConfig())
    # Doji: open == close exactly. Prior floor 1e-9 caused ratios ~1e10.
    doji = _bar(0, o=100.0, h=100.5, l=99.5, c=100.0, v=10.0)
    long_ratio = det._wick_ratio_long(doji)
    short_ratio = det._wick_ratio_short(doji)
    assert long_ratio <= 20.0
    assert short_ratio <= 20.0


def test_wick_ratio_still_sensible_on_normal_bars() -> None:
    det = SweepDetector(StrategyConfig())
    # Clear bullish rejection: close well above open with tall lower wick
    bar = _bar(0, o=100.0, h=100.3, l=99.0, c=100.5, v=10.0)
    long_ratio = det._wick_ratio_long(bar)
    assert 1.5 <= long_ratio <= 20.0
```

- [ ] **Step 2.2: Run tests, confirm FAIL** (long_ratio will be ~1e9 on the doji)
- [ ] **Step 2.3: Patch the two wick_ratio methods**

```python
def _wick_ratio_short(self, bar: Bar) -> float:
    body = abs(bar.close - bar.open)
    body_floor = max((bar.high - bar.low) * 0.01, 1e-9)
    body = max(body, body_floor)
    upper_wick = max(0.0, bar.high - max(bar.open, bar.close))
    return min(upper_wick / body, 20.0)

def _wick_ratio_long(self, bar: Bar) -> float:
    body = abs(bar.close - bar.open)
    body_floor = max((bar.high - bar.low) * 0.01, 1e-9)
    body = max(body, body_floor)
    lower_wick = max(0.0, min(bar.open, bar.close) - bar.low)
    return min(lower_wick / body, 20.0)
```

Rationale: using 1% of the bar range as the body floor means a doji's ratio caps at 100 before the 20-cap; the 20-cap then protects any remaining pathological input. 20 is high enough that real strong-rejection candles still rank above weak ones (true rejections are 2–8x in the data).

- [ ] **Step 2.4: Run tests, confirm PASS**
- [ ] **Step 2.5: Run full suite**

## Task 3 — Capture: persist `coin` field end-to-end

**Files:**
- Modify: `src/hliq_bot/analytics/market_capture.py:22-47`
- Modify: `src/hliq_bot/replay/loader.py:10-64`
- Test:   create `tests/test_market_capture.py`

- [ ] **Step 3.1: Write failing test for roundtrip**

```python
# tests/test_market_capture.py
from __future__ import annotations

from pathlib import Path

from hliq_bot.analytics.market_capture import MarketCaptureWriter
from hliq_bot.models import BookTopEvent, MarketEvent, TradeEvent
from hliq_bot.replay.loader import load_market_events


def test_capture_writes_coin_field_and_loader_passes_it_through(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = MarketCaptureWriter(path=str(path))

    writer.write(MarketEvent(
        kind="trade",
        ts_ms=1_700_000_000_000,
        coin="ETH",
        trade=TradeEvent(ts_ms=1_700_000_000_000, price=2350.0, size=0.1, side="buy"),
    ))
    writer.write(MarketEvent(
        kind="book",
        ts_ms=1_700_000_000_500,
        coin="SOL",
        book=BookTopEvent(ts_ms=1_700_000_000_500, best_bid=90.0, best_ask=90.05, bid_size=5.0, ask_size=4.0),
    ))

    events = list(load_market_events(str(path)))
    assert len(events) == 2
    assert events[0].coin == "ETH"
    assert events[0].kind == "trade"
    assert events[1].coin == "SOL"
    assert events[1].kind == "book"
```

- [ ] **Step 3.2: Run test, confirm FAIL** (coin currently stripped)
- [ ] **Step 3.3: Patch writer + loader**

In `market_capture.py`:
```python
def write(self, event: MarketEvent) -> None:
    row: dict[str, object] = {
        "kind": event.kind,
        "ts_ms": event.ts_ms,
    }
    coin = (event.coin or "").strip().upper()
    if coin:
        row["coin"] = coin
    # ... trade/book blocks unchanged ...
```

In `replay/loader.py`, pass `coin=str(row.get("coin", "")).strip().upper()` into every `MarketEvent(...)` call.

- [ ] **Step 3.4: Run test, confirm PASS**
- [ ] **Step 3.5: Full suite green**

## Task 4 — Config: unstick the gates + fix the exits

**Files:**
- Modify: `.env`

No code change or tests — this is tuning of already-tested knobs. Rationale comes from the performance-forensics and risk-gates audit agents.

### 4a. Edge-pause widening (unstick the bot)

| Key | Before | After | Why |
|---|---|---|---|
| `BOT_SESSION_EDGE_PAUSE_MIN_TRADES` | 4 | **15** | 4-trade windows have ~30% false-positive rate on a 40%-WR strategy |
| `BOT_SESSION_EDGE_PAUSE_AVG_R` | -0.20 | **-0.40** | -0.20 trips on one -1R scratch in a 5-trade window |
| `BOT_LEVEL_EDGE_PAUSE_MIN_TRADES` | 3 | **10** | 3-trade windows are statistical noise |
| `BOT_LEVEL_EDGE_PAUSE_AVG_R` | -0.20 | **-0.40** | Same rationale |
| `BOT_SIDE_EDGE_PAUSE_MIN_TRADES` | 4 | **8** | Has time-based cooldown already; still tight |
| `BOT_LOSS_COOLDOWN_SEC` | 300 | **0** | Redundant with hard_loss_cooldown |
| `BOT_LEVEL_HARD_LOSS_COOLDOWN_SEC` | 21600 | **3600** | 6h on a single -0.9R is a day-killer |

### 4b. Exit loosening (stop leaking MFE)

| Key | Before | After | Why |
|---|---|---|---|
| `BOT_TIME_STOP_SEC` (second / effective def) | 240 | **600** | 240s cuts 35/59 trades that went favorable |
| `BOT_EARLY_EXIT_SEC` | 120 | **240** | Same reasoning, smaller adjustment |
| `BOT_EARLY_EXIT_R_THRESHOLD` | -0.3 | **-0.5** | Allow more latitude before cutting losers |
| `BOT_TRAIL_FROM_ENTRY_FACTOR` | 0.25 | **0.10** | 0.25 scratches at +$0.06; loosen to let trades breathe |
| `BOT_MIN_TP1_BPS` | 25 | **40** | 25 bps TP is whipsawed routinely |
| `BOT_MIN_TP2_BPS` | 50 | **90** | Same |

### 4c. Sizing

| Key | Before | After | Why |
|---|---|---|---|
| `BOT_RISK_PER_TRADE_PCT` | 0.30 | **0.50** | $3/trade too small for any measurable P&L; still 10% of default 0.75 |

- [ ] **Step 4.1: Apply all of the above to `.env`**
- [ ] **Step 4.2: Full test suite green**

## Task 5 — Verification

- [ ] **Step 5.1: `PYTHONPATH=src pytest` — all passing, including new tests**
- [ ] **Step 5.2: `docker compose down && docker compose up -d` — container restarts cleanly**
- [ ] **Step 5.3: `docker logs hliq-paper-bot --tail 30` — verify no errors, detector diagnostics showing traffic**
- [ ] **Step 5.4: Wait 5–10 minutes, check `tail runtime/bot.log` — confirm `signals=N blocked=<N` (blocked strictly less than generated, proving the gate has released)**

## Task 6 — Commits

Commit in logical chunks, each a passing unit, with HEREDOC messages:
1. `fix(risk): clear session/level deques on UTC day roll` (governor + test)
2. `fix(detector): bound wick_ratio to prevent doji artefact` (detector + tests)
3. `fix(capture): persist coin field through writer and loader` (capture + loader + test)
4. `chore(config): widen edge-pause thresholds and loosen exits` (.env)

## What this plan deliberately does NOT change

- Signal scoring formula — requires replay validation on the 42-day corpus first; not this session.
- Strategy inversion (breakout vs retest) — same.
- Live Hyperliquid execution adapter — needs HL SDK, agent wallet, testnet validation; multi-day project.
- ML gate — stays disabled (`BOT_ML_ENABLED=false`); overfit model needs retraining on ≥500 samples.
- WS subscription extensions (L2 depth, funding, liquidations) — separate session.
- Predictor training on 3 GB corpus — separate multi-day project.
