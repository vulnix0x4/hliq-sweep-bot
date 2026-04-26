# Better Levels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new level sources (PDH/PDL, session open, round numbers, VWAP, prior session high/low) to widen the signal funnel.

**Architecture:** New `SessionTracker` and `VWAPTracker` stateful classes feed into an expanded `derive_levels()`. Each level source is independently toggleable via config. The `SweepDetector` and `Bot` are updated to wire the new trackers through.

**Tech Stack:** Python 3.11+, pytest, no new dependencies.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/hliq_bot/config.py` | Modify | Add `LevelConfig` dataclass + env loading |
| `src/hliq_bot/signal/session_tracker.py` | Create | Track session open, prior session H/L, PDH/PDL |
| `src/hliq_bot/signal/vwap_tracker.py` | Create | Rolling VWAP with daily reset |
| `src/hliq_bot/signal/levels.py` | Modify | Add 5 new level derivation functions |
| `src/hliq_bot/signal/sweep_detector.py` | Modify | Accept new tracker args, apply level-type weight |
| `src/hliq_bot/bot.py` | Modify | Create trackers, feed bars through them |
| `tests/test_session_tracker.py` | Create | Session rollover, PDH/PDL, prior session H/L |
| `tests/test_vwap_tracker.py` | Create | VWAP accumulation, daily reset |
| `tests/test_levels.py` | Create | Each new level source |
| `tests/test_sweep_detector.py` | Modify | Signals from new level types |
| `tests/test_bot_runtime.py` | Modify | Tracker wiring |

---

### Task 1: Add LevelConfig to config.py

**Files:**
- Modify: `src/hliq_bot/config.py`

- [ ] **Step 1: Add `LevelConfig` dataclass after `StrategyConfig`**

Add this right after the `StrategyConfig` class definition (after line 90):

```python
@dataclass(slots=True)
class LevelConfig:
    pdh_pdl: bool = True
    session_open: bool = True
    round_numbers: bool = True
    vwap: bool = True
    prior_session: bool = True
    round_number_range_pct: float = 1.5
```

- [ ] **Step 2: Add `levels` field to `AppConfig`**

Add `levels: LevelConfig` field to the `AppConfig` dataclass.

- [ ] **Step 3: Load `LevelConfig` from env in `load_config()`**

Add this block before the `return AppConfig(...)` statement:

```python
levels = LevelConfig(
    pdh_pdl=_env_bool("BOT_LEVELS_PDH_PDL", True),
    session_open=_env_bool("BOT_LEVELS_SESSION_OPEN", True),
    round_numbers=_env_bool("BOT_LEVELS_ROUND_NUMBERS", True),
    vwap=_env_bool("BOT_LEVELS_VWAP", True),
    prior_session=_env_bool("BOT_LEVELS_PRIOR_SESSION", True),
    round_number_range_pct=_env_float("BOT_ROUND_NUMBER_RANGE_PCT", 1.5),
)
```

Pass `levels=levels` into the `AppConfig(...)` constructor.

- [ ] **Step 4: Update test helper in `tests/test_bot_runtime.py`**

The `_app_config` helper constructs `AppConfig` directly. Add `levels=LevelConfig()` to it so existing tests keep passing.

- [ ] **Step 5: Run tests**

Run: `pytest -q`
Expected: All 29 tests PASS (no behavior changed, just added config).

- [ ] **Step 6: Commit**

```
git add src/hliq_bot/config.py tests/test_bot_runtime.py
git commit -m "feat(config): add LevelConfig for new level sources"
```

---

### Task 2: Create SessionTracker

**Files:**
- Create: `src/hliq_bot/signal/session_tracker.py`
- Create: `tests/test_session_tracker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_session_tracker.py`:

```python
from __future__ import annotations

from hliq_bot.models import Bar
from hliq_bot.signal.session_tracker import SessionTracker


def _bar(ts_ms: int, o: float, h: float, l: float, c: float) -> Bar:
    return Bar(
        start_ms=ts_ms,
        end_ms=ts_ms + 60_000,
        open=o, high=h, low=l, close=c,
        volume=1.0, trade_count=1, vwap=(o + c) / 2.0, avg_spread_bps=0.1,
    )


def test_session_from_hour():
    st = SessionTracker()
    assert st._session_from_hour(0) == "asia"
    assert st._session_from_hour(6) == "asia"
    assert st._session_from_hour(7) == "eu"
    assert st._session_from_hour(12) == "eu"
    assert st._session_from_hour(13) == "us"
    assert st._session_from_hour(21) == "us"
    assert st._session_from_hour(22) == "late"
    assert st._session_from_hour(23) == "late"


def test_session_open_tracked_on_first_bar():
    st = SessionTracker()
    # 2026-04-06 00:01 UTC -> asia session
    ts = 1775433660000
    st.on_bar(_bar(ts, 70000.0, 70100.0, 69900.0, 70050.0))
    assert st.current_session == "asia"
    assert st.current_session_open == 70000.0


def test_session_rollover_updates_prior():
    st = SessionTracker()
    # Feed Asia bars (hour 0-6)
    ts = 1775433600000  # 2026-04-06 00:00:00 UTC
    for i in range(6):
        bar_ts = ts + i * 60_000
        st.on_bar(_bar(bar_ts, 70000.0 + i, 70100.0 + i, 69900.0, 70050.0 + i))

    # Now feed an EU bar (hour 7 = +7h = +25200s)
    eu_ts = ts + 7 * 3600 * 1000
    st.on_bar(_bar(eu_ts, 70200.0, 70300.0, 70100.0, 70250.0))

    assert st.current_session == "eu"
    assert st.current_session_open == 70200.0
    assert st.prior_session == "asia"
    assert st.prior_session_open == 70000.0
    assert st.prior_session_high == 70105.0  # max high from asia bars
    assert st.prior_session_low == 69900.0   # min low from asia bars


def test_pdh_pdl_on_day_rollover():
    st = SessionTracker()
    # Day 1 bars
    day1_start = 1775347200000  # 2026-04-05 00:00:00 UTC
    for i in range(10):
        bar_ts = day1_start + i * 60_000
        st.on_bar(_bar(bar_ts, 69000.0, 69500.0 + i * 10, 68500.0 - i * 5, 69100.0))

    # Day 2 bar
    day2_start = day1_start + 86400 * 1000  # 2026-04-06 00:00:00 UTC
    st.on_bar(_bar(day2_start, 69200.0, 69300.0, 69100.0, 69250.0))

    assert st.prior_day_high == 69590.0  # 69500 + 9*10
    assert st.prior_day_low == 68455.0   # 68500 - 9*5


def test_levels_returns_empty_before_any_bars():
    st = SessionTracker()
    assert st.get_levels() == ([], [])


def test_levels_include_pdh_pdl_after_rollover():
    st = SessionTracker()
    day1_start = 1775347200000
    for i in range(10):
        st.on_bar(_bar(day1_start + i * 60_000, 69000.0, 69500.0, 68500.0, 69100.0))
    day2_start = day1_start + 86400 * 1000
    st.on_bar(_bar(day2_start, 69200.0, 69300.0, 69100.0, 69250.0))

    short_levels, long_levels = st.get_levels()
    short_labels = [label for label, _ in short_levels]
    long_labels = [label for label, _ in long_levels]
    assert "pdh" in short_labels
    assert "pdl" in long_labels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session_tracker.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `SessionTracker`**

Create `src/hliq_bot/signal/session_tracker.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from hliq_bot.models import Bar


class SessionTracker:
    def __init__(self) -> None:
        self.current_session: str = ""
        self.current_session_open: float = 0.0
        self._current_session_high: float = 0.0
        self._current_session_low: float = float("inf")

        self.prior_session: str = ""
        self.prior_session_open: float = 0.0
        self.prior_session_high: float = 0.0
        self.prior_session_low: float = 0.0

        self._current_day: str = ""
        self._current_day_high: float = 0.0
        self._current_day_low: float = float("inf")

        self.prior_day_high: float = 0.0
        self.prior_day_low: float = 0.0

        self._has_prior_session: bool = False
        self._has_prior_day: bool = False

    def on_bar(self, bar: Bar) -> None:
        dt = datetime.fromtimestamp(bar.start_ms / 1000.0, tz=timezone.utc)
        session = self._session_from_hour(dt.hour)
        day_key = dt.strftime("%Y-%m-%d")

        # Day rollover
        if self._current_day and day_key != self._current_day:
            self.prior_day_high = self._current_day_high
            self.prior_day_low = self._current_day_low
            self._has_prior_day = True
            self._current_day_high = 0.0
            self._current_day_low = float("inf")

        self._current_day = day_key
        self._current_day_high = max(self._current_day_high, bar.high)
        self._current_day_low = min(self._current_day_low, bar.low)

        # Session rollover
        if self.current_session and session != self.current_session:
            self.prior_session = self.current_session
            self.prior_session_open = self.current_session_open
            self.prior_session_high = self._current_session_high
            self.prior_session_low = self._current_session_low
            self._has_prior_session = True
            self._current_session_high = 0.0
            self._current_session_low = float("inf")
            self.current_session_open = bar.open

        if not self.current_session:
            self.current_session_open = bar.open

        self.current_session = session
        self._current_session_high = max(self._current_session_high, bar.high)
        self._current_session_low = min(self._current_session_low, bar.low)

    def get_levels(self) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        short_levels: list[tuple[str, float]] = []
        long_levels: list[tuple[str, float]] = []

        if self._has_prior_day and self.prior_day_high > 0:
            short_levels.append(("pdh", self.prior_day_high))
        if self._has_prior_day and self.prior_day_low > 0:
            long_levels.append(("pdl", self.prior_day_low))

        if self.current_session_open > 0:
            short_levels.append(("session_open_current", self.current_session_open))
            long_levels.append(("session_open_current", self.current_session_open))

        if self._has_prior_session and self.prior_session_open > 0:
            short_levels.append(("session_open_prior", self.prior_session_open))
            long_levels.append(("session_open_prior", self.prior_session_open))

        if self._has_prior_session and self.prior_session_high > 0:
            short_levels.append(("prior_session_high", self.prior_session_high))
        if self._has_prior_session and self.prior_session_low > 0:
            long_levels.append(("prior_session_low", self.prior_session_low))

        return short_levels, long_levels

    @staticmethod
    def _session_from_hour(hour_utc: int) -> str:
        if 0 <= hour_utc < 7:
            return "asia"
        if 7 <= hour_utc < 13:
            return "eu"
        if 13 <= hour_utc < 22:
            return "us"
        return "late"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_session_tracker.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: All tests PASS (no regressions).

- [ ] **Step 6: Commit**

```
git add src/hliq_bot/signal/session_tracker.py tests/test_session_tracker.py
git commit -m "feat(levels): add SessionTracker for PDH/PDL, session open, prior session H/L"
```

---

### Task 3: Create VWAPTracker

**Files:**
- Create: `src/hliq_bot/signal/vwap_tracker.py`
- Create: `tests/test_vwap_tracker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_vwap_tracker.py`:

```python
from __future__ import annotations

from hliq_bot.models import Bar
from hliq_bot.signal.vwap_tracker import VWAPTracker


def _bar(ts_ms: int, o: float, h: float, l: float, c: float, v: float) -> Bar:
    return Bar(
        start_ms=ts_ms,
        end_ms=ts_ms + 60_000,
        open=o, high=h, low=l, close=c,
        volume=v, trade_count=1, vwap=(o + c) / 2.0, avg_spread_bps=0.1,
    )


def test_vwap_single_bar():
    vt = VWAPTracker()
    vt.on_bar(_bar(1000, 100.0, 101.0, 99.0, 100.5, 10.0))
    # VWAP uses bar's own vwap (which is (open+close)/2 = 100.25) * volume
    # Actually we should use the bar's vwap field for accuracy
    assert vt.vwap > 0


def test_vwap_accumulates_across_bars():
    vt = VWAPTracker()
    # bar1: vwap=100.0, vol=10 -> pv=1000
    vt.on_bar(_bar(1000, 100.0, 101.0, 99.0, 100.0, 10.0))
    # bar2: vwap=110.0, vol=20 -> pv=2200
    vt.on_bar(_bar(61000, 110.0, 111.0, 109.0, 110.0, 20.0))
    # cumulative: pv=3200, vol=30, vwap=106.666...
    expected = (100.0 * 10.0 + 110.0 * 20.0) / 30.0
    assert abs(vt.vwap - expected) < 0.01


def test_vwap_resets_on_new_day():
    vt = VWAPTracker()
    day1_ts = 1775347200000  # 2026-04-05 00:00:00 UTC
    vt.on_bar(_bar(day1_ts, 69000.0, 69100.0, 68900.0, 69050.0, 100.0))

    day2_ts = day1_ts + 86400 * 1000
    vt.on_bar(_bar(day2_ts, 70000.0, 70100.0, 69900.0, 70050.0, 50.0))

    # After reset, VWAP should be from day2 bar only
    expected = (70000.0 + 70050.0) / 2.0  # bar vwap = 70025
    assert abs(vt.vwap - expected) < 1.0


def test_vwap_zero_before_any_bars():
    vt = VWAPTracker()
    assert vt.vwap == 0.0


def test_get_levels_returns_vwap_as_dual_sided():
    vt = VWAPTracker()
    vt.on_bar(_bar(1000, 100.0, 101.0, 99.0, 100.0, 10.0))
    short_levels, long_levels = vt.get_levels()
    assert any(label == "vwap_daily" for label, _ in short_levels)
    assert any(label == "vwap_daily" for label, _ in long_levels)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vwap_tracker.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `VWAPTracker`**

Create `src/hliq_bot/signal/vwap_tracker.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from hliq_bot.models import Bar


class VWAPTracker:
    def __init__(self) -> None:
        self._cum_pv: float = 0.0
        self._cum_vol: float = 0.0
        self._current_day: str = ""

    @property
    def vwap(self) -> float:
        if self._cum_vol <= 0:
            return 0.0
        return self._cum_pv / self._cum_vol

    def on_bar(self, bar: Bar) -> None:
        day_key = datetime.fromtimestamp(
            bar.start_ms / 1000.0, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        if self._current_day and day_key != self._current_day:
            self._cum_pv = 0.0
            self._cum_vol = 0.0

        self._current_day = day_key
        self._cum_pv += bar.vwap * bar.volume
        self._cum_vol += bar.volume

    def get_levels(self) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        v = self.vwap
        if v <= 0:
            return [], []
        return [("vwap_daily", v)], [("vwap_daily", v)]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vwap_tracker.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```
git add src/hliq_bot/signal/vwap_tracker.py tests/test_vwap_tracker.py
git commit -m "feat(levels): add VWAPTracker with daily reset"
```

---

### Task 4: Expand levels.py with new level sources

**Files:**
- Modify: `src/hliq_bot/signal/levels.py`
- Create: `tests/test_levels.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_levels.py`:

```python
from __future__ import annotations

from hliq_bot.config import LevelConfig
from hliq_bot.models import Bar
from hliq_bot.signal.levels import derive_levels, round_number_levels
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.vwap_tracker import VWAPTracker


def _bar(ts_ms: int, o: float, h: float, l: float, c: float, v: float = 1.0) -> Bar:
    return Bar(
        start_ms=ts_ms,
        end_ms=ts_ms + 60_000,
        open=o, high=h, low=l, close=c,
        volume=v, trade_count=1, vwap=(o + c) / 2.0, avg_spread_bps=0.1,
    )


def test_round_number_levels_btc():
    short_levels, long_levels = round_number_levels(
        current_price=68500.0,
        coin="BTC",
        range_pct=1.5,
    )
    short_prices = [px for _, px in short_levels]
    long_prices = [px for _, px in long_levels]
    assert 69000.0 in short_prices  # above current price
    assert 68000.0 in long_prices   # below current price
    # All labels start with "round_"
    assert all(label.startswith("round_") for label, _ in short_levels)
    assert all(label.startswith("round_") for label, _ in long_levels)


def test_round_number_levels_eth():
    short_levels, long_levels = round_number_levels(
        current_price=3450.0,
        coin="ETH",
        range_pct=1.5,
    )
    short_prices = [px for _, px in short_levels]
    long_prices = [px for _, px in long_levels]
    assert 3500.0 in short_prices
    assert 3400.0 in long_prices


def test_round_number_levels_sol():
    short_levels, long_levels = round_number_levels(
        current_price=135.0,
        coin="SOL",
        range_pct=1.5,
    )
    short_prices = [px for _, px in short_levels]
    long_prices = [px for _, px in long_levels]
    assert 140.0 in short_prices
    assert 130.0 in long_prices


def test_round_number_levels_unknown_coin():
    short_levels, long_levels = round_number_levels(
        current_price=5.0,
        coin="DOGE",
        range_pct=2.0,
    )
    # Should still produce some levels
    assert len(short_levels) + len(long_levels) > 0


def test_derive_levels_includes_new_sources():
    history = [_bar(i * 60_000, 70000.0, 70100.0, 69900.0, 70050.0) for i in range(20)]
    st = SessionTracker()
    vt = VWAPTracker()
    for bar in history:
        st.on_bar(bar)
        vt.on_bar(bar)

    cfg = LevelConfig(
        pdh_pdl=True,
        session_open=True,
        round_numbers=True,
        vwap=True,
        prior_session=True,
    )
    result = derive_levels(
        history=history,
        timeframe_sec=60,
        equal_band_bps=6.0,
        level_config=cfg,
        session_tracker=st,
        vwap_tracker=vt,
        current_price=70050.0,
        coin="BTC",
    )
    all_labels = [l for l, _ in result.short_levels] + [l for l, _ in result.long_levels]
    # Should have at least one round number and VWAP
    assert any(l.startswith("round_") for l in all_labels)
    assert any(l == "vwap_daily" for l in all_labels)


def test_derive_levels_respects_disabled_flags():
    history = [_bar(i * 60_000, 70000.0, 70100.0, 69900.0, 70050.0) for i in range(20)]
    st = SessionTracker()
    vt = VWAPTracker()
    for bar in history:
        st.on_bar(bar)
        vt.on_bar(bar)

    cfg = LevelConfig(
        pdh_pdl=False,
        session_open=False,
        round_numbers=False,
        vwap=False,
        prior_session=False,
    )
    result = derive_levels(
        history=history,
        timeframe_sec=60,
        equal_band_bps=6.0,
        level_config=cfg,
        session_tracker=st,
        vwap_tracker=vt,
        current_price=70050.0,
        coin="BTC",
    )
    all_labels = [l for l, _ in result.short_levels] + [l for l, _ in result.long_levels]
    assert not any(l.startswith("round_") for l in all_labels)
    assert not any(l == "vwap_daily" for l in all_labels)
    assert not any(l == "pdh" for l in all_labels)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_levels.py -v`
Expected: FAIL — `round_number_levels` not found, `derive_levels` signature mismatch.

- [ ] **Step 3: Implement expanded `levels.py`**

Replace `src/hliq_bot/signal/levels.py` with:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

from hliq_bot.config import LevelConfig
from hliq_bot.models import Bar
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.vwap_tracker import VWAPTracker

_ROUND_INTERVALS: dict[str, float] = {
    "BTC": 1000.0,
    "ETH": 100.0,
    "SOL": 10.0,
    "AVAX": 10.0,
    "DOGE": 0.01,
    "ARB": 0.1,
    "OP": 0.5,
    "MATIC": 0.1,
    "LINK": 1.0,
    "WIF": 0.1,
}


@dataclass(slots=True)
class LevelSet:
    short_levels: list[tuple[str, float]]
    long_levels: list[tuple[str, float]]


def derive_levels(
    history: list[Bar],
    timeframe_sec: int,
    equal_band_bps: float,
    level_config: LevelConfig | None = None,
    session_tracker: SessionTracker | None = None,
    vwap_tracker: VWAPTracker | None = None,
    current_price: float = 0.0,
    coin: str = "BTC",
) -> LevelSet:
    if not history:
        return LevelSet(short_levels=[], long_levels=[])

    bars_15m = max(1, int((15 * 60) / timeframe_sec))
    bars_1h = max(1, int((60 * 60) / timeframe_sec))

    look_15m = history[-bars_15m:]
    look_1h = history[-bars_1h:]

    short_levels: list[tuple[str, float]] = []
    long_levels: list[tuple[str, float]] = []

    # --- Existing: prior 15m/1h high/low ---
    high_15 = max(b.high for b in look_15m)
    low_15 = min(b.low for b in look_15m)
    high_1h = max(b.high for b in look_1h)
    low_1h = min(b.low for b in look_1h)

    short_levels.append(("prior_15m_high", high_15))
    short_levels.append(("prior_1h_high", high_1h))
    long_levels.append(("prior_15m_low", low_15))
    long_levels.append(("prior_1h_low", low_1h))

    # --- Existing: equal levels ---
    eq_highs, eq_lows = _equal_levels(history, equal_band_bps)
    for i, level in enumerate(eq_highs, start=1):
        short_levels.append((f"equal_high_{i}", level))
    for i, level in enumerate(eq_lows, start=1):
        long_levels.append((f"equal_low_{i}", level))

    # --- New level sources ---
    cfg = level_config or LevelConfig()

    if session_tracker is not None and (cfg.pdh_pdl or cfg.session_open or cfg.prior_session):
        s_short, s_long = session_tracker.get_levels()
        for label, px in s_short:
            if label == "pdh" and not cfg.pdh_pdl:
                continue
            if label.startswith("session_open") and not cfg.session_open:
                continue
            if label.startswith("prior_session") and not cfg.prior_session:
                continue
            short_levels.append((label, px))
        for label, px in s_long:
            if label == "pdl" and not cfg.pdh_pdl:
                continue
            if label.startswith("session_open") and not cfg.session_open:
                continue
            if label.startswith("prior_session") and not cfg.prior_session:
                continue
            long_levels.append((label, px))

    if cfg.vwap and vwap_tracker is not None:
        v_short, v_long = vwap_tracker.get_levels()
        short_levels.extend(v_short)
        long_levels.extend(v_long)

    if cfg.round_numbers and current_price > 0:
        r_short, r_long = round_number_levels(
            current_price=current_price,
            coin=coin,
            range_pct=cfg.round_number_range_pct,
        )
        short_levels.extend(r_short)
        long_levels.extend(r_long)

    short_levels = _dedupe_levels(short_levels, equal_band_bps)
    long_levels = _dedupe_levels(long_levels, equal_band_bps)
    return LevelSet(short_levels=short_levels, long_levels=long_levels)


def round_number_levels(
    current_price: float,
    coin: str,
    range_pct: float,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    interval = _ROUND_INTERVALS.get(coin.upper(), 0.0)
    if interval <= 0:
        interval = _auto_interval(current_price)
    if interval <= 0 or current_price <= 0:
        return [], []

    range_abs = current_price * (range_pct / 100.0)
    lo = current_price - range_abs
    hi = current_price + range_abs

    first = math.ceil(lo / interval) * interval
    short_levels: list[tuple[str, float]] = []
    long_levels: list[tuple[str, float]] = []

    level = first
    while level <= hi:
        label = f"round_{level:g}"
        if level > current_price:
            short_levels.append((label, level))
        elif level < current_price:
            long_levels.append((label, level))
        level += interval

    return short_levels, long_levels


def _auto_interval(price: float) -> float:
    if price <= 0:
        return 0.0
    magnitude = 10 ** math.floor(math.log10(price))
    return max(magnitude * 0.01, 0.01)


def _equal_levels(history: list[Bar], band_bps: float) -> tuple[list[float], list[float]]:
    pivot_highs: list[float] = []
    pivot_lows: list[float] = []
    if len(history) < 3:
        return pivot_highs, pivot_lows

    for i in range(1, len(history) - 1):
        left = history[i - 1]
        mid = history[i]
        right = history[i + 1]
        if mid.high > left.high and mid.high > right.high:
            pivot_highs.append(mid.high)
        if mid.low < left.low and mid.low < right.low:
            pivot_lows.append(mid.low)

    return _cluster_levels(pivot_highs, band_bps), _cluster_levels(pivot_lows, band_bps)


def _cluster_levels(pivots: list[float], band_bps: float) -> list[float]:
    if len(pivots) < 2:
        return []
    clusters: list[list[float]] = []
    for px in pivots:
        matched = False
        for c in clusters:
            ref = sum(c) / len(c)
            diff_bps = abs(px - ref) / ref * 10_000.0
            if diff_bps <= band_bps:
                c.append(px)
                matched = True
                break
        if not matched:
            clusters.append([px])

    return [sum(c) / len(c) for c in clusters if len(c) >= 2]


def _dedupe_levels(levels: list[tuple[str, float]], band_bps: float) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for label, px in levels:
        if px <= 0:
            continue
        duplicate = False
        for _, existing in out:
            diff_bps = abs(px - existing) / existing * 10_000.0
            if diff_bps <= band_bps:
                duplicate = True
                break
        if not duplicate:
            out.append((label, px))
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_levels.py tests/test_sweep_detector.py -v`
Expected: All PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```
git add src/hliq_bot/signal/levels.py tests/test_levels.py
git commit -m "feat(levels): add PDH/PDL, session open, round numbers, VWAP, prior session H/L"
```

---

### Task 5: Update SweepDetector to use new level sources and apply level-type weight

**Files:**
- Modify: `src/hliq_bot/signal/sweep_detector.py`
- Modify: `tests/test_sweep_detector.py`

- [ ] **Step 1: Write failing test for level-type weight bonus**

Add to `tests/test_sweep_detector.py`:

```python
def test_signal_score_includes_level_type_weight() -> None:
    cfg = StrategyConfig(
        timeframe_sec=60,
        min_sweep_bps=4.0,
        max_sweep_bps=40.0,
        min_reclaim_bps=2.0,
        volume_lookback_bars=5,
        volume_spike_mult=1.1,
        wick_body_ratio_min=1.2,
    )
    det = SweepDetector(cfg)

    bar = _bar(0, o=100.95, h=101.25, l=100.70, c=100.85, v=240.0, spread=1.0)

    signal_base = det._short_signal(
        bar,
        short_levels=[("prior_15m_high", 101.0)],
        avg_vol=100.0,
    )
    signal_pdh = det._short_signal(
        bar,
        short_levels=[("pdh", 101.0)],
        avg_vol=100.0,
    )

    assert signal_base is not None
    assert signal_pdh is not None
    # PDH should get a higher score due to level-type weight bonus
    assert signal_pdh.signal_score > signal_base.signal_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep_detector.py::test_signal_score_includes_level_type_weight -v`
Expected: FAIL — scores are equal (no bonus applied yet).

- [ ] **Step 3: Update `SweepDetector`**

Modify `src/hliq_bot/signal/sweep_detector.py`:

1. Add imports at top:

```python
from hliq_bot.config import LevelConfig, StrategyConfig
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.vwap_tracker import VWAPTracker
```

2. Add the `__init__` parameters and store trackers:

```python
def __init__(
    self,
    config: StrategyConfig,
    level_config: LevelConfig | None = None,
    session_tracker: SessionTracker | None = None,
    vwap_tracker: VWAPTracker | None = None,
    coin: str = "BTC",
) -> None:
    self.cfg = config
    self._level_config = level_config
    self._session_tracker = session_tracker
    self._vwap_tracker = vwap_tracker
    self._coin = coin
    # ... rest of existing __init__ unchanged
```

3. Update `on_bar` to pass new args to `derive_levels`:

```python
def on_bar(self, bar: Bar) -> SweepSignal | None:
    history = list(self._history)
    signal = None

    if len(history) < self._min_history_bars:
        self._diag_counts["skip_history"] += 1
    else:
        levels = derive_levels(
            history,
            self.cfg.timeframe_sec,
            self.cfg.equal_level_band_bps,
            level_config=self._level_config,
            session_tracker=self._session_tracker,
            vwap_tracker=self._vwap_tracker,
            current_price=bar.close,
            coin=self._coin,
        )
        avg_vol = self._avg_volume(history)
        if avg_vol <= 0:
            self._diag_counts["skip_avg_volume"] += 1
        else:
            signal = self._find_signal(bar, levels.short_levels, levels.long_levels, avg_vol)
            if signal is None and self._is_trending(history):
                self._diag_counts["trend_context"] += 1

    self._history.append(bar)
    return signal
```

4. Add level-type weight bonus mapping and helper:

```python
_LEVEL_TYPE_WEIGHT: dict[str, float] = {
    "pdh": 0.05,
    "pdl": 0.05,
    "vwap_daily": 0.04,
    "session_open_current": 0.03,
    "session_open_prior": 0.03,
    "prior_session_high": 0.02,
    "prior_session_low": 0.02,
}
```

5. In both `_short_signal` and `_long_signal`, after computing `score = self._signal_score(...)`, add:

```python
bonus = _level_type_bonus(label)
score = max(0.0, min(1.0, score + bonus))
```

6. Add the helper function at module level:

```python
def _level_type_bonus(label: str) -> float:
    for prefix, bonus in _LEVEL_TYPE_WEIGHT.items():
        if label == prefix or label.startswith(prefix):
            return bonus
    if label.startswith("round_"):
        return 0.01
    return 0.0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sweep_detector.py -v`
Expected: All tests PASS including the new one.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```
git add src/hliq_bot/signal/sweep_detector.py tests/test_sweep_detector.py
git commit -m "feat(detector): wire new level sources + level-type weight bonus"
```

---

### Task 6: Wire trackers into Bot

**Files:**
- Modify: `src/hliq_bot/bot.py`
- Modify: `tests/test_bot_runtime.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_bot_runtime.py`:

```python
def test_bot_creates_session_and_vwap_trackers(tmp_path: Path) -> None:
    cfg = _app_config(tmp_path)
    bot = SweepBot(cfg)
    assert bot._session_tracker is not None
    assert bot._vwap_tracker is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bot_runtime.py::test_bot_creates_session_and_vwap_trackers -v`
Expected: FAIL — `_session_tracker` attribute not found.

- [ ] **Step 3: Update `bot.py`**

1. Add imports at top of `bot.py`:

```python
from hliq_bot.signal.session_tracker import SessionTracker
from hliq_bot.signal.vwap_tracker import VWAPTracker
```

2. In `SweepBot.__init__`, create the trackers before `SweepDetector`:

```python
self._session_tracker = SessionTracker()
self._vwap_tracker = VWAPTracker()
```

3. Update the `SweepDetector` construction to pass trackers:

```python
self.detector = SweepDetector(
    config.strategy,
    level_config=config.levels,
    session_tracker=self._session_tracker,
    vwap_tracker=self._vwap_tracker,
    coin=config.feed.coin,
)
```

4. In the bar processing section of `_handle_event` (around the `for bar in closed_bars:` loop), feed each bar to the trackers **before** calling `self.detector.on_bar(bar)`:

```python
for bar in closed_bars:
    self._bars_closed += 1
    self._recent_bar_ranges.append(bar.range_pct)
    self._recent_closes.append(bar.close)
    self._session_tracker.on_bar(bar)
    self._vwap_tracker.on_bar(bar)
    signal = self.detector.on_bar(bar)
    # ... rest unchanged
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_bot_runtime.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```
git add src/hliq_bot/bot.py tests/test_bot_runtime.py
git commit -m "feat(bot): wire SessionTracker + VWAPTracker into bar processing"
```

---

### Task 7: Add level source env vars to .env

**Files:**
- Modify: `.env`

- [ ] **Step 1: Append new config lines to `.env`**

Add after the existing config block:

```
BOT_LEVELS_PDH_PDL=true
BOT_LEVELS_SESSION_OPEN=true
BOT_LEVELS_ROUND_NUMBERS=true
BOT_LEVELS_VWAP=true
BOT_LEVELS_PRIOR_SESSION=true
BOT_ROUND_NUMBER_RANGE_PCT=1.5
```

- [ ] **Step 2: Run full test suite one final time**

Run: `pytest -q`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```
git add .env
git commit -m "feat(config): add level source env vars to .env"
```
