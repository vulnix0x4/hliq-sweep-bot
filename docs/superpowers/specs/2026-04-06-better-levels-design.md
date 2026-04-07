# Better Levels — Design Spec

## Problem

The bot currently derives only 4 level types: prior_15m_high/low, prior_1h_high/low, and clustered equal highs/lows. This limits the signal surface area. The journal shows `prior_15m_low` is the best-performing level type (+$10.29 across 13 trades), but there are high-probability institutional reference levels being missed entirely.

## Goal

Add 5 new level sources to widen the signal funnel without degrading per-trade quality. Each level type is independently toggleable.

## New Level Types

### 1. Prior Day High/Low (PDH/PDL)

The single most-traded intraday level. Institutional algos and retail traders both watch PDH/PDL.

- **Definition:** The high and low of the prior UTC day (00:00–00:00 UTC).
- **Implementation:** Track daily OHLC from bar history. When a new UTC day starts, the previous day's high/low become PDH/PDL. Need at least 1440 bars (24h of 1-min bars) of history to derive. If history is shorter, derive from whatever is available from the prior day.
- **Labels:** `pdh` (short level), `pdl` (long level).
- **Config:** `BOT_LEVELS_PDH_PDL=true` (default true).

### 2. Session Open

Session open prices are reference points. A sweep below session open that reclaims is a classic long setup.

- **Definition:** The opening price (first trade) of each trading session.
  - Asia: 00:00 UTC
  - EU: 07:00 UTC
  - US: 13:00 UTC
  - Late: 22:00 UTC
- **Implementation:** Track the close of the last bar before each session boundary, or the open of the first bar after. Store in a `SessionTracker` that the `SweepDetector` (or level derivation) queries. Only the current session's open and the prior session's open are active levels.
- **Labels:** `session_open_current`, `session_open_prior`.
- **Both sides:** Session open is both a short level (sweep above + reject) and a long level (sweep below + reclaim).
- **Config:** `BOT_LEVELS_SESSION_OPEN=true` (default true).

### 3. Round Numbers

Psychological S/R. Large resting orders cluster at round numbers.

- **Definition:** Price levels at fixed intervals depending on the asset:
  - BTC: every $1,000 (e.g., $68,000, $69,000)
  - ETH: every $100 (e.g., $3,400, $3,500)
  - SOL: every $10 (e.g., $130, $140)
  - Default: every 1% of current price (for unknown coins)
- **Implementation:** Given current price, enumerate round levels within `round_number_range_pct` (default 1.5%) above and below. Levels above current price → short levels. Levels below → long levels.
- **Labels:** `round_<price>` (e.g., `round_69000`).
- **Config:** `BOT_LEVELS_ROUND_NUMBERS=true` (default true), `BOT_ROUND_NUMBER_RANGE_PCT=1.5`.
- **Per-coin intervals** stored in a simple dict in config, extensible via env var `BOT_ROUND_NUMBER_INTERVAL` (default auto-detected from coin).

### 4. VWAP (Volume-Weighted Average Price)

VWAP is the benchmark price. Sweeps through VWAP that reclaim are high-probability mean-reversion entries.

- **Definition:** Cumulative (price * volume) / cumulative volume, reset at the start of each UTC day (or configurable session).
- **Implementation:** Add a `VWAPTracker` that accumulates on each bar close. Exposes `current_vwap` and `prior_session_vwap`. Both are active levels (dual-sided: can be swept from above or below).
- **Labels:** `vwap_daily`, `vwap_prior_session`.
- **Config:** `BOT_LEVELS_VWAP=true` (default true).

### 5. Prior Session High/Low

Similar to PDH/PDL but at session granularity. The high/low of the most recently completed session.

- **Definition:** The high and low prices from the prior trading session (Asia/EU/US/Late).
- **Implementation:** The `SessionTracker` from (2) also tracks per-session high/low. When a session ends, its high/low become prior session high/low levels.
- **Labels:** `prior_session_high`, `prior_session_low`.
- **Config:** `BOT_LEVELS_PRIOR_SESSION=true` (default true).

## Architecture Changes

### levels.py Refactor

Current `derive_levels()` takes bar history and returns a `LevelSet`. It will be extended:

```
def derive_levels(
    history: list[Bar],
    timeframe_sec: int,
    equal_band_bps: float,
    level_config: LevelConfig,       # NEW: toggles for each level type
    session_tracker: SessionTracker,  # NEW: session open/high/low state
    vwap_tracker: VWAPTracker,       # NEW: running VWAP state
    current_price: float,            # NEW: for round number derivation
    coin: str,                       # NEW: for round number intervals
) -> LevelSet:
```

Each level source is a separate private function that returns `list[tuple[str, float]]`. The main function assembles them and deduplicates.

### New: SessionTracker

Lightweight stateful object that tracks session boundaries:

```python
@dataclass
class SessionTracker:
    current_session: str
    current_open: float
    prior_session: str
    prior_open: float
    prior_high: float
    prior_low: float
    prior_day_high: float
    prior_day_low: float
```

Updated on each bar close. Detects session/day rollovers from bar timestamps.

### New: VWAPTracker

```python
@dataclass
class VWAPTracker:
    cum_pv: float = 0.0       # cumulative price*volume
    cum_vol: float = 0.0       # cumulative volume
    current_day: str = ""      # UTC day key for reset detection
    
    @property
    def vwap(self) -> float: ...
    
    def on_bar(self, bar: Bar) -> None: ...
```

### New: LevelConfig dataclass

```python
@dataclass
class LevelConfig:
    pdh_pdl: bool = True
    session_open: bool = True
    round_numbers: bool = True
    vwap: bool = True
    prior_session: bool = True
    round_number_range_pct: float = 1.5
```

Populated from env vars in `load_config()`.

### Signal Score: Level-Type Weighting

Different level types have different base reliability. Add a small bonus/penalty to the signal score based on level type:

| Level Type | Weight Bonus |
|---|---|
| pdh, pdl | +0.05 |
| vwap_daily | +0.04 |
| session_open_current | +0.03 |
| prior_session_high/low | +0.02 |
| round_* | +0.01 |
| prior_15m_high/low (existing) | 0 (baseline) |
| equal_* (existing) | 0 (baseline) |

Applied as an additive term after the existing `_signal_score()` calculation, clamped to [0, 1].

## Config Summary

New env vars (all optional with sensible defaults):
- `BOT_LEVELS_PDH_PDL=true`
- `BOT_LEVELS_SESSION_OPEN=true`
- `BOT_LEVELS_ROUND_NUMBERS=true`
- `BOT_LEVELS_VWAP=true`
- `BOT_LEVELS_PRIOR_SESSION=true`
- `BOT_ROUND_NUMBER_RANGE_PCT=1.5`

## Files Modified

- `src/hliq_bot/config.py` — add `LevelConfig` dataclass + env var loading
- `src/hliq_bot/signal/levels.py` — refactor `derive_levels()`, add 5 new level sources, dedup
- `src/hliq_bot/signal/sweep_detector.py` — pass new args to `derive_levels()`, apply level-type weight bonus
- `src/hliq_bot/bot.py` — create `SessionTracker` + `VWAPTracker`, pass to detector, update on each bar
- `src/hliq_bot/models.py` — no changes expected

## New Files

- `src/hliq_bot/signal/session_tracker.py` — `SessionTracker` class
- `src/hliq_bot/signal/vwap_tracker.py` — `VWAPTracker` class

## Tests

- `tests/test_levels.py` — test each new level source independently
- `tests/test_session_tracker.py` — test session rollover, PDH/PDL derivation
- `tests/test_vwap_tracker.py` — test VWAP accumulation, daily reset
- Update `tests/test_sweep_detector.py` — ensure new levels generate valid signals
- Update `tests/test_bot_runtime.py` — ensure bot wires up trackers correctly

## What's NOT in Scope

- Multi-coin support (Phase 2)
- TP/exit improvements (Phase 1)
- ML gate improvements (Phase 4)
- Risk multiplier fixes (Phase 1)
