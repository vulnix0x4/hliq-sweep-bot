# Strategy Redesign Brief

**Date:** 2026-05-23
**Trigger:** Edge gate fail; user instruction "stop and replan"
**Status:** Bot remains running in paper mode for data collection; not trading.

## TL;DR

The current "sweep-and-reclaim" strategy is structurally broken in the current
Hyperliquid tape. Across 32 real paper trades, **31 hit stop loss and only 1
reached TP2**. The targets ask for 40-90 bps of favorable follow-through after
entry; the market actually delivers a median of 5 bps. Expectancy is negative
at *every* TP level tested, so simple parameter retuning will not fix this.

A genuine redesign is required.

## Evidence

### Candle backtest (3.5 days, all HL has)
| Metric            | Value       | Gate target |
|-------------------|-------------|-------------|
| Total trades      | 93          |             |
| avg_r             | **−0.233**  | ≥ 0.05      |
| profit_factor     | **0.72**    | ≥ 1.20      |
| Best ≥8-trade slice | none      |             |

Top positive slices are 4-6 trade samples (statistical noise):
- SPX asia long equal_low_*: 5 trades, +1.413R, pf=6.04
- NEAR eu long prior_15m_low: 4 trades, +0.740R, pf=3.18

### Real paper journal (32 closed trades, multiple runs)
| By side            | n  | avg_r   | win_rate |
|--------------------|----|---------|----------|
| short              | 16 | +0.015  | 18.8%    |
| long               | 16 | −0.250  | 18.8%    |

| By (session, side) | n  | avg_r   | pnl_R   |
|--------------------|----|---------|---------|
| asia short         | 5  | +0.537  | +2.69   |
| us short           | 1  | +0.488  | +0.49   |
| asia long          | 2  | −0.332  | −0.66   |
| us long            | 7  | −0.179  | −1.25   |
| eu long            | 7  | −0.297  | −2.08   |
| eu short           | 10 | −0.294  | −2.94   |

### MFE analysis (favorable distance trades actually reach)
| Reach ≥ X bps | n / 32 | %   |
|---------------|--------|-----|
| 5 bps         | 16     | 50  |
| 10 bps        | 11     | 34  |
| 15 bps        | 8      | 25  |
| 20 bps        | 7      | 22  |
| **40 bps (current TP1)** | **2**  | **6**  |
| 50 bps        | 1      | 3   |

**Median MFE = 5 bps. Target TP1 = 40 bps. The mismatch is 8x.**

### Expectancy at every TP level tested (assuming stop=17 bps, no scale-out)
| TP target | Win % (from MFE data) | R per win | Expectancy R |
|-----------|----------------------|-----------|--------------|
| 5 bps     | 50%                  | +0.29     | **−0.36**    |
| 10 bps    | 34%                  | +0.59     | **−0.46**    |
| 15 bps    | 25%                  | +0.88     | **−0.53**    |
| 20 bps    | 22%                  | +1.18     | **−0.52**    |
| 40 bps    | 6%                   | +2.35     | **−0.80**    |

Negative at every level. The problem is not "wrong TP" — the strategy is
producing entries with **negative edge per signal**.

## Root-cause hypotheses

1. **Sweep-reclaim doesn't reverse in this tape.** The thesis is that a wick
   past a level followed by a reclaim signals stop hunters were absorbed and
   price reverts. In practice, 81% of these signals continue against us
   (stop_loss is the dominant exit). The wick + reclaim may itself be noise
   in a one-directional market, not a meaningful reversal.

2. **Stops are too tight for the tape's noise floor.** Median stop = 17 bps;
   intra-bar volatility on these coins is comparable. A 17 bp stop in a
   noisy retail-coin tape is fundamentally chase-the-noise.

3. **The signal scoring isn't ranking quality.** Best-quality candidates
   (signal_score 0.9-1.0 bucket) still produce avg_r = −0.118. ML / score
   inputs aren't discriminating winners from losers.

4. **Wrong direction on most setups.** 31/32 stops suggests the market is
   reliably continuing AGAINST our signal. Inverting the signal (fading
   reclaim, riding continuation) may have positive edge — though it inverts
   the thesis entirely.

## Redesign options (ranked by promise)

### A. **Fade the signal** (test first, cheapest)
If the bot says "long the reclaim," go SHORT instead — bet on continuation,
not reversal. Estimated effort: add an `invert_signals` flag and re-run
candle_backtest. Hours, not days.

Inversion math on existing 32 trades: if each loss (−1R) becomes a win
(+stop_bps proxy) and each win becomes a loss, expectancy could flip
positive. Worth running first.

### B. **Replace sweep-reclaim with continuation/trend-follow**
Different thesis: enter on the breakout side, not the reclaim side. Stops
behind the swing, targets ride the move. Effort: new detector module
alongside `SweepDetector`. Days of work. Has the advantage that crypto
intraday has known trend-following edge.

### C. **Asia-shorts-only with current strategy** (defensive minimum)
Only asia + short shows positive expectancy (+0.537R, n=5). Cap trading to
that slice, accept tiny trade count, see if the edge persists with more
samples. Effort: edit .env. Acceptable as a parallel experiment but not a
real strategy on its own.

### D. **Statistical mean-reversion on Z-score**
Different entry primitive: instead of level sweeps, enter when 1-minute
return Z-score exceeds N (over a rolling window) with a tight time-stop.
Effort: new detector + new tests. Days of work. Well-studied edge in liquid
markets, less so in retail alts.

### E. **Stop and accept that this codebase isn't profitable yet**
Treat this as research infrastructure (the bot, journal, replay, backtest
are all valuable). Don't trade money with it until edge can be demonstrated.
Effort: 0. The honest baseline.

## Recommended next step

**Run option A** as a 1-hour experiment. If inversion shows positive
expectancy in the candle backtest AND in a replay over the paper journal,
that's the cheapest path to "the bot makes money." If it doesn't, escalate
to option B (continuation strategy) as a multi-day project.

In the meantime, the bot stays in paper mode (BOT_ALLOW_LIVE=false) and the
edge gate continues to refuse trading — both correct behaviors.

## What I will NOT do without explicit informed consent

- Flip `BOT_ALLOW_LIVE=true` on a strategy with negative backtest expectancy
- Disable the edge gate to "make the bot trade"
- Cherry-pick a 5-trade slice and present it as a profitable strategy
- Increase `BOT_RISK_PER_TRADE_PCT` to compound a losing edge

These are the things that turn "the bot isn't trading" into "the account is
empty."
