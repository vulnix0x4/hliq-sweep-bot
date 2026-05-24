# AI Mode (Tier 3) — LLM-driven Trading

When `BOT_AI_ENABLED=true`, the rule-based sweep detector is bypassed and an
LLM is polled per-coin every `BOT_AI_INTERVAL_SEC` (default 300s = 5 min).
The LLM has full position-management authority within safety bounds.

## What still applies (safety)

The AI **cannot bypass**:
- `BOT_ALLOW_LIVE` — paper unless explicitly set
- `RiskGovernor` — daily loss limit, hard-loss cooldowns, edge-pause, size_position
- `HyperliquidOrderManager` — max_notional, native stops with slippage buffer,
  deadman timer, lot rounding, sig-fig price rounding
- `max_concurrent_same_side` cap (prevents 4-long-alts stealth leverage)
- Volatility-targeted sizing (when enabled)
- Operator override flag (`runtime/ai_override.flag`) — `pause`/`no_new`/`close_all`

## What the AI sees per decision (context)

Built by `src/hliq_bot/ai/context.py`:

**Per-coin core**
- Recent N OHLCV bars (default 30 1-min bars, ~30 min of history)
- Current price, spread, recent trade flow bias, realized vol, range_5m / range_30m
- Session VWAP + distance, session tag (asia/eu/us/late)
- Current open position (entry/stop/TP/qty/unrealized R, time held)

**HL enrichment (from `meta_and_asset_ctxs` + `l2_snapshot`)**
- Funding rate, open interest, mark price
- 24h change %, 24h volume USD
- L2 top-5 bids + asks, depth imbalance, true spread bps

**Multi-coin / portfolio**
- `other_coins`: top-8 by volume — each with day_change_pct, funding, flow_bias_5m
- `portfolio`: all open positions/pending entries + concurrent_long/short counts

**Account + self-reflection**
- Account equity, daily PnL, daily R
- `recent_outcomes`: last 10 AI trades — pulled from persistent memory
  (`runtime/ai_memory.jsonl`), survives restarts

## Actions the AI can take

Beyond `hold` / `open_long` / `open_short` / `close`:

- `move_stop_to_breakeven` — set stop to entry. Locks in protection after a
  favorable move without committing to a specific tighter level.
- `modify_stop` — set `new_stop_price`. Strict-improvement validated (never
  widen). For trailing tighter or moving past BE.
- `scale_out` — partial close. `scale_fraction` in (0,1). For taking some
  off at a target while keeping a runner.
- `add_to_position` — `add_qty_fraction`. Only allowed when unrealized
  R ≥ +0.5. **Currently logs intent only**; executor add-on API is the next
  TODO.

## Resilience

`ResilientLLM` wrapper handles:
- Retries with exponential backoff (`BOT_AI_RETRY_ATTEMPTS`, default 3)
  on `AIRetryableError` (timeouts, 5xx, 429)
- Fallback model chain (`BOT_AI_FALLBACK_MODELS` env, comma-separated)
- Circuit breaker (`BOT_AI_CIRCUIT_THRESHOLD` consecutive failures →
  `BOT_AI_CIRCUIT_COOLDOWN_SEC` open window)

## Cost / cadence math

Default cadence (5 min, 4 coins): ~48 calls/hour = ~1150/day.

Gemini 3.5 Flash (est): ~$0.10/M input + $0.40/M output. Typical call
(~3k input + 200 output) = ~$0.0004. 1150 calls/day = **~$0.46/day**.

Soft budget warning at `BOT_AI_DAILY_BUDGET_USD` (default $5). Over-budget
calls are skipped (`skip_reason=daily_budget_exhausted`).

## Operator commands

```
scripts/ai_status.py             # snapshot: positions, memory stats, recent decisions
scripts/ai_pause.py pause        # block new opens; manage existing normally
scripts/ai_pause.py close-all    # force-close everything on next AI tick
scripts/ai_pause.py resume       # clear override; resume normal trading
scripts/ai_flatten.py            # alias for ai_pause.py close-all
scripts/ai_decisions_report.py   # cross-reference decisions with outcomes
scripts/ai_decisions_report.py --by reasoning_word --min-trades 3
scripts/ai_backtest.py --dry-run --days 1 --coins HYPE  # cost estimate
scripts/ai_backtest.py --days 1 --coins HYPE --interval-min 15  # real run
```

## Enabling

1. Get an OpenRouter API key: <https://openrouter.ai>
2. Add to your env (NOT committed):
   ```
   export OPENROUTER_API_KEY=sk-or-v1-...
   ```
   Or in a non-committed `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   BOT_AI_ENABLED=true
   ```
3. Rebuild + restart:
   ```
   docker compose build bot && docker compose up -d --force-recreate bot
   docker logs -f hliq-paper-bot
   ```
4. First-run sanity checks in the logs:
   - `AI strategy ENABLED: model=... memory=... (loaded=0 ...)` — config + memory ok
   - `AI hold on COIN: <reasoning>` — first decision arrived

## Iterating on the prompt

The prompt lives at `src/hliq_bot/ai/prompts.py:SYSTEM_PROMPT`. Procedure:

1. Edit the prompt.
2. Bump `BOT_AI_PROMPT_VERSION` in `.env` (e.g. `v1` → `v2_tighter_stops`).
3. Rebuild + restart.
4. After a day:
   ```
   scripts/ai_decisions_report.py --by prompt_version
   ```
5. The report groups outcomes by version so you can A/B real prompt changes.

## Backtest before live

```
# Estimate cost first
python scripts/ai_backtest.py --dry-run --days 3 --coins HYPE,NEAR

# If cost is acceptable, run for real (~$0.50 for 3 days × 4 coins)
python scripts/ai_backtest.py --days 3 --coins HYPE,NEAR --interval-min 15 --json-out runtime/ai_backtest.json
```

The backtest re-runs the exact AI strategy pipeline against historical 1m
candles. Simulated execution is conservative OHLC (stop wins ties). Use
multiple coin / interval / days settings to map the cost/edge surface.

## Going live (mainnet, real money)

Same gates as the rule-based strategy. Before flipping `BOT_ALLOW_LIVE=true`:
- ≥ 3 days of profitable paper with current prompt version
- AI decisions report shows reasonable per-action breakdown (mostly `hold`,
  not 90% `open_long`)
- Memory stats show win_rate ≥ 35% AND avg_r > 0
- Backtest agrees with paper (no divergence > 2x)
- `HL_MAX_NOTIONAL_PER_TRADE` set to a small value for first live run
- You've run `scripts/ai_flatten.py` once successfully (know how to stop fast)

## Files

- `src/hliq_bot/ai/client.py` — OpenAI-compatible HTTP client + ResilientLLM + CostBudget
- `src/hliq_bot/ai/context.py` — market-state aggregation
- `src/hliq_bot/ai/market_data.py` — HL meta/L2 cache
- `src/hliq_bot/ai/memory.py` — persistent decision+outcome journal
- `src/hliq_bot/ai/prompts.py` — system prompt + decision schema
- `src/hliq_bot/ai/strategy.py` — main AIStrategy class
- `scripts/ai_status.py` — operator status snapshot
- `scripts/ai_pause.py` — pause / close-all / resume
- `scripts/ai_flatten.py` — emergency flatten alias
- `scripts/ai_backtest.py` — offline AI replay
- `scripts/ai_decisions_report.py` — outcomes report
- `tests/test_ai_strategy.py` — unit tests (memory, resilience, validation)
