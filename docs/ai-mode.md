# AI Mode (Tier 3) — LLM-driven Trading

When `BOT_AI_ENABLED=true`, the rule-based sweep detector is bypassed and an
LLM is polled per-coin every `BOT_AI_INTERVAL_SEC` (default 300s = 5 min).
The LLM has full position-management authority within safety bounds: open,
hold, or close.

## What still applies (safety)

The AI **cannot bypass**:
- `BOT_ALLOW_LIVE` — paper unless explicitly set (and you've validated)
- `RiskGovernor` — daily loss limit, hard-loss cooldowns, edge-pause, size_position
- `HyperliquidOrderManager` — max_notional, native stops with slippage buffer,
  deadman timer, lot rounding, sig-fig price rounding
- Operator pause (`runtime/trade_pause.flag`) — except `edge_check_*` pauses,
  which are sweep-strategy-specific and don't apply to AI

The AI's stop_price suggestion is what the executor actually uses. The AI's
confidence is passed to the risk governor as `risk_multiplier`.

## What the AI sees per decision (context)

Built by `src/hliq_bot/ai/context.py`:
- Recent N OHLCV bars (default 30 1-min bars, ~30 min of history)
- Current price, spread, recent trade flow bias, realized vol, range
- Session VWAP + distance, session tag (asia/eu/us/late)
- Current open position (entry/stop/TP/qty/unrealized R, time held)
- Account equity, daily PnL, daily R
- Last 10 closed AI trades (for self-reflection)

## What the AI returns (schema)

`src/hliq_bot/ai/prompts.py:decision_schema` — strict JSON:
- `action`: `open_long` | `open_short` | `close` | `hold`
- `stop_price`: required for open. Must be on losing side, 8-80 bps from price.
- `tp1_price`, `tp2_price`: optional. Default to 2R/4R if omitted.
- `confidence`: 0-1.
- `reasoning`: short paragraph (logged for review).

The strategy validates everything before submission; invalid stops/TPs are
rejected with reason logged to journal.

## Cost / cadence math

At default cadence (5 min, 4 coins): ~48 calls/hour = ~1150/day.

Gemini 3.5 Flash (estimate): ~$0.10/M input + $0.40/M output.
A typical call: ~3k input + 200 output = ~$0.0004. 1150 calls/day = **~$0.46/day**.

Soft budget warning at `BOT_AI_DAILY_BUDGET_USD` (default $5). Calls above
budget are skipped (logged with `skip_reason=daily_budget_exhausted`); strict
enforcement only — billing happens at OpenRouter regardless.

## Enabling

1. Get an OpenRouter API key: <https://openrouter.ai>
2. Add `OPENROUTER_API_KEY=sk-or-v1-...` to your environment (shell env or
   `.env` — keep out of git either way; `.env` is already in `.gitignore`).
3. Set in `.env`:
   ```
   BOT_AI_ENABLED=true
   BOT_AI_MODEL=google/gemini-3.5-flash
   # keep paper / testnet for the first run:
   BOT_MODE=paper
   ```
4. Rebuild + restart: `docker compose build bot && docker compose up -d --force-recreate bot`
5. Tail logs: `docker logs -f hliq-paper-bot` — look for:
   - `AI strategy ENABLED: provider=openrouter model=...`
   - `AI hold/open_long/close on COIN: <reasoning>`
6. Inspect `runtime/signals.jsonl` for `event_type=ai_decision` rows.

## Stopping / reverting

`BOT_AI_ENABLED=false` and restart — the bot reverts to the rule-based sweep
detector. AI state (call history, budget) is in-process; no persistence to
clean up.

## Switching models

Set `BOT_AI_MODEL` to any OpenRouter-supported model. The pricing table in
`src/hliq_bot/ai/client.py:_PRICING_PER_M_TOKENS` should be updated when
adding new models so the budget tracker isn't blind. Missing models bill at
$0 in the tracker (no warnings), but you still pay OpenRouter.

## Evaluating performance

After a week of paper running, the journal has `event_type=ai_decision` rows
side-by-side with `outcome` rows. Build a report mapping `ai_decision.reasoning`
to subsequent outcome `r_multiple` to see which prompt patterns produce winners.

Existing `scripts/journal_report.py --input runtime/signals.jsonl --last-run`
already produces per-coin / per-session expectancy — those still work in AI
mode (the AI's signal_id naming carries through).

## Going live

Same rules as the rule-based strategy. Requires:
- `BOT_ALLOW_LIVE=true`, `HL_NETWORK=mainnet`, valid `HL_AGENT_PRIVATE_KEY`
- Proven positive expectancy over at least a few days of paper
- Operator review of `runtime/signals.jsonl` showing AI decisions are sane
- Small initial `BOT_ACCOUNT_EQUITY` (e.g. $50) and tight `HL_MAX_NOTIONAL_PER_TRADE`

Do NOT skip the paper period. An LLM "looks smart" plenty of times before it
loses you money.
