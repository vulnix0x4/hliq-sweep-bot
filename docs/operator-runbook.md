# Operator Runbook

## Pre-deployment checks

Before flipping `BOT_MODE=live`:

1. **Rebuild the Docker image and verify the SDK.** Live mode depends on `hyperliquid-python-sdk`; confirm the running image has it before any live/testnet run:
   ```bash
   docker compose build
   docker compose down && docker compose up -d
   docker exec hliq-paper-bot python3 -c "from hyperliquid.exchange import Exchange; print('OK')"
   ```
   The third command must print `OK` before going live.

2. **Approve an agent wallet** via `scripts/approve_agent.py`. NEVER put your main private key in `.env`. The agent key has trade-only permission.

3. **Set both safety flags.** Live activates only when `BOT_MODE=live` AND `BOT_ALLOW_LIVE=true`. Boot will fail if only one is set.

4. **Start with `HL_NETWORK=testnet`** for at least 7 days. Compare testnet outcomes to paper expectations.

5. **Run the go-live gate.** `./scripts/botctl.sh go-live-check` must pass before enabling real-money mode. By default it checks the latest run only, requires a clean post-fix paper sample, requires recent candle evidence to remain positive, and fails if risky live knobs are armed.

## Replay data hygiene

`runtime/market_events.jsonl` (the captured event stream used by `scripts/replay_capture.py`) was originally captured before the `coin` field was added to event records. Legacy rows are coin-less and can bias multi-coin replay results if routed to the first worker.

Mitigation:
- Multi-coin replays skip coin-less rows by default.
- Use `scripts/replay_capture.py --allow-untagged-events` only for old single-coin studies where the missing coin is intentional.
- Newly captured events are properly tagged.

## Policy checks

After changing `.env` allowlists, blocklists, or quality floors, run:

```bash
./scripts/botctl.sh policy-report --input runtime/archive/signals_pre_tune_20260511T224920Z.jsonl --show-selected
```

This applies the current operator policy to historical closed trades and reports the selected slice PnL. Allowlists and blocklists accept exact values or shell-style globs like `equal_high_*`. Use `BOT_ALLOW_COIN_LEVELS` / `BOT_BLOCK_COIN_LEVELS` entries like `HYPE:equal_low_*` when one coin is only acceptable on a narrower level family than the global `BOT_ALLOW_LEVEL_LABELS` set. Use `BOT_ALLOW_COIN_SESSIONS` / `BOT_BLOCK_COIN_SESSIONS` entries like `LINK:asia` when a coin is acceptable only in specific sessions. Use `BOT_ALLOW_COIN_SESSION_LEVELS` / `BOT_BLOCK_COIN_SESSION_LEVELS` entries like `HYPE:asia:equal_high_*` when only the exact coin/session/level combination is acceptable. Treat a positive policy report as a hypothesis only; real-money mode still requires `./scripts/botctl.sh go-live-check` to pass on a fresh paper sample.

For current-market context, run:

```bash
./scripts/botctl.sh candle-backtest --days 7 --search --search-min-trades 5
```

This pulls recent Hyperliquid candles and applies the current detector and operator policy with a conservative OHLC fill model. It is useful for rejecting obviously bad slices before paper trading, but it is still not a go-live substitute.
The `--search` flag ranks session/side/level subsets that already passed the current detector thresholds, which keeps policy changes tied to current-market evidence.
`--recommend-env` is stricter than the ranked subset list: by default a recommendation needs at least 5 trades for the coin, 2 trades for each selected session, and 2 trades for each selected level family, all with positive expectancy and profit factor at or above 1.20. This prevents a tiny or mixed-quality subset from being copied directly into `.env`.
The report prints both the requested lookback and the actual `span_days` returned by Hyperliquid. It also prints `candle_fetch_chunks`, `nonempty`, and `empty` so API history caps are visible. The go-live gate requires `recent_candle_edge` to meet `--min-candle-span-days` (default `3.5`) so a capped candle response cannot masquerade as a full requested lookback.

Before copying a candidate into `.env`, validate it against the strict freshness gate without changing runtime config:

```bash
./scripts/botctl.sh policy-candidate-check --clear-policy \
  --set HL_COINS=TON \
  --set BOT_ALLOW_COINS=TON \
  --set BOT_ALLOW_LEVEL_LABELS=prior_15m_low \
  --set BOT_ALLOW_COIN_LEVELS=TON:prior_15m_low \
  --set BOT_ALLOW_COIN_SESSIONS=TON:eu \
  --set BOT_ALLOW_COIN_SESSION_LEVELS=TON:eu:prior_15m_low \
  --set BOT_ALLOW_SESSIONS=eu \
  --set BOT_ALLOW_SIDES=long
```

To test whether the latest blocked paper signal should be admitted, validate that rejected slice directly:

```bash
./scripts/botctl.sh rejected-signal-check
```

To distinguish healthy paper collection from actual live readiness:

```bash
./scripts/botctl.sh trade-readiness
```

To watch the current paper run without leaving an unbounded log follow open, run:

```bash
./scripts/botctl.sh money-status
./scripts/botctl.sh proof-watch --timeout-sec 900 --poll-sec 15
./scripts/botctl.sh proof-watch --once --json
```

The watch prints the current run funnel, paper PnL, UTC session gate, and next allowed session. It exits early on a fresh paper entry or closed trade, and exits with a timeout if no new proof appears.

`./scripts/botctl.sh replay` intentionally ignores the live runtime pause file so replay output reflects the strategy and operator policy, not the profit watcher's session wait pause.

For actual profit proof, wait for a resolved profitable trade instead of just an entry:

```bash
./scripts/botctl.sh profit-watch --timeout-sec 3600 --poll-sec 30
```

To wait until the next US session and then watch for profitable close proof:

```bash
./scripts/botctl.sh us-profit-watch --dry-run --max-wait-sec 86400 --timeout-sec 3600 --poll-sec 30
./scripts/botctl.sh us-profit-watch --max-wait-sec 86400 --timeout-sec 3600 --poll-sec 30
./scripts/botctl.sh profit-watch-status
```

Because the Docker container does not bind-mount `src/`, verify the running image after code changes:

```bash
./scripts/botctl.sh runtime-code-check
```

For a concise objective-level check, run:

```bash
./scripts/botctl.sh completion-audit
```

It exits nonzero until the objective is actually achieved.

`./scripts/botctl.sh edge-status` also runs a current-session opportunity scan across the configured major-coin scan basket with operator policy ignored. The displayed subset search and recommendation are both restricted to the current UTC strategy session. Use that section to decide whether the bot is idle because there is no positive current-session recommendation, or because the active policy is too restrictive relative to fresh evidence. Override the basket with `BOT_EDGE_STATUS_SCAN_COINS` when testing a narrower hypothesis.

## History warm-start

`BOT_HISTORY_WARM_START_ENABLED=true` seeds the detector from recent Hyperliquid candles before the websocket loop starts. This avoids the first 15+ minutes of `skip_history` after restarts. In paper mode, candles are selected from the configured public websocket environment, not from `HL_NETWORK`, because `HL_NETWORK` only controls live execution.

## Log rotation

`runtime/bot.log` is rotated at container start when it exceeds `BOT_LOG_MAX_BYTES`. `runtime/market_events.jsonl` is rotated by the capture writer when `BOT_MARKET_CAPTURE_MAX_BYTES` is set.

```bash
BOT_LOG_MAX_BYTES=52428800
BOT_MARKET_CAPTURE_MAX_BYTES=1073741824
BOT_MARKET_CAPTURE_BACKUPS=3
```

## Known design choices (not bugs)

- **Paper fills are optimistic.** Paper mode fills entries on price-touch; live mode places real maker limits that may not fill due to queue position. Phase E (testnet observation) is the place to measure this drift.
- **Stops/TPs are code-watched, not exchange-native.** The bot watches each tick and submits market_close on trigger. This is simpler than HL's native trigger orders but pays taker fees on TP exits and depends on the bot being alive. The deadman switch (60s TTL) is the safety net for unattended bot death.
- **TP1/TP2 use market_close (taker), not resting limits (maker rebate).** Documented trade-off: simpler + matches paper's exit timing, costs ~$0.04/trade in extra fees. v2 candidate.
- **Every bot start gets a run_id.** Use `scripts/journal_report.py --last-run` for post-restart performance instead of mixing months of old journal rows into current decisions.
