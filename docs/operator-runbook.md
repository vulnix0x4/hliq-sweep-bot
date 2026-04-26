# Operator Runbook

## Pre-deployment checks

Before flipping `BOT_MODE=live`:

1. **Rebuild the Docker image.** The currently-running paper container does NOT have the `hyperliquid-python-sdk` installed (the SDK was added after the image was built). Live mode will crash with `ModuleNotFoundError`. Run:
   ```bash
   docker compose build
   docker compose down && docker compose up -d
   docker exec hliq-paper-bot python3 -c "from hyperliquid.exchange import Exchange; print('OK')"
   ```
   The third command must print `OK` before going live.

2. **Approve an agent wallet** via `scripts/approve_agent.py`. NEVER put your main private key in `.env`. The agent key has trade-only permission.

3. **Set both safety flags.** Live activates only when `BOT_MODE=live` AND `BOT_ALLOW_LIVE=true`. Boot will fail if only one is set.

4. **Start with `HL_NETWORK=testnet`** for at least 7 days. Compare testnet outcomes to paper expectations.

## Replay data hygiene

`runtime/market_events.jsonl` (the captured event stream used by `scripts/replay_capture.py`) was originally captured before the `coin` field was added to event records. ~76% of legacy rows are coin-less and get routed to the first worker (BTC) on replay, which biases multi-coin replay results.

Mitigation:
- For multi-coin replay studies, use only events captured AFTER commit `9877350` (coin-field fix).
- Newly captured events ARE properly tagged.
- A clean per-coin split would be the right long-term fix.

## Log rotation

`runtime/bot.log` (~35 MB and growing) and `runtime/market_events.jsonl` (~4.4 GB and growing) are appended to indefinitely. `docker-compose.yml` uses `tee -a` with no rotation. For long-running deployments, configure `logrotate` or manually truncate periodically:

```bash
# Truncate without bot restart
truncate -s 0 runtime/bot.log

# Or rotate market_events.jsonl monthly
mv runtime/market_events.jsonl runtime/archive/market_events.$(date +%Y%m%d).jsonl
touch runtime/market_events.jsonl
```

## Known design choices (not bugs)

- **Paper fills are optimistic.** Paper mode fills entries on price-touch; live mode places real maker limits that may not fill due to queue position. Phase E (testnet observation) is the place to measure this drift.
- **Stops/TPs are code-watched, not exchange-native.** The bot watches each tick and submits market_close on trigger. This is simpler than HL's native trigger orders but pays taker fees on TP exits and depends on the bot being alive. The deadman switch (60s TTL) is the safety net for unattended bot death.
- **TP1/TP2 use market_close (taker), not resting limits (maker rebate).** Documented trade-off: simpler + matches paper's exit timing, costs ~$0.04/trade in extra fees. v2 candidate.
