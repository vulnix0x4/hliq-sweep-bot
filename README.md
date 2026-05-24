# Hyperliquid Liquidity Sweep Bot (MVP)

This is a paper-trading-first implementation of the strategy you sketched:

- Real-time feed: Hyperliquid WebSocket (`trades` + `l2Book`)
- Deterministic signal: liquidity sweep + rejection + volume/spread/trend filters
- Regime adaptation: confidence/risk shaping by trend/volatility/spread/session
- ML gate (optional): model-based allow/deny after deterministic signal generation
- Execution: post-only style pending limit entries, hard stop, TP1/TP2, time stop
- Risk governor: dynamic position sizing with daily `-R` lockout and circuit breakers

## Current status

- **Paper mode** and offline **replay mode** — production-stable, default
- **Live mode** (Hyperliquid mainnet/testnet) - adapter implemented, gated behind `BOT_MODE=live` + `BOT_ALLOW_LIVE=true`. Requires a one-time agent-wallet approval via `scripts/approve_agent.py`. Operator-facing emergency exit via `scripts/flatten_live.py`. **Run on testnet for at least a week before mainnet.**
- Unit tests cover paper, replay, and live safety paths.

See `docs/plans/2026-04-25-hyperliquid-live-execution.md` for the live rollout playbook (Phases E/F/G).

## Project layout

```text
src/hliq_bot/
  config.py
  data/
    hyperliquid_ws.py
    bar_builder.py
  signal/
    levels.py
    sweep_detector.py
  risk/
    governor.py
  execution/
    order_manager.py
  bot.py
  main.py
tests/
scripts/
  train_gate.py
```

## Quick start

1. Copy env.

```bash
cp .env.example .env
```

2. Start in background with Docker.

```bash
./scripts/botctl.sh up
```

3. Watch logs / status / stats.

```bash
./scripts/botctl.sh status
./scripts/botctl.sh follow
./scripts/botctl.sh stats
./scripts/botctl.sh report
./scripts/botctl.sh report-last
./scripts/botctl.sh policy-report --input runtime/archive/signals_pre_tune_20260511T224920Z.jsonl --show-selected
./scripts/botctl.sh candle-backtest --days 7 --search --search-min-trades 5
./scripts/botctl.sh policy-candidate-check --clear-policy --set HL_COINS=TON --set BOT_ALLOW_COINS=TON
./scripts/botctl.sh rejected-signal-check
./scripts/botctl.sh trade-readiness
./scripts/botctl.sh money-status
./scripts/botctl.sh edge-status
./scripts/botctl.sh completion-audit
./scripts/botctl.sh proof-watch --timeout-sec 900 --poll-sec 15
./scripts/botctl.sh profit-watch --timeout-sec 3600 --poll-sec 30
./scripts/botctl.sh us-profit-watch --timeout-sec 3600 --poll-sec 30
./scripts/botctl.sh profit-watch-status
./scripts/botctl.sh replay
./scripts/botctl.sh snapshot
./scripts/botctl.sh ml-status
./scripts/botctl.sh codex-status
```

4. Train/update ML gate (once you have enough resolved trades).

```bash
./scripts/botctl.sh train-ml
./scripts/botctl.sh ml-status
./scripts/botctl.sh apply-ml-threshold
```

5. Stop bot.

```bash
./scripts/botctl.sh down
```

Logs are persisted to `runtime/bot.log` so stats survive container restarts/rebuilds.
If `BOT_MARKET_CAPTURE_ENABLED=true`, raw market events are also captured to `runtime/market_events.jsonl` for replay.

## Why Docker first

If your host is missing `python3-venv` or `pip`, Docker avoids local Python setup issues and runs the bot with pinned dependencies.

## Local (non-Docker) option

If you prefer local execution:

```bash
sudo apt update
sudo apt install -y python3.13-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m hliq_bot.main
```

## Key env vars

```bash
export BOT_MODE=paper
export HL_WS_URL=wss://api.hyperliquid.xyz/ws
export HL_COINS=TON,SPX,HYPE,NEAR
export CODEX_AUTH_DIR=/home/vulnix/.codex

export BOT_TIMEFRAME_SEC=60
export BOT_MIN_SWEEP_BPS=2
export BOT_MAX_SWEEP_BPS=24
export BOT_MIN_RECLAIM_BPS=2
export BOT_VOLUME_LOOKBACK_BARS=20
export BOT_VOLUME_SPIKE_MULT=0.8
export BOT_WICK_BODY_RATIO_MIN=1.1
export BOT_MIN_STOP_DISTANCE_BPS=6
export BOT_MAX_STOP_DISTANCE_BPS=55
export BOT_TP1_BPS=35
export BOT_TP2_BPS=80
export BOT_MIN_RR_TP1=0.6
export BOT_MIN_RR_TP2=1.2
export BOT_BREAK_EVEN_PROGRESS_TP1_FRAC=0.45
export BOT_ENTRY_EXPIRY_SEC=180
export BOT_ENTRY_TOUCH_TOL_BPS=2
export BOT_TIME_STOP_SEC=600
export BOT_MAX_HOLDING_SEC=1800
export BOT_MIN_SIGNAL_SCORE=0.40
export BOT_MIN_CONF_RANGE=0.65
export BOT_MIN_CONF_TREND=0.72
export BOT_USE_MICRO_CONFIRM=true
export BOT_MICRO_FLOW_WINDOW_SEC=5
export BOT_MIN_OFI_RATIO=0.06
export BOT_MIN_QUEUE_IMBALANCE=0.03
export BOT_USE_FUNDING_BLACKOUT=true
export BOT_FUNDING_BLACKOUT_SEC=90
export BOT_WARMUP_ENABLED=false
export BOT_WARMUP_TARGET_RESOLVED=30
export BOT_WARMUP_MICRO_RELAX=false
export BOT_WARMUP_OFI_SCALE=0.5
export BOT_WARMUP_QIMB_SCALE=0.5
export BOT_WARMUP_MICRO_OR_LOGIC=false
export BOT_WARMUP_RISK_MULT_CAP=0.75
export BOT_LONG_ONLY=false
export BOT_ALLOW_COINS=TON,SPX,HYPE,NEAR
export BOT_ALLOW_LEVEL_LABELS=equal_low_*,prior_15m_low,equal_high_*
export BOT_ALLOW_COIN_LEVELS=TON:prior_15m_low,TON:equal_high_*,SPX:equal_low_*,HYPE:equal_high_*,HYPE:equal_low_*,NEAR:equal_low_*,NEAR:prior_15m_low
export BOT_ALLOW_COIN_SESSIONS=TON:eu,SPX:asia,HYPE:eu,HYPE:us,NEAR:eu,NEAR:us
export BOT_ALLOW_COIN_SESSION_LEVELS=TON:eu:prior_15m_low,TON:eu:equal_high_*,SPX:asia:equal_low_*,HYPE:eu:equal_high_*,HYPE:us:equal_low_*,NEAR:eu:equal_low_*,NEAR:eu:prior_15m_low,NEAR:us:equal_low_*
export BOT_ALLOW_SESSIONS=asia,eu,us
export BOT_ALLOW_SIDES=long,short
export BOT_BLOCK_COINS=
export BOT_BLOCK_LEVEL_LABELS=
export BOT_BLOCK_COIN_LEVELS=
export BOT_BLOCK_COIN_SESSIONS=
export BOT_BLOCK_COIN_SESSION_LEVELS=
export BOT_BLOCK_SESSIONS=late
export BOT_MARKET_CAPTURE_ENABLED=true
export BOT_MARKET_CAPTURE_PATH=runtime/market_events.jsonl
export BOT_REPLAY_INPUT_PATH=runtime/market_events.jsonl
export BOT_ML_STATE_PATH=runtime/ml_state.json
export BOT_ML_ENABLED=false
export BOT_ML_DECISION_MODE=rank
export BOT_ML_PROVIDER=ensemble
export BOT_ML_MODEL_MIN_SAMPLES=20
export BOT_ML_MIN_PROB=0.58
export BOT_ML_THRESHOLD_FLOOR=0.52
export BOT_ML_THRESHOLD_CAP=0.85
export BOT_ML_ADAPTIVE_THRESHOLD=true
export BOT_ML_ADAPTIVE_WINDOW=40
export BOT_ML_ADAPTIVE_MIN_TRADES=12
export BOT_ML_ADAPTIVE_STEP=0.03
export BOT_ML_ENSEMBLE_CODEX_WEIGHT=0.65
export BOT_ML_AUTO_TRAIN=true
export BOT_ML_AUTO_TRAIN_INTERVAL_SEC=1800
export BOT_ML_AUTO_TRAIN_MIN_RESOLVED=8
export BOT_ML_AUTO_TRAIN_MIN_NEW_TRADES=3
export BOT_ML_AUTO_APPLY_THRESHOLD=true
export BOT_ML_FAIL_OPEN=false
export BOT_CODEX_MODEL=gpt-5.3-codex
export BOT_CODEX_TIMEOUT_SEC=20
export BOT_CODEX_MIN_INTERVAL_SEC=15

export BOT_ACCOUNT_EQUITY=50
export BOT_RISK_PER_TRADE_PCT=0.50
export BOT_DAILY_LOSS_LIMIT_R=1.5
export BOT_MAX_LEVERAGE=1
export BOT_MAX_OPEN_POSITIONS=1
export BOT_MAX_POSITIONS_PER_COIN=1
export BOT_PORTFOLIO_MAX_POSITIONS=1
export BOT_LOSS_COOLDOWN_SEC=900
export BOT_HARD_LOSS_R=-0.90
export BOT_HARD_LOSS_COOLDOWN_SEC=300
export BOT_SIDE_HARD_LOSS_R=-0.90
export BOT_SIDE_HARD_LOSS_COOLDOWN_SEC=1800
export BOT_LEVEL_HARD_LOSS_R=-0.90
export BOT_LEVEL_HARD_LOSS_COOLDOWN_SEC=3600
export BOT_EDGE_PAUSE_AVG_R=-0.25
export BOT_EDGE_PAUSE_MIN_TRADES=30
export BOT_SIDE_EDGE_PAUSE_AVG_R=-0.25
export BOT_SIDE_EDGE_PAUSE_MIN_TRADES=30
export BOT_SIDE_EDGE_PAUSE_COOLDOWN_SEC=900
export BOT_SESSION_EDGE_PAUSE_AVG_R=-0.40
export BOT_SESSION_EDGE_PAUSE_MIN_TRADES=30
export BOT_LEVEL_EDGE_PAUSE_AVG_R=-0.40
export BOT_LEVEL_EDGE_PAUSE_MIN_TRADES=30
export BOT_MAKER_FEE_PCT=0.00015
export BOT_TAKER_FEE_PCT=0.00045
export BOT_PAPER_ENTRY_SLIPPAGE_BPS=0
export BOT_PAPER_EXIT_SLIPPAGE_BPS=1.5
export BOT_PAPER_TP1_IS_TAKER=true
export BOT_ALLOW_LIVE=false
export HL_NETWORK=testnet
export HL_MAX_NOTIONAL_PER_TRADE=50
```

## Strategy defaults in this MVP

- Trigger timeframe: `1m` bars
- Levels: prior 15m high/low, prior 1h high/low, and clustered equal highs/lows
- Trigger: highest-ranked sweep by `4-16 bps` + close back inside + wick ratio + volume filter
- Reclaim quality floor: require close back inside level by `BOT_MIN_RECLAIM_BPS`
- Entries: paper post-only style limit at level retest
- Stop: sweep extreme + `12 bps`
- Promote stop to break-even once price reaches `BOT_BREAK_EVEN_PROGRESS_TP1_FRAC` of TP1 path
- Stop-distance filter: skip setups with too-tight or too-wide stop distance
- RR floor: enforce minimum reward/risk to TP1 and TP2
- TP: `0.65%` then `1.4%`
- Microstructure confirmation: require signed flow and queue imbalance alignment before entry
- Funding blackout: avoid entries around 00:00/08:00/16:00 UTC windows
- Time stop: `3 min` if still not profitable
- Max holding time: `30 min` absolute cap
- Entry touch tolerance: `1 bps` (paper fill realism)
- Adaptive confidence floor by regime (`range` vs `trend`)
- Portfolio-style allocator scales risk by recent R-performance and session/regime
- Edge kill-switch: pauses new trades if rolling expectancy degrades beyond threshold
- Side kill-switch: pauses a side (long/short) if that side's recent expectancy degrades
- Session kill-switch: pauses weak sessions after repeated underperformance
- Level kill-switch: pauses weak level labels after repeated underperformance
- Operator blocklists can suppress coins, sessions, sides, or level labels; level labels accept shell-style globs like `equal_low_*`
- Loss cooldown: short no-trade period after a loss to avoid immediate re-entry churn
- Hard-loss quarantine: global/side/level cooldowns after large adverse R outcomes
- Optional ML gate can rank or veto low-quality signals (`BOT_ML_ENABLED=true`, `BOT_ML_DECISION_MODE=rank|gate`)
- Ensemble ML mode combines local logistic + Codex probability (`BOT_ML_PROVIDER=ensemble`)
- Adaptive ML threshold tightens/loosens allowed probability floor from recent realized outcomes
- Captured market events can be replayed through the exact bot logic with `./scripts/botctl.sh replay`
- Each bot start writes a `run_id` into `runtime/signals.jsonl`; use `./scripts/botctl.sh report` for all history or `./scripts/botctl.sh report-last` for the latest run only
- `./scripts/botctl.sh go-live-check` gives a hard pass/fail before real-money mode is enabled
- Warmup mode can temporarily relax micro-confirmation logic with capped risk until enough resolved trades are collected
- Circuit breakers:
  - spread too wide
  - stale data / WS unhealthy
  - 30s move spike above threshold
  - consecutive wide-range bars

## Notes for live deployment

- Keep real-money trading off unless `BOT_MODE=live`, `BOT_ALLOW_LIVE=true`, and `HL_NETWORK` are deliberately set for that run.
- Start or resume on testnet after strategy changes, especially when blocklists, fee assumptions, or execution behavior change.
- Do not allow model outputs to override stops, size, blocklists, or circuit breakers.
- Live positions place native reduce-only exchange stops, and the bot reconciles or replaces them after restart and partial exits. Keep the process running for trailing-stop updates, TP exits, journaling, and policy enforcement.

## Codex OAuth Gate (OpenClaw-style)

- Set `BOT_ML_ENABLED=true` and choose:
  - `BOT_ML_PROVIDER=codex_cli` (Codex only), or
  - `BOT_ML_PROVIDER=ensemble` (Codex + local logistic blend).
- The bot calls `codex exec` as a subprocess for candidate-signal gating.
- In Docker, OAuth tokens are read from `${CODEX_AUTH_DIR:-/home/vulnix/.codex}` mounted to `/root/.codex`.
- If Codex is unavailable or rate-limited, behavior follows `BOT_ML_FAIL_OPEN`:
  - `true`: allow deterministic strategy to continue.
  - `false`: deny new signals when Codex gate fails.
- `./scripts/botctl.sh train-ml` now writes a `recommended_min_prob` into the model file based on out-of-sample expectancy.
- `./scripts/botctl.sh apply-ml-threshold` copies that recommendation into `.env` and restarts the bot.
- If `BOT_ML_AUTO_TRAIN=true`, the bot retrains itself in background after enough resolved trades and auto-applies `recommended_min_prob` without manual commands.
