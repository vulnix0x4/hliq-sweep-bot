#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE="${COMPOSE_BIN:-docker compose}"

usage() {
  cat <<'EOF'
Usage: ./scripts/botctl.sh <command>

Commands:
  up         Build and start bot in background
  down       Stop and remove bot container
  restart    Restart bot container
  status     Show container status
  logs       Show recent logs (tail=200)
  follow     Follow live logs
  stats      Summarize trading stats from logs
  report     Analyze runtime/signals.jsonl funnel + expectancy
  report-last  Analyze only the latest journaled run_id
  policy-report  Apply current .env policy filters to a historical journal
  candle-backtest Approximate current policy expectancy from recent HL candles
  policy-candidate-check  Validate candidate policy KEY=VALUE overrides
  rejected-signal-check  Validate the latest blocked signal as a candidate policy
  trade-readiness  Summarize paper collection health and live blockers
  session-status  Show current UTC strategy session and next allowed window
  proof-watch  Bounded watch for fresh paper entry/close proof
  money-status  One-shot PnL/exposure/session status
  profit-watch  Wait for a new profitable closed paper trade
  us-profit-watch  Wait through the next EU/US window for profitable close proof
  start-us-profit-watch  Start container-side EU/US profit watcher in background
  profit-watch-status  Show container-side profit watcher process and log
  go-live-check  Hard pass/fail readiness check before enabling live mode
  runtime-code-check  Compare host trading code to running container image
  edge-status  Current status + policy evidence + go-live gate
  completion-audit  Check objective evidence: code, PnL/exposure, go-live gate
  replay     Replay runtime/market_events.jsonl into runtime/replay_signals.jsonl
  train-ml   Train ML gate model from runtime/signals.jsonl
  ml-status  Show ML model metadata if present
  apply-ml-threshold  Sync model's recommended_min_prob into .env and restart bot
  codex-status  Show Codex CLI/OAuth availability inside container
  snapshot   Status + recent logs + stats
  help       Show this message
EOF
}

ensure_prereqs() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or not in PATH." >&2
    exit 1
  fi
}

ensure_env_file() {
  if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
  fi
}

run_compose() {
  # shellcheck disable=SC2086
  $COMPOSE "$@"
}

stats_input() {
  if [[ -s runtime/bot.log ]]; then
    python3 scripts/paper_stats.py runtime/bot.log
  else
    run_compose logs --no-color --no-log-prefix bot | python3 scripts/paper_stats.py
  fi
}

runtime_code_check() {
  ensure_prereqs
  local paths=(
    src/hliq_bot/bot.py
    src/hliq_bot/config.py
    src/hliq_bot/models.py
    src/hliq_bot/risk/governor.py
    src/hliq_bot/signal/sweep_detector.py
    src/hliq_bot/execution/order_manager.py
    src/hliq_bot/execution/hyperliquid_order_manager.py
    scripts/proof_watch.py
    scripts/session_profit_watch.py
    scripts/session_status.py
    scripts/go_live_check.py
    scripts/candle_backtest.py
    scripts/policy_candidate_check.py
    scripts/policy_freshness_check.py
    scripts/policy_report.py
    scripts/rejected_signal_check.py
    scripts/replay_capture.py
    scripts/botctl.sh
    scripts/trade_readiness.py
  )
  local mismatch=0
  echo "Runtime Code Check"
  echo "========================================"
  for path in "${paths[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "[FAIL] $path missing on host"
      mismatch=1
      continue
    fi
    local host_sum
    host_sum="$(sha256sum "$path" | awk '{print $1}')"
    local container_sum
    if ! container_sum="$(run_compose exec -T bot sha256sum "/app/$path" 2>/dev/null | awk '{print $1}')"; then
      container_sum=""
    fi
    if [[ -z "$container_sum" ]]; then
      echo "[FAIL] $path missing in running container"
      mismatch=1
    elif [[ "$host_sum" == "$container_sum" ]]; then
      echo "[PASS] $path"
    else
      echo "[FAIL] $path host=$host_sum container=$container_sum"
      mismatch=1
    fi
  done
  if [[ "$mismatch" -eq 0 ]]; then
    echo "status: PASS"
  else
    echo "status: FAIL - rebuild/recreate with: ./scripts/botctl.sh up"
  fi
  return "$mismatch"
}

start_us_profit_watch() {
  ensure_prereqs
  mkdir -p runtime
  if profit_watch_running; then
    echo "Session profit watcher already running."
  else
    rm -f runtime/us_profit_watch.log
    run_compose exec -T -d bot sh -lc \
      'cd /app && while true; do python /app/scripts/session_profit_watch.py --loop --max-wait-sec 86400 --timeout-sec 28800 --poll-sec 30 --edge-min-session-trades "${BOT_EDGE_MIN_SESSION_TRADES:-2}" --edge-min-coin-trades "${BOT_EDGE_MIN_COIN_TRADES:-5}" --edge-min-level-trades "${BOT_EDGE_MIN_LEVEL_TRADES:-2}"; rc="$?"; if [ "$rc" -eq 0 ]; then echo "Session profit watcher completed successfully."; break; fi; echo "Session profit watcher exited $rc; restarting in 30s."; sleep 30; done >> /app/runtime/us_profit_watch.log 2>&1'
    echo "Started session profit watcher. Log: runtime/us_profit_watch.log"
  fi
}

profit_watch_running() {
  run_compose top bot | grep -F "python /app/scripts/session_profit_watch.py" | grep -vq "sh -lc"
}

profit_watch_supervised() {
  run_compose top bot | grep -F "while true; do python /app/scripts/session_profit_watch.py" | grep -q "restarting in 30s"
}

cmd="${1:-help}"

case "$cmd" in
  help|--help|-h)
    usage
    ;;
  up)
    ensure_prereqs
    ensure_env_file
    mkdir -p runtime
    run_compose up -d --build
    run_compose ps
    start_us_profit_watch
    ;;
  down)
    ensure_prereqs
    run_compose down
    ;;
  restart)
    ensure_prereqs
    run_compose restart bot
    ;;
  status)
    ensure_prereqs
    run_compose ps
    ;;
  logs)
    ensure_prereqs
    run_compose logs --tail=200 bot
    ;;
  follow)
    ensure_prereqs
    run_compose logs -f bot
    ;;
  stats)
    ensure_prereqs
    stats_input
    ;;
  report)
    python3 scripts/journal_report.py --input runtime/signals.jsonl
    ;;
  report-last)
    python3 scripts/journal_report.py --input runtime/signals.jsonl --last-run
    ;;
  policy-report)
    python3 scripts/policy_report.py "${@:2}"
    ;;
  candle-backtest)
    shift
    candle_args=()
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --current-session)
          candle_args+=("--recommend-current-session")
          shift
          ;;
        --ignore-policy)
          candle_args+=("--ignore-operator-policy")
          shift
          ;;
        *)
          candle_args+=("$1")
          shift
          ;;
      esac
    done
    python3 scripts/candle_backtest.py "${candle_args[@]}"
    ;;
  policy-candidate-check)
    python3 scripts/policy_candidate_check.py "${@:2}"
    ;;
  rejected-signal-check)
    python3 scripts/rejected_signal_check.py "${@:2}"
    ;;
  trade-readiness)
    python3 scripts/trade_readiness.py "${@:2}"
    ;;
  session-status)
    python3 scripts/session_status.py
    ;;
  proof-watch)
    python3 scripts/proof_watch.py "${@:2}"
    ;;
  money-status)
    python3 scripts/proof_watch.py --once --active-policy-runs
    ;;
  profit-watch)
    python3 scripts/proof_watch.py --exit-on close --require-profit "${@:2}"
    ;;
  us-profit-watch)
    python3 scripts/session_profit_watch.py "${@:2}"
    ;;
  start-us-profit-watch)
    start_us_profit_watch
    ;;
  profit-watch-status)
    ensure_prereqs
    echo "== Watcher Process =="
    if ! run_compose top bot | grep -F "python /app/scripts/session_profit_watch.py" | grep -v "sh -lc"; then
      echo "not running"
    fi
    echo
    echo "== Watcher Supervisor =="
    if profit_watch_supervised; then
      echo "supervised retry loop active"
    else
      echo "supervised retry loop not found"
    fi
    echo
    echo "== Watcher Log =="
    if [[ -f runtime/us_profit_watch.log ]]; then
      tail -n 40 runtime/us_profit_watch.log
    else
      echo "No runtime/us_profit_watch.log yet."
    fi
    ;;
  go-live-check)
    python3 scripts/go_live_check.py "${@:2}"
    ;;
  runtime-code-check)
    runtime_code_check
    ;;
  edge-status)
    ensure_prereqs
    echo "== Status =="
    run_compose ps
    echo
    echo "== Runtime Code Check =="
    runtime_code_check || true
    echo
    echo "== Latest Paper Run =="
    python3 scripts/journal_report.py --input runtime/signals.jsonl --last-run || true
    echo
    echo "== Session Status =="
    python3 scripts/session_status.py || true
    echo
    echo "== Current Policy On Archived Outcomes =="
    if [[ -f runtime/archive/signals_pre_tune_20260511T224920Z.jsonl ]]; then
      python3 scripts/policy_report.py \
        --input runtime/archive/signals_pre_tune_20260511T224920Z.jsonl \
        --show-selected || true
    else
      echo "No archived policy sample found."
    fi
    echo
    echo "== Current Policy Candle Backtest =="
    python3 scripts/candle_backtest.py \
      --days "${BOT_EDGE_STATUS_BACKTEST_DAYS:-7}" \
      --search \
      --search-min-trades "${BOT_EDGE_STATUS_SEARCH_MIN_TRADES:-5}" || true
    echo
    echo "== Current Session Opportunity Scan =="
    python3 scripts/candle_backtest.py \
      --days "${BOT_EDGE_STATUS_BACKTEST_DAYS:-7}" \
      --coins "${BOT_EDGE_STATUS_SCAN_COINS:-BTC,DOGE,ETH,SOL,HYPE,XRP,BNB,AVAX,LINK,SUI}" \
      --ignore-operator-policy \
      --search \
      --search-min-trades "${BOT_EDGE_STATUS_SEARCH_MIN_TRADES:-5}" \
      --search-current-session \
      --recommend-env \
      --recommend-current-session || true
    echo
    echo "== Current Policy Replay =="
    if [[ -f runtime/replay_current_policy_signals.jsonl ]]; then
      python3 scripts/journal_report.py --input runtime/replay_current_policy_signals.jsonl || true
    else
      echo "No current-policy replay found. Run: docker compose run --rm --entrypoint sh bot -c 'python /app/scripts/replay_capture.py --input /app/runtime/market_events.jsonl --journal /app/runtime/replay_current_policy_signals.jsonl'"
    fi
    echo
    echo "== Go-Live Gate =="
    python3 scripts/go_live_check.py \
      --active-policy-runs \
      --min-candle-session-trades "${BOT_EDGE_STATUS_MIN_CANDLE_SESSION_TRADES:-2}" \
      --min-candle-coin-trades "${BOT_EDGE_STATUS_MIN_CANDLE_COIN_TRADES:-5}" \
      --min-candle-level-trades "${BOT_EDGE_STATUS_MIN_CANDLE_LEVEL_TRADES:-2}" \
      || true
    ;;
  completion-audit)
    ensure_prereqs
    audit_status=0
    echo "== Objective =="
    echo "make this trading bot make money"
    echo
    echo "== Success Criteria =="
    echo "- running bot code matches host trading code"
    echo "- paper/live PnL is positive on fresh closed trades"
    echo "- no unresolved exposure is hiding losses"
    echo "- go-live gate passes before any real-money mode"
    echo
    echo "== Runtime Code =="
    runtime_code_check || audit_status=1
    echo
    echo "== Money / Exposure =="
    python3 scripts/proof_watch.py --once --active-policy-runs --require-positive-pnl || audit_status=1
    echo
    echo "== Profit Watcher =="
    if profit_watch_running; then
      echo "[PASS] profit watcher running"
    else
      echo "[FAIL] profit watcher not running"
      audit_status=1
    fi
    if profit_watch_supervised; then
      echo "[PASS] profit watcher supervised"
    else
      echo "[FAIL] profit watcher supervisor not found"
      audit_status=1
    fi
    echo
    echo "== Go-Live Gate =="
    python3 scripts/go_live_check.py \
      --active-policy-runs \
      --min-candle-session-trades "${BOT_COMPLETION_MIN_CANDLE_SESSION_TRADES:-2}" \
      --min-candle-coin-trades "${BOT_COMPLETION_MIN_CANDLE_COIN_TRADES:-5}" \
      --min-candle-level-trades "${BOT_COMPLETION_MIN_CANDLE_LEVEL_TRADES:-2}" \
      || audit_status=1
    echo
    if [[ "$audit_status" -eq 0 ]]; then
      echo "completion: ACHIEVED"
    else
      echo "completion: NOT ACHIEVED"
    fi
    exit "$audit_status"
    ;;
  replay)
    python3 scripts/replay_capture.py \
      --input runtime/market_events.jsonl \
      --journal runtime/replay_signals.jsonl \
      --ignore-runtime-pause
    python3 scripts/journal_report.py --input runtime/replay_signals.jsonl || true
    ;;
  snapshot)
    ensure_prereqs
    echo "== Status =="
    run_compose ps
    echo
    echo "== Recent Logs =="
    run_compose logs --tail=40 bot
    echo
    echo "== Stats =="
    stats_input
    echo
    echo "== Journal Report =="
    python3 scripts/journal_report.py --input runtime/signals.jsonl || true
    ;;
  train-ml)
    default_min_prob="$(awk -F= '/^BOT_ML_MIN_PROB=/{print $2}' .env 2>/dev/null | tail -n1)"
    if [[ -z "${default_min_prob:-}" ]]; then
      default_min_prob="0.62"
    fi
    python3 scripts/train_gate.py \
      --input runtime/signals.jsonl \
      --output runtime/models/gate_model.json \
      --default-min-prob "$default_min_prob"
    ;;
  ml-status)
    if [[ -f runtime/models/gate_model.json ]]; then
      python3 - <<'PY'
import json
from pathlib import Path
p = Path("runtime/models/gate_model.json")
raw = json.loads(p.read_text(encoding="utf-8"))
print("model:", p)
print("model_type:", raw.get("model_type"))
print("train_samples:", raw.get("train_samples"))
print("positive_rate:", raw.get("positive_rate"))
print("feature_count:", len(raw.get("features", [])))
print("recommended_min_prob:", raw.get("recommended_min_prob"))
val = raw.get("validation", {})
if isinstance(val, dict) and val:
    print("validation:", json.dumps(val, sort_keys=True))
PY
    else
      echo "No model at runtime/models/gate_model.json"
    fi
    ;;
  apply-ml-threshold)
    ensure_env_file
    if [[ ! -f runtime/models/gate_model.json ]]; then
      echo "No model at runtime/models/gate_model.json" >&2
      exit 1
    fi
    rec_prob="$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("runtime/models/gate_model.json")
raw = json.loads(p.read_text(encoding="utf-8"))
val = raw.get("recommended_min_prob")
if val is None:
    print("")
else:
    try:
        x = float(val)
    except Exception:
        print("")
    else:
        x = max(0.0, min(1.0, x))
        print(f"{x:.2f}")
PY
)"
    if [[ -z "${rec_prob:-}" ]]; then
      echo "Model does not contain recommended_min_prob; run train-ml first." >&2
      exit 1
    fi
    old_prob="$(awk -F= '/^BOT_ML_MIN_PROB=/{print $2}' .env 2>/dev/null | tail -n1)"
    if grep -q '^BOT_ML_MIN_PROB=' .env; then
      sed -i "s/^BOT_ML_MIN_PROB=.*/BOT_ML_MIN_PROB=${rec_prob}/" .env
    else
      echo "BOT_ML_MIN_PROB=${rec_prob}" >> .env
    fi
    echo "Updated BOT_ML_MIN_PROB: ${old_prob:-unset} -> ${rec_prob}"
    ensure_prereqs
    run_compose restart bot
    ;;
  codex-status)
    ensure_prereqs
    run_compose exec -T bot sh -lc '
      set -e
      if command -v codex >/dev/null 2>&1; then
        codex --version
      else
        echo "codex CLI not found in container"
      fi
      if [ -f /root/.codex/auth.json ]; then
        echo "oauth_auth_file: present (/root/.codex/auth.json)"
      else
        echo "oauth_auth_file: missing (/root/.codex/auth.json)"
      fi
    '
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
