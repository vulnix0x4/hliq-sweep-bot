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
  replay)
    python3 scripts/replay_capture.py \
      --input runtime/market_events.jsonl \
      --journal runtime/replay_signals.jsonl
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
