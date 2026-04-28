#!/bin/bash
# Auto-runs after the testnet faucet drips. Mints an agent, updates .env,
# restarts the container in live testnet mode. Idempotent: re-running
# regenerates the agent and overwrites the .env entries.
set -euo pipefail

ROOT=/opt/docker/crypto-trade
WALLET_FILE="$ROOT/.testnet_main_wallet.txt"
ENV_FILE="$ROOT/.env"

if [ ! -f "$WALLET_FILE" ]; then
  echo "ERROR: $WALLET_FILE not found" >&2
  exit 2
fi

MAIN_KEY=$(grep -E "^PRIVATE_KEY:" "$WALLET_FILE" | awk '{print $2}')
MAIN_ADDR=$(grep -E "^ADDRESS:" "$WALLET_FILE" | awk '{print $2}')
if [ -z "$MAIN_KEY" ] || [ -z "$MAIN_ADDR" ]; then
  echo "ERROR: could not parse wallet keys from $WALLET_FILE" >&2
  exit 2
fi

echo "Step 1/4: approving agent for main=$MAIN_ADDR on testnet ..."
APPROVE_OUTPUT=$(docker exec -e HL_MAIN_PRIVATE_KEY="$MAIN_KEY" -e HL_NETWORK=testnet \
  hliq-paper-bot python /app/scripts/approve_agent.py 2>&1)
echo "$APPROVE_OUTPUT"

AGENT_KEY=$(echo "$APPROVE_OUTPUT" | grep -E "^HL_AGENT_PRIVATE_KEY=" | head -1 | cut -d= -f2-)
if [ -z "$AGENT_KEY" ]; then
  echo "ERROR: approval did not return an agent key. Faucet may not have settled yet." >&2
  exit 1
fi

echo
echo "Step 2/4: writing live config to $ENV_FILE ..."
# Strip stale live entries, write fresh ones.
grep -vE "^(BOT_MODE|BOT_ALLOW_LIVE|HL_NETWORK|HL_AGENT_PRIVATE_KEY|HL_MAIN_WALLET_ADDRESS)=" "$ENV_FILE" > "${ENV_FILE}.tmp"
{
  echo "BOT_MODE=live"
  echo "BOT_ALLOW_LIVE=true"
  echo "HL_NETWORK=testnet"
  echo "HL_AGENT_PRIVATE_KEY=$AGENT_KEY"
  echo "HL_MAIN_WALLET_ADDRESS=$MAIN_ADDR"
} >> "${ENV_FILE}.tmp"
mv "${ENV_FILE}.tmp" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "Step 3/4: restarting container in live testnet mode ..."
docker compose -f "$ROOT/docker-compose.yml" down
docker compose -f "$ROOT/docker-compose.yml" up -d

echo "Step 4/4: waiting for first heartbeat ..."
until docker logs hliq-paper-bot 2>&1 | grep -q "Heartbeat"; do
  sleep 3
done

echo
echo "=========================================="
echo "LIVE TESTNET MODE ACTIVE"
echo "=========================================="
docker logs hliq-paper-bot 2>&1 | tail -10
