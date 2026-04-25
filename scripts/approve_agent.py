#!/usr/bin/env python3
"""One-time: approve a fresh agent wallet for the configured network.

Usage:
  HL_MAIN_PRIVATE_KEY=0x... HL_NETWORK=testnet python3 scripts/approve_agent.py

The MAIN private key is read from env, used ONCE to sign approve_agent, and
discarded. The script prints the new agent key + main address. Paste both
into .env (HL_AGENT_PRIVATE_KEY and HL_MAIN_WALLET_ADDRESS).

Agent permissions: trade only. Cannot withdraw. If the agent key leaks, the
attacker can lose your trading capital but cannot drain your wallet.
"""
import os
import sys

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants


def main() -> int:
    main_key = os.getenv("HL_MAIN_PRIVATE_KEY", "").strip()
    if not main_key:
        print("ERROR: set HL_MAIN_PRIVATE_KEY in env (read once, not stored)", file=sys.stderr)
        return 2
    network = os.getenv("HL_NETWORK", "testnet").strip().lower()
    if network not in ("testnet", "mainnet"):
        print(f"ERROR: HL_NETWORK must be testnet|mainnet, got {network!r}", file=sys.stderr)
        return 2
    api_url = constants.MAINNET_API_URL if network == "mainnet" else constants.TESTNET_API_URL

    main_wallet = eth_account.Account.from_key(main_key)
    main_exchange = Exchange(main_wallet, api_url)

    print(f"Approving agent on {network} for main address {main_wallet.address} ...", file=sys.stderr)
    approve_result, agent_key = main_exchange.approve_agent()
    if approve_result.get("status") != "ok":
        print(f"FAILED: {approve_result}", file=sys.stderr)
        return 1

    agent_wallet = eth_account.Account.from_key(agent_key)
    print()
    print("=" * 60)
    print("AGENT APPROVED — paste these into .env:")
    print("=" * 60)
    print(f"HL_NETWORK={network}")
    print(f"HL_MAIN_WALLET_ADDRESS={main_wallet.address}")
    print(f"HL_AGENT_PRIVATE_KEY={agent_key}")
    print(f"# (agent address: {agent_wallet.address}; trade-only, no withdraw)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
