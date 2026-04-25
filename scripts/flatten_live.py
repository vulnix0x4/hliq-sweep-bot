#!/usr/bin/env python3
"""Emergency flatten: cancel all open orders and market-close all positions.

Usage:
  HL_AGENT_PRIVATE_KEY=0x... HL_MAIN_WALLET_ADDRESS=0x... HL_NETWORK=testnet \\
    python3 scripts/flatten_live.py

Operator-facing emergency tool. Bypasses the bot entirely. Use when:
- Bot is unresponsive or misbehaving
- You want to immediately exit all live exposure
- After a manual position you want to close from a known-clean state

The agent wallet is sufficient (cannot withdraw, only trade). Main wallet
key is NOT needed.

Exit codes:
  0 = success (everything cancelled and closed)
  1 = HL returned non-ok on at least one operation
  2 = config error (missing env vars)
"""
import os
import sys

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants


def main() -> int:
    key = os.getenv("HL_AGENT_PRIVATE_KEY", "").strip()
    main_addr = os.getenv("HL_MAIN_WALLET_ADDRESS", "").strip()
    network = os.getenv("HL_NETWORK", "testnet").strip().lower()
    if not key:
        print("ERROR: set HL_AGENT_PRIVATE_KEY in env", file=sys.stderr)
        return 2
    if not main_addr:
        print("ERROR: set HL_MAIN_WALLET_ADDRESS in env", file=sys.stderr)
        return 2
    if network not in ("testnet", "mainnet"):
        print(f"ERROR: HL_NETWORK must be testnet|mainnet, got {network!r}", file=sys.stderr)
        return 2
    api_url = constants.MAINNET_API_URL if network == "mainnet" else constants.TESTNET_API_URL

    wallet = eth_account.Account.from_key(key)
    exchange = Exchange(wallet, api_url, account_address=main_addr)
    info = Info(api_url, skip_ws=True)

    print(f"Flatten on {network} for {main_addr} (agent {wallet.address})", file=sys.stderr)
    failed = False

    # 1. Cancel all resting orders
    try:
        open_orders = info.open_orders(main_addr)
        if open_orders:
            cancels = [{"coin": o["coin"], "oid": o["oid"]} for o in open_orders if o.get("oid")]
            if cancels:
                result = exchange.bulk_cancel(cancels)
                print(f"Cancelled {len(cancels)} orders: {result}", file=sys.stderr)
                if isinstance(result, dict) and result.get("status") != "ok":
                    failed = True
        else:
            print("No open orders", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR cancelling orders: {exc}", file=sys.stderr)
        failed = True

    # 2. Market-close every nonzero position
    try:
        state = info.user_state(main_addr)
        positions = state.get("assetPositions", []) if isinstance(state, dict) else []
        any_position = False
        for ap in positions:
            pos = ap.get("position", {})
            coin = pos.get("coin")
            szi_raw = pos.get("szi")
            szi = float(szi_raw.get("base", 0)) if isinstance(szi_raw, dict) else float(szi_raw or 0)
            if abs(szi) < 1e-9 or not coin:
                continue
            any_position = True
            try:
                result = exchange.market_close(coin=coin, sz=None, slippage=0.01)
                print(f"Closed {coin} szi={szi}: {result}", file=sys.stderr)
                if not (isinstance(result, dict) and result.get("status") == "ok"):
                    failed = True
            except Exception as exc:
                print(f"ERROR closing {coin}: {exc}", file=sys.stderr)
                failed = True
        if not any_position:
            print("No open positions", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR fetching positions: {exc}", file=sys.stderr)
        failed = True

    if failed:
        print("FLATTEN COMPLETED WITH ERRORS — verify manually on HL UI", file=sys.stderr)
        return 1
    print("FLATTEN COMPLETE", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
