# Hyperliquid Live Execution Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `PaperOrderManager` with a real `HyperliquidOrderManager` that submits signed orders to Hyperliquid (testnet first, then mainnet at 1/10th sizing), without changing any of the strategy code or risk gates.

**Architecture:** Mirror the `PaperOrderManager` interface exactly so `bot.py` doesn't need to know which executor is active. Adapter selection is `.env`-driven (`BOT_MODE=paper|live`). Live mode uses the official `hyperliquid-python-sdk`, signs actions via an **agent wallet** (not the main wallet — limits blast radius), and uses `schedule_cancel` as a deadman switch.

**Tech Stack:**
- Python 3.13 (existing)
- `hyperliquid-python-sdk` (PyPI, MIT, official; pulls in `eth_account` + `requests` + `websocket-client`)
- Existing test infra (`pytest`, `PYTHONPATH=src`)
- Docker compose (existing, `.env`-driven)

**Safety stance:**
- Paper mode is the default. Going live is `BOT_MODE=live` + `BOT_ALLOW_LIVE=true` (two env flags must both flip).
- Phase A is testnet only. Mainnet is gated by Phase G.
- All orders carry a Cloid keyed by our internal `signal_id` so we can reconcile state on restart.
- Deadman switch: `schedule_cancel` is reset every 30s; if the bot dies, all open orders are auto-cancelled within 60s.

**Concurrency note:** the HL SDK is synchronous. `HyperliquidOrderManager.on_trade()` is called from the bot's event loop. Each SDK call (typically <100ms) briefly blocks the loop. Acceptable for v1 — the bot processes ~1 signal/min on average. v2 may wrap SDK calls in `asyncio.to_thread()` if measured latency becomes a problem.

**SDK verified (Apr 25, 2026):**
- pip name: `hyperliquid-python-sdk`, latest 0.23.0, MIT, actively maintained
- `Alo` is the post-only TIF flag (verified against HL API docs)
- `info.user_state` returns `assetPositions` with `szi` field that may be either a JSON string (signed) or a dict — handle both
- `info.open_orders` and `info.query_order_by_oid` available for state queries
- `exchange.schedule_cancel(timestamp_ms)` requires timestamp ≥5s in the future

---

## Phase 0 — Research lock-in & dependency setup

### Task 0.1: Install hyperliquid-python-sdk into the Docker image

**Files:**
- Modify: `pyproject.toml`
- Modify: `Dockerfile` (verify pip install picks it up)

**Step 1: Add dep to pyproject.toml**

In `pyproject.toml`, change:
```toml
dependencies = [
  "websockets>=12.0",
]
```
to:
```toml
dependencies = [
  "websockets>=12.0",
  "hyperliquid-python-sdk>=0.20.0",
]
```

**Step 2: Rebuild the Docker image**

Run: `docker compose build`
Expected: build completes, no errors. Layer for `pip install --no-cache-dir .` pulls in `hyperliquid-python-sdk` and its transitive deps (`eth_account`, `requests`, `websocket-client`, `msgpack`).

**Step 3: Verify import works inside the container**

Run: `docker compose run --rm bot python -c "from hyperliquid.exchange import Exchange; from hyperliquid.info import Info; from hyperliquid.utils import constants; print('OK', constants.TESTNET_API_URL)"`
Expected: `OK https://api.hyperliquid-testnet.xyz`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add hyperliquid-python-sdk dependency"
```

---

## Phase A — Foundation: config + skeleton adapter (testnet only, no real risk)

### Task A.1: Add live-mode config knobs

**Files:**
- Modify: `src/hliq_bot/config.py:35-53` (FeedConfig) and `162-200` (RuntimeConfig)
- Modify: `.env.example`

**Step 1: Write failing test**

Create test in `tests/test_config_live.py`:
```python
import os
from hliq_bot.config import load_config


def test_live_mode_defaults_paper(monkeypatch):
    """Default config must be paper. Live requires explicit opt-in."""
    monkeypatch.delenv("BOT_MODE", raising=False)
    monkeypatch.delenv("BOT_ALLOW_LIVE", raising=False)
    cfg = load_config()
    assert cfg.mode == "paper"
    assert cfg.live.allow_live is False
    assert cfg.live.network == "testnet"
    assert cfg.live.agent_private_key == ""
    assert cfg.live.main_wallet_address == ""


def test_live_mode_loads_from_env(monkeypatch):
    monkeypatch.setenv("BOT_MODE", "live")
    monkeypatch.setenv("BOT_ALLOW_LIVE", "true")
    monkeypatch.setenv("HL_NETWORK", "mainnet")
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", "0x" + "a" * 64)
    monkeypatch.setenv("HL_MAIN_WALLET_ADDRESS", "0x" + "b" * 40)
    cfg = load_config()
    assert cfg.mode == "live"
    assert cfg.live.allow_live is True
    assert cfg.live.network == "mainnet"
    assert cfg.live.agent_private_key.startswith("0x")
    assert cfg.live.main_wallet_address.startswith("0x")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_config_live.py -v`
Expected: FAIL — `cfg.live` doesn't exist.

**Step 3: Add `LiveConfig` dataclass and load it**

In `src/hliq_bot/config.py`, after the `RuntimeConfig` block (~line 200), add:
```python
@dataclass(slots=True)
class LiveConfig:
    """Live (real-money) execution settings. Default safe = OFF."""
    allow_live: bool = False  # MUST be explicitly set to true to allow live
    network: str = "testnet"  # "testnet" or "mainnet"
    agent_private_key: str = ""  # private key of the agent wallet (NOT main)
    main_wallet_address: str = ""  # main wallet that approved the agent
    max_notional_per_trade: float = 200.0  # hard cap; refuse orders above this
    deadman_cancel_sec: int = 60  # auto-cancel-all if bot doesn't refresh
    deadman_refresh_sec: int = 30  # how often to push the cancel timer forward
```

In `AppConfig` (~line 200), add `live: LiveConfig` after `levels`.

In `load_config()` (around the bottom), add before the final `return AppConfig(...)`:
```python
live = LiveConfig(
    allow_live=_env_bool("BOT_ALLOW_LIVE", False),
    network=_env_str("HL_NETWORK", "testnet").lower(),
    agent_private_key=_env_str("HL_AGENT_PRIVATE_KEY", ""),
    main_wallet_address=_env_str("HL_MAIN_WALLET_ADDRESS", ""),
    max_notional_per_trade=_env_float("HL_MAX_NOTIONAL_PER_TRADE", 200.0),
    deadman_cancel_sec=_env_int("HL_DEADMAN_CANCEL_SEC", 60),
    deadman_refresh_sec=_env_int("HL_DEADMAN_REFRESH_SEC", 30),
)
```
Then add `live=live` to the `AppConfig(...)` return.

**Step 4: Run tests to verify pass**

Run: `PYTHONPATH=src pytest tests/test_config_live.py -v`
Expected: PASS, both tests.

Run: `PYTHONPATH=src pytest`
Expected: 87 passed (was 85, +2 new).

**Step 5: Update `.env.example`**

Append:
```
# === Live trading (HL real money) — DO NOT enable without testnet validation ===
BOT_ALLOW_LIVE=false
HL_NETWORK=testnet
HL_AGENT_PRIVATE_KEY=
HL_MAIN_WALLET_ADDRESS=
HL_MAX_NOTIONAL_PER_TRADE=200
HL_DEADMAN_CANCEL_SEC=60
HL_DEADMAN_REFRESH_SEC=30
```

**Step 6: Commit**

```bash
git add src/hliq_bot/config.py tests/test_config_live.py .env.example
git commit -m "feat(config): add LiveConfig (paper-default, requires BOT_ALLOW_LIVE)"
```

---

### Task A.2: Build agent-wallet bootstrap script

**Files:**
- Create: `scripts/approve_agent.py`

**Purpose:** One-time helper. The user runs this manually with their MAIN wallet's private key in an env var. It calls `approve_agent()` to mint a new agent key, prints the agent key to stdout, and the user pastes it into `.env` as `HL_AGENT_PRIVATE_KEY`. The main key NEVER touches the bot.

**Step 1: Write the script**

```python
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
```

**Step 2: Make executable**

Run: `chmod +x scripts/approve_agent.py`

**Step 3: Smoke test the script with a fake key (verify it errors cleanly)**

Run: `HL_MAIN_PRIVATE_KEY="" python3 scripts/approve_agent.py; echo "exit=$?"`
Expected: `ERROR: set HL_MAIN_PRIVATE_KEY ... exit=2`

**Step 4: Commit**

```bash
git add scripts/approve_agent.py
git commit -m "feat(live): add agent-wallet approval helper script"
```

---

### Task A.3: Skeleton `HyperliquidOrderManager` mirroring PaperOrderManager interface

**Files:**
- Create: `src/hliq_bot/execution/hyperliquid_order_manager.py`
- Create: `tests/test_hyperliquid_order_manager.py`

**Step 1: Write failing test for interface parity**

```python
# tests/test_hyperliquid_order_manager.py
from __future__ import annotations

import pytest

from hliq_bot.config import LiveConfig, StrategyConfig
from hliq_bot.execution.hyperliquid_order_manager import HyperliquidOrderManager
from hliq_bot.models import Side, SweepSignal, TradeEvent


def _signal() -> SweepSignal:
    return SweepSignal(
        side=Side.LONG, level=100.0, level_label="prior_15m_low",
        sweep_extreme=99.4, entry_price=100.0, stop_price=99.0,
        tp1_price=102.0, tp2_price=104.0, confidence=0.9,
        reason="test", created_ms=1_000,
    )


def test_constructs_with_live_config(tmp_path):
    """Adapter constructs without touching the network when allow_live=False."""
    live_cfg = LiveConfig(allow_live=False)
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    assert mgr.has_exposure() is False
    assert mgr.pending_entry is None
    assert mgr.position is None


def test_refuses_to_submit_when_allow_live_false(tmp_path):
    """submit_entry must reject when allow_live=False (safety guard)."""
    live_cfg = LiveConfig(allow_live=False)
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    with pytest.raises(RuntimeError, match="allow_live"):
        mgr.submit_entry(_signal(), signal_id="abc", qty=1.0, risk_dollars=1.0)


def test_refuses_to_submit_when_no_agent_key():
    """Even with allow_live=true, missing agent key must abort."""
    live_cfg = LiveConfig(allow_live=True, agent_private_key="")
    with pytest.raises(RuntimeError, match="agent_private_key"):
        HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")


def test_refuses_when_notional_exceeds_cap():
    """Hard notional cap must reject oversized orders even live."""
    # Use a fake but valid-format key for instantiation; SDK won't be called yet
    live_cfg = LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=50.0,  # very low cap
    )
    mgr = HyperliquidOrderManager(StrategyConfig(), live_cfg, coin="BTC")
    sig = _signal()  # qty=1.0, entry=100.0 -> notional=$100
    with pytest.raises(RuntimeError, match="max_notional"):
        mgr.submit_entry(sig, signal_id="abc", qty=1.0, risk_dollars=1.0)
```

**Step 2: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Write the skeleton**

```python
# src/hliq_bot/execution/hyperliquid_order_manager.py
from __future__ import annotations

import logging
from typing import Any

from hliq_bot.config import LiveConfig, StrategyConfig
from hliq_bot.models import (
    ClosedTrade,
    ExecEventType,
    ExecutionUpdate,
    OpenPosition,
    PendingEntry,
    Side,
    SweepSignal,
    TradeEvent,
)

log = logging.getLogger(__name__)

_TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
_MAINNET_URL = "https://api.hyperliquid.xyz"


class HyperliquidOrderManager:
    """Live-execution adapter mirroring PaperOrderManager's public surface.

    Safety:
      - Refuses any operation if cfg.allow_live is False.
      - Refuses construction if agent_private_key is empty.
      - Refuses any order whose notional exceeds cfg.max_notional_per_trade.

    The SDK clients (Exchange, Info) are constructed lazily on first use to
    keep import-time and test-time clean (no network calls in __init__).
    """

    def __init__(
        self,
        strategy_cfg: StrategyConfig,
        live_cfg: LiveConfig,
        coin: str,
    ) -> None:
        if live_cfg.allow_live and not live_cfg.agent_private_key:
            raise RuntimeError(
                "LiveConfig.agent_private_key must be set when allow_live=True"
            )
        self.cfg = strategy_cfg
        self.live_cfg = live_cfg
        self.coin = coin
        self.pending_entry: PendingEntry | None = None
        self.position: OpenPosition | None = None
        self._last_trade_ms: int = 0
        # SDK clients — constructed on first network operation.
        self._exchange: Any = None
        self._info: Any = None

    # ---- Public surface (mirrors PaperOrderManager) ----

    def has_exposure(self) -> bool:
        return self.pending_entry is not None or self.position is not None

    def submit_entry(
        self,
        signal: SweepSignal,
        signal_id: str,
        qty: float,
        risk_dollars: float,
    ) -> ExecutionUpdate:
        self._guard_live()
        notional = signal.entry_price * qty
        if notional > self.live_cfg.max_notional_per_trade:
            raise RuntimeError(
                f"order notional ${notional:.2f} exceeds "
                f"max_notional_per_trade=${self.live_cfg.max_notional_per_trade:.2f}"
            )
        # Phase B will fill this in — for now we just record a stub.
        raise NotImplementedError("submit_entry implementation pending Phase B")

    def on_trade(self, trade: TradeEvent) -> list[ExecutionUpdate]:
        # Phase B will fill this in — for now no-op.
        return []

    # ---- Guards ----

    def _guard_live(self) -> None:
        if not self.live_cfg.allow_live:
            raise RuntimeError(
                "HyperliquidOrderManager called with allow_live=False; refusing to send."
            )

    @property
    def api_url(self) -> str:
        return _MAINNET_URL if self.live_cfg.network == "mainnet" else _TESTNET_URL
```

**Step 4: Run tests to verify pass**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py -v`
Expected: 4 passed.

Run: `PYTHONPATH=src pytest`
Expected: 91 passed (87 + 4 new).

**Step 5: Commit**

```bash
git add src/hliq_bot/execution/hyperliquid_order_manager.py tests/test_hyperliquid_order_manager.py
git commit -m "feat(live): skeleton HyperliquidOrderManager with safety guards"
```

---

### Task A.4: Wire executor selection by `BOT_MODE` in `bot.py`

**Files:**
- Modify: `src/hliq_bot/bot.py:19, 37, 92` (executor type + instantiation)

**Step 1: Write failing test**

Add to `tests/test_bot_runtime.py`:
```python
def test_bot_uses_paper_executor_when_mode_paper(tmp_path):
    cfg = _app_config(tmp_path)
    cfg.mode = "paper"
    bot = SweepBot(cfg)
    from hliq_bot.execution.order_manager import PaperOrderManager
    for w in bot._workers.values():
        assert isinstance(w.executor, PaperOrderManager)


def test_bot_uses_hyperliquid_executor_when_mode_live(tmp_path):
    cfg = _app_config(tmp_path)
    cfg.mode = "live"
    cfg.live.allow_live = True
    cfg.live.agent_private_key = "0x" + "a" * 64
    cfg.live.main_wallet_address = "0x" + "b" * 40
    bot = SweepBot(cfg)
    from hliq_bot.execution.hyperliquid_order_manager import HyperliquidOrderManager
    for w in bot._workers.values():
        assert isinstance(w.executor, HyperliquidOrderManager)
```

**Step 2: Run, expect failure**

Run: `PYTHONPATH=src pytest tests/test_bot_runtime.py::test_bot_uses_hyperliquid_executor_when_mode_live -v`
Expected: FAIL — bot.py always constructs PaperOrderManager.

**Step 3: Update `bot.py`**

In `bot.py`, around line 92 where `PaperOrderManager` is constructed, change:
```python
executor=PaperOrderManager(config.strategy),
```
to:
```python
executor=_make_executor(config, coin=coin),
```

Add a helper near the top (after imports):
```python
def _make_executor(config: AppConfig, coin: str):
    """Pick paper or live executor based on cfg.mode."""
    if config.mode == "live":
        from hliq_bot.execution.hyperliquid_order_manager import HyperliquidOrderManager
        return HyperliquidOrderManager(config.strategy, config.live, coin=coin)
    return PaperOrderManager(config.strategy)
```

Also update the `executor:` field type in the worker dataclass to `PaperOrderManager | HyperliquidOrderManager` (or simply the common supertype if defined; otherwise `Any`).

**Step 4: Run tests**

Run: `PYTHONPATH=src pytest`
Expected: 93 passed (91 + 2 new).

**Step 5: Commit**

```bash
git add src/hliq_bot/bot.py tests/test_bot_runtime.py
git commit -m "feat(live): select executor by BOT_MODE (paper|live)"
```

---

## Phase B — Order primitives (testnet only)

### Task B.1: Implement `submit_entry` (post-only limit at retest level)

**Files:**
- Modify: `src/hliq_bot/execution/hyperliquid_order_manager.py`
- Modify: `tests/test_hyperliquid_order_manager.py`

**Step 1: Failing test using a mocked Exchange**

```python
# in tests/test_hyperliquid_order_manager.py
from unittest.mock import MagicMock


def _live_cfg():
    return LiveConfig(
        allow_live=True,
        agent_private_key="0x" + "a" * 64,
        main_wallet_address="0x" + "b" * 40,
        max_notional_per_trade=10000.0,
    )


def test_submit_entry_places_post_only_limit(monkeypatch):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    mgr._exchange = fake_exchange  # inject mock

    sig = _signal()
    update = mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)

    assert update.event_type == ExecEventType.ENTRY_PLACED
    assert update.signal_id == "abc"
    fake_exchange.order.assert_called_once()
    args, kwargs = fake_exchange.order.call_args
    # Verify the kwargs are correct
    assert kwargs.get("name") == "BTC" or args[0] == "BTC"
    # is_buy must reflect side
    assert kwargs.get("is_buy") is True or args[1] is True
    # Post-only via Alo
    order_type = kwargs.get("order_type") or args[4]
    assert order_type == {"limit": {"tif": "Alo"}}
    # Pending entry was recorded
    assert mgr.pending_entry is not None
    assert mgr.pending_entry.qty == 0.001
```

**Step 2: Run, expect failure**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py::test_submit_entry_places_post_only_limit -v`
Expected: FAIL with NotImplementedError.

**Step 3: Implement `submit_entry`**

In `hyperliquid_order_manager.py`, replace the `submit_entry` body:
```python
def submit_entry(
    self,
    signal: SweepSignal,
    signal_id: str,
    qty: float,
    risk_dollars: float,
) -> ExecutionUpdate:
    self._guard_live()
    notional = signal.entry_price * qty
    if notional > self.live_cfg.max_notional_per_trade:
        raise RuntimeError(
            f"order notional ${notional:.2f} exceeds "
            f"max_notional_per_trade=${self.live_cfg.max_notional_per_trade:.2f}"
        )

    exchange = self._ensure_exchange()
    is_buy = signal.side == Side.LONG
    # Alo = "Add Liquidity Only" = post-only. Refused if it would cross the spread.
    result = exchange.order(
        name=self.coin,
        is_buy=is_buy,
        sz=qty,
        limit_px=signal.entry_price,
        order_type={"limit": {"tif": "Alo"}},
        reduce_only=False,
    )
    if result.get("status") != "ok":
        raise RuntimeError(f"HL order failed: {result}")

    statuses = result["response"]["data"]["statuses"]
    status = statuses[0] if statuses else {}
    if "error" in status:
        raise RuntimeError(f"HL order rejected: {status['error']}")

    oid = None
    if "resting" in status:
        oid = status["resting"].get("oid")
    elif "filled" in status:
        # Edge case: post-only filled at limit. Shouldn't happen with Alo but handle it.
        oid = status["filled"].get("oid")

    self.pending_entry = PendingEntry(
        signal_id=signal_id,
        side=signal.side,
        qty=qty,
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
        tp1_price=signal.tp1_price,
        tp2_price=signal.tp2_price,
        created_ms=signal.created_ms,
        expiry_sec=self.cfg.pending_entry_expiry_sec,
        level_label=signal.level_label,
        risk_dollars=risk_dollars,
        coin=self.coin,
    )
    # Stash the HL order id on the pending entry's signal_id mapping.
    self._pending_oid = oid

    return ExecutionUpdate(
        ts_ms=signal.created_ms,
        event_type=ExecEventType.ENTRY_PLACED,
        message=f"hl entry placed: {signal.side.value} qty={qty:.6f} @ {signal.entry_price:.2f} oid={oid}",
        signal_id=signal_id,
    )

def _ensure_exchange(self):
    if self._exchange is not None:
        return self._exchange
    import eth_account
    from hyperliquid.exchange import Exchange
    wallet = eth_account.Account.from_key(self.live_cfg.agent_private_key)
    self._exchange = Exchange(
        wallet,
        self.api_url,
        account_address=self.live_cfg.main_wallet_address or None,
    )
    return self._exchange
```

Also add `self._pending_oid: int | None = None` to `__init__`.

**Step 4: Run test, verify pass**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py -v`
Expected: all pass (5 tests).

Run: `PYTHONPATH=src pytest`
Expected: 94 passed (was 93, +1).

**Step 5: Commit**

```bash
git add src/hliq_bot/execution/hyperliquid_order_manager.py tests/test_hyperliquid_order_manager.py
git commit -m "feat(live): submit_entry posts Alo (post-only) limit via HL SDK"
```

---

### Task B.2: Implement entry expiry & cancel

**Files:**
- Modify: `src/hliq_bot/execution/hyperliquid_order_manager.py`
- Modify: `tests/test_hyperliquid_order_manager.py`

**Step 1: Failing test**

```python
def test_pending_entry_cancels_on_expiry(monkeypatch):
    mgr = HyperliquidOrderManager(StrategyConfig(pending_entry_expiry_sec=120), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_exchange.cancel.return_value = {"status": "ok"}
    mgr._exchange = fake_exchange

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)
    assert mgr.pending_entry is not None

    # Trade event well past expiry — should trigger cancel
    expired_ms = sig.created_ms + (200 * 1000)
    updates = mgr.on_trade(TradeEvent(ts_ms=expired_ms, price=100.5, size=1.0))

    fake_exchange.cancel.assert_called_once_with("BTC", 12345)
    assert mgr.pending_entry is None
    assert any(u.event_type == ExecEventType.ORDER_CANCELED for u in updates)
```

**Step 2: Run, fail**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py::test_pending_entry_cancels_on_expiry -v`
Expected: FAIL — `on_trade` returns `[]`.

**Step 3: Implement expiry path in `on_trade`**

```python
def on_trade(self, trade: TradeEvent) -> list[ExecutionUpdate]:
    self._last_trade_ms = trade.ts_ms
    updates: list[ExecutionUpdate] = []
    updates.extend(self._maybe_expire_pending(trade.ts_ms))
    # Phase B.3 will add fill detection
    # Phase C will add position management
    return updates

def _maybe_expire_pending(self, now_ms: int) -> list[ExecutionUpdate]:
    if self.pending_entry is None:
        return []
    age_sec = (now_ms - self.pending_entry.created_ms) / 1000.0
    if age_sec < self.pending_entry.expiry_sec:
        return []
    pe = self.pending_entry
    if self._pending_oid is not None:
        try:
            self._ensure_exchange().cancel(self.coin, self._pending_oid)
        except Exception as exc:
            log.warning("HL cancel failed for oid=%s: %s", self._pending_oid, exc)
    msg = (
        f"hl pending entry expired after {pe.expiry_sec}s: "
        f"{pe.side.value} @ {pe.entry_price:.2f} oid={self._pending_oid}"
    )
    sid = pe.signal_id
    self.pending_entry = None
    self._pending_oid = None
    return [
        ExecutionUpdate(
            ts_ms=now_ms,
            event_type=ExecEventType.ORDER_CANCELED,
            message=msg,
            signal_id=sid,
        )
    ]
```

**Step 4: Run, pass**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py -v`
Expected: all 6 pass.

**Step 5: Commit**

```bash
git add src/hliq_bot/execution/hyperliquid_order_manager.py tests/test_hyperliquid_order_manager.py
git commit -m "feat(live): expire & cancel pending HL entry orders"
```

---

### Task B.3: Implement fill detection via `info.user_state` polling

**Note:** v1 polls `info.user_state` once per `on_trade` (cheap REST call). v2 will subscribe to `userFills` WS for true async fills. Polling is simpler and correct.

**Files:**
- Modify: `src/hliq_bot/execution/hyperliquid_order_manager.py`
- Modify: `tests/test_hyperliquid_order_manager.py`

**Step 1: Failing test**

```python
def test_pending_entry_transitions_to_position_on_fill(monkeypatch):
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.order.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}},
    }
    fake_info = MagicMock()
    # First call: empty positions. Second: BTC long open at 100.0 size 0.001.
    fake_info.user_state.side_effect = [
        {"assetPositions": []},
        {"assetPositions": [{"position": {"coin": "BTC", "szi": "0.001", "entryPx": "100.0"}}]},
    ]
    mgr._exchange = fake_exchange
    mgr._info = fake_info

    sig = _signal()
    mgr.submit_entry(sig, signal_id="abc", qty=0.001, risk_dollars=1.0)

    # First tick: not yet filled
    out1 = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 500, price=100.0, size=1.0))
    assert mgr.position is None
    assert mgr.pending_entry is not None

    # Second tick: HL reports the position now exists
    out2 = mgr.on_trade(TradeEvent(ts_ms=sig.created_ms + 1500, price=100.0, size=1.0))
    assert mgr.position is not None
    assert mgr.position.entry_price == 100.0
    assert mgr.position.qty_initial == 0.001
    assert mgr.pending_entry is None
    assert any(u.event_type == ExecEventType.ENTRY_FILLED for u in out2)
```

**Step 2: Run, fail**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py::test_pending_entry_transitions_to_position_on_fill -v`
Expected: FAIL.

**Step 3: Implement fill detection**

Add to `on_trade` after expiry check:
```python
updates.extend(self._maybe_detect_fill(trade.ts_ms))
```

Add method:
```python
def _maybe_detect_fill(self, now_ms: int) -> list[ExecutionUpdate]:
    if self.pending_entry is None or self.position is not None:
        return []
    info = self._ensure_info()
    address = self.live_cfg.main_wallet_address or self._agent_address()
    state = info.user_state(address)
    target_qty = self.pending_entry.qty
    for ap in state.get("assetPositions", []):
        pos = ap.get("position", {})
        if str(pos.get("coin", "")).upper() != self.coin.upper():
            continue
        szi_raw = pos.get("szi")
        # szi may be a string (signed) or a dict {"base": "...", ...}
        if isinstance(szi_raw, dict):
            szi = float(szi_raw.get("base", 0))
        else:
            szi = float(szi_raw or 0)
        if abs(szi) < target_qty * 0.5:
            continue  # not yet (partially) filled to a meaningful degree
        entry_px = float(pos.get("entryPx", self.pending_entry.entry_price))
        pe = self.pending_entry
        self.position = OpenPosition(
            signal_id=pe.signal_id,
            side=pe.side,
            entry_price=entry_px,
            stop_price=pe.stop_price,
            tp1_price=pe.tp1_price,
            tp2_price=pe.tp2_price,
            opened_ms=now_ms,
            qty_initial=abs(szi),
            qty_remaining=abs(szi),
            risk_dollars=pe.risk_dollars,
            coin=self.coin,
            best_price=entry_px,
            worst_price=entry_px,
        )
        self.pending_entry = None
        self._pending_oid = None
        return [ExecutionUpdate(
            ts_ms=now_ms,
            event_type=ExecEventType.ENTRY_FILLED,
            message=f"hl entry filled: {pe.side.value} qty={szi:.6f} @ {entry_px:.2f}",
            signal_id=pe.signal_id,
        )]
    return []

def _ensure_info(self):
    if self._info is not None:
        return self._info
    from hyperliquid.info import Info
    self._info = Info(self.api_url, skip_ws=True)
    return self._info

def _agent_address(self) -> str:
    import eth_account
    return eth_account.Account.from_key(self.live_cfg.agent_private_key).address
```

**Step 4: Run, pass**

Run: `PYTHONPATH=src pytest tests/test_hyperliquid_order_manager.py -v`
Expected: 7 pass.

**Step 5: Commit**

```bash
git add src/hliq_bot/execution/hyperliquid_order_manager.py tests/test_hyperliquid_order_manager.py
git commit -m "feat(live): detect fills by polling info.user_state"
```

---

## Phase C — Position management (testnet only)

### Task C.1: Stop-loss exit via `market_close`

**Files:**
- Modify: `src/hliq_bot/execution/hyperliquid_order_manager.py`
- Modify: `tests/test_hyperliquid_order_manager.py`

The bot watches `trade.price` against `position.stop_price` (same as paper). When triggered, submits `market_close` (taker fee).

**Step 1: Failing test**

```python
def test_stop_loss_triggers_market_close():
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.market_close.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalSz": "0.001", "avgPx": "99.0"}}]}},
    }
    mgr._exchange = fake_exchange
    # Pre-seed an open long position
    from hliq_bot.models import OpenPosition
    mgr.position = OpenPosition(
        signal_id="abc", side=Side.LONG, entry_price=100.0,
        stop_price=99.0, tp1_price=102.0, tp2_price=104.0,
        opened_ms=1000, qty_initial=0.001, qty_remaining=0.001,
        risk_dollars=1.0, coin="BTC", best_price=100.0, worst_price=100.0,
    )

    # Price hits stop
    updates = mgr.on_trade(TradeEvent(ts_ms=2000, price=98.5, size=1.0))
    fake_exchange.market_close.assert_called_once()
    assert mgr.position is None
    assert any(u.event_type == ExecEventType.POSITION_CLOSED for u in updates)
    closed = [u for u in updates if u.closed_trade is not None][0].closed_trade
    assert closed.exit_reason == "stop_loss"
    assert closed.pnl_gross < 0
```

**Step 2: Run, fail**

**Step 3: Implement `_maybe_manage_open_position` with stop-loss only first**

Mirror `PaperOrderManager._maybe_manage_open_position` but submit live orders. Start with stop_loss only:
```python
def _maybe_manage_open_position(self, trade: TradeEvent) -> list[ExecutionUpdate]:
    if self.position is None:
        return []
    p = self.position
    self._update_excursions(trade.price)

    if p.side == Side.LONG:
        stop_hit = trade.price <= p.stop_price
    else:
        stop_hit = trade.price >= p.stop_price

    if stop_hit:
        return self._close_via_market(trade.ts_ms, "stop_loss")

    # Phases C.2-C.4: TP1/TP2/trail/time exits
    return []

def _close_via_market(self, ts_ms: int, reason: str) -> list[ExecutionUpdate]:
    p = self.position
    if p is None:
        return []
    exchange = self._ensure_exchange()
    result = exchange.market_close(
        coin=self.coin,
        sz=p.qty_remaining,
        slippage=0.005,  # 0.5% slippage tolerance
    )
    fill_px = p.entry_price  # fallback
    if result.get("status") == "ok":
        st = result["response"]["data"]["statuses"][0]
        if "filled" in st:
            fill_px = float(st["filled"].get("avgPx", p.entry_price))
    pnl_gross = (fill_px - p.entry_price) * p.qty_remaining if p.side == Side.LONG \
                else (p.entry_price - fill_px) * p.qty_remaining
    pnl_gross += p.realized_pnl
    # Fees: taker on the market_close
    final_fee = (fill_px * p.qty_remaining) * self.cfg.taker_fee_pct
    fees_paid = p.realized_fees + final_fee
    pnl_net = pnl_gross - fees_paid
    risk = max(p.risk_dollars, 1e-9)
    closed = ClosedTrade(
        signal_id=p.signal_id, side=p.side, entry_price=p.entry_price,
        exit_price=fill_px, qty=p.qty_initial, pnl=pnl_net, pnl_gross=pnl_gross,
        fees_paid=fees_paid, risk_dollars=risk, r_multiple=pnl_net / risk,
        opened_ms=p.opened_ms, closed_ms=ts_ms, exit_reason=reason,
        coin=self.coin,
        mfe_pnl=self._price_to_pnl(p.best_price), mae_pnl=self._price_to_pnl(p.worst_price),
    )
    self.position = None
    return [ExecutionUpdate(
        ts_ms=ts_ms, event_type=ExecEventType.POSITION_CLOSED,
        message=f"hl position closed ({reason}): pnl={pnl_net:.4f} fees={fees_paid:.4f}",
        signal_id=p.signal_id, closed_trade=closed,
    )]

def _update_excursions(self, price: float) -> None:
    p = self.position
    if p is None:
        return
    if p.side == Side.LONG:
        p.best_price = max(p.best_price or price, price)
        p.worst_price = min(p.worst_price or price, price) if p.worst_price > 0 else price
    else:
        p.best_price = min(p.best_price or price, price) if p.best_price > 0 else price
        p.worst_price = max(p.worst_price or price, price)

def _price_to_pnl(self, price: float) -> float:
    p = self.position
    if p is None:
        return 0.0
    if p.side == Side.LONG:
        return (price - p.entry_price) * p.qty_initial
    return (p.entry_price - price) * p.qty_initial
```

Wire into `on_trade`:
```python
updates.extend(self._maybe_manage_open_position(trade))
```

**Step 4: Run, pass**

**Step 5: Commit**

```bash
git add src/hliq_bot/execution/hyperliquid_order_manager.py tests/test_hyperliquid_order_manager.py
git commit -m "feat(live): stop_loss closes position via market_close (taker)"
```

### Task C.2: Add TP1 partial scale, TP2 full close, time/early/max exits, trailing stop

Mirror `PaperOrderManager._maybe_manage_open_position` exits 1-by-1 with the same logic, but submit live orders:
- **TP1 partial:** post-only limit at tp1 for `qty_remaining * 0.5`. On fill (detected via user_state polling next tick), update `realized_pnl`, `realized_fees`, set `tp1_filled=True`, move stop to BE.
- **TP2 close:** post-only limit at tp2 for `qty_remaining`. On fill, close position.
- **Trail / time_stop / early_exit / max_hold:** market_close at trigger.

This task is large; break into 4 sub-tasks (C.2.1 TP1, C.2.2 TP2, C.2.3 trail-from-entry/post-TP1 trail, C.2.4 time/early/max). Each sub-task: failing test → implementation → pass → commit.

**Pattern to follow for each sub-task:**

1. Copy the equivalent block from `PaperOrderManager`
2. Replace `pnl_fn(...)` arithmetic with `exchange.order(...)` or `exchange.market_close(...)` SDK calls
3. Reconcile state via `info.user_state` polling on the next `on_trade`
4. Test with mocked Exchange + Info

For time/early/max stops (taker exits), use `_close_via_market(ts_ms, reason)`.
For TP1/TP2 (maker exits), use a separate `_close_via_limit(ts_ms, price, qty, reason)` helper.

Example commit messages:
```
feat(live): TP1 partial scale via post-only limit
feat(live): TP2 full close via post-only limit
feat(live): trail-from-entry + post-TP1 trail (live)
feat(live): time_stop / early_exit / max_hold exits (live)
```

---

### Task C.3: Position state reconciliation on bot restart

**Files:**
- Modify: `src/hliq_bot/execution/hyperliquid_order_manager.py`

When the bot starts in live mode and finds a position already on HL (from a prior run that crashed), reconstruct `OpenPosition` from `info.user_state`. Without this, the bot would think no exposure exists and start placing duplicate entries.

**Step 1: Failing test**

```python
def test_reconciles_existing_position_on_startup():
    mgr = HyperliquidOrderManager(StrategyConfig(), _live_cfg(), coin="BTC")
    fake_info = MagicMock()
    fake_info.user_state.return_value = {
        "assetPositions": [{
            "position": {"coin": "BTC", "szi": "0.005", "entryPx": "100.0"},
        }],
    }
    mgr._info = fake_info
    mgr.reconcile_on_startup()
    assert mgr.position is not None
    assert mgr.position.qty_initial == 0.005
    assert mgr.position.entry_price == 100.0
    assert mgr.position.side == Side.LONG
```

**Step 2: Run, fail**

**Step 3: Add `reconcile_on_startup` method**

```python
def reconcile_on_startup(self) -> None:
    """If a position already exists on HL (e.g. after a crash), restore state.

    Does NOT recover stop/TP levels — those are lost. Best practice: manually
    flatten any open positions before restarting the bot.
    """
    if not self.live_cfg.allow_live:
        return
    info = self._ensure_info()
    address = self.live_cfg.main_wallet_address or self._agent_address()
    state = info.user_state(address)
    for ap in state.get("assetPositions", []):
        pos = ap.get("position", {})
        if str(pos.get("coin", "")).upper() != self.coin.upper():
            continue
        szi_raw = pos.get("szi")
        szi = float(szi_raw.get("base", 0)) if isinstance(szi_raw, dict) else float(szi_raw or 0)
        if szi == 0:
            continue
        entry_px = float(pos.get("entryPx", 0))
        side = Side.LONG if szi > 0 else Side.SHORT
        # Stop/TP levels lost — set conservative wide stop at -2% to avoid surprises
        wide_stop = entry_px * (0.98 if side == Side.LONG else 1.02)
        self.position = OpenPosition(
            signal_id="reconciled",
            side=side, entry_price=entry_px,
            stop_price=wide_stop,
            tp1_price=entry_px,  # disable TP1
            tp2_price=entry_px,  # disable TP2
            opened_ms=int(self._last_trade_ms or 0),
            qty_initial=abs(szi), qty_remaining=abs(szi),
            risk_dollars=abs(szi * entry_px) * 0.02,  # rough estimate
            coin=self.coin,
            best_price=entry_px, worst_price=entry_px,
        )
        log.warning(
            "Reconciled existing %s position: %s qty=%.6f entry=%.2f stop=%.2f (manual cleanup recommended)",
            self.coin, side.value, abs(szi), entry_px, wide_stop,
        )
```

Wire into `bot.py` startup: in `SweepBot.__init__` or first heartbeat, call `executor.reconcile_on_startup()` for each worker if mode=live.

**Step 4: Run, pass**

**Step 5: Commit**

```bash
git add src/hliq_bot/execution/hyperliquid_order_manager.py tests/test_hyperliquid_order_manager.py src/hliq_bot/bot.py
git commit -m "feat(live): reconcile open positions on bot startup"
```

---

## Phase D — Safety: deadman switch + reconciliation + manual flatten

### Task D.1: Deadman switch via `schedule_cancel`

Every 30s the bot pushes `schedule_cancel(now + 60s)` while it's running. If the bot dies, HL automatically cancels all the bot's resting orders within 60s.

**Files:**
- Modify: `src/hliq_bot/execution/hyperliquid_order_manager.py`
- Modify: `src/hliq_bot/bot.py` (heartbeat hook)

**Step 1: Failing test**

```python
def test_deadman_refresh_pushes_cancel_timer(monkeypatch):
    cfg = _live_cfg()
    cfg.deadman_cancel_sec = 60
    cfg.deadman_refresh_sec = 30
    mgr = HyperliquidOrderManager(StrategyConfig(), cfg, coin="BTC")
    fake_exchange = MagicMock()
    fake_exchange.schedule_cancel.return_value = {"status": "ok"}
    mgr._exchange = fake_exchange

    mgr.refresh_deadman(now_ms=1_000_000_000_000)
    fake_exchange.schedule_cancel.assert_called_once()
    arg = fake_exchange.schedule_cancel.call_args[0][0]
    assert arg == 1_000_000_000_000 + 60_000
```

**Step 2: Run, fail**

**Step 3: Implement**

```python
def refresh_deadman(self, now_ms: int) -> None:
    if not self.live_cfg.allow_live:
        return
    cancel_at = now_ms + self.live_cfg.deadman_cancel_sec * 1000
    try:
        self._ensure_exchange().schedule_cancel(cancel_at)
        self._last_deadman_refresh_ms = now_ms
    except Exception as exc:
        log.warning("Deadman refresh failed: %s", exc)

def should_refresh_deadman(self, now_ms: int) -> bool:
    if not self.live_cfg.allow_live:
        return False
    elapsed = (now_ms - getattr(self, "_last_deadman_refresh_ms", 0)) / 1000.0
    return elapsed >= self.live_cfg.deadman_refresh_sec
```

In `bot.py` heartbeat (every 60s already), call:
```python
for w in self._workers.values():
    if hasattr(w.executor, "should_refresh_deadman") and w.executor.should_refresh_deadman(now_ms):
        w.executor.refresh_deadman(now_ms)
```

**Step 4: Run, pass**

**Step 5: Commit**

```bash
git add src/hliq_bot/execution/hyperliquid_order_manager.py tests/test_hyperliquid_order_manager.py src/hliq_bot/bot.py
git commit -m "feat(live): deadman switch via HL schedule_cancel (60s ttl, 30s refresh)"
```

### Task D.2: Manual flatten script

**Files:**
- Create: `scripts/flatten_live.py`

Use case: emergency. User wants to close ALL positions and cancel ALL orders right now without going through the bot.

```python
#!/usr/bin/env python3
"""Emergency: cancel all open orders and market-close all positions."""
import os, sys
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

def main() -> int:
    key = os.getenv("HL_AGENT_PRIVATE_KEY", "").strip()
    main_addr = os.getenv("HL_MAIN_WALLET_ADDRESS", "").strip()
    network = os.getenv("HL_NETWORK", "testnet").strip().lower()
    if not key or not main_addr:
        print("ERROR: set HL_AGENT_PRIVATE_KEY and HL_MAIN_WALLET_ADDRESS", file=sys.stderr)
        return 2
    api_url = constants.MAINNET_API_URL if network == "mainnet" else constants.TESTNET_API_URL
    wallet = eth_account.Account.from_key(key)
    exchange = Exchange(wallet, api_url, account_address=main_addr)
    info = Info(api_url, skip_ws=True)

    # 1. Cancel all resting orders
    open_orders = info.open_orders(main_addr)
    if open_orders:
        cancels = [{"coin": o["coin"], "oid": o["oid"]} for o in open_orders]
        result = exchange.bulk_cancel(cancels)
        print(f"Cancelled {len(cancels)} orders: {result}")
    else:
        print("No open orders")

    # 2. Market close every nonzero position
    state = info.user_state(main_addr)
    for ap in state.get("assetPositions", []):
        pos = ap.get("position", {})
        coin = pos.get("coin")
        szi_raw = pos.get("szi")
        szi = float(szi_raw.get("base", 0)) if isinstance(szi_raw, dict) else float(szi_raw or 0)
        if abs(szi) < 1e-9:
            continue
        result = exchange.market_close(coin=coin, sz=None, slippage=0.01)
        print(f"Closed {coin} szi={szi}: {result}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

**Commit:**
```bash
chmod +x scripts/flatten_live.py
git add scripts/flatten_live.py
git commit -m "feat(live): emergency flatten script"
```

---

## Phase E — Testnet validation (no code changes — operational)

This phase is a 1-week run, not a code task. The plan documents the procedure.

### Task E.1: Bootstrap testnet account

1. Create main wallet: `python3 -c "import eth_account; w = eth_account.Account.create(); print('ADDR', w.address); print('KEY', w.key.hex())"`
2. Save the key in a password manager. **Never put it in `.env`.**
3. Get testnet USDC from https://app.hyperliquid-testnet.xyz/drip
4. Run `HL_MAIN_PRIVATE_KEY=<main_key> HL_NETWORK=testnet python3 scripts/approve_agent.py`
5. Paste output into `.env`:
   ```
   BOT_MODE=live
   BOT_ALLOW_LIVE=true
   HL_NETWORK=testnet
   HL_MAIN_WALLET_ADDRESS=0x...
   HL_AGENT_PRIVATE_KEY=0x...
   HL_MAX_NOTIONAL_PER_TRADE=200
   ```
6. Restart: `./scripts/botctl.sh down && ./scripts/botctl.sh up`
7. Verify in logs: `Connected to Hyperliquid WS` and `Live executor active (testnet)`

### Task E.2: 7-day testnet observation

Watch for:
- Fill rate (live) vs paper expectations (paper showed 60-80% fill rate)
- Slippage on entries (paper assumes ±2 bps tolerance; live may differ)
- Slippage on market_close exits
- Any HL errors (rate limiting, signing issues, network)
- Deadman switch fires during deliberate kill of bot
- Position reconciliation works after restart
- Fee math matches reality (compare HL fee receipts to our fees_paid)

### Task E.3: Compare testnet vs paper

After 7 days, replay the SAME captured data through paper and compare:
- Trade count: testnet should be within ±20% of paper
- Win rate: within ±10pp
- Net PnL: within ±30%

If testnet underperforms by >30%, **DO NOT proceed to mainnet**. Investigate the gap (likely fill rate or slippage). Tune `entry_touch_tolerance_bps` and re-validate.

---

## Phase F — Reduce paper-vs-live drift (only if Phase E surfaces issues)

Likely drift sources and fixes:

1. **Fill rate too low.** Paper assumes 60-80% fill. If testnet shows 30-40%, increase `BOT_ENTRY_TOUCH_TOL_BPS` from 2 → 4 (looser touch). Or: change entry from limit-at-level to limit-at-(level + 1bp) to be slightly more aggressive about getting filled.
2. **Slippage on market_close.** Increase `slippage` arg in `market_close` from 0.005 to 0.01.
3. **Latency-sensitive exits firing late.** Move stop_loss to a native HL trigger order (placed at entry time). v2 enhancement.

Each fix: replay-validate first, then deploy.

---

## Phase G — Mainnet rollout (gated)

### Task G.1: Pre-flight checklist

- [ ] Phase E passed (testnet matches paper ±30%)
- [ ] Phase F applied if needed
- [ ] Mainnet wallet funded with **$100 USDC max** (1/10th sizing)
- [ ] `scripts/approve_agent.py` run on mainnet — agent key in `.env`
- [ ] `BOT_ACCOUNT_EQUITY=100` (forces small risk per trade)
- [ ] `HL_NETWORK=mainnet`
- [ ] `HL_MAX_NOTIONAL_PER_TRADE=50` (hard cap; refuses larger orders)
- [ ] `BOT_ALLOW_LIVE=true`
- [ ] Backup of working `.env` before any change
- [ ] Monitoring set up: alert on errors in `bot.log`, alert on >5 consecutive losses

### Task G.2: Go-live + 100-trade observation

1. `./scripts/botctl.sh down && ./scripts/botctl.sh up`
2. Watch first trade end-to-end. Verify entry fill, exit fill, fee accounting.
3. After 100 closed trades, compare to paper expectations on same window.
4. **Scale-up criteria:** net PnL within 30% of paper, win rate within 10pp, no execution errors.

If criteria met: increase `BOT_ACCOUNT_EQUITY` and `HL_MAX_NOTIONAL_PER_TRADE` gradually (2× steps, 100 trades between increases). If not: revert to paper, investigate.

---

## Out of scope (explicit deferrals)

- Native HL TP/SL trigger orders (we use code-watched + market_close instead). v2.
- WebSocket `userFills` subscription (we poll `user_state`). v2 for lower latency.
- Cross-margin balance management (we run isolated per-coin). v2.
- Funding rate harvesting (separate strategy).
- Multi-account support (single agent, single main wallet).

---

## Total estimated effort

- Phase 0 + A: 0.5 day (config + skeleton)
- Phase B: 1 day (entry/expiry/fill detection)
- Phase C: 1.5 days (full position management)
- Phase D: 0.5 day (deadman + flatten script)
- **Code complete: ~3.5 days**
- Phase E: 7 days (passive testnet observation)
- Phase F: 0–2 days (only if needed)
- Phase G: gated rollout, 1-3 weeks to scale up safely

**Earliest realistic real-money date: ~3 weeks from start.**
