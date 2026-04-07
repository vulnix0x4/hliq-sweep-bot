from __future__ import annotations

import json

from hliq_bot.config import FeedConfig
from hliq_bot.data.hyperliquid_ws import HyperliquidWsClient


def test_parse_trades_maps_hyperliquid_side_tokens() -> None:
    client = HyperliquidWsClient(FeedConfig())
    payload = {
        "channel": "trades",
        "data": [
            {"px": "100.0", "sz": "0.25", "side": "B", "time": 1700000000000},
            {"px": "101.0", "sz": "0.40", "side": "A", "time": 1700000000100},
        ],
    }

    events = client._parse_message(json.dumps(payload))
    assert len(events) == 2
    assert events[0].trade is not None
    assert events[1].trade is not None
    assert events[0].trade.side == "buy"
    assert events[1].trade.side == "sell"


def test_parse_trades_handles_nested_data_shape() -> None:
    client = HyperliquidWsClient(FeedConfig())
    payload = {
        "channel": "trades",
        "data": {"trades": [{"px": "100.0", "sz": "0.25", "side": "a", "time": 1700000000000}]},
    }

    events = client._parse_message(json.dumps(payload))
    assert len(events) == 1
    assert events[0].trade is not None
    assert events[0].trade.side == "sell"


def test_parse_trades_extracts_coin_from_data() -> None:
    client = HyperliquidWsClient(FeedConfig(coins_str="BTC,ETH"))
    payload = {
        "channel": "trades",
        "data": [
            {"px": "50000.0", "sz": "0.1", "side": "B", "time": 1700000000000, "coin": "BTC"},
            {"px": "3000.0", "sz": "1.0", "side": "A", "time": 1700000000100, "coin": "ETH"},
        ],
    }

    events = client._parse_message(json.dumps(payload))
    assert len(events) == 2
    assert events[0].coin == "BTC"
    assert events[1].coin == "ETH"


def test_parse_book_extracts_coin() -> None:
    client = HyperliquidWsClient(FeedConfig(coins_str="ETH"))
    payload = {
        "channel": "l2Book",
        "data": {
            "coin": "ETH",
            "time": 1700000000000,
            "levels": [
                [{"px": "3000.0", "sz": "10.0"}],
                [{"px": "3001.0", "sz": "8.0"}],
            ],
        },
    }

    events = client._parse_message(json.dumps(payload))
    assert len(events) == 1
    assert events[0].coin == "ETH"
    assert events[0].kind == "book"


def test_subscribe_sends_messages_for_all_coins() -> None:
    import asyncio

    sent: list[str] = []

    class FakeWs:
        async def send(self, msg: str) -> None:
            sent.append(msg)

    client = HyperliquidWsClient(FeedConfig(coins_str="BTC,ETH,SOL"))
    asyncio.run(client._subscribe(FakeWs()))

    # Should have 2 subscriptions per coin (trades + l2Book) = 6 total
    assert len(sent) == 6
    parsed = [json.loads(s) for s in sent]
    coins_seen = set()
    for p in parsed:
        coins_seen.add(p["subscription"]["coin"])
    assert coins_seen == {"BTC", "ETH", "SOL"}
