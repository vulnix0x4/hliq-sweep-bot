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
