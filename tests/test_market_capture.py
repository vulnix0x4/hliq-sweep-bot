from __future__ import annotations

from pathlib import Path

from hliq_bot.analytics.market_capture import MarketCaptureWriter
from hliq_bot.models import BookTopEvent, MarketEvent, TradeEvent
from hliq_bot.replay.loader import load_market_events


def test_capture_writes_coin_field_and_loader_passes_it_through(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = MarketCaptureWriter(path=str(path))

    writer.write(
        MarketEvent(
            kind="trade",
            ts_ms=1_700_000_000_000,
            coin="ETH",
            trade=TradeEvent(ts_ms=1_700_000_000_000, price=2350.0, size=0.1, side="buy"),
        )
    )
    writer.write(
        MarketEvent(
            kind="book",
            ts_ms=1_700_000_000_500,
            coin="SOL",
            book=BookTopEvent(
                ts_ms=1_700_000_000_500,
                best_bid=90.0,
                best_ask=90.05,
                bid_size=5.0,
                ask_size=4.0,
            ),
        )
    )

    events = list(load_market_events(str(path)))

    assert len(events) == 2
    assert events[0].coin == "ETH"
    assert events[0].kind == "trade"
    assert events[0].trade is not None
    assert events[0].trade.price == 2350.0
    assert events[1].coin == "SOL"
    assert events[1].kind == "book"
    assert events[1].book is not None
    assert events[1].book.best_ask == 90.05


def test_loader_backward_compat_reads_rows_without_coin(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        '{"kind":"trade","ts_ms":1700000000000,"price":75000.0,"size":0.01,"side":"buy"}\n'
        '{"kind":"book","ts_ms":1700000000500,"best_bid":75000.0,"best_ask":75001.0,'
        '"bid_size":1.0,"ask_size":2.0}\n',
        encoding="utf-8",
    )

    events = list(load_market_events(str(path)))

    assert len(events) == 2
    assert events[0].coin == ""
    assert events[1].coin == ""


def test_loader_can_skip_legacy_rows_without_coin(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"kind":"trade","ts_ms":1700000000000,"price":75000.0,"size":0.01,"side":"buy"}\n'
        '{"kind":"trade","ts_ms":1700000001000,"coin":"SOL","price":90.0,"size":1.0,"side":"sell"}\n',
        encoding="utf-8",
    )

    events = list(load_market_events(str(path), require_coin=True))

    assert len(events) == 1
    assert events[0].coin == "SOL"


def test_capture_rotates_when_file_exceeds_max_bytes(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("x" * 128, encoding="utf-8")
    writer = MarketCaptureWriter(path=str(path), max_bytes=64, backups=2)

    writer.write(
        MarketEvent(
            kind="trade",
            ts_ms=1_700_000_000_000,
            coin="BTC",
            trade=TradeEvent(ts_ms=1_700_000_000_000, price=75000.0, size=0.01, side="buy"),
        )
    )

    assert (tmp_path / "events.jsonl.1").exists()
    assert path.exists()
    assert "BTC" in path.read_text(encoding="utf-8")
