from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(slots=True)
class TradeEvent:
    ts_ms: int
    price: float
    size: float
    side: str = "unknown"


@dataclass(slots=True)
class BookTopEvent:
    ts_ms: int
    best_bid: float
    best_ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def spread_bps(self) -> float:
        if self.best_bid <= 0 or self.best_ask <= 0:
            return 0.0
        mid = (self.best_bid + self.best_ask) / 2.0
        return ((self.best_ask - self.best_bid) / mid) * 10_000.0


@dataclass(slots=True)
class MarketEvent:
    kind: str
    ts_ms: int
    coin: str = ""
    trade: TradeEvent | None = None
    book: BookTopEvent | None = None
    raw: dict | None = None


@dataclass(slots=True)
class Bar:
    start_ms: int
    end_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    vwap: float
    avg_spread_bps: float

    @property
    def range_pct(self) -> float:
        if self.open <= 0:
            return 0.0
        return ((self.high - self.low) / self.open) * 100.0


@dataclass(slots=True)
class SweepSignal:
    side: Side
    level: float
    level_label: str
    sweep_extreme: float
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    confidence: float
    reason: str
    created_ms: int
    coin: str = ""
    overshoot_bps: float = 0.0
    reclaim_bps: float = 0.0
    volume_ratio: float = 0.0
    wick_ratio: float = 0.0
    signal_score: float = 0.0


@dataclass(slots=True)
class PositionSize:
    qty: float
    notional: float
    risk_dollars: float
    stop_distance_abs: float


@dataclass(slots=True)
class RiskCheck:
    allowed: bool
    reason: str


@dataclass(slots=True)
class MarketState:
    ts_ms: int
    ws_healthy: bool
    data_stale: bool
    spread_bps: float
    recent_bar_ranges_pct: list[float]
    move_30s_pct: float


@dataclass(slots=True)
class PendingEntry:
    side: Side
    qty: float
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    created_ms: int
    expiry_sec: int
    level_label: str
    risk_dollars: float
    coin: str = ""
    signal_id: str = ""
    external_oid: int | None = None  # exchange's order id (for live cancel)


@dataclass(slots=True)
class OpenPosition:
    side: Side
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    opened_ms: int
    qty_initial: float
    qty_remaining: float
    risk_dollars: float
    coin: str = ""
    tp1_filled: bool = False
    realized_pnl: float = 0.0
    realized_fees: float = 0.0  # cumulative fees+rebates from entry + any partial exits
    best_price: float = 0.0
    worst_price: float = 0.0
    signal_id: str = ""


@dataclass(slots=True)
class ClosedTrade:
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    risk_dollars: float
    r_multiple: float
    opened_ms: int
    closed_ms: int
    exit_reason: str
    coin: str = ""
    mfe_pnl: float = 0.0
    mae_pnl: float = 0.0
    signal_id: str = ""
    fees_paid: float = 0.0  # net fees (positive = paid, negative = received)
    pnl_gross: float = 0.0  # pnl before fees


class ExecEventType(str, Enum):
    ENTRY_PLACED = "entry_placed"
    ENTRY_FILLED = "entry_filled"
    ENTRY_REJECTED = "entry_rejected"  # pre-flight reject (sub-min-notional, sub-min-lot, etc.)
    PARTIAL_TP = "partial_tp"
    POSITION_CLOSED = "position_closed"
    ORDER_CANCELED = "order_canceled"


@dataclass(slots=True)
class ExecutionUpdate:
    ts_ms: int
    event_type: ExecEventType
    message: str
    signal_id: str = ""
    closed_trade: ClosedTrade | None = None
