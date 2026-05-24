"""Unit tests for the AI trading strategy.

Covers: client parsing, context aggregation, decision validation, and the
strategy's open-position safety rules. The LLM HTTP call is mocked
throughout — no network is touched.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from hliq_bot.ai.client import AICallResult, AIClientError, CostBudget, OpenRouterClient, _cost_for
from hliq_bot.ai.context import build_coin_context
from hliq_bot.ai.prompts import SYSTEM_PROMPT, build_user_message, decision_schema
from hliq_bot.ai.strategy import AIDecisionResult, AIStrategy
from hliq_bot.config import AIConfig
from hliq_bot.models import Bar, Side


# ---- Test fixtures ----


@dataclass
class _FakeExecutor:
    position: object | None = None
    pending_entry: object | None = None
    def has_exposure(self) -> bool:
        return self.position is not None or self.pending_entry is not None


@dataclass
class _FakeVWAP:
    val: float | None = None
    def session_vwap(self) -> float | None:
        return self.val


@dataclass
class _FakeWorker:
    coin: str = "BTC"
    last_spread_bps: float = 1.0
    last_best_bid: float = 99.95
    last_best_ask: float = 100.05
    recent_signed_flow: deque = field(default_factory=lambda: deque([(1000, 1.0), (2000, -0.5)]))
    recent_trade_prices: deque = field(default_factory=lambda: deque([(1000, 100.0), (2000, 100.5), (3000, 99.8)]))
    executor: object = field(default_factory=_FakeExecutor)
    vwap_tracker: object = field(default_factory=_FakeVWAP)


def _bar(start_ms: int, close: float, high: float | None = None, low: float | None = None) -> Bar:
    return Bar(
        start_ms=start_ms, end_ms=start_ms + 60_000,
        open=close, high=high or close, low=low or close,
        close=close, volume=10.0, trade_count=20, vwap=close, avg_spread_bps=1.0,
    )


def _ai_cfg(enabled: bool = True) -> AIConfig:
    return AIConfig(
        enabled=enabled, provider="openrouter", model="google/gemini-3.5-flash",
        interval_sec=300, max_calls_hourly=60, timeout_sec=30.0,
        daily_budget_usd=5.0, context_bars=10,
    )


# ---- Client ----


def test_cost_calculator_uses_pricing_table():
    cost = _cost_for("google/gemini-3.5-flash", prompt_tokens=10000, completion_tokens=500)
    assert cost > 0
    # missing model bills at 0 so unknown providers don't false-warn
    assert _cost_for("unknown/model", 10000, 500) == 0.0


def test_client_rejects_empty_api_key():
    with pytest.raises(AIClientError):
        OpenRouterClient(api_key="", model="any")


def test_chat_json_parses_structured_response():
    """When the model returns valid JSON, decision is parsed."""
    client = OpenRouterClient(api_key="sk-test", model="google/gemini-3.5-flash")
    fake_resp = MagicMock()
    fake_resp.read.return_value = (
        b'{"choices":[{"message":{"content":"{\\"action\\":\\"hold\\",\\"confidence\\":0.5,'
        b'\\"reasoning\\":\\"r\\",\\"stop_price\\":null,\\"tp1_price\\":null,\\"tp2_price\\":null}"},'
        b'"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20}}'
    )
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = client.chat_json(system="s", user="u", schema=decision_schema())
    assert result.decision is not None
    assert result.decision["action"] == "hold"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.cost_usd > 0


def test_chat_json_extracts_json_from_code_fence():
    """Some models wrap output in ```json ... ```; client handles it."""
    client = OpenRouterClient(api_key="sk-test", model="google/gemini-3.5-flash")
    body = (
        '{"choices":[{"message":{"content":"```json\\n{\\"action\\":\\"hold\\",\\"confidence\\":0.5,'
        '\\"reasoning\\":\\"r\\",\\"stop_price\\":null,\\"tp1_price\\":null,\\"tp2_price\\":null}\\n```"},'
        '"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10}}'
    )
    fake_resp = MagicMock()
    fake_resp.read.return_value = body.encode("utf-8")
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = client.chat_json(system="s", user="u")
    assert result.decision is not None
    assert result.decision["action"] == "hold"


# ---- Budget ----


def test_cost_budget_tracks_24h_window_and_warns_once():
    b = CostBudget(daily_budget_usd=0.01)
    now = 1_000_000_000_000  # arbitrary ms
    b.record(0.004, now_ms=now)
    assert not b.over_budget(now_ms=now)
    b.record(0.011, now_ms=now)
    assert b.over_budget(now_ms=now)
    # 24h+1ms later, the early cost falls out and we're back under
    future = now + 24 * 60 * 60 * 1000 + 1
    assert b.spent_last_24h(now_ms=future) == 0.0
    assert not b.over_budget(now_ms=future)


# ---- Context ----


def test_build_coin_context_emits_recent_bars_and_position():
    w = _FakeWorker()
    bars = [_bar(0, 100.0), _bar(60_000, 100.5, high=100.7, low=100.3)]
    ctx = build_coin_context(
        w, bars=bars, now_ms=10_000,
        account_equity=50.0, daily_pnl=0.0, daily_r=0.0,
        recent_outcomes=[], context_bars=5,
    )
    d = ctx.to_prompt_dict()
    assert d["coin"] == "BTC"
    assert len(d["recent_bars"]) == 2
    assert d["recent_bars"][-1]["c"] == 100.5
    assert d["last_price"] == pytest.approx(100.0)  # (bid+ask)/2
    assert d["open_position"] is None


def test_build_coin_context_summarizes_open_position():
    w = _FakeWorker()
    from hliq_bot.models import OpenPosition
    w.executor.position = OpenPosition(
        signal_id="s", side=Side.LONG, entry_price=100.0, stop_price=99.5,
        tp1_price=101.0, tp2_price=102.0, opened_ms=9_000,
        qty_initial=0.1, qty_remaining=0.1, risk_dollars=0.05,
        coin="BTC", best_price=100.5, worst_price=99.8,
    )
    ctx = build_coin_context(
        w, bars=[_bar(0, 100.0)], now_ms=10_000,
        account_equity=50.0, daily_pnl=0.0, daily_r=0.0, recent_outcomes=[],
    )
    pos = ctx.to_prompt_dict()["open_position"]
    assert pos is not None
    assert pos["side"] == "long"
    assert pos["entry_price"] == 100.0
    assert "opened_ms_ago" in pos


# ---- Strategy ----


def test_strategy_disabled_does_not_construct_client():
    s = AIStrategy(_ai_cfg(enabled=False))
    assert s._client is None


def test_strategy_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        AIStrategy(_ai_cfg(enabled=True))


def test_strategy_constructs_with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    assert s._client is not None
    assert s._client.model == "google/gemini-3.5-flash"


def test_should_decide_respects_per_coin_interval(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    # First call always allowed
    assert s.should_decide("BTC", now_ms=1_000_000)
    s._last_decision_ms["BTC"] = 1_000_000
    # 100s later — still under 300s interval
    assert not s.should_decide("BTC", now_ms=1_100_000)
    # 301s later — interval elapsed
    assert s.should_decide("BTC", now_ms=1_301_000)


def test_strategy_skips_on_rate_limit(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _ai_cfg(enabled=True)
    cfg.max_calls_hourly = 2
    s = AIStrategy(cfg)
    s._call_times_ms = [1000, 2000]  # already at the cap
    w = _FakeWorker()
    result = s.decide_for_coin(
        w, bars=[_bar(0, 100.0)], now_ms=3000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    assert result.action == "skipped"
    assert result.skip_reason == "hourly_rate_limit"


def test_strategy_skips_when_budget_exhausted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _ai_cfg(enabled=True)
    cfg.daily_budget_usd = 0.001
    s = AIStrategy(cfg)
    s.budget.record(0.01, now_ms=1000)
    result = s.decide_for_coin(
        _FakeWorker(), bars=[_bar(0, 100.0)], now_ms=2000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    assert result.action == "skipped"
    assert result.skip_reason == "daily_budget_exhausted"


def test_strategy_validate_open_rejects_stop_on_wrong_side(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    ctx = build_coin_context(
        _FakeWorker(), bars=[_bar(0, 100.0)], now_ms=1000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    # LONG with stop ABOVE entry — invalid
    sig, err = s._validate_open("BTC", ctx, {
        "action": "open_long", "stop_price": 101.0,
        "tp1_price": None, "tp2_price": None, "confidence": 0.5, "reasoning": "x",
    })
    assert sig is None
    assert "stop_not_below_entry_for_long" in err


def test_strategy_validate_open_rejects_too_tight_stop(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    ctx = build_coin_context(
        _FakeWorker(), bars=[_bar(0, 100.0)], now_ms=1000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    # 1bp stop — too tight
    sig, err = s._validate_open("BTC", ctx, {
        "action": "open_long", "stop_price": 99.99,
        "tp1_price": None, "tp2_price": None, "confidence": 0.5, "reasoning": "x",
    })
    assert sig is None
    assert "stop_distance_out_of_range" in err


def test_strategy_validate_open_accepts_valid_long(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    ctx = build_coin_context(
        _FakeWorker(), bars=[_bar(0, 100.0)], now_ms=1000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    sig, err = s._validate_open("BTC", ctx, {
        "action": "open_long", "stop_price": 99.7,  # 30bp stop
        "tp1_price": 100.6, "tp2_price": 101.2,
        "confidence": 0.7, "reasoning": "valid setup",
    })
    assert err is None
    assert sig is not None
    assert sig.side == Side.LONG
    assert sig.stop_price < sig.entry_price
    assert sig.tp1_price > sig.entry_price


def test_strategy_validate_open_fills_default_tps_when_omitted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    ctx = build_coin_context(
        _FakeWorker(), bars=[_bar(0, 100.0)], now_ms=1000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    sig, err = s._validate_open("BTC", ctx, {
        "action": "open_short", "stop_price": 100.3,  # 30bp above
        "tp1_price": None, "tp2_price": None,
        "confidence": 0.6, "reasoning": "fade",
    })
    assert err is None
    assert sig is not None
    assert sig.side == Side.SHORT
    # Defaults: tp1 at 2R, tp2 at 4R below entry for SHORT
    assert sig.tp1_price < sig.entry_price
    assert sig.tp2_price < sig.tp1_price


# ---- Prompt / schema ----


def test_decision_schema_is_valid_json_schema_shape():
    schema = decision_schema()
    assert schema["type"] == "object"
    assert "action" in schema["properties"]
    assert set(schema["properties"]["action"]["enum"]) == {
        "open_long", "open_short", "close", "hold",
        "move_stop_to_breakeven", "modify_stop",
        "scale_out", "add_to_position",
    }
    # strict mode requires every property listed in required
    for prop in schema["properties"]:
        assert prop in schema["required"]


def test_strategy_validates_modify_stop_requires_new_price(monkeypatch):
    from hliq_bot.ai.client import AICallResult
    from unittest.mock import MagicMock
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    s._llm = MagicMock()
    s._llm.chat_json.return_value = AICallResult(
        decision={
            "action": "modify_stop", "confidence": 0.6, "reasoning": "r",
            "stop_price": None, "tp1_price": None, "tp2_price": None,
            "new_stop_price": None, "scale_fraction": None, "add_qty_fraction": None,
        },
        raw_text="{}", model="test", prompt_tokens=10, completion_tokens=5,
        cost_usd=0.0, latency_ms=10, finish_reason="stop",
    )
    w = _FakeWorker()
    result = s.decide_for_coin(
        w, bars=[_bar(0, 100.0)], now_ms=1000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    assert result.action == "error"
    assert result.error == "missing_new_stop_price"


def test_strategy_validates_scale_out_fraction_range(monkeypatch):
    from hliq_bot.ai.client import AICallResult
    from unittest.mock import MagicMock
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    s._llm = MagicMock()
    s._llm.chat_json.return_value = AICallResult(
        decision={
            "action": "scale_out", "confidence": 0.6, "reasoning": "r",
            "stop_price": None, "tp1_price": None, "tp2_price": None,
            "new_stop_price": None, "scale_fraction": 1.5,  # invalid
            "add_qty_fraction": None,
        },
        raw_text="{}", model="test", prompt_tokens=10, completion_tokens=5,
        cost_usd=0.0, latency_ms=10, finish_reason="stop",
    )
    result = s.decide_for_coin(
        _FakeWorker(), bars=[_bar(0, 100.0)], now_ms=1000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    assert result.action == "error"
    assert result.error == "invalid_scale_fraction"


def test_strategy_accepts_scale_out_valid_fraction(monkeypatch):
    from hliq_bot.ai.client import AICallResult
    from unittest.mock import MagicMock
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = AIStrategy(_ai_cfg(enabled=True))
    s._llm = MagicMock()
    s._llm.chat_json.return_value = AICallResult(
        decision={
            "action": "scale_out", "confidence": 0.6, "reasoning": "take half",
            "stop_price": None, "tp1_price": None, "tp2_price": None,
            "new_stop_price": None, "scale_fraction": 0.5,
            "add_qty_fraction": None,
        },
        raw_text="{}", model="test", prompt_tokens=10, completion_tokens=5,
        cost_usd=0.0, latency_ms=10, finish_reason="stop",
    )
    result = s.decide_for_coin(
        _FakeWorker(), bars=[_bar(0, 100.0)], now_ms=1000,
        account_equity=50, daily_pnl=0, daily_r=0, recent_outcomes=[],
    )
    assert result.action == "scale_out"
    assert result.scale_fraction == 0.5


def test_build_user_message_contains_context():
    msg = build_user_message({"coin": "BTC", "x": 1})
    assert "BTC" in msg
    assert "Decide" in msg


# ---- Memory persistence ----


def test_memory_records_decision_and_outcome(tmp_path):
    from hliq_bot.ai.memory import AIMemory, MemoryEntry
    mem = AIMemory(tmp_path / "ai_memory.jsonl", max_entries=5)
    mem.record_decision(MemoryEntry(
        decision_id="d1", ts_ms=1000, coin="BTC",
        action="open_long", reasoning="r", confidence=0.7,
        stop_price=99.0, entry_price=100.0,
    ))
    assert mem.record_outcome(
        "d1", ts_ms=2000, exit_reason="tp1",
        pnl=0.5, r_multiple=1.2, hold_sec=120.0,
    )
    entries = mem.recent_for_prompt(coin="BTC", limit=5)
    assert len(entries) == 1
    assert entries[0]["r"] == 1.2
    assert entries[0]["exit"] == "tp1"


def test_memory_persists_and_reloads(tmp_path):
    from hliq_bot.ai.memory import AIMemory, MemoryEntry
    path = tmp_path / "ai_memory.jsonl"
    mem1 = AIMemory(path, max_entries=10)
    mem1.record_decision(MemoryEntry(
        decision_id="x1", ts_ms=1000, coin="ETH",
        action="open_short", reasoning="r", confidence=0.5,
    ))
    mem1.record_outcome("x1", ts_ms=2000, exit_reason="stop",
                       pnl=-0.5, r_multiple=-1.0, hold_sec=60.0)
    # Re-open the same file in a fresh instance — should reconstruct state.
    mem2 = AIMemory(path, max_entries=10)
    mem2.load()
    stats = mem2.summary_stats()
    assert stats["total_decisions"] == 1
    assert stats["resolved_trades"] == 1
    assert stats["wins"] == 0
    assert stats["total_pnl"] == pytest.approx(-0.5)


def test_memory_summary_stats_empty(tmp_path):
    from hliq_bot.ai.memory import AIMemory
    mem = AIMemory(tmp_path / "ai_memory.jsonl")
    s = mem.summary_stats()
    assert s["total_decisions"] == 0
    assert s["resolved_trades"] == 0


def test_memory_trims_to_max(tmp_path):
    from hliq_bot.ai.memory import AIMemory, MemoryEntry
    mem = AIMemory(tmp_path / "m.jsonl", max_entries=3)
    for i in range(5):
        mem.record_decision(MemoryEntry(
            decision_id=f"d{i}", ts_ms=1000 + i, coin="BTC",
            action="hold", reasoning="r", confidence=0.5,
        ))
    # Only the 3 most recent should be in memory after trim.
    assert mem.summary_stats()["total_decisions"] == 3


# ---- Resilient client ----


def test_resilient_llm_retries_then_succeeds():
    from hliq_bot.ai.client import (
        AICallResult, AIRetryableError, OpenRouterClient, ResilientLLM,
    )
    primary = MagicMock(spec=OpenRouterClient)
    primary.model = "primary-model"
    # fail once, succeed on retry
    primary.chat_json.side_effect = [
        AIRetryableError("timeout"),
        AICallResult(decision={"action": "hold"}, raw_text="{}", model="primary-model",
                     prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
                     latency_ms=100, finish_reason="stop"),
    ]
    llm = ResilientLLM(primary=primary, max_retries=2, retry_base_sec=0.001)
    result = llm.chat_json(system="s", user="u")
    assert result.decision["action"] == "hold"
    assert primary.chat_json.call_count == 2


def test_resilient_llm_falls_back_after_primary_exhausted():
    from hliq_bot.ai.client import (
        AICallResult, AIRetryableError, OpenRouterClient, ResilientLLM,
    )
    primary = MagicMock(spec=OpenRouterClient)
    primary.model = "primary"
    primary.chat_json.side_effect = AIRetryableError("primary down")
    secondary = MagicMock(spec=OpenRouterClient)
    secondary.model = "secondary"
    secondary.chat_json.return_value = AICallResult(
        decision={"action": "hold"}, raw_text="{}", model="secondary",
        prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
        latency_ms=100, finish_reason="stop",
    )
    llm = ResilientLLM(primary=primary, fallbacks=[secondary],
                       max_retries=1, retry_base_sec=0.001)
    result = llm.chat_json(system="s", user="u")
    assert result.model == "secondary"
    # primary was retried, then secondary was called once.
    assert primary.chat_json.call_count == 2
    assert secondary.chat_json.call_count == 1


def test_resilient_llm_circuit_breaker_opens_after_threshold():
    from hliq_bot.ai.client import (
        AIClientError, AIRetryableError, OpenRouterClient, ResilientLLM,
    )
    primary = MagicMock(spec=OpenRouterClient)
    primary.model = "primary"
    primary.chat_json.side_effect = AIRetryableError("down")
    llm = ResilientLLM(primary=primary, max_retries=2, retry_base_sec=0.001,
                       cb_threshold=3, cb_cool_down_sec=300)
    # First call should exhaust retries (3 calls = primary 1 + retries 2)
    with pytest.raises(AIClientError):
        llm.chat_json(system="s", user="u")
    # Breaker should now be open (3 failures >= threshold)
    assert llm.circuit_open()
    # Subsequent call short-circuits with "circuit_breaker_open"
    with pytest.raises(AIClientError, match="circuit_breaker_open"):
        llm.chat_json(system="s", user="u")


def test_resilient_llm_breaker_does_not_open_below_threshold():
    from hliq_bot.ai.client import (
        AIClientError, AIRetryableError, OpenRouterClient, ResilientLLM,
    )
    primary = MagicMock(spec=OpenRouterClient)
    primary.model = "primary"
    primary.chat_json.side_effect = AIRetryableError("down")
    llm = ResilientLLM(primary=primary, max_retries=1, retry_base_sec=0.001,
                       cb_threshold=10, cb_cool_down_sec=60)
    with pytest.raises(AIClientError):
        llm.chat_json(system="s", user="u")
    # Only 2 failures (< 10 threshold) — breaker stays closed.
    assert not llm.circuit_open()


def test_resilient_llm_resets_breaker_on_success():
    from hliq_bot.ai.client import (
        AICallResult, AIRetryableError, OpenRouterClient, ResilientLLM,
    )
    primary = MagicMock(spec=OpenRouterClient)
    primary.model = "primary"
    primary.chat_json.side_effect = [
        AIRetryableError("d"),
        AICallResult(decision={"action": "hold"}, raw_text="{}", model="primary",
                     prompt_tokens=10, completion_tokens=5, cost_usd=0,
                     latency_ms=10, finish_reason="stop"),
        AIRetryableError("d"),
        AIRetryableError("d"),
    ]
    llm = ResilientLLM(primary=primary, max_retries=3, retry_base_sec=0.001,
                       cb_threshold=3, cb_cool_down_sec=60)
    # First call: retries once and succeeds. Counter goes 1 -> 0.
    result = llm.chat_json(system="s", user="u")
    assert result.decision["action"] == "hold"
    # Sanity: breaker reset.
    assert not llm.circuit_open()
    assert llm._cb.consecutive_failures == 0
