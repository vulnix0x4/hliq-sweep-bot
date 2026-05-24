"""AI-driven trading strategy.

When BOT_AI_ENABLED=true, this module is the source of trading signals
instead of SweepDetector. On a per-coin timer (default 5 min), it polls the
LLM with the current market context and acts on the returned decision:
  - open_long / open_short: convert to a SweepSignal-shape and submit_entry
  - close: force-close the current position on the next trade tick
  - hold: do nothing

The risk governor still computes position size from equity * risk_per_trade_pct,
the executor still enforces max_notional / native_stop / deadman, and the
order journal still records every event. The AI cannot bypass any safety.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import time
from typing import Any

from hliq_bot.ai.client import AICallResult, AIClientError, CostBudget, OpenRouterClient, ResilientLLM
from hliq_bot.ai.context import CoinContext, build_coin_context
from hliq_bot.ai.market_data import MarketDataCache
from hliq_bot.ai.memory import AIMemory, MemoryEntry
from hliq_bot.ai.prompts import SYSTEM_PROMPT, build_user_message, decision_schema
from hliq_bot.config import AIConfig
from hliq_bot.models import Bar, Side, SweepSignal

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AIDecisionResult:
    """Outcome of one decide_for_coin call. The bot consumes this."""
    coin: str
    action: str               # open_long | open_short | close | hold | modify_stop |
                              # move_stop_to_breakeven | scale_out | add_to_position |
                              # error | skipped
    signal: SweepSignal | None  # populated for open_long / open_short / add_to_position
    reasoning: str
    confidence: float
    raw_decision: dict[str, Any] | None
    cost_usd: float
    latency_ms: int
    model: str
    # Action-specific parameters (only set for the relevant action):
    new_stop_price: float | None = None        # for modify_stop
    scale_fraction: float | None = None        # for scale_out
    add_qty_fraction: float | None = None      # for add_to_position
    error: str | None = None
    skip_reason: str | None = None


def _round_to_sig_figs(value: float, sig_figs: int = 5) -> float:
    """Match HL's wire-format tick precision so the executor's pre-flight
    rounding doesn't drift the AI's prices."""
    if value <= 0:
        return value
    return float(f"{value:.{sig_figs}g}")


def read_override_flag(runtime_dir: str | Path) -> str | None:
    """Return the AI override mode set by an operator script, or None.

    Modes: "pause"|"no_new" — block new opens; "close_all" — also force-close.
    """
    path = Path(runtime_dir) / "ai_override.flag"
    if not path.exists():
        return None
    try:
        first = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if not first:
            return None
        mode = first[0].strip().lower()
        if mode in {"pause", "no_new", "close_all"}:
            return mode
        return mode  # surface anything weird so operator sees it
    except OSError:
        return None


class AIStrategy:
    def __init__(
        self,
        cfg: AIConfig,
        *,
        api_key: str | None = None,
        memory: AIMemory | None = None,
        market_data: MarketDataCache | None = None,
    ) -> None:
        self.cfg = cfg
        # Track last-decision timestamps to enforce per-coin cadence.
        self._last_decision_ms: dict[str, int] = {}
        # Per-call timestamps for hourly rate limit (across all coins).
        self._call_times_ms: list[int] = []
        # Cost tracker (24h rolling window).
        self.budget = CostBudget(cfg.daily_budget_usd)
        # Persistent AI trade memory (optional — callers pass it in).
        self.memory: AIMemory | None = memory
        if self.memory is not None:
            self.memory.load()
        # HL market-data cache for funding/OI/L2/etc enrichment (optional).
        self.market_data: MarketDataCache | None = market_data
        # LLM client. When enabled, wrap in ResilientLLM with fallback chain.
        self._client: OpenRouterClient | None = None
        self._llm: ResilientLLM | None = None
        if cfg.enabled:
            key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "BOT_AI_ENABLED=true but OPENROUTER_API_KEY is not set in env"
                )
            self._client = OpenRouterClient(
                api_key=key,
                model=cfg.model,
                base_url=cfg.api_base_url,
            )
            fallbacks: list[OpenRouterClient] = []
            for fb_model in (cfg.fallback_models or []):
                fb_model = fb_model.strip()
                if not fb_model or fb_model == cfg.model:
                    continue
                fallbacks.append(OpenRouterClient(
                    api_key=key, model=fb_model, base_url=cfg.api_base_url,
                ))
            self._llm = ResilientLLM(
                primary=self._client,
                fallbacks=fallbacks,
                max_retries=cfg.retry_attempts,
                retry_base_sec=cfg.retry_base_sec,
                cb_threshold=cfg.circuit_breaker_threshold,
                cb_cool_down_sec=cfg.circuit_breaker_cool_down_sec,
            )

    # ---- Public API ----

    def should_decide(self, coin: str, now_ms: int) -> bool:
        """Has enough time passed since the last decision for this coin?"""
        last = self._last_decision_ms.get(coin, 0)
        if last == 0:
            return True
        return (now_ms - last) >= max(1, self.cfg.interval_sec) * 1000

    def decide_for_coin(
        self,
        worker: Any,
        *,
        bars: list[Bar],
        now_ms: int,
        account_equity: float,
        daily_pnl: float,
        daily_r: float,
        recent_outcomes: list[dict],
        workers_by_coin: dict[str, Any] | None = None,
    ) -> AIDecisionResult:
        """Run one decision cycle for a single coin's worker."""
        coin = worker.coin

        # Rate-limit + budget gates BEFORE building context (which is cheap
        # but the LLM call is the dollar cost; bail early if we'd skip).
        skip = self._pre_decide_check(now_ms)
        if skip is not None:
            self._last_decision_ms[coin] = now_ms  # avoid hot-loop retries
            return AIDecisionResult(
                coin=coin,
                action="skipped",
                signal=None,
                reasoning="",
                confidence=0.0,
                raw_decision=None,
                cost_usd=0.0,
                latency_ms=0,
                model=self.cfg.model,
                skip_reason=skip,
            )

        if not bars:
            self._last_decision_ms[coin] = now_ms
            return AIDecisionResult(
                coin=coin, action="skipped", signal=None, reasoning="",
                confidence=0.0, raw_decision=None, cost_usd=0.0,
                latency_ms=0, model=self.cfg.model,
                skip_reason="no_bars_yet",
            )

        ctx = build_coin_context(
            worker,
            bars=bars,
            now_ms=now_ms,
            account_equity=account_equity,
            daily_pnl=daily_pnl,
            daily_r=daily_r,
            recent_outcomes=recent_outcomes,
            context_bars=self.cfg.context_bars,
            market_data=self.market_data,
            workers_by_coin=workers_by_coin,
        )

        try:
            call = self._call_llm(ctx)
        except AIClientError as exc:
            log.warning("AI call failed for %s: %s", coin, exc)
            self._last_decision_ms[coin] = now_ms
            return AIDecisionResult(
                coin=coin, action="error", signal=None,
                reasoning="", confidence=0.0, raw_decision=None,
                cost_usd=0.0, latency_ms=0, model=self.cfg.model,
                error=str(exc),
            )

        self._last_decision_ms[coin] = now_ms
        self._call_times_ms.append(now_ms)
        self.budget.record(call.cost_usd, now_ms=now_ms)

        decision = call.decision
        if not isinstance(decision, dict):
            log.warning(
                "AI non_json_response for %s (finish=%s tokens=%d): raw[:300]=%r",
                coin, call.finish_reason, call.completion_tokens,
                call.raw_text[:300],
            )
            return AIDecisionResult(
                coin=coin, action="error", signal=None,
                reasoning=call.raw_text[:200], confidence=0.0,
                raw_decision=None, cost_usd=call.cost_usd,
                latency_ms=call.latency_ms, model=call.model,
                error="non_json_response",
            )

        return self._build_result(coin, ctx, call, decision)

    # ---- Internal ----

    def _pre_decide_check(self, now_ms: int) -> str | None:
        # Trim hourly window
        cutoff = now_ms - 60 * 60 * 1000
        self._call_times_ms = [t for t in self._call_times_ms if t >= cutoff]
        if len(self._call_times_ms) >= max(1, self.cfg.max_calls_hourly):
            return "hourly_rate_limit"
        if self.budget.over_budget(now_ms=now_ms):
            return "daily_budget_exhausted"
        return None

    def _call_llm(self, ctx: CoinContext) -> AICallResult:
        # Prefer the resilient wrapper when available (retries + fallbacks);
        # fall back to the raw client for older test fixtures.
        llm = self._llm if self._llm is not None else self._client
        assert llm is not None
        # Inject persistent memory into the prompt's recent_outcomes slot.
        # Falls back to whatever the worker passed in if memory is unset.
        prompt_dict = ctx.to_prompt_dict()
        if self.memory is not None:
            from_memory = self.memory.recent_for_prompt(coin=ctx.coin, limit=10)
            if from_memory:
                prompt_dict["recent_outcomes"] = from_memory
        return llm.chat_json(
            system=SYSTEM_PROMPT,
            user=build_user_message(prompt_dict),
            schema=decision_schema(),
            timeout_sec=self.cfg.timeout_sec,
            max_tokens=self.cfg.max_response_tokens,
        )

    def _build_result(
        self,
        coin: str,
        ctx: CoinContext,
        call: AICallResult,
        decision: dict[str, Any],
    ) -> AIDecisionResult:
        action = str(decision.get("action", "")).strip().lower()
        reasoning = str(decision.get("reasoning", "") or "")[:500]
        try:
            confidence = float(decision.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if action in {"open_long", "open_short"}:
            sig, err = self._validate_open(coin, ctx, decision)
            if sig is None:
                return self._error_result(coin, ctx, call, reasoning, confidence,
                                          decision, err or "invalid_open")
            return AIDecisionResult(
                coin=coin, action=action, signal=sig,
                reasoning=reasoning, confidence=confidence,
                raw_decision=decision, cost_usd=call.cost_usd,
                latency_ms=call.latency_ms, model=call.model,
            )

        if action in {"close", "hold", "move_stop_to_breakeven"}:
            # These actions need an open position (close + move_be) or none (hold);
            # the bot.py handler decides what to do with each.
            return AIDecisionResult(
                coin=coin, action=action, signal=None,
                reasoning=reasoning, confidence=confidence,
                raw_decision=decision, cost_usd=call.cost_usd,
                latency_ms=call.latency_ms, model=call.model,
            )

        if action == "modify_stop":
            new_stop = _coerce_optional_float(decision.get("new_stop_price"))
            if new_stop is None:
                return self._error_result(coin, ctx, call, reasoning, confidence,
                                          decision, "missing_new_stop_price")
            return AIDecisionResult(
                coin=coin, action=action, signal=None,
                reasoning=reasoning, confidence=confidence,
                raw_decision=decision, cost_usd=call.cost_usd,
                latency_ms=call.latency_ms, model=call.model,
                new_stop_price=new_stop,
            )

        if action == "scale_out":
            frac = _coerce_optional_float(decision.get("scale_fraction"))
            if frac is None or not (0.0 < frac < 1.0):
                return self._error_result(coin, ctx, call, reasoning, confidence,
                                          decision, "invalid_scale_fraction")
            return AIDecisionResult(
                coin=coin, action=action, signal=None,
                reasoning=reasoning, confidence=confidence,
                raw_decision=decision, cost_usd=call.cost_usd,
                latency_ms=call.latency_ms, model=call.model,
                scale_fraction=frac,
            )

        if action == "add_to_position":
            sig, err = self._validate_open(coin, ctx, decision)
            if sig is None:
                return self._error_result(coin, ctx, call, reasoning, confidence,
                                          decision, err or "invalid_add")
            qf = _coerce_optional_float(decision.get("add_qty_fraction"))
            if qf is None or not (0.0 < qf <= 1.0):
                return self._error_result(coin, ctx, call, reasoning, confidence,
                                          decision, "invalid_add_qty_fraction")
            return AIDecisionResult(
                coin=coin, action=action, signal=sig,
                reasoning=reasoning, confidence=confidence,
                raw_decision=decision, cost_usd=call.cost_usd,
                latency_ms=call.latency_ms, model=call.model,
                add_qty_fraction=qf,
            )

        return self._error_result(coin, ctx, call, reasoning, confidence,
                                  decision, f"unknown_action:{action!r}")

    def _error_result(
        self,
        coin: str, ctx: CoinContext, call: AICallResult,
        reasoning: str, confidence: float, decision: dict[str, Any], error: str,
    ) -> AIDecisionResult:
        return AIDecisionResult(
            coin=coin, action="error", signal=None,
            reasoning=reasoning, confidence=confidence,
            raw_decision=decision, cost_usd=call.cost_usd,
            latency_ms=call.latency_ms, model=call.model,
            error=error,
        )

    def _validate_open(
        self,
        coin: str,
        ctx: CoinContext,
        decision: dict[str, Any],
    ) -> tuple[SweepSignal | None, str | None]:
        action = decision["action"]
        side = Side.LONG if action == "open_long" else Side.SHORT
        last_price = ctx.last_price
        if last_price <= 0:
            return None, "no_last_price"

        stop_raw = decision.get("stop_price")
        if stop_raw is None:
            return None, "missing_stop_price"
        try:
            stop_price = float(stop_raw)
        except (TypeError, ValueError):
            return None, "non_numeric_stop"

        # Stop must be on the LOSING side of entry.
        if side == Side.LONG and stop_price >= last_price:
            return None, "stop_not_below_entry_for_long"
        if side == Side.SHORT and stop_price <= last_price:
            return None, "stop_not_above_entry_for_short"

        stop_distance_bps = abs(last_price - stop_price) / last_price * 10000
        # Prompt says 8-80 bps; enforce here so the AI can't sneak in a 1bp stop.
        if not (8 <= stop_distance_bps <= 80):
            return None, f"stop_distance_out_of_range:{stop_distance_bps:.1f}bps"

        # TPs (optional). If supplied, validate side.
        tp1_raw = decision.get("tp1_price")
        tp2_raw = decision.get("tp2_price")
        tp1 = _coerce_optional_float(tp1_raw)
        tp2 = _coerce_optional_float(tp2_raw)

        if tp1 is not None:
            if side == Side.LONG and tp1 <= last_price:
                return None, "tp1_not_above_entry_for_long"
            if side == Side.SHORT and tp1 >= last_price:
                return None, "tp1_not_below_entry_for_short"
        if tp2 is not None:
            if side == Side.LONG and tp2 <= last_price:
                return None, "tp2_not_above_entry_for_long"
            if side == Side.SHORT and tp2 >= last_price:
                return None, "tp2_not_below_entry_for_short"

        # Provide sensible defaults if AI omits TPs: 2R and 4R for a "swing" hold.
        risk_dist = abs(last_price - stop_price)
        if tp1 is None:
            tp1 = last_price + (2.0 * risk_dist if side == Side.LONG else -2.0 * risk_dist)
        if tp2 is None:
            tp2 = last_price + (4.0 * risk_dist if side == Side.LONG else -4.0 * risk_dist)

        # Round prices to HL-compatible tick (5 sig figs) so the executor's
        # downstream rounding doesn't drift the AI's intended levels.
        entry_px = _round_to_sig_figs(last_price)
        stop_px = _round_to_sig_figs(stop_price)
        tp1_px = _round_to_sig_figs(tp1)
        tp2_px = _round_to_sig_figs(tp2)

        sig = SweepSignal(
            side=side,
            level=entry_px,
            level_label="ai_decision",
            sweep_extreme=entry_px,
            entry_price=entry_px,
            stop_price=stop_px,
            tp1_price=tp1_px,
            tp2_price=tp2_px,
            confidence=max(ctx.last_spread_bps and 0.5 or 0.5, float(decision.get("confidence", 0.5))),
            reason="ai:" + (str(decision.get("reasoning", ""))[:80].replace("\n", " ")),
            created_ms=ctx.now_ms,
            coin=coin,
        )
        return sig, None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out <= 0:
        return None
    return out
