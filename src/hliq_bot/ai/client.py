"""OpenRouter / OpenAI-compatible chat-completions client.

Sync, stdlib-only (no new deps). Used by the AI trading strategy to ask an
LLM "what should we do on this coin right now?".

Why sync: the AI strategy runs on a slow (300s default) timer, so blocking
HTTP is fine and avoids dragging asyncio into the call site. The bot already
runs the AI loop in its own thread.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class AIClientError(RuntimeError):
    """Raised when the AI provider returns a non-recoverable error."""


@dataclass(slots=True)
class AICallResult:
    """One round-trip with the LLM. `decision` is parsed JSON when the model
    obeyed the schema; falls back to raw text in `raw_text` for diagnostics."""
    decision: dict[str, Any] | None
    raw_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    finish_reason: str


# OpenRouter pricing snapshot (USD per 1M tokens, sourced 2026-05).
# Used only for budget tracking — actual billing is via OpenRouter's invoice.
# Missing models bill at 0 here (no warning triggered) but you still pay OR;
# add entries when you know the rate, otherwise the budget gate is inert.
_PRICING_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # model: (input_price, output_price)
    # gemini 3.5 flash just released — pricing unconfirmed; update when
    # known so the daily-budget tracker isn't flying blind.
    "google/gemini-3.5-flash":       (0.10,  0.40),  # estimate; verify on OpenRouter
    "google/gemini-2.5-flash":       (0.075, 0.30),
    "google/gemini-2.5-pro":         (1.25,  5.00),
    "anthropic/claude-haiku-4.5":    (1.00,  5.00),
    "anthropic/claude-sonnet-4.6":   (3.00, 15.00),
    "anthropic/claude-opus-4.7":     (15.00, 75.00),
    "openai/gpt-5":                  (1.25, 10.00),
    "openai/gpt-5-mini":             (0.25,  2.00),
    "meta-llama/llama-3.3-70b":      (0.20,  0.60),
}


def _cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = _PRICING_PER_M_TOKENS.get(model)
    if price is None:
        return 0.0
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000.0


class OpenRouterClient:
    """Thin OpenAI-compatible chat-completions client targeting OpenRouter.

    Usage:
        client = OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"],
                                  model="google/gemini-2.5-flash")
        result = client.chat_json(
            system="You are a trading agent...",
            user="Context: {...}",
            schema={"type": "object", "properties": {...}, "required": [...]},
            timeout_sec=30.0,
        )
        decision = result.decision
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        # OpenRouter wants these headers to attribute usage to your app.
        # No PII — these only need to identify your project for OR's
        # leaderboard, they don't affect billing or routing.
        http_referer: str = "https://github.com/local/hliq-sweep-bot",
        x_title: str = "hliq-sweep-bot",
    ) -> None:
        if not api_key:
            raise AIClientError("OPENROUTER_API_KEY is empty — set it in .env")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http_referer = http_referer
        self._x_title = x_title

    @property
    def model(self) -> str:
        return self._model

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_sec: float = 30.0,
    ) -> AICallResult:
        """Send a chat-completion request that expects a JSON-shaped response.

        When `schema` is provided, we ask the model to emit JSON conforming to
        it via OpenAI's response_format=json_schema. Most modern models on
        OpenRouter honor this; for models that don't, we fall back to parsing
        the raw text as JSON (which is why temperature is kept low).
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "trade_decision",
                    "strict": True,
                    "schema": schema,
                },
            }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._http_referer,
                "X-Title": self._x_title,
            },
            method="POST",
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise AIClientError(
                f"OpenRouter HTTP {exc.code}: {err_body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AIClientError(f"OpenRouter network error: {exc}") from exc
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AIClientError(
                f"OpenRouter returned non-JSON ({len(raw)}B): {raw[:200]}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise AIClientError(f"OpenRouter response has no choices: {data}")
        message = choices[0].get("message") or {}
        text = str(message.get("content", "") or "")
        finish = str(choices[0].get("finish_reason", "") or "")
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = _cost_for(self._model, prompt_tokens, completion_tokens)

        decision: dict[str, Any] | None = None
        if text:
            try:
                decision = json.loads(text)
            except json.JSONDecodeError:
                # Some models wrap JSON in ```json ... ``` fences when schema
                # isn't honored. Try to extract.
                stripped = text.strip()
                if stripped.startswith("```"):
                    inner = stripped.strip("`")
                    if inner.startswith("json\n"):
                        inner = inner[len("json\n"):]
                    if inner.endswith("```"):
                        inner = inner[: -len("```")]
                    try:
                        decision = json.loads(inner.strip())
                    except json.JSONDecodeError:
                        decision = None

        return AICallResult(
            decision=decision,
            raw_text=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=finish,
        )


class CostBudget:
    """Rolling-window USD budget tracker. Logs a warning when crossed but
    does NOT block calls — the AI strategy decides whether to soft-pause."""

    def __init__(self, daily_budget_usd: float) -> None:
        self._budget = max(0.0, daily_budget_usd)
        # (ts_ms, cost_usd) tuples; trimmed to a 24h window on every check.
        self._calls: list[tuple[int, float]] = []
        self._warned: bool = False

    def record(self, cost_usd: float, now_ms: int | None = None) -> None:
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        self._calls.append((now_ms, max(0.0, cost_usd)))

    def spent_last_24h(self, now_ms: int | None = None) -> float:
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - 24 * 60 * 60 * 1000
        self._calls = [(t, c) for t, c in self._calls if t >= cutoff]
        return sum(c for _, c in self._calls)

    def over_budget(self, now_ms: int | None = None) -> bool:
        if self._budget <= 0:
            return False
        spent = self.spent_last_24h(now_ms)
        over = spent > self._budget
        if over and not self._warned:
            log.warning(
                "AI daily budget exceeded: spent=$%.4f budget=$%.2f (24h rolling)",
                spent, self._budget,
            )
            self._warned = True
        elif not over and self._warned:
            self._warned = False  # re-arm if we drop back under
        return over
