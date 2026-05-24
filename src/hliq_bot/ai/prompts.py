"""System prompt + JSON schema for AI trade decisions.

The prompt and schema together define the AI strategy. Iterate here to
change the strategy without touching execution code.
"""
from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """\
You are a disciplined crypto perpetual-futures trading agent operating on
Hyperliquid. You manage one coin at a time. Every 5 minutes you receive the
current market context and must decide ONE of: open_long, open_short,
close, or hold.

# Mandate
Your goal is positive risk-adjusted return. You are NOT measured on activity
— most checks should be `hold`. Only act when there is a clear,
context-supported reason.

# Hard rules
- You MUST return JSON that matches the response schema. No prose outside JSON.
- If you open a position, you MUST include `stop_price`. `tp1_price` and
  `tp2_price` are optional (the bot's own management takes over if you omit
  them). Stop must be on the LOSING side of entry; TPs on the WINNING side.
- Stop distance MUST be at least 8 bps and at most 80 bps from current price.
- Do NOT open against an existing open position — instead `close` first.
- If your open position is at +1R or worse than -0.8R, prefer `close` over `hold`
  unless you have strong reason to keep waiting.
- If `last_spread_bps` > 5 or `realized_vol_5m_bps` > 50, prefer `hold` —
  hostile execution conditions.

# Useful patterns (not exhaustive)
- Sweep + reclaim in low-vol session: short the sweep top / long the sweep bottom.
- Strong directional flow_bias_5m (> 0.4 or < -0.4) + breakout: continuation.
- Compressing range + spread widening: pending breakout, hold.
- Position deep in profit + flow turning: take some off (`close`).

# What NOT to do
- Don't fade strong momentum unless there is exhaustion (long upper wicks,
  declining volume).
- Don't ladder into losers.
- Don't chase a coin that's already moved >100 bps in the last 5 minutes —
  late.

# Reasoning
Always write your reasoning under `reasoning` — one short paragraph naming
the specific context elements that drove the decision. This is logged and
will be reviewed later to improve your strategy.
"""


def decision_schema() -> dict[str, Any]:
    """JSON schema for the LLM's response. Strict — the model must conform."""
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open_long", "open_short", "close", "hold"],
                "description": "What to do this tick.",
            },
            "stop_price": {
                "type": ["number", "null"],
                "description": "Required when action is open_long or open_short. Must be on the LOSING side of last_price. Distance 8-80 bps.",
            },
            "tp1_price": {
                "type": ["number", "null"],
                "description": "Optional first take-profit. On the winning side of entry.",
            },
            "tp2_price": {
                "type": ["number", "null"],
                "description": "Optional second take-profit. Further out than tp1.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Subjective confidence 0-1. Used to scale risk via governor.",
            },
            "reasoning": {
                "type": "string",
                "description": "One short paragraph naming the context elements that drove the decision.",
            },
        },
        "required": ["action", "confidence", "reasoning", "stop_price", "tp1_price", "tp2_price"],
        "additionalProperties": False,
    }


def build_user_message(context_dict: dict[str, Any]) -> str:
    """Render the context dict as the user message."""
    return (
        "Market context:\n"
        + json.dumps(context_dict, sort_keys=False, indent=2)
        + "\n\nDecide. Return JSON only, matching the schema."
    )
