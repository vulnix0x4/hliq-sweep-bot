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
current market context and must decide ONE action.

# Actions available
- `hold` — do nothing this tick (most common; default when uncertain).
- `open_long` / `open_short` — only when flat. Requires `stop_price`; `tp1_price`
  and `tp2_price` optional (default 2R/4R if omitted).
- `close` — fully exit the current position now.
- `move_stop_to_breakeven` — set stop to entry price. Use after a meaningful
  favorable move to remove downside risk.
- `modify_stop` — change the current stop (must be closer to entry than current
  for a trailing tighten, or in the favorable direction). Set `new_stop_price`.
- `scale_out` — partial close. Set `scale_fraction` in (0, 1) for the fraction
  of CURRENT remaining size to close. Typical: 0.5 to take half off at a target.
- `add_to_position` — add to a winning position. Set `add_qty_fraction` of risk.
  ONLY allowed when current position is at >= +0.5R and pattern still valid.

# Mandate
Maximize risk-adjusted return. You are NOT measured on activity — most
checks should be `hold`. Most positions should be managed (trail, scale)
rather than closed-and-reopened, because fees and slippage compound.

# Hard rules
- Return JSON that matches the response schema. No prose outside JSON.
- Open positions: stop MUST be on the LOSING side; distance 8-80 bps.
- TPs (when supplied): on the WINNING side, tp2 further than tp1.
- modify_stop: new_stop_price must be a strict improvement (tighter, on
  the favorable side) — never widen a stop.
- scale_out: fraction strictly in (0, 1); never use to fully close (use `close`).
- Don't ladder into losers (no add_to_position when unrealized R < 0).
- If your open position is at +1.5R or worse than -0.8R, prefer `close` /
  `move_stop_to_breakeven` over `hold` unless you have a strong reason.
- If `last_spread_bps` > 5 or `realized_vol_5m_bps` > 50, prefer `hold`.
- Respect the portfolio view in context: don't stack more than 2 same-side
  positions across correlated alt-coins.

# Useful patterns
- Sweep + reclaim in low-vol session: fade the sweep direction with tight stop.
- Strong flow_bias_5m (|x| > 0.4) + breakout in same direction: continuation.
- Position at +1R: move_stop_to_breakeven to lock in.
- Position at +2R with weakening flow: scale_out 0.5.
- Position at +2R+ with continuing flow: modify_stop tighter, let it run.
- Funding strongly negative (e.g. < -0.0005) on a long-favored coin: caution
  on longs (paying funding) — prefer shorts or hold.

# What NOT to do
- Don't fade strong momentum without exhaustion signals (long wicks, vol drop).
- Don't add to losers.
- Don't chase a coin that's moved >100 bps in the last 5 minutes — late.
- Don't open a new trade when 2+ same-side positions are already open
  elsewhere — concentration risk.

# Reasoning — HARD LIMIT
`reasoning` MUST be one sentence, ≤ 15 words. Cite numbers only, no prose
(e.g. "flow=+0.62, broke 60.5, OI rising"). DO NOT write paragraphs. DO NOT
explain your reasoning step by step. Just the key numbers + your conclusion.
Responses over 15 words will be truncated.
"""


def decision_schema() -> dict[str, Any]:
    """JSON schema for the LLM's response. Strict — the model must conform."""
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "open_long", "open_short", "close", "hold",
                    "move_stop_to_breakeven", "modify_stop",
                    "scale_out", "add_to_position",
                ],
                "description": "What to do this tick.",
            },
            "stop_price": {
                "type": ["number", "null"],
                "description": "Required when action is open_long/open_short. Must be on the LOSING side of last_price. Distance 8-80 bps.",
            },
            "tp1_price": {
                "type": ["number", "null"],
                "description": "Optional first take-profit. On the winning side of entry.",
            },
            "tp2_price": {
                "type": ["number", "null"],
                "description": "Optional second take-profit. Further out than tp1.",
            },
            "new_stop_price": {
                "type": ["number", "null"],
                "description": "Required when action=modify_stop. Strict improvement only (closer to favorable side than current stop).",
            },
            "scale_fraction": {
                "type": ["number", "null"],
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Required when action=scale_out. Fraction of CURRENT remaining qty to close, in (0, 1).",
            },
            "add_qty_fraction": {
                "type": ["number", "null"],
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Required when action=add_to_position. Fraction of normal risk_dollars to add as a fresh entry on top.",
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
        "required": [
            "action", "confidence", "reasoning",
            "stop_price", "tp1_price", "tp2_price",
            "new_stop_price", "scale_fraction", "add_qty_fraction",
        ],
        "additionalProperties": False,
    }


def build_user_message(context_dict: dict[str, Any]) -> str:
    """Render the context dict as the user message."""
    return (
        "Market context:\n"
        + json.dumps(context_dict, sort_keys=False, indent=2)
        + "\n\nDecide. Return JSON only, matching the schema."
    )
