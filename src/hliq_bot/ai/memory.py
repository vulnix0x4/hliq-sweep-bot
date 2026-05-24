"""Disk-backed AI trading memory.

The LLM has no inherent memory between calls. Without persistence, each
restart starts from zero context — the AI doesn't know which of its own
recent trades won or lost, can't notice it's been making the same mistake
for hours, and can't build on what worked last week.

This module gives the AI a rolling journal of its own decisions+outcomes,
persisted to disk and reloaded on boot. The most recent N entries are
injected into every prompt as `recent_outcomes` so the AI has continuity.

Storage: `runtime/ai_memory.jsonl` (one event per line). Both decision
events (when made) and outcome events (when the trade closes) are appended.
On load we replay the file to reconstruct the per-decision_id state map.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryEntry:
    """One AI trade in memory: decision + (eventually) its outcome."""
    decision_id: str
    ts_ms: int                  # decision time
    coin: str
    action: str                 # open_long | open_short | close | modify_stop | ...
    reasoning: str
    confidence: float
    stop_price: float | None = None
    tp1_price: float | None = None
    tp2_price: float | None = None
    entry_price: float | None = None
    # Filled in when the corresponding trade closes:
    outcome_ts_ms: int | None = None
    outcome_exit_reason: str | None = None
    outcome_pnl: float | None = None
    outcome_r_multiple: float | None = None
    outcome_hold_sec: float | None = None

    @property
    def resolved(self) -> bool:
        return self.outcome_pnl is not None

    def to_compact_prompt_dict(self) -> dict[str, Any]:
        """The compact form fed into the LLM's prompt — small to save tokens."""
        d: dict[str, Any] = {
            "coin": self.coin,
            "action": self.action,
            "conf": round(self.confidence, 2),
            "reason": (self.reasoning or "")[:120],
        }
        if self.resolved:
            d["r"] = round(self.outcome_r_multiple or 0.0, 3)
            d["pnl"] = round(self.outcome_pnl or 0.0, 4)
            d["exit"] = self.outcome_exit_reason
            d["hold_sec"] = int(self.outcome_hold_sec or 0)
        else:
            d["resolved"] = False
        return d


class AIMemory:
    """Rolling persistent memory of AI decisions + outcomes.

    Keeps the last `max_entries` decisions in memory and on disk. Resolved
    entries (with outcome) and unresolved (still-open) entries coexist;
    `recent_for_prompt` prefers resolved ones since they teach more.

    Thread safety: appends are atomic per-line writes; the in-memory deque
    is mutated only from the bot's event-worker thread.
    """

    def __init__(self, path: str | Path, *, max_entries: int = 50) -> None:
        self._path = Path(path)
        self._max = max(1, max_entries)
        # Keyed by decision_id so outcome updates can find the matching entry.
        self._by_id: dict[str, MemoryEntry] = {}
        # Insertion-ordered for "most recent N" queries. Allow some slack
        # above _max so trim has room to work without auto-evicting.
        self._order: deque[str] = deque(maxlen=max(self._max * 2, self._max + 5))
        self._loaded = False

    def load(self) -> int:
        """Read the on-disk journal and reconstruct memory. Idempotent."""
        if self._loaded:
            return len(self._by_id)
        if not self._path.exists():
            self._loaded = True
            return 0
        loaded = 0
        try:
            for raw in self._path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = row.get("kind")
                if kind == "decision":
                    entry = MemoryEntry(
                        decision_id=str(row["decision_id"]),
                        ts_ms=int(row.get("ts_ms", 0)),
                        coin=str(row.get("coin", "")),
                        action=str(row.get("action", "")),
                        reasoning=str(row.get("reasoning", "")),
                        confidence=float(row.get("confidence", 0.0) or 0.0),
                        stop_price=row.get("stop_price"),
                        tp1_price=row.get("tp1_price"),
                        tp2_price=row.get("tp2_price"),
                        entry_price=row.get("entry_price"),
                    )
                    self._by_id[entry.decision_id] = entry
                    if entry.decision_id not in self._order:
                        self._order.append(entry.decision_id)
                    loaded += 1
                elif kind == "outcome":
                    decision_id = str(row.get("decision_id", ""))
                    entry = self._by_id.get(decision_id)
                    if entry is None:
                        continue
                    entry.outcome_ts_ms = int(row.get("outcome_ts_ms", 0))
                    entry.outcome_exit_reason = row.get("outcome_exit_reason")
                    entry.outcome_pnl = row.get("outcome_pnl")
                    entry.outcome_r_multiple = row.get("outcome_r_multiple")
                    entry.outcome_hold_sec = row.get("outcome_hold_sec")
            self._trim_to_max()
            log.info("AI memory loaded: %d entries from %s", loaded, self._path)
        except OSError as exc:
            log.warning("AI memory load failed: %s — starting empty", exc)
        self._loaded = True
        return loaded

    def record_decision(self, entry: MemoryEntry) -> None:
        """Add a fresh decision. Persists to disk and trims the in-memory deque."""
        if not self._loaded:
            self.load()
        self._by_id[entry.decision_id] = entry
        if entry.decision_id in self._order:
            self._order.remove(entry.decision_id)
        self._order.append(entry.decision_id)
        self._trim_to_max()
        self._append_line({"kind": "decision", **{
            k: v for k, v in asdict(entry).items()
            if not k.startswith("outcome_")
        }})

    def record_outcome(
        self,
        decision_id: str,
        *,
        ts_ms: int,
        exit_reason: str,
        pnl: float,
        r_multiple: float,
        hold_sec: float,
    ) -> bool:
        """Attach an outcome to a previously-recorded decision.

        Returns True if a matching decision was found and updated.
        Useful when the bot closes a trade and wants to update memory.
        """
        if not self._loaded:
            self.load()
        entry = self._by_id.get(decision_id)
        if entry is None:
            return False
        entry.outcome_ts_ms = ts_ms
        entry.outcome_exit_reason = exit_reason
        entry.outcome_pnl = pnl
        entry.outcome_r_multiple = r_multiple
        entry.outcome_hold_sec = hold_sec
        self._append_line({
            "kind": "outcome",
            "decision_id": decision_id,
            "outcome_ts_ms": ts_ms,
            "outcome_exit_reason": exit_reason,
            "outcome_pnl": pnl,
            "outcome_r_multiple": r_multiple,
            "outcome_hold_sec": hold_sec,
        })
        return True

    def recent_for_prompt(self, *, coin: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Return up to `limit` recent entries in compact dict form for the LLM.

        Prefers resolved entries (where we know the outcome) since they teach
        more. If `coin` is given, restricts to that coin.
        """
        if not self._loaded:
            self.load()
        resolved: list[MemoryEntry] = []
        unresolved: list[MemoryEntry] = []
        for did in reversed(self._order):
            e = self._by_id.get(did)
            if e is None:
                continue
            if coin and e.coin != coin:
                continue
            if e.resolved:
                resolved.append(e)
            else:
                unresolved.append(e)
            if len(resolved) >= limit:
                break
        out = (resolved + unresolved)[:limit]
        return [e.to_compact_prompt_dict() for e in out]

    def summary_stats(self) -> dict[str, Any]:
        """High-level numbers for logging / dashboards."""
        if not self._loaded:
            self.load()
        resolved = [e for e in self._by_id.values() if e.resolved]
        wins = [e for e in resolved if (e.outcome_r_multiple or 0) > 0]
        total_pnl = sum((e.outcome_pnl or 0) for e in resolved)
        avg_r = (sum((e.outcome_r_multiple or 0) for e in resolved) / len(resolved)) if resolved else 0.0
        return {
            "total_decisions": len(self._by_id),
            "resolved_trades": len(resolved),
            "wins": len(wins),
            "win_rate": (len(wins) / len(resolved) * 100.0) if resolved else 0.0,
            "avg_r": round(avg_r, 3),
            "total_pnl": round(total_pnl, 4),
        }

    # ---- Internals ----

    def _append_line(self, payload: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError as exc:
            log.warning("AI memory append failed (%s): %s", self._path, exc)

    def _trim_to_max(self) -> None:
        """Drop oldest entries from the dict when over capacity."""
        while len(self._order) > self._max:
            old = self._order.popleft()
            self._by_id.pop(old, None)


def default_memory_path(runtime_dir: str) -> Path:
    return Path(runtime_dir) / "ai_memory.jsonl"


def now_ms() -> int:
    return int(time.time() * 1000)
