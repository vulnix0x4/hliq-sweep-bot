"""Daily VWAP tracker with UTC day-boundary reset."""

from __future__ import annotations

from hliq_bot.models import Bar

# Milliseconds in one UTC day.
_MS_PER_DAY = 86_400_000


class VWAPTracker:
    """Accumulates running VWAP and resets at UTC midnight.

    VWAP is a key institutional benchmark. Sweep-and-reject at VWAP
    provides high-probability mean-reversion entries because it is
    dual-sided: price can sweep above VWAP (short setup) or below
    (long setup).
    """

    def __init__(self) -> None:
        self._cum_pv: float = 0.0
        self._cum_vol: float = 0.0
        self._current_day: int = -1  # UTC day number of latest bar

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vwap(self) -> float:
        """Current running VWAP, or 0.0 if no data has been ingested."""
        if self._cum_vol == 0.0:
            return 0.0
        return self._cum_pv / self._cum_vol

    def on_bar(self, bar: Bar) -> None:
        """Accumulate a new bar into the running VWAP.

        Uses ``bar.vwap * bar.volume`` for accurate price-volume
        weighting.  Resets accumulators when the bar falls on a
        different UTC day than the previous bar.
        """
        day = bar.start_ms // _MS_PER_DAY

        if day != self._current_day:
            self._cum_pv = 0.0
            self._cum_vol = 0.0
            self._current_day = day

        self._cum_pv += bar.vwap * bar.volume
        self._cum_vol += bar.volume

    def get_levels(self) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Return ``(short_levels, long_levels)`` for sweep detection.

        VWAP is dual-sided: it appears in both short and long level
        lists with the label ``"vwap_daily"``.  Returns empty lists
        when no VWAP is available yet.
        """
        v = self.vwap
        if v == 0.0:
            return [], []

        short_levels: list[tuple[str, float]] = [("vwap_daily", v)]
        long_levels: list[tuple[str, float]] = [("vwap_daily", v)]
        return short_levels, long_levels
