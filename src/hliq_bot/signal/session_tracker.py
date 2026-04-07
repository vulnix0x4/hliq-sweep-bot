"""Track trading sessions (Asia/EU/US/Late) and prior-day levels."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from hliq_bot.models import Bar


class SessionTracker:
    """Tracks session opens, highs/lows, and prior-day high/low (PDH/PDL).

    Sessions are defined by UTC hour:
        Asia : 0-6   (0 <= hour < 7)
        EU   : 7-12  (7 <= hour < 13)
        US   : 13-21 (13 <= hour < 22)
        Late : 22-23 (22 <= hour < 24)

    Days are UTC days (00:00-00:00).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        # Current session state
        self.current_session: Optional[str] = None
        self.current_session_open: Optional[float] = None
        self._current_session_high: Optional[float] = None
        self._current_session_low: Optional[float] = None

        # Prior session state
        self.prior_session: Optional[str] = None
        self.prior_session_open: Optional[float] = None
        self.prior_session_high: Optional[float] = None
        self.prior_session_low: Optional[float] = None

        # Current day tracking
        self._current_day: Optional[int] = None  # ordinal
        self._current_day_high: Optional[float] = None
        self._current_day_low: Optional[float] = None

        # Prior day levels (PDH / PDL)
        self.prior_day_high: Optional[float] = None
        self.prior_day_low: Optional[float] = None

    # ------------------------------------------------------------------
    # Session classification
    # ------------------------------------------------------------------
    @staticmethod
    def _session_from_hour(hour: int) -> str:
        if hour < 7:
            return "asia"
        if hour < 13:
            return "eu"
        if hour < 22:
            return "us"
        return "late"

    # ------------------------------------------------------------------
    # Bar ingestion
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        dt = datetime.fromtimestamp(bar.start_ms / 1000.0, tz=timezone.utc)
        day_ordinal = dt.toordinal()
        session = self._session_from_hour(dt.hour)

        # --- Day rollover ---
        if self._current_day is not None and day_ordinal != self._current_day:
            self.prior_day_high = self._current_day_high
            self.prior_day_low = self._current_day_low
            self._current_day_high = None
            self._current_day_low = None

        # --- Session rollover ---
        if self.current_session is not None and session != self.current_session:
            self.prior_session = self.current_session
            self.prior_session_open = self.current_session_open
            self.prior_session_high = self._current_session_high
            self.prior_session_low = self._current_session_low
            self.current_session = session
            self.current_session_open = bar.open
            self._current_session_high = bar.high
            self._current_session_low = bar.low
        elif self.current_session is None:
            # First bar ever
            self.current_session = session
            self.current_session_open = bar.open
            self._current_session_high = bar.high
            self._current_session_low = bar.low
        else:
            # Same session -- update running high/low
            if bar.high > self._current_session_high:  # type: ignore[operator]
                self._current_session_high = bar.high
            if bar.low < self._current_session_low:  # type: ignore[operator]
                self._current_session_low = bar.low

        # --- Day high/low tracking ---
        self._current_day = day_ordinal
        if self._current_day_high is None or bar.high > self._current_day_high:
            self._current_day_high = bar.high
        if self._current_day_low is None or bar.low < self._current_day_low:
            self._current_day_low = bar.low

    # ------------------------------------------------------------------
    # Level export
    # ------------------------------------------------------------------
    def get_levels(
        self,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Return ``(short_levels, long_levels)`` tuples of ``(label, price)``.

        short_levels: pdh, session_open_current, session_open_prior, prior_session_high
        long_levels:  pdl, session_open_current, session_open_prior, prior_session_low
        """
        if self.current_session_open is None:
            return ([], [])

        short_levels: list[tuple[str, float]] = []
        long_levels: list[tuple[str, float]] = []

        # PDH / PDL
        if self.prior_day_high is not None:
            short_levels.append(("pdh", self.prior_day_high))
        if self.prior_day_low is not None:
            long_levels.append(("pdl", self.prior_day_low))

        # Current session open (both sides)
        short_levels.append(("session_open_current", self.current_session_open))
        long_levels.append(("session_open_current", self.current_session_open))

        # Prior session open (both sides)
        if self.prior_session_open is not None:
            short_levels.append(("session_open_prior", self.prior_session_open))
            long_levels.append(("session_open_prior", self.prior_session_open))

        # Prior session high / low
        if self.prior_session_high is not None:
            short_levels.append(("prior_session_high", self.prior_session_high))
        if self.prior_session_low is not None:
            long_levels.append(("prior_session_low", self.prior_session_low))

        return (short_levels, long_levels)
