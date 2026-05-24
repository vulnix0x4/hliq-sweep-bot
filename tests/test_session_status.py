from datetime import datetime, timezone

import scripts.session_status as session_status


def test_session_boundaries_are_utc_hour_based():
    assert session_status._session_from_hour(0) == "asia"
    assert session_status._session_from_hour(6) == "asia"
    assert session_status._session_from_hour(7) == "eu"
    assert session_status._session_from_hour(12) == "eu"
    assert session_status._session_from_hour(13) == "us"
    assert session_status._session_from_hour(21) == "us"
    assert session_status._session_from_hour(22) == "late"


def test_next_allowed_rounds_to_next_allowed_utc_hour():
    now = datetime(2026, 5, 12, 0, 42, tzinfo=timezone.utc)

    nxt = session_status._next_allowed(now, {"us"})

    assert nxt == datetime(2026, 5, 12, 13, 0, tzinfo=timezone.utc)


def test_next_allowed_returns_now_when_current_session_allowed():
    now = datetime(2026, 5, 12, 7, 42, tzinfo=timezone.utc)

    nxt = session_status._next_allowed(now, {"eu", "us"})

    assert nxt == now
