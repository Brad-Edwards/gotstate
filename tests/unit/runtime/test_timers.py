# tests/unit/test_timers.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import time
from unittest.mock import patch

import pytest


def test_timer():
    from gotstate.runtime.timers import Timer

    now = time.time()
    t = Timer(deadline=now + 0.5)
    assert t.deadline == pytest.approx(now + 0.5)
    assert t.is_expired() is False

    # Simulate time passing by mocking time.time()
    with patch("time.time", return_value=now + 1.0):
        assert t.is_expired() is True


def test_timeout_scheduler():
    from gotstate.core.events import TimeoutEvent
    from gotstate.runtime.timers import TimeoutScheduler

    sched = TimeoutScheduler()
    now = time.time()
    e = TimeoutEvent(name="TestTimeout", deadline=now + 0.1)
    sched.schedule_timeout(e)

    # Initially no expired
    assert len(sched.check_timeouts()) == 0

    # After time passes
    with patch("time.time", return_value=now + 0.2):
        expired = sched.check_timeouts()
        assert len(expired) == 1
        assert expired[0].name == "TestTimeout"
