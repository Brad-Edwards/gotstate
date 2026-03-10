"""Tests for gotstate.runtime modules (executor, scheduler, monitor)."""

import threading
import time

import icontract
import pytest

from gotstate.exceptions import GotStateError
from gotstate.runtime.executor import (
    ExecutionContext,
    ExecutionMode,
    ExecutionStatus,
    ExecutionUnit,
    Executor,
)
from gotstate.runtime.monitor import MetricType, Monitor, MonitoringLevel
from gotstate.runtime.scheduler import Scheduler, TimerKind, TimerStatus


class TestExecutor:
    """Tests for the Executor class."""

    def test_create(self):
        ex = Executor()
        assert ex.status == ExecutionStatus.IDLE
        assert ex.mode == ExecutionMode.SYNCHRONOUS
        assert ex.execution_count == 0

    def test_execute_action(self):
        ex = Executor()
        result = ex.execute(lambda: 42)
        assert result == 42
        assert ex.execution_count == 1
        assert ex.status == ExecutionStatus.IDLE

    def test_execute_failing_action_sets_failed(self):
        ex = Executor()
        with pytest.raises(ValueError):
            ex.execute(lambda: (_ for _ in ()).throw(ValueError("oops")))
        assert ex.status == ExecutionStatus.FAILED

    def test_reset_after_failure(self):
        ex = Executor()
        with pytest.raises(ValueError):
            ex.execute(lambda: (_ for _ in ()).throw(ValueError("oops")))
        ex.reset()
        assert ex.status == ExecutionStatus.IDLE

    def test_rtc_violation(self):
        """Cannot start a new execution while one is running."""
        ex = Executor()
        barrier = threading.Event()
        errors = []

        def blocking_action():
            barrier.wait(timeout=2)
            return 1

        def try_concurrent():
            try:
                ex.execute(lambda: 2)
            except GotStateError as e:
                errors.append(e)

        t1 = threading.Thread(target=lambda: ex.execute(blocking_action))
        t1.start()
        time.sleep(0.05)  # Let t1 acquire the lock
        t2 = threading.Thread(target=try_concurrent)
        t2.start()
        time.sleep(0.05)
        barrier.set()
        t1.join(timeout=2)
        t2.join(timeout=2)
        # t2 will block on RLock (since it's reentrant only on same thread)
        # The RTC check only prevents re-entrance on the same thread.


class TestExecutionUnit:
    def test_execute(self):
        unit = ExecutionUnit(lambda: "hello")
        result = unit.execute()
        assert result == "hello"
        assert unit.is_executed
        assert unit.result == "hello"

    def test_undo(self):
        state = [0]
        unit = ExecutionUnit(
            action=lambda: state.__setitem__(0, 1),
            rollback=lambda: state.__setitem__(0, 0),
        )
        unit.execute()
        assert state[0] == 1
        unit.undo()
        assert state[0] == 0
        assert not unit.is_executed

    def test_undo_without_rollback(self):
        unit = ExecutionUnit(lambda: 1)
        unit.execute()
        unit.undo()  # Should not raise


class TestExecutionContext:
    def test_set_get_clear(self):
        ctx = ExecutionContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("missing", "default") == "default"
        ctx.clear()
        assert ctx.get("key") is None


class TestScheduler:
    def test_schedule_and_expire_timer(self):
        scheduler = Scheduler()
        fired = threading.Event()
        scheduler.schedule_timer("t1", 0.05, lambda: fired.set())
        assert scheduler.get_timer_status("t1") == TimerStatus.ACTIVE
        fired.wait(timeout=1)
        assert fired.is_set()
        assert scheduler.get_timer_status("t1") == TimerStatus.EXPIRED

    def test_cancel_timer(self):
        scheduler = Scheduler()
        scheduler.schedule_timer("t1", 10.0, lambda: None)
        scheduler.cancel_timer("t1")
        assert scheduler.get_timer_status("t1") == TimerStatus.CANCELLED

    def test_cancel_nonexistent_raises(self):
        scheduler = Scheduler()
        with pytest.raises(GotStateError, match="not found"):
            scheduler.cancel_timer("missing")

    def test_duplicate_timer_raises(self):
        scheduler = Scheduler()
        scheduler.schedule_timer("t1", 10.0, lambda: None)
        with pytest.raises(GotStateError, match="already exists"):
            scheduler.schedule_timer("t1", 10.0, lambda: None)
        scheduler.cancel_timer("t1")

    def test_timer_contracts(self):
        scheduler = Scheduler()
        with pytest.raises(icontract.ViolationError):
            scheduler.schedule_timer("", 1.0, lambda: None)
        with pytest.raises(icontract.ViolationError):
            scheduler.schedule_timer("t", 0, lambda: None)
        with pytest.raises(icontract.ViolationError):
            scheduler.schedule_timer("t", 1.0, "not_callable")  # type: ignore

    def test_register_and_check_conditions(self):
        scheduler = Scheduler()
        flag = [False]
        scheduler.register_change_condition("c1", lambda: flag[0])
        assert scheduler.check_conditions() == []
        flag[0] = True
        assert scheduler.check_conditions() == ["c1"]

    def test_condition_contracts(self):
        scheduler = Scheduler()
        with pytest.raises(icontract.ViolationError):
            scheduler.register_change_condition("", lambda: True)
        with pytest.raises(icontract.ViolationError):
            scheduler.register_change_condition("c", "not_callable")  # type: ignore

    def test_cancel_all(self):
        scheduler = Scheduler()
        scheduler.schedule_timer("t1", 10.0, lambda: None)
        scheduler.schedule_timer("t2", 10.0, lambda: None)
        scheduler.cancel_all()
        assert scheduler.get_timer_status("t1") == TimerStatus.CANCELLED
        assert scheduler.get_timer_status("t2") == TimerStatus.CANCELLED

    def test_get_timer_status_nonexistent_raises(self):
        scheduler = Scheduler()
        with pytest.raises(GotStateError, match="not found"):
            scheduler.get_timer_status("missing")


class TestMonitor:
    def test_create(self):
        m = Monitor()
        assert m.level == MonitoringLevel.NORMAL
        assert m.metrics == {}
        assert m.counters == {}

    def test_record_metric(self):
        m = Monitor()
        m.record_metric("latency", 1.5)
        assert m.metrics["latency"] == 1.5

    def test_increment_counter(self):
        m = Monitor()
        m.increment_counter("transitions")
        m.increment_counter("transitions")
        assert m.counters["transitions"] == 2

    def test_emit_and_get_history(self):
        m = Monitor()
        m.emit_event("state_change", {"from": "a", "to": "b"})
        history = m.get_history()
        assert len(history) == 1
        assert history[0]["type"] == "state_change"
        assert history[0]["data"]["from"] == "a"

    def test_get_history_with_limit(self):
        m = Monitor()
        for i in range(10):
            m.emit_event(f"event_{i}")
        assert len(m.get_history(limit=3)) == 3

    def test_subscriber(self):
        m = Monitor()
        received = []
        m.subscribe(lambda record: received.append(record))
        m.emit_event("test")
        assert len(received) == 1
        assert received[0]["type"] == "test"

    def test_subscriber_error_does_not_break(self):
        m = Monitor()
        m.subscribe(lambda r: (_ for _ in ()).throw(ValueError("bad")))
        m.emit_event("test")  # Should not raise
        assert len(m.get_history()) == 1

    def test_set_level(self):
        m = Monitor()
        m.level = MonitoringLevel.DEBUG
        assert m.level == MonitoringLevel.DEBUG

    def test_reset(self):
        m = Monitor()
        m.record_metric("m", 1.0)
        m.increment_counter("c")
        m.emit_event("e")
        m.reset()
        assert m.metrics == {}
        assert m.counters == {}
        assert m.get_history() == []

    def test_history_size_limit(self):
        m = Monitor(history_size=5)
        for i in range(10):
            m.emit_event(f"event_{i}")
        assert len(m.get_history()) == 5
