"""Tests for gotstate.core.transition module."""

import icontract
import pytest

from gotstate.core.event import Event, EventKind
from gotstate.core.state import State, StateType
from gotstate.core.transition import (
    ChangeTransition,
    CompoundTransition,
    ExternalTransition,
    InternalTransition,
    LocalTransition,
    ProtocolTransition,
    TimeTransition,
    Transition,
    TransitionKind,
    TransitionPriority,
)
from gotstate.exceptions import GuardError, InvalidTransitionError


class TestTransition:
    """Tests for the base Transition class."""

    def test_create_transition(self):
        s1 = State("s1")
        s2 = State("s2")
        t = Transition(s1, s2)
        assert t.source is s1
        assert t.target is s2
        assert t.kind == TransitionKind.EXTERNAL
        assert t.guard is None
        assert t.action is None
        assert t.trigger is None
        assert t.priority == TransitionPriority.NORMAL

    def test_source_required_contract(self):
        with pytest.raises(icontract.ViolationError):
            Transition(None, State("s"))  # type: ignore

    def test_kind_contract(self):
        with pytest.raises(icontract.ViolationError):
            Transition(State("s"), State("t"), kind="bad")  # type: ignore

    def test_is_enabled_no_trigger_no_guard(self):
        t = Transition(State("s"), State("t"))
        assert t.is_enabled(None)
        assert not t.is_enabled(Event("ev", EventKind.SIGNAL))

    def test_is_enabled_with_trigger(self):
        t = Transition(State("s"), State("t"), trigger="click")
        e_match = Event("click", EventKind.SIGNAL)
        e_no_match = Event("other", EventKind.SIGNAL)
        assert t.is_enabled(e_match)
        assert not t.is_enabled(e_no_match)
        assert not t.is_enabled(None)

    def test_is_enabled_with_guard_true(self):
        t = Transition(State("s"), State("t"), guard=lambda: True, trigger="ev")
        assert t.is_enabled(Event("ev", EventKind.SIGNAL))

    def test_is_enabled_with_guard_false(self):
        t = Transition(State("s"), State("t"), guard=lambda: False, trigger="ev")
        assert not t.is_enabled(Event("ev", EventKind.SIGNAL))

    def test_guard_error_raises(self):
        def bad_guard():
            raise ValueError("oops")

        t = Transition(State("s"), State("t"), guard=bad_guard, trigger="ev")
        with pytest.raises(GuardError, match="Guard evaluation failed"):
            t.is_enabled(Event("ev", EventKind.SIGNAL))

    def test_execute_external(self):
        s1 = State("s1")
        s2 = State("s2")
        actions = []
        t = Transition(s1, s2, action=lambda: actions.append("action"))
        s1.enter()
        assert s1.is_active

        t.execute()
        assert not s1.is_active
        assert s2.is_active
        assert actions == ["action"]

    def test_execute_internal(self):
        s = State("s")
        actions = []
        t = Transition(s, s, kind=TransitionKind.INTERNAL, action=lambda: actions.append("action"))
        s.enter()

        t.execute()
        assert s.is_active  # No exit/entry for internal
        assert actions == ["action"]

    def test_priority_ordering(self):
        high = Transition(State("s"), State("t"), priority=TransitionPriority.HIGH)
        low = Transition(State("s"), State("t"), priority=TransitionPriority.LOW)
        assert high < low

    def test_repr(self):
        t = Transition(State("a"), State("b"), trigger="click")
        assert "a" in repr(t)
        assert "b" in repr(t)
        assert "click" in repr(t)


class TestExternalTransition:
    def test_create(self):
        t = ExternalTransition(State("s"), State("t"))
        assert t.kind == TransitionKind.EXTERNAL

    def test_target_required(self):
        with pytest.raises(InvalidTransitionError):
            ExternalTransition(State("s"), None)  # type: ignore


class TestInternalTransition:
    def test_create(self):
        s = State("s")
        t = InternalTransition(s)
        assert t.kind == TransitionKind.INTERNAL
        assert t.source is s
        assert t.target is s

    def test_no_state_exit_entry(self):
        s = State("s")
        entered = []
        exited = []
        s.on_entry(lambda: entered.append(True))
        s.on_exit(lambda: exited.append(True))
        s.enter()
        entered.clear()

        t = InternalTransition(s, action=lambda: None)
        t.execute()
        assert s.is_active
        assert len(entered) == 0
        assert len(exited) == 0


class TestLocalTransition:
    def test_create(self):
        t = LocalTransition(State("s"), State("t"))
        assert t.kind == TransitionKind.LOCAL


class TestCompoundTransition:
    def test_create(self):
        s = State("s")
        t = State("t")
        ct = CompoundTransition(s, t)
        assert ct.kind == TransitionKind.COMPOUND
        assert ct.segments == []

    def test_add_and_execute_segments(self):
        s1 = State("s1")
        s2 = State("s2")
        s3 = State("s3")
        seg1 = Transition(s1, s2)
        seg2 = Transition(s2, s3)
        ct = CompoundTransition(s1, s3, segments=[seg1, seg2])
        assert len(ct.segments) == 2

        s1.enter()
        ct.execute()


class TestProtocolTransition:
    def test_pre_condition_blocks(self):
        s1 = State("s1")
        s2 = State("s2")
        t = ProtocolTransition(s1, s2, pre_condition=lambda: False)
        assert not t.is_enabled()

    def test_post_condition_violation(self):
        s1 = State("s1")
        s2 = State("s2")
        s1.enter()
        t = ProtocolTransition(s1, s2, post_condition=lambda: False)
        with pytest.raises(InvalidTransitionError, match="post-condition"):
            t.execute()


class TestTimeTransition:
    def test_create(self):
        t = TimeTransition(State("s"), State("t"), duration=5.0)
        assert t.duration == 5.0

    def test_positive_duration_contract(self):
        with pytest.raises(icontract.ViolationError):
            TimeTransition(State("s"), State("t"), duration=0)
        with pytest.raises(icontract.ViolationError):
            TimeTransition(State("s"), State("t"), duration=-1)


class TestChangeTransition:
    def test_create(self):
        cond = lambda: True
        t = ChangeTransition(State("s"), State("t"), condition=cond)
        assert t.condition is cond

    def test_non_callable_condition_contract(self):
        with pytest.raises(icontract.ViolationError):
            ChangeTransition(State("s"), State("t"), condition="not_callable")  # type: ignore
