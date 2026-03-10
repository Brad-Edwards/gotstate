"""Tests for gotstate.core.machine module."""

import icontract
import pytest

from gotstate.core.event import Event, EventKind, SignalEvent
from gotstate.core.machine import (
    MachineBuilder,
    MachineModifier,
    MachineMonitor,
    MachineStatus,
    ProtocolMachine,
    StateMachine,
    SubmachineMachine,
)
from gotstate.core.state import State, StateType
from gotstate.core.transition import Transition, TransitionKind, TransitionPriority
from gotstate.exceptions import (
    MachineAlreadyRunningError,
    MachineError,
    MachineNotInitializedError,
)


class TestStateMachine:
    """Tests for the StateMachine class."""

    def _make_simple_machine(self):
        """Helper: create a machine with two states and a transition."""
        m = StateMachine("test")
        s1 = State("idle")
        s2 = State("active")
        m.add_state(s1)
        m.add_state(s2)
        m.set_initial_state(s1)
        t = Transition(s1, s2, trigger="go")
        m.add_transition(t)
        return m, s1, s2, t

    def test_create(self):
        m = StateMachine("test")
        assert m.name == "test"
        assert m.status == MachineStatus.UNINITIALIZED
        assert m.current_state is None
        assert m.states == {}
        assert m.transitions == []

    def test_name_contract(self):
        with pytest.raises(icontract.ViolationError):
            StateMachine("")

    def test_add_state(self):
        m = StateMachine("test")
        s = State("idle")
        m.add_state(s)
        assert "idle" in m.states

    def test_add_duplicate_state_raises(self):
        m = StateMachine("test")
        m.add_state(State("idle"))
        with pytest.raises(MachineError, match="already exists"):
            m.add_state(State("idle"))

    def test_set_initial_state_unregistered_raises(self):
        m = StateMachine("test")
        with pytest.raises(MachineError, match="not registered"):
            m.set_initial_state(State("foreign"))

    def test_start(self):
        m, s1, s2, _ = self._make_simple_machine()
        m.start()
        assert m.status == MachineStatus.ACTIVE
        assert m.current_state is s1
        assert s1.is_active

    def test_start_already_running_raises(self):
        m, *_ = self._make_simple_machine()
        m.start()
        with pytest.raises(MachineAlreadyRunningError):
            m.start()

    def test_start_without_initial_raises(self):
        m = StateMachine("test")
        m.add_state(State("s"))
        with pytest.raises(MachineNotInitializedError):
            m.start()

    def test_stop(self):
        m, s1, _, _ = self._make_simple_machine()
        m.start()
        m.stop()
        assert m.status == MachineStatus.TERMINATED
        assert m.current_state is None
        assert not s1.is_active

    def test_stop_when_not_active_is_noop(self):
        m = StateMachine("test")
        m.add_state(State("s"))
        m.stop()  # Should not raise

    def test_process_event_fires_transition(self):
        m, s1, s2, _ = self._make_simple_machine()
        m.start()
        event = SignalEvent("go")
        result = m.process_event(event)
        assert result is True
        assert m.current_state is s2
        assert s2.is_active
        assert event.is_consumed

    def test_process_event_no_match_returns_false(self):
        m, s1, s2, _ = self._make_simple_machine()
        m.start()
        result = m.process_event(SignalEvent("unknown"))
        assert result is False
        assert m.current_state is s1

    def test_process_event_when_not_active_raises(self):
        m, *_ = self._make_simple_machine()
        with pytest.raises(MachineNotInitializedError):
            m.process_event(SignalEvent("go"))

    def test_process_event_with_guard(self):
        m = StateMachine("test")
        s1, s2 = State("s1"), State("s2")
        m.add_state(s1)
        m.add_state(s2)
        m.set_initial_state(s1)
        allow = [False]
        t = Transition(s1, s2, trigger="go", guard=lambda: allow[0])
        m.add_transition(t)
        m.start()

        assert not m.process_event(SignalEvent("go"))
        assert m.current_state is s1

        allow[0] = True
        assert m.process_event(SignalEvent("go"))
        assert m.current_state is s2

    def test_process_event_priority_resolution(self):
        """Higher priority transitions should fire first."""
        m = StateMachine("test")
        s1, s2, s3 = State("s1"), State("s2"), State("s3")
        m.add_state(s1)
        m.add_state(s2)
        m.add_state(s3)
        m.set_initial_state(s1)
        # Low priority goes to s2
        m.add_transition(Transition(s1, s2, trigger="go", priority=TransitionPriority.LOW))
        # High priority goes to s3
        m.add_transition(Transition(s1, s3, trigger="go", priority=TransitionPriority.HIGH))
        m.start()

        m.process_event(SignalEvent("go"))
        assert m.current_state is s3  # High priority wins

    def test_hierarchical_transition_resolution(self):
        """Transitions on parent states should fire when no child transition matches."""
        m = StateMachine("test")
        parent = State("parent", StateType.COMPOSITE)
        child = State("child")
        parent.add_child(child)
        target = State("target")
        m.add_state(parent)
        m.add_state(child)
        m.add_state(target)
        m.set_initial_state(child)
        # Transition from parent, not from child
        m.add_transition(Transition(parent, target, trigger="escape"))
        m.start()
        assert m.current_state is child

        result = m.process_event(SignalEvent("escape"))
        assert result is True
        assert m.current_state is target

    def test_child_transition_takes_priority_over_parent(self):
        """Inner transitions should fire before parent transitions."""
        m = StateMachine("test")
        parent = State("parent", StateType.COMPOSITE)
        child = State("child")
        parent.add_child(child)
        t1, t2 = State("t1"), State("t2")
        m.add_state(parent)
        m.add_state(child)
        m.add_state(t1)
        m.add_state(t2)
        m.set_initial_state(child)
        # Both parent and child have "go" transition
        m.add_transition(Transition(parent, t1, trigger="go"))
        m.add_transition(Transition(child, t2, trigger="go"))
        m.start()

        m.process_event(SignalEvent("go"))
        assert m.current_state is t2  # Child transition wins

    def test_internal_transition_no_state_change(self):
        m = StateMachine("test")
        s1 = State("s1")
        m.add_state(s1)
        m.set_initial_state(s1)
        actions = []
        m.add_transition(
            Transition(s1, s1, kind=TransitionKind.INTERNAL, trigger="tick",
                       action=lambda: actions.append(1))
        )
        m.start()
        m.process_event(SignalEvent("tick"))
        assert m.current_state is s1
        assert actions == [1]

    def test_on_transition_callback(self):
        m, s1, s2, _ = self._make_simple_machine()
        fired = []
        m.on_transition(lambda t: fired.append(t))
        m.start()
        m.process_event(SignalEvent("go"))
        assert len(fired) == 1
        assert fired[0].target is s2

    def test_on_transition_callback_error_does_not_break(self):
        m, *_ = self._make_simple_machine()
        m.on_transition(lambda t: (_ for _ in ()).throw(ValueError("bad")))
        m.start()
        # Should not raise despite callback failure
        m.process_event(SignalEvent("go"))

    def test_get_state(self):
        m = StateMachine("test")
        s = State("idle")
        m.add_state(s)
        assert m.get_state("idle") is s

    def test_get_state_not_found_raises(self):
        m = StateMachine("test")
        with pytest.raises(MachineError, match="not found"):
            m.get_state("missing")

    def test_repr(self):
        m = StateMachine("test")
        assert "test" in repr(m)
        assert "UNINITIALIZED" in repr(m)

    def test_completion_transition(self):
        """Triggerless (completion) transitions fire when event is None-like."""
        m = StateMachine("test")
        s1, s2 = State("s1"), State("s2")
        m.add_state(s1)
        m.add_state(s2)
        m.set_initial_state(s1)
        # No trigger = completion transition
        m.add_transition(Transition(s1, s2))
        m.start()

        # Completion transitions need None event in process_event's logic
        # but process_event requires an Event. Test the transition directly.
        t = m.transitions[0]
        assert t.is_enabled(None)
        assert not t.is_enabled(SignalEvent("something"))


class TestProtocolMachine:
    def test_allow_and_check_operations(self):
        m = ProtocolMachine("proto")
        s1 = State("open")
        m.add_state(s1)
        m.set_initial_state(s1)
        m.allow_operation("open", "read")
        m.allow_operation("open", "write")
        m.start()
        assert m.is_operation_allowed("read")
        assert m.is_operation_allowed("write")
        assert not m.is_operation_allowed("close")

    def test_no_current_state_disallows(self):
        m = ProtocolMachine("proto")
        assert not m.is_operation_allowed("anything")


class TestSubmachineMachine:
    def test_entry_exit_points(self):
        m = SubmachineMachine("sub")
        s_in = State("in")
        s_out = State("out")
        m.add_entry_point("main", s_in)
        m.add_exit_point("done", s_out)
        assert m.entry_points == {"main": s_in}
        assert m.exit_points == {"done": s_out}


class TestMachineBuilder:
    def test_build_simple_machine(self):
        builder = MachineBuilder("test")
        s1 = builder.add_state("idle")
        s2 = builder.add_state("active")
        builder.set_initial(s1)
        builder.add_transition(s1, s2, trigger="go")
        machine = builder.build()

        assert machine.name == "test"
        assert "idle" in machine.states
        assert "active" in machine.states
        assert len(machine.transitions) == 1

    def test_build_twice_raises(self):
        builder = MachineBuilder("test")
        builder.add_state("s")
        builder.build()
        with pytest.raises(MachineError, match="already been built"):
            builder.build()

    def test_builder_chaining(self):
        builder = MachineBuilder("test")
        s = builder.add_state("s")
        result = builder.set_initial(s)
        assert result is builder


class TestMachineModifier:
    def test_stage_and_apply(self):
        m = StateMachine("test")
        s1 = State("s1")
        m.add_state(s1)
        m.set_initial_state(s1)

        modifier = MachineModifier(m)
        modifier.stage_add_state(State("s2"))
        modifier.stage_add_transition(Transition(s1, State("s2"), trigger="go"))
        modifier.apply()
        assert "s2" in m.states

    def test_rollback_clears_staged(self):
        m = StateMachine("test")
        modifier = MachineModifier(m)
        modifier.stage_add_state(State("s"))
        modifier.rollback()
        modifier.apply()  # Should do nothing
        assert "s" not in m.states


class TestMachineMonitor:
    def test_tracks_transitions(self):
        m = StateMachine("test")
        s1, s2 = State("s1"), State("s2")
        m.add_state(s1)
        m.add_state(s2)
        m.set_initial_state(s1)
        m.add_transition(Transition(s1, s2, trigger="go"))
        monitor = MachineMonitor(m)

        m.start()
        m.process_event(SignalEvent("go"))
        assert monitor.transition_count == 1
        assert monitor.state_history == ["s2"]
