from typing import List, Optional

import pytest

from hsm.core.events import Event
from hsm.core.hooks import HookProtocol
from hsm.core.state_machine import CompositeStateMachine, StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.runtime.async_support import AsyncCompositeStateMachine, AsyncStateMachine


class TestHook(HookProtocol):
    """Hook implementation for testing state transitions and error handling."""
    def __init__(self):
        # Instance attributes instead of class attributes
        self.enter_states = []
        self.exit_states = []
        self.errors = []

    def on_enter(self, state: State) -> None:
        self.enter_states.append(state)

    def on_exit(self, state: State) -> None:
        self.exit_states.append(state)

    def on_error(self, error: Exception) -> None:
        self.errors.append(error)


@pytest.fixture
def test_hook():
    """Provide a fresh TestHook instance."""
    return TestHook()


class TestScenarios:
    def test_basic_state_machine(self):
        """Test basic state machine operations with simple transitions."""
        # Setup states
        state_a = State("StateA")
        state_b = State("StateB")
        state_c = State("StateC")

        # Create events
        event_1 = Event("Event1")
        event_2 = Event("Event2")

        # Create state machine
        sm = StateMachine(initial_state=state_a)

        # Add states and transitions
        sm.add_state(state_b)
        sm.add_state(state_c)
        sm.add_transition(Transition(source=state_a, target=state_b, guards=[lambda e: e.name == "Event1"]))
        sm.add_transition(Transition(source=state_b, target=state_c, guards=[lambda e: e.name == "Event2"]))

        # Start machine and verify initial state
        sm.start()
        assert sm.current_state == state_a

        # Process events and verify transitions
        assert sm.process_event(event_1) is True
        assert sm.current_state == state_b

        assert sm.process_event(event_2) is True
        assert sm.current_state == state_c

    def test_hierarchical_states(self):
        """Test hierarchical state machine with nested states."""
        # Create composite states
        root = CompositeState("Root")
        composite_a = CompositeState("CompositeA")
        composite_b = CompositeState("CompositeB")

        # Create leaf states
        state_a1 = State("StateA1")
        state_a2 = State("StateA2")
        state_b1 = State("StateB1")
        state_b2 = State("StateB2")

        # Create events
        event_1 = Event("Event1")
        event_2 = Event("Event2")
        event_3 = Event("Event3")

        # Create state machine
        sm = StateMachine(initial_state=root)

        # Build hierarchy
        sm.add_state(composite_a, parent=root)
        sm.add_state(composite_b, parent=root)
        sm.add_state(state_a1, parent=composite_a)
        sm.add_state(state_a2, parent=composite_a)
        sm.add_state(state_b1, parent=composite_b)
        sm.add_state(state_b2, parent=composite_b)

        # Add transitions
        sm.add_transition(Transition(source=state_a1, target=state_a2, guards=[lambda e: e.name == "Event1"]))
        sm.add_transition(Transition(source=state_a2, target=state_b1, guards=[lambda e: e.name == "Event2"]))
        sm.add_transition(Transition(source=state_b1, target=state_b2, guards=[lambda e: e.name == "Event3"]))

        # Set initial states
        composite_a._initial_state = state_a1
        composite_b._initial_state = state_b1
        root._initial_state = composite_a

        # Start machine and verify initial state
        sm.start()
        assert sm.current_state == state_a1

        # Process events and verify transitions
        assert sm.process_event(event_1) is True
        assert sm.current_state == state_a2

        assert sm.process_event(event_2) is True
        assert sm.current_state == state_b1

        assert sm.process_event(event_3) is True
        assert sm.current_state == state_b2

    def test_history_states(self):
        """Test history state functionality in hierarchical state machine."""
        # Create states
        root = CompositeState("Root")
        state_a = State("StateA")
        state_b = State("StateB")
        composite_1 = CompositeState("Composite1")
        state_1a = State("State1A")
        state_1b = State("State1B")

        # Create events
        to_composite = Event("ToComposite")
        to_root = Event("ToRoot")
        inner_transition = Event("InnerTransition")

        # Create state machine
        sm = StateMachine(initial_state=root)

        # Build hierarchy
        sm.add_state(state_a, parent=root)
        sm.add_state(state_b, parent=root)
        sm.add_state(composite_1, parent=root)
        sm.add_state(state_1a, parent=composite_1)
        sm.add_state(state_1b, parent=composite_1)

        # Add transitions
        sm.add_transition(Transition(source=state_a, target=state_1a, guards=[lambda e: e.name == "ToComposite"]))
        sm.add_transition(Transition(source=state_b, target=state_1b, guards=[lambda e: e.name == "ToComposite"]))
        sm.add_transition(Transition(source=composite_1, target=state_b, guards=[lambda e: e.name == "ToRoot"]))
        sm.add_transition(Transition(source=state_1a, target=state_1b, guards=[lambda e: e.name == "InnerTransition"]))

        # Set initial states
        root._initial_state = state_a
        composite_1._initial_state = state_1a

        # Start machine
        sm.start()
        assert sm.current_state == state_a

        # Test history preservation
        sm.process_event(to_composite)
        assert sm.current_state == state_1a

        sm.process_event(inner_transition)
        assert sm.current_state == state_1b
        # Record history for composite_1
        sm._graph.record_history(composite_1, state_1b)

        sm.process_event(to_root)
        assert sm.current_state == state_b

        # Verify history is preserved when returning to composite
        sm.process_event(to_composite)
        # Should return to last recorded state
        assert sm.current_state == state_1b

    def test_hooks_and_notifications(self, test_hook):
        """Test hook notifications during state transitions."""
        # Create states
        state_a = State("StateA")
        state_b = State("StateB")

        # Create event
        event = Event("Event")

        # Create state machine with hook
        sm = StateMachine(initial_state=state_a, hooks=[test_hook])

        # Add states and transitions
        sm.add_state(state_b)
        sm.add_transition(Transition(source=state_a, target=state_b, guards=[lambda e: e.name == event.name]))

        # Start machine
        sm.start()
        assert state_a in test_hook.enter_states

        # Process event
        sm.process_event(event)
        assert state_a in test_hook.exit_states
        assert state_b in test_hook.enter_states

    @pytest.mark.asyncio
    async def test_composite_state_machines(self):
        """Test composite state machines with submachines."""
        # Create main states
        main_a = State("MainA")
        main_composite = CompositeState("MainComposite")
        main_b = State("MainB")

        # Create submachine states
        sub_a = State("SubA")
        sub_b = State("SubB")

        # Create events
        to_composite = Event("ToComposite")
        sub_event = Event("SubEvent")
        to_main_b = Event("ToMainB")

        # Create submachine
        submachine = AsyncStateMachine(initial_state=sub_a)
        submachine.add_state(sub_b)
        submachine.add_transition(Transition(source=sub_a, target=sub_b, guards=[lambda e: e.name == "SubEvent"]))

        # Create main machine
        main_machine = AsyncCompositeStateMachine(initial_state=main_a)
        main_machine.add_state(main_composite)
        main_machine.add_state(main_b)
        main_machine.add_transition(
            Transition(source=main_a, target=main_composite, guards=[lambda e: e.name == "ToComposite"])
        )
        main_machine.add_transition(
            Transition(source=main_composite, target=main_b, guards=[lambda e: e.name == "ToMainB"])
        )

        # Add submachine
        main_machine.add_submachine(main_composite, submachine)

        # Add transitions that reference submachine states - must be done after adding submachine
        main_machine.add_transition(
            Transition(source=sub_b, target=main_b, guards=[lambda e: e.name == "ToMainB"])
        )

        # Start machines
        await main_machine.start()
        assert main_machine.current_state == main_a

        # Test transitions between machines
        await main_machine.process_event(to_composite)
        assert main_machine.current_state == sub_a

        await main_machine.process_event(sub_event)
        assert main_machine.current_state == sub_b

        await main_machine.process_event(to_main_b)
        assert main_machine.current_state == main_b

    def test_complex_scenario(self, test_hook):
        """Test complex scenario combining multiple features."""
        # Create states for a traffic light system with maintenance mode
        root = CompositeState("Root")

        # Normal operation states
        normal_operation = CompositeState("NormalOperation")
        red = State("Red")
        yellow = State("Yellow")
        green = State("Green")

        # Maintenance mode states
        maintenance = CompositeState("Maintenance")
        diagnostic = State("Diagnostic")
        repair = State("Repair")
        test = State("Test")

        # Create events
        next_light = Event("NextLight")
        enter_maintenance = Event("EnterMaintenance")
        exit_maintenance = Event("ExitMaintenance")
        start_repair = Event("StartRepair")
        start_test = Event("StartTest")

        # Create state machine with hook
        sm = StateMachine(initial_state=root, hooks=[test_hook])

        # Build hierarchy
        sm.add_state(normal_operation, parent=root)
        sm.add_state(maintenance, parent=root)

        sm.add_state(red, parent=normal_operation)
        sm.add_state(yellow, parent=normal_operation)
        sm.add_state(green, parent=normal_operation)

        sm.add_state(diagnostic, parent=maintenance)
        sm.add_state(repair, parent=maintenance)
        sm.add_state(test, parent=maintenance)

        # Add transitions
        # Normal operation transitions
        sm.add_transition(Transition(source=red, target=green, guards=[lambda e: e.name == "NextLight"]))
        sm.add_transition(Transition(source=green, target=yellow, guards=[lambda e: e.name == "NextLight"]))
        sm.add_transition(Transition(source=yellow, target=red, guards=[lambda e: e.name == "NextLight"]))

        # Maintenance transitions
        sm.add_transition(
            Transition(source=normal_operation, target=maintenance, guards=[lambda e: e.name == "EnterMaintenance"])
        )
        sm.add_transition(
            Transition(source=maintenance, target=normal_operation, guards=[lambda e: e.name == "ExitMaintenance"])
        )
        sm.add_transition(Transition(source=diagnostic, target=repair, guards=[lambda e: e.name == "StartRepair"]))
        sm.add_transition(Transition(source=repair, target=test, guards=[lambda e: e.name == "StartTest"]))

        # Set initial states
        root._initial_state = normal_operation
        normal_operation._initial_state = red
        maintenance._initial_state = diagnostic

        # Start machine
        sm.start()
        assert sm.current_state == red

        # Test normal operation
        sm.process_event(next_light)
        assert sm.current_state == green

        sm.process_event(next_light)
        assert sm.current_state == yellow

        sm.process_event(next_light)
        assert sm.current_state == red

        # Enter maintenance mode
        sm.process_event(enter_maintenance)
        assert sm.current_state == diagnostic

        # Perform maintenance operations
        sm.process_event(start_repair)
        assert sm.current_state == repair

        sm.process_event(start_test)
        assert sm.current_state == test

        # Exit maintenance and verify history
        sm.process_event(exit_maintenance)
        assert sm.current_state == red  # Should return to last normal operation state

        # Verify hooks were called
        assert len(test_hook.enter_states) > 0
        assert len(test_hook.exit_states) > 0

    def test_traffic_light_with_maintenance(self):
        """Test a traffic light state machine with maintenance mode."""
        # Create states
        root = CompositeState("Root")

        # Main operation modes
        normal_operation = CompositeState("NormalOperation")
        maintenance = CompositeState("Maintenance")

        # Traffic light states
        red = State("Red")
        yellow = State("Yellow")
        green = State("Green")

        # Maintenance states
        diagnostic = State("Diagnostic")
        repair = State("Repair")
        test = State("Test")

        # Create events
        next_light = Event("NextLight")
        enter_maintenance = Event("EnterMaintenance")
        exit_maintenance = Event("ExitMaintenance")
        start_repair = Event("StartRepair")
        start_test = Event("StartTest")

        # Create state machine
        sm = StateMachine(initial_state=root)

        # Build hierarchy
        sm.add_state(normal_operation, parent=root)
        sm.add_state(maintenance, parent=root)

        sm.add_state(red, parent=normal_operation)
        sm.add_state(yellow, parent=normal_operation)
        sm.add_state(green, parent=normal_operation)

        sm.add_state(diagnostic, parent=maintenance)
        sm.add_state(repair, parent=maintenance)
        sm.add_state(test, parent=maintenance)

        # Add transitions with proper history handling
        sm.add_transition(
            Transition(
                source=normal_operation,
                target=diagnostic,  # Target the initial maintenance state directly
                guards=[lambda e: e.name == "EnterMaintenance"],
            )
        )
        sm.add_transition(
            Transition(
                source=maintenance,
                target=red,  # Return to specific state instead of composite
                guards=[lambda e: e.name == "ExitMaintenance"],
            )
        )

        # Normal operation transitions
        sm.add_transition(Transition(source=red, target=green, guards=[lambda e: e.name == "NextLight"]))
        sm.add_transition(Transition(source=green, target=yellow, guards=[lambda e: e.name == "NextLight"]))
        sm.add_transition(Transition(source=yellow, target=red, guards=[lambda e: e.name == "NextLight"]))

        # Maintenance transitions
        sm.add_transition(Transition(source=diagnostic, target=repair, guards=[lambda e: e.name == "StartRepair"]))
        sm.add_transition(Transition(source=repair, target=test, guards=[lambda e: e.name == "StartTest"]))

        # Set initial states
        root._initial_state = normal_operation
        normal_operation._initial_state = red
        maintenance._initial_state = diagnostic

        # Start machine
        sm.start()
        assert sm.current_state == red

        # Test normal operation
        sm.process_event(next_light)
        assert sm.current_state == green

        sm.process_event(next_light)
        assert sm.current_state == yellow

        sm.process_event(next_light)
        assert sm.current_state == red

        # Enter maintenance mode
        sm.process_event(enter_maintenance)
        assert sm.current_state == diagnostic

        # Perform maintenance operations
        sm.process_event(start_repair)
        assert sm.current_state == repair

        sm.process_event(start_test)
        assert sm.current_state == test

        # Exit maintenance and verify return to normal operation
        sm.process_event(exit_maintenance)
        assert sm.current_state == red
