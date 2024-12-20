# tests/unit/test_state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock


def test_state_machine_init(dummy_state, validator):
    from hsm.core.state_machine import StateMachine

    m = StateMachine(initial_state=dummy_state, validator=validator)
    assert m.current_state is dummy_state


def test_state_machine_start(dummy_state):
    from hsm.core.state_machine import StateMachine

    m = StateMachine(initial_state=dummy_state)
    # on_enter should be called on initial state start
    dummy_state.on_enter = MagicMock()
    m.start()
    dummy_state.on_enter.assert_called_once()


def test_state_machine_process_event(dummy_state, dummy_event):
    from hsm.core.state_machine import StateMachine

    m = StateMachine(initial_state=dummy_state)
    # Without transitions, event does nothing but should not error
    m.start()
    m.process_event(dummy_event)
    assert m.current_state is dummy_state


def test_state_machine_stop(dummy_state):
    from hsm.core.state_machine import StateMachine

    m = StateMachine(initial_state=dummy_state)
    dummy_state.on_exit = MagicMock()
    m.start()
    m.stop()
    dummy_state.on_exit.assert_called_once()


def test_composite_state_machine_submachines(dummy_state):
    from hsm.core.state_machine import CompositeStateMachine, StateMachine
    from hsm.core.states import CompositeState, State

    root = CompositeState("Root")
    csm = CompositeStateMachine(initial_state=root)
    child_machine = StateMachine(initial_state=State("Sub"))
    csm.add_submachine(root, child_machine)
    assert csm.submachines[root] is child_machine
