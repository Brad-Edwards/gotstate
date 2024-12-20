# tests/unit/test_states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


def test_state_init():
    from hsm.core.states import State

    s = State(name="TestState")
    assert s.name == "TestState"
    assert isinstance(s.data, dict)


def test_state_lifecycle():
    from hsm.core.states import State

    enter_called = False
    exit_called = False

    def on_enter_action():
        nonlocal enter_called
        enter_called = True

    def on_exit_action():
        nonlocal exit_called
        exit_called = True

    s = State("S", entry_actions=[on_enter_action], exit_actions=[on_exit_action])
    s.on_enter()
    assert enter_called is True
    s.on_exit()
    assert exit_called is True


def test_composite_state_children():
    from hsm.core.states import CompositeState, State

    cs = CompositeState("Composite")
    child = State("Child")
    cs.add_child_state(child)
    assert cs.get_child_state("Child") is child
    assert "Child" in cs.children
