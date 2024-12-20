# tests/unit/test_states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


def test_state_init():
    from hsm.core.states import State

    s = State(name="TestState")
    assert s.name == "TestState"
    assert isinstance(s.data, dict)
    assert s.parent is None


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
    child1 = State("Child1")
    child2 = State("Child2")

    cs.add_child_state(child1)
    cs.add_child_state(child2)

    assert cs.get_child_state("Child1") is child1
    assert cs.get_child_state("Child2") is child2
    assert child1.parent is cs
    assert child2.parent is cs


def test_state_data_isolation():
    from hsm.core.states import State

    s1 = State("S1")
    s2 = State("S2")

    s1.data["test"] = "value1"
    s2.data["test"] = "value2"

    assert s1.data["test"] == "value1"
    assert s2.data["test"] == "value2"
