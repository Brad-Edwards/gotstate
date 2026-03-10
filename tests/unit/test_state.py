"""Tests for gotstate.core.state module."""

import threading

import icontract
import pytest

from gotstate.core.state import (
    ChoiceState,
    CompositeState,
    ConnectionPointState,
    HistoryState,
    JunctionState,
    PseudoState,
    State,
    StateType,
)
from gotstate.exceptions import DuplicateStateError, InvalidStateError, StateNotFoundError


class TestState:
    """Tests for the base State class."""

    def test_create_simple_state(self):
        s = State("idle")
        assert s.name == "idle"
        assert s.state_type == StateType.SIMPLE
        assert s.parent is None
        assert s.children == {}
        assert not s.is_active
        assert s.is_leaf
        assert not s.is_composite
        assert not s.is_pseudostate

    def test_create_state_with_type(self):
        s = State("final", StateType.FINAL)
        assert s.state_type == StateType.FINAL

    def test_state_name_contract_rejects_empty(self):
        with pytest.raises(icontract.ViolationError):
            State("")

    def test_state_name_contract_rejects_non_string(self):
        with pytest.raises(icontract.ViolationError):
            State(123)  # type: ignore

    def test_state_type_contract_rejects_invalid(self):
        with pytest.raises(icontract.ViolationError):
            State("s", "not_a_type")  # type: ignore

    def test_add_child(self):
        parent = State("parent", StateType.COMPOSITE)
        child = State("child")
        parent.add_child(child)
        assert "child" in parent.children
        assert child.parent is parent
        assert not parent.is_leaf
        assert parent.is_composite

    def test_add_child_with_parent_arg(self):
        parent = State("parent", StateType.COMPOSITE)
        child = State("child", parent=parent)
        assert child.parent is parent
        assert "child" in parent.children

    def test_add_duplicate_child_raises(self):
        parent = State("parent")
        child1 = State("child")
        parent.add_child(child1)
        child2 = State("child")
        with pytest.raises(DuplicateStateError):
            parent.add_child(child2)

    def test_add_self_as_child_raises(self):
        s = State("s")
        with pytest.raises(InvalidStateError, match="cannot be its own child"):
            s.add_child(s)

    def test_add_child_cycle_detection(self):
        a = State("a")
        b = State("b")
        c = State("c")
        a.add_child(b)
        b.add_child(c)
        with pytest.raises(InvalidStateError, match="cycle"):
            c.add_child(a)

    def test_remove_child(self):
        parent = State("parent")
        child = State("child")
        parent.add_child(child)
        removed = parent.remove_child("child")
        assert removed is child
        assert removed.parent is None
        assert "child" not in parent.children

    def test_remove_nonexistent_child_raises(self):
        parent = State("parent")
        with pytest.raises(StateNotFoundError):
            parent.remove_child("missing")

    def test_get_child(self):
        parent = State("parent")
        child = State("child")
        parent.add_child(child)
        assert parent.get_child("child") is child

    def test_get_child_not_found_raises(self):
        parent = State("parent")
        with pytest.raises(StateNotFoundError):
            parent.get_child("missing")

    def test_get_ancestors(self):
        root = State("root")
        mid = State("mid")
        leaf = State("leaf")
        root.add_child(mid)
        mid.add_child(leaf)
        ancestors = leaf.get_ancestors()
        assert len(ancestors) == 2
        assert ancestors[0] is mid
        assert ancestors[1] is root

    def test_get_root(self):
        root = State("root")
        mid = State("mid")
        leaf = State("leaf")
        root.add_child(mid)
        mid.add_child(leaf)
        assert leaf.get_root() is root
        assert root.get_root() is root

    def test_find_lca(self):
        root = State("root")
        a = State("a")
        b = State("b")
        root.add_child(a)
        root.add_child(b)
        assert State.find_lca(a, b) is root

    def test_find_lca_parent_child(self):
        parent = State("parent")
        child = State("child")
        parent.add_child(child)
        assert State.find_lca(parent, child) is parent

    def test_find_lca_no_common_ancestor(self):
        a = State("a")
        b = State("b")
        assert State.find_lca(a, b) is None

    def test_entry_exit_actions(self):
        s = State("s")
        entered = []
        exited = []
        s.on_entry(lambda: entered.append(True))
        s.on_exit(lambda: exited.append(True))

        s.enter()
        assert s.is_active
        assert len(entered) == 1

        s.exit()
        assert not s.is_active
        assert len(exited) == 1

    def test_state_data(self):
        s = State("s")
        s.set_data("key", "value")
        assert s.get_data("key") == "value"
        assert s.get_data("missing", "default") == "default"

    def test_state_data_inheritance(self):
        parent = State("parent")
        child = State("child")
        parent.add_child(child)
        parent.set_data("inherited", 42)
        assert child.get_data("inherited") == 42

    def test_state_data_override(self):
        parent = State("parent")
        child = State("child")
        parent.add_child(child)
        parent.set_data("key", "parent_value")
        child.set_data("key", "child_value")
        assert child.get_data("key") == "child_value"
        assert parent.get_data("key") == "parent_value"

    def test_set_data_empty_key_contract(self):
        s = State("s")
        with pytest.raises(icontract.ViolationError):
            s.set_data("", "value")

    def test_get_data_empty_key_contract(self):
        s = State("s")
        with pytest.raises(icontract.ViolationError):
            s.get_data("")

    def test_repr(self):
        s = State("idle", StateType.SIMPLE)
        assert "idle" in repr(s)
        assert "SIMPLE" in repr(s)

    def test_thread_safety_add_children(self):
        parent = State("parent")
        errors = []

        def add_child(i):
            try:
                parent.add_child(State(f"child_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_child, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(parent.children) == 20

    def test_pseudostate_types(self):
        for st in (StateType.INITIAL, StateType.CHOICE, StateType.JUNCTION,
                    StateType.SHALLOW_HISTORY, StateType.DEEP_HISTORY,
                    StateType.ENTRY_POINT, StateType.EXIT_POINT, StateType.TERMINATE):
            s = State("ps", st)
            assert s.is_pseudostate


class TestCompositeState:
    """Tests for the CompositeState class."""

    def test_create_composite(self):
        cs = CompositeState("composite")
        assert cs.state_type == StateType.COMPOSITE
        assert cs.initial_state is None

    def test_set_initial_state(self):
        cs = CompositeState("comp")
        child = State("child")
        cs.add_child(child)
        cs.set_initial_state(child)
        assert cs.initial_state is child

    def test_set_initial_state_not_child_raises(self):
        cs = CompositeState("comp")
        foreign = State("foreign")
        with pytest.raises(InvalidStateError):
            cs.set_initial_state(foreign)

    def test_get_active_substates(self):
        cs = CompositeState("comp")
        c1 = State("c1")
        c2 = State("c2")
        cs.add_child(c1)
        cs.add_child(c2)
        c1.enter()
        active = cs.get_active_substates()
        assert len(active) == 1
        assert active[0] is c1


class TestPseudoState:
    """Tests for PseudoState and its subclasses."""

    def test_create_initial_pseudostate(self):
        ps = PseudoState("init", StateType.INITIAL)
        assert ps.is_pseudostate
        assert ps.state_type == StateType.INITIAL

    def test_pseudostate_rejects_non_pseudo_type(self):
        with pytest.raises(icontract.ViolationError):
            PseudoState("bad", StateType.SIMPLE)

    def test_pseudostate_cannot_have_children(self):
        ps = PseudoState("init", StateType.INITIAL)
        child = State("child")
        with pytest.raises(InvalidStateError, match="Pseudostates cannot contain"):
            ps.add_child(child)


class TestHistoryState:
    """Tests for the HistoryState class."""

    def test_create_shallow_history(self):
        hs = HistoryState("history")
        assert hs.state_type == StateType.SHALLOW_HISTORY

    def test_create_deep_history(self):
        hs = HistoryState("history", StateType.DEEP_HISTORY)
        assert hs.state_type == StateType.DEEP_HISTORY

    def test_invalid_history_type(self):
        with pytest.raises(icontract.ViolationError):
            HistoryState("bad", StateType.SIMPLE)

    def test_save_and_restore_configuration(self):
        hs = HistoryState("h")
        s1 = State("s1")
        s2 = State("s2")
        hs.save_configuration([s1, s2])
        restored = hs.restore_configuration()
        assert len(restored) == 2
        assert restored[0] is s1

    def test_restore_default_when_no_history(self):
        hs = HistoryState("h")
        default = State("default")
        hs.default_state = default
        restored = hs.restore_configuration()
        assert len(restored) == 1
        assert restored[0] is default

    def test_restore_empty_when_no_history_or_default(self):
        hs = HistoryState("h")
        assert hs.restore_configuration() == []


class TestConnectionPointState:
    """Tests for ConnectionPointState."""

    def test_create_entry_point(self):
        cp = ConnectionPointState("entry", StateType.ENTRY_POINT)
        assert cp.state_type == StateType.ENTRY_POINT

    def test_create_exit_point(self):
        cp = ConnectionPointState("exit", StateType.EXIT_POINT)
        assert cp.state_type == StateType.EXIT_POINT

    def test_invalid_type_raises(self):
        with pytest.raises(icontract.ViolationError):
            ConnectionPointState("bad", StateType.SIMPLE)


class TestChoiceAndJunction:
    """Tests for ChoiceState and JunctionState."""

    def test_choice_state(self):
        cs = ChoiceState("choice")
        assert cs.state_type == StateType.CHOICE
        assert cs.is_pseudostate

    def test_junction_state(self):
        js = JunctionState("junction")
        assert js.state_type == StateType.JUNCTION
        assert js.is_pseudostate
