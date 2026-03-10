"""Tests for gotstate.core.region module."""

import icontract
import pytest

from gotstate.core.region import (
    HistoryRegion,
    ParallelRegion,
    Region,
    RegionManager,
    RegionStatus,
    SynchronizationRegion,
)
from gotstate.core.state import State
from gotstate.exceptions import RegionError


class TestRegion:
    """Tests for the base Region class."""

    def test_create(self):
        r = Region("main")
        assert r.name == "main"
        assert r.status == RegionStatus.INACTIVE
        assert r.active_state is None
        assert r.initial_state is None
        assert r.states == {}
        assert r.parent_state is None

    def test_create_with_parent(self):
        parent = State("parent")
        r = Region("main", parent)
        assert r.parent_state is parent

    def test_name_contract(self):
        with pytest.raises(icontract.ViolationError):
            Region("")

    def test_add_state(self):
        r = Region("main")
        s = State("idle")
        r.add_state(s)
        assert "idle" in r.states

    def test_add_duplicate_state_raises(self):
        r = Region("main")
        r.add_state(State("s"))
        with pytest.raises(RegionError, match="already exists"):
            r.add_state(State("s"))

    def test_set_initial_state(self):
        r = Region("main")
        s = State("idle")
        r.add_state(s)
        r.set_initial_state(s)
        assert r.initial_state is s

    def test_set_initial_state_not_in_region_raises(self):
        r = Region("main")
        with pytest.raises(RegionError, match="not in region"):
            r.set_initial_state(State("foreign"))

    def test_activate(self):
        r = Region("main")
        s = State("idle")
        r.add_state(s)
        r.set_initial_state(s)
        r.activate()
        assert r.status == RegionStatus.ACTIVE
        assert r.active_state is s
        assert s.is_active

    def test_activate_already_active_is_noop(self):
        r = Region("main")
        s = State("idle")
        r.add_state(s)
        r.set_initial_state(s)
        r.activate()
        r.activate()  # Should not re-enter
        assert r.status == RegionStatus.ACTIVE

    def test_activate_without_initial(self):
        r = Region("main")
        r.activate()
        assert r.status == RegionStatus.ACTIVE
        assert r.active_state is None

    def test_deactivate(self):
        r = Region("main")
        s = State("idle")
        r.add_state(s)
        r.set_initial_state(s)
        r.activate()
        r.deactivate()
        assert r.status == RegionStatus.TERMINATED
        assert r.active_state is None
        assert not s.is_active

    def test_set_active_state(self):
        r = Region("main")
        s1, s2 = State("s1"), State("s2")
        r.add_state(s1)
        r.add_state(s2)
        r.set_active_state(s1)
        assert r.active_state is s1

    def test_set_active_state_not_in_region_raises(self):
        r = Region("main")
        with pytest.raises(RegionError, match="not in region"):
            r.set_active_state(State("foreign"))

    def test_repr(self):
        r = Region("main")
        assert "main" in repr(r)
        assert "INACTIVE" in repr(r)


class TestParallelRegion:
    def test_create(self):
        r = ParallelRegion("parallel")
        assert r.name == "parallel"
        assert r.status == RegionStatus.INACTIVE


class TestSynchronizationRegion:
    def test_sync_points(self):
        r = SynchronizationRegion("sync")
        r.add_sync_point("a")
        r.add_sync_point("b")
        assert not r.is_synchronized

        r.mark_completed("a")
        assert not r.is_synchronized

        r.mark_completed("b")
        assert r.is_synchronized

    def test_reset(self):
        r = SynchronizationRegion("sync")
        r.add_sync_point("a")
        r.mark_completed("a")
        r.reset()
        assert not r.is_synchronized


class TestHistoryRegion:
    def test_save_and_restore_history(self):
        r = HistoryRegion("hist")
        s = State("s")
        r.add_state(s)
        r.set_initial_state(s)
        r.activate()
        r.save_history()
        assert r.restore_history() is s

    def test_restore_no_history(self):
        r = HistoryRegion("hist")
        assert r.restore_history() is None


class TestRegionManager:
    def test_add_and_get_region(self):
        mgr = RegionManager()
        r = Region("main")
        mgr.add_region(r)
        assert mgr.get_region("main") is r
        assert "main" in mgr.regions

    def test_add_duplicate_raises(self):
        mgr = RegionManager()
        mgr.add_region(Region("main"))
        with pytest.raises(RegionError, match="already registered"):
            mgr.add_region(Region("main"))

    def test_remove_region(self):
        mgr = RegionManager()
        mgr.add_region(Region("main"))
        removed = mgr.remove_region("main")
        assert removed.name == "main"
        assert "main" not in mgr.regions

    def test_remove_nonexistent_raises(self):
        mgr = RegionManager()
        with pytest.raises(RegionError, match="not found"):
            mgr.remove_region("missing")

    def test_get_nonexistent_raises(self):
        mgr = RegionManager()
        with pytest.raises(RegionError, match="not found"):
            mgr.get_region("missing")

    def test_activate_deactivate_all(self):
        mgr = RegionManager()
        r1 = Region("r1")
        r2 = Region("r2")
        r1.add_state(State("s1"))
        r1.set_initial_state(r1.states["s1"])
        r2.add_state(State("s2"))
        r2.set_initial_state(r2.states["s2"])
        mgr.add_region(r1)
        mgr.add_region(r2)

        mgr.activate_all()
        active = mgr.get_active_regions()
        assert len(active) == 2

        mgr.deactivate_all()
        active = mgr.get_active_regions()
        assert len(active) == 0
