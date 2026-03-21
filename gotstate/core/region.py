"""
Parallel region and concurrency management.

Implements parallel region execution, synchronization, and cross-region
coordination for the hierarchical state machine.
"""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import Dict, List, Optional, Set

import icontract

from gotstate.core.state import State
from gotstate.exceptions import RegionError


class RegionStatus(Enum):
    """Defines the possible states of a region."""

    INACTIVE = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    TERMINATING = auto()
    TERMINATED = auto()


@icontract.invariant(lambda self: isinstance(self._status, RegionStatus), "Region status must be valid")
@icontract.invariant(
    lambda self: isinstance(self._name, str) and len(self._name) > 0, "Region name must be a non-empty string"
)
class Region:
    """Represents a parallel region in a hierarchical state machine.

    Manages concurrent execution of orthogonal state configurations
    with proper synchronization and isolation.

    Class Invariants:
    1. Region status must be a valid RegionStatus
    2. Region name must be a non-empty string
    """

    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "Name must be a non-empty string")
    def __init__(self, name: str, parent_state: Optional[State] = None) -> None:
        self._name = name
        self._parent_state = parent_state
        self._status = RegionStatus.INACTIVE
        self._states: Dict[str, State] = {}
        self._active_state: Optional[State] = None
        self._initial_state: Optional[State] = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> RegionStatus:
        return self._status

    @property
    def parent_state(self) -> Optional[State]:
        return self._parent_state

    @property
    def active_state(self) -> Optional[State]:
        return self._active_state

    @property
    def initial_state(self) -> Optional[State]:
        return self._initial_state

    @property
    def states(self) -> Dict[str, State]:
        return dict(self._states)

    @icontract.require(lambda state: state is not None, "State must not be None")
    def add_state(self, state: State) -> None:
        """Add a state to this region."""
        with self._lock:
            if state.name in self._states:
                raise RegionError(f"State '{state.name}' already exists in region '{self._name}'")
            self._states[state.name] = state

    @icontract.require(lambda state: state is not None, "State must not be None")
    def set_initial_state(self, state: State) -> None:
        """Designate the initial state for this region."""
        if state.name not in self._states:
            raise RegionError(f"State '{state.name}' is not in region '{self._name}'")
        self._initial_state = state

    def activate(self) -> None:
        """Activate the region, entering its initial state."""
        with self._lock:
            if self._status == RegionStatus.ACTIVE:
                return
            self._status = RegionStatus.ACTIVE
            if self._initial_state is not None:
                self._active_state = self._initial_state
                self._initial_state.enter()

    def deactivate(self) -> None:
        """Deactivate the region, exiting the current active state."""
        with self._lock:
            self._status = RegionStatus.TERMINATING
            if self._active_state is not None:
                self._active_state.exit()
                self._active_state = None
            self._status = RegionStatus.TERMINATED

    def set_active_state(self, state: State) -> None:
        """Change the active state within this region."""
        with self._lock:
            if state.name not in self._states:
                raise RegionError(f"State '{state.name}' is not in region '{self._name}'")
            self._active_state = state

    def __repr__(self) -> str:
        active = self._active_state.name if self._active_state else "None"
        return f"Region(name={self._name!r}, status={self._status.name}, active={active})"


class ParallelRegion(Region):
    """A region that executes in parallel with sibling regions."""

    def __init__(self, name: str, parent_state: Optional[State] = None) -> None:
        super().__init__(name, parent_state)


class SynchronizationRegion(Region):
    """A region that coordinates synchronization points between parallel regions."""

    def __init__(self, name: str, parent_state: Optional[State] = None) -> None:
        super().__init__(name, parent_state)
        self._sync_points: Set[str] = set()
        self._completed: Set[str] = set()

    def add_sync_point(self, point_id: str) -> None:
        self._sync_points.add(point_id)

    def mark_completed(self, point_id: str) -> None:
        self._completed.add(point_id)

    @property
    def is_synchronized(self) -> bool:
        return self._sync_points == self._completed

    def reset(self) -> None:
        self._completed.clear()


class HistoryRegion(Region):
    """A region that maintains history state information."""

    def __init__(self, name: str, parent_state: Optional[State] = None) -> None:
        super().__init__(name, parent_state)
        self._history: Optional[State] = None

    def save_history(self) -> None:
        """Save current active state as history."""
        self._history = self._active_state

    def restore_history(self) -> Optional[State]:
        """Restore saved history state."""
        return self._history


@icontract.invariant(
    lambda self: isinstance(self._regions, dict),
    "Regions collection must be a dict",
)
class RegionManager:
    """Manages multiple regions and their interactions.

    Coordinates parallel region execution, resource allocation,
    and synchronization.
    """

    def __init__(self) -> None:
        self._regions: Dict[str, Region] = {}
        self._lock = threading.RLock()

    @property
    def regions(self) -> Dict[str, Region]:
        return dict(self._regions)

    @icontract.require(lambda region: region is not None, "Region must not be None")
    def add_region(self, region: Region) -> None:
        with self._lock:
            if region.name in self._regions:
                raise RegionError(f"Region '{region.name}' already registered")
            self._regions[region.name] = region

    def remove_region(self, name: str) -> Region:
        with self._lock:
            if name not in self._regions:
                raise RegionError(f"Region '{name}' not found")
            return self._regions.pop(name)

    def get_region(self, name: str) -> Region:
        if name not in self._regions:
            raise RegionError(f"Region '{name}' not found")
        return self._regions[name]

    def activate_all(self) -> None:
        """Activate all managed regions."""
        with self._lock:
            for region in self._regions.values():
                region.activate()

    def deactivate_all(self) -> None:
        """Deactivate all managed regions."""
        with self._lock:
            for region in self._regions.values():
                region.deactivate()

    def get_active_regions(self) -> List[Region]:
        return [r for r in self._regions.values() if r.status == RegionStatus.ACTIVE]
