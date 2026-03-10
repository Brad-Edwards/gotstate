"""
Core package providing the fundamental HFSM functionality.

Implements hierarchical state machine core components including
states, events, transitions, regions, and machine orchestration.
"""

from .event import (
    CallEvent,
    ChangeEvent,
    CompletionEvent,
    Event,
    EventKind,
    EventPriority,
    EventQueue,
    SignalEvent,
    TimeEvent,
)
from .machine import (
    MachineBuilder,
    MachineModifier,
    MachineMonitor,
    MachineStatus,
    ProtocolMachine,
    StateMachine,
    SubmachineMachine,
)
from .region import (
    HistoryRegion,
    ParallelRegion,
    Region,
    RegionManager,
    RegionStatus,
    SynchronizationRegion,
)
from .state import (
    ChoiceState,
    CompositeState,
    ConnectionPointState,
    HistoryState,
    JunctionState,
    PseudoState,
    State,
    StateType,
)
from .transition import (
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

__all__ = [
    "State",
    "StateType",
    "CompositeState",
    "PseudoState",
    "HistoryState",
    "ConnectionPointState",
    "ChoiceState",
    "JunctionState",
    "Event",
    "EventKind",
    "EventPriority",
    "EventQueue",
    "SignalEvent",
    "CallEvent",
    "TimeEvent",
    "ChangeEvent",
    "CompletionEvent",
    "Transition",
    "TransitionKind",
    "TransitionPriority",
    "ExternalTransition",
    "InternalTransition",
    "LocalTransition",
    "CompoundTransition",
    "ProtocolTransition",
    "TimeTransition",
    "ChangeTransition",
    "Region",
    "RegionStatus",
    "ParallelRegion",
    "SynchronizationRegion",
    "HistoryRegion",
    "RegionManager",
    "StateMachine",
    "MachineStatus",
    "ProtocolMachine",
    "SubmachineMachine",
    "MachineBuilder",
    "MachineModifier",
    "MachineMonitor",
]
