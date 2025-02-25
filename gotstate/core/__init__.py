"""
Core package providing the fundamental HFSM functionality.

Architecture:
- Implements hierarchical state machine core components
- Manages state hierarchy, transitions, events, and regions
- Coordinates between components through machine orchestration

Design Patterns:
- Composite Pattern for state hierarchy
- Observer Pattern for state changes
- Command Pattern for transitions
- Strategy Pattern for event processing
- Mediator Pattern for coordination

Security:
- Input validation at module boundaries
- State data isolation
- Resource usage monitoring
- Type system safety checks

Cross-cutting:
- Error handling with consistent propagation
- Performance optimization for state operations
- Monitoring hooks for metrics
- Testing boundaries for validation
"""

from gotstate.core.state import State
from gotstate.core.statemachine import StateMachine
from gotstate.core.transition import Transition
from gotstate.core.event import Event
from gotstate.core.guard import Guard
from gotstate.core.action import Action
from gotstate.core.pseudostate import PseudoState, InitialState, TerminateState
from gotstate.core.region import Region

__all__ = [
    "State", 
    "StateMachine", 
    "Transition", 
    "Event", 
    "Guard", 
    "Action",
    "PseudoState",
    "InitialState", 
    "TerminateState",
    "Region"
]
