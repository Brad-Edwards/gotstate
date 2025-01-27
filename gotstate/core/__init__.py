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

from .state import State
from .transition import Transition
from .event import Event
from .region import Region
from .machine import StateMachine

__all__ = ["State", "Transition", "Event", "Region", "StateMachine"]
