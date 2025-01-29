"""gotstate: UML-compliant Hierarchical Finite State Machine (HFSM) implementation

This package provides a robust implementation of hierarchical state machines following
UML state machine semantics.

Responsibilities:
    - State machine definition and execution
    - Event processing and transition management
    - Hierarchical state composition 
    - Parallel region handling
    - History state tracking
    - Runtime validation

Interactions:
    - Client code through public API
    - Python type system for static/runtime type checking
    - Operating system for concurrency primitives
    - Storage systems for persistence
    - Logging system for diagnostics

Cross-cutting Concerns:
    Thread Safety:
        - All public APIs are thread-safe
        - Internal state protected by appropriate locks
        - Documented thread safety guarantees per component

    Error Handling:
        - Structured error hierarchy
        - Consistent error reporting
        - Clean error recovery paths

    Logging:
        - Structured logging format
        - Configurable verbosity levels
        - Performance impact minimized

    Performance:
        - O(1) state lookup where possible
        - Bounded memory usage
        - Predictable latency

    Security:
        - Input validation on all public APIs
        - Safe serialization/deserialization
        - Protected internal state
"""

__version__ = "0.1.0"
