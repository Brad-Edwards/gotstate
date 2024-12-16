<!-- markdownlint-disable MD037 -->
# MODULE SPECIFICATIONS

## hsm.core.state_machine

### Public Interfaces

- StateMachine (concrete implementation of AbstractStateMachine)
- CompositeStateMachine (adds hierarchical handling)
- Type hints: StateMachine.start() -> None, StateMachine.process_event(event: Event) -> None Async Protocols:
- If async variant provided, AsyncStateMachine in hsm.runtime.async_support.
- Resource Handling: Internal locks for thread safety.
- Extensions: Allows plugging in custom AbstractValidator.
- Allows hooking error recovery strategies.

### hsm.core.states

- State, CompositeState classes
- Manages entry/exit actions, state data initialization/cleanup
- Type hints for state data: State.data: Dict[str, Any]
- Extension points: Custom data initializers from plugins.

### hsm.core.transitions

- Transition class implementing AbstractTransition
- Prioritization logic
- Guard checks, action execution, fallback states
- Extension: Custom prioritization policies via plugins.

### hsm.core.events

- Event class implementing AbstractEvent
- Timeout events, priority handling
- Extension: Custom event subclasses

### hsm.core.guards

- Built-in guard conditions
- Extension: User can provide guards in hsm.plugins.custom_guards

### hsm.core.actions

- Built-in actions
- Extension: User-defined actions in hsm.plugins.custom_actions

#### hsm.core.data_management

- Tools for state data scoping and thread safety
- Context managers for data locking

### hsm.core.hooks

- Hook registration and execution logic
- Extension: Plugins for logging/monitoring

### hsm.core.errors

- Centralized exceptions
- Documented error hierarchies
- hsm.core.validation
- Validator class implementing AbstractValidator
- Integration with construction-time and runtime checks
- Extension: Custom validation rules

### hsm.runtime.executor

- Main event loop logic for synchronous mode
- Possibly uses a worker thread or blocking calls to process events

### hsm.runtime.event_queue

- Default FIFO/priority queue implementation
- Extension: Customizable event queue strategies

### hsm.runtime.async_support

- Async-compatible executor, event loop integration (e.g., asyncio)
- Async event queue variants

### hsm.runtime.timers

- Timeout event scheduling
- Extension: Custom timing sources if needed

### hsm.runtime.concurrency

- Locks, threading primitives
- Ensuring atomic transitions

### hsm.plugins.*

- Example plugins for logging, monitoring
- Custom guard/action definitions

## INTER-MODULE CONTRACTS

### Import Graph

- `hsm.core.*` does not depend on `hsm.runtime.*` (core is independent of runtime mode).
- `hsm.runtime.*` depends on `hsm.core./*` for definitions of states, events.
- `hsm.interfaces.*` is independent and can be imported by `both hsm.core` and `hsm.runtime`.
- `hsm.plugins.*` may depend on `hsm.core` or `hsm.runtime` but not vice versa.

### Async Boundaries

- `hsm.runtime.async_support` provides async variants of event processing. Core modules remain sync-first, enabling a clean async boundary.

### Resource Lifecycle

- States own their data; cleaned up on exit.
- State machine owns event queue; cleaned up on shutdown.
- Timers are cleaned up when state machine stops.

### Error Handling

- Errors raised in `hsm.core.*` bubble up to the state machine level, possibly triggering fallback states.
- `hsm.runtime.*` catches and logs exceptions, may re-raise as HSMError subclasses.

### Initialization Order

- Define machine (states, transitions) in `hsm.core` before starting runtime in hsm.runtime.
- Validation happens after construction, before start.

### Cleanup Requirements

- `hsm.core` ensures state data cleanup after transitions.
- `hsm.runtime` ensures event queues are flushed on stop.
- Plugins must follow documented cleanup hooks.

### ARCHITECTURE SPECIFICATION

| Module                     | Exports                           | Protocols Implemented                | Extensions               | Dependencies                        |
|----------------------------|-----------------------------------|--------------------------------------|--------------------------|-------------------------------------|
| `hsm.core.state_machine`   | StateMachine, CompositeStateMachine | StateMachineProtocol                 | Validators, Hooks        | states, transitions, events, validation |
| `hsm.core.states`          | State, CompositeState             | StateProtocol, CompositeStateProtocol | Custom data init/cleanup | actions, guards (for entry/exit)    |
| `hsm.core.transitions`     | Transition                        | TransitionProtocol                   | Custom priority          | states, guards, actions             |
| `hsm.core.events`          | Event, TimeoutEvent               | EventProtocol                        | Custom event types       | -                                   |
| `hsm.core.guards`          | BasicGuards                       | GuardProtocol                        | Custom guards            | -                                   |
| `hsm.core.actions`         | BasicActions                      | ActionProtocol                       | Custom actions           | -                                   |
| `hsm.core.data_management` | Data access utils                 | -                                    | Custom data policies     | states                              |
| `hsm.core.hooks`           | Hook registration                 | HookProtocol                         | Custom hooks             | states, transitions                 |
| `hsm.core.validation`      | Validator                         | ValidatorProtocol                    | Custom validation rules  | states, transitions, events         |
| `hsm.core.errors`          | Exception classes                 | -                                    | -                        | -                                   |
| `hsm.runtime.executor`     | Executor                          | -                                    | Custom schedulers        | core.*                              |
| `hsm.runtime.event_queue`  | EventQueue                        | EventQueueProtocol                   | Custom queue policies    | events                              |
| `hsm.runtime.async_support`| Async variants                    | AsyncEventQueueProtocol              | Custom async I/O         | event_queue, executor               |
| `hsm.runtime.timers`       | Timer, TimeoutScheduler           | TimerProtocol                        | Custom timing sources    | events                              |
| `hsm.runtime.concurrency`  | Locks, context managers           | -                                    | Different lock strategies | -                                   |
| `hsm.plugins.*`            | Custom functionalities            | -                                    | -                        | core.* or runtime.* (optional)      |

### INTERFACE STABILITY

| API Element                  | Type Hints | Breaking Changes | Migration Path                          |
|------------------------------|------------|-------------------|-----------------------------------------|
| StateMachine.start()         | Yes        | Low               | Backwards-compatible, stable API        |
| StateMachine.process_event() | Yes        | Low               | Well-defined for sync/async             |
| State.data (accessors)       | Yes        | Medium            | Documented getters/setters              |
| Transition guards/actions    | Yes        | Medium            | Use adapter classes if changed          |
| Validator rules              | Yes        | High              | Versioned rules, fallback defaults      |
| EventQueue.enqueue()         | Yes        | Low               | Stable, optional async variant          |
| Timer scheduling methods     | Yes        | Medium            | Deprecation warnings before changes     |
| Hooks (On-Enter/Exit)        | Yes        | Low               | Backwards-compatible, stable hooks      |

This specification provides a structured blueprint for implementing the hierarchical state machine library. It addresses the requirements from architectural design, interface definitions, module responsibilities, inter-module contracts, and API stability while remaining consistent with Python’s best practices and allowing for future extension and customization.
