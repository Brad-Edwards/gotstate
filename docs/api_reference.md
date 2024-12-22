# API Reference

## Core Module (`hsm.core`)

### StateMachine

The primary class for creating and managing state machines.

```python
from hsm.core.state_machine import StateMachine

machine = StateMachine(
    initial_state: State,
    validator: Optional[Validator] = None,
    hooks: Optional[List[HookProtocol]] = None
)
```

#### Constructor Parameters

- `initial_state`: The starting state of the machine
- `validator`: Optional validator for checking machine configuration
- `hooks`: Optional list of hooks for monitoring state machine events

#### Methods

- `start() -> None`: Initialize and start the state machine
- `process_event(event: Event) -> None`: Process an event through the state machine
- `add_state(state: State) -> State`: Add a state to the machine
- `add_transition(transition: Transition) -> Transition`: Add a transition to the machine
- `get_current_state() -> State`: Get the currently active state
- `is_running() -> bool`: Check if the machine is running

### CompositeStateMachine

Extends `StateMachine` with hierarchical state capabilities.

```python
from hsm.core.state_machine import CompositeStateMachine

machine = CompositeStateMachine(
    initial_state: CompositeState,
    validator: Optional[Validator] = None,
    hooks: Optional[List[HookProtocol]] = None
)
```

Additional Methods:
- `get_active_leaf_state() -> State`: Get the currently active leaf state
- `get_state_hierarchy() -> List[State]`: Get the current state hierarchy from root to leaf

### State

Base class for states in the state machine.

```python
from hsm.core.states import State

state = State(
    name: str,
    entry_action: Optional[Callable[[Event], None]] = None,
    exit_action: Optional[Callable[[Event], None]] = None,
    data_initializer: Optional[Callable[[], Dict[str, Any]]] = None
)
```

#### Properties
- `name: str`: The state's identifier
- `data: Dict[str, Any]`: State-specific data storage
- `is_active: bool`: Whether the state is currently active

#### Methods
- `on_enter(event: Optional[Event] = None) -> None`: Called when entering the state
- `on_exit(event: Optional[Event] = None) -> None`: Called when exiting the state
- `get_data(key: str) -> Any`: Get state data by key
- `set_data(key: str, value: Any) -> None`: Set state data

### CompositeState

A state that can contain other states.

```python
from hsm.core.states import CompositeState

state = CompositeState(
    name: str,
    initial_substate: Optional[State] = None,
    history_type: Optional[str] = None  # "shallow" or "deep"
)
```

Additional Methods:
- `add_substate(state: State) -> State`: Add a substate
- `get_substates() -> List[State]`: Get all substates
- `get_active_substate() -> Optional[State]`: Get the currently active substate
- `set_history_type(history_type: str) -> None`: Set history type ("shallow" or "deep")

### Event

Base class for events in the state machine.

```python
from hsm.core.events import Event

event = Event(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    priority: int = 0
)
```

#### Properties
- `name: str`: Event identifier
- `payload: Dict[str, Any]`: Optional event data
- `priority: int`: Event priority for processing order
- `timestamp: float`: When the event was created

### TimeoutEvent

Special event type for handling timeouts.

```python
from hsm.core.events import TimeoutEvent

event = TimeoutEvent(
    name: str,
    duration: float,
    payload: Optional[Dict[str, Any]] = None
)
```

Additional Properties:
- `duration: float`: Timeout duration in seconds
- `deadline: float`: When the timeout will occur

### Transition

Defines state transitions and their conditions.

```python
from hsm.core.transitions import Transition

transition = Transition(
    source: State,
    target: State,
    guards: Optional[List[Callable[[Event], bool]]] = None,
    actions: Optional[List[Callable[[Event], None]]] = None,
    priority: int = 0
)
```

#### Properties
- `source: State`: Source state
- `target: State`: Target state
- `guards: List[Callable[[Event], bool]]`: Guard conditions
- `actions: List[Callable[[Event], None]]`: Transition actions
- `priority: int`: Transition priority

## Runtime Module (`hsm.runtime`)

### EventQueue

Thread-safe event queue implementation.

```python
from hsm.runtime.event_queue import EventQueue

queue = EventQueue()
```

Methods:
- `enqueue(event: Event) -> None`: Add event to queue
- `dequeue() -> Optional[Event]`: Get next event
- `peek() -> Optional[Event]`: View next event without removing
- `is_empty() -> bool`: Check if queue is empty
- `clear() -> None`: Clear all events

### AsyncEventQueue

Asynchronous version of EventQueue.

```python
from hsm.runtime.async_support import AsyncEventQueue

queue = AsyncEventQueue()
```

Additional Methods:
- `async enqueue(event: Event) -> None`
- `async dequeue() -> Event`
- `async peek() -> Optional[Event]`

### TimeoutScheduler

Manages timeout events.

```python
from hsm.runtime.timers import TimeoutScheduler

scheduler = TimeoutScheduler()
```

Methods:
- `schedule(event: TimeoutEvent) -> None`: Schedule a timeout
- `cancel(event_name: str) -> None`: Cancel a timeout
- `get_next_timeout() -> Optional[TimeoutEvent]`: Get next timeout
- `clear() -> None`: Cancel all timeouts

## Hooks Module (`hsm.core.hooks`)

### HookProtocol

Interface for implementing state machine hooks.

```python
from hsm.core.hooks import HookProtocol

class MyHook(HookProtocol):
    def on_enter(self, state: State) -> None: ...
    def on_exit(self, state: State) -> None: ...
    def on_error(self, error: Exception) -> None: ...
```

### HookManager

Manages hook registration and execution.

```python
from hsm.core.hooks import HookManager

manager = HookManager(hooks: Optional[List[HookProtocol]] = None)
```

Methods:
- `register_hook(hook: HookProtocol) -> None`
- `execute_on_enter(state: State) -> None`
- `execute_on_exit(state: State) -> None`
- `execute_on_error(error: Exception) -> None`

## Error Module (`hsm.core.errors`)

### Exception Hierarchy

```python
HSMError
├── StateNotFoundError
├── TransitionError
├── ValidationError
├── EventError
└── ConfigurationError
```

Each error type includes:
- Detailed error message
- Context information
- Stack trace
- Recovery suggestions when applicable 