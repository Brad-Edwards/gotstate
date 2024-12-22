# GotState Usage Guide

## Basic Usage

### Creating a State Machine

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Create states
initial = State("Initial")
running = State("Running")
paused = State("Paused")
stopped = State("Stopped")

# Create state machine
machine = StateMachine(initial_state=initial)

# Add states
machine.add_state(running)
machine.add_state(paused)
machine.add_state(stopped)

# Add transitions
machine.add_transition(
    Transition(
        source=initial,
        target=running,
        guards=[lambda e: e.name == "start"]
    )
)

machine.add_transition(
    Transition(
        source=running,
        target=paused,
        guards=[lambda e: e.name == "pause"]
    )
)

machine.add_transition(
    Transition(
        source=paused,
        target=running,
        guards=[lambda e: e.name == "resume"]
    )
)

# Add transitions from both running and paused to stopped
machine.add_transition(
    Transition(
        source=running,
        target=stopped,
        guards=[lambda e: e.name == "stop"]
    )
)

machine.add_transition(
    Transition(
        source=paused,
        target=stopped,
        guards=[lambda e: e.name == "stop"]
    )
)

# Start the machine
machine.start()

# Trigger transitions
machine.process_event(Event("start"))  # Initial -> Running
machine.process_event(Event("pause"))  # Running -> Paused
machine.process_event(Event("resume")) # Paused -> Running
machine.process_event(Event("stop"))   # Running -> Stopped
```

### Using Actions

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

def log_transition(event: Event) -> None:
    print(f"Transition triggered by {event.name}")

def update_display(event: Event) -> None:
    print(f"Updating display for {event.name}")

# Create states
initial = State("Initial")
running = State("Running")

# Create machine
machine = StateMachine(initial_state=initial)
machine.add_state(running)

# Add transition with actions
machine.add_transition(
    Transition(
        source=initial,
        target=running,
        guards=[lambda e: e.name == "start"],
        actions=[log_transition, update_display]
    )
)

# Start and trigger
machine.start()
machine.process_event(Event("start"))
```

### Using Guards

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

def is_authorized(event: Event) -> bool:
    return event.metadata.get("user_role") == "admin"

def has_permission(event: Event) -> bool:
    return event.metadata.get("permission") == "write"

# Create states
locked = State("Locked")
unlocked = State("Unlocked")

# Create machine
machine = StateMachine(initial_state=locked)
machine.add_state(unlocked)

# Add transition with guards
machine.add_transition(
    Transition(
        source=locked,
        target=unlocked,
        guards=[lambda e: e.name == "unlock", is_authorized, has_permission]
    )
)

# Start and trigger
machine.start()

# This will fail because guards are not satisfied
event = Event("unlock")
machine.process_event(event)

# This will succeed because guards are satisfied
event = Event("unlock")
event.metadata["user_role"] = "admin"
event.metadata["permission"] = "write"
machine.process_event(event)
```

### Using State Data

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Create states
counter = State("Counter")

# Create machine
machine = StateMachine(initial_state=counter)

# Initialize state data
machine._graph.set_state_data(counter, "count", 0)

def increment_counter(event: Event) -> None:
    current = machine._graph.get_state_data(counter).get("count", 0)
    machine._graph.set_state_data(counter, "count", current + 1)

# Add self-transition
machine.add_transition(
    Transition(
        source=counter,
        target=counter,
        guards=[lambda e: e.name == "increment"],
        actions=[increment_counter]
    )
)

# Start and use
machine.start()
print(f"Initial count: {machine._graph.get_state_data(counter).get('count')}")  # 0

machine.process_event(Event("increment"))
print(f"After increment: {machine._graph.get_state_data(counter).get('count')}")  # 1
```

### Async Support

```python
import asyncio
from hsm.runtime.async_support import AsyncStateMachine, AsyncEventQueue
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

async def async_action(event: Event) -> None:
    await asyncio.sleep(1)
    print(f"Async action completed for {event.name}")

# Create states
initial = State("Initial")
final = State("Final")

# Create machine and event queue
machine = AsyncStateMachine(initial_state=initial)
machine.add_state(final)

# Add transition with async action
machine.add_transition(
    Transition(
        source=initial,
        target=final,
        guards=[lambda e: e.name == "process"],
        actions=[async_action]
    )
)

# Run async machine
async def main():
    # Start the machine
    await machine.start()
    assert machine.current_state == initial

    # Process event and wait for completion
    event = Event("process")
    result = await machine.process_event(event)
    assert result is True
    assert machine.current_state == final

    # Stop the machine
    await machine.stop()

# Run the async code
if __name__ == "__main__":
    asyncio.run(main())
```

### Error Handling

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition
from hsm.core.errors import TransitionError

def risky_action(event: Event) -> None:
    if event.metadata.get("fail"):
        raise ValueError("Action failed")

def strict_guard(event: Event) -> bool:
    if event.metadata.get("invalid"):
        raise ValueError("Guard check failed")
    return True

# Create states
initial = State("Initial")
error = State("Error")
success = State("Success")

# Create machine
machine = StateMachine(initial_state=initial)
machine.add_state(error)
machine.add_state(success)

# Add transitions with error handling
machine.add_transition(
    Transition(
        source=initial,
        target=success,
        guards=[lambda e: e.name == "process", strict_guard],
        actions=[risky_action]
    )
)

# Add fallback transition for when guard fails
machine.add_transition(
    Transition(
        source=initial,
        target=error,
        guards=[lambda e: e.name == "process"],
        priority=0  # Lower priority, used as fallback
    )
)

# Start machine
machine.start()

# This will transition to error state because guard fails
event = Event("process")
event.metadata["invalid"] = True
machine.process_event(event)
assert machine.current_state == error

# Reset to initial state
machine.reset()

# This will transition to error state because action fails
event = Event("process")
event.metadata["fail"] = True
machine.process_event(event)
assert machine.current_state == error

# Reset to initial state
machine.reset()

# This will succeed and transition to success state
event = Event("process")
machine.process_event(event)
assert machine.current_state == success
```

### Hierarchical States

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Create states
operational = CompositeState("Operational")
maintenance = CompositeState("Maintenance")
idle = State("Idle")
active = State("Active")
diagnostic = State("Diagnostic")
repair = State("Repair")

# Create machine
machine = StateMachine(initial_state=operational)

# Add states with hierarchy
machine.add_state(maintenance)
machine.add_state(idle, parent=operational)
machine.add_state(active, parent=operational)
machine.add_state(diagnostic, parent=maintenance)
machine.add_state(repair, parent=maintenance)

# Set initial substates
machine._graph.set_initial_state(operational, idle)
machine._graph.set_initial_state(maintenance, diagnostic)

# Add transitions between substates
machine.add_transition(
    Transition(
        source=idle,
        target=active,
        guards=[lambda e: e.name == "activate"]
    )
)

machine.add_transition(
    Transition(
        source=active,
        target=idle,
        guards=[lambda e: e.name == "deactivate"]
    )
)

# Add transitions between composite states
machine.add_transition(
    Transition(
        source=operational,
        target=maintenance,
        guards=[lambda e: e.name == "maintain"]
    )
)

machine.add_transition(
    Transition(
        source=maintenance,
        target=operational,
        guards=[lambda e: e.name == "resume"]
    )
)

# Add transitions between substates of maintenance
machine.add_transition(
    Transition(
        source=diagnostic,
        target=repair,
        guards=[lambda e: e.name == "repair"]
    )
)

machine.add_transition(
    Transition(
        source=repair,
        target=diagnostic,
        guards=[lambda e: e.name == "diagnose"]
    )
)

# Start and use
machine.start()
assert machine.current_state == idle  # Initial substate of operational

machine.process_event(Event("activate"))
assert machine.current_state == active

machine.process_event(Event("maintain"))
assert machine.current_state == diagnostic  # Initial substate of maintenance

machine.process_event(Event("repair"))
assert machine.current_state == repair

machine.process_event(Event("diagnose"))
assert machine.current_state == diagnostic

machine.process_event(Event("resume"))
assert machine.current_state == idle  # Back to initial substate of operational
```

### Parallel States

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Create composite states for parallel regions
audio_player = CompositeState("AudioPlayer")
video_player = CompositeState("VideoPlayer")

# Create substates for audio player
audio_playing = State("AudioPlaying")
audio_paused = State("AudioPaused")

# Create substates for video player
video_playing = State("VideoPlaying")
video_paused = State("VideoPaused")

# Create machine with root composite state
root = CompositeState("Root")
machine = StateMachine(initial_state=root)

# Add parallel regions to root
machine.add_state(audio_player, parent=root)
machine.add_state(video_player, parent=root)

# Add substates to audio player
machine.add_state(audio_playing, parent=audio_player)
machine.add_state(audio_paused, parent=audio_player)

# Add substates to video player
machine.add_state(video_playing, parent=video_player)
machine.add_state(video_paused, parent=video_player)

# Set initial substates
machine._graph.set_initial_state(root, audio_player)
machine._graph.set_initial_state(audio_player, audio_paused)
machine._graph.set_initial_state(video_player, video_paused)

# Add transitions for audio
machine.add_transition(
    Transition(
        source=audio_paused,
        target=audio_playing,
        guards=[lambda e: e.name == "play_audio"]
    )
)

machine.add_transition(
    Transition(
        source=audio_playing,
        target=audio_paused,
        guards=[lambda e: e.name == "pause_audio"]
    )
)

# Add transitions for video
machine.add_transition(
    Transition(
        source=video_paused,
        target=video_playing,
        guards=[lambda e: e.name == "play_video"]
    )
)

machine.add_transition(
    Transition(
        source=video_playing,
        target=video_paused,
        guards=[lambda e: e.name == "pause_video"]
    )
)

# Start and use
machine.start()
assert machine.current_state == audio_paused  # Initial state

# Control audio independently
machine.process_event(Event("play_audio"))
assert machine.current_state == audio_playing

# Control video independently
machine.process_event(Event("play_video"))
assert machine.current_state == video_playing

# Pause both
machine.process_event(Event("pause_audio"))
assert machine.current_state == audio_paused

machine.process_event(Event("pause_video"))
assert machine.current_state == video_paused
```

## Advanced Features

### Event Metadata

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

def process_metadata(event: Event) -> None:
    user = event.metadata.get("user", "anonymous")
    timestamp = event.metadata.get("timestamp", "unknown")
    print(f"Action by {user} at {timestamp}")

# Create states
initial = State("Initial")
processed = State("Processed")

# Create machine
machine = StateMachine(initial_state=initial)
machine.add_state(processed)

# Add transition
machine.add_transition(
    Transition(
        source=initial,
        target=processed,
        guards=[lambda e: e.name == "process"],
        actions=[process_metadata]
    )
)

# Start and use with metadata
machine.start()

event = Event("process")
event.metadata["user"] = "admin"
event.metadata["timestamp"] = "2023-01-01 12:00:00"
event.metadata["priority"] = "high"
machine.process_event(event)
```

### Custom Event Types

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class CustomEvent(Event):
    """Custom event type that adds data field."""
    data: Dict[str, Any]

    def __init__(self, name: str, priority: int = 0, data: Dict[str, Any] = None) -> None:
        super().__init__(name, priority)
        self.data = data or {}

def handle_custom_event(event: CustomEvent) -> None:
    print(f"Processing {event.name} with priority {event.priority}")
    if event.data:
        print(f"Event data: {event.data}")

# Create states
waiting = State("Waiting")
processing = State("Processing")

# Create machine
machine = StateMachine(initial_state=waiting)
machine.add_state(processing)

# Add transition
machine.add_transition(
    Transition(
        source=waiting,
        target=processing,
        guards=[lambda e: e.name == "process"],
        actions=[handle_custom_event]
    )
)

# Start and use with custom event
machine.start()

event = CustomEvent(
    name="process",
    priority=10,  # Higher priority events are processed first
    data={"task_id": 123}
)
event.metadata["user"] = "admin"
machine.process_event(event)
``` 