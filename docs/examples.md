# Examples

This document provides practical examples of using GotState in various scenarios.

## Basic Examples

### Traffic Light Controller

A simple traffic light controller using basic states and transitions:

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Define states
class TrafficLightState(State):
    def on_enter(self, event=None) -> None:
        print(f"Light changed to {self.name}")

red = TrafficLightState("Red")
yellow = TrafficLightState("Yellow")
green = TrafficLightState("Green")

# Create state machine
traffic_light = StateMachine(initial_state=red)
traffic_light.add_state(yellow)
traffic_light.add_state(green)

# Add transitions
traffic_light.add_transition(
    Transition(
        source=red,
        target=green,
        guards=[lambda e: e.name == "NextLight"]
    )
)

traffic_light.add_transition(
    Transition(
        source=green,
        target=yellow,
        guards=[lambda e: e.name == "NextLight"]
    )
)

traffic_light.add_transition(
    Transition(
        source=yellow,
        target=red,
        guards=[lambda e: e.name == "NextLight"]
    )
)

# Start the machine
traffic_light.start()

# Process events
traffic_light.process_event(Event("NextLight"))  # Red -> Green
traffic_light.process_event(Event("NextLight"))  # Green -> Yellow
traffic_light.process_event(Event("NextLight"))  # Yellow -> Red
```

### Document Management System

Example of using composite states for document workflow:

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State, CompositeState
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Define states
class DocumentState(State):
    def on_enter(self, event=None) -> None:
        if event:
            self.set_data("user", event.metadata.get("user"))

# Create states
draft = DocumentState("Draft")
revision = DocumentState("Revision")
pending = DocumentState("Pending")
approved = DocumentState("Approved")
rejected = DocumentState("Rejected")
published = State("Published")

# Create composite states
editing = CompositeState("Editing")
review = CompositeState("Review")

# Create state machine
doc_system = StateMachine(initial_state=draft)

# Add states with hierarchy
doc_system.add_state(editing)
doc_system.add_state(review)
doc_system.add_state(published)
doc_system.add_state(draft, parent=editing)
doc_system.add_state(revision, parent=editing)
doc_system.add_state(pending, parent=review)
doc_system.add_state(approved, parent=review)
doc_system.add_state(rejected, parent=review)

# Add transitions
doc_system.add_transition(
    Transition(
        source=draft,
        target=pending,
        guards=[lambda e: e.name == "submit_for_review"]
    )
)

doc_system.add_transition(
    Transition(
        source=pending,
        target=approved,
        guards=[lambda e: e.name == "approve"]
    )
)

doc_system.add_transition(
    Transition(
        source=pending,
        target=rejected,
        guards=[lambda e: e.name == "reject"]
    )
)

doc_system.add_transition(
    Transition(
        source=rejected,
        target=revision,
        guards=[lambda e: e.name == "revise"]
    )
)

doc_system.add_transition(
    Transition(
        source=approved,
        target=published,
        guards=[lambda e: e.name == "publish"]
    )
)

# Start the machine
doc_system.start()

# Example usage
submit_event = Event("submit_for_review")
submit_event.metadata["user"] = "john"
doc_system.process_event(submit_event)

approve_event = Event("approve")
approve_event.metadata["user"] = "alice"
doc_system.process_event(approve_event)

publish_event = Event("publish")
publish_event.metadata["user"] = "admin"
doc_system.process_event(publish_event)
```

## Advanced Examples

### Login System

Example of a login system with account lockout:

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Create states
logged_out = State("LoggedOut")
authenticating = State("Authenticating")
logged_in = State("LoggedIn")
locked = State("Locked")

# Create state machine
login_system = StateMachine(initial_state=logged_out)
login_system.add_state(authenticating)
login_system.add_state(logged_in)
login_system.add_state(locked)

# Add transitions
login_system.add_transition(
    Transition(
        source=logged_out,
        target=authenticating,
        guards=[lambda e: e.name == "attempt_login"]
    )
)

login_system.add_transition(
    Transition(
        source=authenticating,
        target=logged_in,
        guards=[lambda e: e.name == "auth_success"]
    )
)

login_system.add_transition(
    Transition(
        source=authenticating,
        target=logged_out,
        guards=[lambda e: e.name == "auth_failure"]
    )
)

login_system.add_transition(
    Transition(
        source=logged_in,
        target=logged_out,
        guards=[lambda e: e.name == "logout"]
    )
)

login_system.add_transition(
    Transition(
        source=authenticating,
        target=locked,
        guards=[lambda e: e.name == "too_many_attempts"]
    )
)

# Start system
login_system.start()

# Example usage
login_system.process_event(Event("attempt_login"))  # Logged out -> Authenticating
login_system.process_event(Event("auth_success"))   # Authenticating -> Logged in
login_system.process_event(Event("logout"))         # Logged in -> Logged out

# Example of failed login and lockout
login_system.process_event(Event("attempt_login"))     # Logged out -> Authenticating
login_system.process_event(Event("auth_failure"))      # Authenticating -> Logged out
login_system.process_event(Event("attempt_login"))     # Try again
login_system.process_event(Event("too_many_attempts")) # Lock account
```

### Game Character State Machine

Example of a complex game character state system with movement and combat states:

```python
from hsm.core.state_machine import StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.events import Event
from hsm.core.transitions import Transition

# Base state with common functionality
class CharacterState(State):
    def on_enter(self, event=None) -> None:
        print(f"Character is now {self.name}")

# Create states
idle = CharacterState("Idle")
walking = CharacterState("Walking")
running = CharacterState("Running")
jumping = CharacterState("Jumping")

# Combat states
combat = CompositeState("Combat")
attacking = CharacterState("Attacking")
blocking = CharacterState("Blocking")
dodging = CharacterState("Dodging")

# Create state machine
character = StateMachine(initial_state=idle)
character.add_state(walking)
character.add_state(running)
character.add_state(jumping)
character.add_state(combat)
character.add_state(attacking, parent=combat)  # First added child becomes initial state
character.add_state(blocking, parent=combat)
character.add_state(dodging, parent=combat)

# Add movement transitions
character.add_transition(
    Transition(
        source=idle,
        target=walking,
        guards=[lambda e: e.name == "start_walking"]
    )
)

character.add_transition(
    Transition(
        source=walking,
        target=running,
        guards=[lambda e: e.name == "start_running"]
    )
)

character.add_transition(
    Transition(
        source=running,
        target=walking,
        guards=[lambda e: e.name == "stop_running"]
    )
)

character.add_transition(
    Transition(
        source=walking,
        target=idle,
        guards=[lambda e: e.name == "stop_walking"]
    )
)

character.add_transition(
    Transition(
        source=idle,
        target=jumping,
        guards=[lambda e: e.name == "jump"]
    )
)

character.add_transition(
    Transition(
        source=jumping,
        target=idle,
        guards=[lambda e: e.name == "land"]
    )
)

# Add combat transitions
character.add_transition(
    Transition(
        source=idle,
        target=combat,  # First transition to combat state
        guards=[lambda e: e.name == "enter_combat"]
    )
)

character.add_transition(
    Transition(
        source=combat,
        target=attacking,  # Then transition to initial combat state
        guards=[lambda e: True]  # Always transition to initial state
    )
)

character.add_transition(
    Transition(
        source=attacking,
        target=blocking,
        guards=[lambda e: e.name == "block"]
    )
)

character.add_transition(
    Transition(
        source=blocking,
        target=dodging,
        guards=[lambda e: e.name == "dodge"]
    )
)

character.add_transition(
    Transition(
        source=dodging,
        target=idle,
        guards=[lambda e: e.name == "reset"]
    )
)

# Start the machine
character.start()

# Example usage - Movement
character.process_event(Event("start_walking"))  # Idle -> Walking
character.process_event(Event("start_running"))  # Walking -> Running
character.process_event(Event("stop_running"))   # Running -> Walking
character.process_event(Event("stop_walking"))   # Walking -> Idle

# Example usage - Jumping
character.process_event(Event("jump"))  # Idle -> Jumping
character.process_event(Event("land"))  # Jumping -> Idle

# Example usage - Combat
character.process_event(Event("enter_combat"))  # Idle -> Combat -> Attacking
character.process_event(Event("block"))         # Attacking -> Blocking
character.process_event(Event("dodge"))         # Blocking -> Dodging
character.process_event(Event("reset"))         # Dodging -> Idle
```

## Integration Examples

### Integration with FastAPI

Example of using GotState with FastAPI for a task management system:

```python
from fastapi import FastAPI, HTTPException
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition
from pydantic import BaseModel

app = FastAPI()

# Define models
class Task(BaseModel):
    id: int
    title: str
    status: str

# Define states
todo = State("todo")
in_progress = State("in_progress")
review = State("review")
done = State("done")

# Create state machine
task_machine = StateMachine(initial_state=todo)
task_machine.add_state(in_progress)
task_machine.add_state(review)
task_machine.add_state(done)

# Add transitions
task_machine.add_transition(
    Transition(
        source=todo,
        target=in_progress,
        guards=[lambda e: e.name == "start"]
    )
)

task_machine.add_transition(
    Transition(
        source=in_progress,
        target=review,
        guards=[lambda e: e.name == "submit"]
    )
)

task_machine.add_transition(
    Transition(
        source=review,
        target=done,
        guards=[lambda e: e.name == "approve"]
    )
)

task_machine.add_transition(
    Transition(
        source=review,
        target=in_progress,
        guards=[lambda e: e.name == "reject"]
    )
)

# Start machine
task_machine.start()

# Store tasks
tasks = {}

@app.post("/tasks/")
async def create_task(task: Task):
    tasks[task.id] = task
    return task

@app.put("/tasks/{task_id}/status")
async def update_task_status(task_id: int, action: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        task_machine.process_event(Event(action))
        tasks[task_id].status = task_machine.get_current_state().name
        return tasks[task_id]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task(task_id: int):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]
```

### Integration with SQLAlchemy

Example of persisting state machine state to a database:

```python
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.events import Event
from hsm.core.transitions import Transition
from hsm.core.hooks import HookProtocol

Base = declarative_base()
engine = create_engine('sqlite:///states.db')
Session = sessionmaker(bind=engine)

class StateMachineRecord(Base):
    __tablename__ = 'state_machines'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    current_state = Column(String)
    state_data = Column(JSON)

Base.metadata.create_all(engine)

class DatabaseHook(HookProtocol):
    def __init__(self, machine_id):
        self.machine_id = machine_id
    
    def on_enter(self, state):
        session = Session()
        record = session.query(StateMachineRecord).get(self.machine_id)
        if record:
            record.current_state = state.name
            record.state_data = state.data
            session.commit()
        session.close()

# Define states
state1 = State("state1")
state2 = State("state2")
state3 = State("state3")

# Create and persist machine
def create_machine(name: str) -> StateMachine:
    session = Session()
    record = StateMachineRecord(
        name=name,
        current_state="state1",
        state_data={}
    )
    session.add(record)
    session.commit()
    machine_id = record.id
    session.close()
    
    machine = StateMachine(
        initial_state=state1,
        hooks=[DatabaseHook(machine_id)]
    )
    machine.add_state(state2)
    machine.add_state(state3)
    
    machine.add_transition(
        Transition(
            source=state1,
            target=state2,
            guards=[lambda e: e.name == "advance"]
        )
    )
    
    machine.add_transition(
        Transition(
            source=state2,
            target=state3,
            guards=[lambda e: e.name == "advance"]
        )
    )
    
    return machine

# Load machine state
def load_machine(machine_id: int) -> StateMachine:
    session = Session()
    record = session.query(StateMachineRecord).get(machine_id)
    if not record:
        raise ValueError("Machine not found")
    
    # Create machine with saved state
    states = {"state1": state1, "state2": state2, "state3": state3}
    current_state = states[record.current_state]
    current_state.data = record.state_data
    
    machine = StateMachine(
        initial_state=current_state,
        hooks=[DatabaseHook(machine_id)]
    )
    
    for state in states.values():
        if state != current_state:
            machine.add_state(state)
    
    session.close()
    return machine

# Example usage
machine = create_machine("test_machine")
machine.start()
machine.process_event(Event("advance"))

# Later, load the machine
loaded_machine = load_machine(1)
loaded_machine.process_event(Event("advance"))