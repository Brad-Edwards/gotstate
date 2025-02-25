# GotState: A Robust State Machine Library for Python

[![PyPI version](https://img.shields.io/pypi/v/gotstate.svg)](https://pypi.org/project/gotstate/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gotstate.svg)](https://pypi.org/project/gotstate/)
[![Build Status](https://github.com/username/gotstate/workflows/Run%20Tests/badge.svg)](https://github.com/username/gotstate/actions)
[![License](https://img.shields.io/pypi/l/gotstate.svg)](https://github.com/username/gotstate/blob/main/LICENSE)

GotState is a powerful and flexible state machine library for Python, designed to help you manage complex state-based systems with ease.

## Features

- **Simple and intuitive API** for creating and managing state machines
- Support for **hierarchical state machines** with nesting and inheritance
- **Event-driven architecture** for responsive state transitions
- **Entry and exit actions** for states
- **Guard conditions** for transitions
- **Composite states** and **regions** for complex state machines
- **Asynchronous support** through the AsyncStateMachine extension

## Installation

```bash
pip install gotstate
```

Or with Poetry:

```bash
poetry add gotstate
```

## Quick Example

Here's a simple traffic light implementation:

```python
from gotstate.core.statemachine import StateMachine
from gotstate.core.state import State
from time import sleep

def create_traffic_light():
    # Create a state machine
    traffic_light = StateMachine("TrafficLight")
    
    # Create states
    red_state = State("RED")
    yellow_state = State("YELLOW")
    green_state = State("GREEN")
    
    # Add states to the machine
    traffic_light.add_state(red_state)
    traffic_light.add_state(yellow_state)
    traffic_light.add_state(green_state)
    
    # Set up entry actions
    red_state.add_entry_action(lambda: print("Red light: Stop!"))
    yellow_state.add_entry_action(lambda: print("Yellow light: Prepare to stop!"))
    green_state.add_entry_action(lambda: print("Green light: Go!"))
    
    # Set up transitions
    traffic_light.add_transition(red_state, green_state, "TIMER")
    traffic_light.add_transition(green_state, yellow_state, "TIMER")
    traffic_light.add_transition(yellow_state, red_state, "TIMER")
    
    # Set initial state
    traffic_light.set_initial_state(red_state)
    
    return traffic_light

def run_traffic_light(cycles=3):
    traffic_light = create_traffic_light()
    traffic_light.start()
    
    # Run for specified number of cycles
    for _ in range(cycles * 3):
        sleep(2)  # Wait for 2 seconds
        traffic_light.process_event("TIMER")
    
    traffic_light.stop()

if __name__ == "__main__":
    run_traffic_light()
```

## Advanced Usage

### Hierarchical State Machines

```python
# Coming soon
```

### Asynchronous State Machines

```python
# Coming soon
```

## Documentation

For full documentation, visit [docs.gotstate.io](https://docs.gotstate.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
