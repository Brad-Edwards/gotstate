# gotstate

[![Security](https://github.com/KeplerOps/gotstate/actions/workflows/security.yml/badge.svg)](https://github.com/KeplerOps/gotstate/actions/workflows/security.yml)
[![Quality](https://github.com/KeplerOps/gotstate/actions/workflows/sonar.yml/badge.svg)](https://github.com/KeplerOps/gotstate/actions/workflows/sonar.yml)
[![Tests](https://github.com/KeplerOps/gotstate/actions/workflows/test.yml/badge.svg)](https://github.com/KeplerOps/gotstate/actions/workflows/test.yml)
[![Lint](https://github.com/KeplerOps/gotstate/actions/workflows/lint.yml/badge.svg)](https://github.com/KeplerOps/gotstate/actions/workflows/lint.yml)

A hierarchical finite state machine (HFSM) library for Python, focusing on reliability and ease of use.

## Features

- Hierarchical state machines with composite states
- Type-safe state and event handling
- Thread-safe event processing
- Guard conditions and transition actions
- State data management with lifecycle hooks
- Timeout events
- History states (both shallow and deep)
- Error handling
- Activation hooks for monitoring
- Plugin system

## Status

**Version 1.0**

Features:
- Full test coverage
- Type hints
- Input validation
- Error handling
- API documentation

## Design Philosophy

`gotstate` is designed with the following principles:

- **Safety**: Runtime validation and type checking
- **Clarity**: Intuitive API design
- **Reliability**: Built for real-world applications
- **Performance**: Minimal overhead
- **Flexibility**: Extensible through plugins

## Example

```python
from hsm.core import StateMachine, State, Event, Transition

# Define states
class Idle(State):
    pass

class Running(State):
    pass

# Define events
class Start(Event):
    pass

class Stop(Event):
    pass

# Create state machine
sm = StateMachine("example")
idle = sm.add_state(Idle("idle"))
running = sm.add_state(Running("running"))

# Add transitions
sm.add_transition(Transition(idle, running, Start))
sm.add_transition(Transition(running, idle, Stop))

# Initialize and use
sm.initialize()
assert sm.current_state == idle
sm.handle_event(Start())
assert sm.current_state == running
```

## Installation

Install using pip:

```bash
pip install gotstate
```

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for full dependencies

## Documentation

Documentation is available in the `docs/` directory:
- API Reference
- Usage Guide
- Examples

## License

Licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines in CONTRIBUTING.md.

## Security

This package follows Python security best practices.
