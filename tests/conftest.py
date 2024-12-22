# tests/conftest.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import threading
from unittest.mock import MagicMock

import pytest

from gotstate.core.hooks import Hook


@pytest.fixture
def simple_state():
    """A minimal State object for basic testing."""
    from gotstate.core.states import State

    return State(name="Idle")


@pytest.fixture
def another_state():
    """Another State object for transitions."""
    from gotstate.core.states import State

    return State(name="Active")


@pytest.fixture
def event_queue():
    """A simple, non-priority event queue."""
    from gotstate.runtime.event_queue import EventQueue

    return EventQueue(priority=False)


@pytest.fixture
def hook():
    """A hook object for testing hooks."""
    return Hook()


@pytest.fixture
def timeout_event():
    """A timeout event for testing timers and timeouts."""
    from gotstate.core.events import TimeoutEvent

    return TimeoutEvent(name="Timeout", deadline=10.0)


@pytest.fixture
def basic_event():
    """A generic event for testing transitions."""
    from gotstate.core.events import Event

    return Event(name="TestEvent")


@pytest.fixture
def validator():
    """A default Validator that passes everything for simplicity."""
    from gotstate.core.validations import Validator

    v = Validator()
    # Could mock methods if needed, but a no-op validator is fine.
    return v


@pytest.fixture
def mock_actions():
    """Return a list of action mocks."""
    action = MagicMock()
    return [action]


@pytest.fixture
def mock_guards():
    """Return a list of guard mocks that always pass."""
    guard = MagicMock(return_value=True)
    return [guard]


@pytest.fixture
def machine_factory(validator):
    """Returns a factory function to create a simple state machine for tests."""
    from gotstate.core.state_machine import StateMachine
    from gotstate.core.states import State

    def _factory():
        initial = State("Initial")
        return StateMachine(initial_state=initial, validator=validator, hooks=[])

    return _factory


@pytest.fixture
def dummy_state():
    """A basic State with no entry/exit actions."""
    from gotstate.core.states import State

    return State(name="Dummy")


@pytest.fixture
def dummy_event():
    """A generic Event for testing."""
    from gotstate.core.events import Event

    return Event("TestEvent")


@pytest.fixture
def dummy_guard():
    """A guard function that always returns True."""
    return lambda event: True


@pytest.fixture
def dummy_action():
    """A simple action function (no-op)."""
    return lambda event: None


@pytest.fixture
def dummy_hooks():
    """A list of hook mocks for testing HookManager."""
    hook = MagicMock()
    # hook should have on_enter(state), on_exit(state), on_error(error)
    hook.on_enter = MagicMock()
    hook.on_exit = MagicMock()
    hook.on_error = MagicMock()
    return [hook]


@pytest.fixture
def error_classes():
    """Provides a tuple of error classes for quick reference."""
    from gotstate.core.errors import HSMError, StateNotFoundError, TransitionError, ValidationError

    return (HSMError, StateNotFoundError, TransitionError, ValidationError)


@pytest.fixture
def mock_machine():
    """A mock state machine for runtime tests."""
    m = MagicMock()
    m.current_state = MagicMock(name="CurrentState")
    return m


@pytest.fixture
def mock_event_queue():
    """A mock event queue for executor tests."""
    eq = MagicMock()
    return eq


@pytest.fixture
def mock_event():
    """A mock event object."""
    e = MagicMock()
    e.name = "MockEvent"
    return e


@pytest.fixture(autouse=True)
def cleanup_threads():
    yield
    # Cleanup any remaining threads after each test
    for thread in threading.enumerate():
        if thread != threading.current_thread() and thread.is_alive():
            thread.join(timeout=1.0)
