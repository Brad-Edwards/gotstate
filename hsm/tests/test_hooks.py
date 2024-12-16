# hsm/tests/test_hooks.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import logging
from typing import Any, Dict

import pytest

from hsm.core.errors import HSMError
from hsm.core.hooks import HookError, HookManager
from hsm.interfaces.abc import AbstractHook, AbstractTransition, StateID

# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class MockHook(AbstractHook):
    """
    A mock hook implementing AbstractHook for testing.

    Example:
        hook = MockHook()
        # Test calling hook methods like hook.on_enter("STATE")
    """

    def __init__(self, name: str = "MockHook", fail_on: Dict[str, bool] = None):
        self.name = name
        # A dictionary indicating which methods should fail
        self.fail_on = fail_on or {
            "on_enter": False,
            "on_exit": False,
            "pre_transition": False,
            "post_transition": False,
        }
        self.calls = {
            "on_enter": [],
            "on_exit": [],
            "pre_transition": [],
            "post_transition": [],
        }

    def on_enter(self, state_id: StateID) -> None:
        self.calls["on_enter"].append(state_id)
        if self.fail_on["on_enter"]:
            raise RuntimeError(f"{self.name} on_enter failed")

    def on_exit(self, state_id: StateID) -> None:
        self.calls["on_exit"].append(state_id)
        if self.fail_on["on_exit"]:
            raise RuntimeError("MockHook on_exit failed")

    def pre_transition(self, transition: AbstractTransition) -> None:
        self.calls["pre_transition"].append(transition)
        if self.fail_on["pre_transition"]:
            raise RuntimeError("MockHook pre_transition failed")

    def post_transition(self, transition: AbstractTransition) -> None:
        self.calls["post_transition"].append(transition)
        if self.fail_on["post_transition"]:
            raise RuntimeError("MockHook post_transition failed")


class MockTransition(AbstractTransition):
    """
    A mock transition implementing AbstractTransition for testing.
    """

    def get_source_state_id(self) -> str:
        return "SOURCE"

    def get_target_state_id(self) -> str:
        return "TARGET"

    def get_guard(self):
        return None

    def get_actions(self):
        return []

    def get_priority(self) -> int:
        return 0


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def hook_manager() -> HookManager:
    """Fixture to create and return a fresh HookManager instance."""
    return HookManager()


@pytest.fixture
def mock_hook() -> MockHook:
    """Fixture to create a default MockHook instance."""
    return MockHook()


# -----------------------------------------------------------------------------
# TESTS FOR HOOKERROR EXCEPTION
# -----------------------------------------------------------------------------


def test_hookerror_is_subclass_of_hsmerror() -> None:
    """
    Test that HookError is a subclass of HSMError.
    """
    assert issubclass(HookError, HSMError)


def test_hookerror_initialization() -> None:
    """
    Test initializing a HookError with a message and optional details.
    """
    err = HookError("Test hook error", details={"info": "some detail"})
    assert err.message == "Test hook error"
    assert err.details == {"info": "some detail"}


def test_hookerror_str_representation() -> None:
    """
    Test the string representation of HookError.
    """
    err = HookError("Another test error")
    assert str(err) == "Another test error"


def test_hookerror_without_details() -> None:
    """
    Test HookError with no details provided.
    """
    err = HookError("No details")
    assert err.details == {}


# -----------------------------------------------------------------------------
# TESTS FOR HOOKMANAGER CORE FUNCTIONALITY
# -----------------------------------------------------------------------------


def test_hookmanager_register_valid_hook(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test registering a valid hook that implements AbstractHook.
    No exceptions should be raised.
    """
    hook_manager.register_hook(mock_hook)
    # Call on_enter to ensure the hook is indeed registered
    hook_manager.call_on_enter("STATE_A")
    assert mock_hook.calls["on_enter"] == ["STATE_A"]


def test_hookmanager_register_invalid_hook(hook_manager: HookManager) -> None:
    """
    Test that attempting to register a non-AbstractHook object raises HookError.
    """
    not_a_hook = object()
    with pytest.raises(HookError) as exc_info:
        hook_manager.register_hook(not_a_hook)  # type: ignore
    assert "does not implement AbstractHook" in str(exc_info.value)


def test_hookmanager_unregister_hook(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test unregistering a previously registered hook.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.unregister_hook(mock_hook)
    # Hook should no longer be called
    hook_manager.call_on_enter("STATE_B")
    assert mock_hook.calls["on_enter"] == []


def test_hookmanager_unregister_nonexistent_hook(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Unregistering a hook that isn't registered should do nothing (no errors).
    """
    hook_manager.unregister_hook(mock_hook)  # no error expected
    # still can register and call it later
    hook_manager.register_hook(mock_hook)
    hook_manager.call_on_enter("STATE_C")
    assert mock_hook.calls["on_enter"] == ["STATE_C"]


# -----------------------------------------------------------------------------
# TESTING HOOK METHOD INVOCATIONS
# -----------------------------------------------------------------------------


def test_hookmanager_call_on_enter(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test that call_on_enter invokes on_enter on registered hooks.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.call_on_enter("STATE_X")
    assert mock_hook.calls["on_enter"] == ["STATE_X"]


def test_hookmanager_call_on_exit(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test that call_on_exit invokes on_exit on registered hooks.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.call_on_exit("STATE_Y")
    assert mock_hook.calls["on_exit"] == ["STATE_Y"]


def test_hookmanager_call_pre_transition(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test that call_pre_transition invokes pre_transition on registered hooks.
    """
    tran = MockTransition()
    hook_manager.register_hook(mock_hook)
    hook_manager.call_pre_transition(tran)
    assert mock_hook.calls["pre_transition"] == [tran]


def test_hookmanager_call_post_transition(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test that call_post_transition invokes post_transition on registered hooks.
    """
    tran = MockTransition()
    hook_manager.register_hook(mock_hook)
    hook_manager.call_post_transition(tran)
    assert mock_hook.calls["post_transition"] == [tran]


# -----------------------------------------------------------------------------
# TEST ERROR HANDLING WITH HOOK EXECUTIONS
# -----------------------------------------------------------------------------


def test_hookmanager_handles_hook_exceptions(hook_manager: HookManager, caplog: pytest.LogCaptureFixture) -> None:
    """
    Test that if a hook method raises an exception, HookManager catches and logs it,
    without interrupting execution or raising the exception.
    """
    failing_hook = MockHook(
        fail_on={"on_enter": True, "on_exit": False, "pre_transition": False, "post_transition": False}
    )
    hook_manager.register_hook(failing_hook)
    with caplog.at_level(logging.ERROR):
        hook_manager.call_on_enter("STATE_FAIL")
    assert "MockHook on_enter failed" in caplog.text


def test_hookmanager_multiple_hooks_with_one_failing(
    hook_manager: HookManager, caplog: pytest.LogCaptureFixture
) -> None:
    failing_hook = MockHook(name="FailingHook", fail_on={"on_enter": True})
    good_hook = MockHook(name="GoodHook")
    hook_manager.register_hook(good_hook)
    hook_manager.register_hook(failing_hook)

    with caplog.at_level(logging.ERROR):
        hook_manager.call_on_enter("STATE_TEST")

    # GoodHook still runs even though FailingHook fails
    assert good_hook.calls["on_enter"] == ["STATE_TEST"]
    # Now this assertion should pass because the hook name is in the raised error.
    assert "FailingHook on_enter failed" in caplog.text


def test_hookmanager_no_hooks(hook_manager: HookManager) -> None:
    """
    Calling hook manager methods with no registered hooks should do nothing and not fail.
    """
    # No hooks are registered
    hook_manager.call_on_enter("NO_HOOKS_STATE")
    # Nothing to assert directly, just ensure no errors occur.


# -----------------------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------------------


def test_hookmanager_invalid_state_id(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    If an invalid (empty) state_id is passed, hooks are still called.
    HookManager doesn't validate state_id. Test no errors occur.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.call_on_enter("")
    assert mock_hook.calls["on_enter"] == [""]


def test_hookmanager_transition_none(
    hook_manager: HookManager, mock_hook: MockHook, caplog: pytest.LogCaptureFixture
) -> None:
    """
    If a None transition is passed to pre_transition or post_transition,
    no exceptions are raised. Hooks are still called with None.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.call_pre_transition(None)  # type: ignore
    hook_manager.call_post_transition(None)  # type: ignore

    # Hooks should record these calls
    assert mock_hook.calls["pre_transition"] == [None]
    assert mock_hook.calls["post_transition"] == [None]


# -----------------------------------------------------------------------------
# INTEGRATION TESTS (LOGGING)
# -----------------------------------------------------------------------------


def test_hookmanager_logging_when_no_exceptions(
    hook_manager: HookManager, mock_hook: MockHook, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test that HookManager doesn't log errors when no exceptions occur.
    """
    hook_manager.register_hook(mock_hook)
    with caplog.at_level(logging.ERROR):
        hook_manager.call_on_enter("STATE_LOG_OK")

    # No errors should be logged
    assert caplog.text == ""


def test_hookerror_chaining() -> None:
    """
    Test raising a HookError from within another exception to check exception chaining.
    """
    try:
        try:
            raise ValueError("Original cause")
        except ValueError as e:
            raise HookError("Hook error with chain") from e
    except HookError as hook_exc:
        assert hook_exc.__cause__ is not None
        assert isinstance(hook_exc.__cause__, ValueError)
        assert str(hook_exc.__cause__) == "Original cause"
        assert str(hook_exc) == "Hook error with chain"


def test_hookmanager_hook_raises_hookerror(hook_manager: HookManager, caplog: pytest.LogCaptureFixture) -> None:
    """
    Test that if a hook raises a HookError, the HookManager logs it and does not re-raise.
    """

    class HookErrorRaisingHook(AbstractHook):
        def on_enter(self, state_id: StateID) -> None:
            raise HookError("Error from hook", details={"state_id": state_id})

        def on_exit(self, state_id: StateID) -> None:
            pass

        def pre_transition(self, transition: AbstractTransition) -> None:
            pass

        def post_transition(self, transition: AbstractTransition) -> None:
            pass

    failing_hook = HookErrorRaisingHook()
    hook_manager.register_hook(failing_hook)
    with caplog.at_level(logging.ERROR):
        hook_manager.call_on_enter("STATE_FOR_HOOKERROR")
    assert "Error from hook" in caplog.text
    assert "state_id" in caplog.text  # Check that details are logged


def test_hookmanager_multiple_registrations(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test registering the same hook multiple times.
    The hook should be called once per registration.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.register_hook(mock_hook)  # register same hook again
    hook_manager.call_on_enter("MULTIPLE_HOOK_STATE")
    # Hook should be called twice
    assert mock_hook.calls["on_enter"] == ["MULTIPLE_HOOK_STATE", "MULTIPLE_HOOK_STATE"]


def test_hookmanager_on_exit_with_none_state_id(
    hook_manager: HookManager, mock_hook: MockHook, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test calling on_exit with None as state_id. This shouldn't raise an error.
    Hooks should still be called with None passed in.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.call_on_exit(None)  # type: ignore
    assert mock_hook.calls["on_exit"] == [None]
    # No errors expected
    assert "ERROR" not in caplog.text
