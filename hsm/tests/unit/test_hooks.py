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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockTransition):
            return NotImplemented
        return (
            self.get_source_state_id() == other.get_source_state_id()
            and self.get_target_state_id() == other.get_target_state_id()
            and self.get_guard() == other.get_guard()
            and self.get_actions() == other.get_actions()
            and self.get_priority() == other.get_priority()
        )

    def __repr__(self) -> str:
        return f"MockTransition(source={self.get_source_state_id()}, target={self.get_target_state_id()})"


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
# TEST HELPERS
# -----------------------------------------------------------------------------


def assert_hook_method_called(
    hook_manager: HookManager, mock_hook: MockHook, method_name: str, arg: Any, expected_calls: list
) -> None:
    """Helper to test hook method invocation."""
    hook_manager.register_hook(mock_hook)
    getattr(hook_manager, f"call_{method_name}")(arg)
    assert mock_hook.calls[method_name] == expected_calls


def create_failing_hook(failing_method: str, name: str = "FailingHook") -> MockHook:
    """Helper to create a hook that fails on a specific method."""
    fail_on = {
        "on_enter": False,
        "on_exit": False,
        "pre_transition": False,
        "post_transition": False,
    }
    fail_on[failing_method] = True
    return MockHook(name=name, fail_on=fail_on)


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


@pytest.mark.parametrize(
    "method_name,arg,expected",
    [
        ("on_enter", "STATE_X", ["STATE_X"]),
        ("on_exit", "STATE_Y", ["STATE_Y"]),
        ("pre_transition", MockTransition(), [MockTransition()]),
        ("post_transition", MockTransition(), [MockTransition()]),
    ],
)
def test_hookmanager_method_calls(
    hook_manager: HookManager, mock_hook: MockHook, method_name: str, arg: Any, expected: list
) -> None:
    """
    Test that hook manager methods properly invoke corresponding hook methods.
    """
    assert_hook_method_called(hook_manager, mock_hook, method_name, arg, expected)


# -----------------------------------------------------------------------------
# TEST ERROR HANDLING WITH HOOK EXECUTIONS
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failing_method,state_id,expected_error",
    [
        ("on_enter", "STATE_FAIL", "FailingHook on_enter failed"),
        ("on_exit", "STATE_FAIL", "MockHook on_exit failed"),
        ("pre_transition", MockTransition(), "MockHook pre_transition failed"),
        ("post_transition", MockTransition(), "MockHook post_transition failed"),
    ],
)
def test_hookmanager_error_handling(
    hook_manager: HookManager, caplog: pytest.LogCaptureFixture, failing_method: str, state_id: Any, expected_error: str
) -> None:
    """
    Test error handling for various hook methods.
    """
    failing_hook = create_failing_hook(failing_method)
    hook_manager.register_hook(failing_hook)

    with caplog.at_level(logging.ERROR):
        getattr(hook_manager, f"call_{failing_method}")(state_id)
    assert expected_error in caplog.text


def test_hookmanager_multiple_hooks_with_one_failing(
    hook_manager: HookManager, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test that when one hook fails, others still execute.
    """
    failing_hook = create_failing_hook("on_enter", "FailingHook")
    good_hook = MockHook(name="GoodHook")
    hook_manager.register_hook(good_hook)
    hook_manager.register_hook(failing_hook)

    with caplog.at_level(logging.ERROR):
        hook_manager.call_on_enter("STATE_TEST")

    assert good_hook.calls["on_enter"] == ["STATE_TEST"]
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

    failing_hook = HookErrorRaisingHook()
    hook_manager.register_hook(failing_hook)
    with caplog.at_level(logging.ERROR):
        hook_manager.call_on_enter("STATE_FOR_HOOKERROR")
    assert "Error from hook" in caplog.text
    assert "state_id" in caplog.text  # Check that details are logged


def test_hookmanager_multiple_registrations(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test that registering the same hook multiple times has no effect (idempotent).
    Each hook should only be called once regardless of how many times it's registered.
    """
    hook_manager.register_hook(mock_hook)
    hook_manager.register_hook(mock_hook)  # Should have no effect
    hook_manager.call_on_enter("MULTIPLE_HOOK_STATE")
    # Hook should be called only once due to idempotent registration
    assert mock_hook.calls["on_enter"] == ["MULTIPLE_HOOK_STATE"]


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


# -----------------------------------------------------------------------------
# HOOK ORDER AND STATE TESTS
# -----------------------------------------------------------------------------


def test_hookmanager_hook_execution_order(hook_manager: HookManager) -> None:
    """
    Test that hooks are called in the order they were registered.
    """
    execution_order = []

    class OrderedHook(AbstractHook):
        def __init__(self, name: str):
            self.name = name

        def on_enter(self, state_id: StateID) -> None:
            execution_order.append(self.name)

        def on_exit(self, state_id: StateID) -> None:
            execution_order.append(self.name)

        def pre_transition(self, transition: AbstractTransition) -> None:
            execution_order.append(self.name)

        def post_transition(self, transition: AbstractTransition) -> None:
            execution_order.append(self.name)

    # Register hooks in specific order
    hook1 = OrderedHook("hook1")
    hook2 = OrderedHook("hook2")
    hook3 = OrderedHook("hook3")

    hook_manager.register_hook(hook1)
    hook_manager.register_hook(hook2)
    hook_manager.register_hook(hook3)

    # Test all hook methods
    hook_manager.call_on_enter("STATE")
    assert execution_order == ["hook1", "hook2", "hook3"]

    execution_order.clear()
    hook_manager.call_on_exit("STATE")
    assert execution_order == ["hook1", "hook2", "hook3"]

    # Test order preservation after unregister/re-register
    hook_manager.unregister_hook(hook2)
    execution_order.clear()
    hook_manager.call_on_enter("STATE")
    assert execution_order == ["hook1", "hook3"]

    hook_manager.register_hook(hook2)  # Re-register at end
    execution_order.clear()
    hook_manager.call_on_enter("STATE")
    assert execution_order == ["hook1", "hook3", "hook2"]


def test_hookmanager_hook_state_isolation(hook_manager: HookManager) -> None:
    """
    Test that hooks maintain independent state and failures don't affect other hooks' state.
    """

    class StatefulHook(AbstractHook):
        def __init__(self, name: str, fail_on_count: int = -1):
            self.name = name
            self.call_count = 0
            self.fail_on_count = fail_on_count

        def on_enter(self, state_id: StateID) -> None:
            self.call_count += 1
            if self.call_count == self.fail_on_count:
                raise RuntimeError(f"{self.name} failed on count {self.call_count}")

    # Create hooks with different failure conditions
    hook1 = StatefulHook("hook1")  # Never fails
    hook2 = StatefulHook("hook2", fail_on_count=2)  # Fails on second call
    hook3 = StatefulHook("hook3")  # Never fails

    hook_manager.register_hook(hook1)
    hook_manager.register_hook(hook2)
    hook_manager.register_hook(hook3)

    # First call - all succeed
    hook_manager.call_on_enter("STATE1")
    assert hook1.call_count == 1
    assert hook2.call_count == 1
    assert hook3.call_count == 1

    # Second call - hook2 fails but others continue
    hook_manager.call_on_enter("STATE2")
    assert hook1.call_count == 2  # Succeeded
    assert hook2.call_count == 2  # Failed but count still incremented
    assert hook3.call_count == 2  # Succeeded

    # Third call - all continue from previous state
    hook_manager.call_on_enter("STATE3")
    assert hook1.call_count == 3
    assert hook2.call_count == 3
    assert hook3.call_count == 3


def test_hookmanager_hook_registration_idempotency(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test that registering the same hook multiple times has no effect (idempotent),
    and unregistering it removes all instances.
    """
    # Register same hook multiple times
    hook_manager.register_hook(mock_hook)
    hook_manager.register_hook(mock_hook)  # Should have no effect
    hook_manager.register_hook(mock_hook)  # Should have no effect

    # Should be called only once since registration is idempotent
    hook_manager.call_on_enter("STATE")
    assert mock_hook.calls["on_enter"] == ["STATE"]

    # Unregister should remove the hook entirely
    hook_manager.unregister_hook(mock_hook)
    hook_manager.call_on_enter("STATE2")
    assert mock_hook.calls["on_enter"] == ["STATE"]  # No new calls


def test_hookmanager_multiple_distinct_hooks(hook_manager: HookManager) -> None:
    """
    Test registering multiple distinct hooks of the same type.
    Each hook should be registered and called independently.
    """
    hook1 = MockHook(name="hook1")
    hook2 = MockHook(name="hook2")  # Different instance
    hook3 = MockHook(name="hook3")  # Different instance

    # Register multiple distinct hooks
    hook_manager.register_hook(hook1)
    hook_manager.register_hook(hook2)
    hook_manager.register_hook(hook3)

    # Each hook should be called once
    hook_manager.call_on_enter("STATE")
    assert hook1.calls["on_enter"] == ["STATE"]
    assert hook2.calls["on_enter"] == ["STATE"]
    assert hook3.calls["on_enter"] == ["STATE"]

    # Unregistering one hook shouldn't affect others
    hook_manager.unregister_hook(hook2)
    hook_manager.call_on_enter("STATE2")
    assert hook1.calls["on_enter"] == ["STATE", "STATE2"]
    assert hook2.calls["on_enter"] == ["STATE"]  # No new calls
    assert hook3.calls["on_enter"] == ["STATE", "STATE2"]


# -----------------------------------------------------------------------------
# ADDITIONAL EDGE CASES AND CORNER CASES
# -----------------------------------------------------------------------------


def test_hookmanager_hook_return_values(hook_manager: HookManager) -> None:
    """
    Test that hook method return values are ignored.
    Hooks should be side-effect only.
    """

    class ReturningHook(AbstractHook):
        def on_enter(self, state_id: StateID) -> str:
            return "This return value should be ignored"  # type: ignore

        def on_exit(self, state_id: StateID) -> int:
            return 42  # type: ignore

        def pre_transition(self, transition: AbstractTransition) -> bool:
            return True  # type: ignore

        def post_transition(self, transition: AbstractTransition) -> list:
            return []  # type: ignore

    hook = ReturningHook()
    hook_manager.register_hook(hook)

    # These should not raise any errors despite return values
    hook_manager.call_on_enter("STATE")
    hook_manager.call_on_exit("STATE")
    hook_manager.call_pre_transition(MockTransition())
    hook_manager.call_post_transition(MockTransition())


def test_hookmanager_complex_state_ids(hook_manager: HookManager, mock_hook: MockHook) -> None:
    """
    Test that complex state IDs are passed unmodified to hooks.
    """
    complex_state = ("STATE", 123, {"nested": True})  # type: ignore
    hook_manager.register_hook(mock_hook)
    hook_manager.call_on_enter(complex_state)
    assert mock_hook.calls["on_enter"] == [complex_state]

    hook_manager.call_on_exit(complex_state)
    assert mock_hook.calls["on_exit"] == [complex_state]


def test_hookmanager_register_none_hook(hook_manager: HookManager) -> None:
    """
    Test that attempting to register None as a hook raises HookError.
    """
    with pytest.raises(HookError) as exc_info:
        hook_manager.register_hook(None)  # type: ignore
    assert "does not implement AbstractHook" in str(exc_info.value)


class DynamicHook:
    """A hook that can dynamically lose its interface compliance."""

    def __init__(self):
        self.compliant = True

    def on_enter(self, state_id: StateID) -> None:
        if not self.compliant:
            raise AttributeError("No longer compliant")

    def on_exit(self, state_id: StateID) -> None:
        if not self.compliant:
            raise AttributeError("No longer compliant")

    def pre_transition(self, transition: AbstractTransition) -> None:
        if not self.compliant:
            raise AttributeError("No longer compliant")

    def post_transition(self, transition: AbstractTransition) -> None:
        if not self.compliant:
            raise AttributeError("No longer compliant")


def test_hookmanager_dynamic_hook_compliance(hook_manager: HookManager, caplog: pytest.LogCaptureFixture) -> None:
    """
    Test behavior when a hook dynamically loses interface compliance.
    HookManager should handle the failure gracefully.
    """
    hook = DynamicHook()
    hook_manager.register_hook(hook)  # type: ignore

    # Initially compliant
    hook_manager.call_on_enter("STATE1")
    assert "ERROR" not in caplog.text

    # Make hook non-compliant
    hook.compliant = False
    with caplog.at_level(logging.ERROR):
        hook_manager.call_on_enter("STATE2")
    assert "No longer compliant" in caplog.text
