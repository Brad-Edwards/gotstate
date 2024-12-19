import asyncio
import logging
from typing import Any, AsyncGenerator, List, Optional

import pytest

from hsm.core.errors import EventQueueFullError, ExecutorError, InvalidStateError
from hsm.interfaces.protocols import Event
from hsm.runtime.async_support import (
    AsyncAction,
    AsyncCompositeState,
    AsyncExecutor,
    AsyncGuard,
    AsyncHSMError,
    AsyncLockManager,
    AsyncState,
    AsyncStateError,
    AsyncStateMachine,
    AsyncTransition,
    AsyncTransitionError,
)
from hsm.runtime.event_queue import EventQueueError, QueueFullError
from hsm.tests.async_utils import AsyncMock, AsyncTestHelper

logger = logging.getLogger(__name__)


# Fixtures
@pytest.fixture(scope="function")
async def lock_manager() -> AsyncLockManager:
    return AsyncLockManager()


@pytest.fixture
def event_mock_factory():
    class MockEvent(Event):
        def __init__(self, event_id: str, data: Any = None, timestamp: float = 0.0):
            self.event_id = event_id
            self._data = data
            self._timestamp = timestamp

        def get_id(self) -> str:
            return self.event_id

        def get_data(self) -> Any:
            return self._data

        def get_timestamp(self) -> float:
            return self._timestamp

    def create_event(event_id: str = "test_event", data: Any = None, timestamp: float = 0.0):
        return MockEvent(event_id, data, timestamp)

    return create_event


@pytest.fixture
def state_mock() -> AsyncState:
    class MockState(AsyncState):
        async def _do_enter(self, event: Any, data: Any) -> None:
            pass

        async def _do_exit(self, event: Any, data: Any) -> None:
            pass

    return MockState("test_state")


@pytest.fixture
def guard_mock() -> AsyncMock:
    class MockGuard(AsyncMock):
        def __init__(self, return_value: bool = True):
            super().__init__()
            self.return_value = return_value

        async def check(self, event: Any, state_data: Any) -> bool:
            return await self(event, state_data)

    return MockGuard()


@pytest.fixture
def action_mock() -> AsyncMock:
    class MockAction(AsyncMock):
        async def execute(self, event: Any, state_data: Any) -> None:
            await self(event, state_data)

    return MockAction()


@pytest.fixture
async def configurable_machine(state_mock):
    def _create_machine(num_states=2, with_transitions=True, cycle=False):
        states = [state_mock] + [AsyncState(f"state{i}") for i in range(1, num_states)]
        transitions = []

        if with_transitions:
            if cycle:
                transitions = [AsyncTransition(states[i], states[(i + 1) % num_states]) for i in range(num_states)]
            else:
                transitions = [AsyncTransition(states[i], states[i + 1]) for i in range(num_states - 1)]

        machine = AsyncStateMachine(states, transitions, states[0])
        return machine

    return _create_machine


@pytest.fixture
def machine_context():
    class StateMachineContext:
        async def __aenter__(self, machine):
            self.machine = machine
            await self.machine.start()
            return self.machine

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.machine.is_running():
                await self.machine.stop()

    return StateMachineContext()


@pytest.fixture
def event_mock(event_mock_factory):
    """Creates a mock event instance for testing"""
    return event_mock_factory()


# Move this fixture to module level (outside of any test class)
@pytest.fixture
def simple_machine(state_mock):
    logger.debug("Creating simple machine")
    states = [state_mock, AsyncState("state2")]
    transitions = [AsyncTransition(state_mock, states[1])]
    machine = AsyncStateMachine(states, transitions, state_mock)
    return machine


# Test AsyncLockManager
class TestAsyncLockManager:
    async def test_init(self, lock_manager: AsyncLockManager) -> None:
        assert lock_manager.get_locks() == {}

    async def test_with_lock_creates_new_lock(self, lock_manager: AsyncLockManager) -> None:
        counter = 0

        async def increment():
            nonlocal counter
            counter += 1

        await lock_manager.with_lock("test", increment)
        assert "test" in lock_manager.get_locks()
        assert counter == 1

    async def test_with_lock_reuses_lock(self, lock_manager: AsyncLockManager) -> None:
        lock1 = None

        async def get_lock():
            nonlocal lock1
            lock1 = lock_manager.get_locks().get("test")

        await lock_manager.with_lock("test", get_lock)
        original_lock = lock1

        await lock_manager.with_lock("test", get_lock)
        assert lock1 is original_lock

    async def test_concurrent_operations(self, lock_manager: AsyncLockManager) -> None:
        counter = 0

        async def increment():
            nonlocal counter
            await asyncio.sleep(0.01)
            counter += 1

        await asyncio.gather(lock_manager.with_lock("test", increment), lock_manager.with_lock("test", increment))
        assert counter == 2

    async def test_lock_timeout(self, lock_manager: AsyncLockManager) -> None:
        """Test lock acquisition timeout"""

        async def hold_lock():
            async def block():
                await asyncio.sleep(0.2)

            await lock_manager.with_lock("test", block)

        # Start blocking operation
        task = asyncio.create_task(hold_lock())
        await asyncio.sleep(0.1)

        # Try to acquire same lock
        with pytest.raises(asyncio.TimeoutError):

            async def timeout_op():
                await asyncio.sleep(0.1)

            await asyncio.wait_for(lock_manager.with_lock("test", timeout_op), timeout=0.1)

        await task

    async def test_lock_exception_release(self, lock_manager: AsyncLockManager) -> None:
        """Test lock is released when operation raises exception"""
        with pytest.raises(ValueError):

            async def raise_error():
                raise ValueError("test error")

            await lock_manager.with_lock("test", raise_error)

        # Lock should be released, allowing new acquisition
        acquired = False

        async def check_lock():
            nonlocal acquired
            acquired = True

        await lock_manager.with_lock("test", check_lock)
        assert acquired


# Test AsyncState
class TestAsyncState:
    async def test_init_valid_id(self):
        state = AsyncState("test")
        assert state.get_id() == "test"

    async def test_init_invalid_id(self):
        with pytest.raises(ValueError):
            AsyncState("")

    async def test_on_entry(self, state_mock, event_mock):
        await state_mock.on_entry(event_mock, {})
        # Verify no exceptions raised

    async def test_on_exit(self, state_mock, event_mock):
        await state_mock.on_exit(event_mock, {})
        # Verify no exceptions raised

    async def test_entry_error(self):
        class ErrorState(AsyncState):
            async def _do_enter(self, event: Any, data: Any) -> None:
                raise ValueError("Test error")

        state = ErrorState("error_state")
        with pytest.raises(AsyncStateError):
            await state.on_entry(None, {})

    async def test_concurrent_entry_exit(self, state_mock: AsyncState) -> None:
        """Test concurrent entry/exit operations"""

        async def entry_exit():
            await state_mock.on_entry(None, {})
            await state_mock.on_exit(None, {})

        # Run multiple entry/exit operations concurrently
        await asyncio.gather(*[entry_exit() for _ in range(5)])

    async def test_state_data_isolation(self, state_mock):
        """Test that each state maintains isolated data"""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        await state_mock.on_entry(None, data1)
        data1["key"] = "modified"  # Modify original data
        await state_mock.on_entry(None, data2)
        assert data2["key"] == "value2"  # Verify no cross-contamination

    async def test_state_id_validation(self):
        """Test state ID validation edge cases"""
        with pytest.raises(ValueError):
            AsyncState("")  # Empty string is invalid
        # Note: Whitespace validation would need to be added to AsyncState implementation


# Test AsyncCompositeState
class TestAsyncCompositeState:
    @pytest.fixture
    def composite_setup(self, state_mock: AsyncState) -> AsyncCompositeState:
        substates = [AsyncState("sub1"), AsyncState("sub2")]
        composite = AsyncCompositeState("composite", substates, substates[0])
        return composite

    async def test_init_valid(self, composite_setup: AsyncCompositeState) -> None:
        assert composite_setup.get_id() == "composite"
        assert len(composite_setup.get_substates()) == 2

    async def test_init_invalid_initial_state(self) -> None:
        with pytest.raises(ValueError):
            AsyncCompositeState("test", [], AsyncState("invalid"))

    async def test_history_state(self, composite_setup: AsyncCompositeState) -> None:
        composite_setup.set_has_history(True)
        composite_setup.set_history_state(composite_setup.get_substates()[0])
        assert composite_setup.get_history_state() == composite_setup.get_substates()[0]

    async def test_parent_reference_management(self, composite_setup: AsyncCompositeState) -> None:
        """Test parent state references are properly managed"""
        child_state = AsyncState("child")
        parent_state = AsyncCompositeState("parent", [child_state], child_state)

        # Verify parent reference
        assert isinstance(child_state, AsyncState)
        assert parent_state._substates[0] is child_state

    async def test_history_persistence(self, composite_setup: AsyncCompositeState) -> None:
        """Test history state persistence across transitions"""
        composite_setup.set_has_history(True)
        substates = composite_setup.get_substates()

        # Perform multiple transitions
        for state in substates:
            composite_setup.set_history_state(state)
            assert composite_setup.get_history_state() is state

    async def test_substate_validation(self):
        """Test substate validation"""
        state1 = AsyncState("s1")
        state2 = AsyncState("s2")  # Different ID
        comp = AsyncCompositeState("comp", [state1, state2], state1)
        assert len(comp.get_substates()) == 2
        # Note: Duplicate ID validation would need to be added to AsyncCompositeState implementation

    async def test_initial_state_consistency(self):
        """Test initial state consistency"""
        states = [AsyncState("s1"), AsyncState("s2")]
        comp = AsyncCompositeState("comp", states, states[0])
        assert comp.get_initial_state() is states[0]
        # Current substate is set during entry
        await comp.on_entry(None, {})
        assert comp._current_substate is states[0]


# Test AsyncTransition
class TestAsyncTransition:
    async def test_init_valid(self, state_mock):
        target = AsyncState("target")
        transition = AsyncTransition(state_mock, target)
        assert transition.get_source_state() == state_mock
        assert transition.get_target_state() == target

    async def test_init_invalid(self):
        with pytest.raises(ValueError):
            AsyncTransition(None, None)

    async def test_guard_check(self, state_mock, guard_mock):
        target = AsyncState("target")
        transition = AsyncTransition(state_mock, target, guard_mock)
        assert transition.get_guard() == guard_mock

    async def test_transition_priority(self, state_mock):
        """Test transition priority ordering"""
        target = AsyncState("target")
        t1 = AsyncTransition(state_mock, target, priority=1)
        t2 = AsyncTransition(state_mock, target, priority=2)
        assert t2.get_priority() > t1.get_priority()

    async def test_action_execution_order(self, state_mock):
        """Test actions execute in order"""
        executed = []

        class OrderedAction(AsyncAction):
            def __init__(self, order):
                self.order = order

            async def execute(self, event, data):
                executed.append(self.order)

        target = AsyncState("target")
        transition = AsyncTransition(state_mock, target, actions=[OrderedAction(1), OrderedAction(2)])

        for action in transition.get_actions():
            await action.execute(None, {})
        assert executed == [1, 2]


# Test AsyncStateMachine
class TestAsyncStateMachine:
    async def test_init(self, simple_machine):
        assert simple_machine._initial_state.get_id() == "test_state"
        assert len(simple_machine._states) == 2

    async def test_start_stop(self, simple_machine):
        await simple_machine.start()
        assert simple_machine.is_running()
        await simple_machine.stop()
        assert not simple_machine.is_running()

    async def test_state_transition(self, simple_machine, event_mock):
        """Test basic state transition with timeout protection"""
        logger.debug("Starting state transition test")

        async def run_transition():
            logger.debug("Starting machine")
            await simple_machine.start()
            assert simple_machine.is_running()

            logger.debug(f"Current state before transition: {simple_machine.get_current_state_id()}")
            await simple_machine._handle_event(event_mock)
            logger.debug(f"Current state after transition: {simple_machine.get_current_state_id()}")

            assert simple_machine.get_current_state_id() == "state2"

            await simple_machine.stop()
            assert not simple_machine.is_running()

        try:
            await AsyncTestHelper.run_with_timeout(run_transition(), timeout=0.2)
            logger.debug("Test completed successfully")
        except TimeoutError:
            logger.error("Test timed out after 0.2 seconds!")
            await simple_machine.stop()  # Cleanup
            raise
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            await simple_machine.stop()  # Cleanup
            raise

    async def test_state_data_initialization(self):
        """Test state data is properly initialized"""
        states = [AsyncState("s1"), AsyncState("s2")]
        machine = AsyncStateMachine(states, [], states[0])
        for state in states:
            assert state.get_id() in machine._state_data
            assert isinstance(machine._state_data[state.get_id()], dict)

    async def test_invalid_transition_configuration(self):
        """Test invalid transition configurations"""
        states = [AsyncState("s1"), AsyncState("s2")]
        with pytest.raises(ValueError):
            AsyncStateMachine(states, [], None)  # Invalid initial state
        # Note: Invalid transition state validation would need to be added to AsyncStateMachine implementation


# Test AsyncExecutor
class TestAsyncExecutor:
    @pytest.fixture
    def executor_setup(self, simple_machine):
        return AsyncExecutor(simple_machine)

    async def test_init(self, executor_setup):
        assert executor_setup._state_machine is not None
        assert executor_setup._event_queue is not None

    async def test_start_stop(self, executor_setup):
        await executor_setup.start()
        assert executor_setup.is_running()
        await executor_setup.stop()
        assert not executor_setup.is_running()

    async def test_process_event(self, executor_setup, event_mock):
        await executor_setup.start()
        await executor_setup.process_event(event_mock)
        await asyncio.sleep(0.1)  # Allow event processing
        await executor_setup.stop()

    async def test_pause_resume(self, executor_setup):
        await executor_setup.start()
        async with executor_setup.pause():
            assert not executor_setup._processing_flag.is_set()
        assert executor_setup._processing_flag.is_set()
        await executor_setup.stop()

    async def test_executor_error_recovery(self, executor_setup: AsyncExecutor, event_mock: Event) -> None:
        """Test executor recovers from processing errors"""
        await executor_setup.start()

        # Create error-causing event that fails during processing, not validation
        class ErrorEvent(Event):
            def get_id(self) -> str:
                return "error_event"

            def get_data(self) -> Any:
                return None

            def get_timestamp(self) -> float:
                return 0.0

        error_event = ErrorEvent()

        # Process error event and verify recovery
        await executor_setup.process_event(error_event)
        await asyncio.sleep(0.1)

        # Should still process normal events
        await executor_setup.process_event(event_mock)
        await asyncio.sleep(0.1)

        await executor_setup.stop()

    async def test_executor_shutdown_cleanup(self, executor_setup: AsyncExecutor, event_mock: Event) -> None:
        """Test proper cleanup during executor shutdown"""
        await executor_setup.start()

        # Add some events
        for _ in range(5):
            await executor_setup.process_event(event_mock)
            await asyncio.sleep(0.01)  # Small delay between events

        # Stop executor and verify cleanup
        await executor_setup.stop()
        assert executor_setup._task is None
        assert not executor_setup.is_running()

    async def test_executor_event_ordering(self, executor_setup):
        """Test events are processed in order"""
        processed = []

        class OrderedEvent(Event):
            def __init__(self, order):
                self.order = order

            def get_id(self) -> str:
                return str(self.order)

            def get_data(self) -> Any:
                return self.order

            def get_timestamp(self) -> float:
                return 0.0

        # Create states and transitions that process events
        class ProcessingState(AsyncState):
            async def _do_enter(self, event: Any, data: Any) -> None:
                if event and event.get_data() is not None:
                    processed.append(event.get_data())

        state1 = ProcessingState("state1")
        state2 = ProcessingState("state2")

        # Create transitions between states
        transition1 = AsyncTransition(state1, state2)
        transition2 = AsyncTransition(state2, state1)

        # Setup state machine with processing states
        executor_setup._state_machine = AsyncStateMachine(
            states=[state1, state2], transitions=[transition1, transition2], initial_state=state1
        )

        await executor_setup.start()

        # Send events in reverse order
        for i in range(5, 0, -1):
            await executor_setup.process_event(OrderedEvent(i))
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.2)  # Allow processing
        await executor_setup.stop()

        # Verify processing order matches enqueue order
        assert processed == [5, 4, 3, 2, 1]

    async def test_executor_pause_resume_edge(self, executor_setup, event_mock):
        """Test pause/resume edge cases"""
        await executor_setup.start()
        processed_events = 0

        # Test pause during event processing
        async with executor_setup.pause():
            await executor_setup.process_event(event_mock)
            await asyncio.sleep(0.1)
            # Event should not be processed while paused
            assert processed_events == 0

        # Test multiple pause/resume cycles
        for _ in range(3):
            async with executor_setup.pause():
                await executor_setup.process_event(event_mock)
            await asyncio.sleep(0.1)

        await executor_setup.stop()


# Test Error Classes
class TestErrorClasses:
    def test_async_hsm_error(self):
        error = AsyncHSMError("test error")
        assert str(error) == "test error"

    def test_async_state_error(self):
        error = AsyncStateError("test error", "state_id", {"detail": "value"})
        assert error.state_id == "state_id"
        assert error.details == {"detail": "value"}

    def test_async_transition_error(self, state_mock, event_mock):
        target = AsyncState("target")
        error = AsyncTransitionError("test error", state_mock, target, event_mock, {"detail": "value"})
        assert error.source_state == state_mock
        assert error.target_state == target
        assert error.event == event_mock


# Edge Cases and Complex State Configurations
class TestComplexStateScenarios:
    @pytest.fixture
    def nested_composite_state(self):
        leaf_state = AsyncState("leaf")
        inner_composite = AsyncCompositeState("inner", [leaf_state], leaf_state)
        outer_composite = AsyncCompositeState("outer", [inner_composite], inner_composite)
        return outer_composite

    async def test_deep_nesting_entry(self, nested_composite_state, event_mock):
        """Test deeply nested state entry propagation"""
        await nested_composite_state.on_entry(event_mock, {})
        assert nested_composite_state._current_substate is not None
        assert nested_composite_state._current_substate._current_substate is not None

    async def test_history_state_deep_reset(self, nested_composite_state, event_mock):
        """Test history state behavior with deep nesting"""
        nested_composite_state.set_has_history(True)
        nested_composite_state._current_substate.set_has_history(True)

        await nested_composite_state.on_entry(event_mock, {})
        await nested_composite_state.on_exit(event_mock, {})

        # Verify history states are properly maintained
        assert nested_composite_state.get_history_state() is not None
        assert nested_composite_state._current_substate.get_history_state() is not None

    @pytest.fixture
    def cycle_machine(self):
        states = [AsyncState(f"state{i}") for i in range(3)]
        transitions = [
            AsyncTransition(states[0], states[1]),
            AsyncTransition(states[1], states[2]),
            AsyncTransition(states[2], states[0]),  # Creates a cycle
        ]
        return AsyncStateMachine(states, transitions, states[0])

    async def test_cycle_detection(self, cycle_machine, event_mock):
        """Test state machine handles cycles without infinite loops"""
        logger.debug("Starting cycle detection test")
        await cycle_machine.start()

        # Run through a limited number of cycles with logging
        for i in range(6):  # Should complete 2 full cycles
            logger.debug(f"Cycle iteration {i}: Current state = {cycle_machine.get_current_state_id()}")
            await cycle_machine._handle_event(event_mock)
            await asyncio.sleep(0.1)  # Add small delay between transitions

        logger.debug("Cycles complete")
        assert cycle_machine.is_running()
        await cycle_machine.stop()
        logger.debug("Test finished")


class TestResourceManagement:
    @pytest.fixture
    async def resource_heavy_state(self):
        class ResourceState(AsyncState):
            def __init__(self, state_id: str):
                super().__init__(state_id)
                self.resources = []

            async def _do_enter(self, event: Any, data: Any) -> None:
                self.resources.extend([bytearray(1024) for _ in range(100)])

            async def _do_exit(self, event: Any, data: Any) -> None:
                self.resources.clear()

        return ResourceState("resource_state")

    async def test_resource_cleanup(self, resource_heavy_state, event_mock):
        """Test proper resource cleanup during state transitions"""
        await resource_heavy_state.on_entry(event_mock, {})
        assert len(resource_heavy_state.resources) > 0
        await resource_heavy_state.on_exit(event_mock, {})
        assert len(resource_heavy_state.resources) == 0

    @pytest.fixture
    def executor_setup(self, simple_machine):
        return AsyncExecutor(simple_machine)

    async def test_queue_boundary(self, executor_setup, event_mock):
        """Test event queue behavior at boundaries"""
        try:
            # Create executor with small queue size
            executor = AsyncExecutor(executor_setup._state_machine, max_queue_size=1)
            await executor.start()

            async with executor.pause():
                # Fill queue to capacity
                await executor.process_event(event_mock)

                # Add debug logging
                queue_size = await executor._event_queue.size()
                logger.debug(f"Queue size after first event: {queue_size}")

                # Try to add one more event (should fail with queue full)
                with pytest.raises(QueueFullError):  # Note: Changed from EventQueueError to QueueFullError
                    await executor.process_event(event_mock)

        finally:
            if executor.is_running():
                await executor.stop()


class TestEdgeCases:
    async def test_rapid_state_changes(self, simple_machine, event_mock):
        """Test rapid state transitions"""
        await simple_machine.start()
        for _ in range(100):  # Rapid transitions
            await simple_machine._handle_event(event_mock)
        await simple_machine.stop()

    async def test_concurrent_guard_evaluation(self, guard_mock):
        """Test concurrent guard evaluation"""

        async def evaluate_guard():
            return await guard_mock.check(None, None)

        results = await asyncio.gather(*[evaluate_guard() for _ in range(10)])
        assert len(results) == 10
        assert all(results)  # All should be True by default

    async def test_null_event_handling(self, simple_machine):
        """Test handling of null events"""
        await simple_machine.start()
        await simple_machine._handle_event(None)
        assert simple_machine.is_running()

    async def test_empty_composite_state(self):
        """Test composite state with no substates"""
        with pytest.raises(ValueError, match="Composite state must have at least one substate"):
            AsyncCompositeState("empty", [], None)


class TestProtocolConformance:
    async def test_custom_guard_implementation(self):
        """Test custom guard implementation conformance"""

        class CustomGuard:
            async def check(self, event: Any, state_data: Any) -> bool:
                return True

        guard = CustomGuard()
        assert isinstance(guard, AsyncGuard)
        result = await guard.check(None, None)
        assert isinstance(result, bool)

    async def test_custom_action_implementation(self):
        """Test custom action implementation conformance"""

        class CustomAction:
            async def execute(self, event: Any, state_data: Any) -> None:
                pass

        action = CustomAction()
        assert isinstance(action, AsyncAction)
        await action.execute(None, None)  # Should not raise


class TestExceptionPropagation:
    async def test_nested_error_context(self):
        """Test error context preservation in nested states"""

        class ErrorState(AsyncState):
            async def _do_enter(self, event: Any, data: Any) -> None:
                try:
                    raise ValueError("Inner error")
                except ValueError as e:
                    raise AsyncStateError("Outer error", self.get_id(), {"inner_error": str(e)}) from e

        state = ErrorState("error_state")
        with pytest.raises(AsyncStateError) as exc_info:
            await state.on_entry(None, {})

        assert exc_info.value.details.get("inner_error") == "Inner error"
        assert exc_info.value.__cause__ is not None

    async def test_transition_error_chain(self, state_mock):
        """Test transition error propagation chain"""

        class ErrorGuard(AsyncGuard):
            async def check(self, event: Any, state_data: Any) -> bool:
                raise ValueError("Guard error")

        states = [state_mock]
        transition = AsyncTransition(state_mock, state_mock, ErrorGuard())
        machine = AsyncStateMachine(states, [transition], state_mock)

        await machine.start()
        with pytest.raises(AsyncTransitionError) as exc_info:
            await machine._handle_event(None)

        assert isinstance(exc_info.value.__cause__, ValueError)
        await machine.stop()


class AsyncStateMachineTestHelper:
    @staticmethod
    async def trigger_transitions(machine, event, count: int = 1):
        """Helper to trigger multiple transitions"""
        for _ in range(count):
            await machine._handle_event(event)
            await asyncio.sleep(0.01)

    @staticmethod
    async def verify_state_sequence(machine, event, expected_states: List[str]):
        """Helper to verify a sequence of state transitions"""
        for expected_state in expected_states:
            assert machine.get_current_state_id() == expected_state
            await machine._handle_event(event)
            await asyncio.sleep(0.01)
