# test_Executor.py

import dataclasses
import threading
import time
import unittest
from itertools import cycle
from typing import Any, Optional, Type
from unittest.mock import Mock, call, patch

from hsm.core.errors import ExecutorError
from hsm.core.events import Event
from hsm.core.state_machine import StateMachine
from hsm.runtime.event_queue import EventQueue
from hsm.runtime.executor import Executor, ExecutorContext, ExecutorState, ExecutorStats
from hsm.runtime.timers import Timer

# Add missing exceptions to EventQueue for tests:
setattr(EventQueue, "QueueError", type("QueueError", (Exception,), {}))
setattr(EventQueue, "QueueFullError", type("QueueFullError", (EventQueue.QueueError,), {}))


# We'll mock events as simple Mocks with needed specs
def make_mock_event(event_id: str = "test_event", priority: int = 0):
    e = Mock(spec=Event)
    e.get_id.return_value = event_id
    e.get_priority.return_value = priority
    return e


# Add at top of file with other imports
try:
    from threading import TimeoutError as ThreadTimeoutError
except ImportError:
    ThreadTimeoutError = TimeoutError  # Use built-in TimeoutError as fallback


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.state_machine = Mock(spec=StateMachine)
        self.executor = Executor(self.state_machine)

    def tearDown(self):
        if self.executor._context.state != ExecutorState.IDLE:
            self.executor.stop(force=True)

    # ------------------ Helper Methods ------------------
    def _start_executor_and_assert_running(self):
        """Helper to start executor and assert that it's running."""
        self.executor.start()
        self.assertTrue(self.executor.is_running())
        self.assertEqual(self.executor._context.state, ExecutorState.RUNNING)

    def _stop_executor_and_assert_stopped(self, force=False):
        """Helper to stop executor and assert that it's stopped."""
        self.executor.stop(force=force)
        self.assertFalse(self.executor.is_running())
        self.assertEqual(self.executor._context.state, ExecutorState.STOPPED)

    def _pause_executor_and_assert_paused(self):
        """Helper to pause executor and assert that it's paused."""
        with self.executor.pause():
            self.assertEqual(self.executor._context.state, ExecutorState.PAUSED)

    def _register_error_handler(self, error_type: Type[Exception], handler: Mock):
        """Helper to register an error handler."""
        self.executor.register_error_handler(error_type, handler)

    # ------------------ Example Tests ------------------
    def test_executor_lifecycle(self):
        """Test basic lifecycle: start -> running -> stop"""
        self._start_executor_and_assert_running()
        self._stop_executor_and_assert_stopped()

    def test_executor_pause_resume(self):
        """Test pause/resume functionality"""
        self._start_executor_and_assert_running()
        self._pause_executor_and_assert_paused()
        # After exiting the with-block, it should be RUNNING again
        self.assertEqual(self.executor._context.state, ExecutorState.RUNNING)

    def test_executor_error_handling_chain(self):
        """Test error handler chain and priorities"""
        self._start_executor_and_assert_running()
        base_handler = Mock()
        specific_handler = Mock()
        self._register_error_handler(Exception, base_handler)
        self._register_error_handler(ValueError, specific_handler)

        # Trigger error
        self.state_machine.process_event.side_effect = ValueError("Test error")
        self.executor.process_event(make_mock_event())

        # Wait a bit for error handling
        time.sleep(0.1)

        # Check calls
        specific_handler.assert_called_once()
        base_handler.assert_not_called()

    def test_executor_force_stop(self):
        """Test force stop behavior"""
        self._start_executor_and_assert_running()
        self._stop_executor_and_assert_stopped(force=True)

    def test_executor_invalid_start_state(self):
        """Test starting executor from invalid state"""
        self._start_executor_and_assert_running()
        with self.assertRaises(ExecutorError):
            self.executor.start()

    def test_executor_invalid_stop_state(self):
        """Test stopping executor from invalid state"""
        with self.assertRaises(ExecutorError):
            self.executor.stop()

    def test_executor_stats_reset(self):
        """Test stats are reset properly"""
        stats = self.executor.get_stats()
        self.assertEqual(stats.events_processed, 0)
        self.assertEqual(stats.transitions_executed, 0)
        self.assertEqual(stats.errors_encountered, 0)

    def test_executor_state_validation(self):
        """Test state transitions validation"""
        self._start_executor_and_assert_running()
        with self.assertRaises(ExecutorError):
            self.executor._context.state = ExecutorState.IDLE  # Can't go back to IDLE from RUNNING

    def test_executor_init_validation(self):
        """Test constructor validation"""
        with self.assertRaises(ValueError):
            Executor(None)  # state_machine cannot be None
        with self.assertRaises(ValueError):
            Executor(self.state_machine, thread_join_timeout=0)  # timeout must be positive
        with self.assertRaises(ValueError):
            Executor(self.state_machine, max_queue_size=0)  # queue size must be positive
        with self.assertRaises(TypeError):
            Executor(self.state_machine, thread_join_timeout="1.0")  # timeout must be numeric
        with self.assertRaises(TypeError):
            Executor(self.state_machine, max_queue_size="100")  # queue size must be int

    def test_executor_get_current_state(self):
        """Test current state retrieval"""
        mock_state = Mock()
        self.state_machine.get_state.return_value = mock_state
        self._start_executor_and_assert_running()
        self.assertEqual(self.executor.get_current_state(), mock_state)

    def test_executor_pause_validation(self):
        """Test pause validation"""
        # Can't pause when not running
        with self.assertRaises(ExecutorError):
            with self.executor.pause():
                pass

    def test_executor_error_handler_registration(self):
        """Test error handler registration"""
        handler = Mock()
        self.executor.register_error_handler(ValueError, handler)
        self.assertIn(ValueError, self.executor._context._error_handlers)
        self.assertEqual(self.executor._context._error_handlers[ValueError], handler)

    def test_executor_error_handler_override(self):
        """Test error handler override behavior"""
        handler1 = Mock()
        handler2 = Mock()
        self.executor.register_error_handler(ValueError, handler1)
        self.executor.register_error_handler(ValueError, handler2)  # Should override
        self.assertEqual(self.executor._context._error_handlers[ValueError], handler2)

    def test_executor_state_transitions(self):
        """Test all valid state transitions"""
        # IDLE -> RUNNING
        self.assertEqual(self.executor._context.state, ExecutorState.IDLE)
        self._start_executor_and_assert_running()

        # RUNNING -> PAUSED
        with self.executor.pause():
            self.assertEqual(self.executor._context.state, ExecutorState.PAUSED)

        # RUNNING -> STOPPING -> STOPPED
        self._stop_executor_and_assert_stopped()

    def test_executor_stats_immutability(self):
        """Test that stats objects are immutable"""
        stats = self.executor.get_stats()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            stats.events_processed = 100

    def test_executor_process_event_validation(self):
        """Test event validation in process_event"""
        self._start_executor_and_assert_running()
        with self.assertRaises(TypeError):
            self.executor.process_event(None)
        with self.assertRaises(TypeError):
            self.executor.process_event("not an event")
        with self.assertRaises(TypeError):
            self.executor.process_event(123)

    def test_executor_context_error_handler_isolation(self):
        """Test that error handlers are isolated per context"""
        context1 = ExecutorContext()
        context2 = ExecutorContext()
        handler1 = Mock()
        handler2 = Mock()

        context1.register_error_handler(ValueError, handler1)
        context2.register_error_handler(ValueError, handler2)

        error = ValueError("test")
        context1.handle_error(error)

        handler1.assert_called_once_with(error)
        handler2.assert_not_called()

    def test_executor_queue_full_handling(self):
        """Test behavior when event queue is full"""
        # Mock the event queue first
        self.executor._event_queue = Mock()
        self.executor._event_queue.enqueue.side_effect = EventQueue.QueueFullError()

        self._start_executor_and_assert_running()
        with self.assertRaises(ExecutorError):
            self.executor.process_event(make_mock_event())

    def test_executor_state_machine_stop_error(self):
        """Test handling of state machine stop errors"""
        self._start_executor_and_assert_running()
        self.state_machine.stop.side_effect = RuntimeError("Stop error")
        self._stop_executor_and_assert_stopped(force=True)  # Should not raise, force=True should handle errors

    def test_executor_timer_shutdown(self):
        """Test timer shutdown during stop"""

        # Create a Timer class with the required method
        class TimerSpec:
            def shutdown(self):
                pass

        mock_timer = Mock(spec=TimerSpec)
        self.executor._timer = mock_timer
        self._start_executor_and_assert_running()
        self._stop_executor_and_assert_stopped()
        mock_timer.shutdown.assert_called_once()

    def test_executor_handle_error_unhandled(self):
        """Test handling of unhandled errors (no registered handler)."""
        self._start_executor_and_assert_running()

        # Create an error with no registered handler
        custom_error = Exception("Custom error")

        with patch("hsm.runtime.executor.logger") as mock_logger:
            self.executor._context.handle_error(custom_error)

            # Verify logger was called with unhandled error message
            mock_logger.error.assert_called_with("Unhandled error in executor: %s", str(custom_error))

            # Verify state changed to ERROR
            self.assertEqual(self.executor._context.state, ExecutorState.ERROR)

            # Verify error count increased
            stats = self.executor.get_stats()
            self.assertEqual(stats.errors_encountered, 1)

    def test_executor_handle_error_handler_fails(self):
        """Test handling of errors when the error handler itself fails."""
        self._start_executor_and_assert_running()

        # Register a handler that raises an exception
        def failing_handler(e):
            raise RuntimeError("Handler failed")

        self.executor.register_error_handler(ValueError, failing_handler)

        with patch("hsm.runtime.executor.logger") as mock_logger:
            self.executor._context.handle_error(ValueError("Test error"))

            # Verify logger was called for handler failure
            mock_logger.error.assert_called_with("Error handler failed: %s", "Handler failed")

            # Verify state changed to ERROR
            self.assertEqual(self.executor._context.state, ExecutorState.ERROR)

    def test_executor_timer_callback_error(self):
        """Test error handling in timer callback."""
        self._start_executor_and_assert_running()

        # Create a mock event that will cause an error
        mock_event = make_mock_event()

        # Create a mock timer with the callback we want to test
        def timer_callback(timer_id: str, event: Any) -> None:
            if self.executor._context.state == ExecutorState.RUNNING:
                self.executor.process_event(event)

        self.executor._timer = Mock()
        self.executor._timer._callback = timer_callback

        # Set up the state machine to raise an error
        self.state_machine.process_event.side_effect = RuntimeError("Timer callback error")

        with patch("hsm.runtime.executor.logger") as mock_logger:
            # Call timer callback directly
            self.executor._timer._callback("timer1", mock_event)

            # Wait for error handling
            time.sleep(0.2)

            # Verify error was logged with the correct message
            mock_logger.error.assert_called_with("Unhandled error in executor: %s", "Timer callback error")

            # Verify error was handled
            stats = self.executor.get_stats()
            self.assertEqual(stats.errors_encountered, 1)

    def test_executor_stop_timeout(self):
        """Test handling of timeout during executor stop."""
        self._start_executor_and_assert_running()

        # Mock thread.join to simulate timeout
        with patch.object(threading.Thread, "join", side_effect=ThreadTimeoutError()):
            with self.assertRaises(ExecutorError):
                self.executor.stop()  # Should raise ExecutorError due to timeout

            # Try again with force=True
            self.executor.stop(force=True)  # Should succeed
            self.assertEqual(self.executor._context.state, ExecutorState.STOPPED)

    def test_executor_process_events_paused(self):
        """Test event processing behavior when executor is paused."""
        self._start_executor_and_assert_running()

        # Mock the event queue to track if events are processed
        mock_queue = Mock(spec=EventQueue)
        mock_queue.is_empty.return_value = False
        mock_queue.try_dequeue.return_value = None  # Return None to simulate no events while paused
        self.executor._event_queue = mock_queue

        # Pause the executor
        with self.executor.pause():
            # Wait a bit to ensure worker thread sees pause
            time.sleep(0.2)

            # Verify try_dequeue was not called (events not processed while paused)
            mock_queue.try_dequeue.assert_not_called()

        # After pause, verify try_dequeue is called again
        time.sleep(0.2)
        mock_queue.try_dequeue.assert_called()

    def test_executor_process_events_error_handling(self):
        """Test error handling during event processing."""
        self._start_executor_and_assert_running()

        # Create an event that will cause an error
        event = make_mock_event()

        # Mock the event queue to ensure our event is processed
        mock_queue = Mock(spec=EventQueue)
        mock_queue.is_empty.return_value = False
        mock_queue.try_dequeue.return_value = event
        self.executor._event_queue = mock_queue

        # Set up the state machine to raise an error
        error = RuntimeError("Processing error")
        self.state_machine.process_event.side_effect = error

        with patch("hsm.runtime.executor.logger") as mock_logger:
            # Let the executor process the event
            time.sleep(0.2)

            # Verify error was logged with the correct message
            # The error is handled by handle_error which logs it as unhandled
            mock_logger.error.assert_called_with("Unhandled error in executor: %s", str(error))

            # Verify error was handled
            stats = self.executor.get_stats()
            self.assertEqual(stats.errors_encountered, 1)

            # Verify executor state changed to ERROR
            self.assertEqual(self.executor._context.state, ExecutorState.ERROR)


if __name__ == "__main__":
    unittest.main()
