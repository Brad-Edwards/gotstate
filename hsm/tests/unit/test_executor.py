# test_Executor.py

import dataclasses
import threading
import time
import unittest
from itertools import cycle
from typing import Optional
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

    def test_executor_lifecycle(self):
        """Test basic lifecycle: start -> running -> stop"""
        # Start
        self.executor.start()
        self.assertTrue(self.executor.is_running())
        self.assertEqual(self.executor._context.state, ExecutorState.RUNNING)

        # Stop
        self.executor.stop()
        self.assertEqual(self.executor._context.state, ExecutorState.STOPPED)
        self.assertFalse(self.executor.is_running())

    def test_executor_pause_resume(self):
        """Test pause/resume functionality"""
        self.executor.start()

        with self.executor.pause():
            self.assertEqual(self.executor._context.state, ExecutorState.PAUSED)

        # Should be running after pause
        self.assertEqual(self.executor._context.state, ExecutorState.RUNNING)

    def test_executor_error_handling_chain(self):
        """Test error handler chain and priorities"""
        self.executor.start()

        # Register handlers
        base_handler = Mock()
        specific_handler = Mock()

        self.executor.register_error_handler(Exception, base_handler)
        self.executor.register_error_handler(ValueError, specific_handler)

        # Trigger error
        self.state_machine.process_event.side_effect = ValueError("Test error")
        self.executor.process_event(make_mock_event())

        # Wait a bit for error handling
        time.sleep(0.1)

        # Specific handler should be called, not base
        specific_handler.assert_called_once()
        base_handler.assert_not_called()

    def test_executor_force_stop(self):
        """Test force stop behavior"""
        self.executor.start()
        self.executor.stop(force=True)
        self.assertEqual(self.executor._context.state, ExecutorState.STOPPED)

    def test_executor_invalid_start_state(self):
        """Test starting executor from invalid state"""
        self.executor.start()
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
        self.executor.start()
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
        self.executor.start()
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
        self.executor.start()
        self.assertEqual(self.executor._context.state, ExecutorState.RUNNING)

        # RUNNING -> PAUSED
        with self.executor.pause():
            self.assertEqual(self.executor._context.state, ExecutorState.PAUSED)

        # RUNNING -> STOPPING -> STOPPED
        self.executor.stop()
        self.assertEqual(self.executor._context.state, ExecutorState.STOPPED)

    def test_executor_stats_immutability(self):
        """Test that stats objects are immutable"""
        stats = self.executor.get_stats()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            stats.events_processed = 100

    def test_executor_process_event_validation(self):
        """Test event validation in process_event"""
        self.executor.start()
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

        self.executor.start()
        with self.assertRaises(ExecutorError):
            self.executor.process_event(make_mock_event())

    def test_executor_state_machine_stop_error(self):
        """Test handling of state machine stop errors"""
        self.executor.start()
        self.state_machine.stop.side_effect = RuntimeError("Stop error")
        self.executor.stop(force=True)  # Should not raise, force=True should handle errors
        self.assertEqual(self.executor._context.state, ExecutorState.STOPPED)

    def test_executor_timer_shutdown(self):
        """Test timer shutdown during stop"""

        # Create a Timer class with the required method
        class TimerSpec:
            def shutdown(self):
                pass

        mock_timer = Mock(spec=TimerSpec)
        self.executor._timer = mock_timer
        self.executor.start()
        self.executor.stop()
        mock_timer.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
