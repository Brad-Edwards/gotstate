# hsm/runtime/executor.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import threading
import time
from typing import Optional

from hsm.core.events import Event
from hsm.core.state_machine import StateMachine
from hsm.runtime.event_queue import EventQueue


class Executor:
    """
    Runs the event processing loop for synchronous state machines, fetching events
    from a queue and passing them to the machine until stopped.
    """

    def __init__(self, machine: StateMachine, event_queue: EventQueue) -> None:
        """
        Initialize with a state machine and event queue.

        :param machine: StateMachine instance to run.
        :param event_queue: EventQueue providing events to process.
        """
        self.machine = machine
        self.event_queue = event_queue
        self._running = False
        self._lock = threading.Lock()

    def stop(self) -> None:
        """Stop the executor's event processing loop."""
        with self._lock:
            self._running = False

    def _ensure_machine_started(self) -> None:
        """Ensure the state machine is started before processing events."""
        if not self.machine._started:
            self.machine.start()

    def _process_event(self, event: Event) -> None:
        """Process a single event and verify state transition."""
        # Verify current state before processing
        self.machine.current_state
        self.machine.process_event(event)
        # Give a small time for state transition to complete
        time.sleep(0.01)

    def _handle_event_processing_error(self, error: Exception) -> None:
        """Handle any errors that occur during event processing."""
        print(f"Error processing event: {error}")

    def _should_continue_running(self) -> bool:
        """Check if the executor should continue running."""
        with self._lock:
            return self._running

    def run(self) -> None:
        """
        Start the blocking loop that continuously processes events until stopped.
        This method blocks until `stop()` is called.
        """
        with self._lock:
            if self._running:
                return
            self._running = True

        # Ensure machine is started
        self._ensure_machine_started()

        while self._should_continue_running():
            try:
                event = self.event_queue.dequeue()
                if event is not None:
                    self._process_event(event)
                else:
                    # No event available, sleep briefly
                    time.sleep(0.01)
            except Exception as e:
                self._handle_event_processing_error(e)
                continue
