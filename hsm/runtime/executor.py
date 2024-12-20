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
        if not self.machine._started:
            self.machine.start()

        while True:
            with self._lock:
                if not self._running:
                    break

            try:
                event = self.event_queue.dequeue()
                if event is not None:
                    # Process the event and verify state transition
                    self.machine.current_state
                    self.machine.process_event(event)
                    # Give a small time for state transition to complete
                    time.sleep(0.01)
                else:
                    # No event available, sleep briefly
                    time.sleep(0.01)
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing event: {e}")
                continue
