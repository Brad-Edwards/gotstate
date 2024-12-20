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

    def run(self) -> None:
        """
        Start the blocking loop that continuously processes events until stopped.
        This method blocks until `stop()` is called.
        """
        with self._lock:
            self._running = True

        # Ensure machine is started if not already
        if not self.machine.current_state:
            self.machine.start()

        while True:
            with self._lock:
                if not self._running:
                    break

            try:
                event = self.event_queue.dequeue()
                if event is not None:
                    # Process the event
                    self.machine.process_event(event)
                else:
                    # No event available, no immediate stop requested
                    # Sleep briefly to avoid tight loop spinning.
                    time.sleep(0.01)
            except StopIteration:
                # Handle case where mock queue runs out of events in tests
                break

    def stop(self) -> None:
        """
        Signal the event loop to stop after finishing current tasks.
        """
        with self._lock:
            self._running = False
