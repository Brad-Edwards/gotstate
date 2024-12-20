# tests/unit/test_executor.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


def test_executor_init(mock_machine, mock_event_queue):
    from hsm.runtime.executor import Executor

    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)
    assert ex.machine is mock_machine
    assert ex.event_queue is mock_event_queue


def test_executor_run_stop(mock_machine, mock_event_queue):
    import threading
    import time

    from hsm.runtime.executor import Executor

    # Simulate event queue returning None (no events)
    mock_event_queue.dequeue.return_value = None
    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)

    # Run executor in separate thread
    thread = threading.Thread(target=ex.run)
    thread.start()

    # Give it a moment to start
    time.sleep(0.1)

    # Stop the executor
    ex.stop()
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_executor_process_events(mock_machine, mock_event_queue, mock_event):
    import threading
    import time

    from hsm.runtime.executor import Executor

    # Simulate one event then None
    mock_event_queue.dequeue.side_effect = [mock_event, None]

    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)

    # Run executor in separate thread
    thread = threading.Thread(target=ex.run)
    thread.start()

    # Give it a moment to process events
    time.sleep(0.1)

    # Stop the executor
    ex.stop()
    thread.join(timeout=1)

    # Verify the event was processed
    mock_machine.process_event.assert_called_once_with(mock_event)
