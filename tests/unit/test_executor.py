# tests/unit/test_executor.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


def test_executor_init(mock_machine, mock_event_queue):
    from hsm.runtime.executor import Executor

    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)
    assert ex.machine is mock_machine
    assert ex.event_queue is mock_event_queue


def test_executor_run_stop(mock_machine, mock_event_queue):
    from hsm.runtime.executor import Executor

    # Simulate event queue returning None (no events) to end the loop.
    mock_event_queue.dequeue.side_effect = [None]  # Immediately no events
    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)

    # run should exit gracefully if no events are available
    ex.run()
    # No exceptions = pass

    # Test stop method: should just set an internal flag
    ex.stop()
    # Cannot assert much without knowing internal state, but no error is good.


def test_executor_process_events(mock_machine, mock_event_queue, mock_event):
    from hsm.runtime.executor import Executor

    # Simulate one event then None
    mock_event_queue.dequeue.side_effect = [mock_event, None]

    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)
    ex.run()
    mock_machine.process_event.assert_called_once_with(mock_event)
