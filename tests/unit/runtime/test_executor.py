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


def test_executor_double_run(mock_machine, mock_event_queue):
    import threading
    import time
    from hsm.runtime.executor import Executor

    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)
    mock_event_queue.dequeue.return_value = None

    # First run
    thread1 = threading.Thread(target=ex.run)
    thread1.start()
    time.sleep(0.1)

    # Second run should return immediately
    ex.run()  # Should return immediately due to running check

    ex.stop()
    thread1.join(timeout=1)
    assert not thread1.is_alive()


def test_executor_machine_start(mock_machine, mock_event_queue):
    from hsm.runtime.executor import Executor
    import threading
    import time

    mock_machine._started = False
    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)
    mock_event_queue.dequeue.return_value = None

    thread = threading.Thread(target=ex.run)
    thread.start()
    time.sleep(0.1)

    # Verify machine was started
    mock_machine.start.assert_called_once()

    ex.stop()
    thread.join(timeout=1)


def test_executor_error_handling(mock_machine, mock_event_queue, mock_event):
    from hsm.runtime.executor import Executor
    import threading
    import time

    # Make process_event raise an exception
    mock_machine.process_event.side_effect = Exception("Test error")
    mock_event_queue.dequeue.side_effect = [mock_event, None]

    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)

    thread = threading.Thread(target=ex.run)
    thread.start()
    time.sleep(0.1)

    ex.stop()
    thread.join(timeout=1)
    
    # Verify the executor continued running despite the error
    mock_machine.process_event.assert_called_once_with(mock_event)


def test_executor_stop_before_run(mock_machine, mock_event_queue):
    from hsm.runtime.executor import Executor

    ex = Executor(machine=mock_machine, event_queue=mock_event_queue)
    ex.stop()  # Should not raise any errors
