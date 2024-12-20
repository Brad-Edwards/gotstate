class Executor:
    """
    Runs the event processing loop for synchronous state machines, fetching events
    from a queue and passing them to the machine until stopped.
    """

    def __init__(self, machine: "StateMachine", event_queue: "EventQueue") -> None:
        """
        Initialize with a state machine and event queue.

        :param machine: StateMachine instance to run.
        :param event_queue: EventQueue providing events to process.
        """
        raise NotImplementedError()

    def run(self) -> None:
        """
        Start the blocking loop that continuously processes events until stopped.
        """
        raise NotImplementedError()

    def stop(self) -> None:
        """
        Signal the event loop to stop after finishing current tasks.
        """
        raise NotImplementedError()


class _EventProcessingLoop:
    """
    Internal loop class handling the actual iteration over events, invoking the
    state machine, and catching errors.
    """

    def __init__(self, machine: "StateMachine", event_queue: "EventQueue") -> None:
        """
        Store references and prepare for event iteration.
        """
        raise NotImplementedError()

    def start_loop(self) -> None:
        """
        Begin processing events.
        """
        raise NotImplementedError()

    def stop_loop(self) -> None:
        """
        Stop processing events.
        """
        raise NotImplementedError()


class _ErrorLogger:
    """
    Internal tool for logging or handling errors encountered during runtime event
    processing.
    """

    def __init__(self) -> None:
        """
        Prepare error logging mechanism.
        """

    def log(self, error: Exception) -> None:
        """
        Log or otherwise handle an encountered error.
        """
