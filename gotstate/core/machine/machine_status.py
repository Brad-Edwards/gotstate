from enum import Enum, auto


class MachineStatus(Enum):
    """Defines the possible states of a state machine.

    Used to track machine lifecycle and coordinate operations.
    """

    UNINITIALIZED = auto()  # Machine not yet configured
    INITIALIZING = auto()  # Machine being configured
    ACTIVE = auto()  # Machine running normally
    MODIFYING = auto()  # Machine being modified
    TERMINATING = auto()  # Machine shutting down
    TERMINATED = auto()  # Machine fully stopped
