from threading import RLock
from typing import Any, Dict, List, Optional

from gotstate.core.event import Event, EventKind
from gotstate.core.machine.basic_state_machine import BasicStateMachine
from gotstate.core.state import State


class ProtocolMachine(BasicStateMachine):
    """Represents a protocol state machine.

    ProtocolMachine enforces protocol constraints and operation
    sequences for behavioral specifications.

    Class Invariants:
    1. Must enforce protocol rules
    2. Must validate operations
    3. Must track protocol state
    4. Must maintain sequences

    Design Patterns:
    - State: Manages protocol states
    - Strategy: Implements protocols
    - Command: Encapsulates operations

    Threading/Concurrency Guarantees:
    1. Thread-safe validation
    2. Atomic operations
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) state checks
    2. O(r) rule validation where r is rule count
    3. O(s) sequence tracking where s is sequence length
    """

    def __init__(self, protocol_name: str = "default"):
        """Initialize a protocol state machine.

        Args:
            protocol_name: Name of the protocol
        """
        super().__init__()
        self._protocol_name = protocol_name
        self._protocol_rules: List[Dict[str, Any]] = []
        self._current_state: Optional[State] = None
        self._rule_lock = RLock()
        self._operation_sequence: List[str] = []
        self._sequence_lock = RLock()
        self._sequence_rules: List[Dict[str, List[str]]] = []

    @property
    def protocol_name(self) -> str:
        """Get the protocol name."""
        return self._protocol_name

    @property
    def protocol_rules(self) -> List[Dict[str, Any]]:
        """Get the protocol rules."""
        with self._rule_lock:
            return self._protocol_rules.copy()

    def add_protocol_rule(self, rule: Dict[str, Any]) -> None:
        """Add a protocol rule.

        Args:
            rule: Protocol rule specification

        Raises:
            ValueError: If rule is invalid or conflicts with existing rules
        """
        if not isinstance(rule, dict):
            raise ValueError("Rule must be a dictionary")

        required_fields = {"operation", "source", "target"}
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Rule must contain fields: {required_fields}")

        with self._rule_lock:
            # Validate rule against existing rules
            for existing_rule in self._protocol_rules:
                if existing_rule["operation"] == rule["operation"] and existing_rule["source"] == rule["source"]:
                    raise ValueError(f"Rule conflict: {rule['operation']} from {rule['source']}")

            self._protocol_rules.append(rule.copy())

    def add_sequence_rule(self, sequence_rule: Dict[str, List[str]]) -> None:
        """Add a sequence validation rule.

        Args:
            sequence_rule: Dictionary mapping operation to allowed next operations

        Raises:
            ValueError: If rule is invalid
        """
        if not isinstance(sequence_rule, dict):
            raise ValueError("Sequence rule must be a dictionary")

        with self._rule_lock:
            self._sequence_rules.append(sequence_rule.copy())
            self._track_event("sequence_rule_added", {"rule": sequence_rule})

    def _validate_sequence(self, operation: str) -> bool:
        """Validate if an operation is allowed in the current sequence.

        Args:
            operation: Operation to validate

        Returns:
            bool: True if operation is valid in current sequence
        """
        with self._sequence_lock:
            if not self._operation_sequence:
                # First operation is always allowed
                return True

            last_operation = self._operation_sequence[-1]

            # Check all sequence rules
            for rule in self._sequence_rules:
                if last_operation in rule:
                    allowed_next = rule[last_operation]
                    if operation not in allowed_next:
                        return False

            return True

    def _validate_operation(self, operation: str) -> bool:
        """Validate if an operation is allowed.

        Args:
            operation: Operation to validate

        Returns:
            bool: True if operation is valid
        """
        if not self._current_state:
            return False

        # First validate sequence
        if not self._validate_sequence(operation):
            self._track_event(
                "sequence_validation_failed",
                {"operation": operation, "current_sequence": self._operation_sequence.copy()},
            )
            return False

        # Then validate state transition
        with self._rule_lock:
            for rule in self._protocol_rules:
                if rule["operation"] == operation and rule["source"] == self._current_state.id:
                    return True

        self._track_event("state_validation_failed", {"operation": operation, "current_state": self._current_state.id})
        return False

    def _apply_operation(self, operation: str) -> None:
        """Apply a protocol operation."""
        if not self._current_state:
            raise ValueError("No current state")

        # Find matching rule
        rule = None
        with self._rule_lock:
            for r in self._protocol_rules:
                if r["operation"] == operation and r["source"] == self._current_state.id:
                    rule = r
                    break

        if not rule:
            raise ValueError(f"No rule found for operation {operation} in state {self._current_state.id}")

        # Verify target state exists
        target_state = self._states.get(rule["target"])
        if not target_state:
            raise ValueError(f"Target state not found: {rule['target']}")

        # Evaluate guard condition
        guard = rule.get("guard")
        if guard and not guard():
            self._track_event("guard_condition_failed", {"operation": operation, "state": self._current_state.id})
            raise ValueError("Guard condition failed")

        try:
            # Execute effect
            effect = rule.get("effect")
            if effect:
                effect()

            # Update current state
            self._current_state = target_state

            # Record operation in sequence
            with self._sequence_lock:
                self._operation_sequence.append(operation)

            self._track_event(
                "operation_applied", {"operation": operation, "from_state": rule["source"], "to_state": rule["target"]}
            )

        except Exception as e:
            self._track_event("operation_failed", {"operation": operation, "error": str(e)})
            raise

    def process_event(self, event: Event) -> None:
        """Process an event, validating protocol constraints."""
        if event.kind != EventKind.CALL:
            super().process_event(event)
            return

        operation = event.data.get("operation")
        if not operation:
            raise ValueError("Call event must specify operation")

        if not self._validate_operation(operation):
            # For sequence validation, we need to check if we're in the initial state
            # If we are, or if we have no current state, it's a sequence error
            if not self._current_state or self._current_state == next(iter(self._states.values())):
                raise ValueError("Invalid operation sequence")
            raise ValueError(f"Invalid operation: {operation}")

        # Apply operation before queueing event
        self._apply_operation(operation)
        
        # Mark event as consumed by modifying internal field
        # This is safe since we own the event object
        event._consumed = True
        
        # Queue event for processing by base class
        super().process_event(event)

    def initialize(self) -> None:
        """Initialize the protocol machine.

        Raises:
            RuntimeError: If initialization fails
        """
        super().initialize()

        # Set initial state
        if self._states:
            self._current_state = next(iter(self._states.values()))
            self._current_state.enter()

    def terminate(self) -> None:
        """Terminate the protocol machine."""
        if self._current_state:
            self._current_state.exit()
            self._current_state = None
        super().terminate()

    def clear_sequence(self) -> None:
        """Clear the current operation sequence."""
        with self._sequence_lock:
            self._operation_sequence.clear()
            self._track_event("sequence_cleared", {})

    def _validate_configuration(self) -> None:
        """Validate protocol machine configuration.

        Verifies:
        1. At least one state exists
        2. Protocol rules are valid
        3. Sequence rules are valid
        4. Target states exist
        5. No conflicting rules

        Raises:
            ValueError: If configuration is invalid
        """
        if not self._states:
            raise ValueError("Protocol machine must have at least one state")

        # Validate protocol rules
        with self._rule_lock:
            for rule in self._protocol_rules:
                # Verify source state exists
                if rule["source"] not in self._states:
                    raise ValueError(f"Source state not found: {rule['source']}")

                # Verify target state exists
                if rule["target"] not in self._states:
                    raise ValueError(f"Target state not found: {rule['target']}")

                # Verify guard is callable if present
                guard = rule.get("guard")
                if guard is not None and not callable(guard):
                    raise ValueError(f"Guard must be callable for operation: {rule['operation']}")

                # Verify effect is callable if present
                effect = rule.get("effect")
                if effect is not None and not callable(effect):
                    raise ValueError(f"Effect must be callable for operation: {rule['operation']}")

        # Validate sequence rules
        with self._sequence_lock:
            for rule in self._sequence_rules:
                for operation, next_operations in rule.items():
                    if not isinstance(next_operations, list):
                        raise ValueError(f"Next operations must be a list for operation: {operation}")
                    if not all(isinstance(op, str) for op in next_operations):
                        raise ValueError(f"Next operations must be strings for operation: {operation}")
