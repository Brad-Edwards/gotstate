import threading
from typing import Any, Dict, List, Set, Type

from gotstate.core.machine.basic_state_machine import BasicStateMachine
from gotstate.core.machine.state_machine import StateMachine


class MachineBuilder:
    """Builds state machine configurations.

    MachineBuilder implements the Builder pattern to construct
    valid state machine configurations.

    Class Invariants:
    1. Must validate configuration
    2. Must enforce constraints
    3. Must maintain consistency
    4. Must track dependencies

    Design Patterns:
    - Builder: Constructs machines
    - Factory: Creates components
    - Validator: Checks configuration

    Threading/Concurrency Guarantees:
    1. Thread-safe construction
    2. Atomic validation
    3. Safe concurrent access

    Performance Characteristics:
    1. O(c) configuration where c is component count
    2. O(v) validation where v is rule count
    3. O(d) dependency resolution where d is dependency count
    """

    def __init__(self):
        """Initialize the machine builder."""
        self._machine_type = BasicStateMachine
        self._components: Dict[str, List[Any]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._component_lock = threading.Lock()
        self._dependency_lock = threading.Lock()

    @property
    def machine_type(self) -> type:
        """Get the machine type.

        Returns:
            The machine type class
        """
        return self._machine_type

    @property
    def components(self) -> Dict[str, List[Any]]:
        """Get the component collections.

        Returns:
            Dictionary mapping component types to lists of components
        """
        with self._component_lock:
            return {k: list(v) for k, v in self._components.items()}

    @property
    def dependencies(self) -> Dict[str, Set[str]]:
        """Get the component dependencies.

        Returns:
            Dictionary mapping component types to their dependencies
        """
        with self._dependency_lock:
            return {k: set(v) for k, v in self._dependencies.items()}

    def set_machine_type(self, machine_type: Type[StateMachine]) -> None:
        """Set the type of machine to build.

        Args:
            machine_type: The machine type class

        Raises:
            ValueError: If machine_type is not a StateMachine subclass
        """
        if not issubclass(machine_type, StateMachine):
            raise ValueError("Machine type must be a StateMachine subclass")
        self._machine_type = machine_type

    def add_component(self, component_type: str, component: Any) -> None:
        """Add a component to the configuration.

        Args:
            component_type: Type of component (e.g. "states", "regions")
            component: The component to add
        """
        with self._component_lock:
            if component_type not in self._components:
                self._components[component_type] = []
            self._components[component_type].append(component)

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Add a component dependency.

        Args:
            dependent: The dependent component type
            dependency: The required component type

        Raises:
            ValueError: If dependency would create a cycle
        """
        with self._dependency_lock:
            # Check for cyclic dependencies
            if self._would_create_cycle(dependent, dependency):
                raise ValueError(f"Cyclic dependency detected: {dependent} -> {dependency}")

            if dependent not in self._dependencies:
                self._dependencies[dependent] = set()
            self._dependencies[dependent].add(dependency)

    def _would_create_cycle(self, dependent: str, dependency: str) -> bool:
        """Check if adding a dependency would create a cycle.

        Args:
            dependent: The dependent component type
            dependency: The required component type

        Returns:
            True if adding the dependency would create a cycle
        """
        # If the dependency already depends on the dependent, it would create a cycle
        if dependency in self._dependencies and dependent in self._dependencies[dependency]:
            return True

        # Check if there's a path from dependency back to dependent
        visited = set()

        def has_path(start: str, target: str) -> bool:
            if start == target:
                return True
            if start in visited:
                return False
            visited.add(start)
            if start in self._dependencies:
                for next_dep in self._dependencies[start]:
                    if has_path(next_dep, target):
                        return True
            return False

        return has_path(dependency, dependent)

    def build(self) -> StateMachine:
        """Build and validate a machine instance.

        Returns:
            The constructed state machine

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate machine type
        if not self._machine_type:
            raise ValueError("Machine type not set")

        # Define validation order and required types
        validation_order = ["states", "regions", "transitions"]
        required_types = ["states", "regions"]  # Transitions are optional

        # Validate required components
        with self._component_lock:
            # Check if any components are added
            if not self._components:
                raise ValueError("No components added to machine")

            # Check if all required component types are present
            missing_types = [t for t in required_types if t not in self._components]
            if missing_types:
                raise ValueError(f"Missing required component types: {', '.join(missing_types)}")

        # Validate components in specific order
        with self._component_lock:
            for component_type in validation_order:
                if component_type not in self._components:
                    continue

                # Remove trailing 's' from component type for error message
                type_name = component_type[:-1] if component_type.endswith("s") else component_type

                # Validate all components of this type
                for component in self._components[component_type]:
                    if hasattr(component, "is_valid") and not component.is_valid():
                        # Clear components of this type to prevent re-validation
                        self._components[component_type] = []
                        raise ValueError(f"Invalid {type_name} configuration")

        # Validate dependencies
        with self._dependency_lock, self._component_lock:
            for dependent, dependencies in self._dependencies.items():
                for dependency in dependencies:
                    # Check if the dependency exists in any component type
                    dependency_found = False
                    for components in self._components.values():
                        for component in components:
                            if hasattr(component, "id") and component.id == dependency:
                                dependency_found = True
                                break
                        if dependency_found:
                            break
                    if not dependency_found:
                        raise ValueError(f"Unresolved dependency: {dependency} required by {dependent}")

        # Create machine instance
        machine = self._machine_type()

        # Add components in specific order
        with self._component_lock:
            for component_type in validation_order:
                if component_type not in self._components:
                    continue
                add_method = getattr(machine, f"add_{component_type[:-1]}", None)
                if add_method:
                    for component in self._components[component_type]:
                        add_method(component)

        return machine
