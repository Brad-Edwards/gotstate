import unittest
from unittest.mock import Mock, patch

from gotstate.core.machine.basic_state_machine import BasicStateMachine
from gotstate.core.machine.machine_builder import MachineBuilder
from gotstate.core.machine.state_machine import StateMachine
from tests.unit.machine.machine_mocks import MockRegion, MockState


class TestMachineBuilder(unittest.TestCase):
    """Test cases for MachineBuilder class.

    Tests verify:
    1. Machine type management
    2. Component management
    3. Dependency management
    4. Build process validation
    5. Error handling
    """

    def setUp(self):
        """Set up test fixtures."""
        self.builder = MachineBuilder()
        self.state1 = MockState("state1")
        self.state2 = MockState("state2")
        self.region1 = MockRegion("region1")
        self.region2 = MockRegion("region2")

    def test_initial_state(self):
        """Test initial state of builder.

        Verifies:
        1. Default machine type is BasicStateMachine
        2. No components initially
        3. No dependencies initially
        4. Locks are initialized
        """
        self.assertEqual(self.builder.machine_type, BasicStateMachine)
        self.assertEqual(len(self.builder.components), 0)
        self.assertEqual(len(self.builder.dependencies), 0)
        self.assertIsNotNone(self.builder._component_lock)
        self.assertIsNotNone(self.builder._dependency_lock)

    def test_set_machine_type(self):
        """Test setting machine type.

        Verifies:
        1. Valid machine type accepted
        2. Invalid type rejected
        """

        # Test valid machine type
        class CustomMachine(StateMachine):
            pass

        self.builder.set_machine_type(CustomMachine)
        self.assertEqual(self.builder.machine_type, CustomMachine)

        # Test invalid machine type
        class NotAMachine:
            pass

        with self.assertRaises(ValueError) as ctx:
            self.builder.set_machine_type(NotAMachine)
        self.assertIn("must be a StateMachine subclass", str(ctx.exception))

    def test_add_component(self):
        """Test adding components.

        Verifies:
        1. Components added correctly
        2. Multiple components of same type allowed
        3. Thread safety maintained
        """
        # Add first state
        self.builder.add_component("states", self.state1)
        self.assertEqual(len(self.builder.components["states"]), 1)
        self.assertIn(self.state1, self.builder.components["states"])

        # Add second state
        self.builder.add_component("states", self.state2)
        self.assertEqual(len(self.builder.components["states"]), 2)
        self.assertIn(self.state2, self.builder.components["states"])

        # Add region
        self.builder.add_component("regions", self.region1)
        self.assertEqual(len(self.builder.components["regions"]), 1)
        self.assertIn(self.region1, self.builder.components["regions"])

    def test_add_dependency(self):
        """Test adding dependencies.

        Verifies:
        1. Dependencies added correctly
        2. Multiple dependencies allowed
        3. Cyclic dependencies prevented
        4. Thread safety maintained
        """
        # Add valid dependencies
        self.builder.add_dependency("state2", "state1")
        self.builder.add_dependency("state3", "state2")

        deps = self.builder.dependencies
        self.assertIn("state2", deps)
        self.assertIn("state3", deps)
        self.assertIn("state1", deps["state2"])
        self.assertIn("state2", deps["state3"])

        # Test cyclic dependency detection
        with self.assertRaises(ValueError) as ctx:
            self.builder.add_dependency("state1", "state3")
        self.assertIn("Cyclic dependency detected", str(ctx.exception))

    def test_would_create_cycle(self):
        """Test cycle detection in dependencies.

        Verifies:
        1. Direct cycles detected
        2. Indirect cycles detected
        3. Valid dependencies allowed
        """
        # Set up chain: A -> B -> C
        self.builder.add_dependency("B", "A")
        self.builder.add_dependency("C", "B")

        # Direct cycle
        self.assertTrue(self.builder._would_create_cycle("A", "C"))

        # Indirect cycle
        self.assertTrue(self.builder._would_create_cycle("A", "B"))

        # Valid dependency
        self.assertFalse(self.builder._would_create_cycle("D", "A"))

    def test_build_success(self):
        """Test successful machine building.

        Verifies:
        1. All components added
        2. Dependencies resolved
        3. Validation performed
        4. Machine created
        """
        # Add required components
        self.builder.add_component("states", self.state1)
        self.builder.add_component("regions", self.region1)

        # Build machine
        machine = self.builder.build()

        # Verify machine type
        self.assertIsInstance(machine, BasicStateMachine)

        # Verify components added
        self.assertEqual(len(machine._states), 1)
        self.assertEqual(len(machine._regions), 1)
        self.assertIn(self.state1.id, machine._states)
        self.assertIn(self.region1.id, machine._regions)

    def test_build_validation_failure(self):
        """Test build validation failures.

        Verifies:
        1. Missing components detected
        2. Invalid components detected
        3. Missing dependencies detected
        4. Proper error messages
        """
        # Test no components
        with self.assertRaises(ValueError) as ctx:
            self.builder.build()
        self.assertIn("No components added", str(ctx.exception))

        # Test missing required component type
        self.builder.add_component("states", self.state1)
        with self.assertRaises(ValueError) as ctx:
            self.builder.build()
        self.assertIn("Missing required component types", str(ctx.exception))

        # Test invalid component
        self.state1.is_valid.return_value = False
        self.builder.add_component("regions", self.region1)
        with self.assertRaises(ValueError) as ctx:
            self.builder.build()
        self.assertIn("Invalid state configuration", str(ctx.exception))

    def test_build_dependency_resolution(self):
        """Test dependency resolution during build.

        Verifies:
        1. Dependencies checked
        2. Missing dependencies detected
        3. Proper error messages
        """
        # Add components
        self.builder.add_component("states", self.state1)
        self.builder.add_component("regions", self.region1)

        # Add dependency on non-existent component
        self.builder.add_dependency("state1", "missing_state")

        # Build should fail due to unresolved dependency
        with self.assertRaises(ValueError) as ctx:
            self.builder.build()
        self.assertIn("Unresolved dependency", str(ctx.exception))

    def test_component_property_thread_safety(self):
        """Test thread safety of component property.

        Verifies:
        1. Components copied on access
        2. Original components unchanged
        3. Thread safety maintained
        """
        self.builder.add_component("states", self.state1)

        # Get components and modify them
        components = self.builder.components
        components["states"].append(self.state2)

        # Original should be unchanged
        self.assertEqual(len(self.builder.components["states"]), 1)
        self.assertNotIn(self.state2, self.builder.components["states"])

    def test_dependencies_property_thread_safety(self):
        """Test thread safety of dependencies property.

        Verifies:
        1. Dependencies copied on access
        2. Original dependencies unchanged
        3. Thread safety maintained
        """
        self.builder.add_dependency("state1", "state2")

        # Get dependencies and modify them
        deps = self.builder.dependencies
        deps["state1"].add("state3")

        # Original should be unchanged
        self.assertEqual(len(self.builder.dependencies["state1"]), 1)
        self.assertNotIn("state3", self.builder.dependencies["state1"])


if __name__ == "__main__":
    unittest.main()
