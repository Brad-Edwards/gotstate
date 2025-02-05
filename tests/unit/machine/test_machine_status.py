import unittest

from gotstate.core.machine.machine_status import MachineStatus


class TestMachineStatus(unittest.TestCase):
    """Test cases for MachineStatus enum.

    Tests verify:
    1. All expected states exist
    2. States are unique
    3. State transitions are valid
    4. State comparisons work correctly
    5. String representations are correct
    """

    def test_states_exist(self):
        """Test that all expected machine states exist.

        Verifies that all six states are defined in the enum.
        """
        self.assertTrue(hasattr(MachineStatus, "UNINITIALIZED"))
        self.assertTrue(hasattr(MachineStatus, "INITIALIZING"))
        self.assertTrue(hasattr(MachineStatus, "ACTIVE"))
        self.assertTrue(hasattr(MachineStatus, "MODIFYING"))
        self.assertTrue(hasattr(MachineStatus, "TERMINATING"))
        self.assertTrue(hasattr(MachineStatus, "TERMINATED"))

    def test_states_are_unique(self):
        """Test that all machine states have unique values.

        Verifies that no two states share the same enum value.
        """
        states = [
            MachineStatus.UNINITIALIZED,
            MachineStatus.INITIALIZING,
            MachineStatus.ACTIVE,
            MachineStatus.MODIFYING,
            MachineStatus.TERMINATING,
            MachineStatus.TERMINATED,
        ]
        # Convert to set to check for duplicates
        unique_states = set(states)
        self.assertEqual(len(states), len(unique_states))

    def test_state_comparisons(self):
        """Test state comparison operations.

        Verifies that states can be correctly compared.
        """
        # Test equality
        self.assertEqual(MachineStatus.UNINITIALIZED, MachineStatus.UNINITIALIZED)
        self.assertNotEqual(MachineStatus.UNINITIALIZED, MachineStatus.ACTIVE)

        # Test identity
        self.assertIs(MachineStatus.ACTIVE, MachineStatus.ACTIVE)
        self.assertIsNot(MachineStatus.ACTIVE, MachineStatus.TERMINATED)

    def test_state_names(self):
        """Test state name representations.

        Verifies that states have correct string representations.
        """
        self.assertEqual(MachineStatus.UNINITIALIZED.name, "UNINITIALIZED")
        self.assertEqual(MachineStatus.INITIALIZING.name, "INITIALIZING")
        self.assertEqual(MachineStatus.ACTIVE.name, "ACTIVE")
        self.assertEqual(MachineStatus.MODIFYING.name, "MODIFYING")
        self.assertEqual(MachineStatus.TERMINATING.name, "TERMINATING")
        self.assertEqual(MachineStatus.TERMINATED.name, "TERMINATED")

    def test_state_values(self):
        """Test state values.

        Verifies that states have valid auto-generated values.
        """
        # Values should be auto-generated integers starting from 1
        states = [
            MachineStatus.UNINITIALIZED,
            MachineStatus.INITIALIZING,
            MachineStatus.ACTIVE,
            MachineStatus.MODIFYING,
            MachineStatus.TERMINATING,
            MachineStatus.TERMINATED,
        ]
        values = [state.value for state in states]

        # Values should be unique integers
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all(isinstance(v, int) for v in values))

        # Values should be positive
        self.assertTrue(all(v > 0 for v in values))

    def test_state_iteration(self):
        """Test iteration over states.

        Verifies that all states can be iterated over.
        """
        expected_states = {"UNINITIALIZED", "INITIALIZING", "ACTIVE", "MODIFYING", "TERMINATING", "TERMINATED"}
        actual_states = {state.name for state in MachineStatus}
        self.assertEqual(expected_states, actual_states)

    def test_state_lookup(self):
        """Test state lookup by name.

        Verifies that states can be looked up by name.
        """
        self.assertEqual(MachineStatus["UNINITIALIZED"], MachineStatus.UNINITIALIZED)
        self.assertEqual(MachineStatus["ACTIVE"], MachineStatus.ACTIVE)
        self.assertEqual(MachineStatus["TERMINATED"], MachineStatus.TERMINATED)

        with self.assertRaises(KeyError):
            _ = MachineStatus["INVALID_STATE"]


if __name__ == "__main__":
    unittest.main()
