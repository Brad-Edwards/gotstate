from unittest.mock import Mock

from gotstate.core.region import Region
from gotstate.core.state import State, StateType


class MockState(State):
    """Mock state for testing."""

    def __init__(self, state_id: str):
        """Initialize mock state."""
        super().__init__(state_id=state_id, state_type=StateType.SIMPLE)
        self.initialize = Mock()
        self.enter = Mock()
        self.exit = Mock()
        self.is_valid = Mock(return_value=True)
        self._parent = None  # Add parent storage

    @property
    def parent(self):
        """Get the parent state."""
        return self._parent

    @parent.setter
    def parent(self, value):
        """Set the parent state."""
        self._parent = value


class MockRegion(Region):
    """Mock region for testing."""

    def __init__(self, region_id: str):
        """Initialize mock region."""
        mock_parent = MockState("mock_parent")
        super().__init__(region_id=region_id, parent_state=mock_parent)
        self.initialize = Mock()
        self.activate = Mock()
        self.deactivate = Mock()
        self.is_valid = Mock(return_value=True)
