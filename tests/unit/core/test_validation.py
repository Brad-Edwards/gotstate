import pytest

from hsm.core.states import CompositeState
from hsm.core.validations import ValidationError


def test_circular_dependency_detection():
    """Test that circular parent-child relationships are detected"""
    state1 = CompositeState("State1")
    state2 = CompositeState("State2")

    state1.add_child_state(state2)

    with pytest.raises(ValidationError, match="Circular dependency"):
        state2.add_child_state(state1)
