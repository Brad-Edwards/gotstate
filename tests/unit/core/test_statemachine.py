import pytest
from gotstate import StateMachine, State, Transition, InitialState


def test_machine_creation():
    """Test basic state machine creation."""
    machine = StateMachine("test_machine")
    
    assert machine.name == "test_machine"
    assert len(machine.states) == 0
    assert len(machine.pseudostates) == 0
    assert len(machine.transitions) == 0
    assert len(machine.active_states) == 0
    assert machine.is_started is False
    assert machine.is_terminated is False


def test_adding_states():
    """Test adding states to the state machine."""
    machine = StateMachine("test_machine")
    
    state1 = State("state1")
    state2 = State("state2")
    
    machine.add_state(state1, initial=True)
    machine.add_state(state2)
    
    assert len(machine.states) == 2
    assert machine.get_state("state1") is state1
    assert machine.get_state("state2") is state2
    
    # Test with duplicate state
    with pytest.raises(Exception):
        machine.add_state(State("state1"))


def test_adding_transitions():
    """Test adding transitions to the state machine."""
    machine = StateMachine("test_machine")
    
    state1 = State("state1")
    state2 = State("state2")
    
    machine.add_state(state1)
    machine.add_state(state2)
    
    transition = machine.add_transition(state1, state2, "test_event")
    
    assert len(machine.transitions) == 1
    assert transition.source is state1
    assert transition.target is state2
    assert transition.event_id == "test_event"


def test_machine_start_stop():
    """Test starting and stopping the state machine."""
    machine = StateMachine("test_machine")
    
    state1 = State("state1")
    state2 = State("state2")
    
    # Track action execution
    action_log = []
    
    @state1.on_entry
    def state1_entry(event_id, event_data):
        action_log.append("state1_entry")
    
    @state1.on_exit
    def state1_exit(event_id, event_data):
        action_log.append("state1_exit")
    
    machine.add_state(state1, initial=True)
    machine.add_state(state2)
    
    # Start the machine
    machine.start()
    
    assert machine.is_started is True
    assert machine.is_state_active(state1) is True
    assert len(action_log) == 1
    assert action_log[0] == "state1_entry"
    
    # Stop the machine
    machine.stop()
    
    assert machine.is_started is False
    assert len(machine.active_states) == 0
    assert len(action_log) == 2
    assert action_log[1] == "state1_exit"


def test_event_processing():
    """Test processing events in the state machine."""
    machine = StateMachine("test_machine")
    
    state1 = State("state1")
    state2 = State("state2")
    
    # Track state changes
    state_log = []
    
    @state1.on_entry
    def state1_entry(event_id, event_data):
        state_log.append(f"enter:state1:{event_id}")
    
    @state1.on_exit
    def state1_exit(event_id, event_data):
        state_log.append(f"exit:state1:{event_id}")
    
    @state2.on_entry
    def state2_entry(event_id, event_data):
        state_log.append(f"enter:state2:{event_id}")
    
    machine.add_state(state1, initial=True)
    machine.add_state(state2)
    machine.add_transition(state1, state2, "test_event")
    
    # Start machine and process event
    machine.start()
    machine.process_event("test_event")
    
    assert machine.is_state_active(state1) is False
    assert machine.is_state_active(state2) is True
    
    assert len(state_log) == 3
    assert state_log[0] == "enter:state1:None"  # Initial entry
    assert state_log[1] == "exit:state1:test_event"
    assert state_log[2] == "enter:state2:test_event" 