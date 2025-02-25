import pytest
import asyncio
from gotstate import State
from gotstate.extensions.async_sm import AsyncStateMachine


@pytest.mark.asyncio
async def test_async_machine_creation():
    """Test basic async state machine creation."""
    machine = AsyncStateMachine("test_machine")
    
    assert machine.name == "test_machine"
    assert len(machine.states) == 0
    assert len(machine.pseudostates) == 0
    assert len(machine.transitions) == 0
    assert len(machine.active_states) == 0
    assert machine.is_started is False
    assert machine.is_terminated is False


@pytest.mark.asyncio
async def test_async_machine_start_stop():
    """Test starting and stopping the async state machine."""
    machine = AsyncStateMachine("test_machine")
    
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
    await machine.start()
    
    assert machine.is_started is True
    assert machine.is_state_active(state1) is True
    assert len(action_log) == 1
    assert action_log[0] == "state1_entry"
    
    # Stop the machine
    await machine.stop()
    
    assert machine.is_started is False
    assert len(machine.active_states) == 0
    assert len(action_log) == 2
    assert action_log[1] == "state1_exit"


@pytest.mark.asyncio
async def test_async_event_processing():
    """Test processing events in the async state machine."""
    machine = AsyncStateMachine("test_machine")
    
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
    
    # Start machine
    await machine.start()
    
    # Start event loop in background and process event
    event_loop_task = asyncio.create_task(machine.run_event_loop())
    
    try:
        # Process event asynchronously
        await machine.process_event("test_event")
        
        # Wait for event to be processed
        await asyncio.sleep(0.1)
        
        assert machine.is_state_active(state1) is False
        assert machine.is_state_active(state2) is True
        
        assert len(state_log) == 3
        assert state_log[0] == "enter:state1:None"  # Initial entry
        assert state_log[1] == "exit:state1:test_event"
        assert state_log[2] == "enter:state2:test_event"
    finally:
        # Stop the machine and cancel the event loop
        await machine.stop()
        event_loop_task.cancel()
        try:
            await event_loop_task
        except asyncio.CancelledError:
            pass 