"""Async StateMachine extension for gotstate."""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from gotstate.types.common import StateId, EventId, EventData
from gotstate.core.statemachine import StateMachine
from gotstate.core.state import State
from gotstate.core.transition import Transition
from gotstate.core.event import Event
from gotstate.core.pseudostate import PseudoState
from gotstate.core.guard import Guard
from gotstate.core.action import Action
from gotstate.core.exceptions import StateMachineNotStartedError


class AsyncStateMachine(StateMachine):
    """
    Asynchronous version of the StateMachine class.
    
    This class extends StateMachine to provide asynchronous event processing.
    """
    
    def __init__(self, name: str):
        """
        Initialize a new async state machine.
        
        Args:
            name: Name of the state machine
        """
        super().__init__(name)
        self._event_queue_lock = asyncio.Lock()
        self._event_processing_lock = asyncio.Lock()
        self._event_available = asyncio.Event()
    
    async def start(self) -> None:
        """
        Start the state machine asynchronously.
        
        This activates the initial state and begins processing events.
        """
        async with self._event_processing_lock:
            # Use the parent class implementation but with a lock
            super().start()
    
    async def stop(self) -> None:
        """
        Stop the state machine asynchronously.
        
        This deactivates all active states and stops processing events.
        """
        async with self._event_processing_lock:
            # Use the parent class implementation but with a lock
            super().stop()
    
    async def reset(self) -> None:
        """
        Reset the state machine asynchronously.
        
        This stops the state machine and clears all history states.
        """
        async with self._event_processing_lock:
            # Use the parent class implementation but with a lock
            super().reset()
    
    async def process_event(self, event_id: str, event_data: Optional[EventData] = None) -> bool:
        """
        Process an event asynchronously.
        
        Args:
            event_id: ID of the event to process
            event_data: Optional data associated with the event
            
        Returns:
            True if the event was consumed, False otherwise
            
        Raises:
            StateMachineNotStartedError: If the state machine has not been started
        """
        if not self._started:
            raise StateMachineNotStartedError("State machine has not been started")
        
        if self._terminated:
            return False
        
        # Create an event object
        event_id_typed = EventId(event_id)
        event_data_typed = {} if event_data is None else event_data
        
        async with self._event_queue_lock:
            # Add the event to the queue
            self._event_queue.append((event_id_typed, event_data_typed))
            self._event_available.set()
        
        # Try to acquire the processing lock to process the event
        if not self._processing_event and self._event_processing_lock.locked():
            return await self._process_event_queue_async()
        
        return True
    
    async def _process_event_queue_async(self) -> bool:
        """
        Process the event queue asynchronously.
        
        Returns:
            True if at least one event was consumed, False otherwise
        """
        consumed = False
        
        # Try to acquire the processing lock
        if not await self._event_processing_lock.acquire():
            return False
        
        try:
            self._processing_event = True
            
            while not self._terminated:
                # Check if there are events in the queue
                async with self._event_queue_lock:
                    if not self._event_queue:
                        self._event_available.clear()
                        break
                    
                    event_id, event_data = self._event_queue.pop(0)
                
                # Process the event
                event_consumed = await self._process_event_internal_async(event_id, event_data)
                consumed = consumed or event_consumed
        finally:
            self._processing_event = False
            self._event_processing_lock.release()
        
        return consumed
    
    async def _process_event_internal_async(self, event_id: EventId, event_data: EventData) -> bool:
        """
        Process an event internally asynchronously.
        
        Args:
            event_id: ID of the event to process
            event_data: Data associated with the event
            
        Returns:
            True if the event was consumed, False otherwise
        """
        # Find transitions that can be triggered by this event
        enabled_transitions = self._find_enabled_transitions(event_id, event_data)
        
        # If no transitions are enabled, the event is not consumed
        if not enabled_transitions:
            return False
        
        # Execute the transitions
        for transition in enabled_transitions:
            await self._execute_transition_async(transition, event_id, event_data)
        
        return True
    
    async def _execute_transition_async(self, transition: Transition, event_id: EventId, event_data: EventData) -> None:
        """
        Execute a transition asynchronously.
        
        Args:
            transition: The transition to execute
            event_id: ID of the event that triggered the transition
            event_data: Data associated with the event
        """
        # This is a simplified version for now; in a real implementation, 
        # we would need to make all the transition execution methods async
        self._execute_transition(transition, event_id, event_data)
    
    async def run_event_loop(self) -> None:
        """
        Run the event processing loop asynchronously.
        
        This method will continuously process events from the queue until the
        state machine is terminated or stopped.
        """
        while self._started and not self._terminated:
            # Wait for an event to be available
            await self._event_available.wait()
            
            # Process the event queue
            await self._process_event_queue_async()
            
            # Sleep a bit to avoid hogging the CPU
            await asyncio.sleep(0.001) 