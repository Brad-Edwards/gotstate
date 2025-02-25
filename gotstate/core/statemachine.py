"""StateMachine module for gotstate."""

import threading
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from uuid import uuid4

from gotstate.types.common import StateId, EventId, EventData, EventQueue, EventQueueEntry
from gotstate.core.state import State
from gotstate.core.transition import Transition
from gotstate.core.event import Event
from gotstate.core.pseudostate import PseudoState, InitialState, TerminateState
from gotstate.core.region import Region
from gotstate.core.guard import Guard
from gotstate.core.action import Action
from gotstate.core.exceptions import (
    StateMachineError,
    StateMachineNotStartedError,
    StateMachineAlreadyStartedError,
    StateError,
    StateNotFoundError,
    TransitionError,
    EventError,
)


class StateMachine:
    """
    Represents a hierarchical state machine.
    
    A state machine consists of states, transitions, and regions, and processes events
    to transition between states according to UML state machine semantics.
    """
    
    def __init__(self, name: str):
        """
        Initialize a new state machine.
        
        Args:
            name: Name of the state machine
        """
        self._name = name
        self._states: Dict[StateId, State] = {}
        self._pseudostates: Dict[StateId, PseudoState] = {}
        self._regions: Dict[str, Region] = {}
        self._transitions: List[Transition] = []
        
        # Current state configuration
        self._active_states: Set[State] = set()
        self._root_state: Optional[State] = None
        self._initial_state: Optional[State] = None
        
        # Event processing
        self._event_queue: EventQueue = []
        self._processing_event = False
        self._terminated = False
        self._started = False
        
        # Thread safety
        self._lock = threading.RLock()
    
    @property
    def name(self) -> str:
        """Get the state machine name."""
        return self._name
    
    @property
    def states(self) -> List[State]:
        """Get all states in the state machine."""
        return list(self._states.values())
    
    @property
    def pseudostates(self) -> List[PseudoState]:
        """Get all pseudostates in the state machine."""
        return list(self._pseudostates.values())
    
    @property
    def transitions(self) -> List[Transition]:
        """Get all transitions in the state machine."""
        return self._transitions.copy()
    
    @property
    def active_states(self) -> Set[State]:
        """Get the currently active states."""
        return self._active_states.copy()
    
    @property
    def is_started(self) -> bool:
        """Check if the state machine has been started."""
        return self._started
    
    @property
    def is_terminated(self) -> bool:
        """Check if the state machine has been terminated."""
        return self._terminated
    
    def add_state(self, state: State, initial: bool = False) -> None:
        """
        Add a state to the state machine.
        
        Args:
            state: The state to add
            initial: Whether this state is the initial state of the state machine
            
        Raises:
            StateError: If a state with the same ID already exists
        """
        with self._lock:
            if state.id in self._states:
                raise StateError(f"State with ID '{state.id}' already exists")
            
            self._states[state.id] = state
            
            if initial:
                self._initial_state = state
    
    def add_pseudostate(self, pseudostate: PseudoState) -> None:
        """
        Add a pseudostate to the state machine.
        
        Args:
            pseudostate: The pseudostate to add
            
        Raises:
            StateError: If a pseudostate with the same ID already exists
        """
        with self._lock:
            if pseudostate.id in self._pseudostates:
                raise StateError(f"Pseudostate with ID '{pseudostate.id}' already exists")
            
            self._pseudostates[pseudostate.id] = pseudostate
    
    def add_region(self, region: Region) -> None:
        """
        Add a region to the state machine.
        
        Args:
            region: The region to add
            
        Raises:
            RegionError: If a region with the same ID already exists
        """
        with self._lock:
            if region.id in self._regions:
                raise StateError(f"Region with ID '{region.id}' already exists")
            
            self._regions[region.id] = region
    
    def add_transition(
        self,
        source: Union[State, PseudoState],
        target: Optional[Union[State, PseudoState]],
        event_id: Optional[str] = None,
        guard: Optional[Guard] = None,
        actions: Optional[List[Action]] = None,
    ) -> Transition:
        """
        Add a transition to the state machine.
        
        Args:
            source: Source state or pseudostate of the transition
            target: Target state or pseudostate of the transition (None for internal transitions)
            event_id: ID of the event that triggers the transition (None for completion transitions)
            guard: Optional guard condition for the transition
            actions: Optional list of actions to execute during the transition
            
        Returns:
            The created transition
            
        Raises:
            StateError: If the source or target state is not in the state machine
        """
        with self._lock:
            # Verify that source and target states are in the state machine
            source_id = source.id
            if source_id not in self._states and source_id not in self._pseudostates:
                raise StateError(f"Source state or pseudostate with ID '{source_id}' not found in state machine")
            
            if target is not None:
                target_id = target.id
                if target_id not in self._states and target_id not in self._pseudostates:
                    raise StateError(f"Target state or pseudostate with ID '{target_id}' not found in state machine")
            
            # Create the transition
            transition = Transition(source, target, event_id, guard, actions)
            self._transitions.append(transition)
            
            return transition
    
    def get_state(self, state_id: Union[str, StateId]) -> State:
        """
        Get a state by its ID.
        
        Args:
            state_id: ID of the state to get
            
        Returns:
            The state with the given ID
            
        Raises:
            StateNotFoundError: If no state with the given ID exists
        """
        state_id_typed = StateId(state_id) if isinstance(state_id, str) else state_id
        if state_id_typed not in self._states:
            raise StateNotFoundError(f"State with ID '{state_id}' not found")
        
        return self._states[state_id_typed]
    
    def get_pseudostate(self, state_id: Union[str, StateId]) -> PseudoState:
        """
        Get a pseudostate by its ID.
        
        Args:
            state_id: ID of the pseudostate to get
            
        Returns:
            The pseudostate with the given ID
            
        Raises:
            StateNotFoundError: If no pseudostate with the given ID exists
        """
        state_id_typed = StateId(state_id) if isinstance(state_id, str) else state_id
        if state_id_typed not in self._pseudostates:
            raise StateNotFoundError(f"Pseudostate with ID '{state_id}' not found")
        
        return self._pseudostates[state_id_typed]
    
    def is_state_active(self, state: Union[State, str, StateId]) -> bool:
        """
        Check if a state is active.
        
        Args:
            state: The state or state ID to check
            
        Returns:
            True if the state is active, False otherwise
            
        Raises:
            StateNotFoundError: If no state with the given ID exists
        """
        if isinstance(state, (str, type(StateId("")))):
            state = self.get_state(state)
        
        return state in self._active_states
    
    def start(self) -> None:
        """
        Start the state machine.
        
        This activates the initial state and begins processing events.
        
        Raises:
            StateMachineAlreadyStartedError: If the state machine has already been started
            StateMachineError: If no initial state has been set
        """
        with self._lock:
            if self._started:
                raise StateMachineAlreadyStartedError("State machine has already been started")
            
            if self._initial_state is None:
                raise StateMachineError("No initial state has been set")
            
            self._started = True
            self._enter_state(self._initial_state, None, None)
    
    def _enter_state(self, state: State, event_id: Optional[EventId], event_data: Optional[EventData]) -> None:
        """
        Enter a state and all its ancestors.
        
        Args:
            state: The state to enter
            event_id: Optional ID of the event that triggered the entry
            event_data: Optional data associated with the event
        """
        # Enter ancestors first (from root to leaf)
        ancestors = state.ancestors
        
        for ancestor in reversed(ancestors):
            if ancestor not in self._active_states:
                self._active_states.add(ancestor)
                # Always execute entry actions, even if event is None
                event_id_to_use = event_id if event_id is not None else EventId("None")
                event_data_to_use = event_data if event_data is not None else {}
                ancestor.execute_entry_actions(event_id_to_use, event_data_to_use)
        
        # Enter the state itself
        if state not in self._active_states:
            self._active_states.add(state)
            # Always execute entry actions, even if event is None
            event_id_to_use = event_id if event_id is not None else EventId("None")
            event_data_to_use = event_data if event_data is not None else {}
            state.execute_entry_actions(event_id_to_use, event_data_to_use)
        
        # If the state is composite, enter its initial substate
        if state.is_composite:
            # Find an initial pseudostate within this state
            initial = None
            for ps in self._pseudostates.values():
                if isinstance(ps, InitialState) and ps.parent is state:
                    initial = ps
                    break
            
            # If an initial pseudostate was found, follow its transition
            if initial is not None:
                for transition in self._transitions:
                    if transition.source is initial:
                        if event_id is not None and event_data is not None:
                            self._execute_transition(transition, event_id, event_data)
                        else:
                            self._execute_transition(transition, event_id_to_use, event_data_to_use)
                        break
    
    def stop(self) -> None:
        """
        Stop the state machine.
        
        This deactivates all active states and stops processing events.
        """
        with self._lock:
            self._exit_all_states(None, None)
            self._active_states.clear()
            self._event_queue.clear()
            self._started = False
            self._terminated = False
    
    def reset(self) -> None:
        """
        Reset the state machine.
        
        This stops the state machine and clears all history states.
        """
        with self._lock:
            self.stop()
            
            # Reset all history states
            for pseudostate in self._pseudostates.values():
                if hasattr(pseudostate, "reset"):
                    pseudostate.reset()
    
    def process_event(self, event_id: str, event_data: Optional[EventData] = None) -> bool:
        """
        Process an event.
        
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
        
        with self._lock:
            # Create an event object
            event_id_typed = EventId(event_id)
            event_data_typed = {} if event_data is None else event_data
            
            # Add the event to the queue
            self._event_queue.append((event_id_typed, event_data_typed))
            
            # Process the event queue if not already processing
            if not self._processing_event:
                return self._process_event_queue()
            
            return True
    
    def _process_event_queue(self) -> bool:
        """
        Process the event queue.
        
        Returns:
            True if at least one event was consumed, False otherwise
        """
        consumed = False
        self._processing_event = True
        
        try:
            while self._event_queue and not self._terminated:
                event_id, event_data = self._event_queue.pop(0)
                event_consumed = self._process_event_internal(event_id, event_data)
                consumed = consumed or event_consumed
        finally:
            self._processing_event = False
        
        return consumed
    
    def _process_event_internal(self, event_id: EventId, event_data: EventData) -> bool:
        """
        Process an event internally.
        
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
            self._execute_transition(transition, event_id, event_data)
        
        return True
    
    def _find_enabled_transitions(self, event_id: EventId, event_data: EventData) -> List[Transition]:
        """
        Find all transitions that can be triggered by the given event.
        
        Args:
            event_id: ID of the event to check
            event_data: Data associated with the event
            
        Returns:
            List of transitions that can be triggered
        """
        enabled_transitions = []
        
        for transition in self._transitions:
            source_state = transition.source
            
            # Check if the source state is active
            if isinstance(source_state, State) and source_state not in self._active_states:
                continue
            
            # Check if the transition can be triggered by this event
            if transition.can_trigger(event_id, event_data):
                enabled_transitions.append(transition)
        
        return enabled_transitions
    
    def _execute_transition(self, transition: Transition, event_id: EventId, event_data: EventData) -> None:
        """
        Execute a transition.
        
        Args:
            transition: The transition to execute
            event_id: ID of the event that triggered the transition
            event_data: Data associated with the event
        """
        source = transition.source
        target = transition.target
        
        # Handle internal transitions
        if transition.is_internal:
            # Execute transition actions
            transition.execute_actions(event_id, event_data)
            return
        
        # Find the LCA (Least Common Ancestor) of source and target states
        if isinstance(source, State) and isinstance(target, State):
            lca = self._find_lca(source, target)
        else:
            # For transitions involving pseudostates, use the parent of the pseudostate
            if isinstance(source, PseudoState):
                source_parent = source.parent
                if source_parent is None:
                    lca = None
                else:
                    lca = source_parent
            elif isinstance(target, PseudoState):
                target_parent = target.parent
                if target_parent is None:
                    lca = None
                else:
                    lca = target_parent
            else:
                lca = None
        
        # Exit source state and all its ancestors up to but not including the LCA
        self._exit_state_to_lca(source, lca, event_id, event_data)
        
        # Execute transition actions
        transition.execute_actions(event_id, event_data)
        
        # Enter target state and all its ancestors from the LCA
        if target is not None:
            self._enter_state_from_lca(target, lca, event_id, event_data)
            
            # Handle special case for TerminateState
            if isinstance(target, TerminateState):
                self._terminated = True
    
    def _find_lca(self, state1: State, state2: State) -> Optional[State]:
        """
        Find the least common ancestor of two states.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            The least common ancestor, or None if no common ancestor exists
        """
        # If either state is None, there is no LCA
        if state1 is None or state2 is None:
            return None
        
        # If the states are the same, the LCA is the state itself
        if state1 is state2:
            return state1
        
        # Get the ancestors of each state
        ancestors1 = [state1] + state1.ancestors
        ancestors2 = [state2] + state2.ancestors
        
        # Find the common ancestors
        common_ancestors = set(ancestors1) & set(ancestors2)
        if not common_ancestors:
            return None
        
        # Find the shallowest common ancestor
        best_lca = None
        min_depth = float('inf')
        
        for ancestor in common_ancestors:
            depth = len(ancestor.ancestors)
            if depth < min_depth:
                min_depth = depth
                best_lca = ancestor
        
        return best_lca
    
    def _exit_state_to_lca(self, state: Union[State, PseudoState], lca: Optional[State], event_id: EventId, event_data: EventData) -> None:
        """
        Exit a state and all its ancestors up to but not including the LCA.
        
        Args:
            state: The state to exit
            lca: The LCA to stop at
            event_id: ID of the event that triggered the exit
            event_data: Data associated with the event
        """
        # If the state is a pseudostate, we don't need to exit it
        if isinstance(state, PseudoState):
            return
        
        # Exit all child states first
        for child in sorted(state.children, key=lambda s: str(s.id)):
            if isinstance(child, State) and child in self._active_states:
                self._exit_state_to_lca(child, None, event_id, event_data)
        
        # Exit the state if it's active and not the LCA
        if state in self._active_states and state is not lca:
            # Execute exit actions
            state.execute_exit_actions(event_id, event_data)
            
            # Remove from active states
            self._active_states.remove(state)
            
            # If the state has a parent and it's not the LCA, exit the parent too
            if state.parent is not None and state.parent is not lca:
                self._exit_state_to_lca(state.parent, lca, event_id, event_data)
    
    def _exit_all_states(self, event_id: Optional[EventId], event_data: Optional[EventData]) -> None:
        """
        Exit all active states.
        
        Args:
            event_id: Optional ID of the event that triggered the exit
            event_data: Optional data associated with the event
        """
        # Sort states by depth (deepest first) to ensure proper exit order
        sorted_states = sorted(
            self._active_states,
            key=lambda s: len(s.ancestors),
            reverse=True
        )
        
        # Use default values if event info is not provided
        event_id_to_use = event_id if event_id is not None else EventId("None")
        event_data_to_use = event_data if event_data is not None else {}
        
        for state in sorted_states:
            state.execute_exit_actions(event_id_to_use, event_data_to_use)
    
    def _enter_state_from_lca(self, state: Union[State, PseudoState], lca: Optional[State], event_id: EventId, event_data: EventData) -> None:
        """
        Enter a state and all its ancestors from the LCA.
        
        Args:
            state: The state to enter
            lca: The LCA to start from
            event_id: ID of the event that triggered the entry
            event_data: Data associated with the event
        """
        # Handle pseudostates
        if isinstance(state, PseudoState):
            # Special handling for different pseudostate kinds
            if isinstance(state, InitialState):
                # Find the transition from the initial state
                for transition in self._transitions:
                    if transition.source is state:
                        self._execute_transition(transition, event_id, event_data)
                        break
            elif isinstance(state, TerminateState):
                self._terminated = True
            
            return
        
        # Enter ancestors from LCA to the state
        ancestors = state.ancestors
        start_index = ancestors.index(lca) + 1 if lca in ancestors else 0
        
        for ancestor in reversed(ancestors[start_index:]):
            if ancestor not in self._active_states:
                self._active_states.add(ancestor)
                ancestor.execute_entry_actions(event_id, event_data)
        
        # Enter the state itself
        if state not in self._active_states:
            self._active_states.add(state)
            state.execute_entry_actions(event_id, event_data)
        
        # If the state is composite, enter its initial substate
        if state.is_composite:
            # Find an initial pseudostate within this state
            initial = None
            for ps in self._pseudostates.values():
                if isinstance(ps, InitialState) and ps.parent is state:
                    initial = ps
                    break
            
            # If an initial pseudostate was found, follow its transition
            if initial is not None:
                for transition in self._transitions:
                    if transition.source is initial:
                        self._execute_transition(transition, event_id, event_data)
                        break
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state machine to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the state machine
        """
        with self._lock:
            return {
                "name": self._name,
                "states": {state_id: state.to_dict() for state_id, state in self._states.items()},
                "pseudostates": {ps_id: ps.to_dict() for ps_id, ps in self._pseudostates.items()},
                "transitions": [t.to_dict() for t in self._transitions],
                "initial_state_id": self._initial_state.id if self._initial_state else None,
                "active_state_ids": [state.id for state in self._active_states],
                "started": self._started,
                "terminated": self._terminated
            } 