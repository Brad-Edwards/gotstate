// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model interfaces for state machine implementations.

/// A `ModelDefinition` describes states, events, and transitions in the FSM.
///
/// **Key Considerations:**
/// - This acts as the blueprint for the FSM.
/// - `root_state` is the entry point of the FSM.
/// - Must ensure all references (states, events, transitions) are consistent.
pub trait ModelDefinition: Send + Sync {
    type StateDefinition: StateDefinition;
    type EventDefinition: EventDefinition;
    type TransitionDefinition: TransitionDefinition<
        StateDef = Self::StateDefinition,
        EventDef = Self::EventDefinition,
    >;

    fn root_state(&self) -> &Self::StateDefinition;
    fn states(&self) -> &[Self::StateDefinition];
    fn events(&self) -> &[Self::EventDefinition];
    fn transitions(&self) -> &[Self::TransitionDefinition];
}

/// `StateDefinition` represents a single state within the FSM, potentially hierarchical.
///
/// **Key Considerations:**
/// - Composite states may contain child states.
/// - Hierarchy validation is crucial to ensure well-formed state trees.
pub trait StateDefinition: Send + Sync + Sized {
    fn name(&self) -> &str;
    fn children(&self) -> &[Self];
    fn is_composite(&self) -> bool;
}

/// `EventDefinition` describes an event type that triggers transitions.
///
/// **Key Considerations:**
/// - Names must be unique for easy lookup.
/// - Events may map to external signals or system occurrences.
pub trait EventDefinition: Send + Sync {
    fn name(&self) -> &str;
}

/// `TransitionDefinition` couples source state, target state, and the event that triggers the transition.
/// It may also include a guard condition.
///
/// **Key Considerations:**
/// - Must ensure `source_state` and `target_state` are valid states from the model.
/// - `has_guard` indicates whether guard evaluation is required.
pub trait TransitionDefinition: Send + Sync {
    type StateDef: StateDefinition;
    type EventDef: EventDefinition;

    fn source_state(&self) -> &Self::StateDef;
    fn target_state(&self) -> &Self::StateDef;
    fn event(&self) -> &Self::EventDef;
    fn has_guard(&self) -> bool;
}

/// `ModelBuilderInternal` allows incremental construction of the model.
///
/// **Key Considerations:**
/// - Used internally by builders to define states, events, transitions before finalizing.
/// - Ensures that after `finalize`, a consistent model set is returned.
pub trait ModelBuilderInternal: Send {
    type Error;
    type StateDef: StateDefinition;
    type EventDef: EventDefinition;
    type TransitionDef: TransitionDefinition<StateDef = Self::StateDef, EventDef = Self::EventDef>;

    fn define_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn define_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn define_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;

    fn finalize(
        self,
    ) -> Result<
        (
            Vec<Self::StateDef>,
            Vec<Self::EventDef>,
            Vec<Self::TransitionDef>,
        ),
        Self::Error,
    >;
}

/// `ImmutableModelStore` is a read-only model repository.
///
/// **Key Considerations:**
/// - Allows quick lookups by name.
/// - Useful at runtime for referencing states/events without mutating the model.
pub trait ImmutableModelStore: Send + Sync {
    type StateDef: StateDefinition;
    type EventDef: EventDefinition;
    type TransitionDef: TransitionDefinition<StateDef = Self::StateDef, EventDef = Self::EventDef>;

    fn root_state(&self) -> &Self::StateDef;
    fn states(&self) -> &[Self::StateDef];
    fn events(&self) -> &[Self::EventDef];
    fn transitions(&self) -> &[Self::TransitionDef];
    fn find_state_by_name(&self, name: &str) -> Option<&Self::StateDef>;
    fn find_event_by_name(&self, name: &str) -> Option<&Self::EventDef>;
}

/// `HierarchyValidator` ensures proper structuring of states.
///
/// **Key Considerations:**
/// - Must detect cycles or malformed hierarchies.
/// - Called typically after model building and before runtime execution.
pub trait HierarchyValidator: Send + Sync {
    type StateDef: StateDefinition;
    type Error;

    fn validate_hierarchy(&self, root_state: &Self::StateDef) -> Result<(), Self::Error>;
}
