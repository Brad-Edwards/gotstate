// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core interfaces for state machine implementations.
use crate::behavior::Behavior;
use crate::diagnostics::Diagnostics;
use crate::model::{ModelDefinition, TransitionDefinition};
use crate::resource::ResourceManager;

/// The `CoreEngine` trait represents the central FSM engine interface.
/// It defines lifecycle controls (start, stop) and event processing.
///
/// **Key Considerations:**
/// - `Error` type: Defines error types specific to the engine's operations.
/// - `StateHandle`: Represents an opaque handle or reference to the current state. This might be a state ID or a more complex structure.
/// - `Event` & `TransitionResult`: Types related to event handling and state transitions.
///
/// **Gotchas:**
/// - Implementations must ensure thread-safety (`Send + Sync`) due to concurrency expectations.
/// - Transition results must be deterministic or well-defined even if events arrive in an unexpected order.
pub trait CoreEngine: Send + Sync {
    type Error;
    type StateHandle: Send + Sync;
    type Event: Send;
    type TransitionResult: Send;

    /// Starts the FSM engine. Typically sets the initial state.
    fn start(&self) -> Result<(), Self::Error>;

    /// Stops the FSM engine. Typically halts transitions and frees resources.
    fn stop(&self) -> Result<(), Self::Error>;

    /// Processes an incoming event, potentially causing a state transition.
    fn process_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error>;

    /// Returns a handle/reference to the current state.
    fn current_state(&self) -> Self::StateHandle;
}

/// The `CoreExecutor` trait defines the low-level execution logic for transitions.
///
/// **Key Considerations:**
/// - Provides `execute_transition` which is central to how a transition occurs given an event.
/// - `Context` acts as a provider for diagnostics and resource management.
/// - `Behavior` defines how guards and actions are executed.
///
/// **Gotchas:**
/// - Must ensure that the transition definition is consistent with the model.
/// - Contexts must remain valid throughout the transition's execution.
pub trait CoreExecutor: Send + Sync {
    type Error;
    type StateHandle: Send + Sync;
    type Event: Send;
    type TransitionResult: Send;
    type TransitionDef: TransitionDefinition;
    type Behavior: Behavior<TransitionDef = Self::TransitionDef>;
    type ResourceManager: ResourceManager;
    type Diagnostics: Diagnostics;
    type Model: ModelDefinition;

    /// Executes a given transition in response to an event.
    /// This may involve evaluating guards, performing actions, and updating state.
    fn execute_transition(
        &self,
        transition: &Self::TransitionDef,
        event: &Self::Event,
    ) -> Result<Self::TransitionResult, Self::Error>;

    /// Returns the current state after executing any transitions.
    fn current_state(&self) -> Self::StateHandle;

    /// Provides contextual information (diagnostics, resources) needed during execution.
    fn context(
        &self,
    ) -> &dyn CoreContextProvider<
        StateHandle = Self::StateHandle,
        Diagnostics = Self::Diagnostics,
        ResourceManager = Self::ResourceManager,
    >;
}

/// The `CoreContextProvider` trait supplies the current state handle, diagnostics and resource manager.
///
/// **Key Considerations:**
/// - Helps decouple execution logic from the environment and resources.
/// - Ensures uniform access to diagnostics and resource management across different implementations.
pub trait CoreContextProvider: Send + Sync {
    type StateHandle: Send + Sync;
    type Diagnostics: Diagnostics;
    type ResourceManager: ResourceManager;

    fn state_handle(&self) -> Self::StateHandle;
    fn diagnostics(&self) -> &Self::Diagnostics;
    fn resource_manager(&self) -> &Self::ResourceManager;
}

/// The `CoreDispatcher` trait abstracts the mechanism of dispatching events.
///
/// **Key Considerations:**
/// - Event queue processing and direct dispatch methods.
/// - Ensures that queued events can be processed in a controlled manner.
///
/// **Gotchas:**
/// - Must consider concurrency and ordering of events.
pub trait CoreDispatcher: Send + Sync {
    type Error;
    type Event: Send;
    type TransitionResult: Send;
    type Executor: CoreExecutor<Event = Self::Event, TransitionResult = Self::TransitionResult>;

    fn dispatch_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error>;
    fn process_queued_events(&self) -> Result<(), Self::Error>;
}

/// The `LifecycleController` trait provides generic lifecycle management beyond the FSM states.
///
/// **Key Considerations:**
/// - Often used for initialization, shutdown, and cleanup tasks unrelated directly to state transitions.
/// - May tie into external system resources or runtime environments.
pub trait LifecycleController: Send + Sync {
    type Error;

    fn start(&self) -> Result<(), Self::Error>;
    fn stop(&self) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}
