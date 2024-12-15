// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core interfaces for state machine implementations.
use crate::behavior::Behavior;
use crate::diagnostics::Diagnostics;
use crate::model::{ModelDefinition, TransitionDefinition};
use crate::resource::ResourceManager;
use crate::types::EventId;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

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
    type Error: std::fmt::Debug;
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
    type Model: ModelDefinition<TransitionDefinition = Self::TransitionDef>;

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

    fn context_version(&self) -> u64;
    fn is_valid(&self) -> bool;
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

#[derive(Debug)]
pub enum EngineError {
    AlreadyStarted,
    NotStarted,
    Stopped,
    InvalidTransition,
    ExecutionError(String),
    DispatchError(String),
    UnknownError(String),
    PoisonError,
}

pub trait PoisonErrorConverter {
    fn from_poison_error() -> Self;
}

impl PoisonErrorConverter for EngineError {
    fn from_poison_error() -> Self {
        EngineError::PoisonError
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateHandle {
    pub state_id: String,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub name: EventId,
    pub payload: Option<String>,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Clone, Debug)]
pub struct TransitionResult {
    pub from_state: String,
    pub to_state: String,
    pub event_name: String,
}

pub struct FsmEngine<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T, Context = FsmContext<StateHandle, D, R>> + Send + Sync + 'static,
    D: Diagnostics + Send + Sync + 'static,
    M: ModelDefinition<TransitionDefinition = T> + Send + Sync + 'static,
    R: ResourceManager + Send + Sync + 'static,
    T: TransitionDefinition + Send + Sync + 'static,
{
    lifecycle: Arc<FsmLifecycleController<EngineError>>,
    dispatcher:
        Arc<FsmDispatcher<EngineError, Event, TransitionResult, FsmExecutor<B, D, M, R, T>>>,
    context: Arc<FsmContext<StateHandle, D, R>>,
    executor: Arc<FsmExecutor<B, D, M, R, T>>,
    started: Mutex<bool>,
}

impl<B, D, M, R, T> FsmEngine<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T, Context = FsmContext<StateHandle, D, R>> + Send + Sync,
    D: Diagnostics + Send + Sync,
    M: ModelDefinition<TransitionDefinition = T> + Send + Sync,
    R: ResourceManager + Send + Sync,
    T: TransitionDefinition + Send + Sync,
{
    pub fn new(
        lifecycle: FsmLifecycleController<EngineError>,
        dispatcher: FsmDispatcher<EngineError, Event, TransitionResult, FsmExecutor<B, D, M, R, T>>,
        context: FsmContext<StateHandle, D, R>,
        executor: FsmExecutor<B, D, M, R, T>,
    ) -> Self {
        Self {
            lifecycle: Arc::new(lifecycle),
            dispatcher: Arc::new(dispatcher),
            context: Arc::new(context),
            executor: Arc::new(executor),
            started: Mutex::new(false),
        }
    }
}

impl<B, D, M, R, T> CoreEngine for FsmEngine<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T, Context = FsmContext<StateHandle, D, R>> + Send + Sync + 'static,
    D: Diagnostics + Send + Sync + 'static,
    M: ModelDefinition<TransitionDefinition = T> + Send + Sync + 'static,
    R: ResourceManager + Send + Sync + 'static,
    T: TransitionDefinition + Send + Sync + 'static,
{
    type Error = EngineError;
    type StateHandle = StateHandle;
    type Event = Event;
    type TransitionResult = TransitionResult;

    fn start(&self) -> Result<(), Self::Error> {
        let mut started = self
            .started
            .lock()
            .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        if *started {
            return Err(EngineError::AlreadyStarted);
        }
        self.lifecycle.start()?;
        *started = true;
        Ok(())
    }

    fn stop(&self) -> Result<(), Self::Error> {
        let mut started = self
            .started
            .lock()
            .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        if !*started {
            return Err(EngineError::NotStarted);
        }
        self.lifecycle.stop()?;
        *started = false;
        Ok(())
    }

    fn process_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error> {
        let started = self
            .started
            .lock()
            .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        if !*started {
            return Err(EngineError::NotStarted);
        }
        self.dispatcher.dispatch_event(event)
    }

    fn current_state(&self) -> Self::StateHandle {
        self.executor.current_state()
    }
}

pub struct FsmExecutor<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T, Context = FsmContext<StateHandle, D, R>> + Send + Sync + 'static,
    B::Error: std::fmt::Debug,
    D: Diagnostics + Send + Sync + 'static,
    M: ModelDefinition<TransitionDefinition = T> + Send + Sync + 'static,
    R: ResourceManager + Send + Sync + 'static,
    T: TransitionDefinition + Clone + Send + Sync + 'static,
{
    model: Arc<M>,
    behavior: Arc<B>,
    context: Arc<FsmContext<StateHandle, D, R>>,
    current_state: Mutex<StateHandle>,
}

impl<B, D, M, R, T> FsmExecutor<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T, Context = FsmContext<StateHandle, D, R>> + Send + Sync + 'static,
    D: Diagnostics + Send + Sync + 'static,
    M: ModelDefinition<TransitionDefinition = T> + Send + Sync + 'static,
    R: ResourceManager + Send + Sync + 'static,
    T: TransitionDefinition + Clone + Send + Sync + 'static,
{
    pub fn new(
        model: M,
        behavior: B,
        context: FsmContext<StateHandle, D, R>,
        initial_state: StateHandle,
    ) -> Self {
        Self {
            model: Arc::new(model),
            behavior: Arc::new(behavior),
            context: Arc::new(context),
            current_state: Mutex::new(initial_state),
        }
    }
}

impl<B, D, M, R, T> CoreExecutor for FsmExecutor<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T, Context = FsmContext<StateHandle, D, R>> + Send + Sync + 'static,
    D: Diagnostics + Send + Sync + 'static,
    M: ModelDefinition<TransitionDefinition = T> + Send + Sync + 'static,
    R: ResourceManager + Send + Sync + 'static,
    T: TransitionDefinition + Send + Sync + 'static,
{
    type Error = EngineError;
    type StateHandle = StateHandle;
    type Event = Event;
    type TransitionResult = TransitionResult;
    type TransitionDef = T;
    type Behavior = B;
    type ResourceManager = R;
    type Diagnostics = D;
    type Model = M;

    fn execute_transition(
        &self,
        transition: &Self::TransitionDef,
        event: &Self::Event,
    ) -> Result<Self::TransitionResult, Self::Error> {
        // Evaluate guard conditions
        if !self
            .behavior
            .evaluate_guard(transition, self.context.as_ref())
            .map_err(|e| EngineError::ExecutionError(format!("Guard evaluation failed: {:?}", e)))?
        {
            return Err(EngineError::InvalidTransition);
        }

        let old_state = {
            let lock = self
                .current_state
                .lock()
                .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
            lock.state_id.clone()
        };

        // Execute actions
        if let Err(e) =
            self.behavior
                .execute_actions(transition, event, self.context.resource_manager())
        {
            return Err(EngineError::ExecutionError(format!(
                "Action execution failed: {:?}",
                e
            )));
        }

        // Determine next state from transition
        let new_state_id = match self.model.resolve_target_state(transition) {
            Some(s) => s,
            None => return Err(EngineError::InvalidTransition),
        };

        {
            let mut lock = self
                .current_state
                .lock()
                .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
            lock.state_id = new_state_id.clone();
        }

        Ok(TransitionResult {
            from_state: old_state,
            to_state: new_state_id,
            event_name: event.name.to_string(),
        })
    }

    fn current_state(&self) -> Self::StateHandle {
        let lock = self.current_state.lock().unwrap();
        StateHandle {
            state_id: lock.state_id.clone(),
        }
    }

    fn context(
        &self,
    ) -> &dyn CoreContextProvider<
        StateHandle = Self::StateHandle,
        Diagnostics = Self::Diagnostics,
        ResourceManager = Self::ResourceManager,
    > {
        self.context.as_ref()
    }
}

pub struct FsmContext<S, D, R>
where
    S: Send + Sync,
    D: Diagnostics + Send + Sync,
    R: ResourceManager + Send + Sync,
{
    diagnostics: D,
    resource_manager: R,
    initial_state: S,
}

impl<S, D, R> FsmContext<S, D, R>
where
    S: Send + Sync,
    D: Diagnostics + Send + Sync,
    R: ResourceManager + Send + Sync,
{
    pub fn new(diagnostics: D, resource_manager: R, initial_state: S) -> Self {
        Self {
            diagnostics,
            resource_manager,
            initial_state,
        }
    }
}

impl<D, R> CoreContextProvider for FsmContext<StateHandle, D, R>
where
    D: Diagnostics + Send + Sync,
    R: ResourceManager + Send + Sync,
{
    type StateHandle = StateHandle;
    type Diagnostics = D;
    type ResourceManager = R;

    fn context_version(&self) -> u64 {
        1
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn state_handle(&self) -> Self::StateHandle {
        self.initial_state.clone()
    }

    fn diagnostics(&self) -> &Self::Diagnostics {
        &self.diagnostics
    }

    fn resource_manager(&self) -> &Self::ResourceManager {
        &self.resource_manager
    }
}

pub struct FsmDispatcher<E, Ev, Tr, Ex>
where
    E: Send + Sync + PoisonErrorConverter,
    Ev: Send,
    Tr: Send,
    Ex: CoreExecutor<Event = Ev, TransitionResult = Tr>
        + ExecutorExt<Error = E, Event = Ev, TransitionResult = Tr>,
{
    executor: Arc<Ex>,
    event_queue: Mutex<VecDeque<Ev>>,
    _phantom_e: std::marker::PhantomData<E>,
}

impl<E, Ev, Tr, Ex> FsmDispatcher<E, Ev, Tr, Ex>
where
    E: Send + Sync + PoisonErrorConverter,
    Ev: Send,
    Tr: Send,
    Ex: CoreExecutor<Event = Ev, TransitionResult = Tr>
        + ExecutorExt<Error = E, Event = Ev, TransitionResult = Tr>,
{
    pub fn new(executor: Ex) -> Self {
        Self {
            executor: Arc::new(executor),
            event_queue: Mutex::new(VecDeque::new()),
            _phantom_e: std::marker::PhantomData,
        }
    }

    pub fn enqueue_event(&self, event: Ev) -> Result<(), E> {
        let mut queue = self
            .event_queue
            .lock()
            .map_err(|_| E::from_poison_error())?;
        queue.push_back(event);
        Ok(())
    }

    pub fn queue_size(&self) -> Result<usize, E> {
        Ok(self
            .event_queue
            .lock()
            .map_err(|_| E::from_poison_error())?
            .len())
    }

    pub fn clear_queue(&self) -> Result<(), E> {
        self.event_queue
            .lock()
            .map_err(|_| E::from_poison_error())?
            .clear();
        Ok(())
    }
}

impl<E, Ev, Tr, Ex> CoreDispatcher for FsmDispatcher<E, Ev, Tr, Ex>
where
    E: Send + Sync + From<EngineError> + PoisonErrorConverter,
    Ev: Send,
    Tr: Send,
    Ex: CoreExecutor<Event = Ev, TransitionResult = Tr>
        + ExecutorExt<Error = E, Event = Ev, TransitionResult = Tr>,
{
    type Error = E;
    type Event = Ev;
    type TransitionResult = Tr;
    type Executor = Ex;

    fn dispatch_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error> {
        // Direct dispatch without enqueuing
        self.executor.execute_transition_for_event(event)
    }

    fn process_queued_events(&self) -> Result<(), Self::Error> {
        let mut queue = self
            .event_queue
            .lock()
            .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        while let Some(ev) = queue.pop_front() {
            let _ = self.executor.execute_transition_for_event(ev)?;
        }
        Ok(())
    }
}

// Add a helper method on CoreExecutor to handle direct event-based transitions:
pub trait ExecutorExt {
    type Error;
    type Event;
    type TransitionResult;
    fn execute_transition_for_event(
        &self,
        event: Self::Event,
    ) -> Result<Self::TransitionResult, Self::Error>;
}

impl<B, D, M, R, T> ExecutorExt for FsmExecutor<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T, Context = FsmContext<StateHandle, D, R>> + Send + Sync + 'static,
    D: Diagnostics + Send + Sync + 'static,
    M: ModelDefinition<TransitionDefinition = T> + Send + Sync + 'static,
    R: ResourceManager + Send + Sync + 'static,
    T: TransitionDefinition + Clone + Send + Sync + 'static,
{
    type Error = EngineError;
    type Event = Event;
    type TransitionResult = TransitionResult;

    fn execute_transition_for_event(
        &self,
        event: Self::Event,
    ) -> Result<Self::TransitionResult, Self::Error> {
        let current = self.current_state();
        let transitions = (*self.model).available_transitions(&current.state_id, &event);
        if transitions.is_empty() {
            return Err(EngineError::InvalidTransition);
        }

        let chosen_transition = transitions[0].clone();
        self.execute_transition(&chosen_transition, &event)
    }
}

#[derive(Debug, Default)]
pub struct FsmLifecycleController<E: Send + Sync> {
    stopped: std::sync::Mutex<bool>,
    _phantom_e: std::marker::PhantomData<E>,
}

impl<E: Send + Sync> FsmLifecycleController<E> {
    pub fn new() -> Self {
        Self {
            stopped: std::sync::Mutex::new(false),
            _phantom_e: std::marker::PhantomData,
        }
    }
}

impl<E: Send + Sync + From<EngineError>> LifecycleController for FsmLifecycleController<E> {
    type Error = E;

    fn start(&self) -> Result<(), Self::Error> {
        let mut s = self
            .stopped
            .lock()
            .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        *s = false;
        Ok(())
    }

    fn stop(&self) -> Result<(), Self::Error> {
        let mut s = self
            .stopped
            .lock()
            .map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        *s = true;
        Ok(())
    }

    fn cleanup(&self) -> Result<(), Self::Error> {
        // Perform resource cleanup if needed
        Ok(())
    }
}
