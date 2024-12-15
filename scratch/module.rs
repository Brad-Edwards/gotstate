// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core interfaces for state machine implementations.
pub trait CoreEngine: Send + Sync {
    type Error;
    type StateHandle: Send + Sync;
    type Event: Send;
    type TransitionResult: Send;

    fn start(&self) -> Result<(), Self::Error>;
    fn stop(&self) -> Result<(), Self::Error>;
    fn process_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error>;
    fn current_state(&self) -> Self::StateHandle;
}

//// Core Submodules

pub trait CoreExecutor: Send + Sync {
    type Error;
    type StateHandle: Send + Sync;
    type Event: Send;
    type TransitionResult: Send;
    type TransitionDef: TransitionDefinition;
    type Behavior: Behavior<TransitionDef=Self::TransitionDef>;
    type ResourceManager: ResourceManager;
    type Diagnostics: Diagnostics;
    type Model: ModelDefinition;

    fn execute_transition(
        &self,
        transition: &Self::TransitionDef,
        event: &Self::Event
    ) -> Result<Self::TransitionResult, Self::Error>;

    fn current_state(&self) -> Self::StateHandle;

    fn context(&self) -> &dyn CoreContextProvider<
        StateHandle = Self::StateHandle,
        Diagnostics = Self::Diagnostics,
        ResourceManager = Self::ResourceManager
    >;
}

pub trait CoreContextProvider: Send + Sync {
    type StateHandle: Send + Sync;
    type Diagnostics: Diagnostics;
    type ResourceManager: ResourceManager;

    fn state_handle(&self) -> Self::StateHandle;
    fn diagnostics(&self) -> &Self::Diagnostics;
    fn resource_manager(&self) -> &Self::ResourceManager;
}

pub trait CoreDispatcher: Send + Sync {
    type Error;
    type Event: Send;
    type TransitionResult: Send;
    type Executor: CoreExecutor<Event=Self::Event, TransitionResult=Self::TransitionResult>;

    fn dispatch_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error>;
    fn process_queued_events(&self) -> Result<(), Self::Error>;
}

pub trait LifecycleController: Send + Sync {
    type Error;

    fn start(&self) -> Result<(), Self::Error>;
    fn stop(&self) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}

// Model Module

pub trait ModelDefinition: Send + Sync {
    type StateDefinition: StateDefinition;
    type EventDefinition: EventDefinition;
    type TransitionDefinition: TransitionDefinition<
        StateDef=Self::StateDefinition,
        EventDef=Self::EventDefinition
    >;

    fn root_state(&self) -> &Self::StateDefinition;
    fn states(&self) -> &[Self::StateDefinition];
    fn events(&self) -> &[Self::EventDefinition];
    fn transitions(&self) -> &[Self::TransitionDefinition];
}

pub trait StateDefinition: Send + Sync {
    fn name(&self) -> &str;
    fn children(&self) -> &[Self];
    fn is_composite(&self) -> bool;
}

pub trait EventDefinition: Send + Sync {
    fn name(&self) -> &str;
}

pub trait TransitionDefinition: Send + Sync {
    type StateDef: StateDefinition;
    type EventDef: EventDefinition;

    fn source_state(&self) -> &Self::StateDef;
    fn target_state(&self) -> &Self::StateDef;
    fn event(&self) -> &Self::EventDef;
    fn has_guard(&self) -> bool;
}

//// Model Module

pub trait ModelBuilderInternal: Send {
    type Error;
    type StateDef: StateDefinition;
    type EventDef: EventDefinition;
    type TransitionDef: TransitionDefinition<StateDef=Self::StateDef, EventDef=Self::EventDef>;

    fn define_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn define_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn define_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;
    fn finalize(self) -> Result<(Vec<Self::StateDef>, Vec<Self::EventDef>, Vec<Self::TransitionDef>), Self::Error>;
}

pub trait ImmutableModelStore: Send + Sync {
    type StateDef: StateDefinition;
    type EventDef: EventDefinition;
    type TransitionDef: TransitionDefinition<StateDef=Self::StateDef, EventDef=Self::EventDef>;

    fn root_state(&self) -> &Self::StateDef;
    fn states(&self) -> &[Self::StateDef];
    fn events(&self) -> &[Self::EventDef];
    fn transitions(&self) -> &[Self::TransitionDef];
    fn find_state_by_name(&self, name: &str) -> Option<&Self::StateDef>;
    fn find_event_by_name(&self, name: &str) -> Option<&Self::EventDef>;
}

pub trait HierarchyValidator: Send + Sync {
    type StateDef: StateDefinition;
    type Error;

    fn validate_hierarchy(&self, root_state: &Self::StateDef) -> Result<(), Self::Error>;
}

// Behavior Module

pub trait BehaviorContext: Send + Sync {
    type StateHandle: Send + Sync;
    type Diagnostics: Diagnostics;
    type ResourceHandle: Send + Sync;

    fn current_state(&self) -> Self::StateHandle;
    fn diagnostics(&self) -> &Self::Diagnostics;
    fn resource_handle(&self) -> &Self::ResourceHandle;
}

pub trait Behavior: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context: BehaviorContext;

    fn evaluate_guard(&self, transition: &Self::TransitionDef, ctx: &Self::Context) -> Result<bool, Self::Error>;
    fn execute_action(&self, transition: &Self::TransitionDef, ctx: &Self::Context) -> Result<(), Self::Error>;
    fn on_state_entry(&self, state: &<Self::TransitionDef as TransitionDefinition>::StateDef, ctx: &Self::Context) -> Result<(), Self::Error>;
    fn on_state_exit(&self, state: &<Self::TransitionDef as TransitionDefinition>::StateDef, ctx: &Self::Context) -> Result<(), Self::Error>;
}

//// Behavior Module

pub trait GuardEvaluator: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context: BehaviorContext;

    fn evaluate_guard(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context
    ) -> Result<bool, Self::Error>;
}

pub trait ActionExecutor: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context: BehaviorContext;

    fn execute_action(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context
    ) -> Result<(), Self::Error>;
}

pub trait StateLifecycleHandler: Send + Sync {
    type Error;
    type StateDef: StateDefinition;
    type Context: BehaviorContext;

    fn on_state_entry(
        &self,
        state: &Self::StateDef,
        ctx: &Self::Context
    ) -> Result<(), Self::Error>;

    fn on_state_exit(
        &self,
        state: &Self::StateDef,
        ctx: &Self::Context
    ) -> Result<(), Self::Error>;
}

// Builder Module

pub trait FSMBuilder: Send {
    type Error;
    type Model: ModelDefinition;
    type Validator: Validator<Model = Self::Model>;

    fn add_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;
    fn build(self) -> Result<Self::Model, Self::Error>;
    fn validate_and_build(self, validator: &Self::Validator) -> Result<Self::Model, Self::Error>;
}

//// Builder Submodules

pub trait BuilderContext: Send {
    type StateDefinition: StateDefinition;
    type EventDefinition: EventDefinition;
    type TransitionDefinition: TransitionDefinition<
        StateDef=Self::StateDefinition,
        EventDef=Self::EventDefinition
    >;
    type Error;

    fn add_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;

    fn finalize(self) -> Result<(
        Vec<Self::StateDefinition>,
        Vec<Self::EventDefinition>,
        Vec<Self::TransitionDefinition>
    ), Self::Error>;
}

// Concurrency Module

pub trait Concurrency: Send + Sync {
    type Event: Send;
    type LockHandle: Send;
    type Error;
    type Duration;

    fn enqueue_event(&self, event: Self::Event) -> Result<(), Self::Error>;
    fn dequeue_event(&self) -> Option<Self::Event>;
    fn acquire_lock(&self) -> Result<Self::LockHandle, Self::Error>;
    fn release_lock(&self, handle: Self::LockHandle);
    fn schedule_timeout(&self, event: Self::Event, delay: Self::Duration) -> Result<(), Self::Error>;
    fn cancel_timeout(&self, event: &Self::Event) -> Result<(), Self::Error>;
}

// Concurrency Submodules

pub trait EventQueue: Send + Sync {
    type Event: Send;
    type Error;

    fn enqueue(&self, event: Self::Event) -> Result<(), Self::Error>;
    fn dequeue(&self) -> Option<Self::Event>;
    fn is_empty(&self) -> bool;
}

pub trait LockManager: Send + Sync {
    type LockHandle: Send;
    type Error;

    fn acquire_lock(&self) -> Result<Self::LockHandle, Self::Error>;
    fn release_lock(&self, handle: Self::LockHandle);
}

pub trait TimeoutScheduler: Send + Sync {
    type Event: Send;
    type Duration;
    type Error;

    fn schedule_timeout(&self, event: Self::Event, delay: Self::Duration) -> Result<(), Self::Error>;
    fn cancel_timeout(&self, event: &Self::Event) -> Result<(), Self::Error>;
    fn check_expired(&self) -> Result<Vec<Self::Event>, Self::Error>;
}

// Resource Module

pub trait ResourceManager: Send + Sync {
    type ResourceHandle: Send + Sync;
    type ResourceType: Send + Sync;
    type ResourceConfig: Send + Sync;
    type Error;

    fn allocate_resource(&self, rtype: &Self::ResourceType, config: &Self::ResourceConfig) -> Result<Self::ResourceHandle, Self::Error>;
    fn release_resource(&self, handle: Self::ResourceHandle) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}

// Resource Submodules

pub trait ResourcePool: Send + Sync {
    type ResourceHandle: Send + Sync;
    type ResourceType: Send + Sync;
    type ResourceConfig: Send + Sync;
    type Error;

    fn allocate(&self, rtype: &Self::ResourceType, config: &Self::ResourceConfig) -> Result<Self::ResourceHandle, Self::Error>;
    fn release(&self, handle: Self::ResourceHandle) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}

pub trait ResourceTracker: Send + Sync {
    type ResourceHandle: Send + Sync;

    fn increment_ref(&self, handle: &Self::ResourceHandle);
    fn decrement_ref(&self, handle: &Self::ResourceHandle);
    fn ref_count(&self, handle: &Self::ResourceHandle) -> usize;
}

// Diagnostics Module

pub trait Diagnostics: Send + Sync {
    type ErrorInfo: Send + Sync;
    type LogLevel: Send + Sync;
    type DiagnosticData: Send + Sync;

    fn log(&self, message: &str, level: &Self::LogLevel);
    fn report_error(&self, error: &Self::ErrorInfo);
    fn get_diagnostic_data(&self) -> Self::DiagnosticData;
}

// Diagnostics Submodules

pub trait Logger: Send + Sync {
    type LogLevel: Send + Sync;
    fn log(&self, message: &str, level: &Self::LogLevel);
}

pub trait ErrorReporter: Send + Sync {
    type ErrorInfo: Send + Sync;
    fn report_error(&self, error: &Self::ErrorInfo);
}

pub trait DiagnosticCollector: Send + Sync {
    type DiagnosticData: Send + Sync;
    fn collect_data(&self) -> Self::DiagnosticData;
}

// Validation Module

pub trait Validator: Send + Sync {
    type Model: ModelDefinition;
    type ValidationResult;
    type Error;

    fn validate_model(&self, model: &Self::Model) -> Result<Self::ValidationResult, Self::Error>;
}

// Validation Submodules

pub trait StructuralValidator: Send + Sync {
    type Model: ModelDefinition;
    type Error;

    fn validate_model_structure(&self, model: &Self::Model) -> Result<(), Self::Error>;
}

pub trait RuntimeValidator: Send + Sync {
    type StateDef: StateDefinition;
    type TransitionDef: TransitionDefinition;
    type Error;

    fn validate_state_activation(&self, state: &Self::StateDef) -> Result<(), Self::Error>;
    fn validate_transition_consistency(&self, transition: &Self::TransitionDef) -> Result<(), Self::Error>;
}

pub trait RuleEngine: Send + Sync {
    type Model: ModelDefinition;
    type Error;

    fn apply_rules(&self, model: &Self::Model) -> Result<(), Self::Error>;
    fn register_rule(&mut self, rule_name: &str, rule_fn: &dyn Fn(&Self::Model) -> Result<(), Self::Error>);
}

pub trait RecoveryCoordinator: Send + Sync {
    type Model: ModelDefinition;
    type Error;

    fn recover_from_error(&self, model: &Self::Model, error: &Self::Error) -> Result<(), Self::Error>;
}


// API Modules

pub trait FsmApi: Send + Sync {
    type Engine: CoreEngine;
    type Model: ModelDefinition;
    type Behavior: Behavior<TransitionDef=<Self::Model as ModelDefinition>::TransitionDefinition>;
    type Concurrency: Concurrency<Event=<Self::Engine as CoreEngine>::Event>;
    type Diagnostics: Diagnostics;
    type ResourceManager: ResourceManager;
    type Validator: Validator<Model=Self::Model>;
    type Error;

    fn create_fsm(
        &self,
        model: Self::Model,
        concurrency: Self::Concurrency,
        behavior: Self::Behavior,
        diagnostics: Self::Diagnostics,
        resource_manager: Self::ResourceManager,
        validator: Self::Validator,
    ) -> Result<Self::Engine, Self::Error>;

    fn get_state_info(&self, engine: &Self::Engine) -> <Self::Engine as CoreEngine>::StateHandle;

    fn dispatch_event(
        &self,
        engine: &Self::Engine,
        event: <Self::Engine as CoreEngine>::Event
    ) -> Result<<Self::Engine as CoreEngine>::TransitionResult, Self::Error>;
}