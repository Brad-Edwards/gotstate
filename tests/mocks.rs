// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use gotstate::behavior::{Behavior, BehaviorContext};
use gotstate::builder::FSMBuilder;
use gotstate::concurrency::{Concurrency, EventQueue, LockManager, TimeoutScheduler};
use gotstate::core::{CoreContextProvider, CoreDispatcher, CoreEngine, CoreExecutor, Event};
use gotstate::diagnostics::{
    DiagnosticCollector, Diagnostics, ErrorReporter, Logger, TransitionStats,
};
use gotstate::fsm_api::FsmApi;
use gotstate::model::{
    EventDefinition, HierarchyValidator, ImmutableModelStore, ModelBuilderInternal,
    ModelDefinition, StateDefinition, TransitionDefinition,
};
use gotstate::resource::{ResourceManager, ResourcePool, ResourceTracker};
use gotstate::validator::Validator;

// A general-purpose MockError used across multiple tests
#[derive(Debug, Clone)]
pub struct MockError(pub String);

impl fmt::Display for MockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for MockError {}

// A general-purpose MockState. Adjust fields or add others as required by all tests.
// Some tests only have a `name` field, others have children, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct MockState {
    pub name: String,
    pub children: Vec<MockState>,
}

impl Default for MockState {
    fn default() -> Self {
        MockState {
            name: "default_state".to_string(),
            children: vec![],
        }
    }
}

impl StateDefinition for MockState {
    fn name(&self) -> &str {
        &self.name
    }

    fn children(&self) -> &[Self] {
        &self.children
    }

    fn is_composite(&self) -> bool {
        !self.children.is_empty()
    }
}

// A general-purpose MockEvent. Some tests only need a `name` field, others need `id` and `payload`.
// If some tests use only name, and others require `id` and `payload`, include them all and set defaults as needed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MockEvent {
    pub id: u64,
    pub name: Option<String>,
    pub payload: Option<String>,
}

impl EventDefinition for MockEvent {
    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("") // or handle None as you see fit
    }
}

impl fmt::Display for MockEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MockTransitionResult {
    pub from: MockState,
    pub to: MockState,
    pub success: bool,
}

// A general-purpose MockTransition. Some tests have `has_guard`, others do not.
// Include it for all tests.
#[derive(Debug, Clone, PartialEq)]
pub struct MockTransition {
    pub from: MockState,
    pub to: MockState,
    pub event: MockEvent,
    pub has_guard: bool,
}

impl MockTransition {
    pub fn new(from: MockState, to: MockState, event: MockEvent) -> Self {
        MockTransition {
            from,
            to,
            event,
            has_guard: false,
        }
    }
}

impl TransitionDefinition for MockTransition {
    type StateDef = MockState;
    type EventDef = MockEvent;

    fn source_state(&self) -> &Self::StateDef {
        &self.from
    }

    fn target_state(&self) -> &Self::StateDef {
        &self.to
    }

    fn event(&self) -> &Self::EventDef {
        &self.event
    }

    fn has_guard(&self) -> bool {
        self.has_guard
    }

    fn validate_transition(&self, _from: &str, _to: &str) -> bool {
        true // Default implementation always validates the transition
    }

    fn get_priority(&self) -> u32 {
        0 // Default priority of 0
    }
}

// If multiple tests require a builder or FSM builder, unify them similarly here. For example:
// (Adjust fields and logic as per your common usage across tests.)

#[derive(Debug, Default)]
pub struct MockBuilder {
    pub states: Vec<MockState>,
    pub events: Vec<MockEvent>,
    pub transitions: Vec<MockTransition>,
}

impl MockBuilder {
    pub fn add_state(&mut self, name: &str) -> Result<(), MockError> {
        if self.states.iter().any(|s| s.name == name) {
            return Err(MockError(format!("State {} already exists", name)));
        }
        self.states.push(MockState {
            name: name.to_string(),
            children: vec![],
        });
        Ok(())
    }

    pub fn add_event(&mut self, name: &str) -> Result<(), MockError> {
        if self.events.iter().any(|e| e.name.as_deref() == Some(name)) {
            return Err(MockError(format!("Event {} already exists", name)));
        }
        self.events.push(MockEvent {
            id: 0,
            name: Some(name.to_string()),
            payload: None,
        });
        Ok(())
    }

    pub fn add_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), MockError> {
        let from_state = self
            .states
            .iter()
            .find(|s| s.name == from)
            .ok_or_else(|| MockError(format!("Source state {} not found", from)))?
            .clone();

        let to_state = self
            .states
            .iter()
            .find(|s| s.name == to)
            .ok_or_else(|| MockError(format!("Target state {} not found", to)))?
            .clone();

        let event_def = self
            .events
            .iter()
            .find(|e| e.name.as_deref() == Some(event))
            .ok_or_else(|| MockError(format!("Event {} not found", event)))?
            .clone();

        self.transitions
            .push(MockTransition::new(from_state, to_state, event_def));
        Ok(())
    }

    pub fn finalize(
        self,
    ) -> Result<(Vec<MockState>, Vec<MockEvent>, Vec<MockTransition>), MockError> {
        Ok((self.states, self.events, self.transitions))
    }
}

pub struct MockValidator;

impl Validator for MockValidator {
    type Model = MockModel;
    type Error = MockError;
    type ValidationResult = ();

    fn validate_model(&self, model: &Self::Model) -> Result<Self::ValidationResult, Self::Error> {
        // Simple validation: ensure no empty states
        if model.states().is_empty() {
            return Err(MockError("Model must have at least one state".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct MockModel {
    pub root: MockState,
    pub states: Vec<MockState>,
    pub events: Vec<MockEvent>,
    pub transitions: Vec<MockTransition>,
}

impl ModelDefinition for MockModel {
    type StateDefinition = MockState;
    type EventDefinition = MockEvent;
    type TransitionDefinition = MockTransition;

    fn states(&self) -> &[Self::StateDefinition] {
        &self.states
    }

    fn events(&self) -> &[Self::EventDefinition] {
        &self.events
    }

    fn transitions(&self) -> &[Self::TransitionDefinition] {
        &self.transitions
    }

    fn root_state(&self) -> &Self::StateDefinition {
        &self.states[0]
    }

    fn available_transitions(&self, state_id: &str, event: &Event) -> Vec<MockTransition> {
        self.transitions
            .iter()
            .filter(|t| {
                t.from.name == state_id
                    && match &t.event.name {
                        Some(event_name) => event_name == &event.name.to_string(),
                        None => false,
                    }
            })
            .cloned()
            .collect()
    }

    fn resolve_target_state(&self, transition: &Self::TransitionDefinition) -> Option<String> {
        Some(transition.to.name.clone())
    }
}

#[derive(Default)]
pub struct MockFSMBuilder {
    pub states: Vec<MockState>,
    pub events: Vec<MockEvent>,
    pub transitions: Vec<MockTransition>,
}

impl FSMBuilder for MockFSMBuilder {
    type Error = MockError;
    type Model = MockModel;
    type Validator = MockValidator;

    fn add_state(&mut self, name: &str) -> Result<(), Self::Error> {
        if self.states.iter().any(|s| s.name == name) {
            return Err(MockError(format!("State {} already exists", name)));
        }
        self.states.push(MockState {
            name: name.to_string(),
            children: vec![],
        });
        Ok(())
    }

    fn add_event(&mut self, name: &str) -> Result<(), Self::Error> {
        if self.events.iter().any(|e| e.name.as_deref() == Some(name)) {
            return Err(MockError(format!("Event {} already exists", name)));
        }
        self.events.push(MockEvent {
            id: 0,
            name: Some(name.to_string()),
            payload: None,
        });
        Ok(())
    }

    fn add_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error> {
        let from_state = self
            .states
            .iter()
            .find(|s| s.name == from)
            .ok_or_else(|| MockError(format!("Source state {} not found", from)))?
            .clone();

        let to_state = self
            .states
            .iter()
            .find(|s| s.name == to)
            .ok_or_else(|| MockError(format!("Target state {} not found", to)))?
            .clone();

        let event = self
            .events
            .iter()
            .find(|e| e.name.as_deref() == Some(event))
            .ok_or_else(|| MockError(format!("Event {} not found", event)))?
            .clone();

        self.transitions.push(MockTransition {
            from: from_state,
            to: to_state,
            event,
            has_guard: false,
        });
        Ok(())
    }

    fn build(self) -> Result<Self::Model, Self::Error> {
        Ok(MockModel {
            root: self.states[0].clone(),
            states: self.states,
            events: self.events,
            transitions: self.transitions,
        })
    }

    fn validate_and_build(self, validator: &Self::Validator) -> Result<Self::Model, Self::Error> {
        let model = self.build()?;
        validator.validate_model(&model)?;
        Ok(model)
    }
}

#[derive(Debug, Default)]
pub struct MockEventQueue {
    pub events: Arc<Mutex<VecDeque<MockEvent>>>,
}

impl EventQueue for MockEventQueue {
    type Event = MockEvent;
    type Error = MockError;

    fn enqueue(&self, event: Self::Event) -> Result<(), Self::Error> {
        let mut queue = self
            .events
            .lock()
            .map_err(|_| MockError("Lock poisoned".to_string()))?;
        queue.push_back(event);
        Ok(())
    }

    fn dequeue(&self) -> Option<Self::Event> {
        self.events.lock().ok()?.pop_front()
    }

    fn is_empty(&self) -> bool {
        self.events.lock().map(|q| q.is_empty()).unwrap_or(true)
    }
}

#[derive(Clone)]
pub struct MockLockHandle {
    pub id: u64,
}

impl MockLockHandle {
    pub fn id(&self) -> u64 {
        self.id
    }
}

#[derive(Debug, Default)]
pub struct MockLockManager {
    pub active_locks: Arc<RwLock<HashMap<u64, bool>>>,
    pub next_lock_id: Arc<Mutex<u64>>,
}

impl LockManager for MockLockManager {
    type LockHandle = MockLockHandle;
    type Error = MockError;

    fn acquire_lock(&self) -> Result<Self::LockHandle, Self::Error> {
        let mut next_id = self
            .next_lock_id
            .lock()
            .map_err(|_| MockError("Lock poisoned".to_string()))?;
        let id = *next_id;
        *next_id += 1;

        let mut locks = self
            .active_locks
            .write()
            .map_err(|_| MockError("Lock poisoned".to_string()))?;
        locks.insert(id, true);
        Ok(MockLockHandle { id: id as u64 })
    }

    fn release_lock(&self, handle: Self::LockHandle) {
        if let Ok(mut locks) = self.active_locks.write() {
            locks.remove(&(handle.id as u64));
        }
    }
}

#[derive(Debug, Default)]
pub struct MockTimeoutScheduler {
    pub timeouts: Arc<RwLock<HashMap<MockEvent, (Instant, Duration)>>>,
}

impl TimeoutScheduler for MockTimeoutScheduler {
    type Event = MockEvent;
    type Duration = Duration;
    type Error = MockError;

    fn schedule_timeout(
        &self,
        event: Self::Event,
        delay: Self::Duration,
    ) -> Result<(), Self::Error> {
        let mut timeouts = self
            .timeouts
            .write()
            .map_err(|_| MockError("Lock poisoned".to_string()))?;
        timeouts.insert(event, (Instant::now(), delay));
        Ok(())
    }

    fn cancel_timeout(&self, event: &Self::Event) -> Result<(), Self::Error> {
        let mut timeouts = self
            .timeouts
            .write()
            .map_err(|_| MockError("Lock poisoned".to_string()))?;
        timeouts.remove(event);
        Ok(())
    }

    fn check_expired(&self) -> Result<Vec<Self::Event>, Self::Error> {
        let mut timeouts = self
            .timeouts
            .write()
            .map_err(|_| MockError("Lock poisoned".to_string()))?;
        let now = Instant::now();
        let expired: Vec<_> = timeouts
            .iter()
            .filter(|(_, (start, duration))| now.duration_since(*start) >= *duration)
            .map(|(event, _)| event.clone())
            .collect();

        for event in &expired {
            timeouts.remove(event);
        }
        Ok(expired)
    }
}

#[derive(Debug, Default)]
pub struct MockConcurrency {
    pub event_queue: Arc<MockEventQueue>,
    pub lock_manager: Arc<MockLockManager>,
    pub timeout_scheduler: Arc<MockTimeoutScheduler>,
}

impl Concurrency for MockConcurrency {
    type Event = MockEvent;
    type LockHandle = MockLockHandle;
    type Error = MockError;
    type Duration = Duration;

    fn enqueue_event(&self, event: Self::Event) -> Result<(), Self::Error> {
        self.event_queue.enqueue(event)
    }

    fn dequeue_event(&self) -> Option<Self::Event> {
        self.event_queue.dequeue()
    }

    fn acquire_lock(&self) -> Result<Self::LockHandle, Self::Error> {
        self.lock_manager.acquire_lock()
    }

    fn release_lock(&self, handle: Self::LockHandle) {
        self.lock_manager.release_lock(handle)
    }

    fn schedule_timeout(
        &self,
        event: Self::Event,
        delay: Self::Duration,
    ) -> Result<(), Self::Error> {
        self.timeout_scheduler.schedule_timeout(event, delay)
    }

    fn cancel_timeout(&self, event: &Self::Event) -> Result<(), Self::Error> {
        self.timeout_scheduler.cancel_timeout(event)
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum MockLogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Default)]
pub struct MockLogger {
    log_entries: Arc<Mutex<Vec<(String, MockLogLevel)>>>,
    log_counts: Arc<Mutex<HashMap<MockLogLevel, u32>>>,
}

impl Logger for MockLogger {
    type LogLevel = MockLogLevel;

    fn log(&self, message: &str, level: &Self::LogLevel) {
        let mut entries = self.log_entries.lock().unwrap();
        entries.push((message.to_string(), level.clone()));

        let mut counts = self.log_counts.lock().unwrap();
        *counts.entry(level.clone()).or_insert(0) += 1;
    }
}

impl MockLogger {
    pub fn new() -> Self {
        MockLogger {
            log_entries: Arc::new(Mutex::new(Vec::new())),
            log_counts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_entries(&self) -> Vec<(String, MockLogLevel)> {
        self.log_entries.lock().unwrap().clone()
    }

    pub fn get_count_for_level(&self, level: &MockLogLevel) -> u32 {
        *self.log_counts.lock().unwrap().get(level).unwrap_or(&0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MockErrorInfo {
    pub error_code: u32,
    pub message: String,
    pub timestamp: SystemTime,
    pub severity: MockLogLevel,
}

pub struct MockEngine {
    pub current_state: Arc<Mutex<MockState>>,
    pub is_running: Arc<Mutex<bool>>,
}

impl CoreEngine for MockEngine {
    type Error = MockError;
    type StateHandle = MockState;
    type Event = MockEvent;
    type TransitionResult = MockTransitionResult;

    fn start(&self) -> Result<(), Self::Error> {
        let mut running = self.is_running.lock().unwrap();
        if *running {
            return Err(MockError("Engine already running".to_string()));
        }
        *running = true;
        Ok(())
    }

    fn stop(&self) -> Result<(), Self::Error> {
        let mut running = self.is_running.lock().unwrap();
        *running = false;
        Ok(())
    }

    fn process_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error> {
        let current = self.current_state.lock().unwrap();
        Ok(MockTransitionResult {
            from: current.clone(),
            to: MockState {
                name: format!("state_{}", event.id),
                children: vec![],
            },
            success: true,
        })
    }

    fn current_state(&self) -> Self::StateHandle {
        self.current_state.lock().unwrap().clone()
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MockDiagnosticData {
    pub total_errors: u32,
    pub total_logs: u32,
    pub error_counts_by_type: HashMap<u32, u32>,
    pub last_error: Option<MockErrorInfo>,
    pub log_counts_by_level: HashMap<MockLogLevel, u32>,
}

pub struct MockDiagnostics {
    pub data: Arc<Mutex<MockDiagnosticData>>,
    pub logger: Arc<MockLogger>,
    pub error_reporter: Arc<MockErrorReporter>,
    pub collector: MockDiagnosticCollector,
}

impl Diagnostics for MockDiagnostics {
    type ErrorInfo = MockErrorInfo;
    type LogLevel = MockLogLevel;
    type DiagnosticData = MockDiagnosticData;

    fn log(&self, message: &str, level: &Self::LogLevel) {
        // Update internal diagnostic data
        let mut data = self.data.lock().unwrap();
        data.total_logs += 1;
        *data.log_counts_by_level.entry(level.clone()).or_insert(0) += 1;

        // Also log via the embedded logger if needed
        self.logger.log(message, level);
    }

    fn report_error(&self, error: &Self::ErrorInfo) {
        let mut data = self.data.lock().unwrap();
        data.total_errors += 1;
        *data
            .error_counts_by_type
            .entry(error.error_code)
            .or_insert(0) += 1;
        data.last_error = Some(error.clone());

        // Also report via the embedded error_reporter
        self.error_reporter.report_error(error);
    }

    fn get_diagnostic_data(&self) -> Self::DiagnosticData {
        self.data.lock().unwrap().clone()
    }

    fn log_transition_attempt(&self, from_state: &str, to_state: &str, success: bool) {
        let message = format!(
            "Transition attempt from {} to {}: {}",
            from_state, to_state, success
        );
        self.log(&message, &MockLogLevel::Info);
    }

    fn get_transition_statistics(&self) -> HashMap<String, TransitionStats> {
        HashMap::new() // Or implement actual statistics tracking
    }
}

impl MockDiagnostics {
    pub fn new() -> Self {
        let data = Arc::new(Mutex::new(MockDiagnosticData::default()));
        let logger = Arc::new(MockLogger::new());
        let error_reporter = Arc::new(MockErrorReporter::new());
        let collector = MockDiagnosticCollector::new(logger.clone(), error_reporter.clone());

        MockDiagnostics {
            data,
            logger,
            error_reporter,
            collector,
        }
    }
}

pub struct MockResourceManager {
    pub allocated: Arc<Mutex<HashMap<u64, String>>>,
    pub next_handle: Arc<Mutex<u64>>,
}

impl ResourceManager for MockResourceManager {
    type ResourceHandle = u64;
    type ResourceType = String;
    type ResourceConfig = ();
    type Error = MockError;

    fn allocate_resource(
        &self,
        rtype: &Self::ResourceType,
        _: &Self::ResourceConfig,
    ) -> Result<Self::ResourceHandle, Self::Error> {
        let mut next_handle = self.next_handle.lock().unwrap();
        let handle = *next_handle;
        *next_handle += 1;

        let mut allocated = self.allocated.lock().unwrap();
        allocated.insert(handle, rtype.clone());
        Ok(handle)
    }

    fn release_resource(&self, handle: Self::ResourceHandle) -> Result<(), Self::Error> {
        let mut allocated = self.allocated.lock().unwrap();
        allocated
            .remove(&handle)
            .ok_or_else(|| MockError("Invalid resource handle".to_string()))?;
        Ok(())
    }

    fn cleanup(&self) -> Result<(), Self::Error> {
        let mut allocated = self.allocated.lock().unwrap();
        allocated.clear();
        Ok(())
    }
}

#[derive(Default)]
pub struct MockContext {
    pub state: Option<MockState>,
    pub diagnostics: Option<MockDiagnostics>,
    pub resource_handle: Option<u64>,
}

impl gotstate::behavior::BehaviorContext for MockContext {
    type StateHandle = MockState;
    type Diagnostics = MockDiagnostics;
    type ResourceHandle = u64;

    fn current_state(&self) -> Self::StateHandle {
        self.state.as_ref().unwrap().clone()
    }

    fn diagnostics(&self) -> &Self::Diagnostics {
        self.diagnostics.as_ref().unwrap()
    }

    fn resource_handle(&self) -> &Self::ResourceHandle {
        self.resource_handle.as_ref().unwrap()
    }
}

pub struct MockContextProvider {
    pub state: MockState,
    pub diagnostics: Arc<MockDiagnostics>,
    pub resource_manager: Arc<MockResourceManager>,
}

impl CoreContextProvider for MockContextProvider {
    type StateHandle = MockState;
    type Diagnostics = MockDiagnostics;
    type ResourceManager = MockResourceManager;

    fn state_handle(&self) -> Self::StateHandle {
        self.state.clone()
    }

    fn diagnostics(&self) -> &Self::Diagnostics {
        &self.diagnostics
    }

    fn resource_manager(&self) -> &Self::ResourceManager {
        &self.resource_manager
    }

    fn context_version(&self) -> u64 {
        1 // Return a version number for the context
    }

    fn is_valid(&self) -> bool {
        true // Return whether the context is valid
    }
}

impl BehaviorContext for MockContextProvider {
    type StateHandle = MockState;
    type Diagnostics = MockDiagnostics;
    type ResourceHandle = u64;

    fn current_state(&self) -> Self::StateHandle {
        self.state.clone()
    }

    fn diagnostics(&self) -> &Self::Diagnostics {
        &self.diagnostics
    }

    fn resource_handle(&self) -> &Self::ResourceHandle {
        static DEFAULT_HANDLE: u64 = 0;
        &DEFAULT_HANDLE
    }
}

pub struct MockExecutor {
    pub context: Arc<MockContextProvider>,
    pub behavior: Arc<MockBehavior>,
}

impl CoreExecutor for MockExecutor {
    type Error = MockError;
    type StateHandle = MockState;
    type Event = MockEvent;
    type TransitionResult = MockTransitionResult;
    type TransitionDef = MockTransition;
    type Behavior = MockBehavior;
    type ResourceManager = MockResourceManager;
    type Diagnostics = MockDiagnostics;
    type Model = MockModel;

    fn execute_transition(
        &self,
        transition: &Self::TransitionDef,
        _event: &Self::Event,
    ) -> Result<Self::TransitionResult, Self::Error> {
        if self.behavior.evaluate_guard(transition, &self.context)? {
            self.behavior.execute_action(transition, &self.context)?;
            <MockContextProvider as BehaviorContext>::diagnostics(&*self.context)
                .log_transition_attempt(&transition.from.name, &transition.to.name, true);
            Ok(MockTransitionResult {
                from: transition.from.clone(),
                to: transition.to.clone(),
                success: true,
            })
        } else {
            <MockContextProvider as BehaviorContext>::diagnostics(&*self.context)
                .log_transition_attempt(&transition.from.name, &transition.to.name, false);
            Ok(MockTransitionResult {
                from: transition.from.clone(),
                to: transition.from.clone(),
                success: false,
            })
        }
    }

    fn current_state(&self) -> Self::StateHandle {
        self.context.state_handle()
    }

    fn context(
        &self,
    ) -> &dyn CoreContextProvider<
        StateHandle = Self::StateHandle,
        Diagnostics = Self::Diagnostics,
        ResourceManager = Self::ResourceManager,
    > {
        &*self.context
    }
}

pub struct MockBehavior {
    pub action_results: Arc<Mutex<Vec<MockTransition>>>,
    pub guard_results: Arc<Mutex<HashMap<String, bool>>>,
}

impl Behavior for MockBehavior {
    type TransitionDef = MockTransition;
    type Error = MockError;
    type Context = MockContextProvider;

    fn execute_actions<E, H, T, C>(
        &self,
        transition: &Self::TransitionDef,
        event: &Event,
        resources: &dyn ResourceManager<
            Error = E,
            ResourceHandle = H,
            ResourceType = T,
            ResourceConfig = C,
        >,
    ) -> Result<(), Self::Error> {
        let mut results = self.action_results.lock().unwrap();
        results.push(transition.clone());
        Ok(())
    }

    fn execute_action(
        &self,
        transition: &Self::TransitionDef,
        context: &Self::Context,
    ) -> Result<(), Self::Error> {
        let mut results = self.action_results.lock().unwrap();
        results.push(transition.clone());
        Ok(())
    }

    fn evaluate_guard(
        &self,
        transition: &Self::TransitionDef,
        _context: &Self::Context,
    ) -> Result<bool, Self::Error> {
        let guards = self.guard_results.lock().unwrap();
        Ok(*guards.get(&transition.from.name).unwrap_or(&true))
    }

    fn on_state_entry(
        &self,
        _state: &MockState,
        _context: &Self::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn on_state_exit(
        &self,
        _state: &MockState,
        _context: &Self::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl MockBehavior {
    pub fn new() -> Self {
        MockBehavior {
            action_results: Arc::new(Mutex::new(Vec::new())),
            guard_results: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

pub struct MockBehaviorContext<St, D, R> {
    pub current_state_val: St,
    pub diagnostics_val: Arc<D>,
    pub resource_handle_val: R,
}

impl<St, D, R> MockBehaviorContext<St, D, R>
where
    St: StateDefinition + Send + Sync + Clone + 'static,
    D: Diagnostics + Send + Sync + 'static,
    R: Send + Sync + Default + 'static,
{
    /// Create a new `MockBehaviorContext` with the given state, diagnostics, and resource handle.
    pub fn new(current_state: St, diagnostics: Arc<D>, resource_handle: R) -> Self {
        Self {
            current_state_val: current_state,
            diagnostics_val: diagnostics,
            resource_handle_val: resource_handle,
        }
    }

    /// Create a `MockBehaviorContext` with defaults for convenience in tests.
    /// - State: "test_state"
    /// - Diagnostics: A new empty diagnostics instance
    /// - Resource Handle: 42
    pub fn default_with_test_state<DefaultD, DefaultR>() -> Self
    where
        DefaultD: Diagnostics + Default + Send + Sync + 'static,
        DefaultR: Default + Send + Sync + 'static,
        St: Default,
        D: Default,
    {
        Self {
            current_state_val: Default::default(),
            diagnostics_val: Arc::new(D::default()),
            resource_handle_val: R::default(),
        }
    }
}

impl<St, D, R> BehaviorContext for MockBehaviorContext<St, D, R>
where
    St: StateDefinition + Send + Sync + Clone + 'static,
    D: Diagnostics + Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    type StateHandle = St;
    type Diagnostics = D;
    type ResourceHandle = R;

    fn current_state(&self) -> Self::StateHandle {
        self.current_state_val.clone()
    }

    fn diagnostics(&self) -> &Self::Diagnostics {
        &self.diagnostics_val
    }

    fn resource_handle(&self) -> &Self::ResourceHandle {
        &self.resource_handle_val
    }
}

pub struct MockStateLifecycleHandler {
    entries: Vec<MockState>,
    exits: Vec<MockState>,
}

impl MockStateLifecycleHandler {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            exits: Vec::new(),
        }
    }

    pub fn on_state_entry(
        &mut self,
        state: &MockState,
        ctx: &MockContext,
    ) -> Result<(), MockError> {
        self.entries.push(state.clone());
        Ok(())
    }

    pub fn on_state_exit(&mut self, state: &MockState, ctx: &MockContext) -> Result<(), MockError> {
        self.exits.push(state.clone());
        Ok(())
    }

    pub fn get_state_entries(&self) -> &Vec<MockState> {
        &self.entries
    }

    pub fn get_state_exits(&self) -> &Vec<MockState> {
        &self.exits
    }
}

pub struct MockGuardEvaluator;

impl MockGuardEvaluator {
    pub fn new() -> Self {
        MockGuardEvaluator
    }

    pub fn evaluate_guard(
        &self,
        _transition: &MockTransition,
        _ctx: &MockBehaviorContext<MockState, MockDiagnostics, u64>,
    ) -> Result<bool, ()> {
        Ok(true)
    }
}

pub struct MockActionExecutor {
    executed_actions: Vec<MockEvent>,
}

impl MockActionExecutor {
    pub fn new() -> Self {
        MockActionExecutor {
            executed_actions: Vec::new(),
        }
    }

    pub fn execute_action(&mut self, event: MockEvent) {
        self.executed_actions.push(event);
    }

    pub fn get_executed_actions(&self) -> &Vec<MockEvent> {
        &self.executed_actions
    }
}

pub struct MockDispatcher {
    pub executor: Arc<MockExecutor>,
    pub queued_events: Arc<Mutex<VecDeque<MockEvent>>>,
}

impl CoreDispatcher for MockDispatcher {
    type Error = MockError;
    type Event = MockEvent;
    type TransitionResult = MockTransitionResult;
    type Executor = MockExecutor;

    fn dispatch_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error> {
        let transition = MockTransition {
            from: self.executor.current_state(),
            to: MockState {
                name: format!("state_{}", event.id),
                children: vec![],
            },
            event: event.clone(),
            has_guard: false,
        };
        self.executor.execute_transition(&transition, &event)
    }

    fn process_queued_events(&self) -> Result<(), Self::Error> {
        while let Some(event) = self.queued_events.lock().unwrap().pop_front() {
            self.dispatch_event(event)?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct MockErrorReporter {
    pub reported_errors: Arc<Mutex<Vec<MockErrorInfo>>>,
    pub error_counts: Arc<Mutex<HashMap<u32, u32>>>,
}

impl ErrorReporter for MockErrorReporter {
    type ErrorInfo = MockErrorInfo;

    fn report_error(&self, error: &Self::ErrorInfo) {
        let mut errors = self.reported_errors.lock().unwrap();
        errors.push(error.clone());

        let mut counts = self.error_counts.lock().unwrap();
        *counts.entry(error.error_code).or_insert(0) += 1;
    }
}

impl MockErrorReporter {
    pub fn new() -> Self {
        MockErrorReporter {
            reported_errors: Arc::new(Mutex::new(Vec::new())),
            error_counts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_reported_errors(&self) -> Vec<MockErrorInfo> {
        self.reported_errors.lock().unwrap().clone()
    }

    pub fn get_error_count(&self, error_code: u32) -> u32 {
        *self
            .error_counts
            .lock()
            .unwrap()
            .get(&error_code)
            .unwrap_or(&0)
    }
}

#[derive(Default)]
pub struct MockDiagnosticCollector {
    pub logger: Arc<MockLogger>,
    pub error_reporter: Arc<MockErrorReporter>,
}

impl DiagnosticCollector for MockDiagnosticCollector {
    type DiagnosticData = MockDiagnosticData;

    fn collect_data(&self) -> Self::DiagnosticData {
        let mut data = MockDiagnosticData::default();

        // Collect log statistics
        let log_entries = self.logger.get_entries();
        data.total_logs = log_entries.len() as u32;

        for (_, level) in log_entries {
            *data.log_counts_by_level.entry(level).or_insert(0) += 1;
        }

        // Collect error statistics
        let reported_errors = self.error_reporter.get_reported_errors();
        data.total_errors = reported_errors.len() as u32;

        if let Some(last_error) = reported_errors.last() {
            data.last_error = Some(last_error.clone());
        }

        for error in reported_errors {
            *data
                .error_counts_by_type
                .entry(error.error_code)
                .or_insert(0) += 1;
        }

        data
    }
}

impl MockDiagnosticCollector {
    pub fn new(logger: Arc<MockLogger>, error_reporter: Arc<MockErrorReporter>) -> Self {
        MockDiagnosticCollector {
            logger,
            error_reporter,
        }
    }
}

pub struct MockFsmApi;

impl FsmApi for MockFsmApi {
    type Engine = MockEngine;
    type Model = MockModel;
    type Behavior = MockBehavior;
    type Concurrency = MockConcurrency;
    type Diagnostics = MockDiagnostics;
    type ResourceManager = MockResourceManager;
    type Validator = MockValidator;
    type Error = MockError;

    fn create_fsm(
        &self,
        model: Self::Model,
        _concurrency: Self::Concurrency,
        _behavior: Self::Behavior,
        _diagnostics: Self::Diagnostics,
        _resource_manager: Self::ResourceManager,
        validator: Self::Validator,
    ) -> Result<Self::Engine, Self::Error> {
        validator.validate_model(&model)?;
        let root = model.root_state().clone();
        Ok(MockEngine {
            current_state: Arc::new(Mutex::new(root)),
            is_running: Arc::new(Mutex::new(false)),
        })
    }

    fn get_state_info(&self, engine: &Self::Engine) -> <Self::Engine as CoreEngine>::StateHandle {
        engine.current_state()
    }

    fn dispatch_event(
        &self,
        engine: &Self::Engine,
        event: <Self::Engine as CoreEngine>::Event,
    ) -> Result<<Self::Engine as CoreEngine>::TransitionResult, Self::Error> {
        engine.process_event(event)
    }
}

pub struct MockModelBuilder {
    pub states: Vec<MockState>,
    pub events: Vec<MockEvent>,
    pub transitions: Vec<MockTransition>,
}

impl MockModelBuilder {
    fn new() -> Self {
        MockModelBuilder {
            states: Vec::new(),
            events: Vec::new(),
            transitions: Vec::new(),
        }
    }
}

impl ModelBuilderInternal for MockModelBuilder {
    type Error = MockError;
    type StateDef = MockState;
    type EventDef = MockEvent;
    type TransitionDef = MockTransition;

    fn define_state(&mut self, name: &str) -> Result<(), Self::Error> {
        if self.states.iter().any(|s| s.name == name) {
            return Err(MockError(format!("State {} already exists", name)));
        }
        self.states.push(MockState {
            name: name.to_string(),
            children: Vec::new(),
        });
        Ok(())
    }

    fn define_event(&mut self, name: &str) -> Result<(), Self::Error> {
        if self.events.iter().any(|e| e.name == Some(name.to_string())) {
            return Err(MockError(format!("Event {} already exists", name)));
        }
        self.events.push(MockEvent {
            name: Some(name.to_string()),
            id: 0,
            payload: None,
        });
        Ok(())
    }

    fn define_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error> {
        let from_state = self
            .states
            .iter()
            .find(|s| s.name == from)
            .ok_or_else(|| MockError(format!("Source state {} not found", from)))?
            .clone();

        let to_state = self
            .states
            .iter()
            .find(|s| s.name == to)
            .ok_or_else(|| MockError(format!("Target state {} not found", to)))?
            .clone();

        let event_def = self
            .events
            .iter()
            .find(|e| e.name == Some(event.to_string()))
            .ok_or_else(|| MockError(format!("Event {} not found", event)))?
            .clone();

        self.transitions.push(MockTransition {
            from: from_state,
            to: to_state,
            event: event_def,
            has_guard: false,
        });
        Ok(())
    }

    fn finalize(
        self,
    ) -> Result<
        (
            Vec<Self::StateDef>,
            Vec<Self::EventDef>,
            Vec<Self::TransitionDef>,
        ),
        Self::Error,
    > {
        if self.states.is_empty() {
            return Err(MockError("No states defined".to_string()));
        }
        Ok((self.states, self.events, self.transitions))
    }
}

pub struct MockModelStore {
    pub root: MockState,
    pub states: Vec<MockState>,
    pub events: Vec<MockEvent>,
    pub transitions: Vec<MockTransition>,
}

impl ImmutableModelStore for MockModelStore {
    type StateDef = MockState;
    type EventDef = MockEvent;
    type TransitionDef = MockTransition;

    fn root_state(&self) -> &Self::StateDef {
        &self.root
    }

    fn states(&self) -> &[Self::StateDef] {
        &self.states
    }

    fn events(&self) -> &[Self::EventDef] {
        &self.events
    }

    fn transitions(&self) -> &[Self::TransitionDef] {
        &self.transitions
    }

    fn find_state_by_name(&self, name: &str) -> Option<&Self::StateDef> {
        self.states.iter().find(|s| s.name == name)
    }

    fn find_event_by_name(&self, name: &str) -> Option<&Self::EventDef> {
        self.events
            .iter()
            .find(|e| e.name.as_ref() == Some(&name.to_string()))
    }
}

pub struct MockHierarchyValidator;

impl HierarchyValidator for MockHierarchyValidator {
    type StateDef = MockState;
    type Error = MockError;

    fn validate_hierarchy(&self, root_state: &Self::StateDef) -> Result<(), Self::Error> {
        // Simple rule: no state named "invalid"
        if root_state.name == "invalid" {
            return Err(MockError("Invalid hierarchy detected".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MockResourceHandle(pub u64);

#[derive(Debug, Clone)]
pub struct MockResourceConfig {
    pub timeout_ms: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone)]
pub struct MockResourceType {
    pub name: String,
    pub id: u64,
}

pub struct MockResourcePool {
    pub allocated: Arc<Mutex<HashMap<MockResourceHandle, MockResourceType>>>,
    pub next_handle: Arc<Mutex<u64>>,
}

impl ResourcePool for MockResourcePool {
    type ResourceHandle = MockResourceHandle;
    type ResourceType = MockResourceType;
    type ResourceConfig = MockResourceConfig;
    type Error = MockError;

    fn allocate(
        &self,
        rtype: &Self::ResourceType,
        _config: &Self::ResourceConfig,
    ) -> Result<Self::ResourceHandle, Self::Error> {
        let mut next_handle = self.next_handle.lock().unwrap();
        let handle = MockResourceHandle(*next_handle);
        *next_handle += 1;

        let mut allocated = self.allocated.lock().unwrap();
        allocated.insert(handle.clone(), rtype.clone());
        Ok(handle)
    }

    fn release(&self, handle: Self::ResourceHandle) -> Result<(), Self::Error> {
        let mut allocated = self.allocated.lock().unwrap();
        allocated
            .remove(&handle)
            .ok_or_else(|| MockError("Attempted to release non-existent handle".to_string()))?;
        Ok(())
    }

    fn cleanup(&self) -> Result<(), Self::Error> {
        let mut allocated = self.allocated.lock().unwrap();
        allocated.clear();
        Ok(())
    }
}

pub struct MockResourceTracker {
    pub ref_counts: Arc<Mutex<HashMap<MockResourceHandle, usize>>>,
}

impl ResourceTracker for MockResourceTracker {
    type ResourceHandle = MockResourceHandle;

    fn increment_ref(&self, handle: &Self::ResourceHandle) {
        let mut counts = self.ref_counts.lock().unwrap();
        *counts.entry(handle.clone()).or_insert(0) += 1;
    }

    fn decrement_ref(&self, handle: &Self::ResourceHandle) {
        let mut counts = self.ref_counts.lock().unwrap();
        if let Some(count) = counts.get_mut(handle) {
            if *count > 0 {
                *count -= 1;
            }
            if *count == 0 {
                counts.remove(handle);
            }
        }
    }

    fn ref_count(&self, handle: &Self::ResourceHandle) -> usize {
        let counts = self.ref_counts.lock().unwrap();
        *counts.get(handle).unwrap_or(&0)
    }
}

// Add these mocks to mocks.rs

pub struct MockStructuralValidator {
    pub should_fail: bool,
}

impl Default for MockStructuralValidator {
    fn default() -> Self {
        MockStructuralValidator { should_fail: false }
    }
}

impl gotstate::validator::StructuralValidator for MockStructuralValidator {
    type Model = MockModel;
    type Error = MockError;

    fn validate_model_structure(&self, model: &Self::Model) -> Result<(), Self::Error> {
        if self.should_fail {
            Err(MockError("Structural validation failed".to_string()))
        } else if model.states.is_empty() {
            Err(MockError("No states defined".to_string()))
        } else {
            Ok(())
        }
    }
}

pub struct MockRuntimeValidator {
    pub invalid_states: Vec<String>,
    pub invalid_transitions: Vec<String>,
    pub inconsistent_transition: bool,
}

impl Default for MockRuntimeValidator {
    fn default() -> Self {
        MockRuntimeValidator {
            invalid_states: vec![],
            invalid_transitions: vec![],
            inconsistent_transition: false,
        }
    }
}

impl gotstate::validator::RuntimeValidator for MockRuntimeValidator {
    type StateDef = MockState;
    type TransitionDef = MockTransition;
    type Error = MockError;

    fn validate_state_activation(&self, state: &Self::StateDef) -> Result<(), Self::Error> {
        if self.invalid_states.contains(&state.name) {
            Err(MockError(format!(
                "State {} cannot be activated",
                state.name
            )))
        } else {
            Ok(())
        }
    }

    fn validate_transition_consistency(
        &self,
        transition: &Self::TransitionDef,
    ) -> Result<(), Self::Error> {
        if self.invalid_transitions.contains(&transition.from.name) {
            Err(MockError(format!(
                "Transition from {} is invalid",
                transition.from.name
            )))
        } else if self.inconsistent_transition {
            Err(MockError("Transition is inconsistent".to_string()))
        } else {
            Ok(())
        }
    }
}

pub struct MockRuleEngine {
    pub fail_rules: bool,
}

impl Default for MockRuleEngine {
    fn default() -> Self {
        MockRuleEngine { fail_rules: false }
    }
}

impl gotstate::validator::RuleEngine for MockRuleEngine {
    type Model = MockModel;
    type Error = MockError;

    fn apply_rules(&self, _model: &Self::Model) -> Result<(), Self::Error> {
        if self.fail_rules {
            Err(MockError("Rule engine failed a rule".to_string()))
        } else {
            Ok(())
        }
    }

    fn register_rule(
        &mut self,
        _rule_name: &str,
        _rule_fn: Box<dyn Fn(&Self::Model) -> Result<(), Self::Error> + Send + Sync + 'static>,
    ) {
        // For simplicity, do nothing here.
    }
}

pub struct MockRecoveryCoordinator {
    pub can_recover: bool,
}

impl Default for MockRecoveryCoordinator {
    fn default() -> Self {
        MockRecoveryCoordinator { can_recover: true }
    }
}

impl gotstate::validator::RecoveryCoordinator for MockRecoveryCoordinator {
    type Model = MockModel;
    type Error = MockError;

    fn recover_from_error(
        &self,
        _model: &Self::Model,
        error: &Self::Error,
    ) -> Result<(), Self::Error> {
        if self.can_recover {
            Ok(())
        } else {
            Err(MockError(format!("Cannot recover from: {}", error)))
        }
    }
}
