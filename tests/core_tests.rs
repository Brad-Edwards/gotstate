// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use gotstate::core::{CoreContextProvider, CoreDispatcher, CoreEngine, CoreExecutor};
use gotstate::diagnostics::Diagnostics;
use gotstate::resource::ResourceManager;

mod mocks;
use mocks::{
    MockBehavior, MockContextProvider, MockDiagnostics, MockDispatcher, MockEngine, MockEvent,
    MockExecutor, MockLogLevel, MockResourceManager, MockState, MockTransition,
};

fn create_test_state(name: &str) -> MockState {
    MockState {
        name: name.to_string(),
        children: vec![],
    }
}

fn create_test_event(id: u64, name: &str) -> MockEvent {
    MockEvent {
        id,
        name: Some(name.to_string()),
        payload: None,
    }
}

fn create_test_components() -> (
    Arc<MockEngine>,
    Arc<MockExecutor>,
    Arc<MockDispatcher>,
    Arc<MockContextProvider>,
) {
    let initial_state = create_test_state("initial");

    let diagnostics = Arc::new(MockDiagnostics::new());

    let resource_manager = Arc::new(MockResourceManager {
        allocated: Arc::new(Mutex::new(HashMap::new())),
        next_handle: Arc::new(Mutex::new(0)),
    });

    let context_provider = Arc::new(MockContextProvider {
        state: initial_state.clone(),
        diagnostics: diagnostics.clone(),
        resource_manager: resource_manager.clone(),
    });

    let behavior = Arc::new(MockBehavior {
        action_results: Arc::new(Mutex::new(Vec::new())),
        guard_results: Arc::new(Mutex::new(HashMap::new())),
    });

    let executor = Arc::new(MockExecutor {
        context: context_provider.clone(),
        behavior: behavior.clone(),
    });

    let engine = Arc::new(MockEngine {
        current_state: Arc::new(Mutex::new(initial_state)),
        is_running: Arc::new(Mutex::new(false)),
    });

    let dispatcher = Arc::new(MockDispatcher {
        executor: executor.clone(),
        queued_events: Arc::new(Mutex::new(VecDeque::new())),
    });

    (engine, executor, dispatcher, context_provider)
}

// CoreEngine Tests
#[test]
fn test_engine_lifecycle() {
    let (engine, _, _, _) = create_test_components();

    // Test initial state
    assert!(!*engine.is_running.lock().unwrap());

    // Test start
    assert!(engine.start().is_ok());
    assert!(*engine.is_running.lock().unwrap());

    // Test double start
    assert!(engine.start().is_err());

    // Test stop
    assert!(engine.stop().is_ok());
    assert!(!*engine.is_running.lock().unwrap());
}

#[test]
fn test_engine_event_processing() {
    let (engine, _, _, _) = create_test_components();

    let event = create_test_event(1, "test_event");
    let result = engine.process_event(event.clone()).unwrap();

    assert_eq!(result.from, engine.current_state());
    assert_eq!(result.to.name, "state_1");
    assert!(result.success);
}

#[test]
fn test_engine_concurrent_access() {
    use std::thread;

    let (engine, _, _, _) = create_test_components();
    let mut handles = vec![];

    for i in 0..10 {
        let engine = engine.clone();
        let handle = thread::spawn(move || {
            let event = create_test_event(i, &format!("event_{}", i));
            engine.process_event(event).unwrap()
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.success);
    }
}

// CoreExecutor Tests
#[test]
fn test_executor_transition_execution() {
    let (_, executor, _, _) = create_test_components();

    let event = create_test_event(1, "test_event");
    let from_state = create_test_state("state1");
    let to_state = create_test_state("state2");

    let transition = MockTransition {
        from: from_state.clone(),
        to: to_state.clone(),
        event: event.clone(),
        has_guard: false,
    };

    let result = executor.execute_transition(&transition, &event).unwrap();
    assert!(result.success);
    assert_eq!(result.from, from_state);
    assert_eq!(result.to, to_state);
}

#[test]
fn test_executor_guard_evaluation() {
    let (_, executor, _, _) = create_test_components();

    // Set up a guard that will return false
    let mut guard_results = executor.behavior.guard_results.lock().unwrap();
    guard_results.insert("state1".to_string(), false);
    drop(guard_results);

    let event = create_test_event(1, "test_event");
    let from_state = create_test_state("state1");
    let to_state = create_test_state("state2");

    let transition = MockTransition {
        from: from_state.clone(),
        to: to_state,
        event: event.clone(),
        has_guard: true,
    };

    let result = executor.execute_transition(&transition, &event).unwrap();
    assert!(!result.success);
    assert_eq!(result.from, from_state);
    assert_eq!(result.to, from_state);
}

#[test]
fn test_executor_action_tracking() {
    let (_, executor, _, _) = create_test_components();

    let event = create_test_event(1, "test_event");
    let transition = MockTransition {
        from: create_test_state("state1"),
        to: create_test_state("state2"),
        event: event.clone(),
        has_guard: false,
    };

    executor.execute_transition(&transition, &event).unwrap();

    let action_results = executor.behavior.action_results.lock().unwrap();
    assert_eq!(action_results.len(), 1);
    assert_eq!(action_results[0], transition);
}

#[test]
fn test_context_provider_state_handle() {
    let (_, _, _, provider) = create_test_components();

    let state = provider.state_handle();
    assert_eq!(state.name, "initial");
}

#[test]
fn test_context_provider_diagnostics() {
    let (_, _, _, provider) = create_test_components();

    CoreContextProvider::diagnostics(&*provider).log("test message", &MockLogLevel::Info);
    let data = CoreContextProvider::diagnostics(provider.as_ref()).get_diagnostic_data();
    assert_eq!(data.total_logs, 1);
}

#[test]
fn test_context_provider_resource_manager() {
    let (_, _, _, provider) = create_test_components();

    let result = provider
        .resource_manager()
        .allocate_resource(&"test_resource".to_string(), &())
        .unwrap();
    assert_eq!(result, 0);
}

#[test]
fn test_dispatcher_event_handling() {
    let (_, _, dispatcher, _) = create_test_components();

    let event = create_test_event(1, "test_event");
    let result = dispatcher.dispatch_event(event).unwrap();
    assert!(result.success);
}

#[test]
fn test_dispatcher_queue_processing() {
    let (_, _, dispatcher, _) = create_test_components();

    // Queue some events
    let mut queued_events = dispatcher.queued_events.lock().unwrap();
    queued_events.push_back(create_test_event(1, "event1"));
    queued_events.push_back(create_test_event(2, "event2"));
    drop(queued_events);

    assert!(dispatcher.process_queued_events().is_ok());
    assert!(dispatcher.queued_events.lock().unwrap().is_empty());
}

#[test]
fn test_full_state_transition_flow() {
    let (engine, executor, dispatcher, _) = create_test_components();

    // Start the engine
    assert!(engine.start().is_ok());

    // Process an event through the dispatcher
    let event = create_test_event(1, "test_event");
    let dispatch_result = dispatcher.dispatch_event(event.clone()).unwrap();
    assert!(dispatch_result.success);

    // Verify the transition was executed
    let action_results = executor.behavior.action_results.lock().unwrap();
    assert_eq!(action_results.len(), 1);

    // Stop the engine
    assert!(engine.stop().is_ok());
}

#[test]
fn test_concurrent_state_transitions() {
    use std::thread;

    let (engine, executor, dispatcher, _) = create_test_components();
    let mut handles = vec![];

    assert!(engine.start().is_ok());

    // Spawn multiple threads to dispatch events
    for i in 0..5 {
        let dispatcher = dispatcher.clone();
        let handle = thread::spawn(move || {
            let event = create_test_event(i, &format!("event_{}", i));
            dispatcher.dispatch_event(event).unwrap()
        });
        handles.push(handle);
    }

    // Wait for all transitions to complete
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.success);
    }

    // Verify all actions were executed
    let action_results = executor.behavior.action_results.lock().unwrap();
    assert_eq!(action_results.len(), 5);

    assert!(engine.stop().is_ok());
}

#[test]
fn test_error_handling_and_diagnostics() {
    let (_, _, dispatcher, provider) = create_test_components();

    // Set up a failing guard
    let mut guard_results = dispatcher.executor.behavior.guard_results.lock().unwrap();
    guard_results.insert("initial".to_string(), false);
    drop(guard_results);

    // Attempt a transition that will fail the guard
    let event = create_test_event(1, "test_event");
    let result = dispatcher.dispatch_event(event).unwrap();
    assert!(!result.success);

    // Give a small delay to ensure diagnostics are processed
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Verify diagnostic data
    let diagnostic_data = provider.diagnostics().get_diagnostic_data();
    assert!(
        diagnostic_data.total_logs > 0,
        "Expected logs from failed transition"
    );

    // Verify log entries
    let log_entries = provider.diagnostics().logger.get_entries();
    assert!(!log_entries.is_empty(), "Expected at least one log entry");

    // Verify that there's a transition attempt log
    let has_transition_log = log_entries
        .iter()
        .any(|(msg, _)| msg.contains("Transition attempt from") && msg.contains("false"));
    assert!(
        has_transition_log,
        "Expected to find transition attempt log"
    );
}

#[test]
fn test_resource_management_during_transitions() {
    let (_, _, dispatcher, provider) = create_test_components();

    // Allocate a resource
    let resource_handle = provider
        .resource_manager()
        .allocate_resource(&"test_resource".to_string(), &())
        .unwrap();

    // Perform a transition
    let event = create_test_event(1, "test_event");
    assert!(dispatcher.dispatch_event(event).unwrap().success);

    // Release the resource
    assert!(provider
        .resource_manager()
        .release_resource(resource_handle)
        .is_ok());
}
