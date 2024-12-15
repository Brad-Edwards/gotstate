use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

use crate::{CoreDispatcher, CoreExecutor};
use super::{EngineError, Event, TransitionResult};

pub struct FsmDispatcher<E, Ev, Tr, Ex>
where
    E: Send + Sync,
    Ev: Send,
    Tr: Send,
    Ex: CoreExecutor<Event = Ev, TransitionResult = Tr>,
{
    executor: Arc<Ex>,
    event_queue: Mutex<VecDeque<Ev>>,
    _phantom_e: std::marker::PhantomData<E>,
}

impl<E, Ev, Tr, Ex> FsmDispatcher<E, Ev, Tr, Ex>
where
    E: Send + Sync,
    Ev: Send,
    Tr: Send,
    Ex: CoreExecutor<Event = Ev, TransitionResult = Tr>,
{
    pub fn new(executor: Ex) -> Self {
        Self {
            executor: Arc::new(executor),
            event_queue: Mutex::new(VecDeque::new()),
            _phantom_e: std::marker::PhantomData,
        }
    }

    pub fn enqueue_event(&self, event: Ev) -> Result<(), E> {
        let mut queue = self.event_queue.lock().map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        queue.push_back(event);
        Ok(())
    }
}

impl<E, Ev, Tr, Ex> CoreDispatcher for FsmDispatcher<E, Ev, Tr, Ex>
where
    E: Send + Sync + From<EngineError>,
    Ev: Send,
    Tr: Send,
    Ex: CoreExecutor<Event = Ev, TransitionResult = Tr>,
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
        let mut queue = self.event_queue.lock().map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
        while let Some(ev) = queue.pop_front() {
            let _ = self.executor.execute_transition_for_event(ev)?;
        }
        Ok(())
    }
}

// Add a helper method on CoreExecutor to handle direct event-based transitions:
trait ExecutorExt {
    type Error;
    type Event;
    type TransitionResult;
    fn execute_transition_for_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error>;
}

impl<B, D, M, R, T> ExecutorExt for super::FsmExecutor<B, D, M, R, T>
where
    B: crate::behavior::Behavior<TransitionDef = T> + Send + Sync + 'static,
    D: crate::diagnostics::Diagnostics + Send + Sync + 'static,
    M: crate::model::ModelDefinition<TransitionDef = T> + Send + Sync + 'static,
    R: crate::resource::ResourceManager + Send + Sync + 'static,
    T: crate::model::TransitionDefinition + Send + Sync + 'static,
{
    type Error = EngineError;
    type Event = Event;
    type TransitionResult = TransitionResult;

    fn execute_transition_for_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error> {
        // Look up a suitable transition from the current state using the model.
        let current = self.current_state();
        let transitions = self.model.available_transitions(&current.state_id, &event);
        if transitions.is_empty() {
            // No valid transition for this event
            return Err(EngineError::InvalidTransition);
        }

        // In a hierarchical FSM, multiple transitions might match;
        // enterprise logic might prioritize or pick the first valid one.
        let chosen_transition = transitions[0].clone();

        self.execute_transition(&chosen_transition, &event)
    }
}
