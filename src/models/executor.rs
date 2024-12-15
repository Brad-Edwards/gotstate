use std::sync::{Arc, Mutex};
use crate::{CoreExecutor, CoreContextProvider};
use crate::behavior::Behavior;
use crate::diagnostics::Diagnostics;
use crate::model::{ModelDefinition, TransitionDefinition};
use crate::resource::ResourceManager;

use super::{EngineError, StateHandle, Event, TransitionResult};
use crate::context::FsmContext;

pub struct FsmExecutor<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T> + Send + Sync,
    D: Diagnostics + Send + Sync,
    M: ModelDefinition<TransitionDef = T> + Send + Sync,
    R: ResourceManager + Send + Sync,
    T: TransitionDefinition + Send + Sync,
{
    model: Arc<M>,
    behavior: Arc<B>,
    context: Arc<FsmContext<StateHandle, D, R>>,
    current_state: Mutex<StateHandle>,
}

impl<B, D, M, R, T> FsmExecutor<B, D, M, R, T>
where
    B: Behavior<TransitionDef = T> + Send + Sync,
    D: Diagnostics + Send + Sync,
    M: ModelDefinition<TransitionDef = T> + Send + Sync,
    R: ResourceManager + Send + Sync,
    T: TransitionDefinition + Send + Sync,
{
    pub fn new(model: M, behavior: B, context: FsmContext<StateHandle, D, R>, initial_state: StateHandle) -> Self {
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
    B: Behavior<TransitionDef = T> + Send + Sync + 'static,
    D: Diagnostics + Send + Sync + 'static,
    M: ModelDefinition<TransitionDef = T> + Send + Sync + 'static,
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
        if !self.behavior.evaluate_guard(transition, event, &self.context.resource_manager()) {
            return Err(EngineError::InvalidTransition);
        }

        let old_state = {
            let lock = self.current_state.lock().map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
            lock.state_id.clone()
        };

        // Execute actions
        if let Err(e) = self.behavior.execute_actions(transition, event, &self.context.resource_manager()) {
            return Err(EngineError::ExecutionError(format!("Action execution failed: {:?}", e)));
        }

        // Determine next state from transition
        let new_state_id = match self.model.resolve_target_state(transition) {
            Some(s) => s,
            None => return Err(EngineError::InvalidTransition),
        };

        {
            let mut lock = self.current_state.lock().map_err(|_| EngineError::UnknownError("Poisoned mutex".into()))?;
            lock.state_id = new_state_id.clone();
        }

        Ok(TransitionResult {
            from_state: old_state,
            to_state: new_state_id,
            event_name: event.name.clone(),
        })
    }

    fn current_state(&self) -> Self::StateHandle {
        let lock = self.current_state.lock().unwrap();
        StateHandle {
            state_id: lock.state_id.clone(),
        }
    }

    fn context(&self) -> &dyn CoreContextProvider<
        StateHandle = Self::StateHandle,
        Diagnostics = Self::Diagnostics,
        ResourceManager = Self::ResourceManager,
    > {
        self.context.as_ref()
    }
}
