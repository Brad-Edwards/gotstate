use crate::{CoreContextProvider};
use crate::diagnostics::Diagnostics;
use crate::resource::ResourceManager;

use super::StateHandle;

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
