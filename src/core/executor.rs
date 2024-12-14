use crate::behavior::Behavior;
use crate::diagnostics::Diagnostics;
use crate::model::TransitionDefinition;
use crate::resource::ResourceManager;

pub trait CoreExecutor: Send + Sync {
    type Error;
    type StateHandle: Send + Sync;
    type Event: Send;
    type TransitionResult: Send;
    type TransitionDef: TransitionDefinition;
    type Behavior: Behavior<TransitionDef = Self::TransitionDef>;
    type ResourceManager: ResourceManager;
    type Diagnostics: Diagnostics;
    type Model;

    fn execute_transition(
        &self,
        transition: &Self::TransitionDef,
        event: &Self::Event,
    ) -> Result<Self::TransitionResult, Self::Error>;

    fn current_state(&self) -> Self::StateHandle;

    fn context(
        &self,
    ) -> &dyn super::CoreContextProvider<
        StateHandle = Self::StateHandle,
        Diagnostics = Self::Diagnostics,
        ResourceManager = Self::ResourceManager,
    >;
}
